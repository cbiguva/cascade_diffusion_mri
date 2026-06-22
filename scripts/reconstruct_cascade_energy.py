"""
reconstruct_cascade_energy.py
-----------------------------
Joint MRI reconstruction using cascaded energy models.

Three approaches implemented:
    A) DPS-style:   Fixed base output, likelihood-guided SR denoising
    B) Joint:       Simultaneous optimization of x_96 and x_384 with
                    JVP-corrected energy scores + data fidelity +
                    cross-resolution consistency
    C) ADMM-style:  Alternating denoising and data-consistency projection

All approaches use:
    - JVP-corrected scores (from energy_denoiser.py)
    - MRI forward model (from mri_forward_model.py)
    - Pre-trained EDM cascade (base 96×96 + SR 96→384)
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from energy_denoiser import Denoiser, giveScore, giveEnergy
from mri_forward_model import MRIForwardOp


# ──────────────────────────────────────────────────────────────────────────────
#  Noise schedule helpers (EDM ρ-schedule)
# ──────────────────────────────────────────────────────────────────────────────

def edm_sigma_schedule(num_steps, sigma_min=0.002, sigma_max=80, rho=7, device='cuda'):
    """Generate EDM σ schedule (descending)."""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return t_steps.to(torch.float32)


def pool_4x4(x):
    """Average-pool (B, 2, 384, 384) → (B, 2, 96, 96)."""
    return F.avg_pool2d(x, kernel_size=4, stride=4)


def upsample_4x(x):
    """Bilinear upsample (B, 2, 96, 96) → (B, 2, 384, 384)."""
    return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)


# ──────────────────────────────────────────────────────────────────────────────
#  Approach A: DPS-style cascade reconstruction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_dps(
    base_denoiser,
    sr_denoiser,
    y_measured,
    mri_op,
    num_steps=50,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    lambda_data=1.0,
    verbose=True,
    device='cuda',
):
    """
    DPS-style reconstruction: generate x_96 from base model,
    then reconstruct x_384 with SR model + data consistency.

    Parameters
    ----------
    base_denoiser : Denoiser
        Wrapped base model (96×96).
    sr_denoiser : Denoiser
        Wrapped SR model (96→384).
    y_measured : (B, 2, H, W) tensor
        Undersampled k-space measurements.
    mri_op : MRIForwardOp
        MRI forward operator.
    num_steps : int
        Number of denoising steps.
    lambda_data : float
        Weight for data fidelity gradient.

    Returns
    -------
    x_384 : (B, 2, 384, 384) — reconstructed image
    x_96  : (B, 2, 96, 96)   — base model output
    """
    B = y_measured.shape[0]
    sigmas = edm_sigma_schedule(num_steps, sigma_min, sigma_max, rho, device)

    # Stage 1: Generate x_96 from base model (standard EDM sampling)
    x_96 = torch.randn(B, 2, 96, 96, device=device) * sigmas[0]

    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        sigma_cur_t = sigma_cur.expand(B)
        denoised = base_denoiser(x_96, sigma_cur_t)
        d = (x_96 - denoised) / sigma_cur
        x_96 = x_96 + (sigma_next - sigma_cur) * d

    if verbose:
        print(f"  Base 96×96 generated, range [{x_96.min():.3f}, {x_96.max():.3f}]")

    # Stage 2: Reconstruct x_384 with SR + data consistency
    x_384 = torch.randn(B, 2, 384, 384, device=device) * sigmas[0]

    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        sigma_cur_t = sigma_cur.expand(B)

        # SR denoising step
        denoised = sr_denoiser(x_384, sigma_cur_t, low_res=x_96)
        d = (x_384 - denoised) / sigma_cur
        x_384 = x_384 + (sigma_next - sigma_cur) * d

        # Data consistency gradient step
        if lambda_data > 0:
            dc_grad = mri_op.gradient(x_384, y_measured)
            step_size = lambda_data * sigma_next ** 2
            x_384 = x_384 - step_size * dc_grad

        if verbose and (i % 10 == 0 or i == num_steps - 2):
            print(f"  step {i+1}/{num_steps-1}, σ={sigma_cur:.4f}")

    return x_384, x_96


# ──────────────────────────────────────────────────────────────────────────────
#  Approach B: Joint multi-resolution posterior sampling
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_joint(
    base_denoiser,
    sr_denoiser,
    y_measured,
    mri_op,
    num_steps=50,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    lambda_data=1.0,
    lambda_consist=0.1,
    use_jvp=True,
    verbose=True,
    device='cuda',
):
    """
    Joint multi-resolution posterior sampling with JVP-corrected
    energy scores.

    Simultaneously optimizes x_96 and x_384 under the joint posterior:
        p(x_96, x_384 | y) ∝ p(y|x_384) · p_base(x_96) · p_SR(x_384|x_96)

    The cross-resolution consistency term couples the two resolutions:
        x_96 ≈ AvgPool(x_384)

    Parameters
    ----------
    use_jvp : bool
        If True, use JVP-corrected energy scores (proper energy model).
        If False, use Tweedie scores (standard diffusion, for ablation).
    lambda_consist : float
        Weight for cross-resolution consistency.

    Returns
    -------
    x_384 : (B, 2, 384, 384) — reconstructed image
    x_96  : (B, 2, 96, 96)   — jointly optimized low-res
    """
    B = y_measured.shape[0]
    sigmas = edm_sigma_schedule(num_steps, sigma_min, sigma_max, rho, device)

    # Initialize from noise
    x_96 = torch.randn(B, 2, 96, 96, device=device) * sigmas[0]
    x_384 = torch.randn(B, 2, 384, 384, device=device) * sigmas[0]

    for i, (sigma_cur, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        sigma_cur_t = sigma_cur.expand(B)
        dt = sigma_next - sigma_cur  # negative (decreasing σ)

        # ── Scores for x_96 (base model) ─────────────────────────────────
        if use_jvp:
            score_96 = giveScore(x_96, base_denoiser, sigma_cur_t,
                                 precondition=True)
        else:
            denoised_96 = base_denoiser(x_96, sigma_cur_t)
            score_96 = (denoised_96 - x_96) / sigma_cur ** 2

        # ── Scores for x_384 (SR model) ──────────────────────────────────
        if use_jvp:
            score_384 = giveScore(x_384, sr_denoiser, sigma_cur_t,
                                  precondition=True, low_res=x_96.detach())
        else:
            denoised_384 = sr_denoiser(x_384, sigma_cur_t, low_res=x_96)
            score_384 = (denoised_384 - x_384) / sigma_cur ** 2

        # ── Data fidelity gradient (on x_384 only) ───────────────────────
        with torch.no_grad():
            dc_grad = mri_op.gradient(x_384, y_measured)

        # ── Cross-resolution consistency ─────────────────────────────────
        # Gradient of ½λ ||x_96 - Pool(x_384)||²
        with torch.no_grad():
            pooled_384 = pool_4x4(x_384)
            consist_residual = x_96 - pooled_384  # (B, 2, 96, 96)

            # Gradient w.r.t. x_96:  λ · (x_96 - Pool(x_384))
            g_consist_96 = lambda_consist * consist_residual

            # Gradient w.r.t. x_384: -λ · Pool^T(x_96 - Pool(x_384))
            # Pool^T is the adjoint of avg_pool = upsample / 16
            g_consist_384 = -lambda_consist * upsample_4x(consist_residual) / 16.0

        # ── Update x_96 ──────────────────────────────────────────────────
        # Euler step: x += dt · dx/dt, where dx/dt = -σ · score
        with torch.no_grad():
            dx_96 = -sigma_cur * score_96 - g_consist_96
            x_96 = x_96 + dt * dx_96

        # ── Update x_384 ─────────────────────────────────────────────────
        with torch.no_grad():
            dx_384 = -sigma_cur * score_384 - lambda_data * dc_grad + g_consist_384
            x_384 = x_384 + dt * dx_384

        if verbose and (i % 10 == 0 or i == num_steps - 2):
            with torch.no_grad():
                dc_err = (mri_op.forward(x_384) - y_measured).pow(2).sum().sqrt()
                consist_err = consist_residual.pow(2).sum().sqrt()
            print(f"  step {i+1}/{num_steps-1}, σ={sigma_cur:.4f}, "
                  f"dc_err={dc_err:.4f}, consist={consist_err:.4f}")

    return x_384, x_96


# ──────────────────────────────────────────────────────────────────────────────
#  Approach C: ADMM-style alternating reconstruction
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_admm(
    base_denoiser,
    sr_denoiser,
    y_measured,
    mri_op,
    num_outer=20,
    inner_steps=5,
    sigma_start=10.0,
    sigma_end=0.01,
    lambda_data=1.0,
    verbose=True,
    device='cuda',
):
    """
    ADMM-style alternating reconstruction.

    Outer loop alternates between:
        1. Prior step:  Denoise using cascade (base → SR)
        2. DC step:     Replace measured k-space lines
        3. Consistency:  Enforce x_96 = Pool(x_384)

    Parameters
    ----------
    num_outer : int
        Number of outer ADMM iterations.
    inner_steps : int
        Denoising sub-steps per outer iteration.
    sigma_start, sigma_end : float
        Noise level annealing schedule.

    Returns
    -------
    x_384 : (B, 2, 384, 384) — reconstructed image
    x_96  : (B, 2, 96, 96)   — low-res estimate
    """
    B = y_measured.shape[0]

    # Initialize from zero-filled reconstruction
    x_384 = mri_op.adjoint(y_measured)
    x_96 = pool_4x4(x_384)

    # Annealing schedule for σ
    sigmas = torch.logspace(
        np.log10(sigma_start), np.log10(sigma_end),
        num_outer, device=device,
    )

    for k in range(num_outer):
        sigma = sigmas[k]
        sigma_t = sigma.expand(B)

        # ── Step 1: Prior denoising (cascade) ────────────────────────────
        # Base model: denoise x_96
        for _ in range(inner_steps):
            denoised_96 = base_denoiser(x_96, sigma_t)
            x_96 = denoised_96  # direct denoising (not sampling)

        # SR model: denoise x_384 conditioned on x_96
        for _ in range(inner_steps):
            denoised_384 = sr_denoiser(x_384, sigma_t, low_res=x_96)
            x_384 = denoised_384

        # ── Step 2: Data consistency ─────────────────────────────────────
        x_384 = mri_op.data_consistency(x_384, y_measured)

        # ── Step 3: Cross-resolution consistency ─────────────────────────
        x_96 = pool_4x4(x_384)

        if verbose and (k % 5 == 0 or k == num_outer - 1):
            dc_err = (mri_op.forward(x_384) - y_measured).pow(2).sum().sqrt()
            print(f"  ADMM iter {k+1}/{num_outer}, σ={sigma:.4f}, dc_err={dc_err:.4f}")

    return x_384, x_96


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(x_recon, x_gt):
    """
    Compute reconstruction metrics.

    Parameters
    ----------
    x_recon, x_gt : (B, 2, H, W) tensors

    Returns
    -------
    dict with PSNR, SSIM (per-sample, averaged)
    """
    # Compute magnitude images for metrics
    mag_recon = torch.sqrt(x_recon[:, 0] ** 2 + x_recon[:, 1] ** 2)
    mag_gt = torch.sqrt(x_gt[:, 0] ** 2 + x_gt[:, 1] ** 2)

    # PSNR
    mse = ((mag_recon - mag_gt) ** 2).mean(dim=(1, 2))
    max_val = mag_gt.amax(dim=(1, 2))
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))

    # NMSE
    nmse = mse / (mag_gt ** 2).mean(dim=(1, 2))

    return {
        'psnr': psnr.mean().item(),
        'nmse': nmse.mean().item(),
        'psnr_std': psnr.std().item(),
    }
