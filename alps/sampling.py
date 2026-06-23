"""
alps/sampling.py
----------------
ALPS sampling functions for cascaded MRI reconstruction.

Core functions (ported directly from JyoChand/ALPS/utils.py):
    giveScore(x, net, sigma, precondition)  — energy gradient
    ALPS(A, net, y, opts, ...)              — annealed Langevin sampler
    giveTsteps(...)                         — noise schedule

Cascade entry point (new):
    cascaded_ALPS(...)                      — two-stage reconstruction

References
----------
Thornton et al., "ALPS: AnneaLed Posterior Sampling for Inverse Imaging
via Diffusion-to-Energy Distillation", TMLR 2025 (arXiv 2601.02594).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .operators import (MRIOperator, avg_pool_complex,
                        make_cartesian_mask, crop_kspace_center)
from .denoiser  import BaseDenoiser, SRDenoiser


# ─────────────────────────────────────────────────────────────────────────────
#  Options dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ALPSOptions:
    """
    Hyperparameters for a single ALPS stage.

    num_steps : int
        Number of noise levels in the annealing schedule.
    sigma_max : float
        Starting noise level (typically 80.0 for EDM VE).
    sigma_min : float
        Final noise level (typically 0.002).
    rho : float
        Schedule exponent (7.0 matches EDM Heun sampler schedule).
    K : int
        Number of Langevin inner steps per noise level.
    """
    num_steps : int   = 20
    sigma_max : float = 80.0
    sigma_min : float = 0.002
    rho       : float = 7.0
    K         : int   = 1


# ─────────────────────────────────────────────────────────────────────────────
#  Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def giveTsteps(sigma_max: float, sigma_min: float, rho: float,
               num_steps: int, device: torch.device) -> torch.Tensor:
    """
    EDM-style rho-power noise schedule.

    Returns a (num_steps,) tensor of decreasing sigma values from
    sigma_max down to sigma_min.
    """
    idx = torch.arange(num_steps, dtype=torch.float64, device=device)
    t   = (
        sigma_max ** (1.0 / rho)
        + idx / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Score computation
# ─────────────────────────────────────────────────────────────────────────────

def giveScore(
    x:            torch.Tensor,
    net:          torch.nn.Module,
    sigma:        torch.Tensor,
    precondition: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the energy and its gradient (score) for

        E(x, sigma) = 0.5 || x - psi(x, sigma) ||^2

    using the vector-Jacobian product to avoid materialising the full Jacobian.

    Parameters
    ----------
    x     : (B, C, H, W)  current iterate  (no grad required on entry)
    net   : BaseDenoiser or SRDenoiser
    sigma : (B, 1, 1, 1) or scalar
    precondition : match ALPS paper's preconditioned score formulation

    Returns
    -------
    E     : (B,)          scalar energy per sample
    score : (B, C, H, W)  gradient  nabla_x E
    """
    # NOTE: we only have a trained EDM *score* network, not an energy model, so
    # we drop the Jacobian term  J_{psi}^T eps  from  nabla_x E.  What remains,
    #     score = x - psi(x, sigma),
    # is the MMSE-residual direction the ALPS sampler actually consumes
    # (it forms d = x - score = psi(x, sigma)).  No autograd is needed, so we
    # run the denoiser under no_grad to avoid building a useless graph.
    # When an energy model is available, restore the VJP path below.
    with torch.no_grad():
        denoised = net(x, sigma)               # psi(x, sigma)
    score = x - denoised                          # nabla_x E (no Jacobian term)
    E     = 0                                      # placeholder; unused by ALPS

    # ── Energy-model path (requires net.vjp / a trained energy model) ────────
    # x   = x.clone().detach().requires_grad_(True)
    # denoised = net(x, sigma)
    # eps      = x - denoised                       # residual
    # E        = 0.5 * torch.sum(eps.abs() ** 2, dim=(1, 2, 3))
    # JT_eps   = net.vjp(
    #     outputs      = denoised,
    #     inputs       = x,
    #     conditioning = sigma,
    #     vector       = eps,
    #     precondition = precondition,
    # )
    # score = eps - JT_eps

    return E, score


# ─────────────────────────────────────────────────────────────────────────────
#  ALPS sampler  (single stage)
# ─────────────────────────────────────────────────────────────────────────────

def ALPS(
    A:                 torch.nn.Module,
    net:               torch.nn.Module,
    y:                 torch.Tensor,
    opts:              ALPSOptions,
    isALPS:            bool = True,
    storeIntermediate: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Annealed Langevin Posterior Sampling (single stage).

    At each noise level t_i the algorithm performs K inner steps of
    preconditioned Langevin dynamics:

        d       = x - score(x, t)          # MMSE denoising direction
        rhs     = A^T y / eta^2 + d / t^2
        x_tilde = B_t * rhs                # B_t = (A^T A/eta^2 + I/t^2)^{-1}
        x       = x_tilde + B_t^{1/2} * n  # add Langevin noise (isALPS=True)

    Parameters
    ----------
    A    : operator with .adjoint, .PreCondition, .NoiseModulation
    net  : denoiser with .forward and .vjp
    y    : (B, C, H, W)  measurements
    opts : ALPSOptions
    isALPS : True → posterior sample;  False → MAP estimate (no noise)
    storeIntermediate : if True return all iterates as second output

    Returns
    -------
    x         : (B, C, H, W)  final reconstruction
    xsample   : (num_steps, B, C, H, W)  intermediate iterates (only if
                storeIntermediate=True)
    """
    device  = y.device
    t_steps = giveTsteps(opts.sigma_max, opts.sigma_min, opts.rho,
                         opts.num_steps, device)

    # Initialise from zero-filled adjoint solution
    Atb    = A.adjoint(y).detach()
    xtilde = A.PreCondition(Atb / A.eta2, t_steps[0].item())

    if isALPS:
        n = A.NoiseModulation(torch.randn_like(xtilde), t_steps[0].item())
        x = (xtilde + n).detach()
    else:
        x = xtilde.detach()

    if storeIntermediate:
        xshape    = [opts.num_steps] + list(x.shape)
        xsample   = torch.zeros(xshape, dtype=x.dtype)

    for i, t in enumerate(t_steps):
        t_val = t.item()
        # Match the iterate dtype: t_steps is float64 (schedule precision), but
        # feeding a float64 sigma promotes the score to float64 through vjp's
        # sigma-preconditioning, which then leaks into the net input.
        sigma = t.reshape(-1, 1, 1, 1).to(device=device, dtype=x.dtype)

        for k in range(opts.K):
            # ── Score step ──────────────────────────────────────────────
            _, score = giveScore(x, net, sigma, precondition=True)
            d = (x - score).detach()

            # ── Preconditioned data-consistency update ───────────────────
            rhs    = Atb / A.eta2 + d / t_val ** 2
            xtilde = A.PreCondition(rhs, t_val).detach()

            # ── Langevin noise injection (all but final inner step) ──────
            if isALPS and (k < opts.K - 1):
                n = A.NoiseModulation(torch.randn_like(x), t_val)
                x = (xtilde + n).detach()
            else:
                x = xtilde

        if storeIntermediate:
            xsample[i] = x.detach().cpu()

        # ── Move to next noise level ─────────────────────────────────────
        if i < len(t_steps) - 1:
            t_next = t_steps[i + 1].item()
            n      = A.NoiseModulation(torch.randn_like(x), t_next)
            x      = (xtilde + n).detach()

    if storeIntermediate:
        return x, xsample
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Cascaded ALPS  (two-stage)
# ─────────────────────────────────────────────────────────────────────────────

def cascaded_ALPS(
    multicoil_kspace_384: torch.Tensor,
    csm_384:              torch.Tensor,
    net_base:             torch.nn.Module,
    net_sr:               torch.nn.Module,
    opts_base:            ALPSOptions,
    opts_sr:              ALPSOptions,
    acceleration:         int   = 4,
    acs_fraction:         float = 0.08,
    eta:                  float = 0.01,
    seed:                 int   = 42,
    cg_iters:             int   = 10,
    device:               torch.device = torch.device('cuda'),
    store_stage1:         bool  = False,
    store_stage2:         bool  = False,
    verbose:              bool  = True,
) -> dict:
    """
    Two-stage cascaded ALPS for MRI reconstruction.

    Stage 1
    -------
    Solve the 96×96 low-resolution posterior with a low-res SENSE model
    (4×-pooled image and coil maps) and the Stage 1 (base) score network.

        y1 = M_96 · F_96 · (S_96 ⊙ x_96) + noise

    Stage 2
    -------
    Fix x_96 from Stage 1 and solve the 384×384 high-resolution posterior
    using the real multi-coil k-space and the SR score network conditioned
    on x_96.

        y2_c = M_384 · F_384 · (S_c ⊙ x_384) + noise

    Parameters
    ----------
    multicoil_kspace_384 : (B, C, 384, 384) complex
        Full, *unmasked* multi-coil k-space (real acquisition).
    csm_384 : (1, C, 384, 384) complex
        Coil-sensitivity maps.
    net_base : EDMPrecond (loaded, eval mode)
        Stage 1 score network (96×96, 2-channel).
    net_sr : EDMSRPrecond (loaded, eval mode)
        Stage 2 SR score network (384×384, conditioned on 96×96).
    opts_base : ALPSOptions  for Stage 1
    opts_sr   : ALPSOptions  for Stage 2
    acceleration : int
        Cartesian acceleration factor (same for both stages).
    acs_fraction : float
    eta : float
        Noise std (same for both stages; adjust if needed).
    seed : int
    device : torch.device
    store_stage1, store_stage2 : bool
        Whether to return intermediate iterates for each stage.
    verbose : bool

    Returns
    -------
    dict with keys:
        'x96'          : (B, 2, 96, 96)   Stage 1 reconstruction
        'x384'         : (B, 2, 384, 384) Stage 2 reconstruction
        'x96_iterates' : tensor or None    (if store_stage1)
        'x384_iterates': tensor or None    (if store_stage2)
    """
    multicoil_kspace_384 = multicoil_kspace_384.to(device)
    csm_384              = csm_384.to(device)

    # ── Build SENSE operators ──────────────────────────────────────────────────
    #   Stage 1 (96×96): average-pool the coil maps and use fft_scale = 4 so the
    #   pooled-image forward model matches the measured k-space centre.
    full_size = multicoil_kspace_384.shape[-1]      # 384
    low_size  = full_size // 4                       # 96
    pool      = full_size // low_size                # 4

    csm_96   = avg_pool_complex(csm_384, pool)
    # Single Cartesian mask; the 96×96 mask is its central crop so both stages
    # sample exactly the same phase-encode lines in the k-space centre.
    mask_384 = make_cartesian_mask(full_size, acceleration, acs_fraction, seed)
    mask_96  = crop_kspace_center(mask_384, low_size)

    A1 = MRIOperator(
        csm=csm_96, mask=mask_96,
        eta=eta, cg_iters=cg_iters,
        fft_scale=full_size / low_size, device=device,
    )
    A2 = MRIOperator(
        csm=csm_384, mask=mask_384,
        eta=eta, cg_iters=cg_iters,
        fft_scale=1.0, device=device,
    )

    # ── Get multi-coil measurements ────────────────────────────────────────────
    y1 = A1.get_measurements(multicoil_kspace_384)            # (B, C, 96, 96)
    y2 = A2.get_measurements(multicoil_kspace_384)            # (B, C, 384, 384)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Stage 1: 96×96 reconstruction from central k-space")
        print(f"  {opts_base.num_steps} noise levels, K={opts_base.K} inner steps")
        print("=" * 60)

    denoiser1 = BaseDenoiser(net_base)

    # Seed the stochastic Langevin/NoiseModulation draws so this matches the
    # cell-by-cell notebook (which reseeds with the same `seed` before Stage 1).
    torch.manual_seed(seed)
    result1 = ALPS(
        A=A1, net=denoiser1, y=y1,
        opts=opts_base,
        isALPS=True,
        storeIntermediate=store_stage1,
    )

    if store_stage1:
        x96, x96_iterates = result1
    else:
        x96, x96_iterates = result1, None

    if verbose:
        print(f"  Stage 1 done.  x96 shape: {tuple(x96.shape)}\n")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Stage 2: 384×384 reconstruction conditioned on 96×96")
        print(f"  {opts_sr.num_steps} noise levels, K={opts_sr.K} inner steps")
        print("=" * 60)

    denoiser2 = SRDenoiser(net_sr, x96_fixed=x96)

    # Reseed before Stage 2 to match the notebook's per-stage seeding.
    torch.manual_seed(seed)
    result2 = ALPS(
        A=A2, net=denoiser2, y=y2,
        opts=opts_sr,
        isALPS=True,
        storeIntermediate=store_stage2,
    )

    if store_stage2:
        x384, x384_iterates = result2
    else:
        x384, x384_iterates = result2, None

    if verbose:
        print(f"  Stage 2 done.  x384 shape: {tuple(x384.shape)}\n")

    return {
        'x96'           : x96,
        'x384'          : x384,
        'x96_iterates'  : x96_iterates,
        'x384_iterates' : x384_iterates,
    }
