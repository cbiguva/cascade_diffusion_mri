"""
sample_edm_cascade.py
---------------------
Cascaded inference:  Base(32×32)  →  NN-upsample  →  SR(64×64)

Uses EDM's Heun 2nd-order ODE sampler (Algorithm 2) for both stages.
Only 18 steps per stage (vs. 250+ for DDPM).

Usage:
    python scripts/sample_edm_cascade.py \\
        --base_model  checkpoints/edm_afhq_base_32/network-snapshot-XXXXXX.pkl \\
        --sr_model    checkpoints/edm_afhq_sr_64/network-snapshot-XXXXXX.pkl \\
        --num_samples 16 --batch_size 8 \\
        --out_dir samples/edm_afhq/
"""

import argparse
import math
import os
import sys
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Make EDM importable ──────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EDM_PATH = os.path.join(_REPO_ROOT, 'edm_repo')
if _EDM_PATH not in sys.path:
    sys.path.insert(0, _EDM_PATH)

# Ensure our custom SR model is importable for pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
#  EDM Heun Sampler (Algorithm 2 from Karras et al., 2022)
# ──────────────────────────────────────────────────────────────────────────────

def edm_sampler(
    net, latents,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    **model_kwargs,
):
    """
    EDM 2nd-order Heun sampler (deterministic by default).

    Parameters
    ----------
    net : EDMPrecond or EDMSRPrecond
        Preconditioned diffusion model.
    latents : (B, C, H, W)
        Initial noise (from torch.randn).
    model_kwargs : dict
        Extra kwargs passed to net() on every call.
        For SR models, pass low_res=... here.
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization  (ρ-schedule)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Main sampling loop
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Optionally increase noise temporarily (stochastic sampler)
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step
        denoised = net(
            x_hat.to(torch.float32), t_hat.to(torch.float32),
            **model_kwargs,
        ).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd-order correction (Heun)
        if i < num_steps - 1:
            denoised = net(
                x_next.to(torch.float32), t_next.to(torch.float32),
                **model_kwargs,
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

def tensor_to_pil(t):
    """Convert (C, H, W) tensor in [-1, 1] to PIL RGB image."""
    t = ((t.float() + 1.0) * 127.5).clamp(0, 255).byte()
    return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())


def make_grid(images, nrow=4):
    """Make a grid of PIL images."""
    n = len(images)
    ncol = nrow
    nrow_actual = math.ceil(n / ncol)
    w, h = images[0].size
    grid = Image.new('RGB', (ncol * w, nrow_actual * h), (0, 0, 0))
    for i, img in enumerate(images):
        grid.paste(img, ((i % ncol) * w, (i // ncol) * h))
    return grid


def nn_upsample_2x(x):
    """NN-upsample (B, C, 32, 32) → (B, C, 64, 64)."""
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def sample_cascade(args, device):
    # ── Load base model ──────────────────────────────────────────────────
    print("Loading base model…")
    with open(args.base_model, 'rb') as f:
        base_data = pickle.load(f)
    model_base = base_data['ema'].to(device).eval()
    print(f"  Base: {sum(p.numel() for p in model_base.parameters()):,} params")
    print(f"  Resolution: {model_base.img_resolution}×{model_base.img_resolution}")

    # ── Load SR model ────────────────────────────────────────────────────
    print("Loading SR model…")
    with open(args.sr_model, 'rb') as f:
        sr_data = pickle.load(f)
    model_sr = sr_data['ema'].to(device).eval()
    print(f"  SR: {sum(p.numel() for p in model_sr.parameters()):,} params")
    print(f"  Resolution: {model_sr.img_resolution}×{model_sr.img_resolution}")

    # ── Sample ───────────────────────────────────────────────────────────
    os.makedirs(os.path.join(args.out_dir, '32'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, '64'), exist_ok=True)

    all_32, all_64 = [], []
    n_done = 0

    while n_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - n_done)
        print(f"\nSampling batch {n_done}..{n_done + bs} / {args.num_samples}")

        # ── Stage 1: Base model → 32×32 ──────────────────────────────────
        with torch.no_grad():
            latents_32 = torch.randn(
                [bs, model_base.img_channels, model_base.img_resolution, model_base.img_resolution],
                device=device,
            )
            sample_32 = edm_sampler(
                model_base, latents_32,
                num_steps=args.base_steps,
                sigma_min=0.002, sigma_max=80,
            ).to(torch.float32)  # (B, 3, 32, 32) in [-1, 1]

        # ── Stage 2: NN-upsample → SR model → 64×64 ─────────────────────
        low_res = nn_upsample_2x(sample_32)  # (B, 3, 64, 64)

        # Note: we pass the original 32×32 sample (not upsampled) as low_res
        # The EDMSRPrecond model handles bilinear upsampling internally
        with torch.no_grad():
            latents_64 = torch.randn(
                [bs, model_sr.img_channels, model_sr.img_resolution, model_sr.img_resolution],
                device=device,
            )
            sample_64 = edm_sampler(
                model_sr, latents_64,
                num_steps=args.sr_steps,
                sigma_min=0.002, sigma_max=80,
                low_res=sample_32,  # pass original 32×32, model upsamples internally
            ).to(torch.float32)  # (B, 3, 64, 64) in [-1, 1]

        # ── Save individual images ───────────────────────────────────────
        for i in range(bs):
            idx = n_done + i
            img32 = tensor_to_pil(sample_32[i])
            img64 = tensor_to_pil(sample_64[i])
            img32.save(os.path.join(args.out_dir, '32', f'sample_{idx:04d}.png'))
            img64.save(os.path.join(args.out_dir, '64', f'sample_{idx:04d}.png'))
            all_32.append(img32)
            all_64.append(img64)

        n_done += bs

    # ── Save grids ───────────────────────────────────────────────────────
    grid_32 = make_grid(all_32, nrow=min(8, len(all_32)))
    grid_64 = make_grid(all_64, nrow=min(8, len(all_64)))
    grid_32.save(os.path.join(args.out_dir, 'grid_32.png'))
    grid_64.save(os.path.join(args.out_dir, 'grid_64.png'))
    print(f"\nDone! Saved {n_done} samples to {args.out_dir}")
    print(f"  Base:    {args.base_steps} Heun steps")
    print(f"  SR:      {args.sr_steps} Heun steps")


def main():
    parser = argparse.ArgumentParser(description='EDM cascaded sampling (32→64)')
    parser.add_argument('--base_model',   required=True,
                        help='Base model .pkl checkpoint')
    parser.add_argument('--sr_model',     required=True,
                        help='SR model .pkl checkpoint')
    parser.add_argument('--num_samples',  type=int, default=16)
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--base_steps',   type=int, default=18,
                        help='Number of Heun steps for base model')
    parser.add_argument('--sr_steps',     type=int, default=18,
                        help='Number of Heun steps for SR model')
    parser.add_argument('--out_dir',      default='samples/edm_afhq/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_cascade(args, device)


if __name__ == '__main__':
    main()
