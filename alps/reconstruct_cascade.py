"""
alps/reconstruct_cascade.py
---------------------------
CLI entry point for cascaded ALPS reconstruction of 2-channel MRI.

Mirrors the style of scripts/sample_mri_cascade.py but performs
posterior sampling from undersampled k-space rather than unconditional
generation.

Usage
-----
Single GPU:
    python alps/reconstruct_cascade.py \\
        --base_ckpt  checkpoints/edm_mri_base_96/network-snapshot-200000.pkl \\
        --sr_ckpt    checkpoints/edm_mri_sr_384/network-snapshot-200000.pkl \\
        --data_dir   /data/MRI_processed/test/AXT2_normalized \\
        --outdir     outputs/alps_cascade \\
        --accel      4

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=2 alps/reconstruct_cascade.py ...

Input format
------------
The dataloader reads the same .pt files used during training:
    { 'slices': (S, 2, 384, 384) float32,  'global_scale': scalar }

The fully-sampled 384×384 k-space is computed internally by FFT of each
slice.  Undersampling masks are generated on the fly.

Output format
-------------
For each reconstructed slice, three files are saved:

    <outdir>/<case>_s<slice>_x96_2ch.pt      — Stage 1 output (2,  96,  96)
    <outdir>/<case>_s<slice>_x384_2ch.pt     — Stage 2 output (2, 384, 384)
    <outdir>/<case>_s<slice>_comparison.png  — side-by-side magnitude figure
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── make edm_repo and scripts/ importable ─────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [os.path.join(_ROOT, 'edm_repo'),
           os.path.join(_ROOT, 'scripts')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from edm_sr_model import EDMSRPrecond          # noqa — needed for pickle
from training.networks import EDMPrecond        # noqa — needed for pickle

from alps.sampling  import cascaded_ALPS, ALPSOptions
from alps.operators import MRIOperator, make_cartesian_mask    # for Fourier cropping helper


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def _image_to_kspace(image_2ch: torch.Tensor) -> torch.Tensor:
    """
    2-channel image → 2-channel k-space via 2D FFT.

    image_2ch : (B, 2, H, W)
    returns   : (B, 2, H, W)
    """
    kc = torch.fft.fft2(_to_complex(image_2ch), norm='ortho')
    return torch.stack([kc.real, kc.imag], dim=1)


def magnitude(x: torch.Tensor) -> np.ndarray:
    """(B, 2, H, W) → (B, H, W) magnitude, normalised to [0,1]."""
    re, im = x[:, 0], x[:, 1]
    mag    = torch.sqrt(re ** 2 + im ** 2).float().cpu()
    b      = mag.shape[0]
    mn     = mag.view(b, -1).min(1).values.view(b, 1, 1)
    mx     = mag.view(b, -1).max(1).values.view(b, 1, 1)
    return ((mag - mn) / (mx - mn + 1e-8)).numpy()


def save_comparison(zf_96: torch.Tensor,
                    x96:   torch.Tensor,
                    x384:  torch.Tensor,
                    path:  str) -> None:
    """Save a 4-panel magnitude figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping figure.")
        return

    # zero-fill 96→384 for visual comparison
    zf_up  = F.interpolate(zf_96.float(), size=384, mode='nearest')

    panels = [
        (magnitude(zf_96)[0],  "Zero-fill 96×96"),
        (magnitude(zf_up)[0],  "ZF up-sampled 384×384"),
        (magnitude(x96)[0],    "ALPS Stage 1 96×96"),
        (magnitude(x384)[0],   "ALPS Stage 2 384×384"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap='gray', interpolation='bilinear')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset iterator
# ─────────────────────────────────────────────────────────────────────────────

def iter_slices(data_dir: str, max_slices: int = None):
    """
    Yield (case_name, slice_idx, slice_2ch) from .pt files in data_dir.

    slice_2ch : (1, 2, 384, 384) float32 in [-1, 1]
    """
    pt_files = sorted(Path(data_dir).glob('*.pt'))
    assert len(pt_files) > 0, f"No .pt files found in {data_dir}"

    count = 0
    for fpath in pt_files:
        try:
            obj = torch.load(str(fpath), map_location='cpu')
        except Exception as e:
            print(f"  WARNING: skipping {fpath.name} ({e})")
            continue

        slices = obj.get('slices', None)   # (S, 2, 384, 384)
        if slices is None:
            continue

        case = fpath.stem
        for s in range(slices.shape[0]):
            yield case, s, slices[s:s+1]   # (1, 2, 384, 384)
            count += 1
            if max_slices is not None and count >= max_slices:
                return


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(pkl_path: str, device: torch.device) -> torch.nn.Module:
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    net = data['ema'].to(device).eval()
    n_p = sum(p.numel() for p in net.parameters())
    print(f"  {n_p:,} parameters  |  img_resolution={net.img_resolution}"
          f"  img_channels={net.img_channels}")
    return net


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    net_base = load_model(args.base_ckpt, device)
    net_sr   = load_model(args.sr_ckpt,   device)

    # ── ALPS options ──────────────────────────────────────────────────────────
    opts_base = ALPSOptions(
        num_steps = args.steps_base,
        sigma_max = args.sigma_max,
        sigma_min = args.sigma_min,
        rho       = args.rho,
        K         = args.K,
    )
    opts_sr = ALPSOptions(
        num_steps = args.steps_sr,
        sigma_max = args.sigma_max,
        sigma_min = args.sigma_min,
        rho       = args.rho,
        K         = args.K,
    )

    # ── Process slices ────────────────────────────────────────────────────────
    for case, sidx, slice_2ch in iter_slices(args.data_dir, args.max_slices):
        print(f"\n{'─'*60}")
        print(f"  Case: {case}  |  Slice: {sidx}")
        print(f"{'─'*60}")

        slice_2ch = slice_2ch.to(device)   # (1, 2, 384, 384)

        # Simulate fully sampled k-space (FFT of magnitude image)
        full_kspace = _image_to_kspace(slice_2ch)   # (1, 2, 384, 384)

        # Run cascaded ALPS
        result = cascaded_ALPS(
            full_kspace_384 = full_kspace,
            net_base        = net_base,
            net_sr          = net_sr,
            opts_base       = opts_base,
            opts_sr         = opts_sr,
            acceleration    = args.accel,
            acs_fraction    = args.acs_fraction,
            eta             = args.eta,
            seed            = args.seed,
            device          = device,
            store_stage1    = False,
            store_stage2    = False,
            verbose         = True,
        )

        x96  = result['x96']    # (1, 2,  96,  96)
        x384 = result['x384']   # (1, 2, 384, 384)

        # Zero-filled 96×96 for reference (single-coil: ones coil map).
        kc      = torch.complex(full_kspace[:, 0], full_kspace[:, 1]).unsqueeze(1)  # (1,1,384,384)
        csm1    = torch.ones_like(kc)
        csm1_96 = csm1[..., ::4, ::4]
        mask_96 = make_cartesian_mask(96, args.accel, args.acs_fraction, args.seed)
        A1 = MRIOperator(
            csm=csm1_96, mask=mask_96,
            eta=args.eta, fft_scale=full_kspace.shape[-1] / 96, device=device,
        )
        y1  = A1.get_measurements(kc)
        zf  = A1.adjoint(y1)   # zero-filled 96×96

        # ── Save ──────────────────────────────────────────────────────────────
        prefix = os.path.join(args.outdir, f"{case}_s{sidx:03d}")

        torch.save(x96.cpu(),  f"{prefix}_x96_2ch.pt")
        torch.save(x384.cpu(), f"{prefix}_x384_2ch.pt")
        save_comparison(zf, x96, x384, f"{prefix}_comparison.png")

        print(f"  Saved: {prefix}_*.pt  +  comparison.png")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cascaded ALPS MRI reconstruction (96×96 → 384×384)'
    )

    # Checkpoints
    parser.add_argument('--base_ckpt', required=True,
                        help='Path to Stage 1 (96×96) EDM checkpoint .pkl')
    parser.add_argument('--sr_ckpt',   required=True,
                        help='Path to Stage 2 (384×384 SR) EDM checkpoint .pkl')

    # Data
    parser.add_argument('--data_dir',   required=True,
                        help='Directory of .pt MRI files (same format as training)')
    parser.add_argument('--outdir',     default='outputs/alps_cascade',
                        help='Output directory')
    parser.add_argument('--max_slices', type=int, default=None,
                        help='Limit total slices processed (default: all)')

    # Forward model
    parser.add_argument('--accel',       type=int,   default=4,
                        help='Cartesian acceleration factor (default: 4)')
    parser.add_argument('--acs_fraction',type=float, default=0.08,
                        help='Fraction of k-space centre always acquired (default: 0.08)')
    parser.add_argument('--eta',         type=float, default=0.01,
                        help='Noise standard deviation (default: 0.01)')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='Mask generation seed (default: 42)')

    # ALPS hyperparameters
    parser.add_argument('--steps_base', type=int,   default=20,
                        help='ALPS noise levels for Stage 1 (default: 20)')
    parser.add_argument('--steps_sr',   type=int,   default=20,
                        help='ALPS noise levels for Stage 2 (default: 20)')
    parser.add_argument('--sigma_max',  type=float, default=80.0)
    parser.add_argument('--sigma_min',  type=float, default=0.002)
    parser.add_argument('--rho',        type=float, default=7.0)
    parser.add_argument('--K',          type=int,   default=1,
                        help='Langevin inner steps per noise level (default: 1)')

    # Hardware
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    main(args)
