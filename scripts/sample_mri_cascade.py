"""
sample_mri_cascade.py
----------------------
TWO-STEP CASCADED INFERENCE:

  Step 1: Base model generates 96×96 MRI from pure noise
  Step 2: SR model takes Step 1 output → generates 384×384 MRI

Saves individual PNG files and a summary grid.

Usage:
    python scripts/sample_mri_cascade.py \
        --base_model     checkpoints/base_96/ema_0.9999_XXXXXX.pt \
        --sr_model       checkpoints/sr_384/ema_0.9999_XXXXXX.pt \
        --num_samples    16 \
        --batch_size     4 \
        --timestep_respacing 250 \
        --out_dir        samples/
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import math

# ── Make guided_diffusion importable ─────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GD_PATH = os.path.join(_REPO_ROOT, 'guided_diffusion_repo')
if _GD_PATH not in sys.path:
    sys.path.insert(0, _GD_PATH)

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


# ─────────────────────────────────────────────────────────────
# Must match exactly what you used in training
# ─────────────────────────────────────────────────────────────
BASE_MODEL_CONFIG = dict(
    image_size            = 96,
    num_channels          = 64,
    num_res_blocks        = 2,
    num_heads             = 4,
    num_head_channels     = 32,
    num_heads_upsample    = -1,
    attention_resolutions = '16',
    channel_mult          = '',
    dropout               = 0.0,         # 0 at inference
    class_cond            = False,
    use_checkpoint        = False,
    use_scale_shift_norm  = True,
    resblock_updown       = True,
    use_new_attention_order = True,
    use_fp16              = False,
    in_channels           = 1,           # grayscale magnitude MRI
    learn_sigma           = True,
    diffusion_steps       = 1000,
    noise_schedule        = 'cosine',
    # diffusion extras
    use_kl                = False,
    predict_xstart        = False,
    rescale_timesteps     = False,
    rescale_learned_sigmas = False,
)

SR_MODEL_CONFIG = dict(
    large_size            = 384,
    small_size            = 96,
    num_channels          = 64,
    num_res_blocks        = 2,
    num_heads             = 4,
    num_head_channels     = 32,
    num_heads_upsample    = -1,
    attention_resolutions = '32,16',
    channel_mult          = '',
    dropout               = 0.0,
    class_cond            = False,
    use_checkpoint        = False,
    use_scale_shift_norm  = True,
    resblock_updown       = True,
    use_fp16              = False,
    in_channels           = 1,           # grayscale magnitude MRI
    learn_sigma           = True,
    diffusion_steps       = 1000,
    noise_schedule        = 'linear',
    # diffusion extras
    use_kl                = False,
    predict_xstart        = False,
    rescale_timesteps     = False,
    rescale_learned_sigmas = False,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def tensor_to_uint16(x):
    """(B, 1, H, W) in [-1,1] → (B, H, W) uint16."""
    x = ((x + 1) / 2).clamp(0, 1)          # [0, 1]
    x = (x * 65535).to(torch.int32)         # [0, 65535]
    return x.squeeze(1).cpu().numpy().astype(np.uint16)


def save_png16(arr, path):
    Image.fromarray(arr, mode='I;16').save(path)


def make_grid(arrays, ncols=4):
    """arrays: list of (H,W) uint16.  Returns one big (H*rows, W*ncols) image."""
    n     = len(arrays)
    nrows = math.ceil(n / ncols)
    H, W  = arrays[0].shape
    grid  = np.zeros((nrows * H, ncols * W), dtype=np.uint16)
    for i, arr in enumerate(arrays):
        r, c = divmod(i, ncols)
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = arr
    return grid


# ─────────────────────────────────────────────────────────────
# Main sampling loop
# ─────────────────────────────────────────────────────────────

def sample_cascade(args, device):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / '96').mkdir(exist_ok=True)
    (out_dir / '384').mkdir(exist_ok=True)

    # ── Load BASE model ───────────────────────────────────────
    print('Loading base 96\u00d796 model\u2026')
    base_cfg = {**BASE_MODEL_CONFIG,
                'timestep_respacing': args.timestep_respacing}
    model_base, diff_base = create_model_and_diffusion(**base_cfg)
    state = torch.load(args.base_model, map_location='cpu', weights_only=True)
    model_base.load_state_dict(state)
    model_base.to(device).eval()

    # ── Load SR model ─────────────────────────────────────────
    print('Loading SR 96\u2192384 model\u2026')
    sr_cfg = {**SR_MODEL_CONFIG,
              'timestep_respacing': args.sr_timestep_respacing}
    model_sr, diff_sr = sr_create_model_and_diffusion(**sr_cfg)
    state_sr = torch.load(args.sr_model, map_location='cpu', weights_only=True)
    model_sr.load_state_dict(state_sr)
    model_sr.to(device).eval()

    all_96, all_384 = [], []
    done = 0

    with torch.no_grad():
        while done < args.num_samples:
            bs = min(args.batch_size, args.num_samples - done)

            # ── STEP 1: generate 96×96 ────────────────────────
            print(f'  Step 1: sampling {bs} × 96×96…')
            sample_96 = diff_base.p_sample_loop(
                model_base,
                shape=(bs, 1, 96, 96),
                clip_denoised=True,
                model_kwargs={},
                device=device,
                progress=True,
            )                                            # (B,1,96,96) in [-1,1]

            # ── STEP 2: upsample 96→384 and condition SR model ─
            # Nearest-neighbour upsample to 384 (same as training)
            low_res = F.interpolate(
                sample_96, size=(384, 384), mode='nearest'
            )                                            # (B,1,384,384)

            print(f'  Step 2: upsampling to 384×384…')
            sample_384 = diff_sr.p_sample_loop(
                model_sr,
                shape=(bs, 1, 384, 384),
                clip_denoised=True,
                model_kwargs={'low_res': low_res},       # conditioning
                device=device,
                progress=True,
            )                                            # (B,1,384,384)

            # ── Save ──────────────────────────────────────────
            imgs_96  = tensor_to_uint16(sample_96)      # (B,96,96)
            imgs_384 = tensor_to_uint16(sample_384)     # (B,384,384)

            for i in range(bs):
                idx = done + i
                save_png16(imgs_96[i],  out_dir / '96'  / f'sample_{idx:04d}.png')
                save_png16(imgs_384[i], out_dir / '384' / f'sample_{idx:04d}.png')

            all_96.extend(imgs_96)
            all_384.extend(imgs_384)
            done += bs
            print(f'  {done}/{args.num_samples} done')

    # ── Save grids ────────────────────────────────────────────
    print('Saving summary grids…')
    grid_96  = make_grid(all_96,  ncols=min(8, args.num_samples))
    grid_384 = make_grid(all_384, ncols=min(4, args.num_samples))
    save_png16(grid_96,  out_dir / 'grid_96.png')
    save_png16(grid_384, out_dir / 'grid_384.png')
    print(f'\nDone. Samples saved to {out_dir}/')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model',  required=True,
                        help='Path to base 96x96 EMA checkpoint (.pt)')
    parser.add_argument('--sr_model',    required=True,
                        help='Path to SR 96→384 EMA checkpoint (.pt)')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--timestep_respacing',    default='250',
                        help='Faster sampling for base model (e.g. 250)')
    parser.add_argument('--sr_timestep_respacing', default='250',
                        help='Faster sampling for SR model')
    parser.add_argument('--out_dir',     default='samples/')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


if __name__ == '__main__':
    args   = create_argparser().parse_args()
    device = torch.device(args.device)
    sample_cascade(args, device)
