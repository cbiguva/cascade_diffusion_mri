"""
sample_afhq_cascade.py
----------------------
Cascaded inference: Base(32x32) -> NN-upsample -> SR(64x64).

Usage:
    python scripts/sample_afhq_cascade.py \
        --base_model   checkpoints/afhq_base_32/ema_0.9999_100000.pt \
        --sr_model     checkpoints/afhq_sr_64/ema_0.9999_100000.pt \
        --num_samples  16 --batch_size 8 \
        --timestep_respacing 250 --sr_timestep_respacing 250 \
        --out_dir samples/afhq/
"""

import argparse, os, sys, math
import numpy as np
import torch
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GD_PATH = os.path.join(_REPO_ROOT, 'guided_diffusion_repo')
if _GD_PATH not in sys.path:
    sys.path.insert(0, _GD_PATH)

import torch
import torch.nn.functional as F
from PIL import Image
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    create_model_and_diffusion, model_and_diffusion_defaults,
    sr_create_model_and_diffusion, sr_model_and_diffusion_defaults,
    create_gaussian_diffusion,
)


# ── Base model config (must match training) ──────────────────────────────────
BASE_CONFIG = dict(
    image_size=32, 
    num_channels=128, 
    num_res_blocks=2,
    num_heads=4, 
    num_head_channels=32, num_heads_upsample=-1,
    attention_resolutions='16', channel_mult='1,2,3,4', dropout=0.1,
    class_cond=False, use_checkpoint=False, use_scale_shift_norm=True,
    resblock_updown=True, use_new_attention_order=True, use_fp16=False,
    in_channels=3, learn_sigma=True,
    diffusion_steps=1000, noise_schedule='cosine',
    use_kl=False, predict_xstart=False,
    rescale_timesteps=False, rescale_learned_sigmas=False,
)

# ── SR model config (must match training) ────────────────────────────────────
SR_CONFIG = dict(
    large_size=64, small_size=32,
    num_channels=64, num_res_blocks=3,
    num_heads=4, num_head_channels=32, num_heads_upsample=-1,
    attention_resolutions='16', channel_mult='1244', dropout=0.1,
    class_cond=False, use_checkpoint=False, use_scale_shift_norm=True,
    resblock_updown=True, use_fp16=False, in_channels=3, learn_sigma=True,
    diffusion_steps=1000, noise_schedule='linear',
    use_kl=False, predict_xstart=False,
    rescale_timesteps=False, rescale_learned_sigmas=False,
)


def tensor_to_pil(t):
    """Convert (C, H, W) tensor in [-1,1] to PIL RGB image."""
    t = ((t + 1.0) * 127.5).clamp(0, 255).byte()
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
    """NN-upsample (B,C,32,32) -> (B,C,64,64) using repeat_interleave."""
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def sample_cascade(args, device):
    # ── Load base model ──────────────────────────────────────────────────
    print("Loading base model…")
    base_cfg = dict(BASE_CONFIG)
    base_cfg['timestep_respacing'] = args.timestep_respacing
    model_base, diff_base = create_model_and_diffusion(
        **{k: v for k, v in base_cfg.items()
           if k in model_and_diffusion_defaults()
           or k in ('in_channels',)}
    )
    state = torch.load(args.base_model, map_location='cpu')
    model_base.load_state_dict(state)
    model_base.to(device).eval()
    print(f"  Base model loaded ({sum(p.numel() for p in model_base.parameters()):,} params)")

    # ── Load SR model ────────────────────────────────────────────────────
    print("Loading SR model…")
    sr_cfg = dict(SR_CONFIG)
    sr_cfg['timestep_respacing'] = args.sr_timestep_respacing
    model_sr, diff_sr = sr_create_model_and_diffusion(
        **{k: v for k, v in sr_cfg.items()
           if k in sr_model_and_diffusion_defaults()
           or k in ('in_channels',)}
    )
    state = torch.load(args.sr_model, map_location='cpu')
    model_sr.load_state_dict(state)
    model_sr.to(device).eval()
    print(f"  SR model loaded ({sum(p.numel() for p in model_sr.parameters()):,} params)")

    # ── Sample ───────────────────────────────────────────────────────────
    os.makedirs(os.path.join(args.out_dir, '32'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, '64'), exist_ok=True)

    all_32, all_64 = [], []
    n_done = 0

    while n_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - n_done)
        print(f"\nSampling batch {n_done}..{n_done+bs} / {args.num_samples}")

        # Step 1: Base model → 32×32
        with torch.no_grad():
            sample_32 = diff_base.p_sample_loop(
                model_base,
                shape=(bs, 3, 32, 32),
                clip_denoised=True,
                model_kwargs={},
                device=device,
            )  # (B, 3, 32, 32) in [-1, 1]

        # Step 2: NN-upsample 32→64
        low_res = nn_upsample_2x(sample_32)  # (B, 3, 64, 64)

        # Step 3: SR model → 64×64
        with torch.no_grad():
            sample_64 = diff_sr.p_sample_loop(
                model_sr,
                shape=(bs, 3, 64, 64),
                clip_denoised=True,
                model_kwargs={'low_res': low_res},
                device=device,
            )  # (B, 3, 64, 64) in [-1, 1]

        # Save individual images
        for i in range(bs):
            idx = n_done + i
            img32 = tensor_to_pil(sample_32[i])
            img64 = tensor_to_pil(sample_64[i])
            img32.save(os.path.join(args.out_dir, '32', f'sample_{idx:04d}.png'))
            img64.save(os.path.join(args.out_dir, '64', f'sample_{idx:04d}.png'))
            all_32.append(img32)
            all_64.append(img64)

        n_done += bs

    # Save grids
    grid_32 = make_grid(all_32, nrow=min(8, len(all_32)))
    grid_64 = make_grid(all_64, nrow=min(8, len(all_64)))
    grid_32.save(os.path.join(args.out_dir, 'grid_32.png'))
    grid_64.save(os.path.join(args.out_dir, 'grid_64.png'))
    print(f"\nDone! Saved {n_done} samples to {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description='AFHQ cascaded sampling')
    parser.add_argument('--base_model', required=True, help='Base model checkpoint')
    parser.add_argument('--sr_model', required=True, help='SR model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--timestep_respacing', default='250')
    parser.add_argument('--sr_timestep_respacing', default='250')
    parser.add_argument('--out_dir', default='samples/afhq/')
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_cascade(args, device)


if __name__ == '__main__':
    main()
