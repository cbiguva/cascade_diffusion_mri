"""
sample_afhq_base_32.py
----------------------
Sample from the trained AFHQ base model (32×32).

Usage:
    python scripts/sample_afhq_base_32.py \
        --model checkpoints/afhq_base_32/ema_0.9999_510000.pt \
        --num_samples 16 --batch_size 8 \
        --timestep_respacing 250 \
        --out_dir samples/afhq_base_32/
"""

import argparse, os, sys, math

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GD_PATH = os.path.join(_REPO_ROOT, 'guided_diffusion_repo')
if _GD_PATH not in sys.path:
    sys.path.insert(0, _GD_PATH)

import torch
from PIL import Image
from guided_diffusion.script_util import (
    create_model_and_diffusion, model_and_diffusion_defaults,
)


# ── Must match training config in train_afhq_base_32.py ──────────────────────
BASE_CONFIG = dict(
    image_size=32, num_channels=128, num_res_blocks=2,
    num_heads=4, num_head_channels=32, num_heads_upsample=-1,
    attention_resolutions='16', channel_mult='1,2,3,4', dropout=0.0,
    class_cond=False, use_checkpoint=False, use_scale_shift_norm=True,
    resblock_updown=True, use_new_attention_order=True, use_fp16=False,
    in_channels=3, learn_sigma=True,
    diffusion_steps=1000, noise_schedule='cosine',
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


def sample_base(args, device):
    # ── Load model ────────────────────────────────────────────────────────
    print("Loading base model…")
    cfg = dict(BASE_CONFIG)
    cfg['timestep_respacing'] = args.timestep_respacing
    model, diffusion = create_model_and_diffusion(
        **{k: v for k, v in cfg.items()
           if k in model_and_diffusion_defaults()}
    )
    state = torch.load(args.model, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded ({n_params:,} params)")
    print(f"  Diffusion steps: {diffusion.num_timesteps} "
          f"(respaced from {cfg['diffusion_steps']})")

    # ── Sample ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    all_imgs = []
    n_done = 0

    while n_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - n_done)
        print(f"\nSampling batch {n_done}..{n_done+bs} / {args.num_samples}")

        with torch.no_grad():
            samples = diffusion.p_sample_loop(
                model,
                shape=(bs, 3, 32, 32),
                clip_denoised=True,
                model_kwargs={},
                device=device,
                progress=True,
            )  # (B, 3, 32, 32) in [-1, 1]

        for i in range(bs):
            idx = n_done + i
            img = tensor_to_pil(samples[i])
            img.save(os.path.join(args.out_dir, f'sample_{idx:04d}.png'))
            all_imgs.append(img)

        n_done += bs

    # ── Save grid ─────────────────────────────────────────────────────────
    grid = make_grid(all_imgs, nrow=min(8, len(all_imgs)))
    grid.save(os.path.join(args.out_dir, 'grid.png'))
    print(f"\nDone! Saved {n_done} samples + grid.png to {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Sample from the 32×32 AFHQ base diffusion model'
    )
    parser.add_argument('--model', required=True,
                        help='Path to base model checkpoint (e.g. ema_0.9999_510000.pt)')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for sampling')
    parser.add_argument('--timestep_respacing', default='250',
                        help='Number of denoising steps (e.g. 250, 100, ddim25)')
    parser.add_argument('--out_dir', default='samples/afhq_base_32/',
                        help='Output directory for generated images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_base(args, device)


if __name__ == '__main__':
    main()
