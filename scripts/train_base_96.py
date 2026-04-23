"""
train_base_96.py
----------------
Trains the BASE diffusion model on 96×96 MRI slices.

Data:
    .pt files from /data/Sahil_dataset/MRI_processed/train/AXT2_normalized/
    Each file: {'slices': (S, 2, 384, 384), 'global_scale': scalar}
    We compute magnitude = sqrt(real² + imag²), then average-pool to 96×96.

Architecture: UNET_SMALL  (appropriate for small datasets)
Conditioning: NONE  (single contrast AXT2, unconditional)
Channels:     1 (grayscale magnitude)

Usage:
    conda run -n fastmri python scripts/train_base_96.py \
        --pt_dir      /data/Sahil_dataset/MRI_processed/train/AXT2_normalized \
        --save_dir    checkpoints/base_96 \
        --batch_size  8 \
        --lr          1e-4 \
        --save_interval 5000

    Or use the launch script:  bash scripts/run_train_base.sh
"""

import argparse
import os
import sys

# ── Make guided_diffusion importable ─────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GD_PATH = os.path.join(_REPO_ROOT, 'guided_diffusion_repo')
if _GD_PATH not in sys.path:
    sys.path.insert(0, _GD_PATH)

import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

# ── Import our custom dataloader ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mri_dataloader import load_mri_data


# ─────────────────────────────────────────────────────────────────────────────
# UNET_SMALL config for 96×96 single-channel MRI magnitude images
# ─────────────────────────────────────────────────────────────────────────────
UNET_SMALL = dict(
    image_size            = 96,
    num_channels          = 64,       # small vs 192 in paper's full model
    num_res_blocks        = 2,
    num_heads             = 4,
    num_head_channels     = 32,
    num_heads_upsample    = -1,
    attention_resolutions = '16',     # attend at 1/6× (96//16 = 6)
    channel_mult          = '',       # auto-selected (1,2,2,2) for 96
    dropout               = 0.1,
    class_cond            = False,
    use_checkpoint        = False,
    use_scale_shift_norm  = True,
    resblock_updown       = True,
    use_new_attention_order = True,
    use_fp16              = False,
    in_channels           = 1,        # ← grayscale magnitude
    learn_sigma           = True,     # predict both mean and variance
)

DIFFUSION_CONFIG = dict(
    diffusion_steps    = 1000,
    noise_schedule     = 'cosine',    # cosine for base model (paper §3)
    timestep_respacing = '',
    use_kl             = False,
    predict_xstart     = False,
    rescale_timesteps  = False,
    rescale_learned_sigmas = False,
    learn_sigma        = True,
)


def create_argparser():
    defaults = dict(
        pt_dir            = '/data/Sahil_dataset/MRI_processed/train/AXT2_normalized',
        save_dir          = 'checkpoints/base_96',
        batch_size        = 8,
        microbatch        = -1,       # -1 = same as batch_size
        lr                = 1e-4,
        ema_rate          = '0.9999',
        log_interval      = 100,
        save_interval     = 5000,
        resume_checkpoint = '',
        use_fp16          = False,
        fp16_scale_growth = 1e-3,
        schedule_sampler  = 'uniform',
        weight_decay      = 0.0,
        lr_anneal_steps   = 0,
        num_workers       = 4,
    )
    defaults.update(UNET_SMALL)
    defaults.update(DIFFUSION_CONFIG)
    parser = argparse.ArgumentParser(description='Train 96x96 MRI base diffusion model')
    add_dict_to_argparser(parser, defaults)
    # add_dict_to_argparser doesn't know about pt_dir / save_dir / num_workers
    # if they're already added from defaults that's fine; if not, add them:
    try:
        parser.add_argument('--pt_dir',      default=defaults['pt_dir'])
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--save_dir',    default=defaults['save_dir'])
    except argparse.ArgumentError:
        pass
    try:
        parser.add_argument('--num_workers', type=int, default=defaults['num_workers'])
    except argparse.ArgumentError:
        pass
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log('Creating model and diffusion…')

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log(f'Loading data from {args.pt_dir}…')
    data = load_mri_data(
        pt_dir=args.pt_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Model parameters: {n_params:,}')
    logger.log('Training…')

    TrainLoop(
        model             = model,
        diffusion         = diffusion,
        data              = data,
        batch_size        = args.batch_size,
        microbatch        = args.microbatch,
        lr                = args.lr,
        ema_rate          = args.ema_rate,
        log_interval      = args.log_interval,
        save_interval     = args.save_interval,
        resume_checkpoint = args.resume_checkpoint,
        use_fp16          = args.use_fp16,
        fp16_scale_growth = args.fp16_scale_growth,
        schedule_sampler  = schedule_sampler,
        weight_decay      = args.weight_decay,
        lr_anneal_steps   = args.lr_anneal_steps,
    ).run_loop()


if __name__ == '__main__':
    main()
