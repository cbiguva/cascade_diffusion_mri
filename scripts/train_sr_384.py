"""
train_sr_384.py
---------------
Trains the SUPER-RESOLUTION diffusion model:
    Condition:  96×96 average-pooled image → NN-upsampled → 384×384  (blurry)
    Target:     384×384 full-resolution MRI

Data:
    Same .pt files as the base model.
    We produce both HR target and blurry condition on-the-fly from each slice.

This model is trained INDEPENDENTLY of the base model.
During training the condition comes from REAL data (not base-model samples).
During inference, the base model's 96×96 output is NN-upsampled and fed here.

Architecture: UNET_SMALL  (SR variant — receives concat[noisy_384 | condition_384])
Scheduling:   linear  (SR models, paper §3)

Usage:
    conda run -n fastmri python scripts/train_sr_384.py \
        --pt_dir      /data/Sahil_dataset/MRI_processed/train/AXT2_normalized \
        --save_dir    checkpoints/sr_384 \
        --batch_size  2 \
        --lr          1e-4

    Or use the launch script:  bash scripts/run_train_sr.sh
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
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mri_dataloader import load_mri_sr_data



SR_UNET_SMALL = dict(
    large_size            = 384,      # output (and noisy input) resolution
    small_size            = 96,       # low-res input resolution (used by dataloader)
    num_channels          = 64,       # small model
    num_res_blocks        = 2,
    num_heads             = 4,
    num_head_channels     = 32,
    num_heads_upsample    = -1,
    attention_resolutions = '32,16',  # attend at 384//32=12, 384//16=24
    channel_mult          = '',       # auto-selected (1,1,2,2,4,4) for 384
    dropout               = 0.1,
    class_cond            = False,
    use_checkpoint        = False,
    use_scale_shift_norm  = True,
    resblock_updown       = True,
    use_fp16              = False,
    # SuperResModel internally doubles in_channels to cat[noisy | condition].
    # We set in_channels=1 here; the model will use 2 internally.
    in_channels           = 1,        # ← grayscale magnitude
    learn_sigma           = True,
)

SR_DIFFUSION_CONFIG = dict(
    diffusion_steps       = 1000,
    noise_schedule        = 'linear', # linear for SR model (paper §3)
    timestep_respacing    = '',
    use_kl                = False,
    predict_xstart        = False,
    rescale_timesteps     = False,
    rescale_learned_sigmas = False,
    learn_sigma           = True,
)


def create_argparser():
    defaults = dict(
        pt_dir            = '/data/Sahil_dataset/MRI_processed/train/AXT2_normalized',
        save_dir          = 'checkpoints/sr_384',
        batch_size        = 2,        # 384×384 is memory-heavy
        microbatch        = -1,
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
    defaults.update(SR_UNET_SMALL)
    defaults.update(SR_DIFFUSION_CONFIG)
    parser = argparse.ArgumentParser(description='Train 384x384 MRI SR diffusion model')
    add_dict_to_argparser(parser, defaults)
    for key, val in [('pt_dir', defaults['pt_dir']),
                     ('save_dir', defaults['save_dir']),
                     ('num_workers', 4)]:
        try:
            parser.add_argument(f'--{key}', default=val,
                                type=type(val) if not isinstance(val, str) else str)
        except argparse.ArgumentError:
            pass
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)
    logger.log('Creating SR model and diffusion…')

    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log(f'Loading paired data from {args.pt_dir}…')
    data = load_mri_sr_data(
        pt_dir=args.pt_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Model parameters: {n_params:,}')
    logger.log('Training SR model…')

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
