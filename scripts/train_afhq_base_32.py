"""
train_afhq_base_32.py
---------------------
BASE diffusion model on 32×32 AFHQ animal face images.

Data:
    Standard image files in ImageFolder layout:
        data/afhq/train/cat/  dog/  wild/

        --data_dir    data/afhq/train \\
        --save_dir    checkpoints/afhq_base_32 \\
        --batch_size  32 \\
        --lr          1e-4 \\
        --save_interval 5000

   
"""

import argparse
import os
import sys

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afhq_dataloader import load_afhq_data



UNET_CONFIG = dict(
    image_size            = 32,
    num_channels          = 128,      # for this small experiment
    num_res_blocks        = 2,
    num_heads             = 4,
    num_head_channels     = 32,
    num_heads_upsample    = -1,
    attention_resolutions = '16',     # attend at 32//16 = 2× downsampled
    channel_mult          = '1,2,3,4',       # auto-selected (1,2,2) for 32
    dropout               = 0.1,
    class_cond            = False,
    use_checkpoint        = False,
    use_scale_shift_norm  = True,
    resblock_updown       = True,
    use_new_attention_order = True,
    use_fp16              = False,
    in_channels           = 3,        #  RGB
    learn_sigma           = True,     # predict both mean and variance
)

DIFFUSION_CONFIG = dict(
    diffusion_steps    = 1000,
    noise_schedule     = 'cosine',    # cosine for base model (CDM paper)
    timestep_respacing = '',
    use_kl             = False,
    predict_xstart     = False,
    rescale_timesteps  = False,
    rescale_learned_sigmas = False,
    learn_sigma        = True,
)


def create_argparser():
    defaults = dict(
        data_dir          = 'data/afhq',
        save_dir          = 'checkpoints/afhq_base_32',
        batch_size        = 256,
        microbatch        = 128,       # -1 = same as batch_size
        lr                = 1e-4,
        ema_rate          = '0.9999',
        log_interval      = 100,
        save_interval     = 500,
        resume_checkpoint = '',
        use_fp16          = False,
        fp16_scale_growth = 1e-3,
        schedule_sampler  = 'uniform',
        weight_decay      = 0.0,
        lr_anneal_steps   = 0,
        num_workers       = 4,
    )
    defaults.update(UNET_CONFIG)
    defaults.update(DIFFUSION_CONFIG)
    parser = argparse.ArgumentParser(description='Train 32x32 AFHQ base diffusion model')
    add_dict_to_argparser(parser, defaults)
    for key, val in [('data_dir', defaults['data_dir']),
                     ('save_dir', defaults['save_dir']),
                     ('num_workers', defaults['num_workers'])]:
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
    logger.log('Creating model and diffusion…')

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )

    logger.log(f'Loading AFHQ data from {args.data_dir}…')
    data = load_afhq_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Model parameters: {n_params:,}')
    logger.log(f'Image size: {args.image_size}×{args.image_size}, channels: {args.in_channels}')
    logger.log(f'Noise schedule: {args.noise_schedule}')
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
