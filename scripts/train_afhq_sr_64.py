"""
train_afhq_sr_64.py
-------------------
 SUPER-RESOLUTION diffusion model (32->64) on AFHQ.

Gaussian noise on low_res input.
SuperResModel (concat[noisy_64 | condition_64])
Schedule: linear (SR models, CDM paper)

        --data_dir data/afhq/train --save_dir checkpoints/afhq_sr_64 \
        --batch_size 16 --lr 1e-4
"""


import argparse, os, sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GD_PATH = os.path.join(_REPO_ROOT, 'guided_diffusion_repo')
if _GD_PATH not in sys.path:
    sys.path.insert(0, _GD_PATH)


import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults, sr_create_model_and_diffusion,
    args_to_dict, add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afhq_dataloader import load_afhq_sr_data

SR_UNET_CONFIG = dict(
    large_size=64,
    small_size=32,
    num_channels=64, 
    num_res_blocks=3,
    num_heads=4, 
    num_head_channels=32, 
    num_heads_upsample=-1,
    attention_resolutions='16', 
    channel_mult='1244', 
    dropout=0.1,
    class_cond=False, 
    use_checkpoint=False, 
    use_scale_shift_norm=True,
    resblock_updown=True, 
    use_fp16=False, 
    in_channels=3, 
    learn_sigma=True,
)

SR_DIFFUSION_CONFIG = dict(
    diffusion_steps=1000, 
    noise_schedule='linear', 
    timestep_respacing='',
    use_kl=False, 
    predict_xstart=False, 
    rescale_timesteps=False,
    rescale_learned_sigmas=False, 
    learn_sigma=True,
)


def create_argparser():
    defaults = dict(
        data_dir='data/afhq/train', 
        save_dir='checkpoints/afhq_sr_64',
        batch_size=128, 
        microbatch=32, 
        lr=1e-4, 
        ema_rate='0.9999',
        log_interval=100, 
        save_interval=5000, 
        resume_checkpoint='',
        use_fp16=False, 
        fp16_scale_growth=1e-3,
        schedule_sampler='uniform',
        weight_decay=0.0, 
        lr_anneal_steps=0, 
        num_workers=4,
        cond_aug_prob=1.0, 
        cond_aug_max_timestep=200,
    )
    defaults.update(SR_UNET_CONFIG)
    defaults.update(SR_DIFFUSION_CONFIG)
    parser = argparse.ArgumentParser(description='Train 64x64 AFHQ SR model')
    add_dict_to_argparser(parser, defaults)
    for key, val in [('data_dir', defaults['data_dir']),
                     ('save_dir', defaults['save_dir']),
                     ('num_workers', 4),
                     ('cond_aug_prob', 1.0),
                     ('cond_aug_max_timestep', 200)]:
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

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'Loading paired AFHQ data from {args.data_dir}…')
    data = load_afhq_sr_data(
        data_dir=args.data_dir, batch_size=args.batch_size,
        large_size=args.large_size, small_size=args.small_size,
        num_workers=args.num_workers, augment=True,
        cond_aug_prob=args.cond_aug_prob,
        cond_aug_max_timestep=args.cond_aug_max_timestep,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f'Model parameters: {n_params:,}')
    logger.log(f'Target: {args.large_size}x{args.large_size}, '
               f'cond from: {args.small_size}x{args.small_size}')
    logger.log(f'Cond aug: prob={args.cond_aug_prob}, S={args.cond_aug_max_timestep}')
    logger.log('Training SR model…')

    TrainLoop(
        model=model, diffusion=diffusion, data=data,
        batch_size=args.batch_size, microbatch=args.microbatch,
        lr=args.lr, ema_rate=args.ema_rate,
        log_interval=args.log_interval, save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16, fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay, lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()




if __name__ == '__main__':
    main()
