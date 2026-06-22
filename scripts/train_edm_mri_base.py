"""
train_edm_mri_base.py
---------------------
Train a 96×96 base model for 2-channel MRI using EDM.

Architecture : SongUNet (DDPM++ configuration)
Input/Output : 2-channel (Real + Imaginary)
Precond      : EDM (Karras et al., 2022)

Usage:
    python scripts/train_edm_mri_base.py \\
        --data_dir /data/Sahil_dataset/MRI_processed/train/AXT2_normalized \\
        --outdir checkpoints/edm_mri_base_96

    torchrun --standalone --nproc_per_node=2 scripts/train_edm_mri_base.py \\
        --data_dir /data/Sahil_dataset/MRI_processed/train/AXT2_normalized \\
        --outdir checkpoints/edm_mri_base_96
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import psutil
import torch

# ── Make EDM importable ──────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EDM_PATH = os.path.join(_REPO_ROOT, 'edm_repo')
if _EDM_PATH not in sys.path:
    sys.path.insert(0, _EDM_PATH)

from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training.networks import SongUNet, EDMPrecond

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edm_mri_dataloader import MRIBaseDatasetEDM


# ──────────────────────────────────────────────────────────────────────────────
#  Loss (standard EDM loss adapted for 2ch)
# ──────────────────────────────────────────────────────────────────────────────

class EDMBaseLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(images) * sigma
        D_yn = net(images + n, sigma)
        loss = weight * ((D_yn - images) ** 2)
        return loss.sum() / loss.shape[0]


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────────

def base_training_loop(
    outdir,
    data_dir,
    img_resolution      = 96,
    img_channels        = 2,
    num_workers          = 4,
    model_channels       = 128,
    channel_mult         = [1, 2, 2, 2],
    num_blocks           = 4,
    attn_resolutions     = [16],
    dropout              = 0.13,
    batch_size           = 256,
    batch_gpu            = None,
    total_kimg           = 200000,
    lr                   = 1e-3,
    ema_halflife_kimg    = 500,
    ema_rampup_ratio     = 0.05,
    lr_rampup_kimg       = 10000,
    P_mean               = -1.2,
    P_std                = 1.2,
    sigma_data           = 0.5,
    kimg_per_tick        = 50,
    snapshot_ticks       = 50,
    state_dump_ticks     = 100,
    seed                 = 0,
    resume_pkl           = None,
    resume_state_dump    = None,
    resume_kimg          = 0,
    use_fp16             = False,
    augment_prob         = 0.0,
    xflip                = True,
    cudnn_benchmark      = True,
    device               = torch.device('cuda'),
):
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # Batch size
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Dataset
    dist.print0('Loading MRI base dataset...')
    dataset = MRIBaseDatasetEDM(data_dir, augment=xflip)
    data_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=misc.InfiniteSampler(
                dataset=dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                seed=seed,
            ),
            batch_size=batch_gpu,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
        )
    )

    # Network — standard EDMPrecond wrapping SongUNet
    dist.print0('Constructing EDM base model...')
    net = EDMPrecond(
        img_resolution  = img_resolution,
        img_channels    = img_channels,
        use_fp16        = use_fp16,
        sigma_data      = sigma_data,
        model_channels  = model_channels,
        channel_mult    = channel_mult,
        num_blocks      = num_blocks,
        attn_resolutions = attn_resolutions,
        dropout         = dropout,
        # DDPM++ config
        model_type      = 'SongUNet',
        embedding_type  = 'positional',
        channel_mult_noise = 1,
        encoder_type    = 'standard',
        decoder_type    = 'standard',
        resample_filter = [1, 1],
    )
    net.train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        n_params = sum(p.numel() for p in net.parameters())
        dist.print0(f'  Parameters: {n_params:,}')
        dist.print0(f'  Input/Output: {img_channels}ch @ {img_resolution}×{img_resolution}')

    # Optimizer
    dist.print0('Setting up optimizer...')
    loss_fn = EDMBaseLoss(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        import pickle
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with open(resume_pkl, 'rb') as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data

    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data

    # Train
    dist.print0(f'Training for {total_kimg} kimg...')
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images = next(data_iterator)
                images = images.to(device).to(torch.float32)
                loss = loss_fn(net=ddp, images=images)
                training_stats.report('Loss/loss', loss)
                loss.mul(1.0 / batch_gpu_total).backward()

        # Update weights
        for g in optimizer.param_groups:
            g['lr'] = lr * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # EMA
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {_format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / max(cur_nimg - tick_start_nimg, 1) * 1e3):<7.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        if (not done) and dist.should_stop():
            done = True
            dist.print0('Aborting...')

        # Save snapshot
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            import pickle
            data = dict(ema=ema)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
            if dist.get_rank() == 0:
                pkl_path = os.path.join(outdir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(data, f)
                dist.print0(f'  Saved {pkl_path}')
            del data

        # Save training state
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) \
                and cur_tick != 0 and dist.get_rank() == 0:
            state_path = os.path.join(outdir, f'training-state-{cur_nimg//1000:06d}.pt')
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), state_path)

        # Logs
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(outdir, 'stats.jsonl'), 'at')
            stats_jsonl.write(
                json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n'
            )
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break

    dist.print0('Exiting...')


def _format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60:       return f'{s}s'
    elif s < 3600:   return f'{s // 60}m {s % 60:02d}s'
    elif s < 86400:  return f'{s // 3600}h {(s % 3600) // 60:02d}m'
    else:            return f'{s // 86400}d {(s % 86400) // 3600:02d}h'


def main():
    parser = argparse.ArgumentParser(description='Train EDM base model for 2ch MRI (96×96)')
    parser.add_argument('--data_dir',      default='/data/Sahil_dataset/MRI_processed/train/AXT2_normalized')
    parser.add_argument('--outdir',        default='checkpoints/edm_mri_base_96')
    parser.add_argument('--batch',         type=int, default=256, dest='batch_size')
    parser.add_argument('--batch_gpu',     type=int, default=None)
    parser.add_argument('--total_kimg',    type=int, default=200000)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--dropout',       type=float, default=0.13)
    parser.add_argument('--fp16',          action='store_true', dest='use_fp16')
    parser.add_argument('--tick',          type=int, default=50, dest='kimg_per_tick')
    parser.add_argument('--snap',          type=int, default=50, dest='snapshot_ticks')
    parser.add_argument('--seed',          type=int, default=0)
    parser.add_argument('--resume_pkl',    type=str, default=None)
    parser.add_argument('--resume_state',  type=str, default=None, dest='resume_state_dump')
    parser.add_argument('--resume_kimg',   type=int, default=0)
    parser.add_argument('--num_workers',   type=int, default=4)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if dist.get_rank() == 0:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, 'training_options.json'), 'wt') as f:
            json.dump(vars(args), f, indent=2)

    dist.print0()
    dist.print0('=' * 60)
    dist.print0('  EDM MRI Base Training (2ch, 96×96)')
    dist.print0('=' * 60)
    dist.print0(f'  Data:          {args.data_dir}')
    dist.print0(f'  Output:        {args.outdir}')
    dist.print0(f'  Batch size:    {args.batch_size}  (batch_gpu={args.batch_gpu})')
    dist.print0(f'  GPUs:          {dist.get_world_size()}')
    dist.print0()

    base_training_loop(
        outdir           = args.outdir,
        data_dir         = args.data_dir,
        batch_size       = args.batch_size,
        batch_gpu        = args.batch_gpu,
        total_kimg       = args.total_kimg,
        lr               = args.lr,
        model_channels   = args.model_channels,
        dropout          = args.dropout,
        use_fp16         = args.use_fp16,
        kimg_per_tick    = args.kimg_per_tick,
        snapshot_ticks   = args.snapshot_ticks,
        seed             = args.seed,
        resume_pkl       = args.resume_pkl,
        resume_state_dump = args.resume_state_dump,
        resume_kimg      = args.resume_kimg,
        num_workers      = args.num_workers,
    )


if __name__ == '__main__':
    main()
