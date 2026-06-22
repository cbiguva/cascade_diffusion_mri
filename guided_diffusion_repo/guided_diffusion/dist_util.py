"""
Helpers for distributed training.

Supports two modes:
  1. torchrun / torch.distributed.launch (PREFERRED, no MPI needed):
       CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_script.py ...
  2. Single-GPU fallback:
       python train_script.py ...

The original guided-diffusion used MPI via mpi4py.  This version removes that
dependency and uses PyTorch-native distributed primitives only.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.

    Works with torchrun (which sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
    MASTER_PORT env vars automatically).  If those are not set, falls back to
    single-process mode.
    """
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # ── Launched via torchrun ─────────────────────────────────────────
        backend = "nccl" if th.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        th.cuda.set_device(local_rank)
    else:
        # ── Single-process fallback ───────────────────────────────────────
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if th.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.  Rank 0 reads from disk and broadcasts to others.
    """
    if dist.get_world_size() == 1:
        # Single process — just load directly
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        return th.load(io.BytesIO(data), **kwargs)

    # Multi-process: rank 0 reads, broadcasts length then data
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        length = th.tensor([len(data)], dtype=th.long, device="cpu")
    else:
        data = None
        length = th.tensor([0], dtype=th.long, device="cpu")

    dist.broadcast(length, src=0)

    if dist.get_rank() != 0:
        data = bytes(length.item())
        data_tensor = th.zeros(length.item(), dtype=th.uint8)
    else:
        data_tensor = th.frombuffer(bytearray(data), dtype=th.uint8).clone()

    dist.broadcast(data_tensor, src=0)

    if dist.get_rank() != 0:
        data = bytes(data_tensor.numpy())

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
