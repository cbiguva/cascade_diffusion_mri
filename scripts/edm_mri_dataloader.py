"""
edm_mri_dataloader.py
---------------------
FastMRI dataset loaders for EDM-based cascaded diffusion training.

Data format
-----------
Each .pt file contains:
    {
        'slices':       (S, 2, 384, 384)  float32  — channel 0=Real, 1=Imaginary
        'global_scale': scalar
    }

Two datasets are provided:

    MRIBaseDatasetEDM   — avg-pool 384→96, returns (2, 96, 96) in [-1, 1]
    MRISRDatasetEDM     — returns (hr, low_res) pairs for SR training
                          hr: (2, 384, 384), low_res: (2, 96, 96)
                          CDM conditioning augmentation always applied (S=300)
"""

import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pt_files(pt_dir: str):
    """Return sorted list of .pt files in pt_dir."""
    paths = sorted(Path(pt_dir).glob("*.pt"))
    assert len(paths) > 0, f"No .pt files found in {pt_dir}"
    return paths


def _build_index(pt_dir: str):
    """
    Build a flat index of (file_path, slice_idx) pairs.
    Cached in <pt_dir>/index_cache_edm.pkl.
    """
    import pickle
    pt_dir = Path(pt_dir)
    cache_path = pt_dir / "index_cache_edm.pkl"
    paths = _pt_files(str(pt_dir))

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_files") == len(paths):
                index = cached["index"]
                print(f"[edm_mri] {pt_dir}  →  cached index "
                      f"({len(paths)} files, {len(index)} slices)")
                return index
        except Exception:
            pass

    print(f"[edm_mri] Building slice index for {pt_dir} ({len(paths)} files)…")
    index = []
    for i, fpath in enumerate(paths):
        if i % 200 == 0:
            print(f"  {i}/{len(paths)}…", flush=True)
        try:
            obj = torch.load(str(fpath), map_location="cpu")
        except Exception as e:
            print(f"  WARNING: skipping {fpath.name} ({e})")
            continue
        if "slices" not in obj:
            continue
        S = obj["slices"].shape[0]
        for s in range(S):
            index.append((str(fpath), s))
        del obj

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"n_files": len(paths), "index": index}, f)
    except Exception:
        pass

    print(f"[edm_mri] {len(paths)} files, {len(index)} slices total")
    return index


class _LRUFileCache:
    """Simple LRU cache for loaded .pt tensors."""
    def __init__(self, max_size: int = 8):
        self._cache: OrderedDict = OrderedDict()
        self.max_size = max_size

    def get(self, fpath):
        fpath = str(fpath)
        if fpath in self._cache:
            self._cache.move_to_end(fpath)
            return self._cache[fpath]
        try:
            obj = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(fpath, map_location="cpu")
        self._cache[fpath] = obj
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return obj


def _pool4x4(img: torch.Tensor) -> torch.Tensor:
    """Average-pool (C, 384, 384) → (C, 96, 96)."""
    return F.avg_pool2d(img.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)


def _linear_beta_schedule(num_timesteps: int = 1000):
    """Compute √ᾱ_t and √(1−ᾱ_t) for the linear β schedule."""
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end   = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_ac  = torch.from_numpy(np.sqrt(alphas_cumprod)).float()
    sqrt_1m  = torch.from_numpy(np.sqrt(1.0 - alphas_cumprod)).float()
    return sqrt_ac, sqrt_1m


# ─────────────────────────────────────────────────────────────────────────────
#  Base Model Dataset (96×96)
# ─────────────────────────────────────────────────────────────────────────────

class MRIBaseDatasetEDM(Dataset):
    """
    Loads 2-channel MRI slices, avg-pools 384→96.

    Returns: (2, 96, 96) float32 in [-1, 1]
    """

    def __init__(self, pt_dir: str, augment: bool = True, cache_size: int = 8):
        self.index   = _build_index(pt_dir)
        self.augment = augment
        self._cache  = _LRUFileCache(max_size=cache_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, sidx = self.index[idx]
        obj  = self._cache.get(fpath)
        slc  = obj["slices"][sidx]        # (2, 384, 384)

        x96 = _pool4x4(slc)              # (2, 96, 96)

        if self.augment and random.random() < 0.5:
            x96 = x96.flip(-1)           # random horizontal flip

        return x96  # (2, 96, 96) in [-1, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  SR Model Dataset (96→384)
# ─────────────────────────────────────────────────────────────────────────────

class MRISRDatasetEDM(Dataset):
    """
    2-channel MRI super-resolution dataset for EDM.

    Returns (hr, low_res):
        hr      : (2, 384, 384)  — HR target in [-1, 1]
        low_res : (2, 96, 96)    — avg-pooled LR condition (augmented)

    CDM conditioning augmentation (§4.2) is ALWAYS applied:
        s ~ Uniform{0, …, S}
        if s > 0:  low_res = √ᾱ_s · low_res + √(1−ᾱ_s) · ε
    """

    def __init__(
        self,
        pt_dir: str,
        augment: bool = True,
        cache_size: int = 8,
        cond_aug_max_timestep: int = 300,
        num_diffusion_timesteps: int = 1000,
    ):
        self.index   = _build_index(pt_dir)
        self.augment = augment
        self._cache  = _LRUFileCache(max_size=cache_size)
        self.cond_aug_max_timestep = cond_aug_max_timestep

        sqrt_ac, sqrt_1m = _linear_beta_schedule(num_diffusion_timesteps)
        self.sqrt_alphas_cumprod           = sqrt_ac
        self.sqrt_one_minus_alphas_cumprod  = sqrt_1m

        print(f"[edm_mri] SR dataset: {len(self.index)} slices, "
              f"384→96, cond_aug S={cond_aug_max_timestep} (always)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, sidx = self.index[idx]
        obj  = self._cache.get(fpath)
        slc  = obj["slices"][sidx]        # (2, 384, 384)

        hr = slc.clone()                  # (2, 384, 384)
        low_res = _pool4x4(slc)           # (2, 96, 96)

        # Random horizontal flip (same for both)
        if self.augment and random.random() < 0.5:
            hr = hr.flip(-1)
            low_res = low_res.flip(-1)

        # ── CDM conditioning augmentation (§4.2) ─────────────────────────
        s = random.randint(0, self.cond_aug_max_timestep)
        if s > 0:
            sqrt_alpha    = self.sqrt_alphas_cumprod[s - 1]
            sqrt_1m_alpha = self.sqrt_one_minus_alphas_cumprod[s - 1]
            noise = torch.randn_like(low_res)
            low_res = sqrt_alpha * low_res + sqrt_1m_alpha * noise

        return hr, low_res


# ─────────────────────────────────────────────────────────────────────────────
#  Public loaders
# ─────────────────────────────────────────────────────────────────────────────

def _infinite_loader(dataset, batch_size, num_workers=4):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    while True:
        yield from loader
