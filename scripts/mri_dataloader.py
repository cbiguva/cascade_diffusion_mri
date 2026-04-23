"""
mri_dataloader.py
-----------------
Custom dataset loaders for the MRI cascaded diffusion project.

Data format
-----------
Each .pt file in the dataset directory stores:
    {
        'slices':       torch.Tensor  shape (S, 2, 384, 384)  float32
        'global_scale': torch.Tensor  scalar
    }

The two channels are real + imaginary coil images.
We combine them into a magnitude image:
    magnitude = sqrt(real² + imag²)

The data is already normalised so that magnitude ∈ [0, 1] per volume.
We rescale to [-1, 1] before feeding to the model.

Public API (compatible with guided-diffusion's TrainLoop)
---------------------------------------------------------
Both generators yield (x, cond) tuples where:
  • x    : (B, 1, H, W)  float32  in [-1, 1]
  • cond : dict  (empty for base model, {'low_res': tensor} for SR model)

    load_mri_data(pt_dir, batch_size, ...)
        → infinite generator of (B,1,96,96) base-model batches

    load_mri_sr_data(pt_dir, batch_size, ...)
        → infinite generator of
          ( (B,1,384,384) target,
            {'low_res': (B,1,384,384) NN-upsampled condition} )
"""

import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pt_files(pt_dir: str):
    """Return sorted list of .pt files in pt_dir."""
    paths = sorted(Path(pt_dir).glob("*.pt"))
    assert len(paths) > 0, f"No .pt files found in {pt_dir}"
    return paths


def _build_index(pt_dir: str):
    """
    Walk all .pt files and build a flat index of (file_path, slice_idx) pairs.

    Results are cached in <pt_dir>/index_cache.pkl so the expensive scan
    only runs once.  The cache is invalidated if the number of .pt files changes.
    """
    import pickle
    pt_dir = Path(pt_dir)
    cache_path = pt_dir / "index_cache.pkl"
    paths = _pt_files(str(pt_dir))

    # Check cache validity
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_files") == len(paths):
                index = cached["index"]
                print(f"[dataloader] {pt_dir}  →  loaded cached index "
                      f"({len(paths)} files, {len(index)} slices)")
                return index
        except Exception:
            pass  # corrupt cache — rebuild

    print(f"[dataloader] Building slice index for {pt_dir} "
          f"({len(paths)} files)…  [will cache for future runs]")
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
        del obj  # free memory immediately

    # Save cache
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"n_files": len(paths), "index": index}, f)
        print(f"[dataloader] Index cached → {cache_path}")
    except Exception as e:
        print(f"[dataloader] WARNING: could not save index cache: {e}")

    print(f"[dataloader] {pt_dir}  →  {len(paths)} files, {len(index)} slices total")
    return index


class _LRUFileCache:
    """Simple LRU cache for loaded .pt tensors."""
    def __init__(self, max_size: int = 4):
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


def _to_magnitude(slc: torch.Tensor) -> torch.Tensor:
    """
    slc : (2, H, W) float32  – channel 0 = real, channel 1 = imag
    returns: (H, W) magnitude in [-1, 1]

    The raw data is already normalised so magnitude ∈ [0, 1].
    We rescale linearly to [-1, 1].
    """
    mag = torch.sqrt(slc[0] ** 2 + slc[1] ** 2)   # (H, W), in [0, 1]
    mag = mag * 2.0 - 1.0                           # → [-1, 1]
    return mag


def _pool4x4(img: torch.Tensor) -> torch.Tensor:
    """
    Average-pool a (1, 384, 384) tensor → (1, 96, 96).
    """
    return F.avg_pool2d(img.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)


def _nn_upsample(img: torch.Tensor, size: int = 384) -> torch.Tensor:
    """
    Nearest-neighbour upsample (1, 96, 96) → (1, 384, 384).
    Uses torch.repeat to exactly replicate the numpy 4× block approach.
    """
    # img: (1, 96, 96)
    h_rep = img.repeat_interleave(4, dim=1)    # (1, 384, 96)
    hw_rep = h_rep.repeat_interleave(4, dim=2) # (1, 384, 384)
    return hw_rep


# ─────────────────────────────────────────────────────────────────────────────
# 1. Base model dataset  (full 384×384 → average-pool → 96×96)
# ─────────────────────────────────────────────────────────────────────────────

class MRIBaseDataset(Dataset):
    """
    Loads .pt MRI slices, computes magnitude, average-pools to 96×96.

    Returns:
        (x, {})  where  x : (1, 96, 96)  float32 in [-1, 1]
    """

    def __init__(self, pt_dir: str, augment: bool = True, cache_size: int = 4):
        self.index   = _build_index(pt_dir)
        self.augment = augment
        self._cache  = _LRUFileCache(max_size=cache_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, sidx = self.index[idx]
        obj  = self._cache.get(fpath)
        slc  = obj["slices"][sidx]        # (2, 384, 384)

        mag  = _to_magnitude(slc)         # (384, 384) in [-1, 1]
        x    = mag.unsqueeze(0)           # (1, 384, 384)

        if self.augment and random.random() < 0.5:
            x = x.flip(-1)               # random horizontal flip

        x96 = _pool4x4(x)                # (1, 96, 96)
        return x96, {}


# ─────────────────────────────────────────────────────────────────────────────
# 2. SR model dataset  (384×384 target + NN-upsampled 96×96 condition)
# ─────────────────────────────────────────────────────────────────────────────

class MRISRDataset(Dataset):
    """
    Loads .pt MRI slices, computes magnitude at 384×384.
    Also produces the blurry condition: average-pool → 96×96, then NN-upsample → 384×384.

    Returns:
        (x, {'low_res': low_res})
        where:
            x       : (1, 384, 384)  float32 in [-1, 1]  – HR target
            low_res : (1, 384, 384)  float32 in [-1, 1]  – blurry NN condition
    """

    def __init__(self, pt_dir: str, augment: bool = True, cache_size: int = 4):
        self.index   = _build_index(pt_dir)
        self.augment = augment
        self._cache  = _LRUFileCache(max_size=cache_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, sidx = self.index[idx]
        obj  = self._cache.get(fpath)
        slc  = obj["slices"][sidx]        # (2, 384, 384)

        mag  = _to_magnitude(slc)         # (384, 384) in [-1, 1]
        x    = mag.unsqueeze(0)           # (1, 384, 384)

        if self.augment and random.random() < 0.5:
            x = x.flip(-1)

        # Build blurry condition: pool → upsample
        x96     = _pool4x4(x)            # (1,  96,  96)
        low_res = _nn_upsample(x96)      # (1, 384, 384)  pixelated blocks

        return x, {'low_res': low_res}


# ─────────────────────────────────────────────────────────────────────────────
# Public generators (infinite, for use with TrainLoop)
# ─────────────────────────────────────────────────────────────────────────────

def _infinite_loader(dataset, batch_size, num_workers=4):
    """Wrap a Dataset in an infinite DataLoader."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    while True:
        yield from loader


def load_mri_data(pt_dir: str, batch_size: int, num_workers: int = 4, augment: bool = True):
    """
    Infinite generator for the BASE 96×96 model.

    Yields:  (x, {})
      x : (B, 1, 96, 96)  float32  in [-1, 1]
    """
    ds = MRIBaseDataset(pt_dir, augment=augment)
    return _infinite_loader(ds, batch_size, num_workers)


def load_mri_sr_data(pt_dir: str, batch_size: int, num_workers: int = 4, augment: bool = True):
    """
    Infinite generator for the SR 96→384 model.

    Yields:  (x, {'low_res': low_res})
      x       : (B, 1, 384, 384)  float32  in [-1, 1]  – HR target
      low_res : (B, 1, 384, 384)  float32  in [-1, 1]  – blurry NN condition
    """
    ds = MRISRDataset(pt_dir, augment=augment)
    return _infinite_loader(ds, batch_size, num_workers)
