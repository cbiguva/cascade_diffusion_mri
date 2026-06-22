"""
edm_sr_dataloader.py
--------------------
AFHQ SR dataset for EDM-based cascaded diffusion training.

Data format
-----------
Standard image files (jpg/png) in flat or ImageFolder layout:
    data/afhq/*.png   or   data/afhq/train/cat/ dog/ wild/

Conditioning augmentation (CDM paper §4.2)
------------------------------------------
ALWAYS applied (100% of samples).  A discrete timestep s is sampled
uniformly from {0, 1, …, S} where S = cond_aug_max_timestep (default 300,
i.e. 30% of T=1000).  The forward diffusion kernel is applied to the
low-resolution condition:

    z_s = √ᾱ_s · z₀  +  √(1 − ᾱ_s) · ε,   ε ~ N(0, I)

When s = 0, no corruption is applied (identity).

Returns
-------
(hr, low_res)  — both float32 tensors in [-1, 1].
  hr      : (3, 64, 64)   HR target
  low_res : (3, 32, 32)   augmented LR condition (NOT upsampled — the
                           model's bilinear interpolation handles that)
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _list_image_files(data_dir: str):
    """Return sorted list of image file paths (jpg/jpeg/png) recursively."""
    exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    paths = []
    for p in sorted(Path(data_dir).rglob('*')):
        if p.suffix.lower() in exts and p.is_file():
            paths.append(str(p))
    assert len(paths) > 0, f"No image files found in {data_dir}"
    return paths


def _linear_beta_schedule(num_timesteps: int = 1000):
    """
    Compute √ᾱ_t and √(1−ᾱ_t) for the linear β schedule.

    Matches guided_diffusion / DDPM linear schedule.
    Returns two float32 tensors of shape (num_timesteps,).
    """
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end   = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod          = torch.from_numpy(np.sqrt(alphas_cumprod)).float()
    sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.sqrt(1.0 - alphas_cumprod)).float()
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AFHQSRDatasetEDM(Dataset):
    """
    AFHQ super-resolution dataset for EDM training.

    Returns (hr, low_res) pairs in [-1, 1]:
        hr      : (3, large_size, large_size)   — HR target
        low_res : (3, small_size, small_size)    — augmented LR condition

    Conditioning augmentation (CDM §4.2) is ALWAYS applied:
        s ~ Uniform{0, …, S}
        if s > 0:  low_res = √ᾱ_s · low_res + √(1−ᾱ_s) · ε
    """

    def __init__(
        self,
        data_dir: str,
        large_size: int = 64,
        small_size: int = 32,
        augment: bool = True,
        cond_aug_max_timestep: int = 300,      # S — 30% of T=1000
        num_diffusion_timesteps: int = 1000,   # T
    ):
        self.image_paths = _list_image_files(data_dir)
        self.large_size = large_size
        self.small_size = small_size
        self.augment = augment
        self.cond_aug_max_timestep = cond_aug_max_timestep

        # Precompute schedule coefficients for conditioning augmentation
        sqrt_ac, sqrt_1m_ac = _linear_beta_schedule(num_diffusion_timesteps)
        self.sqrt_alphas_cumprod           = sqrt_ac       # (T,)
        self.sqrt_one_minus_alphas_cumprod  = sqrt_1m_ac    # (T,)

        # HR transform
        self.transform_hr = transforms.Compose([
            transforms.Resize(large_size,
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(large_size),
            transforms.ToTensor(),  # → [0, 1]
        ])

        # LR transform (downsample)
        self.transform_lr = transforms.Compose([
            transforms.Resize(small_size,
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(small_size),
            transforms.ToTensor(),  # → [0, 1]
        ])

        print(f"[edm_sr_dataloader] SR dataset: {len(self.image_paths)} images, "
              f"target {large_size}×{large_size}, cond {small_size}×{small_size}, "
              f"cond_aug: S={cond_aug_max_timestep}/{num_diffusion_timesteps} (always applied)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')

        # HR target
        x = self.transform_hr(img)            # (3, 64, 64) in [0, 1]
        x = x * 2.0 - 1.0                     # → [-1, 1]

        # LR condition
        x_lr = self.transform_lr(img)         # (3, 32, 32) in [0, 1]
        x_lr = x_lr * 2.0 - 1.0              # → [-1, 1]

        # Random horizontal flip (same for both)
        if self.augment and random.random() < 0.5:
            x = x.flip(-1)
            x_lr = x_lr.flip(-1)

        # ── Truncated conditioning augmentation (CDM paper §4.2) ──────────
        # ALWAYS applied.  Sample s from {0, …, S}.
        # s = 0 means identity (no corruption).
        s = random.randint(0, self.cond_aug_max_timestep)
        if s > 0:
            sqrt_alpha    = self.sqrt_alphas_cumprod[s - 1]          # 0-indexed
            sqrt_1m_alpha = self.sqrt_one_minus_alphas_cumprod[s - 1]
            noise = torch.randn_like(x_lr)
            x_lr = sqrt_alpha * x_lr + sqrt_1m_alpha * noise

        # Return (hr, low_res) — NOT upsampled.
        # The model's bilinear interpolation handles upsampling internally.
        return x, x_lr


# ─────────────────────────────────────────────────────────────────────────────
#  Public loaders
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


def load_edm_sr_data(
    data_dir: str,
    batch_size: int,
    large_size: int = 64,
    small_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    cond_aug_max_timestep: int = 300,
    num_diffusion_timesteps: int = 1000,
):
    """
    Infinite generator for EDM SR training.

    Yields:  (hr, low_res)
      hr      : (B, 3, 64, 64)  float32 in [-1, 1]
      low_res : (B, 3, 32, 32)  float32 (augmented, NOT upsampled)
    """
    ds = AFHQSRDatasetEDM(
        data_dir,
        large_size=large_size,
        small_size=small_size,
        augment=augment,
        cond_aug_max_timestep=cond_aug_max_timestep,
        num_diffusion_timesteps=num_diffusion_timesteps,
    )
    return _infinite_loader(ds, batch_size, num_workers)
