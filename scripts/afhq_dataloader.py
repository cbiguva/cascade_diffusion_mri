"""
afhq_dataloader.py
------------------
Custom dataset loaders for the AFHQ cascaded diffusion experiment.

Data format
-----------
Standard image files (jpg/png) organised in ImageFolder layout:
    data/afhq/train/cat/  dog/  wild/
    data/afhq/val/cat/    dog/  wild/

Images are 512×512 RGB. We resize on-the-fly to the required resolution.

Public API (compatible with guided-diffusion's TrainLoop)
---------------------------------------------------------
Both generators yield (x, cond) tuples where:
  • x    : (B, 3, H, W)  float32 in [-1, 1]
  • cond : dict  (empty for base model, {'low_res': tensor} for SR model)

    load_afhq_data(data_dir, batch_size, ...)
        → infinite generator of (B,3,32,32) base-model batches

    load_afhq_sr_data(data_dir, batch_size, ...)
        → infinite generator of
          ( (B,3,64,64) target,
            {'low_res': (B,3,64,64) NN-upsampled condition} )

Conditioning augmentation (CDM paper §4.2)
------------------------------------------
The SR dataloader applies *truncated conditioning augmentation* to the
low-res condition during training.  A discrete timestep s is sampled
uniformly from {0, …, S} (S = cond_aug_max_timestep) and the forward
diffusion kernel is applied:

    z_s = √ᾱ_s · z_0 + √(1 − ᾱ_s) · ε,   ε ~ N(0, I)

where ᾱ_s comes from the same linear β-schedule used by the SR diffusion
model.  When s = 0 no corruption is applied (identity).
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



def _list_image_files(data_dir: str):
    """Return sorted list of image file paths (jpg/jpeg/png) recursively."""
    exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    paths = []
    for p in sorted(Path(data_dir).rglob('*')):
        if p.suffix.lower() in exts and p.is_file():
            paths.append(str(p))
    assert len(paths) > 0, f"No image files found in {data_dir}"
    return paths


def _nn_upsample_2x(img: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbour upsample (C, 32, 32) → (C, 64, 64).
    Uses repeat_interleave for exact integer-factor upsampling.
    """
    h_rep = img.repeat_interleave(2, dim=1)    # (C, 64, 32)
    hw_rep = h_rep.repeat_interleave(2, dim=2)  # (C, 64, 64)
    return hw_rep


def _linear_beta_schedule(num_timesteps: int = 1000):
    """
    Compute √ᾱ_t and √(1−ᾱ_t) for the linear β schedule.

    Matches guided_diffusion.gaussian_diffusion.get_named_beta_schedule("linear", T).
    Returns two float32 tensors of shape (num_timesteps,).
    """
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end   = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod         = torch.from_numpy(np.sqrt(alphas_cumprod)).float()
    sqrt_one_minus_alphas_cumprod = torch.from_numpy(np.sqrt(1.0 - alphas_cumprod)).float()
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

#base model

class AFHQBaseDataset(Dataset):
    """
    Loads AFHQ images, resizes to 32×32.

    Returns:
        (x, {})  where  x : (3, 32, 32)  float32 in [-1, 1]
    """

    def __init__(self, data_dir: str, image_size: int = 32, augment: bool = True):
        self.image_paths = _list_image_files(data_dir)
        self.image_size = image_size
        self.augment = augment

        # Deterministic transforms (resize + center crop)
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),            # → [0, 1]
        ])

        print(f"[afhq_dataloader] Base dataset: {len(self.image_paths)} images, "
              f"target size {image_size}×{image_size}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')

        x = self.transform(img)               # (3, 32, 32) in [0, 1]
        x = x * 2.0 - 1.0                     # → [-1, 1]

        if self.augment and random.random() < 0.5:
            x = x.flip(-1)                    # random horizontal flip

        return x, {}


# ─────────────────────────────────────────────────────────────────────────────
# SR dataset with truncated conditioning augmentation (CDM §4.2)
# ─────────────────────────────────────────────────────────────────────────────

class AFHQSRDataset(Dataset):
    """
    Loads AFHQ images at 64×64 (HR target) and 32×32 (LR condition).

    Truncated conditioning augmentation (CDM paper §4.2)
    ----------------------------------------------------
    With probability ``cond_aug_prob``, a discrete timestep
    ``s ~ Uniform{1, …, cond_aug_max_timestep}`` is sampled and the
    forward diffusion kernel is applied to the low-res condition:

        z_s = √ᾱ_s · z_0  +  √(1 − ᾱ_s) · ε,   ε ~ N(0, I)

    This attenuates the clean signal and adds appropriately scaled noise,
    making the SR model robust to imperfect base-model outputs at inference.

    Returns:
        (x, {'low_res': low_res})
        x       : (3, 64, 64) float32 in [-1, 1]  – HR target
        low_res : (3, 64, 64) float32              – augmented condition
    """

    def __init__(
        self,
        data_dir: str,
        large_size: int = 64,
        small_size: int = 32,
        augment: bool = True,
        cond_aug_prob: float = 1.0,            # CDM paper: augment every sample
        cond_aug_max_timestep: int = 200,      # S — max diffusion timestep for augmentation
        num_diffusion_timesteps: int = 1000,   # T — total diffusion timesteps (must match SR model)
    ):
        self.image_paths = _list_image_files(data_dir)
        self.large_size = large_size
        self.small_size = small_size
        self.augment = augment
        self.cond_aug_prob = cond_aug_prob
        self.cond_aug_max_timestep = cond_aug_max_timestep
        self.scale_factor = large_size // small_size  # 2 for 32→64

        # Precompute schedule coefficients for conditioning augmentation
        sqrt_ac, sqrt_1m_ac = _linear_beta_schedule(num_diffusion_timesteps)
        self.sqrt_alphas_cumprod = sqrt_ac             # (T,)
        self.sqrt_one_minus_alphas_cumprod = sqrt_1m_ac  # (T,)

        # Transform to get HR image
        self.transform_hr = transforms.Compose([
            transforms.Resize(large_size, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(large_size),
            transforms.ToTensor(),
        ])

        # Transform to get LR image (for downsampling)
        self.transform_lr = transforms.Compose([
            transforms.Resize(small_size, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(small_size),
            transforms.ToTensor(),
        ])

        print(f"[afhq_dataloader] SR dataset: {len(self.image_paths)} images, "
              f"target {large_size}×{large_size}, condition {small_size}→{large_size}, "
              f"cond_aug: prob={cond_aug_prob}, S={cond_aug_max_timestep}/{num_diffusion_timesteps}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')

        # HR target
        x = self.transform_hr(img)            # (3, 64, 64) in [0, 1]
        x = x * 2.0 - 1.0                     # → [-1, 1]

        # LR condition: downsample then NN-upsample
        x_lr = self.transform_lr(img)         # (3, 32, 32) in [0, 1]
        x_lr = x_lr * 2.0 - 1.0              # → [-1, 1]

        if self.augment and random.random() < 0.5:
            x = x.flip(-1)                    # same flip for both
            x_lr = x_lr.flip(-1)

        # NN-upsample to match target resolution
        low_res = _nn_upsample_2x(x_lr)      # (3, 64, 64) pixelated blocks

        # ── Truncated conditioning augmentation (CDM paper §4.2) ──────────
        # Sample a discrete timestep s from {0, …, S}.  s=0 means identity
        # (no corruption).  For s>0 apply the forward diffusion kernel:
        #   z_s = √ᾱ_s · z_0  +  √(1 − ᾱ_s) · ε
        if self.augment and random.random() < self.cond_aug_prob:
            s = random.randint(0, self.cond_aug_max_timestep)  # 0 = identity
            if s > 0:
                sqrt_alpha = self.sqrt_alphas_cumprod[s - 1]       # 0-indexed
                sqrt_1m_alpha = self.sqrt_one_minus_alphas_cumprod[s - 1]
                noise = torch.randn_like(low_res)
                low_res = sqrt_alpha * low_res + sqrt_1m_alpha * noise

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


def load_afhq_data(
    data_dir: str,
    batch_size: int,
    image_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
):
    """
    Infinite generator for the BASE 32×32 model.

    Yields:  (x, {})
      x : (B, 3, 32, 32)  float32  in [-1, 1]
    """
    ds = AFHQBaseDataset(data_dir, image_size=image_size, augment=augment)
    return _infinite_loader(ds, batch_size, num_workers)


def load_afhq_sr_data(
    data_dir: str,
    batch_size: int,
    large_size: int = 64,
    small_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    cond_aug_prob: float = 1.0,
    cond_aug_max_timestep: int = 200,
    num_diffusion_timesteps: int = 1000,
):
    """
    Infinite generator for the SR 32→64 model.

    Yields:  (x, {'low_res': low_res})
      x       : (B, 3, 64, 64)  float32  in [-1, 1]  – HR target
      low_res : (B, 3, 64, 64)  float32               – augmented condition
                (resized to 32, NN-upsampled to 64, then truncated diffusion aug)

    Truncated conditioning augmentation (CDM paper §4.2):
      cond_aug_prob           – fraction of examples that get augmented (default 0.5)
      cond_aug_max_timestep   – S: max diffusion timestep for augmentation (default 200)
      num_diffusion_timesteps – T: total diffusion steps, must match SR model (default 1000)
    """
    ds = AFHQSRDataset(
        data_dir,
        large_size=large_size,
        small_size=small_size,
        augment=augment,
        cond_aug_prob=cond_aug_prob,
        cond_aug_max_timestep=cond_aug_max_timestep,
        num_diffusion_timesteps=num_diffusion_timesteps,
    )
    return _infinite_loader(ds, batch_size, num_workers)
