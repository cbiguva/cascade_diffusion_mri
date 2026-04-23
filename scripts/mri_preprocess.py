"""
mri_preprocess.py
-----------------
Converts raw fastMRI AXT2 slices into two datasets:
  1. data/train_96/   – 96x96 PNGs  (4x4 average-pooled from 384x384)
  2. data/train_384/  – 384x384 PNGs (original, for SR target)

Both are written as single-channel (grayscale) 16-bit PNGs,
normalised per-volume to [0, 65535].

Usage:
    python scripts/mri_preprocess.py \
        --fastmri_root /path/to/fastMRI/knee/multicoil_train \
        --out_dir    data/ \
        --contrast   AXFLAIR    # or AXT2, AXT1, etc.
        --split_seed 42
"""

import argparse
import os
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

TARGET_SIZE = 384   # original fastMRI knee slice size after centre-crop
SMALL_SIZE  = 96    # 384 / 4


def centre_crop(arr, size=TARGET_SIZE):
    """Centre-crop a 2-D array to (size x size)."""
    h, w = arr.shape[-2:]
    top  = (h - size) // 2
    left = (w - size) // 2
    return arr[..., top:top+size, left:left+size]


def pool4x4(img384):
    """Average 4x4 blocks → 96x96."""
    # img384: (384, 384) float32
    h = img384.reshape(SMALL_SIZE, 4, SMALL_SIZE, 4)
    return h.mean(axis=(1, 3))          # (96, 96)


def to_png(arr, path):
    """Save a float32 array in [0,1] as 16-bit grayscale PNG."""
    arr_u16 = (arr.clip(0, 1) * 65535).astype(np.uint16)
    Image.fromarray(arr_u16, mode='I;16').save(path)


def process_file(h5_path, out96, out384, contrast):
    with h5py.File(h5_path, 'r') as f:
        # fastMRI stores slices in 'reconstruction_rss' or 'reconstruction_esc'
        key = 'reconstruction_rss' if 'reconstruction_rss' in f else \
              'reconstruction_esc'
        meta_contrast = f.attrs.get('acquisition', b'').decode()

        if contrast and meta_contrast != contrast:
            return 0   # skip wrong contrast

        volume = f[key][:]           # (slices, H, W)  float32

    # discard top/bottom 5 slices (often pure noise)
    volume = volume[5:-5]
    if len(volume) == 0:
        return 0

    # normalise entire volume to [0, 1]
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-6:
        return 0
    volume = (volume - vmin) / (vmax - vmin)

    stem = Path(h5_path).stem
    count = 0
    for i, sl in enumerate(volume):
        sl384 = centre_crop(sl, TARGET_SIZE)  # (384, 384)
        sl96  = pool4x4(sl384)                # (96,  96)

        fname = f"{stem}_sl{i:03d}.png"
        to_png(sl384, out384 / fname)
        to_png(sl96,  out96  / fname)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastmri_root', required=True)
    parser.add_argument('--out_dir',      default='data')
    parser.add_argument('--contrast',     default='AXT2',
                        help='Filter to one acquisition type. Empty = all.')
    parser.add_argument('--val_frac',     type=float, default=0.1)
    parser.add_argument('--split_seed',   type=int,   default=42)
    args = parser.parse_args()

    root   = Path(args.fastmri_root)
    outdir = Path(args.out_dir)

    for split in ('train', 'val'):
        (outdir / f'{split}_96').mkdir(parents=True, exist_ok=True)
        (outdir / f'{split}_384').mkdir(parents=True, exist_ok=True)

    h5_files = sorted(root.rglob('*.h5'))
    random.seed(args.split_seed)
    random.shuffle(h5_files)
    n_val    = max(1, int(len(h5_files) * args.val_frac))
    val_set  = set(str(f) for f in h5_files[:n_val])

    total = 0
    for h5 in tqdm(h5_files, desc='Processing volumes'):
        split  = 'val' if str(h5) in val_set else 'train'
        out96  = outdir / f'{split}_96'
        out384 = outdir / f'{split}_384'
        total += process_file(h5, out96, out384, args.contrast)

    print(f'\nDone. Saved {total} slice pairs.')
    print(f'  data/{split}_96/   – 96×96 images for base model')
    print(f'  data/{split}_384/  – 384×384 images for SR model')


if __name__ == '__main__':
    main()
