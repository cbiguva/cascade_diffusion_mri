# MRI Cascaded Diffusion — Instructions

## Architecture

```
fastMRI AXT2 (384x384)
  |
  |-- pool 4x4 --> data/train_96/    (96x96 PNGs)  ← base model trains here
  |-- crop      --> data/train_384/  (384x384 PNGs) ← SR model trains here

MODEL 1: Base diffusion  96x96   (cosine schedule, UNET_SMALL, unconditional)
MODEL 2: SR diffusion    96→384  (linear schedule, UNET_SMALL, conditioned on
                                  NN-upsampled low-res image)

INFERENCE (chained):
  Gaussian noise → [Model 1] → 96x96 sample
                                    |
                              NN upsample ×4
                                    |
  Gaussian noise → [Model 2] → 384x384 final sample
```

---

## Step 0 — Preprocess fastMRI Data

```bash
python scripts/mri_preprocess.py \
    --fastmri_root /path/to/fastMRI/knee/multicoil_train \
    --out_dir      data/ \
    --contrast     AXT2 \
    --val_frac     0.1
```

Produces:
- data/train_96/   — 96x96 16-bit grayscale PNGs
- data/train_384/  — 384x384 16-bit grayscale PNGs (same filenames = paired)
- data/val_96/ and data/val_384/ for validation

---

## Step 1 — Train Base Model (96x96)

```bash
python scripts/train_base_96.py \
    --data_dir      data/train_96 \
    --save_dir      checkpoints/base_96 \
    --batch_size    8 \
    --lr            1e-4 \
    --save_interval 10000
```

Key architecture settings (UNET_SMALL):
  - num_channels = 64
  - num_res_blocks = 2
  - in_channels = 1          ← grayscale MRI
  - noise_schedule = cosine  ← base model uses cosine (from paper)
  - class_cond = False       ← single contrast, no labels
  - learn_sigma = True

Train for 100–200k steps. Checkpoints saved as:
  - model_XXXXXX.pt          ← raw weights
  - ema_0.9999_XXXXXX.pt     ← EMA weights (USE THIS for sampling)

GPU memory: batch_size=4 ≈ 8GB, batch_size=8 ≈ 14GB

---

## Step 2 — Train SR Model (96→384)

```bash
python scripts/train_sr_384.py \
    --base_dir      data/train_384 \
    --small_dir     data/train_96 \
    --save_dir      checkpoints/sr_384 \
    --batch_size    2 \
    --lr            1e-4 \
    --save_interval 10000
```

Key differences from base model:
  - noise_schedule = linear  ← SR model uses linear (from paper)
  - batch_size = 2           ← 384x384 is memory-heavy
  - Conditioning: 96x96 image is nearest-neighbour upsampled to 384x384
                  and concatenated with the noisy target

IMPORTANT: Models 1 and 2 are INDEPENDENT. Run them in parallel
on different GPUs if available.

---

## Step 3 — Cascaded Inference

```bash
python scripts/sample_mri_cascade.py \
    --base_model   checkpoints/base_96/ema_0.9999_200000.pt \
    --sr_model     checkpoints/sr_384/ema_0.9999_200000.pt \
    --num_samples  16 \
    --batch_size   4 \
    --timestep_respacing    250 \
    --sr_timestep_respacing 250 \
    --out_dir      samples/
```

Output:
  samples/96/sample_XXXX.png   ← 96x96 base model output
  samples/384/sample_XXXX.png  ← 384x384 final output
  samples/grid_96.png          ← summary grid
  samples/grid_384.png

timestep_respacing=250 means 250 steps instead of 1000 (4× faster).

---

## Critical Notes

1. in_channels=1 EVERYWHERE
   guided-diffusion defaults to RGB (3 channels). You must set
   in_channels=1 for grayscale MRI in ALL configs.

2. learn_sigma=True means out_channels = 2
   The model predicts both image and variance → out_channels = 2*in_channels = 2.
   This is already handled in the scripts.

3. Always use EMA checkpoints for sampling
   ema_0.9999_XXXXXX.pt, NOT model_XXXXXX.pt.

4. Nearest-neighbour upsampling for SR conditioning
   The blurry condition image uses NN upsample (np.repeat), matching
   how the training data was created.

5. 16-bit PNG for MRI
   All images saved as PIL mode 'I;16' (uint16, [0, 65535]).
   Training normalises to [-1, 1] before feeding to model.

6. Noise schedules match the paper
   Base model = cosine  (better for natural/clean image generation)
   SR model   = linear  (better for refinement / conditional generation)

---

## File Overview

scripts/
  mri_preprocess.py       Step 0: raw h5 volumes → paired PNG datasets
  mri_dataloader.py       Custom PyTorch datasets (used by training scripts)
  train_base_96.py        Step 1: train 96x96 base diffusion model
  train_sr_384.py         Step 2: train 96→384 SR diffusion model
  sample_mri_cascade.py   Step 3: run 2-step cascaded inference
