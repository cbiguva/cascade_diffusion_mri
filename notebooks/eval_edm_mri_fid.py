"""
EDM Cascaded Diffusion — MRI FID Evaluation
============================================
Evaluate FID for up to 2 EDM cascade stages:
  - Stage 1 / Base  (96×96, 2-channel complex MRI)
  - Stage 2 / SR    (96→384, 2-channel complex MRI)

MRI data is 2-channel (Real + Imaginary).
FID is computed on the magnitude image: mag = sqrt(Re² + Im²),
converted to 3-ch grayscale for InceptionV3.

Real reference images are loaded from .pt files (edm_mri_dataloader format):
    each file contains {'slices': (S, 2, 384, 384), 'global_scale': scalar}
"""

# %% [markdown]
# # EDM MRI Cascade — FID Evaluation
# Evaluate FID for:
# - **Base 96×96**  (pure noise → 2ch MRI at 96×96)
# - **SR 384×384**  (96→384 super-resolution)

# %% Imports & setup
import os, sys, glob, math, pickle, re, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy import linalg

# ── Make EDM + scripts importable ─────────────────────────────────────────────
if os.path.basename(os.getcwd()) == 'notebooks':
    PROJECT_ROOT = os.path.dirname(os.getcwd())
else:
    PROJECT_ROOT = '/data/Sahil/mri_cascaded_diffusion'

EDM_PATH     = os.path.join(PROJECT_ROOT, 'edm_repo')
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, 'scripts')
for p in [EDM_PATH, SCRIPTS_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"Project root : {PROJECT_ROOT}")

# %% [markdown]
# ## 1. Configuration

# %%
GPU_ID = 0
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Real MRI data directory (.pt files) ───────────────────────────────────────
# Each .pt: {'slices': (S, 2, 384, 384), 'global_scale': scalar}
MRI_DATA_DIR = os.path.join(PROJECT_ROOT, 'data/mri_pt')   # ← set to your .pt dir

# ── Checkpoints (None = auto-detect) ─────────────────────────────────────────
base_ckpt   = None
sr_384_ckpt = None

# ── Sampling ──────────────────────────────────────────────────────────────────
NUM_SAMPLES = 1000    # images for FID  (≥1000 recommended; AFHQ used 5000)
BATCH_SIZE  = 8
NUM_STEPS   = 18      # Heun steps per stage
SIGMA_MIN   = 0.002
SIGMA_MAX   = 80.0
RHO         = 7
SEED        = 42
COND_AUG_TIMESTEP = 0   # 0 = clean conditioning (recommended at inference)

# ── Which stages to evaluate ──────────────────────────────────────────────────
EVAL_BASE  = True
EVAL_SR384 = True

# ── Auto-detect latest checkpoints ───────────────────────────────────────────
def find_latest_pkl(pattern):
    matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    return matches[-1] if matches else None

if base_ckpt is None:
    base_ckpt = find_latest_pkl("checkpoints/edm_mri_base_96/**/network-snapshot-*.pkl")
if sr_384_ckpt is None:
    sr_384_ckpt = find_latest_pkl("checkpoints/edm_mri_sr_384/**/network-snapshot-*.pkl")

for name, path in [("Base-96", base_ckpt), ("SR-384", sr_384_ckpt)]:
    if path:
        m = re.search(r'snapshot-(\d+)', path)
        kimg = int(m.group(1)) if m else '?'
        print(f"  {name:10s}: kimg={kimg}  →  {os.path.basename(path)}")
    else:
        print(f"  {name:10s}: not found (will skip)")

if SEED is not None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

# %% [markdown]
# ## 2. EDM Heun Sampler

# %%
@torch.no_grad()
def edm_heun_sampler(net, latents, num_steps=18, sigma_min=0.002,
                     sigma_max=80.0, rho=7, **model_kwargs):
    """EDM 2nd-order Heun sampler (Karras et al. 2022, Algorithm 2)."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat = x_next
        denoised = net(x_hat.to(torch.float32), t_cur.to(torch.float32),
                       **model_kwargs).to(torch.float64)
        d_cur = (x_hat - denoised) / t_cur
        x_next = x_hat + (t_next - t_cur) * d_cur
        if i < num_steps - 1:   # Heun correction
            denoised = net(x_next.to(torch.float32), t_next.to(torch.float32),
                           **model_kwargs).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.to(torch.float32)

# %% [markdown]
# ## 3. CDM Conditioning Augmentation

# %%
def _linear_beta_schedule(T=1000):
    scale = 1000 / T
    betas = np.linspace(scale * 0.0001, scale * 0.02, T, dtype=np.float64)
    ac = np.cumprod(1.0 - betas)
    return (torch.from_numpy(np.sqrt(ac)).float(),
            torch.from_numpy(np.sqrt(1.0 - ac)).float())

_sqrt_ac, _sqrt_1m = _linear_beta_schedule()

def apply_cond_aug(low_res: torch.Tensor, s: int) -> torch.Tensor:
    """CDM §4.2 conditioning augmentation. s=0 → no-op."""
    if s <= 0:
        return low_res
    a = _sqrt_ac[s - 1].to(low_res.device)
    b = _sqrt_1m[s - 1].to(low_res.device)
    return a * low_res + b * torch.randn_like(low_res)

# %% [markdown]
# ## 4. Load Models

# %%
from edm_sr_model import EDMSRPrecond          # noqa – needed for pickle
from training.networks import EDMPrecond        # noqa – needed for pickle

models = {}

if base_ckpt and EVAL_BASE:
    print(f"Loading base model …")
    with open(base_ckpt, 'rb') as f:
        data = pickle.load(f)
    models['base'] = data['ema'].to(device).eval()
    print(f"  ✓  {sum(p.numel() for p in models['base'].parameters()):,} params, "
          f"res={models['base'].img_resolution}")
    del data

if sr_384_ckpt and EVAL_SR384:
    print(f"Loading SR-384 model …")
    with open(sr_384_ckpt, 'rb') as f:
        data = pickle.load(f)
    models['sr_384'] = data['ema'].to(device).eval()
    print(f"  ✓  {sum(p.numel() for p in models['sr_384'].parameters()):,} params, "
          f"res={models['sr_384'].img_resolution}")
    del data

print(f"\nLoaded: {list(models.keys())}")
torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Magnitude Conversion Helpers
#
# MRI data is 2-channel (Real, Imaginary).  For FID we convert to magnitude
# and tile to 3-ch for InceptionV3:
#   mag = sqrt(Re² + Im²), normalised to [0, 1], then broadcast → (B, 3, H, W).

# %%
def to_magnitude_3ch(x: torch.Tensor) -> torch.Tensor:
    """
    (B, 2, H, W) float32 in [-1,1]  →  (B, 3, H, W) float32 in [0, 1]

    Magnitude is computed per image, normalised independently, then tiled to 3ch
    so InceptionV3 can process it.
    """
    re, im = x[:, 0], x[:, 1]          # (B, H, W)
    mag = torch.sqrt(re ** 2 + im ** 2).float()
    # per-image normalisation → [0, 1]
    b = mag.shape[0]
    mn = mag.view(b, -1).min(1).values.view(b, 1, 1)
    mx = mag.view(b, -1).max(1).values.view(b, 1, 1)
    mag = (mag - mn) / (mx - mn + 1e-8)
    # tile grayscale → 3ch  (B, 1, H, W) → (B, 3, H, W)
    return mag.unsqueeze(1).repeat(1, 3, 1, 1)

def mri_magnitude_np(x: torch.Tensor) -> np.ndarray:
    """(B, 2, H, W) → (B, H, W) numpy in [0, 1] for display."""
    return to_magnitude_3ch(x)[:, 0].cpu().numpy()

# %% [markdown]
# ## 6. Sampling Functions

# %%
def sample_base(n, bs=BATCH_SIZE):
    """Generate n base 96×96 samples. Returns (N, 2, 96, 96) in [-1, 1]."""
    net = models['base']
    res = net.img_resolution
    ch  = net.img_channels
    all_s = []
    for i in tqdm(range(0, n, bs), desc='Base 96×96'):
        cur_bs = min(bs, n - i)
        latents = torch.randn([cur_bs, ch, res, res], device=device)
        s = edm_heun_sampler(net, latents, num_steps=NUM_STEPS,
                             sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO)
        all_s.append(s.cpu())
    return torch.cat(all_s, 0)[:n]

def sample_sr384(low_res_samples, bs=BATCH_SIZE):
    """Run SR on (N, 2, 96, 96) → (N, 2, 384, 384)."""
    net = models['sr_384']
    res = net.img_resolution
    ch  = net.img_channels
    n   = low_res_samples.shape[0]
    all_s = []
    for i in tqdm(range(0, n, bs), desc='SR 96→384'):
        cur_bs = min(bs, n - i)
        low = apply_cond_aug(low_res_samples[i:i+cur_bs].to(device),
                             s=COND_AUG_TIMESTEP)
        latents = torch.randn([cur_bs, ch, res, res], device=device)
        s = edm_heun_sampler(net, latents, num_steps=NUM_STEPS,
                             sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
                             low_res=low)
        all_s.append(s.cpu())
    return torch.cat(all_s, 0)[:n]

# %% [markdown]
# ## 7. Quick Visualisation

# %%
def plot_magnitude_grid(x, title, nrow=8):
    mag = mri_magnitude_np(x)
    n   = len(mag)
    ncol = min(nrow, n)
    nrows = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 1.8, nrows * 1.8))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < n:
            ax.imshow(mag[i], cmap='gray', interpolation='bilinear')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show()

N_VIS = 8
if SEED is not None:
    torch.manual_seed(SEED)

print(f"Generating {N_VIS} visualisation samples…")
vis_base = sample_base(N_VIS, bs=min(4, N_VIS))
plot_magnitude_grid(vis_base, f'Base 96×96  ({NUM_STEPS} Heun steps, magnitude)', nrow=4)

if 'sr_384' in models:
    vis_sr = sample_sr384(vis_base, bs=min(4, N_VIS))
    plot_magnitude_grid(vis_sr, f'SR 384×384  ({NUM_STEPS} Heun steps, magnitude)', nrow=4)

# Side-by-side cascade comparison
if 'sr_384' in models:
    n_show = min(4, N_VIS)
    fig, axes = plt.subplots(n_show, 3, figsize=(9, n_show * 2.5))
    base_mag = mri_magnitude_np(vis_base[:n_show])
    sr_mag   = mri_magnitude_np(vis_sr[:n_show])
    base_up  = F.interpolate(vis_base[:n_show], size=384, mode='nearest')
    up_mag   = mri_magnitude_np(base_up)
    for row in range(n_show):
        for col, (data, title) in enumerate([(base_mag, '96 (Base)'),
                                              (up_mag,  '384 (NN-up)'),
                                              (sr_mag,  '384 (EDM-SR)')]):
            axes[row, col].imshow(data[row], cmap='gray')
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(title, fontsize=10, fontweight='bold')
    fig.suptitle('EDM MRI Cascade Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. InceptionV3 Feature Extractor

# %%
from torchvision import models as tv_models

class InceptionV3Features(nn.Module):
    """Extract pool3 (2048-d) features from InceptionV3 for FID."""
    def __init__(self):
        super().__init__()
        inception = tv_models.inception_v3(pretrained=True)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1]."""
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.pool(self.blocks(x)).flatten(1)

print("Loading InceptionV3…")
inception = InceptionV3Features().to(device)
print("✓ InceptionV3 ready.")

# %% [markdown]
# ## 9. Real MRI Statistics

# %%
def compute_statistics(features):
    mu    = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

@torch.no_grad()
def extract_features(images_3ch_01, batch_size=32):
    """images_3ch_01: (N, 3, H, W) in [0,1]. Returns (N, 2048) numpy."""
    feats = []
    for i in tqdm(range(0, len(images_3ch_01), batch_size), desc='InceptionV3'):
        batch = images_3ch_01[i:i+batch_size].to(device)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)

# ── Load real MRI slices from .pt files ──────────────────────────────────────
def load_real_mri_slices(pt_dir: str, target_res: int, max_images: int = None):
    """
    Load real 2-channel MRI slices from .pt files and return as
    (N, 3, target_res, target_res) float32 in [0, 1] (magnitude, 3-ch).

    Each .pt file format (edm_mri_dataloader):
        {'slices': (S, 2, 384, 384), 'global_scale': scalar}
    """
    pt_files = sorted(Path(pt_dir).glob("*.pt"))
    assert len(pt_files) > 0, f"No .pt files in {pt_dir}"
    all_slices = []
    pbar = tqdm(pt_files, desc=f'Loading real MRI ({target_res}×{target_res})')
    for fpath in pbar:
        if max_images and len(all_slices) >= max_images:
            break
        try:
            obj = torch.load(str(fpath), map_location='cpu')
        except Exception:
            try:
                obj = torch.load(str(fpath), map_location='cpu',
                                 weights_only=False)
            except Exception as e:
                print(f"  Skip {fpath.name}: {e}")
                continue
        slices = obj['slices']  # (S, 2, 384, 384)
        all_slices.append(slices)
        if max_images and sum(s.shape[0] for s in all_slices) >= max_images:
            break

    real = torch.cat(all_slices, dim=0)[:max_images]  # (N, 2, 384, 384)

    # Resize to target resolution if needed
    if target_res != real.shape[-1]:
        real = F.interpolate(real, size=target_res, mode='bilinear',
                             align_corners=False)

    mag3ch = to_magnitude_3ch(real)  # (N, 3, target_res, target_res) in [0,1]
    print(f"  Loaded {mag3ch.shape[0]} real slices → {tuple(mag3ch.shape)}")
    return mag3ch

# ── Compute real stats for each resolution needed ────────────────────────────
real_stats = {}
resolutions_needed = []
if EVAL_BASE  and 'base'   in models:
    resolutions_needed.append(96)
if EVAL_SR384 and 'sr_384' in models:
    resolutions_needed.append(384)

for res in resolutions_needed:
    print(f"\nComputing real MRI statistics at {res}×{res}…")
    real_imgs = load_real_mri_slices(MRI_DATA_DIR, target_res=res,
                                     max_images=NUM_SAMPLES)
    feats = extract_features(real_imgs)
    real_stats[res] = compute_statistics(feats)
    print(f"  ✓  {len(feats)} real images, features {feats.shape}")
    del real_imgs, feats

# %% [markdown]
# ## 10. Generate Samples & Compute FID

# %%
if SEED is not None:
    torch.manual_seed(SEED)

fid_results = {}
gen_cache   = {}

# ── Base 96×96 FID ────────────────────────────────────────────────────────────
if EVAL_BASE and 'base' in models and 96 in real_stats:
    print(f"\nGenerating {NUM_SAMPLES} base 96×96 samples for FID…")
    gen_96 = sample_base(NUM_SAMPLES, bs=BATCH_SIZE)
    gen_cache[96] = gen_96

    feats = extract_features(to_magnitude_3ch(gen_96))
    mu_g, sig_g = compute_statistics(feats)
    fid_96 = frechet_distance(*real_stats[96], mu_g, sig_g)
    fid_results['Base 96×96'] = fid_96
    print(f"  ★  FID (Base 96×96)  : {fid_96:.2f}")
    del feats

# ── SR 384×384 FID ────────────────────────────────────────────────────────────
if EVAL_SR384 and 'sr_384' in models and 384 in real_stats:
    if 96 not in gen_cache:
        print(f"Generating {NUM_SAMPLES} base samples for SR input…")
        gen_cache[96] = sample_base(NUM_SAMPLES, bs=BATCH_SIZE)

    print(f"\nRunning SR 96→384 on {NUM_SAMPLES} samples for FID…")
    gen_384 = sample_sr384(gen_cache[96], bs=BATCH_SIZE)
    gen_cache[384] = gen_384

    feats = extract_features(to_magnitude_3ch(gen_384))
    mu_g, sig_g = compute_statistics(feats)
    fid_384 = frechet_distance(*real_stats[384], mu_g, sig_g)
    fid_results['SR 384×384'] = fid_384
    print(f"  ★  FID (SR 384×384)  : {fid_384:.2f}")
    del feats

# %% [markdown]
# ## 11. FID Summary & Bar Chart

# %%
print("\n" + "=" * 60)
print("  FID RESULTS — EDM MRI CASCADE")
print("=" * 60)
for name, path in [("Base-96", base_ckpt), ("SR-384", sr_384_ckpt)]:
    if path:
        print(f"  {name:10s}: {os.path.basename(path)}")
print(f"  Samples   : {NUM_SAMPLES:,}")
print(f"  Steps     : {NUM_STEPS} (Heun)")
print(f"  Cond-aug  : s={COND_AUG_TIMESTEP}  (SR stage)")
print("-" * 60)
for stage, fid in fid_results.items():
    print(f"  FID {stage:18s}:  {fid:.2f}")
print("=" * 60)

if fid_results:
    fig, ax = plt.subplots(figsize=(6, 4))
    names  = list(fid_results.keys())
    vals   = list(fid_results.values())
    colors = ['#2196F3', '#4CAF50'][:len(names)]
    bars   = ax.bar(names, vals, color=colors, width=0.45,
                    edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', va='bottom',
                fontweight='bold', fontsize=12)
    ax.set_ylabel('FID ↓', fontsize=12)
    ax.set_title('EDM MRI Cascade — FID Scores', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 12. Training Loss Curves (Optional)

# %%
def plot_training_loss(stats_path, title="Training Loss"):
    if not os.path.exists(stats_path):
        print(f"Not found: {stats_path}")
        return
    kimg_vals, loss_vals = [], []
    with open(stats_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                if 'Progress/kimg' in d and 'Loss/loss' in d:
                    kimg_vals.append(d['Progress/kimg']['mean'])
                    loss_vals.append(d['Loss/loss']['mean'])
            except Exception:
                pass
    if not kimg_vals:
        print(f"  (no data in {stats_path})")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kimg_vals, loss_vals, linewidth=1.5, alpha=0.85)
    ax.set_xlabel('kimg', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    print(f"  Latest: kimg={kimg_vals[-1]:.0f},  loss={loss_vals[-1]:.5f}")

for pattern, title in [
    ('checkpoints/edm_mri_base_96/**/stats.jsonl', 'Base Model Loss (96×96)'),
    ('checkpoints/edm_mri_sr_384/**/stats.jsonl',  'SR Model Loss (96→384)'),
]:
    matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    if matches:
        plot_training_loss(matches[-1], title)
    else:
        print(f"No stats.jsonl found for: {pattern}")

# %% [markdown]
# ## 13. Large Sample Grid

# %%
N_LARGE = 32
if 'base' in models:
    print(f"Generating {N_LARGE} samples for large grid…")
    lg_base = sample_base(N_LARGE, bs=BATCH_SIZE)
    plot_magnitude_grid(lg_base,
                        f'Base 96×96 — {N_LARGE} samples (magnitude)', nrow=8)

    if 'sr_384' in models:
        lg_sr = sample_sr384(lg_base, bs=BATCH_SIZE)
        plot_magnitude_grid(lg_sr,
                            f'SR 384×384 — {N_LARGE} samples (magnitude)', nrow=8)

print("\nDone! ✓")
