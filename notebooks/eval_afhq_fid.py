"""
AFHQ EDM Cascaded Diffusion — FID & Sample Visualization
=========================================================
Evaluate all 3 EDM cascade stages: Base 32×32, SR 64×64, SR 256×256.
Uses EDM's Heun 2nd-order ODE sampler + InceptionV3 FID.
"""

# %% [markdown]
# # AFHQ EDM Cascade — FID & Samples
# Evaluate FID for up to 3 stages:
# - **Base** (32×32)
# - **SR-64** (32→64)
# - **SR-256** (64→256)

# %% Imports & setup
import os, sys, glob, math, pickle, re, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy import linalg

# ── Make EDM + scripts importable ─────────────────────────────────────────────
PROJECT_ROOT = '/data/Sahil/mri_cascaded_diffusion'
EDM_PATH = os.path.join(PROJECT_ROOT, 'edm_repo')
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, 'scripts')
for p in [EDM_PATH, SCRIPTS_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# %% [markdown]
# ## 1. Configuration

# %%
GPU_ID = 4
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/afhq')

# ── Checkpoints (set to None to skip a stage) ────────────────────────────────
base_ckpt   = None   # auto-detected below
sr_64_ckpt  = None
sr_256_ckpt = None

# ── Sampling ──────────────────────────────────────────────────────────────────
NUM_SAMPLES = 5000    # total images for FID (more = more accurate)
BATCH_SIZE  = 32
NUM_STEPS   = 18      # Heun steps per stage
SIGMA_MIN   = 0.002
SIGMA_MAX   = 80
RHO         = 7
SEED        = 42

# ── Which stages to evaluate FID for ─────────────────────────────────────────
EVAL_BASE   = True
EVAL_SR64   = True
EVAL_SR256  = True

# %% Auto-detect checkpoints
def find_latest_pkl(pattern):
    matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    return matches[-1] if matches else None

if base_ckpt is None:
    base_ckpt = find_latest_pkl("checkpoints/edm_afhq_base_32/**/network-snapshot-*.pkl")
if sr_64_ckpt is None:
    sr_64_ckpt = find_latest_pkl("checkpoints/edm_afhq_sr_64/**/network-snapshot-*.pkl")
if sr_256_ckpt is None:
    sr_256_ckpt = find_latest_pkl("checkpoints/edm_afhq_sr_256*/**/network-snapshot-*.pkl")

for name, path in [("Base", base_ckpt), ("SR-64", sr_64_ckpt), ("SR-256", sr_256_ckpt)]:
    if path:
        kimg = re.search(r'snapshot-(\d+)', path)
        kimg = int(kimg.group(1)) if kimg else '?'
        print(f"  {name:8s}: kimg={kimg}  →  {os.path.basename(path)}")
    else:
        print(f"  {name:8s}: not found (will skip)")

# %% [markdown]
# ## 2. EDM Heun Sampler

# %%
@torch.no_grad()
def edm_heun_sampler(net, latents, num_steps=18, sigma_min=0.002,
                     sigma_max=80, rho=7, **model_kwargs):
    """EDM 2nd-order Heun sampler (Algorithm 2, Karras et al. 2022)."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat = x_next
        denoised = net(x_hat.to(torch.float32), t_cur.to(torch.float32),
                       **model_kwargs).to(torch.float64)
        d_cur = (x_hat - denoised) / t_cur
        x_next = x_hat + (t_next - t_cur) * d_cur
        if i < num_steps - 1:  # Heun correction
            denoised = net(x_next.to(torch.float32), t_next.to(torch.float32),
                           **model_kwargs).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.to(torch.float32)

# %% [markdown]
# ## 3. Load Models

# %%
models = {}

if base_ckpt:
    print(f"Loading base model...")
    with open(base_ckpt, 'rb') as f:
        data = pickle.load(f)
    models['base'] = data['ema'].to(device).eval()
    print(f"  ✓ Base: {sum(p.numel() for p in models['base'].parameters()):,} params, "
          f"res={models['base'].img_resolution}")
    del data

if sr_64_ckpt:
    print(f"Loading SR-64 model...")
    with open(sr_64_ckpt, 'rb') as f:
        data = pickle.load(f)
    models['sr_64'] = data['ema'].to(device).eval()
    print(f"  ✓ SR-64: {sum(p.numel() for p in models['sr_64'].parameters()):,} params")
    del data

if sr_256_ckpt:
    print(f"Loading SR-256 model...")
    with open(sr_256_ckpt, 'rb') as f:
        data = pickle.load(f)
    models['sr_256'] = data['ema'].to(device).eval()
    print(f"  ✓ SR-256: {sum(p.numel() for p in models['sr_256'].parameters()):,} params")
    del data

print(f"\nLoaded: {list(models.keys())}")
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Sampling Functions

# %%
def sample_base(n, bs=BATCH_SIZE):
    """Generate n base 32×32 samples. Returns tensor in [-1, 1]."""
    net = models['base']
    res = net.img_resolution
    ch = net.img_channels
    all_s = []
    for i in tqdm(range(0, n, bs), desc='Base 32×32'):
        cur_bs = min(bs, n - i)
        latents = torch.randn([cur_bs, ch, res, res], device=device)
        s = edm_heun_sampler(net, latents, num_steps=NUM_STEPS,
                             sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO)
        all_s.append(s.cpu())
    return torch.cat(all_s, 0)[:n]

def sample_sr(low_res_samples, sr_key, bs=BATCH_SIZE):
    """Run SR on low_res_samples. Returns tensor in [-1, 1]."""
    net = models[sr_key]
    res = net.img_resolution
    ch = net.img_channels
    n = low_res_samples.shape[0]
    all_s = []
    for i in tqdm(range(0, n, bs), desc=f'SR → {res}×{res}'):
        cur_bs = min(bs, n - i)
        low = low_res_samples[i:i+cur_bs].to(device)
        latents = torch.randn([cur_bs, ch, res, res], device=device)
        s = edm_heun_sampler(net, latents, num_steps=NUM_STEPS,
                             sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
                             low_res=low)
        all_s.append(s.cpu())
    return torch.cat(all_s, 0)[:n]

def to_01(t):
    """[-1,1] → [0,1]"""
    return (t.clamp(-1, 1) + 1) / 2

def tensor_to_uint8(t):
    """(B,C,H,W) in [-1,1] → (B,H,W,C) uint8 numpy."""
    return ((t + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()

# %% [markdown]
# ## 5. Visualization — Quick Sample Grids

# %%
N_VIS = 16

def plot_grid(images_np, title, nrow=4, figsize=None):
    n = len(images_np)
    ncol = nrow
    nrow_actual = math.ceil(n / ncol)
    if figsize is None:
        figsize = (ncol * 1.5, nrow_actual * 1.5)
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=figsize)
    axes = np.array(axes).flatten()
    for i in range(len(axes)):
        axes[i].axis('off')
        if i < n:
            axes[i].imshow(images_np[i])
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Generate visualization samples
if SEED is not None:
    torch.manual_seed(SEED)

print(f"Generating {N_VIS} visualization samples...")
vis_results = {}

if 'base' in models:
    vis_results['32×32'] = sample_base(N_VIS, bs=min(8, N_VIS))
    plot_grid(tensor_to_uint8(vis_results['32×32']), 'Base 32×32', nrow=4)

if 'sr_64' in models and '32×32' in vis_results:
    vis_results['64×64'] = sample_sr(vis_results['32×32'], 'sr_64', bs=min(8, N_VIS))
    plot_grid(tensor_to_uint8(vis_results['64×64']), 'SR 64×64', nrow=4)

if 'sr_256' in models and '64×64' in vis_results:
    vis_results['256×256'] = sample_sr(vis_results['64×64'], 'sr_256', bs=min(4, N_VIS))
    plot_grid(tensor_to_uint8(vis_results['256×256']), 'SR 256×256', nrow=4)

# %% Side-by-side cascade comparison
if len(vis_results) > 1:
    stage_names = list(vis_results.keys())
    n_show = min(8, N_VIS)
    n_stages = len(stage_names)
    fig, axes = plt.subplots(n_show, n_stages, figsize=(n_stages * 2.5, n_show * 2.5))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    for col, name in enumerate(stage_names):
        imgs = tensor_to_uint8(vis_results[name][:n_show])
        axes[0, col].set_title(name, fontsize=11, fontweight='bold')
        for row in range(n_show):
            axes[row, col].imshow(imgs[row])
            axes[row, col].axis('off')
    fig.suptitle('EDM Cascade Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. InceptionV3 Feature Extractor

# %%
from torchvision import models as tv_models

class InceptionV3Features(nn.Module):
    """Extract pool3 (2048-d) features from InceptionV3."""
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
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.pool(self.blocks(x)).flatten(1)

print("Loading InceptionV3...")
inception = InceptionV3Features().to(device)
print("✓ InceptionV3 loaded.")

# %% [markdown]
# ## 7. FID Computation

# %% FID helpers
def compute_statistics(features):
    mu = np.mean(features, axis=0)
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

def load_real_images(data_dir, image_size, max_images=None):
    """Load real images, resize to image_size, return (N, 3, H, W) in [0,1]."""
    paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    if max_images:
        paths = paths[:max_images]
    images = []
    for p in tqdm(paths, desc=f'Loading real {image_size}×{image_size}'):
        img = Image.open(p).convert('RGB').resize((image_size, image_size), Image.LANCZOS)
        images.append(np.array(img).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)

@torch.no_grad()
def extract_features(images_01, batch_size=64):
    """images_01: (N, 3, H, W) in [0, 1]. Returns (N, 2048) numpy."""
    feats = []
    for i in tqdm(range(0, len(images_01), batch_size), desc='InceptionV3'):
        batch = images_01[i:i+batch_size].to(device)
        feats.append(inception(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)

# %% [markdown]
# ### 7a. Real image statistics (per resolution)

# %%
real_stats = {}
resolutions_needed = []
if EVAL_BASE and 'base' in models:
    resolutions_needed.append(32)
if EVAL_SR64 and 'sr_64' in models:
    resolutions_needed.append(64)
if EVAL_SR256 and 'sr_256' in models:
    resolutions_needed.append(256)

for res in resolutions_needed:
    print(f"Computing real stats at {res}×{res}...")
    real = load_real_images(DATA_DIR, res, max_images=NUM_SAMPLES)
    feats = extract_features(real)
    real_stats[res] = compute_statistics(feats)
    print(f"  ✓ {len(feats)} images, features {feats.shape}")
    del real, feats

# %% [markdown]
# ### 7b. Generate samples & compute FID

# %%
if SEED is not None:
    torch.manual_seed(SEED)

fid_results = {}
gen_cache = {}  # cache samples for downstream SR stages

# ── Base 32×32 FID ────────────────────────────────────────────────────────────
if EVAL_BASE and 'base' in models:
    print(f"\nGenerating {NUM_SAMPLES} base 32×32 samples...")
    gen_32 = sample_base(NUM_SAMPLES, bs=BATCH_SIZE)
    gen_cache[32] = gen_32

    feats = extract_features(to_01(gen_32))
    mu_g, sig_g = compute_statistics(feats)
    fid_32 = frechet_distance(*real_stats[32], mu_g, sig_g)
    fid_results['Base 32×32'] = fid_32
    print(f"  ★ FID (Base 32×32): {fid_32:.2f}")
    del feats

# ── SR 64×64 FID ──────────────────────────────────────────────────────────────
if EVAL_SR64 and 'sr_64' in models:
    if 32 not in gen_cache:
        print(f"Generating {NUM_SAMPLES} base samples for SR input...")
        gen_cache[32] = sample_base(NUM_SAMPLES, bs=BATCH_SIZE)

    print(f"\nRunning SR 32→64 on {NUM_SAMPLES} samples...")
    gen_64 = sample_sr(gen_cache[32], 'sr_64', bs=BATCH_SIZE)
    gen_cache[64] = gen_64

    feats = extract_features(to_01(gen_64))
    mu_g, sig_g = compute_statistics(feats)
    fid_64 = frechet_distance(*real_stats[64], mu_g, sig_g)
    fid_results['SR 64×64'] = fid_64
    print(f"  ★ FID (SR 64×64): {fid_64:.2f}")
    del feats

# ── SR 256×256 FID ────────────────────────────────────────────────────────────
if EVAL_SR256 and 'sr_256' in models:
    if 64 not in gen_cache:
        if 32 not in gen_cache:
            gen_cache[32] = sample_base(NUM_SAMPLES, bs=BATCH_SIZE)
        if 'sr_64' in models:
            gen_cache[64] = sample_sr(gen_cache[32], 'sr_64', bs=BATCH_SIZE)
        else:
            print("⚠️ SR-256 requires SR-64 output as input! Skipping.")

    if 64 in gen_cache:
        print(f"\nRunning SR 64→256 on {NUM_SAMPLES} samples...")
        gen_256 = sample_sr(gen_cache[64], 'sr_256', bs=min(8, BATCH_SIZE))
        gen_cache[256] = gen_256

        feats = extract_features(to_01(gen_256))
        mu_g, sig_g = compute_statistics(feats)
        fid_256 = frechet_distance(*real_stats[256], mu_g, sig_g)
        fid_results['SR 256×256'] = fid_256
        print(f"  ★ FID (SR 256×256): {fid_256:.2f}")
        del feats

# %% [markdown]
# ### 7c. Results Summary

# %%
print("\n" + "=" * 60)
print("  FID RESULTS SUMMARY")
print("=" * 60)
for name, path in [("Base", base_ckpt), ("SR-64", sr_64_ckpt), ("SR-256", sr_256_ckpt)]:
    if path:
        print(f"  {name:8s}: {os.path.basename(path)}")
print(f"  Samples:  {NUM_SAMPLES:,}")
print(f"  Steps:    {NUM_STEPS} (Heun)")
print("-" * 60)
for stage, fid in fid_results.items():
    print(f"  FID {stage:16s}:  {fid:.2f}")
print("=" * 60)

# Bar chart
if fid_results:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(fid_results.keys())
    vals = list(fid_results.values())
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(names)]
    bars = ax.bar(names, vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('FID ↓', fontsize=12)
    ax.set_title('EDM Cascade — FID Scores', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Training Loss Curves

# %%
def plot_training_loss(stats_path, title="Training Loss"):
    if not os.path.exists(stats_path):
        return
    kimg_vals, loss_vals = [], []
    with open(stats_path) as f:
        for line in f:
            d = json.loads(line)
            if 'Progress/kimg' in d and 'Loss/loss' in d:
                kimg_vals.append(d['Progress/kimg']['mean'])
                loss_vals.append(d['Loss/loss']['mean'])
    if not kimg_vals:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kimg_vals, loss_vals, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('kimg'); ax.set_ylabel('Loss')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_yscale('log')
    plt.tight_layout(); plt.show()
    print(f"  Latest: kimg={kimg_vals[-1]:.0f}, loss={loss_vals[-1]:.4f}")

for pattern, title in [
    ('checkpoints/edm_afhq_base_32/**/stats.jsonl', 'Base Model Loss'),
    ('checkpoints/edm_afhq_sr_64/**/stats.jsonl', 'SR-64 Model Loss'),
    ('checkpoints/edm_afhq_sr_256*/**/stats.jsonl', 'SR-256 Model Loss'),
]:
    matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    if matches:
        plot_training_loss(matches[-1], title)

# %% [markdown]
# ## 9. Large Sample Grid

# %%
N_LARGE = 64
if 'base' in models:
    print(f"Generating {N_LARGE} samples for large grid...")
    lg = sample_base(N_LARGE, bs=BATCH_SIZE)
    plot_grid(tensor_to_uint8(lg), f'Base 32×32 — {N_LARGE} samples', nrow=8, figsize=(16, 16))

    if 'sr_64' in models:
        lg_64 = sample_sr(lg, 'sr_64', bs=BATCH_SIZE)
        plot_grid(tensor_to_uint8(lg_64), f'SR 64×64 — {N_LARGE} samples', nrow=8, figsize=(16, 16))

print("\nDone! ✓")
