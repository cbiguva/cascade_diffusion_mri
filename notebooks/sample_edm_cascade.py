# %% [markdown]
# # EDM Cascaded Diffusion – Sampling Notebook
#
# Sample from the cascaded pipeline:
# - **Stage 1**: Base model → 32×32
# - **Stage 2** (optional): SR model → 64×64
# - **Stage 3** (optional): SR model → 256×256
#
# Uses EDM's **Heun 2nd-order ODE sampler** (Algorithm 2 from Karras et al., 2022).
# Only **18 steps** per stage — vastly faster than DDPM's 250-1000 steps.

# %% [markdown]
# ## 1. Setup & Configuration

# %%
import sys, os, glob, pickle, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display

# ── Make project imports work ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), '..'))) \
    if 'notebooks' in os.getcwd() else os.getcwd()
# Heuristic: if running from notebooks/, go up one level
if os.path.basename(os.getcwd()) == 'notebooks':
    PROJECT_ROOT = os.path.dirname(os.getcwd())
else:
    PROJECT_ROOT = '/data/Sahil/mri_cascaded_diffusion'

EDM_PATH = os.path.join(PROJECT_ROOT, 'edm_repo')
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, 'scripts')
for p in [EDM_PATH, SCRIPTS_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"Project root: {PROJECT_ROOT}")
print(f"EDM repo:     {EDM_PATH}")

# %%
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these paths and settings
# ═══════════════════════════════════════════════════════════════════════════════

# GPU selection
GPU_ID = 4
device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Model checkpoints ────────────────────────────────────────────────────────
# Set to None to skip a stage. At minimum, base_ckpt must be set.

# Base model (32×32) — always required
base_ckpt = None   # ← Set path, e.g.: "checkpoints/edm_afhq_base_32/.../network-snapshot-XXXXXX.pkl"

# SR model 1: 32→64 — optional
sr_64_ckpt = None  # ← Set path, e.g.: "checkpoints/edm_afhq_sr_64/network-snapshot-XXXXXX.pkl"

# SR model 2: 64→256 — optional
sr_256_ckpt = None

# ── Auto-detect latest checkpoints ───────────────────────────────────────────
def find_latest_pkl(pattern):
    """Find the latest .pkl checkpoint matching a glob pattern."""
    matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    if matches:
        return matches[-1]
    return None

if base_ckpt is None:
    base_ckpt = find_latest_pkl("checkpoints/edm_afhq_base_32/**/network-snapshot-*.pkl")
    if base_ckpt:
        print(f"Auto-detected base ckpt: {base_ckpt}")
    else:
        print("⚠️  No base checkpoint found. Set base_ckpt manually.")

if sr_64_ckpt is None:
    sr_64_ckpt = find_latest_pkl("checkpoints/edm_afhq_sr_64/**/network-snapshot-*.pkl")
    if sr_64_ckpt:
        print(f"Auto-detected SR-64 ckpt: {sr_64_ckpt}")
    else:
        print("ℹ️  No SR-64 checkpoint found (will run base only).")

if sr_256_ckpt is None:
    sr_256_ckpt = find_latest_pkl("checkpoints/edm_afhq_sr_256*/**/network-snapshot-*.pkl")
    if sr_256_ckpt:
        print(f"Auto-detected SR-256 ckpt: {sr_256_ckpt}")

# ── Sampling settings ────────────────────────────────────────────────────────
NUM_SAMPLES = 16
BATCH_SIZE  = 8
NUM_STEPS   = 18       # Heun steps per stage (18 is EDM default)
SIGMA_MIN   = 0.002
SIGMA_MAX   = 80
RHO         = 7
SEED        = 42       # Set to None for random

if SEED is not None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

# Which stages to run
stages = ['base']
if sr_64_ckpt:
    stages.append('sr_64')
if sr_256_ckpt:
    stages.append('sr_256')
print(f"\nCascade stages: {' → '.join(stages)}")
print(f"Samples: {NUM_SAMPLES}, Steps/stage: {NUM_STEPS}")

# %% [markdown]
# ## 2. EDM Heun Sampler

# %%
@torch.no_grad()
def edm_heun_sampler(
    net, latents,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    verbose=True, **model_kwargs,
):
    """
    EDM 2nd-order Heun sampler (Algorithm 2 from Karras et al., 2022).

    Parameters
    ----------
    net : EDMPrecond or EDMSRPrecond
    latents : (B, C, H, W) — initial noise
    model_kwargs : extra kwargs passed to net() (e.g. low_res=...)

    Returns
    -------
    x : (B, C, H, H) — denoised samples in [-1, 1]
    """
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization (ρ-schedule)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Main loop
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Optionally increase noise (stochastic sampler)
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step
        denoised = net(
            x_hat.to(torch.float32), t_hat.to(torch.float32),
            **model_kwargs,
        ).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd-order correction (Heun)
        if i < num_steps - 1:
            denoised = net(
                x_next.to(torch.float32), t_next.to(torch.float32),
                **model_kwargs,
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        if verbose and (i % 5 == 0 or i == num_steps - 1):
            print(f"  step {i+1}/{num_steps}, σ={t_cur:.4f} → {t_next:.4f}")

    return x_next.to(torch.float32)

# %% [markdown]
# ## 3. Visualization Helpers

# %%
def tensor_to_numpy(t):
    """(B, C, H, W) tensor in [-1, 1] → (B, H, W, C) numpy in [0, 1]."""
    return ((t.float().cpu().clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()

def show_grid(images_np, title="", nrow=8, figsize=None):
    """
    Display a grid of images.

    Parameters
    ----------
    images_np : (N, H, W, C) numpy array in [0, 1]
    """
    n = len(images_np)
    ncol = min(nrow, n)
    nrows = math.ceil(n / ncol)
    h, w = images_np.shape[1], images_np.shape[2]

    if figsize is None:
        figsize = (ncol * 1.5, nrows * 1.5)

    fig, axes = plt.subplots(nrows, ncol, figsize=figsize)
    if nrows == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    for i in range(nrows):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            ax.axis('off')
            if idx < n:
                ax.imshow(images_np[idx].clip(0, 1))

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def show_cascade_comparison(results, nrow=8, max_show=16):
    """
    Show side-by-side comparison across cascade stages.

    Parameters
    ----------
    results : dict mapping stage name → (B, C, H, W) tensor in [-1, 1]
    """
    n_stages = len(results)
    stage_names = list(results.keys())
    n = min(max_show, results[stage_names[0]].shape[0])

    fig, axes = plt.subplots(n, n_stages, figsize=(n_stages * 2, n * 2))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, name in enumerate(stage_names):
        imgs = tensor_to_numpy(results[name][:n])
        axes[0, col].set_title(name, fontsize=11, fontweight='bold')
        for row in range(n):
            axes[row, col].imshow(imgs[row].clip(0, 1))
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Load Models

# %%
models = {}

# ── Base model ────────────────────────────────────────────────────────────────
if base_ckpt:
    print(f"Loading base model from:\n  {base_ckpt}")
    with open(base_ckpt, 'rb') as f:
        data = pickle.load(f)
    model_base = data['ema'].to(device).eval()
    models['base'] = model_base
    n_params = sum(p.numel() for p in model_base.parameters())
    print(f"  ✓ Base model: {n_params:,} params")
    print(f"    Resolution:  {model_base.img_resolution}×{model_base.img_resolution}")
    print(f"    Channels:    {model_base.img_channels}")
    del data
else:
    print("❌ No base checkpoint — cannot sample!")

# ── SR 64 model ───────────────────────────────────────────────────────────────
if sr_64_ckpt:
    print(f"\nLoading SR-64 model from:\n  {sr_64_ckpt}")
    with open(sr_64_ckpt, 'rb') as f:
        data = pickle.load(f)
    model_sr64 = data['ema'].to(device).eval()
    models['sr_64'] = model_sr64
    n_params = sum(p.numel() for p in model_sr64.parameters())
    print(f"  ✓ SR-64 model: {n_params:,} params")
    print(f"    Resolution:  {model_sr64.img_resolution}×{model_sr64.img_resolution}")
    del data

# ── SR 256 model ──────────────────────────────────────────────────────────────
if sr_256_ckpt:
    print(f"\nLoading SR-256 model from:\n  {sr_256_ckpt}")
    with open(sr_256_ckpt, 'rb') as f:
        data = pickle.load(f)
    model_sr256 = data['ema'].to(device).eval()
    models['sr_256'] = model_sr256
    n_params = sum(p.numel() for p in model_sr256.parameters())
    print(f"  ✓ SR-256 model: {n_params:,} params")
    del data

print(f"\nLoaded models: {list(models.keys())}")
torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Stage 1 — Base Model (32×32)

# %%
assert 'base' in models, "No base model loaded!"
model_base = models['base']

print(f"Generating {NUM_SAMPLES} base samples at {model_base.img_resolution}×{model_base.img_resolution}...")
print(f"  Heun steps: {NUM_STEPS}")
print(f"  σ range: [{SIGMA_MIN}, {SIGMA_MAX}]")

all_base = []
n_done = 0

while n_done < NUM_SAMPLES:
    bs = min(BATCH_SIZE, NUM_SAMPLES - n_done)
    print(f"\n  Batch {n_done+1}–{n_done+bs} / {NUM_SAMPLES}")

    latents = torch.randn(
        [bs, model_base.img_channels,
         model_base.img_resolution, model_base.img_resolution],
        device=device,
    )
    samples = edm_heun_sampler(
        model_base, latents,
        num_steps=NUM_STEPS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
    )
    all_base.append(samples.cpu())
    n_done += bs

base_32 = torch.cat(all_base, dim=0)  # (N, 3, 32, 32)
print(f"\n✓ Generated {base_32.shape[0]} base samples: {tuple(base_32.shape)}")
print(f"  Range: [{base_32.min():.3f}, {base_32.max():.3f}]")

# %%
# Display base samples
show_grid(
    tensor_to_numpy(base_32),
    title=f"Stage 1: Base Model — {model_base.img_resolution}×{model_base.img_resolution}",
    nrow=8,
)

# %% [markdown]
# ## 6. Stage 2 — Super-Resolution 32→64 (optional)

# %%
if 'sr_64' in models:
    model_sr64 = models['sr_64']
    print(f"Running SR cascade: {model_base.img_resolution}×{model_base.img_resolution} → "
          f"{model_sr64.img_resolution}×{model_sr64.img_resolution}")
    print(f"  Heun steps: {NUM_STEPS}")

    all_sr64 = []
    n_done = 0

    while n_done < NUM_SAMPLES:
        bs = min(BATCH_SIZE, NUM_SAMPLES - n_done)
        print(f"\n  Batch {n_done+1}–{n_done+bs} / {NUM_SAMPLES}")

        # Get the low_res conditioning from base samples
        low_res = base_32[n_done : n_done + bs].to(device)

        # Sample HR from noise, conditioned on low_res
        latents = torch.randn(
            [bs, model_sr64.img_channels,
             model_sr64.img_resolution, model_sr64.img_resolution],
            device=device,
        )
        samples = edm_heun_sampler(
            model_sr64, latents,
            num_steps=NUM_STEPS,
            sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
            low_res=low_res,  # ← conditioning input
        )
        all_sr64.append(samples.cpu())
        n_done += bs

    sr_64 = torch.cat(all_sr64, dim=0)  # (N, 3, 64, 64)
    print(f"\n✓ Generated {sr_64.shape[0]} SR-64 samples: {tuple(sr_64.shape)}")

    # Display SR-64 samples
    show_grid(
        tensor_to_numpy(sr_64),
        title=f"Stage 2: SR — {model_sr64.img_resolution}×{model_sr64.img_resolution}",
        nrow=8,
    )
else:
    sr_64 = None
    print("ℹ️  No SR-64 model loaded — skipping Stage 2.")

# %% [markdown]
# ## 7. Stage 3 — Super-Resolution 64→256 (optional)

# %%
if 'sr_256' in models and sr_64 is not None:
    model_sr256 = models['sr_256']
    print(f"Running SR cascade: 64×64 → {model_sr256.img_resolution}×{model_sr256.img_resolution}")
    print(f"  Heun steps: {NUM_STEPS}")

    all_sr256 = []
    n_done = 0

    while n_done < NUM_SAMPLES:
        bs = min(BATCH_SIZE, NUM_SAMPLES - n_done)
        print(f"\n  Batch {n_done+1}–{n_done+bs} / {NUM_SAMPLES}")

        low_res = sr_64[n_done : n_done + bs].to(device)

        latents = torch.randn(
            [bs, model_sr256.img_channels,
             model_sr256.img_resolution, model_sr256.img_resolution],
            device=device,
        )
        samples = edm_heun_sampler(
            model_sr256, latents,
            num_steps=NUM_STEPS,
            sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
            low_res=low_res,
        )
        all_sr256.append(samples.cpu())
        n_done += bs

    sr_256 = torch.cat(all_sr256, dim=0)
    print(f"\n✓ Generated {sr_256.shape[0]} SR-256 samples: {tuple(sr_256.shape)}")

    show_grid(
        tensor_to_numpy(sr_256),
        title=f"Stage 3: SR — {model_sr256.img_resolution}×{model_sr256.img_resolution}",
        nrow=8,
    )
else:
    sr_256 = None
    print("ℹ️  No SR-256 model loaded — skipping Stage 3.")

# %% [markdown]
# ## 8. Cascade Comparison — Side by Side

# %%
# Build results dict for comparison
cascade_results = {}
cascade_results['32×32\n(Base)'] = base_32

if sr_64 is not None:
    # Also include NN-upsampled for reference
    base_up_64 = F.interpolate(base_32, size=64, mode='nearest')
    cascade_results['64×64\n(NN ↑ only)'] = base_up_64
    cascade_results['64×64\n(SR)'] = sr_64

if sr_256 is not None:
    sr64_up_256 = F.interpolate(sr_64, size=256, mode='nearest')
    cascade_results['256×256\n(NN ↑ only)'] = sr64_up_256
    cascade_results['256×256\n(SR)'] = sr_256

# Show comparison
n_show = min(8, NUM_SAMPLES)
n_stages = len(cascade_results)
stage_names = list(cascade_results.keys())

fig, axes = plt.subplots(n_show, n_stages, figsize=(n_stages * 2.5, n_show * 2.5))
if n_show == 1:
    axes = axes[np.newaxis, :]

for col, name in enumerate(stage_names):
    imgs = tensor_to_numpy(cascade_results[name][:n_show])
    axes[0, col].set_title(name, fontsize=10, fontweight='bold')
    for row in range(n_show):
        axes[row, col].imshow(imgs[row].clip(0, 1),
                              interpolation='nearest' if '32' in name else 'bilinear')
        axes[row, col].axis('off')

fig.suptitle('EDM Cascaded Diffusion — Sample Comparison', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Save Samples

# %%
save_dir = os.path.join(PROJECT_ROOT, 'samples', 'edm_cascade_notebook')
os.makedirs(save_dir, exist_ok=True)

def save_samples(tensor, subdir, prefix='sample'):
    """Save individual images from (N, C, H, W) tensor."""
    out = os.path.join(save_dir, subdir)
    os.makedirs(out, exist_ok=True)
    for i in range(tensor.shape[0]):
        img = ((tensor[i].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        Image.fromarray(img).save(os.path.join(out, f'{prefix}_{i:04d}.png'))
    print(f"  Saved {tensor.shape[0]} images to {out}/")

# Save base
save_samples(base_32, '32x32')

# Save SR stages if available
if sr_64 is not None:
    save_samples(sr_64, '64x64')

if sr_256 is not None:
    save_samples(sr_256, '256x256')

print(f"\n✓ All samples saved to: {save_dir}")

# %% [markdown]
# ## 10. Training Loss Curve (Optional)
#
# Plot the training loss from `stats.jsonl` if available.

# %%
import json

def plot_training_loss(stats_path, title="Training Loss"):
    """Parse stats.jsonl and plot loss curve."""
    if not os.path.exists(stats_path):
        print(f"Not found: {stats_path}")
        return

    kimg_vals, loss_vals = [], []
    with open(stats_path) as f:
        for line in f:
            d = json.loads(line)
            if 'Progress/kimg' in d and 'Loss/loss' in d:
                kimg_vals.append(d['Progress/kimg']['mean'])
                loss_vals.append(d['Loss/loss']['mean'])

    if not kimg_vals:
        print("No data found in stats.jsonl")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(kimg_vals, loss_vals, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('kimg', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
    print(f"  Latest: kimg={kimg_vals[-1]:.0f}, loss={loss_vals[-1]:.4f}")

# Find base stats
base_stats = glob.glob(os.path.join(PROJECT_ROOT, 'checkpoints/edm_afhq_base_32/**/stats.jsonl'))
if base_stats:
    plot_training_loss(base_stats[-1], "EDM Base Model — Training Loss")

# Find SR stats
sr_stats = glob.glob(os.path.join(PROJECT_ROOT, 'checkpoints/edm_afhq_sr_64/**/stats.jsonl'))
if sr_stats:
    plot_training_loss(sr_stats[-1], "EDM SR Model — Training Loss")
