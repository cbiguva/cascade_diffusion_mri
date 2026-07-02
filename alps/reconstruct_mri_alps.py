"""
alps/reconstruct_mri_alps.py
----------------------------
Non-cascade MRI reconstruction using ALPS, MALA, DPS, DAPS, PnPULA, MAP, ISTA.

Uses the three new local files in the alps/ folder:
  - sense_new 4.py   -> sense_v1 SENSE operator
  - acc4_c 2.npy     -> 4x acceleration undersampling mask

Run from the project root (mri_cascaded_diffusion/):
    python alps/reconstruct_mri_alps.py

Or from inside alps/:
    python reconstruct_mri_alps.py
"""

import importlib.util
import os
import sys
import json
import math
import re

import numpy as np
import torch
from dataclasses import dataclass, field
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ──────────────────────────────────────────────────────────────────────────────
#  Path setup: make edm_repo and alps/ importable regardless of CWD
# ──────────────────────────────────────────────────────────────────────────────

_ALPS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../alps/
_ROOT     = os.path.dirname(_ALPS_DIR)                    # .../mri_cascaded_diffusion/

for _p in [_ROOT, _ALPS_DIR, os.path.join(_ROOT, 'edm_repo')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ──────────────────────────────────────────────────────────────────────────────
#  Import sense_v1 from "sense_new 4.py"  (filename has a space → use importlib)
# ──────────────────────────────────────────────────────────────────────────────

_sense_path = os.path.join(_ALPS_DIR, "sense_new 4.py")
_spec       = importlib.util.spec_from_file_location("sense_new_4", _sense_path)
_sense_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sense_mod)
sense_v1    = _sense_mod.sense_v1

# ──────────────────────────────────────────────────────────────────────────────
#  GPU selection
# ──────────────────────────────────────────────────────────────────────────────

gpu_id = input("Enter GPU id (e.g. 0, 1) or 'cpu': ")
try:
    device = torch.device("cpu") if gpu_id.lower() == "cpu" else torch.device(f"cuda:{int(gpu_id)}")
except Exception:
    print("Invalid input, defaulting to CPU")
    device = torch.device("cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────────────────────────────────────
#  Load energy model
# ──────────────────────────────────────────────────────────────────────────────

import dnnlib
import pickle
from alps import Denoiser   # alps/alps.py

_energy_pkl = os.path.join(_ROOT, "models", "teacher_model", "Full_fastmri_score_model.pkl")
with dnnlib.util.open_url(_energy_pkl) as f:
    net0 = pickle.load(f)["ema"].to(device)

net = Denoiser(net0).to(device)

net_path = "/CBIG-Standard-ECE/Sahil/stud_teach_fastmri/finetuned_ckpts/ckpt_latest (3).pt"
state    = torch.load(net_path, map_location=device)
state    = state["model"] if isinstance(state, dict) and "model" in state else state
net.load_state_dict(state)
net.eval()
print("Energy model loaded!")

# ──────────────────────────────────────────────────────────────────────────────
#  Load diffusion model
# ──────────────────────────────────────────────────────────────────────────────

with dnnlib.util.open_url(_energy_pkl) as f:
    net_diffusion = pickle.load(f)["ema"].to(device)
net_diffusion.eval()
print("Diffusion model loaded!")

# ──────────────────────────────────────────────────────────────────────────────
#  Class labels
# ──────────────────────────────────────────────────────────────────────────────

batch = 1
class_labels = None
if getattr(net0, "label_dim", 0):
    row0         = torch.eye(net0.label_dim, device=device)[0]
    class_labels = row0.unsqueeze(0).expand(batch, -1).contiguous().float()

# ──────────────────────────────────────────────────────────────────────────────
#  Algorithm option dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Options_ULA:
    num_steps:      int
    sigma_max:      float
    sigma_min:      float
    rho:            float
    K:              int
    beta:           int
    inference_std:  float
    step_size:      float = None
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

@dataclass
class Options_MALA:
    num_steps:      int
    sigma_max:      float
    sigma_min:      float
    rho:            float
    K:              int
    beta:           int
    inference_std:  float
    step_size:      float = None
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

@dataclass
class Options_MAP:
    num_steps:      int
    sigma_max:      float
    sigma_min:      float
    rho:            float
    K:              int
    inference_std:  float
    L:              int
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

@dataclass
class Options_DAPS:
    num_steps:      int
    sigma_max:      float
    sigma_min:      float
    rho:            float
    ode_steps:      int
    ode_rho:        int
    ode_sigma_min:  float
    eta_0:          float
    delta:          float
    langevin_steps: int
    inference_std:  float
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

@dataclass
class Options_DPS:
    num_steps:      int
    sigma_max:      float
    sigma_min:      float
    rho:            float
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

@dataclass
class Options_pnpULA:
    num_steps:      int
    noise_level:    float
    clamp_min:      int
    clamp_max:      int
    step_size:      float
    inference_std:  float
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

@dataclass
class Options_ista:
    num_steps:      int
    sigma:          float
    step_size:      float
    inference_std:  float
    class_labels:   torch.Tensor = field(default_factory=lambda: class_labels)

# ──────────────────────────────────────────────────────────────────────────────
#  Load YAML configs
# ──────────────────────────────────────────────────────────────────────────────

import yaml

_cfg_dir = os.path.join(_ROOT, "config", "MRI_acc_4x1D")

def _load_cfg(name, cls):
    with open(os.path.join(_cfg_dir, name)) as f:
        cfg = yaml.safe_load(f)
    return cls(**cfg)

opts_ula    = _load_cfg("ula.yml",    Options_ULA)
opts_ula.step_size = 1 / math.sqrt(opts_ula.K)

opts_mala   = _load_cfg("mala.yml",   Options_MALA)
opts_mala.step_size = 0.05 / math.sqrt(opts_mala.K)

opts_map    = _load_cfg("map.yml",    Options_MAP)
opts_daps   = _load_cfg("daps.yml",   Options_DAPS)
opts_dps    = _load_cfg("dps.yml",    Options_DPS)
opts_pnpULA = _load_cfg("pnpula.yml", Options_pnpULA)
opts_ista   = _load_cfg("ista.yml",   Options_ista)

print("Configs loaded!")

# ──────────────────────────────────────────────────────────────────────────────
#  Import algorithm implementations (non-cascade)
# ──────────────────────────────────────────────────────────────────────────────

from alps import ALPS_old_stepsize, ALPS_old_stepsize_MALA, mm_without_guidance
from algorithms_mri.Dps    import DPS
from algorithms_mri.Daps   import Daps
from algorithms_mri.ista   import pnp_ista
from algorithms_mri.PnPULA import PnPUla

# ──────────────────────────────────────────────────────────────────────────────
#  Load test data
# ──────────────────────────────────────────────────────────────────────────────

processed_dir = "/CBIG-Project-ECE/Jyothi/subset_test_data"
pt_files      = sorted(
    [f for f in os.listdir(processed_dir) if f.endswith(".pt")],
    key=lambda fn: int(re.search(r"slice(\d+)", fn).group(1))
                   if re.search(r"slice(\d+)", fn) else 0,
)
print(f"Found {len(pt_files)} .pt files in {processed_dir}")
for f in pt_files[:10]:
    print("  ", f)
if len(pt_files) > 10:
    print(f"  ... ({len(pt_files)-10} more)")

loaded_data = [torch.load(os.path.join(processed_dir, f), map_location="cpu") for f in pt_files]
print(f"Loaded {len(loaded_data)} .pt files.")

# ──────────────────────────────────────────────────────────────────────────────
#  Load undersampling mask  ← "acc4_c 2.npy"  (new file in alps/)
# ──────────────────────────────────────────────────────────────────────────────

_mask_path = os.path.join(_ALPS_DIR, "acc4_c 2.npy")
mask_np    = np.load(_mask_path).astype(np.complex64)
mask       = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
tstMask    = mask.to(device)
print(f"Mask loaded from: {_mask_path}  shape={tuple(tstMask.shape)}")

# ──────────────────────────────────────────────────────────────────────────────
#  Output directories
# ──────────────────────────────────────────────────────────────────────────────

for tag in ["4x1D_mri_ula", "4x1D_mri_mala", "4x1D_mri_daps",
            "4x1D_mri_dps", "4x1D_mri_pnpula", "4x1D_mri_map",
            "4x1D_mri_ista", "gt"]:
    os.makedirs(os.path.join(_ROOT, "outputs", tag), exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Reconstruction loop  (non-cascade)
# ──────────────────────────────────────────────────────────────────────────────

Nsamples = 5
A = sense_v1(10)   # sense_v1 from "sense_new 4.py", 10 CG iterations

psnr_all  = {k: [] for k in ["ula", "mala", "daps", "dps", "pnpula", "map", "ista"]}
ssim_all  = {k: [] for k in ["ula", "mala", "daps", "dps", "pnpula", "map", "ista"]}
psnr_samp = {k: [] for k in ["ula", "mala", "daps", "dps", "pnpula"]}

for idx, data in enumerate(loaded_data):

    if idx % 5 == 0:
        print(f"Processing slice {idx} / {len(loaded_data)}")

    x        = data["x"].to(device, dtype=torch.complex64)
    b        = data["b_hat"].to(device)
    tstCsm   = data["csm"].to(device)
    mask_csm = torch.unsqueeze(torch.sum(torch.abs(tstCsm) ** 2, dim=1), 0)
    b_hat    = b * tstMask
    target   = np.squeeze(torch.abs(x).detach().cpu().numpy())

    stacks = {k: torch.empty(Nsamples, 1, 320, 320) for k in ["ula", "mala", "daps", "dps", "pnpula"]}
    psum   = {k: 0.0 for k in stacks}

    # ── Posterior samples ────────────────────────────────────────────────────
    for i in range(Nsamples):

        x_ula = ALPS_old_stepsize(opts_ula, A, net, b_hat, tstCsm, tstMask,
                                   isALPS=True, storeIntermediate=False)
        stacks["ula"][i] = (x_ula * mask_csm).abs().detach().cpu()
        psum["ula"] += peak_signal_noise_ratio(
            target, stacks["ula"][i].abs().squeeze().numpy(), data_range=target.max())

        x_mala = ALPS_old_stepsize_MALA(A, net, b_hat, tstCsm, tstMask, opts_mala,
                                         isALPS=True, storeIntermediate=False)
        stacks["mala"][i] = (x_mala * mask_csm).abs().detach().cpu()
        psum["mala"] += peak_signal_noise_ratio(
            target, stacks["mala"][i].abs().squeeze().numpy(), data_range=target.max())

        x_daps = Daps(A, net_diffusion, b_hat, tstCsm, tstMask, opts_daps,
                       storeIntermediate=False)
        stacks["daps"][i] = (x_daps * mask_csm).abs().detach().cpu()
        psum["daps"] += peak_signal_noise_ratio(
            target, stacks["daps"][i].abs().squeeze().numpy(), data_range=target.max())

        x_dps = DPS(net_diffusion,
                    torch.squeeze(b_hat, 0).squeeze(0),
                    torch.squeeze(tstCsm, 0).squeeze(0),
                    tstMask.squeeze(0), opts_dps)
        stacks["dps"][i] = (x_dps * mask_csm).abs().detach().cpu()
        psum["dps"] += peak_signal_noise_ratio(
            target, stacks["dps"][i].abs().squeeze().numpy(), data_range=target.max())

        x_pnpula = PnPUla(net_diffusion, A, tstCsm, tstMask, b_hat, opts_pnpULA)
        stacks["pnpula"][i] = (x_pnpula * mask_csm).abs().detach().cpu()
        psum["pnpula"] += peak_signal_noise_ratio(
            target, stacks["pnpula"][i].abs().squeeze().numpy(), data_range=target.max())

    for k in psum:
        psnr_samp[k].append(float(psum[k] / Nsamples))

    # ── MMSE (posterior mean) ─────────────────────────────────────────────────
    mmse = {k: torch.mean(stacks[k], dim=0, keepdim=True) for k in stacks}

    # ── MAP / ISTA ────────────────────────────────────────────────────────────
    x_map  = mm_without_guidance(A, net, b_hat, tstCsm, tstMask, opts_map) * mask_csm
    x_ista = pnp_ista(A, net_diffusion, b_hat, tstCsm, tstMask, opts_ista) * mask_csm

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _np(t):
        return np.squeeze(t.abs().detach().cpu().numpy())

    refs = {
        "ula":    _np(mmse["ula"]),
        "mala":   _np(mmse["mala"]),
        "daps":   _np(mmse["daps"]),
        "dps":    _np(mmse["dps"]),
        "pnpula": _np(mmse["pnpula"]),
        "map":    _np(x_map),
        "ista":   _np(x_ista),
    }
    for k, img in refs.items():
        psnr_all[k].append(float(peak_signal_noise_ratio(target, img, data_range=target.max())))
        ssim_all[k].append(float(structural_similarity(target, img, data_range=target.max())))

# ──────────────────────────────────────────────────────────────────────────────
#  Save results
# ──────────────────────────────────────────────────────────────────────────────

results = {
    "MMSE/MAP_psnr":   psnr_all,
    "sample_avg_psnr": psnr_samp,
    "ssim":            ssim_all,
}

os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
out_json = os.path.join(_ROOT, "results", "mri4x1D_alps.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved metrics -> {out_json}")
