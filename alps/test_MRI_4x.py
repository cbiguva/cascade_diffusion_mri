import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import torch
from dataclasses import dataclass, field
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
import os
import json
from torchvision.utils import save_image
import re

from sense_new import sense_v1

 
import dnnlib
import pickle
from operators import *
from algorithms_mri.Alps import ALPS_old_stepsize_MALA
from algorithms_mri.Alps import ALPS_old_stepsize
from algorithms_mri.Alps import mm_without_guidance
from algorithms_mri.Alps import Denoiser
from algorithms_mri.Dps import DPS
from algorithms_mri.Daps import Daps
from algorithms_mri.ista import pnp_ista
from algorithms_mri.PnPULA import PnPUla
############################################################################
#setting the device id 
gpu_id = input("Enter GPU id (e.g. 0, 1) or 'cpu': ")
try:
    if gpu_id.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(gpu_id)}")
except:
    print("Invalid input, defaulting to CPU")
    device = torch.device("cpu")

print(f"Using device: {device}")
############################################################################
#loading energy model 
with dnnlib.util.open_url("./models/teacher_model/Full_fastmri_score_model.pkl") as f:
    net0 = pickle.load(f)["ema"].to(device)

net = Denoiser(net0).to(device)

net_path = "/CBIG-Standard-ECE/Sahil/stud_teach_fastmri/finetuned_ckpts/ckpt_latest (3).pt"
state = torch.load(net_path, map_location=device)
state = state["model"] if isinstance(state, dict) and "model" in state else state
net.load_state_dict(state)
e = net.eval()
print('energy model loaded!')
############################################################################
#loading diffusion model 
network_pkl = "./models/teacher_model/Full_fastmri_score_model.pkl"
with dnnlib.util.open_url(network_pkl) as f:
    net_diffusion = pickle.load(f)['ema'].to(device)
d = net_diffusion.eval()
print('diffusion model loaded!')
############################################################################
#class_label
batch=1
class_labels = None
if getattr(net0, "label_dim", 0):
    row0 = torch.eye(net0.label_dim, device=device)[0]   # [C]
    class_labels = row0.unsqueeze(0).expand(batch, -1)          # [B,C]
    class_labels = class_labels.contiguous().float()
############################################################################

#loading parmeters
@dataclass
class Options_ULA:
    num_steps: int
    sigma_max: float
    sigma_min: float
    rho: float
    K: int
    beta: int
    inference_std: float
    step_size: float = None
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)

    

@dataclass
class Options_MALA:
    num_steps: int
    sigma_max: float
    sigma_min: float
    rho: float
    K: int
    beta:int
    inference_std: float
    step_size: float = None
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)
   

@dataclass
class Options_MAP:
    num_steps: int
    sigma_max: float
    sigma_min: float
    rho: float
    K: int
    inference_std: float
    L: int
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)
print('done')

@dataclass
class Options_DAPS:
    num_steps: int
    sigma_max: float
    sigma_min: float
    rho:float
    ode_steps:int
    ode_rho: int 
    ode_sigma_min: float 
    eta_0: float 
    delta: float 
    langevin_steps: int 
    inference_std: float
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)
    

@dataclass
class Options_DPS:
    num_steps: int
    sigma_max: float
    sigma_min: float
    rho: float
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)


@dataclass
class Options_pnpULA:
    num_steps: int
    noise_level: float
    clamp_min: int
    clamp_max: int
    step_size: float
    inference_std: float
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)
    


@dataclass
class Options_ista:
    num_steps: int
    sigma: float
    step_size: float
    inference_std: float
    class_labels: torch.Tensor = field(default_factory=lambda: class_labels)
    
   

with open("config/MRI_acc_4x1D/ula.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_ula = Options_ULA(**cfg)
# compute derived parameter
opts_ula.step_size = 1 / math.sqrt(opts_ula.K)

with open("config/MRI_acc_4x1D/mala.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_mala = Options_MALA(**cfg)
# compute derived parameter
opts_mala.step_size = 0.05 / math.sqrt(opts_mala.K)

with open("config/MRI_acc_4x1D/map.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_map = Options_MAP(**cfg)

with open("config/MRI_acc_4x1D/daps.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_daps = Options_DAPS(**cfg)

with open("config/MRI_acc_4x1D/dps.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_dps = Options_DPS(**cfg)

with open("config/MRI_acc_4x1D/pnpula.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_pnpULA = Options_pnpULA(**cfg)

with open("config/MRI_acc_4x1D/ista.yml", "r") as f:
    cfg = yaml.safe_load(f)
opts_ista = Options_ista(**cfg)
#############################################################################


Nsamples = 5
psnr_all_ula = []
ssim_all_ula = []
psnr_sample_ula =[]

psnr_all_mala = []
ssim_all_mala = []
psnr_sample_mala =[]


psnr_all_map = []
ssim_all_map = []
psnr_sample_map =[]

psnr_all_daps = []
ssim_all_daps = []
psnr_sample_daps = []

psnr_all_dps = []
ssim_all_dps = []
psnr_sample_dps = []

psnr_all_pnpula = []
ssim_all_pnpula = []
psnr_sample_pnpula = []

psnr_all_ista = []
ssim_all_ista = []
psnr_sample_ista = []

#############################################################################
#loading test data
processed_dir = "/CBIG-Project-ECE/Jyothi/subset_test_data"
pt_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]

def get_slice_idx(filename):
    match = re.search(r"slice(\d+)", filename)
    return int(match.group(1)) if match else 0  

pt_files = sorted(pt_files, key=get_slice_idx)

print(f"Found {len(pt_files)} total .pt files in {processed_dir}")
for f in pt_files[:10]:
    print("  ", f)
if len(pt_files) > 10:
    print(f"... ({len(pt_files)-10} more files)")

# Load all .pt files
loaded_data = []
for f in pt_files:
    path = os.path.join(processed_dir, f)
    data = torch.load(path, map_location="cpu")  # {"x": ..., "b_hat": ..., "csm": ...}
    loaded_data.append(data)

print(f"\n Loaded {len(loaded_data)} .pt files successfully.")
#############################################################################
mask_np = np.load('MRI_masks/acc4_c.npy').astype(np.complex64)
mask = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)  
tstMask = mask.to(device)
#############################################################################
os.makedirs(f"outputs/4x1D_mri_ula", exist_ok=True)
os.makedirs(f"outputs/4x1D_mri_mala", exist_ok=True)
os.makedirs(f"outputs/4x1D_mri_daps", exist_ok=True)
os.makedirs(f"outputs/4x1D_mri_dps", exist_ok=True)
os.makedirs(f"outputs/4x1D_mri_pnpula", exist_ok=True)
os.makedirs(f"outputs/4x1D_mri_map", exist_ok=True)
os.makedirs(f"outputs/4x1D_mri_ista", exist_ok=True)
os.makedirs("outputs/gt", exist_ok=True)


A= sense_v1(10)
for idx, data in enumerate(loaded_data):#
    
    if idx % 5 == 0:
        print(f"Processing {idx} images")

    x = data["x"].to(device, dtype=torch.complex64)
    b = data["b_hat"].to(device)
    tstCsm = data["csm"].to(device)
    mask_csm_temp= torch.sum(torch.abs(tstCsm)**2,dim=1)
    mask_csm = torch.unsqueeze((mask_csm_temp),0)
    b_hat = b * tstMask
    target= np.squeeze(torch.abs(x).detach().cpu().numpy()) 

    
    x_stack_tensor_ula = torch.empty(Nsamples, 1, 320, 320)
    x_stack_tensor_mala = torch.empty(Nsamples, 1, 320, 320)
    x_stack_tensor_daps = torch.empty(Nsamples, 1, 320, 320)
    x_stack_tensor_dps = torch.empty(Nsamples, 1, 320, 320)
    x_stack_tensor_pnpula = torch.empty(Nsamples, 1, 320, 320)
    


    psnr_sum_ula = 0
    psnr_sum_mala = 0
    psnr_sum_daps = 0
    psnr_sum_dps = 0
    psnr_sum_pnpula = 0
    for i in range(Nsamples):
        
        x_ula =ALPS_old_stepsize(opts_ula,A, net, b_hat,tstCsm,tstMask, isALPS=True, storeIntermediate=False)
        x_stack_tensor_ula[i]=(x_ula*mask_csm).abs().detach().cpu()
        x_recUla_np = x_stack_tensor_ula[i].abs().detach().cpu().squeeze().numpy()
        psnr_sum_ula = psnr_sum_ula + peak_signal_noise_ratio(target, x_recUla_np, data_range=target.max())

        

        x_mala = ALPS_old_stepsize_MALA(A, net, b_hat, tstCsm,tstMask, opts_mala, isALPS=True, storeIntermediate=False)
        x_stack_tensor_mala[i]=(x_mala*mask_csm).abs().detach().cpu()
        x_recMala_np = x_stack_tensor_mala[i].abs().detach().cpu().squeeze().numpy()
        psnr_sum_mala = psnr_sum_mala + peak_signal_noise_ratio(target, x_recMala_np, data_range=target.max())

        
        x_daps = Daps(A, net_diffusion, b_hat, tstCsm,tstMask, opts_daps, storeIntermediate=False)
        x_stack_tensor_daps[i]=(x_daps*mask_csm).abs().detach().cpu()
        x_recDaps_np = x_stack_tensor_daps[i].abs().detach().cpu().squeeze().numpy()
        psnr_sum_daps = psnr_sum_daps + peak_signal_noise_ratio(target, x_recDaps_np, data_range=target.max())

        
        x_dps = DPS(net_diffusion, torch.squeeze(torch.squeeze(b_hat,0),0),torch.squeeze(torch.squeeze(tstCsm,0),0),torch.squeeze(tstMask,0), opts_dps)
        x_stack_tensor_dps[i]=(x_dps*mask_csm).abs().detach().cpu()
        x_recDps_np = x_stack_tensor_dps[i].abs().detach().cpu().squeeze().numpy()
        psnr_sum_dps = psnr_sum_dps + peak_signal_noise_ratio(target, x_recDps_np, data_range=target.max())

        
        x_pnpula = PnPUla(net_diffusion, A,tstCsm,tstMask, b_hat,opts_pnpULA)
        x_stack_tensor_pnpula[i] = (x_pnpula*mask_csm).abs().detach().cpu()
        xrecPnPula_np = x_stack_tensor_pnpula[i].abs().detach().cpu().squeeze().numpy()
        psnr_sum_pnpula = psnr_sum_pnpula + peak_signal_noise_ratio(target,xrecPnPula_np,data_range=target.max())
        
        
    psnr_sample_ula.append(float(psnr_sum_ula/Nsamples))
    psnr_sample_mala.append(float(psnr_sum_mala/Nsamples))
    psnr_sample_daps.append(float(psnr_sum_daps/Nsamples))
    psnr_sample_dps.append(float(psnr_sum_dps/Nsamples))
    psnr_sample_pnpula.append(float(psnr_sum_pnpula/Nsamples))
    # Compute MMSE estimate
    xMMSE_ula = torch.mean(x_stack_tensor_ula, dim=0, keepdim=True)
    xMMSE_mala = torch.mean(x_stack_tensor_mala, dim=0, keepdim=True)
    xMMSE_daps = torch.mean(x_stack_tensor_daps, dim=0, keepdim=True)
    xMMSE_dps = torch.mean(x_stack_tensor_dps, dim=0, keepdim=True)
    xMMSE_pnpULA = torch.mean(x_stack_tensor_pnpula, dim=0, keepdim=True)
    #Compute MAP estimate
    
    x_map = mm_without_guidance(A, net, b_hat, tstCsm,tstMask,opts_map)
    x_map = x_map*mask_csm
    x_ista = pnp_ista(A,net_diffusion,b_hat,tstCsm,tstMask,opts_ista)
    x_ista = x_ista*mask_csm
    
    
    
   

    # Compute PSNR and SSIM
    xMMSEula_np = np.squeeze(xMMSE_ula.cpu().numpy())
    xMMSEmala_np = np.squeeze(xMMSE_mala.cpu().numpy())
    xMMSEDAPS_np = np.squeeze(xMMSE_daps.cpu().numpy())
    xMMSEDPS_np = np.squeeze(xMMSE_dps.cpu().numpy())
    xMMSEpnpULA_np = np.squeeze(xMMSE_pnpULA.cpu().numpy())
    map_np = np.squeeze(x_map.abs().detach().cpu().numpy())
    ista_np = np.squeeze(x_ista.abs().detach().cpu().numpy())

    psnr_val_ula = peak_signal_noise_ratio(target, xMMSEula_np, data_range=target.max())
    psnr_all_ula.append(float(psnr_val_ula))
    psnr_val_mala = peak_signal_noise_ratio(target, xMMSEmala_np, data_range=target.max())
    psnr_all_mala.append(float(psnr_val_mala))
    psnr_val_daps = peak_signal_noise_ratio(target, xMMSEDAPS_np,data_range=target.max())
    psnr_all_daps.append(float(psnr_val_daps))
    psnr_val_dps = peak_signal_noise_ratio(target, xMMSEDPS_np, data_range=target.max())
    psnr_all_dps.append(float(psnr_val_dps))
    psnr_val_pnpula = peak_signal_noise_ratio(target, xMMSEpnpULA_np , data_range=target.max())
    psnr_all_pnpula.append(float(psnr_val_pnpula))
    psnr_val_map = peak_signal_noise_ratio(target, map_np,data_range=target.max())
    psnr_all_map.append(float(psnr_val_map))
    psnr_val_ista = peak_signal_noise_ratio(target, ista_np,data_range=target.max())
    psnr_all_ista.append(float(psnr_val_ista))


    ssimf_ula = structural_similarity(target, xMMSEula_np, data_range=target.max())
    ssim_all_ula.append(float(ssimf_ula))
    ssimf_mala = structural_similarity(target, xMMSEmala_np, data_range=target.max())
    ssim_all_mala.append(float(ssimf_mala))
    ssimf_daps = structural_similarity(target, xMMSEDAPS_np, data_range=target.max())
    ssim_all_daps.append(float(ssimf_daps))
    ssimf_dps = structural_similarity(target, xMMSEDPS_np, data_range=target.max())
    ssim_all_dps.append(float(ssimf_dps) )
    ssimf_pnpula = structural_similarity(target, xMMSEpnpULA_np, data_range=target.max())
    ssim_all_pnpula.append(float(ssimf_pnpula) )
    ssimf_map = structural_similarity(target, map_np, data_range=target.max())
    ssim_all_map.append(float(ssimf_map))
    ssimf_ista = structural_similarity(target,  ista_np, data_range=target.max())
    ssim_all_ista.append(float(ssimf_ista))




results = {
    "MMSE/MAP_psnr": {
        "ula": psnr_all_ula,
        "mala": psnr_all_mala,
        "daps": psnr_all_daps,
        "dps": psnr_all_dps,
        "pnpula": psnr_all_pnpula,
        "map": psnr_all_map,
        "ista": psnr_all_ista,
    },
    "sample_avg_psnr":{
        "ula": psnr_sample_ula,
        "mala": psnr_sample_mala,
        "daps": psnr_sample_daps,
        "dps": psnr_sample_dps,
        "pnpula": psnr_sample_pnpula
        },
    "ssim": {
        "ula": ssim_all_ula,
        "mala": ssim_all_mala,
        "daps": ssim_all_daps,
        "dps": ssim_all_dps,
        "pnpula": ssim_all_pnpula,
        "map": ssim_all_map,
        "ista": ssim_all_ista,
    }
}
os.makedirs("results", exist_ok=True)
with open(f"results/mri4x1D.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved metrics to results/mri4x1D.json")

        