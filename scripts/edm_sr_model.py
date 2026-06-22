"""
edm_sr_model.py
---------------
EDM-preconditioned super-resolution model for cascaded diffusion.

Implements the same 6-channel concatenation approach as the CDM paper
(Ho et al., 2022):  the SongUNet receives  [c_in · x_noisy | low_res_up]
along the channel dimension (6 channels in, 3 channels out).

The preconditioning follows EDM (Karras et al., 2022):
    D(x; σ) = c_skip(σ) · x  +  c_out(σ) · F_θ(c_in(σ) · x; c_noise(σ))

Usage:
    model = EDMSRPrecond(img_resolution=64, img_channels=3,
                         model_channels=128, channel_mult=[1,2,2,2])
    D_x = model(x_noisy, sigma, low_res=low_res_32x32)
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F

# ── Make EDM importable ──────────────────────────────────────────────────────
# Guard against __file__ not existing when EDM's persistence.py reconstructs
# this module via exec() during unpickling.
try:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _EDM_PATH = os.path.join(_REPO_ROOT, 'edm_repo')
    if _EDM_PATH not in sys.path:
        sys.path.insert(0, _EDM_PATH)
except NameError:
    pass  # __file__ undefined inside exec() — EDM modules already on sys.path

from torch_utils import persistence
from training.networks import SongUNet


# ──────────────────────────────────────────────────────────────────────────────
#  EDM SR Preconditioner
# ──────────────────────────────────────────────────────────────────────────────

@persistence.persistent_class
class EDMSRPrecond(torch.nn.Module):
    """
    EDM preconditioning wrapper for super-resolution.

    The inner SongUNet takes a 6-channel input:
        [c_in(σ) · x_noisy  |  low_res_upsampled]     (B, 6, H, H)
    and produces a 3-channel output Fθ.

    The final denoised estimate is:
        D(x; σ) = c_skip · x  +  c_out · Fθ

    Conditioning augmentation (CDM §4.2) is NOT applied here — it is
    handled by the dataloader so that each training example is already
    augmented before reaching the model.

    Parameters
    ----------
    img_resolution : int
        Target (HR) spatial resolution (e.g. 64).
    img_channels : int
        Number of image channels (3 for RGB).
    sigma_data : float
        Expected standard deviation of the training data (EDM default 0.5).
    model_channels : int
        Base channel count for the SongUNet.
    channel_mult : list[int]
        Per-resolution channel multipliers.
    num_blocks : int
        Residual blocks per resolution level.
    attn_resolutions : list[int]
        Resolutions at which self-attention is applied.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        img_resolution,                     # e.g. 64
        img_channels        = 3,
        label_dim           = 0,
        use_fp16            = False,
        sigma_min           = 0,
        sigma_max           = float('inf'),
        sigma_data          = 0.5,
        # SongUNet kwargs
        model_channels      = 128,
        channel_mult        = [1, 2, 2, 2],
        channel_mult_emb    = 4,
        num_blocks          = 4,
        attn_resolutions    = [16],
        dropout             = 0.10,
        label_dropout       = 0,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels   = img_channels
        self.label_dim      = label_dim
        self.use_fp16       = use_fp16
        self.sigma_min      = sigma_min
        self.sigma_max      = sigma_max
        self.sigma_data     = sigma_data

        # SongUNet: 6-channel input  (3 noisy + 3 condition) → 3 output
        self.model = SongUNet(
            img_resolution      = img_resolution,
            in_channels         = img_channels * 2,     # 6
            out_channels        = img_channels,          # 3
            label_dim           = label_dim,
            augment_dim         = 0,
            model_channels      = model_channels,
            channel_mult        = channel_mult,
            channel_mult_emb    = channel_mult_emb,
            num_blocks          = num_blocks,
            attn_resolutions    = attn_resolutions,
            dropout             = dropout,
            label_dropout       = label_dropout,
            # DDPM++ config
            embedding_type      = 'positional',
            channel_mult_noise  = 1,
            encoder_type        = 'standard',
            decoder_type        = 'standard',
            resample_filter     = [1, 1],
        )

    def forward(self, x, sigma, low_res=None, class_labels=None,
                force_fp32=False, **model_kwargs):
        """
        Parameters
        ----------
        x : (B, 3, H, H)
            Noisy HR image  (x = x_clean + σ · ε).
        sigma : (B,) or (B, 1, 1, 1)
            Noise level.
        low_res : (B, 3, H_lr, W_lr)
            Low-resolution conditioning image (e.g. 32×32).
            Will be bilinearly upsampled to match x's spatial size.
        class_labels : optional
            Class labels for conditional generation (unused for AFHQ).

        Returns
        -------
        D_x : (B, 3, H, H)
            Denoised estimate.
        """
        assert low_res is not None, "SR model requires low_res conditioning input"

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda')
            else torch.float32
        )

        # ── EDM preconditioning coefficients ──────────────────────────────
        c_skip  = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out   = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # ── Upsample low_res to match target resolution (bilinear) ────────
        low_res_up = F.interpolate(
            low_res.to(torch.float32),
            size=(x.shape[2], x.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        # ── Concatenate along channel dimension: [c_in·x | low_res] ──────
        #    c_in scales the noisy input;  low_res is passed through unscaled
        #    (the network learns how to use the condition)
        model_input = torch.cat([c_in * x, low_res_up], dim=1)  # (B, 6, H, H)

        # ── Forward through SongUNet ──────────────────────────────────────
        F_x = self.model(
            model_input.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype

        # ── Denoised output ───────────────────────────────────────────────
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
