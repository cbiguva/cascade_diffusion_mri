"""
edm_sr_loss.py
--------------
EDM loss function adapted for super-resolution training.

Identical to the standard EDMLoss (Karras et al., 2022) except the
network forward call receives ``low_res`` as an additional keyword argument.

    loss = λ(σ) · ‖ D(x+σε; σ, low_res) − x ‖²

where λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
"""

import torch


class EDMSRLoss:
    """
    EDM denoising loss for super-resolution.

    Parameters
    ----------
    P_mean : float
        Mean of the log-normal noise distribution  (default −1.2).
    P_std : float
        Std of the log-normal noise distribution   (default  1.2).
    sigma_data : float
        Expected std of training data              (default  0.5).
    """

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, low_res, labels=None, augment_pipe=None):
        """
        Parameters
        ----------
        net : EDMSRPrecond
            The SR network with EDM preconditioning.
        images : (B, C, H, H)
            Clean HR target images in [-1, 1].
        low_res : (B, C, H_lr, W_lr)
            Low-resolution conditioning images (already augmented).
        labels : optional
            Class labels (unused for unconditional).
        augment_pipe : optional
            Data augmentation pipeline (applied to HR images only).
        """
        # Sample noise levels from log-normal distribution
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        # EDM weighting
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Optionally augment the HR target
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None
            else (images, None)
        )

        # Add noise to the HR target
        n = torch.randn_like(y) * sigma

        # Denoise with the SR model (receives low_res as conditioning)
        D_yn = net(
            y + n, sigma,
            low_res=low_res,
            class_labels=labels,
            augment_labels=augment_labels,
        )

        # Weighted MSE loss
        loss = weight * ((D_yn - y) ** 2)
        return loss
