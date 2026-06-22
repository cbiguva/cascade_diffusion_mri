"""
energy_denoiser.py
------------------
Convert trained EDM denoisers into proper energy models via the
Jacobian-vector product (JVP) correction.

Theory
------
A standard EDM denoiser D(x; σ) gives the Tweedie score:
    s_tweedie = (D(x;σ) - x) / σ²

But this vector field is NOT conservative (not curl-free), so it
cannot be integrated into a scalar energy E(x).

The denoising energy is defined as:
    E(x; σ) = ½ ||x - D(x; σ)||²

Its gradient (the true energy score) requires a Jacobian correction:
    ∇_x E = (I - J_D^T) · (x - D(x;σ))
           = eps - J_D^T · eps

where J_D is the Jacobian of the denoiser and eps = x - D(x;σ).

The `giveScore` function computes -∇_x E, i.e. the score direction
that DESCENDS the energy landscape.
"""

import torch
import torch.nn as nn


class Denoiser(nn.Module):
    """
    Wraps a trained EDM model (EDMPrecond or EDMSRPrecond) to provide
    energy-based score computation via JVP correction.

    Usage:
        denoiser = Denoiser(edm_model)
        score = giveScore(x, denoiser, sigma, precondition=True)
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    # Expose key attributes from the inner model
    @property
    def img_resolution(self):
        return self.net.img_resolution

    @property
    def img_channels(self):
        return self.net.img_channels

    @property
    def sigma_min(self):
        return getattr(self.net, 'sigma_min', 0)

    @property
    def sigma_max(self):
        return getattr(self.net, 'sigma_max', float('inf'))

    def round_sigma(self, sigma):
        return self.net.round_sigma(sigma)

    def forward(self, inputs, sigma, class_labels=None, **kwargs):
        """Standard denoising forward pass (delegates to inner model)."""
        return self.net(inputs, sigma, class_labels=class_labels, **kwargs)

    def jvp(self, outputs, inputs, conditioning, vector, precondition):
        """
        Compute the Jacobian-vector product J_D^T · vector.

        Parameters
        ----------
        outputs : tensor
            Denoiser output D(x; σ).
        inputs : tensor
            Input x (must have requires_grad=True).
        conditioning : tensor
            Noise level σ (used for preconditioning).
        vector : tensor
            The vector to multiply by J_D^T (typically eps = x - D(x;σ)).
        precondition : bool
            If True, apply EDM preconditioning scaling.

        Returns
        -------
        J_D^T · vector : tensor
        """
        if precondition:
            vector = vector / conditioning
            grad_JT_eps = torch.autograd.grad(
                outputs=outputs, inputs=inputs, grad_outputs=vector,
                create_graph=True, only_inputs=True,
            )[0]
            grad_JT_eps = grad_JT_eps * conditioning
        else:
            grad_JT_eps = torch.autograd.grad(
                outputs=outputs, inputs=inputs, grad_outputs=vector,
                create_graph=True, only_inputs=True,
            )[0]
        return grad_JT_eps


def giveScore(x, net, sigma, precondition=True, class_labels=None, **kwargs):
    """
    Compute the JVP-corrected energy score.

    Returns the score = eps - J_D^T · eps, which is -∇_x E(x; σ)
    where E(x; σ) = ½ ||x - D(x; σ)||².

    Parameters
    ----------
    x : (B, C, H, W) tensor
        Input image (will be detached and re-attached with grad).
    net : Denoiser
        Wrapped EDM model.
    sigma : (B,) or scalar tensor
        Noise level.
    precondition : bool
        Apply EDM preconditioning (recommended True for EDM models).
    class_labels : optional
        Class conditioning (None for unconditional).
    **kwargs : dict
        Extra kwargs passed to net.forward() (e.g. low_res=... for SR).

    Returns
    -------
    score : (B, C, H, W) tensor
        The energy score direction.
    """
    x = x.detach().requires_grad_(True)

    # Ensure sigma has proper shape
    if sigma.dim() == 0:
        sigma = sigma.expand(x.shape[0])
    sigma = sigma.reshape(-1, 1, 1, 1).to(x.device)

    denoised = net(x, sigma.flatten(), class_labels=class_labels, **kwargs)
    eps = x - denoised

    # JVP correction: J_D^T · eps
    base = getattr(net, "module", net)  # unwrap DDP if present
    JVP = base.jvp(
        outputs=denoised, inputs=x,
        conditioning=sigma, vector=eps,
        precondition=precondition,
    )

    score = eps - JVP
    return score.detach()


def giveEnergy(x, net, sigma, class_labels=None, **kwargs):
    """
    Compute the denoising energy E(x; σ) = ½ ||x - D(x; σ)||².

    Parameters
    ----------
    x : (B, C, H, W) tensor
    net : Denoiser
    sigma : scalar or (B,) tensor

    Returns
    -------
    energy : (B,) tensor — scalar energy per sample
    """
    with torch.no_grad():
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        denoised = net(x, sigma, class_labels=class_labels, **kwargs)
        eps = x - denoised
        energy = 0.5 * (eps ** 2).sum(dim=(1, 2, 3))
    return energy
