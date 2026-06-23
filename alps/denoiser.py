"""
alps/denoiser.py
----------------
Denoiser wrappers that bridge the EDM score networks to the ALPS
sampling interface.

ALPS expects a denoiser object with two methods:
    .forward(inputs, sigma)                    → psi(x, sigma)  (x0-prediction)
    .vjp(outputs, inputs, conditioning,
         vector, precondition)                 → J^T v  (vector-Jacobian product)

Two wrappers are provided:

    BaseDenoiser   — wraps EDMPrecond (Stage 1, 96×96, 2-channel)
    SRDenoiser     — wraps EDMSRPrecond (Stage 2, 384×384, 2-channel)
                     The low-res conditioning image is fixed after Stage 1
                     and injected on every forward call.
"""

import os
import sys
import torch
import torch.nn as nn

# ── make edm_repo and scripts/ importable ─────────────────────────────────────
_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EDM     = os.path.join(_ROOT, 'edm_repo')
_SCRIPTS = os.path.join(_ROOT, 'scripts')
for _p in [_EDM, _SCRIPTS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─────────────────────────────────────────────────────────────────────────────
#  Base denoiser  (Stage 1 — 96×96)
# ─────────────────────────────────────────────────────────────────────────────

class BaseDenoiser(nn.Module):
    """
    Wraps an EDMPrecond (or any EDM-style net) as an ALPS denoiser.

    The EDM network signature is:
        net(x, sigma, class_labels=None)  →  x0_prediction

    Parameters
    ----------
    net : EDMPrecond
        Loaded EMA checkpoint, already on the correct device and in eval mode.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute psi(x, sigma)  — the x0-prediction of the EDM network.

        Parameters
        ----------
        inputs : (B, 2, 96, 96)
        sigma  : (B,) or scalar tensor

        Returns
        -------
        (B, 2, 96, 96) denoised estimate
        """
        return self.net(inputs, sigma, None)   # no class labels

    def vjp(self,
            outputs: torch.Tensor,
            inputs:  torch.Tensor,
            conditioning: torch.Tensor,
            vector: torch.Tensor,
            precondition: bool) -> torch.Tensor:
        """
        Compute the vector-Jacobian product  J_{psi}^T  v.

        Used by giveScore to compute the gradient of
            E(x, sigma) = 0.5 || x - psi(x, sigma) ||^2

        Parameters
        ----------
        outputs      : psi(x, sigma)
        inputs       : x  (requires_grad=True)
        conditioning : sigma
        vector       : x - psi(x, sigma)
        precondition : if True, divide vector by sigma before VJP and
                       multiply result by sigma after (matches ALPS paper)
        """
        if precondition:
            vector = vector / conditioning
        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=vector,
            create_graph=True,
            only_inputs=True,
        )[0]
        if precondition:
            grad = grad * conditioning
        return grad


# ─────────────────────────────────────────────────────────────────────────────
#  SR denoiser  (Stage 2 — 384×384)
# ─────────────────────────────────────────────────────────────────────────────

class SRDenoiser(nn.Module):
    """
    Wraps an EDMSRPrecond as an ALPS denoiser with fixed low-res conditioning.

    The SR network signature is:
        net(x, sigma, low_res=lr_image)  →  x0_prediction

    The low-res image (output of Stage 1) is fixed for the entire Stage 2
    sampling run.  ALPS never backpropagates through this conditioning —
    giveScore only differentiates with respect to the HR input x_384.

    Parameters
    ----------
    net         : EDMSRPrecond
        Loaded EMA checkpoint, eval mode.
    x96_fixed   : (B, 2, 96, 96)
        Low-resolution reconstruction from Stage 1.  Will be detached
        and kept fixed throughout Stage 2 sampling.
    """

    def __init__(self, net: nn.Module, x96_fixed: torch.Tensor):
        super().__init__()
        self.net      = net
        # detach so no gradients ever flow back into Stage 1
        self.low_res  = x96_fixed.detach()

    def forward(self, inputs: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute psi(x_384, sigma | x_96).

        Parameters
        ----------
        inputs : (B, 2, 384, 384)  — noisy HR image
        sigma  : (B,) or scalar tensor

        Returns
        -------
        (B, 2, 384, 384) denoised HR estimate
        """
        return self.net(inputs, sigma, low_res=self.low_res)

    def vjp(self,
            outputs: torch.Tensor,
            inputs:  torch.Tensor,
            conditioning: torch.Tensor,
            vector: torch.Tensor,
            precondition: bool) -> torch.Tensor:
        """
        Vector-Jacobian product w.r.t. the HR input only.

        The gradient does NOT flow through self.low_res because it is
        detached.  This is correct — we are sampling from
        p(x_384 | x_96, y) treating x_96 as fixed.
        """
        if precondition:
            vector = vector / conditioning
        grad = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=vector,
            create_graph=True,
            only_inputs=True,
        )[0]
        if precondition:
            grad = grad * conditioning
        return grad
