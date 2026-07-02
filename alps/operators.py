"""
alps/operators.py
-----------------
Multi-coil (SENSE) MRI forward operator for cascaded ALPS reconstruction.

A single :class:`MRIOperator` serves both cascade stages.  It is fully
defined by a set of coil-sensitivity maps (``csm``) and an undersampling
``mask`` at its *own* working resolution, plus an FFT normalisation factor
``fft_scale``.  The caller is responsible for producing the maps/mask at the
right resolution (e.g. by average-pooling the 384×384 maps down to 96×96 for
Stage 1) — the operator itself is resolution-agnostic.

Forward model (per coil c):
    y_c = M · (s · F) · (S_c ⊙ x)  +  noise
where x is the (coil-combined) complex image the diffusion prior models, S_c
are the coil-sensitivity maps, F the centred orthonormal 2D FFT, M the
Cartesian undersampling mask and s = ``fft_scale``.

Cascade usage
-------------
    Stage 2 (384×384):  fft_scale = 1.0
        csm/mask are the native 384×384 maps and mask; the measurement is the
        masked full multi-coil k-space.

    Stage 1 (96×96):    fft_scale = 384 / 96 = 4.0
        csm = avg-pool(csm_384, 4), mask = 96×96 Cartesian mask.  With
        s = 4 the orthonormal 96×96 FFT of (avg-pooled image ⊙ avg-pooled
        coil maps) reproduces the *centre* 96×96 of the full 384×384 k-space
        (the centred orthonormal FFT of an average-pooled image equals the
        k-space centre scaled by 1/k per axis, i.e. 1/k in 2-D, so s = k
        restores the measured amplitude).  ``get_measurements`` therefore
        returns the masked centre crop of the real k-space.

Interface expected by alps/sampling.py:
    .forward(x)              — A x          (image → multi-coil k-space)
    .adjoint(y)              — Aᴴ y         (multi-coil k-space → image)
    .PreCondition(data, t)   — (AᴴA/eta² + I/t²)⁻¹    data
    .NoiseModulation(n, t)   — (AᴴA/eta² + I/t²)^{-1/2} n   (in distribution)
    .eta2                    — measurement noise variance
    .get_measurements(k)     — masked (centre-cropped) multi-coil k-space

Because the coil-sensitivity maps vary in space, AᴴA is NOT diagonal in
k-space, so PreCondition is solved with conjugate gradients (CG) and
NoiseModulation is realised by a perturbation sampler that reuses the same
CG solve.  The CG runs in float64 for numerical stability — with eta≈0.01 the
normal system spans a wide dynamic range and float32 round-off diverges.

All image tensors use the 2-channel (Real, Imaginary) convention:
    x : (B, 2, H, W)              — combined image domain
    y : (B, C, H, W)  complex     — multi-coil k-space (masked)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) → (B, H, W) complex"""
    return torch.complex(x[:, 0], x[:, 1])


def _to_2ch(z: torch.Tensor) -> torch.Tensor:
    """(B, H, W) complex → (B, 2, H, W)"""
    return torch.stack([z.real, z.imag], dim=1)


def _fft2(z: torch.Tensor) -> torch.Tensor:
    """Centred orthonormal 2D FFT (DC in the middle), matching the data + mask."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(z, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )


def _ifft2(z: torch.Tensor) -> torch.Tensor:
    """Centred orthonormal 2D inverse FFT (DC in the middle)."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(z, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )


def avg_pool_complex(z: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool a complex (B, C, H, W) tensor by factor k (real/imag separately)."""
    re = F.avg_pool2d(z.real, kernel_size=k, stride=k)
    im = F.avg_pool2d(z.imag, kernel_size=k, stride=k)
    return torch.complex(re, im)


def crop_kspace_center(kspace: torch.Tensor, size: int) -> torch.Tensor:
    """
    Crop the central ``size×size`` block of a DC-centred k-space tensor.

    The k-space here is ``fftshift``-ed (DC in the middle), so the central
    crop is exactly the low-frequency content.
    """
    H, W = kspace.shape[-2:]
    if H == size and W == size:
        return kspace
    h0 = (H - size) // 2
    w0 = (W - size) // 2
    return kspace[..., h0:h0 + size, w0:w0 + size]


def make_cartesian_mask(size: int, acceleration: int,
                        acs_fraction: float = 0.08,
                        seed: int = 42) -> torch.Tensor:
    """
    Build a 2D Cartesian (phase-encode) undersampling mask.

    Returns a (1, 1, size, size) float32 tensor with 1 = acquired, 0 = skipped.
    Lines run along the readout (last) axis; ACS lines (centre of k-space) are
    always acquired.
    """
    mask = torch.zeros(size, dtype=torch.float32)

    n_acs = max(int(size * acs_fraction), 1)
    c = size // 2
    mask[c - n_acs // 2: c + n_acs // 2] = 1.0

    n_target = max(size // acceleration, n_acs)
    n_extra = n_target - n_acs
    if n_extra > 0:
        avail = [i for i in range(size) if mask[i] == 0]
        rng = np.random.RandomState(seed)
        sel = rng.choice(avail, size=n_extra, replace=False)
        mask[sel] = 1.0

    return mask.view(1, 1, size, 1).expand(1, 1, size, size).clone()

from typing import Optional

def make_radial_mask(size: int, acceleration: int,
                     num_spokes: Optional[int] = None,
                     seed: int = 42) -> torch.Tensor:

    """
    Build a 2D radial (projection) undersampling mask.

    The mask is a set of straight spokes passing through the centre of a
    DC-centred k-space, at golden-angle increments (~111.25°) starting from a
    seeded random angle.  Because every spoke passes through the centre this
    matches the ``fftshift``-ed convention used by :class:`MRIOperator`, and the
    central crop of a radial mask is still radial — consistent with the
    cascade's :func:`crop_kspace_center` step (Stage-1 mask = centre crop of the
    Stage-2 mask).

    Acceleration is defined in the **pixel** (sampling-density) sense — spokes
    are added until the acquired fraction reaches ``1 / acceleration`` — so it is
    directly comparable to :func:`make_cartesian_mask`'s acceleration (which
    acquires ≈ size/acceleration full lines).  Pass ``num_spokes`` to specify the
    spoke count explicitly instead (overrides ``acceleration``).

    Parameters
    ----------
    size : int
        k-space side length.
    acceleration : int
        Target pixel-domain undersampling factor (ignored if ``num_spokes`` set).
    num_spokes : int, optional
        Explicit number of spokes.
    seed : int
        Seeds the random starting angle so different seeds give different
        (but equivalent) spoke sets.

    Returns
    -------
    (1, 1, size, size) float32 mask, 1 = acquired.
    """
    mask   = torch.zeros(size, size, dtype=torch.float32)
    cx = cy = size // 2
    radius = size / 2.0
    # Step the spoke in 0.5-px increments so the rasterised line has no gaps.
    t = torch.arange(-radius, radius, 0.5)

    rng    = np.random.RandomState(seed)
    theta  = float(rng.uniform(0, np.pi))             # random starting angle
    golden = float(np.pi * (np.sqrt(5) - 1) / 2)      # radial golden angle ~111.25°

    def _add_spoke(th: float) -> None:
        xs = (cx + t * float(np.cos(th))).round().long()
        ys = (cy + t * float(np.sin(th))).round().long()
        v  = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        mask[ys[v], xs[v]] = 1.0

    target     = (size * size) / float(acceleration)  # desired #acquired samples
    max_spokes = 4 * size                              # safety cap
    n = 0
    while True:
        _add_spoke(theta)
        theta += golden
        n += 1
        if num_spokes is not None:
            if n >= num_spokes:
                break
        elif mask.sum().item() >= target or n >= max_spokes:
            break

    acq = mask.sum().item()
    print(f"[make_radial_mask] {size}×{size}, {n} golden-angle spokes, "
          f"{acq:.0f}/{size * size} samples "
          f"(pixel accel {size * size / max(acq, 1):.1f}×)")

    return mask.view(1, 1, size, size).clone()


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-coil SENSE operator  (both cascade stages)
# ─────────────────────────────────────────────────────────────────────────────

class MRIOperator(nn.Module):
    """
    Multi-coil Cartesian SENSE forward operator.

    Parameters
    ----------
    csm : (1, C, H, W) complex
        Coil-sensitivity maps at this operator's working resolution.  For the
        low-resolution stage, average-pool the full-res maps *before* passing
        them in (see :func:`avg_pool_complex`).
    mask : (H, W) or (1, 1, H, W) float
        Cartesian undersampling mask (1 = acquired).  Built with
        :func:`make_cartesian_mask`.
    eta : float
        Measurement noise std.
    cg_iters : int
        CG iterations per PreCondition solve.
    cg_tol : float
        Early-stop relative residual for CG.
    fft_scale : float
        FFT normalisation factor s in A = M·(s·F)·S.  Use 1.0 for the native
        (full-resolution) stage and ``full_size / low_size`` for a pooled
        low-resolution stage so the forward model matches the measured
        k-space centre (see module docstring).
    device : torch.device
    """

    def __init__(self,
                 csm:       torch.Tensor,
                 mask:      torch.Tensor,
                 eta:       float = 0.01,
                 cg_iters:  int   = 10,
                 
                 cg_tol:    float = 1e-5,
                 fft_scale: float = 1.0,
                 device                  = torch.device('cuda')):
        super().__init__()

        csm = csm.to(torch.complex64)
        if mask.dim() == 2:
            mask = mask.view(1, 1, *mask.shape)
        mask = mask.to(torch.float32)

        assert csm.shape[-2:] == mask.shape[-2:], \
            f"csm {tuple(csm.shape)} and mask {tuple(mask.shape)} resolution mismatch"

        self.eta2      = float(eta) ** 2
        self.eta       = float(eta)
        self.cg_iters  = int(cg_iters)
        self.cg_tol    = float(cg_tol)
        self.fft_scale = float(fft_scale)
        self.device    = device
        self.n_coils   = csm.shape[1]
        self.img_size  = csm.shape[-1]

        self.register_buffer('csm',  csm)
        self.register_buffer('mask', mask)
        self.to(device)

        n_acq = mask[0, 0, :, 0].sum().item()
        print(f"[MRIOperator] {self.img_size}×{self.img_size} SENSE, "
              f"{self.n_coils} coils, fft_scale={self.fft_scale:g}, "
              f"{n_acq:.0f}/{self.img_size} lines acquired "
              f"(actual {self.img_size / max(n_acq, 1):.1f}×)")

    # ── linear operator ───────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,2,H,W) combined image → (B,C,H,W) complex masked multi-coil k-space."""
        xc   = _to_complex(x).unsqueeze(1)                  # (B,1,H,W)
        coil = self.csm.to(xc.dtype) * xc                   # (B,C,H,W)
        return self.mask * (self.fft_scale * _fft2(coil))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B,C,H,W) complex multi-coil k-space → (B,2,H,W) combined image."""
        coil = _ifft2(self.mask * y)                        # (B,C,H,W)
        comb = (self.csm.conj().to(coil.dtype) * coil).sum(dim=1)   # (B,H,W)
        return _to_2ch(self.fft_scale * comb)

    @staticmethod
    def combine(multicoil_kspace: torch.Tensor, csm: torch.Tensor) -> torch.Tensor:
        """SENSE coil-combine (unmasked) multi-coil k-space → (B,2,H,W) image."""
        coil = _ifft2(multicoil_kspace)
        comb = (csm.conj().to(coil.dtype) * coil).sum(dim=1)
        return _to_2ch(comb)

    def get_measurements(self, multicoil_kspace: torch.Tensor) -> torch.Tensor:
        """
        Masked multi-coil measurement for this operator's resolution.

        If the supplied k-space is higher resolution than the operator, it is
        centre-cropped first (the Stage-1 low-resolution measurement is the
        centre of the real k-space — consistent with the ``fft_scale``-
        normalised forward model).

        Parameters
        ----------
        multicoil_kspace : (B, C, H', W') complex   real acquisition

        Returns
        -------
        y : (B, C, H, W) complex   masked multi-coil k-space
        """
        k = crop_kspace_center(multicoil_kspace.to(self.mask.device), self.img_size)
        return self.mask * k

    # ── normal operator (rescaled by eta² so entries are O(1)) ─────────────────
    def _normal(self, u: torch.Tensor, t: float) -> torch.Tensor:
        """(AᴴA + (eta²/t²) I) u   — the eta²-rescaled normal operator."""
        return self.adjoint(self.forward(u)) + (self.eta2 / t ** 2) * u

    def _cg_solve(self, rhs: torch.Tensor, t: float, n_iter: int) -> torch.Tensor:
        """
        Solve (AᴴA + (eta²/t²) I) u = rhs  by conjugate gradients (float64).

        Returns u.  Note PreCondition multiplies this by eta² (see below).
        """
        dtype_in = rhs.dtype
        b = rhs.double()
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = (r * r).sum()
        r0 = r.norm().clamp(min=1e-30)
        for _ in range(n_iter):
            Ap    = self._normal(p, t)
            alpha = rs / (p * Ap).sum().clamp(min=1e-30)
            x     = x + alpha * p
            r     = r - alpha * Ap
            if (r.norm() / r0) < self.cg_tol:
                break
            rs_new = (r * r).sum()
            p      = r + (rs_new / rs.clamp(min=1e-30)) * p
            rs     = rs_new
        return x.to(dtype_in)

    def PreCondition(self, data: torch.Tensor, t: float) -> torch.Tensor:
        """
        Apply  B_t = (AᴴA/eta² + I/t²)⁻¹  to `data`.

        Using the identity B_t = eta² (AᴴA + (eta²/t²) I)⁻¹, we solve the
        well-scaled system with CG and multiply by eta².
        """
        u = self._cg_solve(data, t, self.cg_iters)
        return self.eta2 * u

    def NoiseModulation(self, n: torch.Tensor, t: float) -> torch.Tensor:
        """
        Draw a sample with covariance  B_t = (AᴴA/eta² + I/t²)⁻¹.

        Construction (exact for any linear A):
            z   = Aᴴ ε₁ / eta + ε₂ / t,   ε₁,ε₂ ~ N(0, I)
            ⇒  Cov(z) = AᴴA/eta² + I/t² = Pₜ
            n'  = B_t z = Pₜ⁻¹ z   ⇒  Cov(n') = B_t
        ε₂ is the image-space noise passed in as `n`; ε₁ is drawn here in
        multi-coil measurement space.  Reuses the same CG solve as
        PreCondition (the Pₜ⁻¹ apply).
        """
        B, _, H, W = n.shape
        eps1 = torch.randn(B, self.n_coils, H, W, device=n.device, dtype=n.dtype) \
             + 1j * torch.randn(B, self.n_coils, H, W, device=n.device, dtype=n.dtype)
        z = self.adjoint(eps1) / self.eta + n / t
        return self.PreCondition(z, t)

    
