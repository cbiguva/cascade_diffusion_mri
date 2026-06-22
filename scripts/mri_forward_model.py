"""
mri_forward_model.py
--------------------
MRI forward operator for 2-channel (real + imaginary) reconstruction.

Implements:
    A  : image → undersampled k-space    (forward)
    A^T: undersampled k-space → image    (adjoint / zero-fill)

The 2-channel representation (Re, Im) is converted to/from complex
internally. All operations are differentiable.

Undersampling patterns:
    - 'equispaced': Equispaced lines with ACS (autocalibration signal)
    - 'random':     Random lines with ACS
    - 'radial':     Radial spokes (future)
"""

import torch
import torch.nn.functional as F
import numpy as np


class MRIForwardOp:
    """
    MRI forward operator: A = M · F

    Where F is the 2D FFT and M is the undersampling mask.

    Works with 2-channel (real, imaginary) representation:
        x : (B, 2, H, W) — image domain
        y : (B, 2, H, W) — k-space (masked)

    Parameters
    ----------
    img_size : int
        Image resolution (assumes square, e.g. 384).
    acceleration : int
        Acceleration factor (e.g. 4, 8, 16).
    pattern : str
        'equispaced' or 'random'.
    acs_fraction : float
        Fraction of center k-space to always acquire (ACS lines).
    seed : int or None
        Random seed for reproducible masks.
    device : torch.device
        Device for mask tensor.
    """

    def __init__(
        self,
        img_size=384,
        acceleration=4,
        pattern='equispaced',
        acs_fraction=0.08,
        seed=42,
        device=torch.device('cuda'),
    ):
        self.img_size = img_size
        self.acceleration = acceleration
        self.pattern = pattern
        self.device = device

        # Generate undersampling mask: (1, 1, H, W)
        self.mask = self._generate_mask(
            img_size, acceleration, pattern, acs_fraction, seed
        ).to(device)

        # Count acquired lines
        n_acquired = self.mask[0, 0, :, 0].sum().item()
        actual_accel = img_size / n_acquired
        print(f"[MRI] mask: {pattern}, {acceleration}× target, "
              f"{n_acquired:.0f}/{img_size} lines acquired "
              f"(actual {actual_accel:.1f}×)")

    def _generate_mask(self, size, accel, pattern, acs_frac, seed):
        """Generate a 1D Cartesian undersampling mask (applied along phase-encode dim)."""
        mask = torch.zeros(size, dtype=torch.float32)

        # ACS (center) lines — always acquired
        n_acs = max(int(size * acs_frac), 1)
        acs_start = size // 2 - n_acs // 2
        acs_end = acs_start + n_acs
        mask[acs_start:acs_end] = 1.0

        # Remaining lines
        n_target = max(size // accel, n_acs)
        n_remaining = n_target - n_acs

        if n_remaining > 0:
            non_acs_indices = [i for i in range(size) if mask[i] == 0]

            if pattern == 'equispaced':
                step = max(len(non_acs_indices) // n_remaining, 1)
                selected = non_acs_indices[::step][:n_remaining]
            elif pattern == 'random':
                rng = np.random.RandomState(seed)
                selected = rng.choice(non_acs_indices, size=n_remaining, replace=False)
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

            for idx in selected:
                mask[idx] = 1.0

        # Expand to (1, 1, H, W) — mask applies along rows (phase-encode)
        return mask.view(1, 1, size, 1).expand(1, 1, size, size).clone()

    def forward(self, x):
        """
        Forward operator: image → undersampled k-space.

        x : (B, 2, H, W) — 2-channel image (real, imaginary)
        Returns: (B, 2, H, W) — masked k-space
        """
        # 2ch → complex
        x_complex = torch.complex(x[:, 0], x[:, 1])  # (B, H, W)

        # 2D FFT
        kspace = torch.fft.fft2(x_complex, norm='ortho')  # (B, H, W)

        # Apply mask
        kspace_masked = kspace * self.mask[:, 0]  # broadcast (B, H, W)

        # complex → 2ch
        y = torch.stack([kspace_masked.real, kspace_masked.imag], dim=1)
        return y

    def adjoint(self, y):
        """
        Adjoint operator: undersampled k-space → zero-filled image.

        y : (B, 2, H, W) — masked k-space
        Returns: (B, 2, H, W) — zero-filled reconstruction
        """
        y_complex = torch.complex(y[:, 0], y[:, 1])
        x_complex = torch.fft.ifft2(y_complex, norm='ortho')
        return torch.stack([x_complex.real, x_complex.imag], dim=1)

    def data_consistency(self, x, y_measured):
        """
        Replace acquired k-space lines with measured data.

        x : (B, 2, H, W) — current image estimate
        y_measured : (B, 2, H, W) — measured (undersampled) k-space
        Returns: (B, 2, H, W) — data-consistent image
        """
        x_complex = torch.complex(x[:, 0], x[:, 1])
        kspace = torch.fft.fft2(x_complex, norm='ortho')

        y_complex = torch.complex(y_measured[:, 0], y_measured[:, 1])

        # Replace measured lines, keep predicted lines
        mask = self.mask[:, 0]  # (1, H, W)
        kspace_dc = kspace * (1 - mask) + y_complex * mask

        x_dc = torch.fft.ifft2(kspace_dc, norm='ortho')
        return torch.stack([x_dc.real, x_dc.imag], dim=1)

    def gradient(self, x, y_measured):
        """
        Gradient of data fidelity: ∇_x ½||Ax - y||²  =  A^T(Ax - y).

        x : (B, 2, H, W)
        y_measured : (B, 2, H, W)
        Returns: (B, 2, H, W)
        """
        residual = self.forward(x) - y_measured  # Ax - y
        return self.adjoint(residual)             # A^T(Ax - y)
