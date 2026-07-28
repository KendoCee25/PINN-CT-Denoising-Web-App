"""Differentiable parallel-beam Radon transform (forward projection) and FBP.

This is the "physics" of a CT scanner expressed in PyTorch. The forward Radon
transform maps an image to its *sinogram* (the set of X-ray line-integral
projections at many angles); filtered back-projection (FBP) is the standard
inverse that reconstructs an image from a sinogram.

Both are implemented with `affine_grid` + `grid_sample` rotations, so the whole
transform is **differentiable** w.r.t. the image. That matters because:
  * Week 2 uses `radon` to turn clean phantoms into sinograms we then corrupt
    with Poisson noise, and `fbp` to reconstruct the noisy image.
  * Week 3 reuses the SAME differentiable `radon` inside the PINN's
    sinogram-consistency loss, where gradients must flow back to the network.

Convention: images are (B, 1, H, W); sinograms are (B, 1, n_angles, W).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _rotation_grid(angle_rad: float, size: int, device, dtype) -> torch.Tensor:
    """Build a sampling grid that rotates an image by `angle_rad`."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    theta = torch.tensor(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], device=device, dtype=dtype
    )
    theta = theta.unsqueeze(0)  # (1, 2, 3)
    return F.affine_grid(theta, (1, 1, size, size), align_corners=False)


def radon(image: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Forward parallel-beam projection.

    Args:
        image: (B, 1, H, W). Assumed square (H == W).
        angles: 1-D tensor of projection angles in **radians**.

    Returns:
        sinogram of shape (B, 1, n_angles, W): each row is the projection at one
        angle, formed by rotating the image and summing along columns.
    """
    b, c, h, w = image.shape
    assert c == 1 and h == w, "expects square single-channel images"
    device, dtype = image.device, image.dtype

    cols = []
    for angle in angles:
        grid = _rotation_grid(float(angle), h, device, dtype).expand(b, -1, -1, -1)
        rotated = F.grid_sample(image, grid, align_corners=False, mode="bilinear")
        # Sum along the vertical axis -> one projection (line integrals).
        proj = rotated.sum(dim=2)  # (B, 1, W)
        cols.append(proj)
    sino = torch.stack(cols, dim=2)  # (B, 1, n_angles, W)
    return sino


def _ramp_filter(width: int, device, dtype) -> torch.Tensor:
    """Ramp (Ram-Lak) filter in the frequency domain for FBP."""
    freqs = torch.fft.fftfreq(width, device=device).abs()  # (W,)
    return freqs.to(dtype)


def fbp(sinogram: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Filtered back-projection: reconstruct an image from a sinogram.

    Args:
        sinogram: (B, 1, n_angles, W).
        angles: 1-D tensor of the SAME angles (radians) used in `radon`.

    Returns:
        reconstructed image (B, 1, W, W).
    """
    b, c, n_angles, w = sinogram.shape
    device, dtype = sinogram.device, sinogram.dtype

    # 1) Filter each projection with the ramp filter (along detector axis).
    ramp = _ramp_filter(w, device, dtype)
    sino_fft = torch.fft.fft(sinogram, dim=3)
    sino_filtered = torch.fft.ifft(sino_fft * ramp, dim=3).real

    # 2) Back-project: smear each filtered projection across the image and rotate.
    recon = torch.zeros(b, 1, w, w, device=device, dtype=dtype)
    for i, angle in enumerate(angles):
        proj = sino_filtered[:, :, i, :]           # (B, 1, W)
        smear = proj.unsqueeze(2).expand(b, 1, w, w)  # broadcast across rows
        # Rotate the smeared band back to its acquisition angle and accumulate.
        grid = _rotation_grid(-float(angle), w, device, dtype).expand(b, -1, -1, -1)
        recon = recon + F.grid_sample(smear, grid, align_corners=False, mode="bilinear")

    # FBP normalisation for angles spanning a HALF turn [0, pi): the discrete
    # back-projection sum approximates (pi / n_angles) * integral over theta.
    # (A pi/(2*n_angles) factor would only be correct for a full [0, 2pi) sweep,
    # where every ray is measured twice.) Getting this right matters because
    # noise.py reconstructs the noise field through fbp and relies on its
    # amplitude being in true image units.
    recon = recon * (math.pi / n_angles)
    return recon


def default_angles(n_angles: int = 180, device=None, dtype=torch.float32) -> torch.Tensor:
    """Evenly spaced projection angles over [0, pi)."""
    return torch.linspace(0.0, math.pi, n_angles + 1, device=device, dtype=dtype)[:-1]


if __name__ == "__main__":
    # Sanity check: phantom -> sinogram -> FBP should roughly recover the phantom.
    import numpy as np

    from training.phantom import generate_phantom

    p = generate_phantom(size=128, seed=0)
    img = torch.from_numpy(p)[None, None]  # (1,1,128,128)
    ang = default_angles(180)

    sino = radon(img, ang)
    recon = fbp(sino, ang)

    # Report reconstruction error (should be small relative to signal).
    err = (recon - img).abs().mean().item()
    print(f"sinogram shape: {tuple(sino.shape)}")
    print(f"recon shape:    {tuple(recon.shape)}")
    print(f"mean abs recon error: {err:.4f}")

    # Confirm differentiability: gradient flows back to the image.
    img.requires_grad_(True)
    loss = radon(img, ang).pow(2).mean()
    loss.backward()
    print(f"grad flows to image: {img.grad is not None and img.grad.abs().sum() > 0}")
