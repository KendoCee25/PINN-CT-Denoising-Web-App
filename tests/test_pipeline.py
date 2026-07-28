"""Sanity tests for the Week 2 data pipeline and model.

Fast, CPU-only checks that each stage produces well-formed output. Run with:
    python -m pytest -q
"""

import numpy as np
import torch

from training.metrics import psnr, ssim
from training.noise import DOSE_I0, add_poisson_noise
from training.phantom import generate_phantom
from training.radon import default_angles, fbp, radon
from training.unet import UNet


def test_phantom_shape_and_range():
    p = generate_phantom(size=128, seed=0)
    assert p.shape == (128, 128)
    assert p.dtype == np.float32
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_phantom_is_deterministic_per_seed():
    a = generate_phantom(size=64, seed=7)
    b = generate_phantom(size=64, seed=7)
    assert np.array_equal(a, b)


def test_radon_shapes_and_differentiable():
    img = torch.rand(2, 1, 64, 64, requires_grad=True)
    ang = default_angles(60)
    sino = radon(img, ang)
    assert sino.shape == (2, 1, 60, 64)
    recon = fbp(sino, ang)
    assert recon.shape == (2, 1, 64, 64)
    # Gradients must flow (required for the PINN loss in Week 3).
    radon(img, ang).pow(2).mean().backward()
    assert img.grad is not None and img.grad.abs().sum() > 0


def test_noise_increases_as_dose_drops():
    clean = torch.from_numpy(generate_phantom(size=64, seed=1))[None, None]
    ang = default_angles(90)
    err = {}
    for dose in ["high", "medium", "low"]:
        noisy, _, _ = add_poisson_noise(clean, dose=dose, angles=ang)
        assert noisy.shape == clean.shape
        assert noisy.min() >= 0.0 and noisy.max() <= 1.0
        err[dose] = (noisy - clean).abs().mean().item()
    # Lower dose -> more noise -> larger deviation from clean.
    assert err["low"] > err["medium"] > err["high"]


def test_metrics_bounds():
    p = generate_phantom(size=64, seed=2)
    # Identical images: perfect SSIM, infinite/very-high PSNR.
    assert ssim(p, p) > 0.999
    assert psnr(p, p) > 60  # near-perfect
    noisy = np.clip(p + 0.1 * np.random.randn(*p.shape), 0, 1).astype(np.float32)
    assert 0.0 <= ssim(noisy, p) <= 1.0
    assert psnr(noisy, p) < psnr(p, p)


def test_unet_forward_shape():
    net = UNet()
    x = torch.rand(1, 1, 128, 128)
    assert net(x).shape == x.shape


def test_dose_levels_ordered():
    assert DOSE_I0["high"] > DOSE_I0["medium"] > DOSE_I0["low"]


def test_fbp_roundtrip_preserves_scale():
    """fbp(radon(x)) must return x's amplitude, not a scaled copy.

    noise.py reconstructs the noise field through fbp and trusts the result to
    be in image units, so a wrong normalisation silently changes how much noise
    the whole dataset gets. Compared mean-centred because the ramp filter zeroes
    the DC bin by construction.
    """
    img = torch.from_numpy(generate_phantom(size=128, seed=0))[None, None]
    ang = default_angles(180)
    recon = fbp(radon(img, ang), ang)

    a = (img - img.mean()).flatten()
    b = (recon - recon.mean()).flatten()
    gain = (a @ b) / (a @ a)
    # Exact value is 1.0; the shortfall is bilinear-interpolation smoothing.
    assert 0.80 < gain < 1.10, f"FBP round-trip gain {gain:.3f} is off-scale"


def test_noise_levels_are_realistic():
    """Each dose must land in the PSNR band real low-dose CT denoising lives in.

    Guards a bug that made the project's core task vacuous: noise so weak that
    the "low dose" input sat at 46 dB, and a trained U-Net scored worse than
    passing the input straight through. Bands are wide; they catch a broken
    calibration, not normal variation.
    """
    imgs = torch.from_numpy(
        np.stack([generate_phantom(size=128, seed=s) for s in range(4)])
    )[:, None]
    ang = default_angles(180)
    bands = {"high": (35.0, 41.0), "medium": (29.0, 35.0), "low": (25.0, 31.0)}

    for dose, (lo, hi) in bands.items():
        gen = torch.Generator().manual_seed(0)
        noisy, _, _ = add_poisson_noise(imgs, dose=dose, angles=ang, generator=gen)
        mean_psnr = float(
            np.mean([psnr(noisy[k, 0], imgs[k, 0]) for k in range(len(imgs))])
        )
        assert lo < mean_psnr < hi, f"{dose} dose input PSNR {mean_psnr:.2f} outside {lo}-{hi} dB"
