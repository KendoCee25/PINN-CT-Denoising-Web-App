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
