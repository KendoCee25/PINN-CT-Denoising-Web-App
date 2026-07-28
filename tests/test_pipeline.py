"""Sanity tests for the data pipeline, the model, and the PINN physics loss.

Fast, CPU-only checks that each stage produces well-formed output. Run with:
    python -m pytest -q
"""

import numpy as np
import pytest
import torch

from training.metrics import psnr, ssim
from training.noise import DOSE_I0, add_poisson_noise
from training.phantom import generate_phantom
from training.pinn import SinogramConsistencyLoss
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


# --------------------------------------------------------------------------
# Week 3: PINN sinogram-consistency loss
# --------------------------------------------------------------------------


def test_returned_sinograms_are_in_raw_units():
    """add_poisson_noise must return sinograms comparable to radon(image).

    The photon model works internally on a sinogram rescaled by a per-image
    factor. If that rescaling leaked into the return value, the physics loss
    would be comparing quantities ~30x apart and lambda would be absorbing the
    mismatch rather than weighting the term.
    """
    img = torch.from_numpy(generate_phantom(size=64, seed=3))[None, None]
    ang = default_angles(60)
    _, sino_clean, sino_noisy = add_poisson_noise(img, dose="low", angles=ang)

    # The clean sinogram returned must BE the forward projection.
    assert torch.allclose(sino_clean, radon(img, ang), rtol=1e-4, atol=1e-3)
    # And the noisy one must sit at the same scale, not 30x off.
    ratio = sino_noisy.abs().mean() / sino_clean.abs().mean()
    assert 0.5 < ratio < 2.0, f"noisy sinogram scale ratio {ratio:.3f} is off"


def test_physics_loss_gradients_reach_the_network():
    """The whole PINN idea depends on this: gradients must flow through radon."""
    net = UNet()
    loss_fn = SinogramConsistencyLoss(lam=0.1, n_angles_full=60, n_angles=30)
    img = torch.from_numpy(generate_phantom(size=64, seed=4))[None, None]
    noisy, _, sino_noisy = add_poisson_noise(img, dose="low", angles=default_angles(60))

    loss = loss_fn(net(noisy), {"sino": sino_noisy})
    loss.backward()

    grads = [p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient from the physics term"
    assert sum(grads) > 0, "physics term produced only zero gradients"


def test_physics_loss_is_minimised_by_the_truth():
    """Projecting the clean image must score better than projecting noise.

    A consistency term that did not prefer the true image would be worse than
    useless — it would actively pull the network away from the answer.
    """
    img = torch.from_numpy(generate_phantom(size=64, seed=5))[None, None]
    ang = default_angles(60)
    _, _, sino_noisy = add_poisson_noise(img, dose="low", angles=ang)
    loss_fn = SinogramConsistencyLoss(lam=1.0, n_angles_full=60)

    truth = loss_fn(img, {"sino": sino_noisy})
    wrong = loss_fn(torch.rand_like(img), {"sino": sino_noisy})
    assert truth < wrong, f"clean image scored {truth:.4f} vs random {wrong:.4f}"


def test_lambda_scales_the_term_linearly():
    img = torch.from_numpy(generate_phantom(size=64, seed=6))[None, None]
    ang = default_angles(60)
    _, _, sino_noisy = add_poisson_noise(img, dose="low", angles=ang)
    pred = torch.rand_like(img)

    l1 = SinogramConsistencyLoss(lam=0.1, n_angles_full=60)(pred, {"sino": sino_noisy})
    l2 = SinogramConsistencyLoss(lam=0.5, n_angles_full=60)(pred, {"sino": sino_noisy})
    assert torch.allclose(l2, l1 * 5.0, rtol=1e-4)


def test_lambda_zero_is_an_exact_no_op():
    """lambda=0 is the ablation's control arm, so it must contribute nothing."""
    img = torch.from_numpy(generate_phantom(size=64, seed=7))[None, None]
    ang = default_angles(60)
    _, _, sino_noisy = add_poisson_noise(img, dose="low", angles=ang)

    loss_fn = SinogramConsistencyLoss(lam=0.0, n_angles_full=60)
    assert loss_fn(torch.rand_like(img), {"sino": sino_noisy}).item() == 0.0


def test_loss_rejects_mismatched_angle_counts():
    """Guards against silently training against the wrong sinogram rows."""
    with pytest.raises(ValueError, match="must divide"):
        SinogramConsistencyLoss(lam=0.1, n_angles_full=180, n_angles=7)

    img = torch.from_numpy(generate_phantom(size=64, seed=8))[None, None]
    _, _, sino_noisy = add_poisson_noise(img, dose="low", angles=default_angles(60))
    with pytest.raises(ValueError, match="angles"):
        SinogramConsistencyLoss(lam=0.1, n_angles_full=180)(img, {"sino": sino_noisy})


def test_subsampled_angles_select_matching_rows():
    """A subsampled loss must compare against the same angles it projects at."""
    img = torch.from_numpy(generate_phantom(size=64, seed=9))[None, None]
    ang = default_angles(60)
    sino_clean = radon(img, ang)
    loss_fn = SinogramConsistencyLoss(lam=1.0, n_angles_full=60, n_angles=20)

    # Scoring the clean image against its own clean sinogram must be ~zero;
    # that only holds if projected angles and selected rows line up.
    assert loss_fn(img, {"sino": sino_clean}).item() < 1e-3
