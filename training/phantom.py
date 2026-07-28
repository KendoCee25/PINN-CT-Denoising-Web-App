"""Shepp-Logan phantom generator with randomised lesion features.

Produces the clean "ground truth" CT images that the whole pipeline is built on.
Each phantom is the classic Shepp-Logan head phantom (a set of nested ellipses)
plus a handful of randomly placed/sized/contrasted extra ellipses that stand in
for lesions. Randomising these gives the diversity the U-Net/PINN need to learn
from (proposal §3, Week 2).

Output is a float32 image in [0, 1], shape (H, W). Pure NumPy so it has no GPU
dependency — generation is cheap and runs fine on the CPU-only dev machine.
"""

from __future__ import annotations

import numpy as np

# Classic Shepp-Logan ellipses (Toft variant), each row:
#   (intensity, semi-axis a, semi-axis b, centre x0, centre y0, rotation deg)
# Coordinates are in a [-1, 1] x [-1, 1] frame; intensities are additive.
_SHEPP_LOGAN = np.array(
    [
        (1.00, 0.6900, 0.9200, 0.0000, 0.0000, 0),
        (-0.80, 0.6624, 0.8740, 0.0000, -0.0184, 0),
        (-0.20, 0.1100, 0.3100, 0.2200, 0.0000, -18),
        (-0.20, 0.1600, 0.4100, -0.2200, 0.0000, 18),
        (0.10, 0.2100, 0.2500, 0.0000, 0.3500, 0),
        (0.10, 0.0460, 0.0460, 0.0000, 0.1000, 0),
        (0.10, 0.0460, 0.0460, 0.0000, -0.1000, 0),
        (0.10, 0.0460, 0.0230, -0.0800, -0.6050, 0),
        (0.10, 0.0230, 0.0230, 0.0000, -0.6060, 0),
        (0.10, 0.0230, 0.0460, 0.0600, -0.6050, 0),
    ],
    dtype=np.float64,
)


def _rasterise_ellipses(ellipses: np.ndarray, size: int) -> np.ndarray:
    """Additively rasterise a set of ellipses onto a `size`x`size` grid."""
    # Pixel-centre coordinate grid in the [-1, 1] frame.
    lin = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(lin, lin)

    img = np.zeros((size, size), dtype=np.float64)
    for intensity, a, b, x0, y0, phi in ellipses:
        phi_rad = np.deg2rad(phi)
        cos_p, sin_p = np.cos(phi_rad), np.sin(phi_rad)
        # Shift and rotate coordinates into the ellipse's local frame.
        x = xx - x0
        y = yy - y0
        x_rot = cos_p * x + sin_p * y
        y_rot = -sin_p * x + cos_p * y
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
        img[mask] += intensity
    return img


def _random_lesions(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate `n` random lesion ellipses (variable size, contrast, position)."""
    lesions = []
    for _ in range(n):
        intensity = rng.uniform(0.1, 0.4) * rng.choice([-1.0, 1.0])  # bright or dark
        a = rng.uniform(0.03, 0.12)          # semi-axis (size)
        b = a * rng.uniform(0.6, 1.4)        # slight anisotropy
        # Keep lesions inside the skull region.
        r = rng.uniform(0.0, 0.55)
        theta = rng.uniform(0.0, 2 * np.pi)
        x0, y0 = r * np.cos(theta), r * np.sin(theta)
        phi = rng.uniform(0.0, 180.0)
        lesions.append((intensity, a, b, x0, y0, phi))
    return np.array(lesions, dtype=np.float64)


def generate_phantom(
    size: int = 256,
    n_lesions: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one randomised Shepp-Logan phantom.

    Args:
        size: output image side length in pixels.
        n_lesions: number of random lesions; if None, a random 1-4 is used.
        seed: optional RNG seed for reproducibility.

    Returns:
        float32 array of shape (size, size), values normalised to [0, 1].
    """
    rng = np.random.default_rng(seed)
    if n_lesions is None:
        n_lesions = int(rng.integers(1, 5))

    ellipses = _SHEPP_LOGAN.copy()
    if n_lesions > 0:
        ellipses = np.vstack([ellipses, _random_lesions(rng, n_lesions)])

    img = _rasterise_ellipses(ellipses, size)

    # Normalise to [0, 1] (min-max); clip guards against tiny numerical spill.
    img = np.clip(img, 0.0, None)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    return img.astype(np.float32)


if __name__ == "__main__":
    # Quick visual smoke test: generate a small grid of phantoms and save a PNG.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        p = generate_phantom(size=256, seed=i)
        ax.imshow(p, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"phantom seed={i}")
        ax.axis("off")
    fig.suptitle("Shepp-Logan phantoms with randomised lesions")
    fig.tight_layout()
    out = "data/phantom_preview.png"
    fig.savefig(out, dpi=110)
    print(f"Saved preview to {out}")
