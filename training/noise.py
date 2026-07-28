"""Low-dose CT noise simulation: Poisson noise in sinogram space + FBP.

Real CT noise does not live in the image — it arises in the *measurement*. As the
X-ray dose drops, fewer photons reach the detector, and photon counting is a
Poisson process. We therefore model noise the physically correct way (proposal
§3, Week 2):

    1. Forward-project the clean phantom to a sinogram of line integrals `p`.
    2. Convert to expected photon counts using a blank-scan intensity I0:
           N_expected = I0 * exp(-p)
       Lower dose  ->  smaller I0  ->  noisier.
    3. Sample actual counts  N ~ Poisson(N_expected).
    4. Convert back to a noisy line-integral sinogram: p_noisy = -log(N / I0).
    5. Reconstruct the noisy image with filtered back-projection (FBP).

Three dose levels (low / medium / high) follow the AAPM low-dose convention of
scaling the tube output (here, the blank-scan photon count I0).
"""

from __future__ import annotations

import torch

from training.radon import default_angles, fbp, radon

# Blank-scan photon counts per ray. Higher = more dose = less noise.
# Ratios follow the AAPM/Mayo low-dose challenge idea of dose fractions of a
# routine scan (high == routine, low ~= quarter dose).
DOSE_I0 = {
    "high": 1.0e5,
    "medium": 3.0e4,
    "low": 1.0e4,
}

# Peak line-integral attenuation the sinogram is scaled to before the photon
# model. Keeps exp(-p) numerically well-behaved (no under/overflow) and fixes a
# consistent contrast scale across phantoms.
P_MAX = 3.0


def add_poisson_noise(
    image: torch.Tensor,
    dose: str = "low",
    angles: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
):
    """Simulate a low-dose acquisition of a clean image.

    Args:
        image: (B, 1, H, W) clean phantom(s) in [0, 1].
        dose: one of "low", "medium", "high".
        angles: projection angles (radians); defaults to 180 over [0, pi).
        generator: optional torch RNG for reproducibility.

    Returns:
        (noisy_image, clean_sinogram, noisy_sinogram)
        noisy_image: (B, 1, H, W) clean image plus the reconstructed noise field,
        clipped to [0, 1] — matched in scale to the clean target.
        clean_sinogram / noisy_sinogram: (B, 1, n_angles, W) line integrals —
        the noisy one is what the PINN's consistency loss compares against.
    """
    if dose not in DOSE_I0:
        raise ValueError(f"dose must be one of {list(DOSE_I0)}, got {dose!r}")
    if angles is None:
        angles = default_angles(180, device=image.device, dtype=image.dtype)

    i0 = DOSE_I0[dose]

    # 1) Forward project, then scale to a fixed peak attenuation per image.
    sino = radon(image, angles)                              # (B,1,A,W)
    peak = sino.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    sino = sino / peak * P_MAX

    # 2-3) Photon counts -> Poisson sampling.
    expected_counts = i0 * torch.exp(-sino)
    noisy_counts = torch.poisson(expected_counts, generator=generator)
    noisy_counts = noisy_counts.clamp_min(1.0)               # avoid log(0)

    # 4) Back to a noisy line-integral sinogram.
    sino_noisy = -torch.log(noisy_counts / i0)

    # 5) Reconstruct ONLY the noise field and add it to the clean image. FBP has
    #    an arbitrary absolute scale, so reconstructing (noisy - clean) cancels
    #    the structure/DC offset and leaves realistic, spatially-correlated CT
    #    noise. Rescale back out of the P_MAX attenuation units into image units.
    noise_sino = sino_noisy - sino
    noise_field = fbp(noise_sino, angles) * (peak / P_MAX)
    noisy_image = (image + noise_field).clamp(0.0, 1.0)

    return noisy_image, sino, sino_noisy


if __name__ == "__main__":
    # Visual check: clean phantom vs noisy reconstructions at each dose level.
    import matplotlib.pyplot as plt

    from training.phantom import generate_phantom

    p = generate_phantom(size=128, seed=1)
    clean = torch.from_numpy(p)[None, None]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(clean[0, 0], cmap="gray")
    axes[0].set_title("clean phantom")
    axes[0].axis("off")

    for ax, dose in zip(axes[1:], ["high", "medium", "low"]):
        noisy, _, _ = add_poisson_noise(clean, dose=dose)
        ax.imshow(noisy[0, 0], cmap="gray")
        ax.set_title(f"{dose} dose")
        ax.axis("off")

    fig.suptitle("Poisson noise simulation across dose levels")
    fig.tight_layout()
    out = "data/noise_preview.png"
    fig.savefig(out, dpi=110)
    print(f"Saved preview to {out}")
