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
# `high` stands for a routine-dose scan; `medium` and `low` are reduced tube
# outputs (~27% and ~10% of routine), following the AAPM/Mayo low-dose challenge
# idea of expressing dose as a fraction of a routine acquisition.
#
# These counts are CALIBRATED, not arbitrary: at 128x128 with 180 angles they
# put the noisy input at roughly 38 / 32 / 28 dB PSNR for high / medium / low,
# which is the regime real low-dose CT denoising operates in. An earlier set of
# values left the "low" dose input at 46 dB — cleaner than a routine clinical
# scan — which made the denoising task vacuous (a plain U-Net could not beat
# passing the input straight through). If you change the image size, the angle
# count, or P_MAX, re-check these against the target PSNRs above.
DOSE_I0 = {
    "high": 6.0e3,
    "medium": 1.6e3,
    "low": 6.0e2,
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
        clean_sinogram / noisy_sinogram: (B, 1, n_angles, W) line integrals in
        **raw units**, i.e. directly comparable to `radon(image)`.

    The raw-units guarantee matters for Week 3. Internally the photon model works
    on a sinogram rescaled to a fixed peak attenuation (P_MAX) using a per-image
    factor derived from the *clean* phantom — a factor the network cannot know at
    training time. Both sinograms are converted back out of those units before
    being returned, so the PINN's consistency loss can compare
    `radon(model_output)` against `noisy_sinogram` with no rescaling. Returning
    them in P_MAX units would leave a per-image ~30x mismatch that the physics
    weight lambda would silently absorb, making a lambda ablation meaningless.
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

    # 5) Undo the P_MAX rescaling so both sinograms are in raw line-integral
    #    units. `sino_raw` is then exactly radon(image), which is the reference
    #    the PINN's consistency loss forward-projects against.
    scale = peak / P_MAX
    sino_raw = sino * scale
    sino_noisy_raw = sino_noisy * scale

    # 6) Reconstruct ONLY the noise field and add it to the clean image.
    #    Reconstructing (noisy - clean) cancels the structure and the DC term
    #    the ramp filter drops, leaving realistic spatially-correlated CT noise.
    noise_field = fbp(sino_noisy_raw - sino_raw, angles)
    noisy_image = (image + noise_field).clamp(0.0, 1.0)

    return noisy_image, sino_raw, sino_noisy_raw


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
