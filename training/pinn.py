r"""Physics-informed loss: differentiable sinogram consistency.

This is the single idea the whole dissertation turns on. The baseline U-Net is
trained only to match the clean phantom in image space (MSE). It has no notion
of how a CT scanner works, so nothing stops it from producing an image that
looks plausible but could never have produced the measurement that was actually
taken.

The PINN adds exactly that constraint. It forward-projects the network's output
through the same differentiable Radon transform the scanner physics is modelled
with, and penalises the L2 distance to the *measured* (noisy) sinogram:

    L = MSE(output, clean)  +  lambda * MSE(radon(output), sinogram_noisy)
        \___ image term ___/    \______ physics term ______/

Because `radon` is built from `grid_sample`, gradients flow from the physics
term back through the projection into the network weights (proposal §3, Week 3).

Two practical points, both anticipated in the proposal's risk table:

* **Units.** `add_poisson_noise` returns sinograms in raw line-integral units,
  so `radon(output)` and the stored sinogram are directly comparable. If they
  were not, lambda would be absorbing a scale factor rather than trading off two
  comparable quantities, and the ablation would measure nothing.
* **Cost.** `radon` loops over projection angles, so the physics term is the
  expensive part of a training step. Angles can be subsampled (`n_angles`) to
  trade fidelity for speed — the proposal's mitigation for a slow forward
  projection is exactly this, 180 -> 60.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.radon import default_angles, radon


class SinogramConsistencyLoss(nn.Module):
    """lambda * MSE(radon(output), measured noisy sinogram).

    Args:
        lam: physics weight lambda. 0.0 disables the term entirely, which makes
            the PINN reduce exactly to the baseline — useful as an ablation
            control that isolates the loss from every other difference.
        n_angles_full: number of angles the stored sinograms were generated
            with. Must match the dataset (see manifest.json).
        n_angles: angles actually used in the loss. Defaults to all of them;
            set lower to subsample for speed. Must divide `n_angles_full` so the
            subsampled projection lines up with stored sinogram rows.
        normalise: divide the residual by the mean squared magnitude of the
            target sinogram, making the term dimensionless.

            This is on by default and it matters. The image term is an MSE over
            [0, 1] pixels (~1e-4 once converged); an unnormalised physics term
            is an MSE over raw line integrals that peak near 45, and it floors
            at the photon-noise variance because a *noisy* sinogram can never be
            matched exactly. The two differ by four to five orders of magnitude,
            so every lambda in the proposal's {0.01, 0.1, 0.5, 1.0} range would
            be physics-dominated and the ablation would compare four flavours of
            "ignore the clean target". Normalising puts both terms on comparable
            footing so lambda reads as a genuine relative weight.
    """

    def __init__(self, lam: float = 0.1, n_angles_full: int = 180,
                 n_angles: int | None = None, normalise: bool = True):
        super().__init__()
        if lam < 0:
            raise ValueError(f"lambda must be non-negative, got {lam}")
        n_angles = n_angles or n_angles_full
        if n_angles_full % n_angles != 0:
            raise ValueError(
                f"n_angles ({n_angles}) must divide n_angles_full ({n_angles_full}) "
                "so subsampled projections align with stored sinogram rows"
            )

        self.lam = float(lam)
        self.n_angles_full = n_angles_full
        self.stride = n_angles_full // n_angles
        self.normalise = normalise

        # Register as buffers so .to(device) moves them with the module.
        all_angles = default_angles(n_angles_full)
        self.register_buffer("angles", all_angles[:: self.stride])
        self.register_buffer("row_idx", torch.arange(0, n_angles_full, self.stride))

    def forward(self, output: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute the weighted physics term.

        Args:
            output: (B, 1, H, W) network output.
            batch: the dict served by CTDenoiseDataset. Only "sino" is read —
                the (B, 1, n_angles_full, W) measured noisy sinogram in raw
                line-integral units. Taking the whole batch rather than a bare
                tensor is what lets `training.train`'s extra_loss hook stay
                fixed as losses start needing more of the sample.
        """
        if self.lam == 0.0:
            # Skip the projection entirely — this is the control arm, and the
            # forward transform is the expensive part of the step.
            return output.sum() * 0.0

        if "sino" not in batch:
            raise KeyError(
                "batch has no 'sino' — build the dataset with sinograms and load "
                "it with CTDenoiseDataset(..., with_sino=True)"
            )
        sino_noisy = batch["sino"]
        if sino_noisy.shape[2] != self.n_angles_full:
            raise ValueError(
                f"sinogram has {sino_noisy.shape[2]} angles but the loss was built for "
                f"{self.n_angles_full}; check manifest.json's n_angles"
            )

        sino_pred = radon(output, self.angles)
        target = sino_noisy[:, :, self.row_idx, :]
        residual = F.mse_loss(sino_pred, target)

        if self.normalise:
            # Relative residual: dimensionless, and independent of image size and
            # attenuation scale, so a lambda tuned at 128x128 still means the
            # same thing at 256x256.
            residual = residual / target.pow(2).mean().clamp_min(1e-8)
        return self.lam * residual

    def extra_repr(self) -> str:
        n_used = self.n_angles_full // self.stride
        return (f"lam={self.lam}, angles={n_used}/{self.n_angles_full}, "
                f"normalise={self.normalise}")
