"""Image-quality metrics: PSNR and SSIM.

These are the two numbers the whole comparison hinges on — shown live in the UI
and used to declare a "winner" between the U-Net and the PINN. Computed against
the clean ground-truth phantom. Thin wrappers over scikit-image so training and
the FastAPI backend share exactly one implementation.

Inputs are single images as float arrays/tensors in [0, 1], shape (H, W) or
(1, H, W). Batched tensors are handled by averaging over the batch.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_2d_numpy(x) -> np.ndarray:
    """Coerce a torch tensor or array of shape (H,W)/(1,H,W)/(1,1,H,W) to 2-D."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"expected a single 2-D image, got shape {x.shape}")
    return np.clip(x, 0.0, 1.0)


def psnr(pred, target) -> float:
    """Peak signal-to-noise ratio (dB), data range 1.0. Higher is better."""
    return float(peak_signal_noise_ratio(_to_2d_numpy(target), _to_2d_numpy(pred), data_range=1.0))


def ssim(pred, target) -> float:
    """Structural similarity in [0, 1]. Higher is better."""
    return float(structural_similarity(_to_2d_numpy(target), _to_2d_numpy(pred), data_range=1.0))


def psnr_ssim(pred, target) -> tuple[float, float]:
    """Convenience: return (PSNR, SSIM) together."""
    return psnr(pred, target), ssim(pred, target)


def batch_mean(pred, target, metric="psnr") -> float:
    """Average a metric over a batch of tensors shaped (B, 1, H, W)."""
    fn = psnr if metric == "psnr" else ssim
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
    pred, target = np.asarray(pred), np.asarray(target)
    return float(np.mean([fn(pred[i], target[i]) for i in range(pred.shape[0])]))
