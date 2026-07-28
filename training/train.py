"""Train the baseline U-Net denoiser (MSE loss).

Week 2 deliverable: a trained baseline checkpoint plus per-epoch PSNR/SSIM curves
showing convergence. The baseline minimises MSE between its output and the clean
phantom. In Week 3 the PINN reuses this exact loop and simply adds a
sinogram-consistency term (see the `extra_loss` hook).

Full training is intended for Colab's GPU; a few epochs run fine on CPU for a
sanity check.

    python -m training.train --data-dir data/dataset --epochs 40 --out models/baseline.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import CTDenoiseDataset
from training.metrics import batch_mean
from training.unet import UNet


@torch.no_grad()
def evaluate(model, loader, device):
    """Return mean (PSNR, SSIM) over a loader."""
    model.eval()
    psnrs, ssims, n = 0.0, 0.0, 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy).clamp(0.0, 1.0)
        bs = noisy.size(0)
        psnrs += batch_mean(out, clean, "psnr") * bs
        ssims += batch_mean(out, clean, "ssim") * bs
        n += bs
    return psnrs / n, ssims / n


def train(
    data_dir: str,
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 1e-3,
    out: str = "models/baseline.pt",
    device: str | None = None,
    extra_loss=None,
):
    """Train a U-Net denoiser.

    Args:
        extra_loss: optional callable(output, noisy, clean) -> scalar tensor,
            added to the MSE term. Used by the PINN (Week 3); None for baseline.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)

    train_ds = CTDenoiseDataset(data_dir / "train.npz")
    test_ds = CTDenoiseDataset(data_dir / "test.npz")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    print(f"Training on {device}: {len(train_ds)} train / {len(test_ds)} test pairs")
    history = {"epoch": [], "train_loss": [], "val_psnr": [], "val_ssim": []}
    best_psnr = -1.0
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out_img = model(noisy)
            loss = mse(out_img, clean)
            if extra_loss is not None:
                loss = loss + extra_loss(out_img, noisy, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * noisy.size(0)
        sched.step()

        train_loss = running / len(train_ds)
        val_psnr, val_ssim = evaluate(model, test_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        print(f"epoch {epoch:3d} | loss {train_loss:.5f} | "
              f"val PSNR {val_psnr:.2f} dB | val SSIM {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_psnr": val_psnr, "val_ssim": val_ssim}, out)

    # Persist training curves next to the checkpoint.
    curves_path = Path(out).with_suffix(".curves.json")
    curves_path.write_text(json.dumps(history, indent=2))
    print(f"Best val PSNR {best_psnr:.2f} dB. Checkpoint: {out} | curves: {curves_path}")
    return history


def main():
    ap = argparse.ArgumentParser(description="Train baseline U-Net denoiser")
    ap.add_argument("--data-dir", type=str, default="data/dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/baseline.pt")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out, args.device)


if __name__ == "__main__":
    main()
