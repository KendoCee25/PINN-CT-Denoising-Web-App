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
    for batch in loader:
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
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
    quiet: bool = False,
):
    """Train a U-Net denoiser.

    Args:
        extra_loss: optional callable(output, batch) -> scalar tensor, added to
            the MSE term, where `batch` is the dict served by CTDenoiseDataset
            ("noisy", "clean", and "sino" when the dataset carries sinograms).
            This is the PINN hook (Week 3); None trains the plain baseline.
            It takes the whole batch rather than loose tensors so a physics term
            can reach the measured sinogram without changing this signature again.
        quiet: suppress per-epoch logging (used by the lambda ablation, which
            prints its own summary per run).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)

    # Only load sinograms if something is going to consume them — they are the
    # bulk of the file and the baseline has no use for them.
    need_sino = extra_loss is not None
    train_ds = CTDenoiseDataset(data_dir / "train.npz", with_sino=need_sino)
    test_ds = CTDenoiseDataset(data_dir / "test.npz", with_sino=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    if not quiet:
        print(f"Training on {device}: {len(train_ds)} train / {len(test_ds)} test pairs"
              f"{' (+ sinograms)' if need_sino else ''}")
    history = {"epoch": [], "train_loss": [], "image_loss": [], "physics_loss": [],
               "val_psnr": [], "val_ssim": []}
    best_psnr = -1.0
    diverged = False
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = running_img = running_phys = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out_img = model(batch["noisy"])

            img_term = mse(out_img, batch["clean"])
            phys_term = extra_loss(out_img, batch) if extra_loss is not None else None
            loss = img_term if phys_term is None else img_term + phys_term

            # A diverging physics term shows up as a non-finite loss. Bailing out
            # here keeps a failed lambda from poisoning the whole ablation sweep
            # and leaves the last good checkpoint on disk.
            if not torch.isfinite(loss):
                print(f"epoch {epoch}: non-finite loss, stopping this run")
                diverged = True
                break

            opt.zero_grad()
            loss.backward()
            # Gradient clipping: the proposal's stated mitigation for PINN
            # training instability, and harmless for the baseline.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = batch["noisy"].size(0)
            running += loss.item() * bs
            running_img += img_term.item() * bs
            running_phys += (phys_term.item() * bs) if phys_term is not None else 0.0

        if diverged:
            break
        sched.step()

        train_loss = running / len(train_ds)
        val_psnr, val_ssim = evaluate(model, test_loader, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["image_loss"].append(running_img / len(train_ds))
        history["physics_loss"].append(running_phys / len(train_ds))
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        if not quiet:
            phys_str = (f" | phys {running_phys / len(train_ds):.5f}"
                        if extra_loss is not None else "")
            print(f"epoch {epoch:3d} | loss {train_loss:.5f}{phys_str} | "
                  f"val PSNR {val_psnr:.2f} dB | val SSIM {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_psnr": val_psnr, "val_ssim": val_ssim}, out)

    history["diverged"] = diverged
    # Persist training curves next to the checkpoint.
    curves_path = Path(out).with_suffix(".curves.json")
    curves_path.write_text(json.dumps(history, indent=2))
    if not quiet:
        print(f"Best val PSNR {best_psnr:.2f} dB. Checkpoint: {out} | curves: {curves_path}")
    return history


def build_extra_loss(lam: float, data_dir: str, n_angles: int | None):
    """Build the PINN physics term, or None for the plain baseline.

    The angle count is read from the dataset manifest so the loss always lines
    up with the sinograms actually on disk.
    """
    if lam is None:
        return None
    from training.pinn import SinogramConsistencyLoss

    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    return SinogramConsistencyLoss(
        lam=lam, n_angles_full=manifest["n_angles"], n_angles=n_angles
    )


def main():
    ap = argparse.ArgumentParser(
        description="Train the baseline U-Net, or the PINN with --lam"
    )
    ap.add_argument("--data-dir", type=str, default="data/dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="models/baseline.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--lam", type=float, default=None,
                    help="physics weight lambda; omit for the plain baseline")
    ap.add_argument("--loss-angles", type=int, default=None,
                    help="angles used in the physics term (default: all). Lower "
                         "is faster; must divide the dataset's n_angles")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    extra = build_extra_loss(args.lam, args.data_dir, args.loss_angles)
    if extra is not None:
        extra = extra.to(device)
        print(f"PINN mode: {extra}")

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out,
          device, extra_loss=extra)


if __name__ == "__main__":
    main()
