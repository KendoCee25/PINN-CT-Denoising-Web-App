"""Lambda ablation: find the stable regime for the physics weight.

Week 3 of the proposal calls for training with lambda in {0.01, 0.1, 0.5, 1.0}
and identifying which values train stably before committing to a full run. This
sweeps them under otherwise identical conditions and writes one comparable row
per value.

lambda = 0.0 is included by default as a control. It runs the same code path as
every other arm with the physics term switched off, so it isolates the effect of
the loss from the effect of everything else that changed between Week 2 and
Week 3 (gradient clipping, the batch-dict refactor). Comparing the PINN against
that control is a fairer test than comparing it against the Week 2 checkpoint.

Runs use fewer epochs than a final training run — the point is to find a stable
regime cheaply, then retrain the winner properly:

    python -m training.ablate --data-dir data/dataset --epochs 15 --device cuda

Results land in models/ablation.json and a per-lambda checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.metrics import psnr, ssim
from training.pinn import SinogramConsistencyLoss
from training.train import train
from training.unet import UNet

DEFAULT_LAMBDAS = [0.0, 0.01, 0.1, 0.5, 1.0]


@torch.no_grad()
def evaluate_per_dose(model, data_dir: Path, device: str) -> dict:
    """Per-dose PSNR/SSIM plus the delta over the untouched noisy input.

    The delta is the number that matters: a denoiser is only useful if it beats
    passing its input straight through.
    """
    data = np.load(data_dir / "test.npz")
    clean, noisy, dose = data["clean"], data["noisy"], data["dose"]
    names = {0: "low", 1: "medium", 2: "high"}
    model.eval()

    out = {}
    for code in [0, 1, 2, None]:
        idx = np.arange(len(clean)) if code is None else np.where(dose == code)[0]
        in_p, out_p, in_s, out_s = [], [], [], []
        for start in range(0, len(idx), 64):
            b = idx[start:start + 64]
            x = torch.from_numpy(noisy[b])[:, None].to(device)
            y = model(x).clamp(0, 1).cpu().numpy()[:, 0]
            for k, i in enumerate(b):
                in_p.append(psnr(noisy[i], clean[i])); out_p.append(psnr(y[k], clean[i]))
                in_s.append(ssim(noisy[i], clean[i])); out_s.append(ssim(y[k], clean[i]))
        out[names.get(code, "all")] = {
            "input_psnr": float(np.mean(in_p)), "psnr": float(np.mean(out_p)),
            "input_ssim": float(np.mean(in_s)), "ssim": float(np.mean(out_s)),
            "delta_psnr": float(np.mean(out_p) - np.mean(in_p)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Ablate the PINN physics weight lambda")
    ap.add_argument("--data-dir", type=str, default="data/dataset")
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-angles", type=int, default=60,
                    help="angles in the physics term; 60 of 180 is the proposal's "
                         "mitigation for a slow forward projection")
    ap.add_argument("--out-dir", type=str, default="models")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    if not manifest.get("has_sinograms", False):
        raise SystemExit(
            "This dataset has no sinograms, so the physics term cannot be trained. "
            "Regenerate with: python -m training.make_dataset (without --no-sinograms)"
        )

    print(f"Ablating lambda over {args.lambdas} | {args.epochs} epochs each | "
          f"{args.loss_angles}/{manifest['n_angles']} angles | device {device}\n")

    results = []
    for lam in args.lambdas:
        tag = f"lam{lam:g}".replace(".", "p")
        ckpt = out_dir / f"pinn_{tag}.pt"
        loss_fn = SinogramConsistencyLoss(
            lam=lam, n_angles_full=manifest["n_angles"], n_angles=args.loss_angles
        ).to(device)

        print(f"--- lambda = {lam:g} " + "-" * 40)
        history = train(
            str(data_dir), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            out=str(ckpt), device=device, extra_loss=loss_fn, quiet=True,
        )

        if history["diverged"]:
            print(f"  DIVERGED after {len(history['epoch'])} epochs — excluded\n")
            results.append({"lam": lam, "diverged": True,
                            "epochs_completed": len(history["epoch"])})
            continue

        saved = torch.load(ckpt, map_location=device)
        model = UNet().to(device)
        model.load_state_dict(saved["model_state"])
        per_dose = evaluate_per_dose(model, data_dir, device)

        row = {
            "lam": lam,
            "diverged": False,
            "best_val_psnr": max(history["val_psnr"]),
            "best_val_ssim": max(history["val_ssim"]),
            "final_image_loss": history["image_loss"][-1],
            "final_physics_loss": history["physics_loss"][-1],
            "per_dose": per_dose,
            "checkpoint": str(ckpt),
        }
        results.append(row)
        print(f"  val PSNR {row['best_val_psnr']:.2f} dB | SSIM {row['best_val_ssim']:.4f} | "
              f"delta vs input {per_dose['all']['delta_psnr']:+.2f} dB\n")

    # Summary table.
    print("=" * 78)
    print(f"{'lambda':<9}{'val PSNR':<11}{'val SSIM':<11}{'delta(all)':<12}"
          f"{'delta(low)':<12}{'status'}")
    print("-" * 78)
    for r in results:
        if r["diverged"]:
            print(f"{r['lam']:<9g}{'-':<11}{'-':<11}{'-':<12}{'-':<12}"
                  f"diverged @ epoch {r['epochs_completed']}")
        else:
            print(f"{r['lam']:<9g}{r['best_val_psnr']:<11.2f}{r['best_val_ssim']:<11.4f}"
                  f"{r['per_dose']['all']['delta_psnr']:<+12.2f}"
                  f"{r['per_dose']['low']['delta_psnr']:<+12.2f}ok")

    stable = [r for r in results if not r["diverged"]]
    if stable:
        best = max(stable, key=lambda r: r["best_val_psnr"])
        control = next((r for r in stable if r["lam"] == 0.0), None)
        print("-" * 78)
        print(f"Best lambda: {best['lam']:g} ({best['best_val_psnr']:.2f} dB)")
        if control is not None and best["lam"] != 0.0:
            gain = best["best_val_psnr"] - control["best_val_psnr"]
            verdict = "physics term helps" if gain > 0 else "physics term does NOT help"
            print(f"vs lambda=0 control: {gain:+.2f} dB — {verdict}")

    out_path = out_dir / "ablation.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
