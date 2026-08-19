"""Evaluate the trained checkpoints on real clinical data (TCIA LDCT-and-
Projection-data), as a generalisation check against models trained entirely on
the synthetic Shepp-Logan pipeline.

This is NOT a training set and NOT part of the weekly plan's required
deliverables — it's the proposal's own risk-mitigation contingency (§6.2:
"Synthetic data insufficiently realistic -> supplement with TCIA public CT
patches"). Build the eval set first:

    python -m training.load_ldct --root "<path to LDCT-and-Projection-data>" --out data/real_ldct

Then run this:

    python -m training.eval_real_ldct

Writes data/real_ldct/eval_results.json (the numbers to cite) and
data/real_ldct_qualitative.png (a few example slices).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from training.metrics import psnr, ssim
from training.unet import UNet


def _load_model(path: str, device: str) -> UNet:
    ckpt = torch.load(path, map_location=device)
    model = UNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _run(model: UNet, x: np.ndarray, device: str, batch_size: int = 32) -> np.ndarray:
    outs = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size])[:, None].to(device)
        outs.append(model(batch).clamp(0.0, 1.0).cpu().numpy()[:, 0])
    return np.concatenate(outs)


def _mean_metrics(pred: np.ndarray, clean: np.ndarray) -> dict:
    return {
        "psnr": float(np.mean([psnr(pred[i], clean[i]) for i in range(len(clean))])),
        "ssim": float(np.mean([ssim(pred[i], clean[i]) for i in range(len(clean))])),
    }


def evaluate(data_dir: str, models_dir: str, device: str) -> dict:
    data = np.load(Path(data_dir) / "real_ldct.npz")
    clean, noisy = data["clean"], data["noisy"]

    # Drop degenerate slices (pure-air scan edges where full-dose and low-dose
    # are pixel-identical after windowing) -- not meaningful denoising cases,
    # and an exact match (MSE=0) produces an infinite PSNR that corrupts the mean.
    keep = clean.std(axis=(1, 2)) > 1e-4
    n_dropped = int((~keep).sum())
    clean, noisy = clean[keep], noisy[keep]

    models = {
        "baseline": _load_model(str(Path(models_dir) / "baseline.pt"), device),
        "pinn": _load_model(str(Path(models_dir) / "pinn.pt"), device),
    }

    results = {"n_pairs": len(clean), "n_dropped_degenerate": n_dropped,
               "input": _mean_metrics(noisy, clean)}
    for name, model in models.items():
        pred = _run(model, noisy, device)
        m = _mean_metrics(pred, clean)
        m["delta_psnr_vs_input"] = m["psnr"] - results["input"]["psnr"]
        results[name] = m

    return results, clean, noisy, models


def save_qualitative(clean, noisy, models, device, out_path: str, n_examples: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(clean), size=n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 4, figsize=(14, 11 * n_examples / 3))
    for row, i in enumerate(picks):
        x = torch.from_numpy(noisy[i])[None, None].to(device)
        with torch.no_grad():
            b_out = models["baseline"](x).clamp(0, 1).cpu().numpy()[0, 0]
            p_out = models["pinn"](x).clamp(0, 1).cpu().numpy()[0, 0]
        panels = [
            (noisy[i], "noisy (real low-dose)"),
            (b_out, "baseline"),
            (p_out, "PINN"),
            (clean[i], "clean (real full-dose)"),
        ]
        for ax, (img, title) in zip(axes[row], panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if "clean" in title:
                ax.set_title(title)
            else:
                ax.set_title(f"{title}\nPSNR {psnr(img, clean[i]):.1f} | SSIM {ssim(img, clean[i]):.3f}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained checkpoints on real TCIA LDCT slices")
    ap.add_argument("--data-dir", type=str, default="data/real_ldct")
    ap.add_argument("--models-dir", type=str, default="models")
    ap.add_argument("--out-image", type=str, default="data/real_ldct_qualitative.png")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results, clean, noisy, models = evaluate(args.data_dir, args.models_dir, device)

    print(f"real LDCT eval set: {results['n_pairs']} pairs "
          f"({results['n_dropped_degenerate']} degenerate blank slices dropped)")
    print(f"{'noisy input':<10}: PSNR {results['input']['psnr']:.2f} dB / SSIM {results['input']['ssim']:.4f}")
    for name in ("baseline", "pinn"):
        r = results[name]
        print(f"{name:<10}: PSNR {r['psnr']:.2f} dB / SSIM {r['ssim']:.4f}  "
              f"(delta {r['delta_psnr_vs_input']:+.2f} dB)")

    out_json = Path(args.data_dir) / "eval_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    save_qualitative(clean, noisy, models, device, args.out_image)
    print(f"Wrote {args.out_image}")


if __name__ == "__main__":
    main()
