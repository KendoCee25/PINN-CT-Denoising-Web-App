"""Generate the synthetic low-dose CT dataset (clean/noisy pairs).

Produces the >=500-slice dataset required in Week 2. For each phantom we render
one clean ground-truth image and a noisy reconstruction at every dose level.
Phantoms are split into train/test *by phantom id* so that no phantom (and none
of its dose variants) leaks across the split — this gives the held-out test set
the evaluation design calls for (Kelly et al., 2019; proposal §4).

Saved layout (under --out, default data/dataset/):
    train.npz / test.npz  with arrays:
        clean  (N, H, W) float32 in [0, 1]
        noisy  (N, H, W) float32 in [0, 1]
        dose   (N,)      int8    0=low, 1=medium, 2=high
        pid    (N,)      int32   phantom id (for grouping / the backend)
    manifest.json  with counts, image size, dose mapping, seed.

Run a small set locally to sanity-check; run the full set on Colab's GPU:
    python -m training.make_dataset --n-phantoms 500 --size 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.noise import add_poisson_noise
from training.phantom import generate_phantom
from training.radon import default_angles

DOSE_CODE = {"low": 0, "medium": 1, "high": 2}


def generate(
    n_phantoms: int,
    size: int,
    doses: list[str],
    n_angles: int,
    seed: int,
    device: str,
    batch_size: int = 64,
):
    """Yield (clean, noisy, dose_code, pid) numpy rows for every phantom x dose."""
    dev = torch.device(device)
    angles = default_angles(n_angles, device=dev)
    # The generator must live on the same device as the tensors torch.poisson
    # samples from, so derive it from `dev` rather than hardcoding CPU.
    gen = torch.Generator(device=dev).manual_seed(seed)

    for start in range(0, n_phantoms, batch_size):
        ids = list(range(start, min(start + batch_size, n_phantoms)))
        # Render this batch of clean phantoms (deterministic per id).
        clean_np = np.stack([generate_phantom(size=size, seed=pid) for pid in ids])
        clean = torch.from_numpy(clean_np)[:, None].to(dev)  # (B,1,H,W)

        for dose in doses:
            noisy, _, _ = add_poisson_noise(clean, dose=dose, angles=angles, generator=gen)
            noisy_np = noisy.squeeze(1).cpu().numpy().astype(np.float32)
            code = DOSE_CODE[dose]
            for k, pid in enumerate(ids):
                yield clean_np[k], noisy_np[k], code, pid


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic low-dose CT dataset")
    ap.add_argument("--n-phantoms", type=int, default=500)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-angles", type=int, default=180)
    ap.add_argument("--doses", nargs="+", default=["low", "medium", "high"])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/dataset")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Split phantom ids first so no phantom leaks across train/test.
    rng = np.random.default_rng(args.seed)
    ids = np.arange(args.n_phantoms)
    rng.shuffle(ids)
    n_test = int(round(args.n_phantoms * args.test_frac))
    test_ids = set(ids[:n_test].tolist())

    buckets = {"train": {"clean": [], "noisy": [], "dose": [], "pid": []},
               "test": {"clean": [], "noisy": [], "dose": [], "pid": []}}

    total = args.n_phantoms * len(args.doses)
    print(f"Generating {total} pairs ({args.n_phantoms} phantoms x {len(args.doses)} doses) "
          f"at {args.size}x{args.size} on {args.device}...")

    for i, (clean, noisy, dose, pid) in enumerate(
        generate(args.n_phantoms, args.size, args.doses, args.n_angles, args.seed, args.device)
    ):
        split = "test" if pid in test_ids else "train"
        b = buckets[split]
        b["clean"].append(clean); b["noisy"].append(noisy)
        b["dose"].append(dose); b["pid"].append(pid)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{total}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, b in buckets.items():
        np.savez_compressed(
            out / f"{split}.npz",
            clean=np.stack(b["clean"]).astype(np.float32),
            noisy=np.stack(b["noisy"]).astype(np.float32),
            dose=np.array(b["dose"], dtype=np.int8),
            pid=np.array(b["pid"], dtype=np.int32),
        )
        print(f"  wrote {split}: {len(b['clean'])} pairs -> {out / f'{split}.npz'}")

    manifest = {
        "n_phantoms": args.n_phantoms,
        "size": args.size,
        "n_angles": args.n_angles,
        "doses": args.doses,
        "dose_code": DOSE_CODE,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "n_train": len(buckets["train"]["clean"]),
        "n_test": len(buckets["test"]["clean"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
