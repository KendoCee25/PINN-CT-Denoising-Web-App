"""Preprocess the TCIA LDCT-and-Projection-data collection into a held-out
real-data validation set (proposal §6.2 risk mitigation: "Supplement with TCIA
public CT patches if needed").

This is NOT training data. It's a real-clinical-acquisition generalisation
check for models trained entirely on the synthetic Shepp-Logan pipeline: same
role as data/dataset/test.npz, but real low-dose/full-dose pairs instead of
simulated Poisson noise. Only 6 patients are available, and there is no raw
projection data in the download (only reconstructed images), so this cannot
train or evaluate the PINN's physics loss — image-domain PSNR/SSIM only.

Each patient has paired Full Dose / Low Dose DICOM series with identical
InstanceNumber/SliceLocation ordering (verified by inspection), so slices are
paired by sorted filename. Pixel data is converted to Hounsfield units via
RescaleSlope/Intercept, windowed to a soft-tissue window, normalised to [0, 1]
to match the synthetic pipeline's value range, and resized to the model's
128x128 input size.

Usage:
    python -m training.load_ldct --root "d:/Data for Project/manifest-.../LDCT-and-Projection-data" --out data/real_ldct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
from skimage.transform import resize

WINDOW_CENTER = 40.0   # soft-tissue window (abdomen), matches the DICOMs' own tag
WINDOW_WIDTH = 300.0


def _dicom_to_unit_image(path: Path, size: int) -> np.ndarray:
    ds = pydicom.dcmread(path)
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    lo, hi = WINDOW_CENTER - WINDOW_WIDTH / 2, WINDOW_CENTER + WINDOW_WIDTH / 2
    windowed = np.clip(hu, lo, hi)
    unit = (windowed - lo) / (hi - lo)  # -> [0, 1], same convention as the phantom pipeline

    return resize(unit, (size, size), anti_aliasing=True, preserve_range=True).astype(np.float32)


def _patient_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir())


def _find_series(patient_dir: Path, label: str) -> Path:
    matches = list(patient_dir.glob(f"*/*{label}*"))
    if not matches:
        raise FileNotFoundError(f"no {label!r} series under {patient_dir}")
    return matches[0]


def build(root: str, out: str, size: int, stride: int) -> None:
    root_path = Path(root)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    clean, noisy, patient_ids, slice_idx = [], [], [], []
    patients = _patient_dirs(root_path)

    for p_idx, patient_dir in enumerate(patients):
        pid = patient_dir.name
        full_dir = _find_series(patient_dir, "Full Dose Images")
        low_dir = _find_series(patient_dir, "Low Dose Images")
        full_files = sorted(full_dir.glob("*.dcm"))
        low_files = sorted(low_dir.glob("*.dcm"))
        if len(full_files) != len(low_files):
            raise ValueError(f"{pid}: full/low slice counts differ ({len(full_files)} vs {len(low_files)})")

        print(f"{pid}: {len(full_files)} slices, taking every {stride}")
        for i in range(0, len(full_files), stride):
            clean.append(_dicom_to_unit_image(full_files[i], size))
            noisy.append(_dicom_to_unit_image(low_files[i], size))
            patient_ids.append(p_idx)
            slice_idx.append(i)

    arrays = {
        "clean": np.stack(clean).astype(np.float32),
        "noisy": np.stack(noisy).astype(np.float32),
        "patient_id": np.array(patient_ids, dtype=np.int32),
        "slice_idx": np.array(slice_idx, dtype=np.int32),
    }
    np.savez_compressed(out_path / "real_ldct.npz", **arrays)

    manifest = {
        "source": "TCIA LDCT-and-Projection-data (McCollough et al.), DOI 10.7937/9npb2637",
        "license": "CC BY 4.0",
        "patients": [d.name for d in patients],
        "n_pairs": len(clean),
        "size": size,
        "stride": stride,
        "window_center_hu": WINDOW_CENTER,
        "window_width_hu": WINDOW_WIDTH,
        "note": "image-domain only; no raw projection data available in this download",
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(clean)} pairs -> {out_path / 'real_ldct.npz'}")


def main():
    ap = argparse.ArgumentParser(description="Preprocess TCIA LDCT DICOMs into a real-data eval set")
    ap.add_argument("--root", type=str, required=True,
                     help="path to the LDCT-and-Projection-data folder (contains C016, C021, ...)")
    ap.add_argument("--out", type=str, default="data/real_ldct")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=10,
                     help="take every Nth slice per patient (adjacent slices are near-duplicates)")
    args = ap.parse_args()
    build(args.root, args.out, args.size, args.stride)


if __name__ == "__main__":
    main()
