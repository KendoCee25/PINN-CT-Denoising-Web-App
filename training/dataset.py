"""PyTorch Dataset for the synthetic low-dose CT pairs.

Loads the .npz files written by make_dataset.py and serves (noisy, clean) tensor
pairs to the training loop. The noisy image is the network input; the clean
phantom is the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CTDenoiseDataset(Dataset):
    """(noisy_input, clean_target) pairs from a train.npz / test.npz file.

    Each item is a dict so the training loop can carry optional extras — namely
    the measured sinogram the PINN's consistency loss needs — without every
    caller having to know whether they are present.

    Args:
        npz_path: path to train.npz / test.npz.
        with_sino: include the measured noisy sinogram under key "sino". The
            file must have been generated without --no-sinograms.
    """

    def __init__(self, npz_path: str | Path, with_sino: bool = False):
        data = np.load(npz_path)
        self.clean = data["clean"]  # (N, H, W) float32
        self.noisy = data["noisy"]
        self.dose = data["dose"]    # (N,) int8
        self.pid = data["pid"]      # (N,) int32

        self.with_sino = with_sino
        if with_sino:
            if "sino" not in data:
                raise KeyError(
                    f"{npz_path} has no 'sino' array — regenerate the dataset without "
                    "--no-sinograms to train the PINN"
                )
            self.sino = data["sino"]  # (N, A, W) float16

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "noisy": torch.from_numpy(self.noisy[idx])[None],  # (1, H, W)
            "clean": torch.from_numpy(self.clean[idx])[None],
        }
        if self.with_sino:
            # Stored float16 to halve the file; widen for the loss.
            item["sino"] = torch.from_numpy(self.sino[idx].astype(np.float32))[None]
        return item
