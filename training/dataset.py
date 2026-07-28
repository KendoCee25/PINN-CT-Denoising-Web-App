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
    """(noisy_input, clean_target) pairs from a train.npz / test.npz file."""

    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.clean = data["clean"]  # (N, H, W) float32
        self.noisy = data["noisy"]
        self.dose = data["dose"]    # (N,) int8
        self.pid = data["pid"]      # (N,) int32

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int):
        noisy = torch.from_numpy(self.noisy[idx])[None]  # (1, H, W)
        clean = torch.from_numpy(self.clean[idx])[None]
        return noisy, clean
