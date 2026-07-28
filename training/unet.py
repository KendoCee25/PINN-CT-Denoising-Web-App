"""U-Net denoiser — the shared backbone for the baseline and the PINN.

An encoder-decoder CNN with skip connections (Ronneberger et al., 2015). The
encoder compresses the noisy image to a low-resolution feature bottleneck; the
decoder upsamples back to full resolution; skip connections carry fine spatial
detail across the bottleneck so edges and small lesions are preserved.

Crucially, the baseline and the PINN use the *identical* architecture. Only the
training loss differs (baseline: MSE; PINN: MSE + sinogram-consistency). Keeping
the backbone fixed isolates the effect of the physics-informed loss term, which
is the core comparison of the whole project (proposal §4).

Input/output: (B, 1, H, W) single-channel images.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x2, the standard U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """A compact 4-level U-Net for grayscale image-to-image denoising.

    Args:
        in_ch: input channels (1 for grayscale CT).
        out_ch: output channels (1).
        base: channel width of the first level; doubles each level down.
        residual: if True, the network predicts the noise/residual and the
            output is `input + prediction`. Residual learning is known to help
            low-dose CT denoising (Chen et al., 2017) and stabilises training.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, residual: bool = True):
        super().__init__()
        self.residual = residual

        # Encoder
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base * 4, base * 8)

        # Decoder (transpose-conv upsampling + skip concat)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path (keep features for skips).
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        # Decoder path with skip connections.
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        if self.residual:
            out = x + out
        return out


if __name__ == "__main__":
    # Smoke test: forward pass shapes + parameter count.
    net = UNet()
    x = torch.randn(2, 1, 128, 128)
    y = net(x)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"parameters: {n_params:,}")
