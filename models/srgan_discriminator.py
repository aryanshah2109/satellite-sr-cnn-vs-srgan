"""VGG-style discriminator for HR super-resolution crops."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SRGANDiscriminator(nn.Module):
    """Reference VGG discriminator with adaptive crop-size support."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: int = 64,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 2 * channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * channels, 2 * channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2 * channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * channels, 4 * channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4 * channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * channels, 4 * channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4 * channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * channels, 8 * channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8 * channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * channels, 8 * channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8 * channels),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * channels * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, out_channels),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x: Tensor) -> Tensor:
        """Return one raw discriminator logit per BCHW image."""
        if x.ndim != 4 or x.shape[-2] < 96 or x.shape[-1] < 96:
            raise ValueError(
                "Discriminator inputs must be BCHW with spatial size >= 96"
            )
        output = self.pool(self.features(x))
        return self.classifier(torch.flatten(output, 1))
