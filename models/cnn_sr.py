"""EDSR-lite CNN super-resolution baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _ResidualBlock(nn.Module):
    """Batch-normalization-free residual block with residual scaling."""

    def __init__(self, channels: int, res_scale: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        """Apply the scaled residual mapping."""
        return x + self.body(x) * self.res_scale


class CNNSuperResolution(nn.Module):
    """EDSR-lite generator accepting and returning [0, 1] BCHW tensors."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
        upscale_factor: int = 4,
    ) -> None:
        super().__init__()
        if upscale_factor < 1 or upscale_factor & (upscale_factor - 1):
            raise ValueError("upscale_factor must be a positive power of two")
        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(
            *[_ResidualBlock(channels, res_scale) for _ in range(num_blocks)],
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        upsample_layers: list[nn.Module] = []
        for _ in range(upscale_factor.bit_length() - 1):
            upsample_layers.extend(
                [nn.Conv2d(channels, channels * 4, 3, 1, 1), nn.PixelShuffle(2)]
            )
        self.upsampler = nn.Sequential(*upsample_layers)
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Map BCHW LR tensors to clamped BCHW HR tensors."""
        head = self.head(x)
        output = self.body(head) + head
        output = self.upsampler(output)
        return torch.clamp(self.tail(output), 0.0, 1.0)
