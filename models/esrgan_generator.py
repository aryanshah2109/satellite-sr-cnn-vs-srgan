"""RRDB-based generator used by ESRGAN-style super-resolution training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _ResidualDenseBlock(nn.Module):
    """Five-layer densely connected block with a scaled residual."""

    def __init__(self, channels: int, growth_channels: int, residual_scale: float) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList(
            nn.Conv2d(channels + index * growth_channels, growth_channels, 3, 1, 1)
            for index in range(4)
        )
        self.final = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.residual_scale = residual_scale

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for convolution in self.convolutions:
            features.append(self.activation(convolution(torch.cat(features, dim=1))))
        return x + self.final(torch.cat(features, dim=1)) * self.residual_scale


class _RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, channels: int, growth_channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                _ResidualDenseBlock(channels, growth_channels, 0.2)
                for _ in range(3)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.blocks(x) * 0.2


class ESRGANGenerator(nn.Module):
    """RRDBNet-style 4× generator accepting normalized BCHW RGB tensors."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 32,
        growth_channels: int = 16,
        num_blocks: int = 6,
        upscale_factor: int = 4,
    ) -> None:
        super().__init__()
        if upscale_factor < 1 or upscale_factor & (upscale_factor - 1):
            raise ValueError("upscale_factor must be a positive power of two")
        if min(in_channels, out_channels, channels, growth_channels, num_blocks) <= 0:
            raise ValueError("All channel and block counts must be positive")

        self.upscale_factor = upscale_factor
        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.trunk = nn.Sequential(
            *[_RRDB(channels, growth_channels) for _ in range(num_blocks)]
        )
        self.trunk_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        upsample_layers: list[nn.Module] = []
        for _ in range(upscale_factor.bit_length() - 1):
            upsample_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, out_channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv_first(x)
        output = features + self.trunk_conv(self.trunk(features))
        output = self.activation(self.conv_hr(self.upsampling(output)))
        return torch.clamp(self.conv_last(output), 0.0, 1.0)
