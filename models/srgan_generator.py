"""State-dict-compatible SRResNet generator."""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


class _ResidualConvBlock(nn.Module):
    """Reference-compatible residual convolution block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block to a BCHW tensor."""
        return self.rcb(x) + x


class _UpsampleBlock(nn.Module):
    """Reference-compatible x2 pixel-shuffle block."""

    def __init__(self, channels: int, upscale_factor: int) -> None:
        super().__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Upsample a BCHW tensor by two."""
        return self.upsample_block(x)


class SRGANGenerator(nn.Module):
    """SRResNet generator compatible with the reference x4 checkpoint.

    The model accepts and returns tensors in [0, 1]. ``forward`` clamps its
    output to that range, matching the reference implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        num_blocks: int = 16,
        upscale_factor: int = 4,
    ) -> None:
        super().__init__()
        if upscale_factor < 1 or upscale_factor & (upscale_factor - 1):
            raise ValueError("upscale_factor must be a positive power of two")
        if num_blocks <= 0 or channels <= 0:
            raise ValueError("channels and num_blocks must be positive")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 9, 1, 4),
            nn.PReLU(),
        )
        self.trunk = nn.Sequential(
            *[_ResidualConvBlock(channels) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.upsampling = nn.Sequential(
            *[
                _UpsampleBlock(channels, 2)
                for _ in range(int(math.log2(upscale_factor)))
            ]
        )
        self.conv3 = nn.Conv2d(channels, out_channels, 9, 1, 4)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize convolution and batch-normalization parameters."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Map BCHW LR tensors to BCHW HR tensors."""
        conv1 = self.conv1(x)
        output = self.trunk(conv1)
        output = self.conv2(output) + conv1
        output = self.upsampling(output)
        output = self.conv3(output)
        return torch.clamp(output, 0.0, 1.0)
