"""Pixel reconstruction losses."""

from __future__ import annotations

from torch import Tensor, nn


class PixelLoss(nn.Module):
    """Calculate an L1 or MSE pixel loss for BCHW tensors."""

    def __init__(self, kind: str = "l1") -> None:
        super().__init__()
        normalized_kind = kind.lower()
        if normalized_kind == "l1":
            self.loss = nn.L1Loss()
        elif normalized_kind == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown pixel loss kind: {kind}; expected 'l1' or 'mse'")

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return the scalar pixel loss for BCHW tensors."""
        return self.loss(prediction, target)
