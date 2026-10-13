"""Adversarial losses for GAN training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GANLoss(nn.Module):
    """Calculate BCE-with-logits or least-squares GAN losses."""

    def __init__(self, kind: str = "bce", real_label_smoothing: float = 0.0) -> None:
        super().__init__()
        normalized_kind = kind.lower()
        if normalized_kind not in {"bce", "lsgan"}:
            raise ValueError(f"Unknown GAN loss kind: {kind}; expected 'bce' or 'lsgan'")
        self.kind = normalized_kind
        self.real_label_smoothing = float(real_label_smoothing)
        self.criterion = nn.BCEWithLogitsLoss() if normalized_kind == "bce" else nn.MSELoss()

    def _target(self, logits: Tensor, value: float) -> Tensor:
        return logits.new_full(logits.shape, value)

    def generator_loss(self, fake_logits: Tensor) -> Tensor:
        """Return the generator loss for fake logits."""
        return self.criterion(fake_logits, self._target(fake_logits, 1.0))

    def discriminator_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Return the discriminator loss for real and fake logits."""
        real_value = 1.0 - self.real_label_smoothing
        real_loss = self.criterion(real_logits, self._target(real_logits, real_value))
        fake_loss = self.criterion(fake_logits, self._target(fake_logits, 0.0))
        return (real_loss + fake_loss) / 2.0


class RelativisticAverageGANLoss(nn.Module):
    """Relativistic-average BCE losses used by ESRGAN-style training."""

    def __init__(self, real_label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.real_label_smoothing = float(real_label_smoothing)
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Train D to rank real samples above the average generated sample."""
        real_relative = real_logits - fake_logits.mean()
        fake_relative = fake_logits - real_logits.mean()
        real_target = torch.full_like(real_relative, 1.0 - self.real_label_smoothing)
        fake_target = torch.zeros_like(fake_relative)
        return (
            self.criterion(real_relative, real_target)
            + self.criterion(fake_relative, fake_target)
        ) / 2.0

    def generator_loss(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        """Train G to make generated samples rank above real samples."""
        real_relative = real_logits - fake_logits.mean()
        fake_relative = fake_logits - real_logits.mean()
        real_target = torch.zeros_like(real_relative)
        fake_target = torch.ones_like(fake_relative)
        return (
            self.criterion(real_relative, real_target)
            + self.criterion(fake_relative, fake_target)
        ) / 2.0
