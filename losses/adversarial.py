"""Adversarial losses for GAN training."""

from __future__ import annotations

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
