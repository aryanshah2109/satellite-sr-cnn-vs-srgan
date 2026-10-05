"""VGG19 perceptual loss."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19


class VGGPerceptualLoss(nn.Module):
    """Compare frozen VGG19 features for normalized RGB BCHW images."""

    def __init__(
        self,
        feature_layer: int = 35,
        kind: str = "mse",
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        if kind.lower() not in {"l1", "mse"}:
            raise ValueError(f"Unknown perceptual loss kind: {kind}; expected 'l1' or 'mse'")
        try:
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except (OSError, RuntimeError) as error:
            cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
            raise RuntimeError(
                f"Could not download VGG19 weights. They are cached in {cache_dir}; "
                "check network access or place the torchvision weights there."
            ) from error
        if feature_layer < 0 or feature_layer >= len(model.features):
            raise ValueError(f"feature_layer must be in [0, {len(model.features) - 1}]")
        # The reference SRGAN uses VGG19 features.35 and MSE by default.
        self.features = nn.Sequential(*list(model.features.children())[: feature_layer + 1])
        self.features.eval()
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss() if kind.lower() == "l1" else nn.MSELoss()

    def train(self, mode: bool = True) -> VGGPerceptualLoss:
        """Keep the frozen feature extractor in evaluation mode."""
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return perceptual loss for [0, 1] RGB BCHW tensors."""
        prediction_features = self.features((prediction - self.mean) / self.std)
        with torch.no_grad():
            target_features = self.features((target - self.mean) / self.std)
        return self.criterion(prediction_features, target_features)
