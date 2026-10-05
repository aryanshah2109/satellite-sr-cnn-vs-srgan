"""Configurable low-resolution image degradation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _value(config: Any, key: str, default: Any = None) -> Any:
    """Read one value from a Config-like or mapping object."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def degrade_image(
    hr: np.ndarray | torch.Tensor,
    scale: int,
    degradation_config: Any,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Downscale an HR CHW image with configured deterministic degradation.

    Args:
        hr: Float CHW image in [0, 1].
        scale: Integer downscale factor.
        degradation_config: Config section containing degradation options.
        generator: Optional torch generator for stochastic options.

    Returns:
        Float32 CHW LR image in [0, 1].
    """
    try:
        tensor = torch.as_tensor(hr, dtype=torch.float32)
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected HR CHW input, received shape {tuple(tensor.shape)}"
            )
        height, width = tensor.shape[-2:]
        if scale <= 0 or height % scale != 0 or width % scale != 0:
            raise ValueError(
                f"HR size {height}x{width} must be divisible by scale {scale}"
            )

        result = functional.interpolate(
            tensor.unsqueeze(0),
            size=(height // scale, width // scale),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        noise_range = _value(degradation_config, "noise_sigma_range")
        if noise_range:
            sigma = float(
                torch.empty(1).uniform_(*noise_range, generator=generator).item()
            )
            result = (
                result
                + torch.randn(
                    result.shape,
                    generator=generator,
                    dtype=result.dtype,
                )
                * sigma
            )

        jpeg_range = _value(degradation_config, "jpeg_quality_range")
        if jpeg_range:
            quality = int(
                torch.randint(
                    int(jpeg_range[0]),
                    int(jpeg_range[1]) + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            array = result.clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(array)
            buffer = __import__("io").BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            result = (
                torch.from_numpy(np.asarray(Image.open(buffer)).copy())
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        output = result.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        LOGGER.debug(
            "Degraded HR shape %s to LR shape %s", tuple(tensor.shape), output.shape
        )
        return output
    except Exception as error:
        LOGGER.error("Failed to degrade HR image: %s", error)
        raise


def degrade(
    hr: np.ndarray | torch.Tensor,
    scale: int,
    degradation_config: Any,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Alias for :func:`degrade_image`."""
    return degrade_image(hr, scale, degradation_config, generator)
