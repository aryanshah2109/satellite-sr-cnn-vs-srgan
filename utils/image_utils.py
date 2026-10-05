"""Image loading, conversion, normalization, and shape utility functions."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from utils.config import load_config

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _image_settings() -> Any:
    """Load image constants from the shared configuration."""
    return load_config("configs/base.yaml").image


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB.

    Args:
        path: Existing image path.

    Returns:
        RGB PIL image.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the extension is unsupported.
    """
    image_path = Path(path)
    settings = _image_settings()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    if image_path.suffix.lower() not in {
        extension.lower() for extension in settings.supported_extensions
    }:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load image {image_path}: {error}") from error


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to float32 CHW tensor in [0, 1].

    Args:
        image: PIL image to convert.

    Returns:
        Tensor with shape (3, H, W), dtype float32, and values in [0, 1].
    """
    rgb_image = image.convert("RGB")
    tensor = torch.frombuffer(bytearray(rgb_image.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(rgb_image.height, rgb_image.width, 3)

    return tensor.permute(2, 0, 1).contiguous().to(torch.float32).div(255.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW or BCHW tensor in [0, 1] to an RGB PIL image.

    Args:
        tensor: Image tensor with shape (3, H, W) or (1, 3, H, W).

    Returns:
        RGB PIL image with uint8 pixels.

    Raises:
        ValueError: If the tensor shape is not CHW or BCHW with one image.
    """
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected tensor shape (3, H, W) or (1, 3, H, W)")

    array = tensor.detach().cpu().float().clamp(0.0, 1.0)
    array = array.mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()

    return Image.fromarray(array, mode="RGB")


def save_image(image_or_tensor: Image.Image | torch.Tensor, path: str | Path) -> None:
    """Save a PIL image or tensor, creating parent directories.

    Args:
        image_or_tensor: RGB PIL image or image tensor.
        path: Destination image path.

    Raises:
        TypeError: If the input is not a PIL image or tensor.
        OSError: If the image cannot be written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = (
        tensor_to_pil(image_or_tensor)
        if isinstance(image_or_tensor, torch.Tensor)
        else image_or_tensor
    )

    if not isinstance(image, Image.Image):
        raise TypeError("image_or_tensor must be a PIL image or torch.Tensor")

    try:
        image.save(destination)
    except OSError as error:
        raise OSError(f"Could not save image to {destination}: {error}") from error


def to_minus_one_one(tensor: torch.Tensor) -> torch.Tensor:
    """Map a [0, 1] tensor to the [-1, 1] range."""
    return tensor.mul(2.0).sub(1.0)


def from_minus_one_one(tensor: torch.Tensor) -> torch.Tensor:
    """Map a [-1, 1] tensor to the [0, 1] range and clamp it."""
    return tensor.add(1.0).div(2.0).clamp(0.0, 1.0)


def rgb_to_y(tensor: torch.Tensor) -> torch.Tensor:
    """Convert RGB data to the BT.601 Y channel.

    Args:
        tensor: RGB tensor shaped (3, H, W) or (N, 3, H, W) in [0, 1].

    Returns:
        Y tensor shaped (1, H, W) or (N, 1, H, W).

    Raises:
        ValueError: If the input does not have three RGB channels.
    """
    if tensor.ndim not in (3, 4) or tensor.shape[-3] != 3:
        raise ValueError("Expected RGB tensor shape (3, H, W) or (N, 3, H, W)")

    settings = _image_settings()
    coefficients = torch.tensor(
        settings.y_channel_coefficients,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    coefficient_shape = (3, 1, 1) if tensor.ndim == 3 else (1, 3, 1, 1)
    channels = tensor * coefficients.view(coefficient_shape)
    return channels.sum(dim=-3, keepdim=True) + settings.y_offset


def crop_border(tensor: torch.Tensor, border: int) -> torch.Tensor:
    """Crop equal borders from a CHW or BCHW tensor.

    Args:
        tensor: Image tensor shaped (C, H, W) or (N, C, H, W).
        border: Number of pixels to remove from each edge.

    Returns:
        Cropped tensor.

    Raises:
        ValueError: If shape or border is invalid.
    """
    if tensor.ndim not in (3, 4) or border < 0:
        raise ValueError("Expected CHW/BCHW tensor and non-negative border")

    height, width = tensor.shape[-2:]
    if border * 2 >= height or border * 2 >= width:
        raise ValueError("border removes all pixels from at least one dimension")

    return tensor[..., border : height - border, border : width - border]


def get_device(preferred: str) -> torch.device:
    """Resolve the requested device, falling back from unavailable CUDA.

    Args:
        preferred: One of ``auto``, ``cuda``, or ``cpu``.

    Returns:
        Resolved torch device.

    Raises:
        ValueError: If the preference is unsupported.
        RuntimeError: If CUDA was explicitly requested but unavailable.
    """
    normalized = preferred.lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device preference: {preferred}")

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    device = torch.device(
        "cuda"
        if normalized == "cuda" or (normalized == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    LOGGER.info("Using device: %s", device)

    return device


def center_crop_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Center-crop spatial dimensions to be divisible by ``multiple``.

    Args:
        tensor: Tensor with spatial dimensions in its final two axes.
        multiple: Positive divisibility factor.

    Returns:
        Center-cropped tensor.

    Raises:
        ValueError: If the multiple is not positive or tensor is too small.
    """
    if multiple <= 0 or tensor.ndim < 2:
        raise ValueError("multiple must be positive and tensor must have spatial axes")

    height, width = tensor.shape[-2:]
    target_height = height - height % multiple
    target_width = width - width % multiple
    if target_height == 0 or target_width == 0:
        raise ValueError("tensor dimensions must be at least the multiple")

    top = (height - target_height) // 2
    left = (width - target_width) // 2

    return tensor[..., top : top + target_height, left : left + target_width]
