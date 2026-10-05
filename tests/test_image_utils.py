"""Tests for image conversion and shape utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from utils.image_utils import (
    center_crop_to_multiple,
    crop_border,
    from_minus_one_one,
    load_image,
    pil_to_tensor,
    rgb_to_y,
    save_image,
    tensor_to_pil,
    to_minus_one_one,
)


def test_pil_tensor_round_trip(tmp_path: Path) -> None:
    """PIL and tensor conversions preserve uint8 RGB pixels."""
    image = Image.new("RGB", (3, 2), (12, 128, 255))
    tensor = pil_to_tensor(image)
    restored = tensor_to_pil(tensor)

    assert tensor.shape == (3, 2, 3)
    assert list(restored.getdata()) == list(image.getdata())

    destination = tmp_path / "nested" / "image.png"
    save_image(tensor, destination)
    assert load_image(destination).size == image.size


def test_range_conversion() -> None:
    """Range conversion maps the endpoints exactly."""
    values = torch.tensor([0.0, 0.5, 1.0])

    assert torch.equal(to_minus_one_one(values), torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(from_minus_one_one(torch.tensor([-1.0, 0.0, 1.0])), values)


def test_y_channel_known_colors() -> None:
    """BT.601 conversion produces expected values for primary colors."""
    colors = torch.eye(3).reshape(3, 3, 1)
    values = rgb_to_y(colors).flatten()

    assert torch.allclose(values, torch.tensor([0.299, 0.587, 0.114]))


def test_crops() -> None:
    """Border and centered divisibility crops have expected shapes."""
    tensor = torch.zeros(3, 7, 9)

    assert crop_border(tensor, 1).shape == (3, 5, 7)
    assert center_crop_to_multiple(tensor, 4).shape == (3, 4, 8)


def test_invalid_image_inputs(tmp_path: Path) -> None:
    """Missing and unsupported image files raise clear errors."""
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")

    unsupported = tmp_path / "image.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_image(unsupported)
