"""Tests for atomic project and pretrained checkpoint handling."""

from __future__ import annotations

from pathlib import Path

import torch

from utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    load_pretrained_weights,
    save_checkpoint,
)
from utils.config import load_config


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A tiny model and optimizer round-trip with metadata."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = load_config("configs/base.yaml")
    destination = tmp_path / "checkpoints" / "last.pt"

    save_checkpoint(
        destination,
        model,
        optimizer,
        epoch=3,
        best_metric=0.5,
        metrics={"loss": 0.5},
        cfg=config,
    )

    restored = torch.nn.Linear(2, 1)
    metadata = load_checkpoint(destination, restored, map_location="cpu")

    assert metadata["epoch"] == 3
    assert metadata["best_metric"] == 0.5
    assert find_latest_checkpoint(destination.parent) == destination
    assert torch.equal(model.weight, restored.weight)


def test_pretrained_module_prefix(tmp_path: Path) -> None:
    """Nested state dicts and DataParallel prefixes are normalized."""
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "pretrained.pth.tar"
    state = {f"module.{key}": value for key, value in model.state_dict().items()}
    torch.save({"state_dict": state}, checkpoint)
    restored = torch.nn.Linear(2, 1)

    result = load_pretrained_weights(checkpoint, restored)

    assert result == {"missing_keys": [], "unexpected_keys": []}
    assert torch.equal(model.weight, restored.weight)
