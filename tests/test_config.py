"""Tests for configuration loading and immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config import config_hash, load_config, save_config


def test_base_load_and_overrides() -> None:
    """Base config values load and existing dotted keys can be overridden."""
    config = load_config("configs/base.yaml", {"device.preferred": "cpu", "seed": 17})

    assert config.seed == 17
    assert config.device.preferred == "cpu"
    assert isinstance(config.paths.logs_dir, Path)


def test_missing_config_key_fails(tmp_path: Path) -> None:
    """Configs without required sections should fail clearly."""
    path = tmp_path / "invalid.yaml"
    path.write_text("_base_: null\nseed: 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required"):
        load_config(path)


def test_hash_is_stable() -> None:
    """Equivalent configurations should have equal hashes."""
    first = load_config("configs/base.yaml")
    second = load_config("configs/base.yaml")

    assert config_hash(first) == config_hash(second)


def test_config_is_immutable() -> None:
    """Accidental mutation should be rejected unless explicitly unlocked."""
    config = load_config("configs/base.yaml")

    with pytest.raises(TypeError):
        config.seed = 1

    with config.unlocked():
        config.seed = 1

    assert config.seed == 1


def test_save_config(tmp_path: Path) -> None:
    """Resolved configs can be written as YAML snapshots."""
    destination = tmp_path / "snapshot.yaml"

    save_config(load_config("configs/base.yaml"), destination)

    assert destination.is_file()
