"""Shared pytest fixtures for Phase 0 utility tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def tiny_image_size() -> tuple[int, int]:
    """Return a small deterministic image size for fast tests."""
    return 4, 6
