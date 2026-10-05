"""Tests for deterministic random seed helpers."""

from __future__ import annotations

import random

import numpy as np
import torch

from utils.seed import get_torch_generator, seed_worker, set_seed


def test_same_seed_repeats_random_numpy_and_torch() -> None:
    """The same seed should reproduce all supported random streams."""
    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(3))

    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_generator_is_reproducible() -> None:
    """Seeded torch generators should produce equal sequences."""
    first = torch.rand(3, generator=get_torch_generator(7))
    second = torch.rand(3, generator=get_torch_generator(7))

    assert torch.equal(first, second)


def test_worker_seed_callback() -> None:
    """The worker callback can be invoked through the DataLoader interface."""
    set_seed(1)
    seed_worker(0)
