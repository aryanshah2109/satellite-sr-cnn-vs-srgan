"""Random-seed utilities for reproducible CPU and CUDA experiments."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA random number generators.

    Args:
        seed: Non-negative seed value.
        deterministic: Whether to request deterministic cuDNN algorithms.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    LOGGER.info("Set seed=%s; deterministic=%s", seed, deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from its initial torch seed.

    Args:
        worker_id: DataLoader worker identifier. It is accepted for the
            standard worker-init callback interface.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch generator for DataLoader shuffling.

    Args:
        seed: Non-negative generator seed.

    Returns:
        A seeded CPU generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
