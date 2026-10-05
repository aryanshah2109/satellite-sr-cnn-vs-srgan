"""Runtime, dependency, hardware, and repository environment reporting."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import subprocess
import sys
from typing import Any

import numpy
import torch

LOGGER = logging.getLogger(__name__)
PINNED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "PyYAML",
    "tqdm",
    "tensorboard",
    "scikit-image",
    "lpips",
    "matplotlib",
)


def _package_version(name: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_info() -> tuple[str | None, bool | None]:
    """Return git commit and dirty state, handling non-repositories."""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None

    if commit_result.returncode != 0 or dirty_result.returncode != 0:
        return None, None

    return commit_result.stdout.strip() or None, bool(dirty_result.stdout.strip())


def collect_environment_info() -> dict[str, Any]:
    """Collect software, hardware, dependency, and git environment facts."""
    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_info.append(
                {
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                }
            )

    git_commit, git_dirty = _git_info()
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except (ImportError, RuntimeError):
        torchvision_version = None

    try:
        import PIL

        pillow_version = PIL.__version__
    except ImportError:
        pillow_version = None

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "numpy_version": numpy.__version__,
        "pillow_version": pillow_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpus": gpu_info,
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "pinned_package_versions": {
            name: _package_version(name) for name in PINNED_PACKAGES
        },
    }


def log_environment_info(logger: logging.Logger) -> None:
    """Collect and log environment information at INFO level."""
    environment = collect_environment_info()

    for key, value in environment.items():
        logger.info("environment.%s=%s", key, value)
