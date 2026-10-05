"""Inspect a PyTorch checkpoint without modifying model state."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from utils.config import load_config
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse checkpoint inspection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    return parser.parse_args()


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Inspect top-level and state-dict key structure."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        weights_only = True
    except Exception as error:
        raise RuntimeError(
            f"Could not load checkpoint with weights_only=True: {error}. "
            "Use the explicit unsafe fallback only after review."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    state_dict = payload.get("state_dict", payload.get("model_state_dict", payload))
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no mapping state dictionary: {path}")
    prefixes = sorted({key.split(".")[0] for key in state_dict})
    shapes = {
        key: list(value.shape)
        for key, value in state_dict.items()
        if hasattr(value, "shape")
    }
    result = {
        "path": str(path),
        "weights_only": weights_only,
        "top_level_keys": list(payload.keys()),
        "state_dict_key": (
            "state_dict" if "state_dict" in payload else "model_state_dict"
        ),
        "state_dict_count": len(state_dict),
        "prefixes": prefixes,
        "keys": list(state_dict.keys()),
        "shapes": shapes,
    }
    LOGGER.info("Checkpoint top-level keys: %s", result["top_level_keys"])
    LOGGER.info(
        "State dictionary contains %s keys with prefixes %s", len(state_dict), prefixes
    )
    for key in list(state_dict)[:10]:
        LOGGER.info("Checkpoint key %s shape=%s", key, shapes.get(key))
    return result


def main() -> int:
    """Run checkpoint inspection."""
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, run_name="inspect_checkpoint")
    inspect_checkpoint(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
