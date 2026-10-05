"""CLI entry point for Phase 0 environment validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.config import Config, load_config
from utils.env_info import log_environment_info
from utils.image_utils import get_device
from utils.logger import setup_logging
from utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)

    return parser.parse_args()


def _verify_paths(cfg: Config) -> None:
    """Create output directories and warn for missing input directories."""
    output_keys = {"checkpoints_dir", "runs_dir", "logs_dir", "results_dir", "docs_dir"}
    input_keys = {
        "data_root",
        "raw_dir",
        "processed_dir",
        "splits_dir",
        "pretrained_dir",
    }

    for key, path in cfg.paths.items():
        if not isinstance(path, Path):
            continue

        if key in output_keys:
            path.mkdir(parents=True, exist_ok=True)
        elif key in input_keys and not path.exists():
            LOGGER.warning("Configured path does not exist: %s", path)


def main() -> int:
    """Run Phase 0 environment checks."""
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, run_name="check_env")

    log_environment_info(LOGGER)
    set_seed(cfg.seed, cfg.deterministic)
    get_device(cfg.device.preferred)
    _verify_paths(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
