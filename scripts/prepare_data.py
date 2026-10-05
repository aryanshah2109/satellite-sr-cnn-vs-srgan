"""Command-line entry point for Phase 1 data preparation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sr_datasets.patching import build_patches
from sr_datasets.splits import create_splits
from sr_datasets.validation import check_alignment, inspect_dataset
from utils.config import Config, config_hash, load_config
from utils.env_info import log_environment_info
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse the Phase 1 subcommand and common options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inspect", "check-alignment", "split", "build", "all"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    return parser.parse_args()


def _run(args: argparse.Namespace, cfg: Config) -> None:
    """Delegate the selected command to the data pipeline modules."""
    if args.command == "inspect":
        inspect_dataset(cfg)
        return
    if args.command == "check-alignment":
        check_alignment(cfg)
        return
    if args.command == "split":
        create_splits(cfg, force=args.force)
        return
    if args.command == "build":
        splits = (args.split,) if args.split else ("train", "val", "test")
        for split in splits:
            build_patches(cfg, split, force=args.force)
        return
    inspect_dataset(cfg)
    create_splits(cfg, force=args.force)
    for split in ("train", "val", "test"):
        build_patches(cfg, split, force=args.force)


def main() -> int:
    """Run a Phase 1 data pipeline command."""
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, run_name="prepare_data")
    log_environment_info(LOGGER)
    LOGGER.info("Using config hash %s", config_hash(cfg))
    _run(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
