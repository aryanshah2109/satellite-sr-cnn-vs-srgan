"""Deterministic tile-level dataset splitting."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sr_datasets.validation import inspect_dataset
from utils.config import Config, config_hash

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "val", "test")


def _report_path(cfg: Config) -> Path:
    """Return the configured validation report path."""
    return Path(cfg.paths.processed_dir) / "dataset_report.json"


def _valid_tiles(cfg: Config) -> list[tuple[str, str]]:
    """Load valid tile IDs and cities from the inspection report."""
    report_path = _report_path(cfg)
    if not report_path.is_file():
        report = inspect_dataset(cfg)
        return [(tile["tile_id"], tile["city"]) for tile in report.tiles]

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return [(tile["tile_id"], tile["city"]) for tile in report["tiles"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid dataset report at {report_path}: {error}") from error


def _allocate_city(
    tile_ids: list[str],
    ratios: list[float],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Allocate one city's tiles while preserving splits where possible."""
    shuffled = list(tile_ids)
    rng.shuffle(shuffled)
    count = len(shuffled)
    if count == 1:
        counts = [1, 0, 0]
    elif count == 2:
        counts = [1, 1, 0]
    else:
        raw = np.asarray(ratios) * count
        counts = np.floor(raw).astype(int).tolist()
        for index in np.argsort(-(raw - np.floor(raw))):
            if sum(counts) < count:
                counts[int(index)] += 1
        for index in range(len(counts)):
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] = 1

    boundaries = np.cumsum([0, *counts])
    return {
        "train": shuffled[boundaries[0] : boundaries[1]],
        "val": shuffled[boundaries[1] : boundaries[2]],
        "test": shuffled[boundaries[2] : boundaries[3]],
    }


def create_splits(cfg: Config, force: bool = False) -> dict[str, list[str]]:
    """Create deterministic, city-stratified tile splits and persist metadata."""
    try:
        split_config = cfg.split
        ratios = [
            float(split_config.train_ratio),
            float(split_config.val_ratio),
            float(split_config.test_ratio),
        ]
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Split ratios must sum to one: {ratios}")

        split_dir = Path(cfg.paths.splits_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / f"split_seed{split_config.seed}.json"
        current_hash = config_hash(cfg)
        if metadata_path.is_file() and not force:
            existing_text = metadata_path.read_text(encoding="utf-8").strip()
            if not existing_text:
                existing_text = "{}"
            try:
                existing = json.loads(existing_text)
            except (OSError, json.JSONDecodeError) as error:
                LOGGER.warning(
                    "Treating invalid split placeholder as not yet created: %s", error
                )
                existing = {}
            if existing.get("config_hash") and existing["config_hash"] != current_hash:
                raise ValueError(
                    f"Existing split config hash {existing['config_hash']} "
                    "differs from "
                    f"current hash {current_hash}; use force=True to replace it"
                )

        city_tiles: dict[str, list[str]] = defaultdict(list)
        for tile_id, city in _valid_tiles(cfg):
            city_tiles[city].append(tile_id)
        rng = np.random.default_rng(int(split_config.seed))
        result = {name: [] for name in SPLIT_NAMES}
        for city in sorted(city_tiles):
            allocation = _allocate_city(city_tiles[city], ratios, rng)
            for name in SPLIT_NAMES:
                result[name].extend(allocation[name])
            if len(city_tiles[city]) < len(SPLIT_NAMES):
                LOGGER.info(
                    "City %s has %s tiles; assigned only feasible splits",
                    city,
                    len(city_tiles[city]),
                )

        for name in SPLIT_NAMES:
            result[name].sort()
            (split_dir / f"{name}.txt").write_text(
                "\n".join(result[name]) + ("\n" if result[name] else ""),
                encoding="utf-8",
            )
        metadata: dict[str, Any] = {
            "seed": int(split_config.seed),
            "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
            "splits": {
                name: {"tile_ids": result[name], "count": len(result[name])}
                for name in SPLIT_NAMES
            },
            "config_hash": current_hash,
            "created_at": datetime.now(UTC).isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info(
            "Created tile splits: %s", {name: len(result[name]) for name in SPLIT_NAMES}
        )
        return result
    except Exception as error:
        LOGGER.error("Failed to create dataset splits: %s", error)
        raise


def load_split(cfg: Config, split: str) -> list[str]:
    """Load one split and validate disjointness against its sibling files."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split name: {split}")
    split_dir = Path(cfg.paths.splits_dir)
    paths = {name: split_dir / f"{name}.txt" for name in SPLIT_NAMES}
    target = paths[split]
    if not target.is_file():
        raise FileNotFoundError(f"Split file does not exist: {target}")

    values = [
        line.strip()
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_values: dict[str, set[str]] = {}
    for name, path in paths.items():
        if path.is_file():
            all_values[name] = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
    for first in SPLIT_NAMES:
        for second in SPLIT_NAMES:
            if first < second and all_values.get(first, set()) & all_values.get(
                second, set()
            ):
                raise ValueError(f"Dataset splits overlap: {first} and {second}")
    return values
