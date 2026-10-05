"""Aligned HR/LR patch extraction and lossless patch persistence."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sr_datasets.degradation import degrade_image
from sr_datasets.raster_io import read_raster
from sr_datasets.splits import load_split
from utils.config import Config

LOGGER = logging.getLogger(__name__)
MANIFEST_FIELDS = ("patch_name", "source_tile", "row", "col", "size")


def extract_aligned_patches(
    hr: np.ndarray,
    lr: np.ndarray | None,
    hr_patch: int,
    stride: int,
    scale: int,
) -> Iterator[tuple[np.ndarray, np.ndarray | None, tuple[int, int]]]:
    """Yield strictly aligned HR/LR patches and their HR coordinates."""
    if hr.ndim != 3:
        raise ValueError(f"Expected HR CHW input, received {hr.shape}")
    if hr_patch <= 0 or stride <= 0 or scale <= 0:
        raise ValueError("hr_patch, stride, and scale must be positive")
    if hr_patch % scale != 0:
        raise ValueError(f"HR patch size {hr_patch} must be divisible by scale {scale}")
    height, width = hr.shape[-2:]

    lr_patch_size = hr_patch // scale
    for row in range(0, height - hr_patch + 1, stride):
        for col in range(0, width - hr_patch + 1, stride):
            hr_window = hr[:, row : row + hr_patch, col : col + hr_patch]
            lr_window = None
            if lr is not None:
                lr_row = row // scale
                lr_col = col // scale
                lr_window = lr[
                    :, lr_row : lr_row + lr_patch_size, lr_col : lr_col + lr_patch_size
                ]
                if lr_window.shape[-2:] != (lr_patch_size, lr_patch_size):
                    raise ValueError(
                        f"LR alignment failed at HR coordinate {(row, col)}"
                    )
            yield hr_window, lr_window, (row, col)


def _save_array(array: np.ndarray, path: Path, output_format: str) -> None:
    """Save one normalized array losslessly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "npy":
        np.save(path, np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
        return
    if output_format != "png":
        raise ValueError(f"Unsupported lossless patch format: {output_format}")
    if array.shape[0] not in (1, 3, 4):
        raise ValueError("PNG output supports only 1, 3, or 4 channels")
    image_array = np.moveaxis(np.clip(array, 0.0, 1.0), 0, -1)
    if array.shape[0] == 1:
        image_array = image_array[..., 0]
    image = Image.fromarray(np.rint(image_array * 255.0).astype(np.uint8))
    image.save(path, format="PNG")


def _read_array(
    path: Path, output_format: str, bands: list[int], max_value: Any
) -> np.ndarray:
    """Read either a source raster or a stored patch."""
    if output_format == "npy":
        return np.load(path).astype(np.float32) / 255.0
    return read_raster(path, bands, max_value)


def build_patches(cfg: Config, split: str, force: bool = False) -> dict[str, int]:
    """Extract one split's aligned patches from validated tile pairs."""
    try:
        split_ids = set(load_split(cfg, split))
        report_path = Path(cfg.paths.processed_dir) / "dataset_report.json"
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        tile_map = {tile["tile_id"]: tile for tile in report["tiles"]}
        output_format = str(cfg.patching.output_format)
        extension = ".npy" if output_format == "npy" else ".png"
        root = Path(cfg.paths.data_root) / split
        hr_dir = root / "HR"
        lr_dir = root / f"LR_x{cfg.dataset.scale_factor}"
        manifest_path = root / "manifest.csv"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []
        counters = {"written": 0, "skipped": 0, "dropped_nodata": 0, "dropped_blank": 0}
        for tile_id in sorted(split_ids):
            if tile_id not in tile_map:
                raise ValueError(f"Split tile {tile_id} is absent from dataset report")
            tile = tile_map[tile_id]
            hr = read_raster(
                tile["hr_path"], list(cfg.bands.indices), cfg.bands.max_value
            )
            lr = None
            if cfg.lr_source == "provided":
                lr = read_raster(
                    tile["lr_path"], list(cfg.bands.indices), cfg.bands.max_value
                )
            stride = int(cfg.patching.hr_stride[split])
            for hr_patch, lr_patch, (row, col) in extract_aligned_patches(
                hr,
                lr,
                int(cfg.patching.hr_patch_size),
                stride,
                int(cfg.dataset.scale_factor),
            ):
                nodata_fraction = float(np.mean(hr_patch == 0.0))
                if nodata_fraction > float(cfg.patching.max_nodata_fraction):
                    counters["dropped_nodata"] += 1
                    continue
                if float(hr_patch.std()) < float(cfg.patching.min_patch_std):
                    counters["dropped_blank"] += 1
                    continue
                name = f"{tile_id}_y{row}_x{col}{extension}"
                hr_path = hr_dir / name
                lr_path = lr_dir / name
                if hr_path.exists() and lr_path.exists() and not force:
                    counters["skipped"] += 1
                else:
                    _save_array(hr_patch, hr_path, output_format)
                    if lr_patch is None:
                        lr_patch = degrade_image(
                            hr_patch,
                            int(cfg.dataset.scale_factor),
                            cfg.degradation,
                        )
                    _save_array(lr_patch, lr_path, output_format)
                    counters["written"] += 1
                rows.append(
                    {
                        "patch_name": name,
                        "source_tile": tile_id,
                        "row": str(row),
                        "col": str(col),
                        "size": str(cfg.patching.hr_patch_size),
                    }
                )
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Built %s patches for %s: %s", len(rows), split, counters)
        return counters
    except Exception as error:
        LOGGER.error("Failed to build %s patches: %s", split, error)
        raise
