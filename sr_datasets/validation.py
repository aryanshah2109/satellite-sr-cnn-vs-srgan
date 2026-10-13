"""Satellite tile discovery and integrity validation."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from sr_datasets.degradation import degrade_image
from sr_datasets.raster_io import raster_metadata, read_raster
from utils.config import Config

LOGGER = logging.getLogger(__name__)
VALID_EXTENSIONS = {".tif", ".tiff"}


@dataclass(frozen=True)
class TileRecord:
    """Describe one validated HR/LR tile pair."""

    tile_id: str
    city: str
    hr_path: str
    lr_path: str
    hr_metadata: dict[str, Any]
    lr_metadata: dict[str, Any]
    nodata_fraction: float
    band_means: list[float]
    band_stds: list[float]


@dataclass(frozen=True)
class DatasetReport:
    """Summarize validation results for the configured dataset."""

    dataset: str
    reader_backend: str
    tiles: list[dict[str, Any]]
    valid_tile_ids: list[str]
    tiles_per_city: dict[str, int]
    aggregate_statistics: dict[str, Any]
    problems: list[str]
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report mapping."""
        return asdict(self)


def _path_from_config(cfg: Config, configured: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    value = Path(configured)
    if value.is_absolute():
        return value

    project_root = Path(cfg.paths.data_root).parent
    return (project_root / value).resolve()


def _dataset_directories(cfg: Config) -> tuple[Path, Path]:
    """Resolve HR and LR input directories from configuration."""
    raw_root = _path_from_config(cfg, cfg.dataset.raw_root)
    return raw_root / cfg.dataset.hr_dir, raw_root / cfg.dataset.lr_dir


def _inspect_image_folder(cfg: Config) -> DatasetReport:
    """Inspect class-organized HR images without requiring paired LR files."""
    raw_root = _path_from_config(cfg, cfg.dataset.raw_root)
    image_root = raw_root / cfg.dataset.hr_dir
    if not image_root.is_dir():
        raise FileNotFoundError(f"Configured image folder does not exist: {image_root}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    class_counts: Counter[str] = Counter()
    for class_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            tile_id = f"{class_dir.name}__{path.stem}"
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    chw = np.moveaxis(np.asarray(rgb, dtype=np.float32) / 255.0, -1, 0)
                if width < int(cfg.patching.hr_patch_size) or height < int(
                    cfg.patching.hr_patch_size
                ):
                    raise ValueError(
                        f"image is {width}x{height}; patch size is "
                        f"{cfg.patching.hr_patch_size}"
                    )
                records.append(
                    {
                        "tile_id": tile_id,
                        "city": class_dir.name,
                        "hr_path": str(path.resolve()),
                        "lr_path": "",
                        "hr_metadata": {
                            "width": width,
                            "height": height,
                            "band_count": 3,
                            "dtype": "uint8",
                            "crs": None,
                            "pixel_size": None,
                            "nodata": None,
                        },
                        "lr_metadata": {
                            "width": width // int(cfg.dataset.scale_factor),
                            "height": height // int(cfg.dataset.scale_factor),
                            "band_count": 3,
                            "dtype": "uint8",
                            "crs": None,
                            "pixel_size": None,
                            "nodata": None,
                        },
                        "nodata_fraction": 0.0,
                        "band_means": chw.mean(axis=(1, 2)).tolist(),
                        "band_stds": chw.std(axis=(1, 2)).tolist(),
                    }
                )
                class_counts[class_dir.name] += 1
            except (OSError, ValueError) as error:
                problems.append(f"Excluded image {path}: {error}")

    if not records:
        raise ValueError(f"No supported images found under {image_root}")
    means = np.asarray([record["band_means"] for record in records], dtype=np.float32)
    stds = np.asarray([record["band_stds"] for record in records], dtype=np.float32)
    report = DatasetReport(
        dataset=str(cfg.dataset.name),
        reader_backend="Pillow RGB",
        tiles=records,
        valid_tile_ids=[record["tile_id"] for record in records],
        tiles_per_city=dict(sorted(class_counts.items())),
        aggregate_statistics={
            "per_band_mean": means.mean(axis=0).tolist(),
            "per_band_std": stds.mean(axis=0).tolist(),
            "nodata_fraction_mean": 0.0,
        },
        problems=problems,
        counts={
            "hr_files": len(records),
            "lr_files": 0,
            "valid_tiles": len(records),
            "excluded_tiles": len(problems),
            "unreadable_files": len(problems),
        },
    )
    _save_report(cfg, report)
    _save_data_card(cfg, report)
    LOGGER.info("Inspected %s RGB images across %s classes", len(records), len(class_counts))
    return report


def _city_from_tile(cfg: Config, tile_id: str) -> str:
    """Parse a city from a tile identifier using the configured regex."""
    match = re.match(cfg.split.city_regex, tile_id)
    return match.group("city") if match else tile_id


def _list_tiles(directory: Path, extension: str) -> dict[str, Path]:
    """List files by stem in one configured raster directory."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Configured raster directory does not exist: {directory}"
        )

    suffix = extension.lower()
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == suffix
    }


def _nodata_fraction(path: Path, metadata: dict[str, Any], bands: list[int]) -> float:
    """Estimate the fraction of pixels containing nodata in selected bands."""
    nodata = metadata.get("nodata")
    if nodata is None:
        return 0.0

    array = read_raster(path, bands, "auto")
    scale = 1.0
    raw_nodata = float(nodata)
    dtype_name = metadata["dtype"]
    if dtype_name.startswith("uint8"):
        scale = 255.0
    elif dtype_name.startswith("uint16"):
        scale = 65535.0
    normalized_nodata = raw_nodata / scale
    mask = np.isclose(array, normalized_nodata).any(axis=0)
    return float(mask.mean())


def _statistics(array: np.ndarray) -> tuple[list[float], list[float]]:
    """Compute per-band means and standard deviations."""
    return (
        [float(value) for value in array.mean(axis=(1, 2))],
        [float(value) for value in array.std(axis=(1, 2))],
    )


def _save_report(cfg: Config, report: DatasetReport) -> Path:
    """Save the report under the configured processed-data directory."""
    destination = Path(cfg.paths.processed_dir) / "dataset_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Saved dataset report to %s", destination)
    return destination


def _save_data_card(cfg: Config, report: DatasetReport) -> None:
    """Write a concise data card from the inspection report."""
    valid_tiles = report.tiles
    hr_sizes = sorted(
        {
            f"{tile['hr_metadata']['height']}x{tile['hr_metadata']['width']}"
            for tile in valid_tiles
        }
    )
    lr_sizes = sorted(
        {
            f"{tile['lr_metadata']['height']}x{tile['lr_metadata']['width']}"
            for tile in valid_tiles
        }
    )
    bands = sorted({tile["hr_metadata"]["band_count"] for tile in valid_tiles})
    dtypes = sorted({tile["hr_metadata"]["dtype"] for tile in valid_tiles})
    issues = (
        "\n".join(f"- {problem}" for problem in report.problems) or "- None reported"
    )
    card = f"""# Super-resolution data card

## Summary

- Valid HR images: {len(report.valid_tile_ids)}
- Classes: {len(report.tiles_per_city)}
- HR sizes: {', '.join(hr_sizes) or 'none'}
- LR sizes: {', '.join(lr_sizes) or 'none'}
- Band counts: {bands or 'none'}
- HR dtypes: {', '.join(dtypes) or 'none'}
- Reader backend: {report.reader_backend}
- LR source: {'synthetic downsampling' if report.counts['lr_files'] == 0 else 'provided pairs'}

## Tiles per city

{json.dumps(report.tiles_per_city, indent=2)}

## Known issues

{issues}

## Provenance

- Source URL: TODO: fill in manually
- License: TODO: fill in manually
- Collection date: TODO: fill in manually
"""
    destination = Path(cfg.paths.data_root) / "README.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(card, encoding="utf-8")
    LOGGER.info("Saved data card to %s", destination)


def inspect_dataset(cfg: Config) -> DatasetReport:
    """Inspect configured source images and save a detailed validation report."""
    try:
        if str(getattr(cfg.dataset, "source", "geotiff")) == "image_folder":
            return _inspect_image_folder(cfg)
        hr_dir, lr_dir = _dataset_directories(cfg)
        extension = str(cfg.dataset.file_extension)
        hr_tiles = _list_tiles(hr_dir, extension)
        lr_tiles = _list_tiles(lr_dir, extension)
        problems: list[str] = []

        for tile_id in sorted(set(hr_tiles) - set(lr_tiles)):
            problems.append(f"Missing LR pair for tile {tile_id}")
            LOGGER.warning("Skipping %s: missing LR pair", tile_id)
        for tile_id in sorted(set(lr_tiles) - set(hr_tiles)):
            problems.append(f"Missing HR pair for tile {tile_id}")
            LOGGER.warning("Skipping %s: missing HR pair", tile_id)

        bands = list(cfg.bands.indices)
        scale = int(cfg.dataset.scale_factor)
        records: list[TileRecord] = []
        unreadable = 0
        shape_errors = 0
        for tile_id in sorted(set(hr_tiles) & set(lr_tiles)):
            hr_path = hr_tiles[tile_id]
            lr_path = lr_tiles[tile_id]
            try:
                hr_meta = raster_metadata(hr_path)
                lr_meta = raster_metadata(lr_path)
                if hr_meta["band_count"] != lr_meta["band_count"]:
                    raise ValueError(
                        f"band-count mismatch HR={hr_meta['band_count']} "
                        f"LR={lr_meta['band_count']}"
                    )
                if hr_meta["dtype"] != lr_meta["dtype"]:
                    raise ValueError(
                        f"dtype mismatch HR={hr_meta['dtype']} LR={lr_meta['dtype']}"
                    )
                expected_height = hr_meta["height"] / scale
                expected_width = hr_meta["width"] / scale
                tolerance = float(getattr(cfg.dataset, "shape_tolerance", 0.0))
                if (
                    abs(lr_meta["height"] - expected_height) > tolerance
                    or abs(lr_meta["width"] - expected_width) > tolerance
                ):
                    shape_errors += 1
                    raise ValueError(
                        f"shape mismatch HR={hr_meta['height']}x{hr_meta['width']} "
                        f"LR={lr_meta['height']}x{lr_meta['width']} expected "
                        f"{expected_height}x{expected_width}"
                    )

                hr_array = read_raster(hr_path, bands, cfg.bands.max_value)
                means, stds = _statistics(hr_array)
                nodata_fraction = _nodata_fraction(hr_path, hr_meta, bands)
                records.append(
                    TileRecord(
                        tile_id=tile_id,
                        city=_city_from_tile(cfg, tile_id),
                        hr_path=str(hr_path),
                        lr_path=str(lr_path),
                        hr_metadata=hr_meta,
                        lr_metadata=lr_meta,
                        nodata_fraction=nodata_fraction,
                        band_means=means,
                        band_stds=stds,
                    )
                )
            except Exception as error:
                unreadable += (
                    1 if isinstance(error, (OSError, FileNotFoundError)) else 0
                )
                problems.append(f"Excluded tile {tile_id}: {error}")
                LOGGER.warning("Skipping tile %s: %s", tile_id, error)

        if not records:
            raise ValueError("No valid HR/LR tile pairs were found")
        if shape_errors == len(records) + shape_errors:
            raise ValueError("All discovered tile pairs have the wrong scale")

        city_counts = Counter(record.city for record in records)
        means = np.asarray([record.band_means for record in records], dtype=np.float32)
        stds = np.asarray([record.band_stds for record in records], dtype=np.float32)
        report = DatasetReport(
            dataset=str(cfg.dataset.name),
            reader_backend="tifffile",
            tiles=[asdict(record) for record in records],
            valid_tile_ids=[record.tile_id for record in records],
            tiles_per_city=dict(sorted(city_counts.items())),
            aggregate_statistics={
                "per_band_mean": means.mean(axis=0).tolist(),
                "per_band_std": stds.mean(axis=0).tolist(),
                "nodata_fraction_mean": float(
                    np.mean([record.nodata_fraction for record in records])
                ),
            },
            problems=problems,
            counts={
                "hr_files": len(hr_tiles),
                "lr_files": len(lr_tiles),
                "valid_tiles": len(records),
                "excluded_tiles": len(problems),
                "unreadable_files": unreadable,
            },
        )
        _save_report(cfg, report)
        _save_data_card(cfg, report)
        LOGGER.info(
            "Inspected %s valid tiles across %s cities", len(records), len(city_counts)
        )
        return report
    except Exception as error:
        LOGGER.error("Failed to inspect dataset: %s", error)
        raise


def _alignment_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calculate PSNR for two equally shaped normalized CHW arrays."""
    mse = float(
        np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    )
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _shifted_psnr(
    reference: np.ndarray,
    candidate: np.ndarray,
    shift_y: int,
    shift_x: int,
) -> float:
    """Compare overlapping pixels after applying a candidate shift."""
    height, width = reference.shape[-2:]
    ref_y0 = max(0, shift_y)
    ref_y1 = min(height, height + shift_y)
    ref_x0 = max(0, shift_x)
    ref_x1 = min(width, width + shift_x)
    cand_y0 = max(0, -shift_y)
    cand_y1 = min(height, height - shift_y)
    cand_x0 = max(0, -shift_x)
    cand_x1 = min(width, width - shift_x)
    return _alignment_psnr(
        reference[:, ref_y0:ref_y1, ref_x0:ref_x1],
        candidate[:, cand_y0:cand_y1, cand_x0:cand_x1],
    )


def _sample_alignment_tiles(
    records: list[dict[str, Any]], sample_count: int, seed: int
) -> list[dict[str, Any]]:
    """Select deterministic samples spread across cities."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["city"]), []).append(record)
    rng = np.random.default_rng(seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    cities = sorted(grouped)
    while len(selected) < sample_count and cities:
        for city in cities:
            if grouped[city] and len(selected) < sample_count:
                selected.append(grouped[city].pop())
    return selected


def check_alignment(cfg: Config) -> dict[str, Any]:
    """Check sampled HR/LR spatial alignment and save a report and plot."""
    try:
        report_path = Path(cfg.paths.processed_dir) / "dataset_report.json"
        if not report_path.is_file():
            report = inspect_dataset(cfg)
            records = report.tiles
        else:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            records = report_data["tiles"]
        alignment_config = cfg.alignment
        samples = _sample_alignment_tiles(
            records,
            int(alignment_config.sample_count),
            int(alignment_config.seed),
        )
        shifts = [
            (shift_y, shift_x) for shift_y in range(-2, 3) for shift_x in range(-2, 3)
        ]
        results: list[dict[str, Any]] = []
        plot_items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for record in samples:
            hr = read_raster(
                record["hr_path"], list(cfg.bands.indices), cfg.bands.max_value
            )
            if cfg.lr_source == "synthetic":
                lr = degrade_image(hr, int(cfg.dataset.scale_factor), cfg.degradation)
            else:
                lr = read_raster(
                    record["lr_path"], list(cfg.bands.indices), cfg.bands.max_value
                )
            target_height = lr.shape[-2] * int(cfg.dataset.scale_factor)
            target_width = lr.shape[-1] * int(cfg.dataset.scale_factor)
            hr = hr[:, :target_height, :target_width]
            upsampled = (
                functional.interpolate(
                    torch.from_numpy(lr).unsqueeze(0),
                    size=(target_height, target_width),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(0)
                .numpy()
                .clip(0.0, 1.0)
            )
            shift_scores = {
                f"{shift_y},{shift_x}": _shifted_psnr(hr, upsampled, shift_y, shift_x)
                for shift_y, shift_x in shifts
            }
            best_key = max(shift_scores, key=shift_scores.get)
            best_shift = tuple(int(value) for value in best_key.split(","))
            result = {
                "tile_id": record["tile_id"],
                "best_shift": list(best_shift),
                "psnr_at_zero": shift_scores["0,0"],
                "best_psnr": shift_scores[best_key],
            }
            results.append(result)
            if len(plot_items) < int(alignment_config.max_plots):
                plot_items.append(
                    (record["tile_id"], upsampled, hr, np.abs(upsampled - hr))
                )
            if best_shift != (0, 0):
                LOGGER.warning(
                    "Tile %s has best alignment shift %s", record["tile_id"], best_shift
                )

        zero_scores = [float(result["psnr_at_zero"]) for result in results]
        misaligned = sum(result["best_shift"] != [0, 0] for result in results)
        mean_zero = float(np.mean(zero_scores)) if zero_scores else float("nan")
        if misaligned > len(results) * 0.25:
            LOGGER.error(
                "Alignment check found %s/%s misaligned tiles", misaligned, len(results)
            )
        if mean_zero < float(alignment_config.min_psnr):
            LOGGER.warning(
                "Mean zero-shift PSNR %.4f is below configured threshold", mean_zero
            )
        destination = Path(cfg.paths.processed_dir) / "alignment_report.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "sample_count": len(results),
            "mean_psnr_at_zero": mean_zero,
            "misaligned_count": misaligned,
            "tiles": results,
        }
        destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
        plot_path = Path(cfg.paths.results_dir) / "plots" / "alignment_check.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(
            len(plot_items), 3, squeeze=False, figsize=(12, 4 * len(plot_items))
        )
        for row, (tile_id, upsampled, hr, difference) in enumerate(plot_items):
            for column, image in enumerate((upsampled, hr, difference)):
                axes[row, column].imshow(np.moveaxis(image, 0, -1))
                axes[row, column].axis("off")
            axes[row, 0].set_title(f"{tile_id} upsampled LR")
            axes[row, 1].set_title("HR")
            axes[row, 2].set_title("absolute difference")
        figure.tight_layout()
        figure.savefig(plot_path)
        plt.close(figure)
        LOGGER.info(
            "Alignment check: %s/%s misaligned, mean PSNR(0,0)=%.4f",
            misaligned,
            len(results),
            mean_zero,
        )
        return output
    except Exception as error:
        LOGGER.error("Failed to check raster alignment: %s", error)
        raise
