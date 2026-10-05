"""GeoTIFF reading and metadata helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_VALUES = {
    "uint8": float(np.iinfo(np.uint8).max),
    "uint16": float(np.iinfo(np.uint16).max),
    "uint32": float(np.iinfo(np.uint32).max),
    "int8": float(np.iinfo(np.int8).max),
    "int16": float(np.iinfo(np.int16).max),
    "int32": float(np.iinfo(np.int32).max),
}


def _as_chw(array: np.ndarray, band_count: int) -> np.ndarray:
    """Convert tifffile's common HWC or CHW layouts to CHW."""
    if array.ndim == 2:
        return array[np.newaxis, ...]

    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D raster, received shape {array.shape}")

    if array.shape[0] == band_count and array.shape[1] != band_count:
        return array

    if array.shape[-1] == band_count:
        return np.moveaxis(array, -1, 0)

    raise ValueError(f"Could not identify band axis for raster shape {array.shape}")


def _normalization_max(dtype: np.dtype[Any]) -> float:
    """Return a sensible normalization maximum for a raster dtype."""
    dtype_name = dtype.name
    if dtype_name in DEFAULT_MAX_VALUES:
        return DEFAULT_MAX_VALUES[dtype_name]

    if np.issubdtype(dtype, np.floating):
        return 1.0

    raise ValueError(f"Unsupported raster dtype for normalization: {dtype}")


def _read_array(path: Path) -> tuple[np.ndarray, tifffile.TiffFile]:
    """Read an array and open TIFF handle with contextual errors."""
    if not path.is_file():
        raise FileNotFoundError(f"Raster file does not exist: {path}")

    try:
        tiff = tifffile.TiffFile(path)
        array = tiff.asarray()
    except Exception as error:
        raise OSError(f"Could not read raster {path}: {error}") from error

    return array, tiff


def _select_bands(array: np.ndarray, bands: list[int] | tuple[int, ...]) -> np.ndarray:
    """Select zero-based bands and validate their bounds."""
    if not bands:
        raise ValueError("At least one raster band must be selected")

    if min(bands) < 0 or max(bands) >= array.shape[0]:
        raise ValueError(
            f"Band selection {bands} is invalid for {array.shape[0]} available bands"
        )

    return array[list(bands)]


def read_raster(
    path: str | Path,
    bands: list[int] | tuple[int, ...],
    max_value: float | str = "auto",
) -> np.ndarray:
    """Read a raster as normalized CHW float32 data.

    Args:
        path: GeoTIFF path.
        bands: Zero-based band indices to return.
        max_value: Explicit normalization maximum or ``"auto"``.

    Returns:
        Raster data in CHW layout and the [0, 1] range.
    """
    raster_path = Path(path)
    tiff: tifffile.TiffFile | None = None

    try:
        array, tiff = _read_array(raster_path)
        band_count = int(tiff.pages[0].samplesperpixel) if array.ndim == 3 else 1
        chw = _as_chw(array, band_count)
        selected = _select_bands(chw, bands)
        scale = (
            _normalization_max(selected.dtype)
            if max_value == "auto"
            else float(max_value)
        )
        if scale <= 0:
            raise ValueError(f"Normalization maximum must be positive: {scale}")

        result = selected.astype(np.float32) / np.float32(scale)
        result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)
        LOGGER.debug("Read raster %s with shape %s", raster_path, result.shape)
        return result
    except Exception as error:
        LOGGER.error("Failed to read raster %s: %s", raster_path, error)
        raise
    finally:
        if tiff is not None:
            tiff.close()


def raster_metadata(path: str | Path) -> dict[str, Any]:
    """Return dimensions, dtype, bands, nodata, CRS, and pixel size metadata."""
    raster_path = Path(path)
    tiff: tifffile.TiffFile | None = None

    try:
        _, tiff = _read_array(raster_path)
        series = tiff.series[0]
        shape = series.shape
        page = tiff.pages[0]
        band_count = int(page.samplesperpixel) if len(shape) == 3 else 1
        if len(shape) == 2:
            height, width = int(shape[0]), int(shape[1])
        elif shape[-1] == band_count:
            height, width = int(shape[0]), int(shape[1])
        else:
            height, width = int(shape[1]), int(shape[2])
        tags = page.tags
        scale_tag = tags.get("ModelPixelScaleTag")
        pixel_size = (
            tuple(float(value) for value in scale_tag.value[:2]) if scale_tag else None
        )
        nodata_tag = tags.get("GDAL_NODATA")
        nodata = None if nodata_tag is None else float(str(nodata_tag.value))

        metadata: dict[str, Any] = {
            "width": width,
            "height": height,
            "band_count": band_count,
            "dtype": np.dtype(series.dtype).name,
            "crs": None,
            "pixel_size": pixel_size,
            "nodata": nodata,
        }
        LOGGER.debug("Raster metadata for %s: %s", raster_path, metadata)
        return metadata
    except Exception as error:
        LOGGER.error("Failed to inspect raster metadata %s: %s", raster_path, error)
        raise
    finally:
        if tiff is not None:
            tiff.close()
