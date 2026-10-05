import numpy as np
import pytest
import tifffile

from sr_datasets.raster_io import raster_metadata, read_raster


def test_raster_normalization_band_selection_and_metadata(tmp_path) -> None:
    path = tmp_path / "sample.tif"
    array = np.zeros((4, 5, 3), dtype=np.uint16)
    array[..., 0] = 65535
    array[..., 1] = 32768
    tifffile.imwrite(path, array)
    result = read_raster(path, [1], "auto")
    metadata = raster_metadata(path)
    assert result.shape == (1, 4, 5)
    assert np.isclose(result.mean(), 32768 / 65535)
    assert metadata["width"] == 5
    assert metadata["height"] == 4
    assert metadata["band_count"] == 3
    assert metadata["dtype"] == "uint16"


def test_raster_uint8_normalization(tmp_path) -> None:
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, np.full((3, 4, 3), 255, dtype=np.uint8))
    assert np.allclose(read_raster(path, [0], "auto"), 1.0)


def test_raster_corrupt_file_has_context(tmp_path) -> None:
    path = tmp_path / "broken.tif"
    path.write_bytes(b"not a TIFF")
    with pytest.raises(OSError, match="broken.tif"):
        read_raster(path, [0])
