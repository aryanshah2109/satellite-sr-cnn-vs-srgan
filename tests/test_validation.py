import json

import numpy as np
import tifffile

from sr_datasets.validation import _shifted_psnr, inspect_dataset
from utils.config import Config


def _config(tmp_path) -> Config:
    raw = tmp_path / "raw"
    return Config(
        {
            "paths": {
                "data_root": str(tmp_path / "data"),
                "processed_dir": str(tmp_path / "processed"),
            },
            "dataset": {
                "name": "test",
                "raw_root": str(raw),
                "hr_dir": "hr/hr",
                "lr_dir": "lr/lr",
                "file_extension": ".tif",
                "scale_factor": 2,
                "shape_tolerance": 0,
            },
            "bands": {"indices": [0, 1, 2], "max_value": "auto"},
            "split": {"city_regex": r"^(?P<city>[A-Za-z]+)(?P<index>\d+)$"},
        }
    )


def test_validation_reports_missing_shape_and_unreadable(tmp_path) -> None:
    cfg = _config(tmp_path)
    hr_dir = tmp_path / "raw" / "hr" / "hr"
    lr_dir = tmp_path / "raw" / "lr" / "lr"
    hr_dir.mkdir(parents=True)
    lr_dir.mkdir(parents=True)
    valid_hr = np.ones((8, 8, 3), dtype=np.uint8)
    valid_lr = np.ones((4, 4, 3), dtype=np.uint8)
    tifffile.imwrite(hr_dir / "good1.tif", valid_hr)
    tifffile.imwrite(lr_dir / "good1.tif", valid_lr)
    tifffile.imwrite(hr_dir / "missing1.tif", valid_hr)
    tifffile.imwrite(hr_dir / "shape1.tif", valid_hr)
    tifffile.imwrite(lr_dir / "shape1.tif", np.ones((3, 4, 3), dtype=np.uint8))
    (hr_dir / "broken1.tif").write_bytes(b"bad")
    (lr_dir / "broken1.tif").write_bytes(b"bad")
    report = inspect_dataset(cfg)
    assert report.valid_tile_ids == ["good1"]
    assert any("Missing LR" in problem for problem in report.problems)
    assert any("shape mismatch" in problem for problem in report.problems)
    assert any("broken1" in problem for problem in report.problems)
    assert json.loads((tmp_path / "processed" / "dataset_report.json").read_text())


def test_alignment_detects_a_deliberate_pixel_shift() -> None:
    reference = np.zeros((3, 8, 8), dtype=np.float32)
    reference[:, 2:6, 2:6] = 1.0
    shifted = np.zeros_like(reference)
    shifted[:, :, 1:] = reference[:, :, :-1]
    assert _shifted_psnr(reference, shifted, 0, -1) > _shifted_psnr(
        reference, shifted, 0, 0
    )
