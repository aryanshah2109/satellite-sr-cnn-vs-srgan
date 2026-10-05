import csv

import numpy as np
import torch

from sr_datasets.sr_dataset import SRDataset, build_dataloader
from utils.config import Config


def _config(tmp_path) -> Config:
    root = tmp_path / "data" / "train"
    (root / "HR").mkdir(parents=True)
    (root / "LR_x2").mkdir()
    name = "tile1_y0_x0.npy"
    hr = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8) / 192.0
    lr = hr[:, ::2, ::2]
    np.save(root / "HR" / name, hr)
    np.save(root / "LR_x2" / name, lr)
    with (root / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("patch_name", "source_tile", "row", "col", "size")
        )
        writer.writeheader()
        writer.writerow(
            {"patch_name": name, "source_tile": "tile1", "row": 0, "col": 0, "size": 8}
        )
    return Config(
        {
            "paths": {"data_root": str(tmp_path / "data")},
            "dataset": {"scale_factor": 2},
            "patching": {"output_format": "npy"},
            "training_crop": {
                "hr_crop_size": 4,
                "augment_flip": False,
                "augment_rot90": False,
            },
            "lr_source": "provided",
            "bands": {"indices": [0, 1, 2], "max_value": "auto"},
            "dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            },
            "split": {"seed": 42},
        }
    )


def test_dataset_shapes_ranges_and_deterministic_val(tmp_path) -> None:
    cfg = _config(tmp_path)
    first = SRDataset(cfg, "train", "val")[0]
    second = SRDataset(cfg, "train", "val")[0]
    assert first["hr"].shape == (3, 8, 8)
    assert first["lr"].shape == (3, 4, 4)
    assert torch.equal(first["hr"], second["hr"])
    assert 0 <= float(first["hr"].min()) <= float(first["hr"].max()) <= 1


def test_dataloader_seed_is_reproducible(tmp_path) -> None:
    cfg = _config(tmp_path)
    first = next(iter(build_dataloader(cfg, "train")))
    second = next(iter(build_dataloader(cfg, "train")))
    assert torch.equal(first["hr"], second["hr"])
