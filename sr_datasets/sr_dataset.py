"""PyTorch dataset and dataloader for stored SR patch pairs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from sr_datasets.degradation import degrade_image
from sr_datasets.raster_io import read_raster
from utils.config import Config
from utils.seed import get_torch_generator, seed_worker

LOGGER = logging.getLogger(__name__)


class SRDataset(Dataset[dict[str, Any]]):
    """Load aligned stored HR/LR patches for one dataset split."""

    def __init__(self, cfg: Config, split: str, mode: str) -> None:
        """Initialize a dataset from its split manifest."""
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")
        self.cfg = cfg
        self.split = split
        self.mode = mode
        self.scale = int(cfg.dataset.scale_factor)
        self.output_format = str(cfg.patching.output_format)
        self.root = Path(cfg.paths.data_root) / split
        self.hr_dir = self.root / "HR"
        self.lr_dir = self.root / f"LR_x{self.scale}"
        self.extension = ".npy" if self.output_format == "npy" else ".png"
        configured_crops = int(cfg.training_crop.get("crops_per_patch", 1))
        self.crops_per_patch = configured_crops if mode == "train" else 1
        self.manifest = self.root / "manifest.csv"
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Patch manifest does not exist: {self.manifest}")
        with self.manifest.open(newline="", encoding="utf-8") as manifest_file:
            self.rows = list(csv.DictReader(manifest_file))
        if not self.rows:
            raise ValueError(f"Patch manifest is empty: {self.manifest}")

    def __len__(self) -> int:
        """Return the number of stored patches."""
        return len(self.rows) * self.crops_per_patch

    def _load(self, path: Path) -> np.ndarray:
        """Load one stored normalized patch."""
        if self.output_format == "npy":
            max_value = float(self.cfg.image.rgb_range) * 255.0
            return np.load(path).astype(np.float32) / max_value
        return read_raster(path, list(self.cfg.bands.indices), self.cfg.bands.max_value)

    def _augment(
        self, hr: torch.Tensor, lr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply identical random geometric transforms to HR and LR."""
        crop_size = int(self.cfg.training_crop.hr_crop_size)
        if self.mode == "train" and crop_size < hr.shape[-1]:
            max_row = hr.shape[-2] - crop_size
            max_col = hr.shape[-1] - crop_size
            row = int(torch.randint(max_row // self.scale + 1, (1,)).item()) * self.scale
            col = int(torch.randint(max_col // self.scale + 1, (1,)).item()) * self.scale
            hr = hr[:, row : row + crop_size, col : col + crop_size]
            lr_size = crop_size // self.scale
            lr = lr[
                :,
                row // self.scale : row // self.scale + lr_size,
                col // self.scale : col // self.scale + lr_size,
            ]
        if self.mode == "train" and bool(self.cfg.training_crop.augment_flip):
            if bool(torch.randint(2, (1,)).item()):
                hr = torch.flip(hr, (-1,))
                lr = torch.flip(lr, (-1,))
        if self.mode == "train" and bool(self.cfg.training_crop.augment_rot90):
            turns = int(torch.randint(4, (1,)).item())
            hr = torch.rot90(hr, turns, (-2, -1))
            lr = torch.rot90(lr, turns, (-2, -1))
        return hr, lr

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one aligned LR/HR tensor pair."""
        try:
            row = self.rows[index % len(self.rows)]
            name = row["patch_name"]
            hr_path = self.hr_dir / name
            lr_path = self.lr_dir / name
            hr = torch.from_numpy(self._load(hr_path))
            if self.cfg.lr_source == "synthetic":
                lr = torch.from_numpy(
                    degrade_image(hr, self.scale, self.cfg.degradation)
                )
            else:
                lr = torch.from_numpy(self._load(lr_path))
            hr, lr = self._augment(hr, lr)
            expected_lr = (hr.shape[-2] // self.scale, hr.shape[-1] // self.scale)
            if hr.ndim != 3 or lr.ndim != 3 or tuple(lr.shape[-2:]) != expected_lr:
                raise ValueError(
                    f"Invalid aligned shapes for {name}: "
                    f"HR={tuple(hr.shape)} LR={tuple(lr.shape)}"
                )
            if not torch.isfinite(hr).all() or not torch.isfinite(lr).all():
                raise ValueError(f"Non-finite values found in patch {name}")
            if not (0.0 <= float(hr.min()) and float(hr.max()) <= 1.0):
                raise ValueError(f"HR values outside [0, 1] in patch {name}")
            if not (0.0 <= float(lr.min()) and float(lr.max()) <= 1.0):
                raise ValueError(f"LR values outside [0, 1] in patch {name}")
            return {"lr": lr.float(), "hr": hr.float(), "name": name}
        except Exception as error:
            LOGGER.error("Failed to load dataset item %s: %s", index, error)
            raise


def build_dataloader(
    cfg: Config,
    split: str,
    indices: list[int] | None = None,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader[dict[str, Any]]:
    """Build a seeded dataloader with train-only shuffling."""
    mode = split if split in {"train", "val", "test"} else "train"
    dataset = SRDataset(cfg, split, mode)
    dataset_object: Dataset[dict[str, Any]] = dataset
    if indices is not None:
        dataset_object = Subset(dataset, indices)
    training = cfg.training if "training" in cfg else cfg.dataloader
    workers = int(training.num_workers) if num_workers is None else int(num_workers)
    loader_batch_size = int(training.batch_size) if batch_size is None else int(batch_size)
    loader_shuffle = mode == "train" if shuffle is None else bool(shuffle)
    loader_kwargs: dict[str, Any] = {
        "batch_size": loader_batch_size,
        "shuffle": loader_shuffle,
        "num_workers": workers,
        "pin_memory": bool(cfg.dataloader.pin_memory),
        "worker_init_fn": seed_worker,
        "generator": get_torch_generator(int(cfg.split.seed)),
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.dataloader.get("persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(cfg.dataloader.get("prefetch_factor", 2))
    return DataLoader(
        dataset_object,
        **loader_kwargs,
    )
