import json

import pytest

from sr_datasets.splits import create_splits, load_split
from utils.config import Config


def _config(tmp_path, seed=42) -> Config:
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    tiles = []
    for city in ("alpha", "beta"):
        for index in range(1, 7):
            tiles.append({"tile_id": f"{city}{index}", "city": city})
    (processed / "dataset_report.json").write_text(json.dumps({"tiles": tiles}))
    return Config(
        {
            "paths": {
                "splits_dir": str(tmp_path / "splits"),
                "processed_dir": str(processed),
            },
            "split": {
                "seed": seed,
                "train_ratio": 0.5,
                "val_ratio": 0.25,
                "test_ratio": 0.25,
            },
        }
    )


def test_splits_are_reproducible_stratified_and_disjoint(tmp_path) -> None:
    cfg = _config(tmp_path)
    first = create_splits(cfg)
    second = create_splits(cfg)
    assert first == second
    assert set().union(*map(set, first.values())) == {
        f"{c}{i}" for c in ("alpha", "beta") for i in range(1, 7)
    }
    assert not (set(first["train"]) & set(first["val"]))
    assert all(
        any(tile.startswith(city) for tile in first[name])
        for city in ("alpha", "beta")
        for name in ("train", "val", "test")
    )
    assert load_split(cfg, "train") == first["train"]


def test_split_hash_mismatch_requires_force(tmp_path) -> None:
    cfg = _config(tmp_path)
    create_splits(cfg)
    with cfg.unlocked():
        cfg.split.val_ratio = 0.2
        cfg.split.test_ratio = 0.3
    with pytest.raises(ValueError, match="config hash"):
        create_splits(cfg)
