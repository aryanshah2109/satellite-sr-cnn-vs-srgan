from pathlib import Path

import pytest
import torch

from models.srgan_generator import SRGANGenerator
from utils.checkpoint import load_pretrained_weights

CHECKPOINT = Path("checkpoints/pretrained/SRGAN_x4-ImageNet.pth.tar")


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="reference checkpoint is absent")
def test_real_generator_loads_strictly_and_runs() -> None:
    model = SRGANGenerator()
    result = load_pretrained_weights(CHECKPOINT, model, strict=True)
    assert result == {"missing_keys": [], "unexpected_keys": []}
    inputs = torch.rand(1, 3, 24, 24)
    first = model(inputs)
    second = model(inputs)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


def test_pretrained_loader_strips_module_prefix(tmp_path) -> None:
    model = torch.nn.Conv2d(3, 3, 1)
    path = tmp_path / "module.pt"
    torch.save(
        {
            "state_dict": {
                f"module.{key}": value for key, value in model.state_dict().items()
            }
        },
        path,
    )
    loaded = torch.nn.Conv2d(3, 3, 1)
    result = load_pretrained_weights(path, loaded, strict=True)
    assert result == {"missing_keys": [], "unexpected_keys": []}
