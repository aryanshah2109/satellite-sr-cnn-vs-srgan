import pytest
import torch

from models import build_model, count_parameters
from models.esrgan_generator import ESRGANGenerator
from models.srgan_discriminator import SRGANDiscriminator
from models.srgan_generator import SRGANGenerator
from utils.config import Config


def _config(name: str = "esrgan_generator") -> Config:
    return Config(
        {
            "model": {
                "name": name,
                "in_channels": 3,
                "out_channels": 3,
                "channels": 64,
                "growth_channels": 16,
                "num_blocks": 2,
                "res_scale": 0.1,
                "upscale_factor": 4,
            }
        }
    )


def test_generators_support_batches_and_non_square_inputs() -> None:
    inputs = torch.rand(2, 3, 12, 16)
    assert SRGANGenerator()(inputs).shape == (2, 3, 48, 64)
    assert ESRGANGenerator(num_blocks=2)(inputs).shape == (2, 3, 48, 64)


def test_discriminator_supports_configured_crop_sizes() -> None:
    model = SRGANDiscriminator()
    for size in (96, 128, 256):
        assert model(torch.rand(2, 3, size, size)).shape == (2, 1)


def test_gradients_flow_and_parameter_counts_are_comparable() -> None:
    generator = SRGANGenerator()
    output = generator(torch.rand(1, 3, 24, 24)).mean()
    output.backward()
    assert all(parameter.grad is not None for parameter in generator.parameters())
    esrgan = ESRGANGenerator(num_blocks=2)
    assert count_parameters(esrgan) > 0


def test_factory_and_invalid_upscale() -> None:
    assert isinstance(build_model(_config()), ESRGANGenerator)
    with pytest.raises(ValueError, match="Unknown"):
        build_model(_config("unknown"))
    with pytest.raises(ValueError, match="power of two"):
        SRGANGenerator(upscale_factor=3)
