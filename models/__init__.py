"""Model factories and summary utilities."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

from models.cnn_sr import CNNSuperResolution
from models.srgan_discriminator import SRGANDiscriminator
from models.srgan_generator import SRGANGenerator
from utils.config import Config

LOGGER = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _model_values(cfg: Config) -> dict[str, Any]:
    """Return model configuration values as a plain mapping."""
    return cfg.model.to_dict()


def build_model(cfg: Config) -> nn.Module:
    """Build the configured CNN or SRGAN generator."""
    values = _model_values(cfg)
    name = str(values.pop("name"))
    ignored_keys = (
        "input_range",
        "output_range",
        "pretrained_weights",
        "allow_weights_only_false",
        "dummy_input_size",
        "discriminator",
    )
    for key in ignored_keys:
        values.pop(key, None)
    if name in {"cnn", "cnn_sr", "edsr_lite"}:
        model = CNNSuperResolution(**values)
    elif name in {"srgan", "srgan_generator", "srresnet"}:
        model = SRGANGenerator(
            num_blocks=values.pop("num_blocks", values.pop("num_rcb", 16)),
            **values,
        )
    else:
        raise ValueError(f"Unknown generator model name: {name}")
    LOGGER.info("Built %s with %s trainable parameters", name, count_parameters(model))
    return model


def build_discriminator(cfg: Config) -> nn.Module:
    """Build the configured VGG-style discriminator."""
    values = _model_values(cfg)
    return SRGANDiscriminator(
        in_channels=int(values.get("in_channels", 3)),
        out_channels=int(values.get("out_channels", 1)),
        channels=int(values.get("channels", 64)),
    )


def log_model_summary(model: nn.Module, logger: logging.Logger = LOGGER) -> None:
    """Log parameter totals, top-level breakdown, and dummy output shape."""
    total = count_parameters(model)
    breakdown = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in model.named_children()
    }
    logger.info("Model parameters: %s", total)
    logger.info("Top-level parameter breakdown: %s", breakdown)
    logger.info("Model output shape is determined by the configured input shape")


__all__ = [
    "CNNSuperResolution",
    "SRGANDiscriminator",
    "SRGANGenerator",
    "build_discriminator",
    "build_model",
    "count_parameters",
    "log_model_summary",
]
