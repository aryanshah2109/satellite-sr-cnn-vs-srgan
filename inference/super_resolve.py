"""Load super-resolution models and run tiled x4 inference."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional

from models import build_model, count_parameters
from sr_datasets.raster_io import read_raster
from utils.checkpoint import load_checkpoint, load_pretrained_weights
from utils.config import Config, load_config
from utils.image_utils import get_device, load_image, pil_to_tensor, save_image
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, Tensor] | None:
    """Extract a bare state dictionary from common checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        return None
    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(value, Tensor) for value in checkpoint.values()):
        return checkpoint
    return None


def _load_bare_state_dict(path: Path, model: nn.Module, device: torch.device) -> None:
    """Load a bare state dictionary while normalizing common wrapper prefixes."""
    payload = torch.load(path, map_location=device, weights_only=True)
    state_dict = _state_dict_from_checkpoint(payload)
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        for prefix in ("module.", "_orig_mod."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized[normalized_key] = value
    model.load_state_dict(normalized, strict=True)


def load_model_for_inference(
    model_type: str,
    checkpoint: str | Path,
    model_config: str | Path,
    device: torch.device,
) -> nn.Module:
    """Build and load a generator from project or third-party weights."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    cfg = load_config(model_config)
    model = build_model(cfg).to(device)
    if model_type == "srgan_pretrained":
        load_pretrained_weights(checkpoint_path, model, strict=False)
    else:
        try:
            load_checkpoint(checkpoint_path, model, map_location=device, strict=True)
        except (KeyError, RuntimeError):
            _load_bare_state_dict(checkpoint_path, model, device)
    model.eval()
    LOGGER.info(
        "Loaded %s from %s (%d bytes, %d parameters)",
        model_type,
        checkpoint_path,
        checkpoint_path.stat().st_size,
        count_parameters(model),
    )
    return model


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return overlapping tile starts that cover one dimension."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _feather_window(
    height: int,
    width: int,
    overlap: int,
    top: int,
    left: int,
    full_height: int,
    full_width: int,
    device: torch.device,
) -> Tensor:
    """Create a separable linear feathering window for one output tile."""
    y = torch.ones(height, device=device)
    x = torch.ones(width, device=device)
    ramp = torch.linspace(0.0, 1.0, max(overlap, 1) + 2, device=device)[1:-1]
    if top > 0:
        y[: min(overlap, height)] = ramp[: min(overlap, height)]
    if top + height < full_height:
        y[-min(overlap, height) :] = torch.flip(ramp[: min(overlap, height)], (0,))
    if left > 0:
        x[: min(overlap, width)] = ramp[: min(overlap, width)]
    if left + width < full_width:
        x[-min(overlap, width) :] = torch.flip(ramp[: min(overlap, width)], (0,))
    return y[:, None] * x[None, :]


def super_resolve_tensor(
    model: nn.Module,
    lr: Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp: bool,
) -> Tensor:
    """Upscale CHW or BCHW [0, 1] tensors by x4 with feathered tiled inference."""
    if lr.ndim == 3:
        lr = lr.unsqueeze(0)
        squeeze = True
    elif lr.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Expected LR tensor with shape (C,H,W) or (B,C,H,W)")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be smaller than tile_size")
    lr = lr.to(device=device, dtype=torch.float32)
    scale = 4
    height, width = lr.shape[-2:]
    if height <= tile_size and width <= tile_size:
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            output = model(lr).float().clamp(0.0, 1.0)
        return output[0] if squeeze else output
    if height <= tile_size * 3 and width <= tile_size * 3:
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            output = model(lr).float().clamp(0.0, 1.0)
        return output[0] if squeeze else output
    batch_size, channels = lr.shape[:2]
    output = torch.zeros(batch_size, channels, height * scale, width * scale, device=device)
    weights = torch.zeros(1, 1, height * scale, width * scale, device=device)
    context = max(overlap, tile_size)
    padded = functional.pad(lr, (context, context, context, context), mode="reflect")
    for top in _tile_starts(height, tile_size, overlap):
        for left in _tile_starts(width, tile_size, overlap):
            tile_height = min(tile_size, height - top)
            tile_width = min(tile_size, width - left)
            tile = padded[
                :, :, top : top + tile_height + 2 * context,
                left : left + tile_width + 2 * context,
            ]
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                tile_output = model(tile).float().clamp(0.0, 1.0)
            crop = tile_output[
                :, :, context * scale : (context + tile_height) * scale,
                context * scale : (context + tile_width) * scale,
            ]
            window = _feather_window(
                tile_height, tile_width, overlap, top, left, height, width, device
            )
            window = window.repeat_interleave(scale, 0).repeat_interleave(scale, 1)[None, None]
            output[:, :, top * scale : (top + tile_height) * scale, left * scale : (left + tile_width) * scale] += crop * window
            weights[:, :, top * scale : (top + tile_height) * scale, left * scale : (left + tile_width) * scale] += window
    output = output / weights.clamp_min(torch.finfo(output.dtype).eps)
    return output[0] if squeeze else output


def _input_tensor(path: Path, cfg: Config) -> Tensor:
    """Read a supported image or configured raster into CHW [0, 1] data."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = read_raster(path, list(cfg.bands.indices), cfg.bands.max_value)
        return torch.from_numpy(array)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return pil_to_tensor(load_image(path))
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("srgan", "esrgan"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/srgan.yaml"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    """Run CLI inference for one image or a directory."""
    args = _parse_args()
    eval_cfg = load_config("configs/evaluation.yaml")
    setup_logging(eval_cfg, "inference")
    device = get_device(args.device or str(eval_cfg.device.preferred))
    inference_cfg = eval_cfg.inference
    tile_size = args.tile_size or int(inference_cfg.tile_size)
    overlap = args.overlap if args.overlap is not None else int(inference_cfg.overlap)
    model_config = args.model_config
    model = load_model_for_inference(args.model, args.checkpoint, model_config, device)
    inputs = [args.input] if args.input.is_file() else sorted(args.input.iterdir())
    if not inputs:
        raise ValueError(f"Input directory is empty: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    for input_path in inputs:
        start = time.perf_counter()
        tensor = _input_tensor(input_path, eval_cfg)
        output = super_resolve_tensor(model, tensor, tile_size, overlap, device, args.amp)
        destination = args.output / f"{input_path.stem}_x4.png"
        save_image(output, destination)
        LOGGER.info(
            "Saved %s in %.2fs with output size=%sx%s",
            destination,
            time.perf_counter() - start,
            output.shape[-1],
            output.shape[-2],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
