"""Shared training, validation, checkpoint, and experiment helpers."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from evaluation.metrics import LPIPSMetric, average_metric, psnr_y, ssim_y
from utils.checkpoint import save_checkpoint
from utils.config import Config, config_hash, save_config
from utils.env_info import collect_environment_info

LOGGER = logging.getLogger(__name__)


def evenly_spaced_indices(size: int, count: int) -> list[int]:
    """Return deterministic, approximately evenly spaced dataset indices."""
    if size <= 0 or count <= 0:
        return []
    requested = min(size, count)
    return torch.linspace(0, size - 1, requested).round().long().unique().tolist()


class RunningLosses:
    """Accumulate detached scalar tensors without synchronizing each iteration."""

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.count = 0

    def update(self, values: dict[str, torch.Tensor]) -> None:
        """Add detached scalar values on their current device."""
        for name, value in values.items():
            detached = value.detach()
            self.values[name] = self.values.get(name, torch.zeros_like(detached)) + detached
        self.count += 1

    def averages(self) -> dict[str, float]:
        """Return synchronized Python averages and reset the accumulator."""
        divisor = max(self.count, 1)
        result = {name: float(value.div(divisor).item()) for name, value in self.values.items()}
        self.values.clear()
        self.count = 0
        return result


def to_channels_last(module: nn.Module, enabled: bool) -> nn.Module:
    """Convert a module to channels-last convolution weights when enabled."""
    if enabled:
        module = module.to(memory_format=torch.channels_last)
    return module


def to_channels_last_input(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Convert BCHW input to channels-last memory format when enabled."""
    if enabled and tensor.ndim == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


class AverageMeter:
    """Track a weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset accumulated values."""
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        """Add a value with its sample count."""
        self.total += value * count
        self.count += count

    @property
    def average(self) -> float:
        """Return the running average."""
        return self.total / max(self.count, 1)


def build_optimizer(parameters: Any, cfg: Config) -> Optimizer:
    """Build an optimizer from an optimizer configuration mapping."""
    name = str(cfg.name).lower()
    kwargs = {
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
    }
    if name == "adam":
        return torch.optim.Adam(parameters, betas=tuple(cfg.betas), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, betas=tuple(cfg.betas), **kwargs)
    raise ValueError(f"Unsupported optimizer: {cfg.name}")


def build_scheduler(optimizer: Optimizer, cfg: Config, epochs: int) -> Any:
    """Build a per-epoch scheduler from configuration."""
    name = str(cfg.name).lower()
    params = cfg.params.to_dict()
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=int(params.get("t_max", epochs)))
    if name == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=[int(value) for value in params.get("milestones", [])],
            gamma=float(params.get("gamma", 0.1)),
        )
    if name == "steplr":
        return StepLR(
            optimizer,
            step_size=int(params["step_size"]),
            gamma=float(params["gamma"]),
        )
    raise ValueError(f"Unsupported scheduler: {cfg.name}")


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set ``requires_grad`` on every parameter in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = flag


def amp_context(device: torch.device, enabled: bool) -> Any:
    """Return a CPU-safe autocast context manager."""
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """Return a CPU-safe gradient scaler."""
    return torch.amp.GradScaler(device=device.type, enabled=enabled and device.type == "cuda")


def validate(
    generator: nn.Module,
    loader: Any,
    device: torch.device,
    metrics: Config,
    amp: bool,
    lpips_metric: LPIPSMetric | None = None,
) -> dict[str, float]:
    """Validate a generator and average PSNR, SSIM, and LPIPS metrics."""
    generator.eval()
    metric = lpips_metric or LPIPSMetric(str(metrics.lpips_net))
    meters = {name: AverageMeter() for name in ("psnr", "ssim", "lpips")}
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            with amp_context(device, amp):
                sr = generator(lr)
            values = {
                "psnr": psnr_y(sr.float(), hr.float(), int(metrics.border)),
                "ssim": ssim_y(
                    sr.float(),
                    hr.float(),
                    int(metrics.border),
                    int(metrics.ssim_window_size),
                    float(metrics.ssim_sigma),
                ),
                "lpips": metric(sr.float(), hr.float()),
            }
            for name, value in values.items():
                meters[name].update(average_metric(value), int(lr.shape[0]))
    generator.train()
    return {name: meter.average for name, meter in meters.items()}


def create_writer(cfg: Config, run_name: str) -> SummaryWriter:
    """Create a TensorBoard writer under the configured runs directory."""
    return SummaryWriter(log_dir=str(Path(cfg.paths.runs_dir) / run_name))


class ThroughputTimer:
    """Track iterations per second and estimated remaining time."""

    def __init__(self, total_iters: int) -> None:
        self.total_iters = total_iters
        self.start = time.perf_counter()

    def update(self, completed_iters: int) -> tuple[float, float]:
        """Return throughput and ETA in seconds."""
        elapsed = max(time.perf_counter() - self.start, 1e-9)
        rate = completed_iters / elapsed
        eta = max(self.total_iters - completed_iters, 0) / max(rate, 1e-9)
        return rate, eta


def setup_run(cfg: Config, run_name: str, model_name: str) -> Path:
    """Create run/checkpoint directories and save config and environment metadata."""
    run_dir = Path(cfg.paths.runs_dir) / run_name
    checkpoint_dir = Path(cfg.paths.checkpoints_dir) / model_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "environment.json").write_text(
        json.dumps(collect_environment_info(), indent=2, default=str), encoding="utf-8"
    )
    LOGGER.info("Run %s config hash=%s", run_name, config_hash(cfg))
    LOGGER.info("Environment=%s", collect_environment_info())
    return checkpoint_dir


def append_experiment_log(
    cfg: Config,
    run_name: str,
    stage: str,
    result: dict[str, float | int],
) -> None:
    """Append a compact completed-run record to the experiment log."""
    environment = collect_environment_info()
    gpu_name = environment["gpus"][0]["name"] if environment["gpus"] else "CPU"
    metric_name = "lpips" if stage == "gan" else "psnr"
    metric_value = result.get(metric_name, "n/a")
    line = (
        f"- {run_name} | hash={config_hash(cfg)} | stage={stage} | "
        f"epochs={result.get('epochs_done', 'n/a')} | {metric_name}={metric_value} | "
        f"time_s={result.get('elapsed_seconds', 'n/a')} | GPU={gpu_name}\n"
    )
    path = Path(cfg.paths.docs_dir) / "experiment_log.md"
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(line)


def append_experiment_history(
    cfg: Config,
    run_name: str,
    stage: str,
    result: dict[str, float | int],
) -> None:
    """Append a structured successful-run record to the JSON history."""
    environment = collect_environment_info()
    gpu_name = environment["gpus"][0]["name"] if environment["gpus"] else "CPU"
    path = Path(cfg.paths.docs_dir) / "experiment_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", encoding="utf-8") as history_file:
            history = json.load(history_file)
        if not isinstance(history, list):
            raise ValueError(f"Experiment history must contain a JSON list: {path}")
    else:
        history = []

    record = {
        "run_number": len(history) + 1,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "success",
        "run_name": run_name,
        "stage": stage,
        "config_hash": config_hash(cfg),
        "epochs": result.get("epochs_done"),
        "metrics": {
            name: value
            for name, value in result.items()
            if name not in {"epochs_done", "elapsed_seconds"}
        },
        "elapsed_seconds": result.get("elapsed_seconds"),
        "gpu": gpu_name,
    }
    history.append(record)

    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float | None,
    metrics: dict[str, float],
    cfg: Config,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save project checkpoint plus scheduler and scaler state."""
    payload = dict(extra or {})
    payload.update(
        {
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
    )
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=epoch,
        best_metric=best_metric,
        metrics=metrics,
        cfg=cfg,
        extra=payload,
    )
