"""Atomic checkpoint persistence and third-party weight loading utilities."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from utils.config import Config, config_hash

LOGGER = logging.getLogger(__name__)


def _git_commit() -> str | None:
    """Read the current git commit without failing outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    value = completed.stdout.strip()
    return value or None


def _config_value(cfg: Config | Mapping[str, Any], key: str, default: Any) -> Any:
    """Read a top-level config value from either supported config type."""
    if isinstance(cfg, Config):
        return cfg[key] if key in cfg else default

    return cfg.get(key, default)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write a torch payload atomically in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    except OSError as error:
        raise OSError(
            f"Could not atomically save checkpoint to {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int,
    best_metric: float | None,
    metrics: Mapping[str, float],
    cfg: Config | Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save a model checkpoint with reproducibility metadata atomically.

    Args:
        path: Destination checkpoint path.
        model: Model whose state should be saved.
        optimizer: Optional optimizer state to save.
        epoch: Completed epoch number.
        best_metric: Best validation metric known at save time.
        metrics: Metrics for this checkpoint.
        cfg: Resolved experiment configuration.
        extra: Optional additional metadata.

    Returns:
        Destination path.
    """
    destination = Path(path)
    metadata = {
        "epoch": epoch,
        "best_metric": best_metric,
        "metrics": dict(metrics),
        "config_hash": config_hash(cfg),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(UTC).isoformat(),
        "torch_version": str(torch.__version__),
    }
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }

    checkpoint_config = _config_value(cfg, "checkpoint", {})
    save_optimizer = (
        checkpoint_config.save_optimizer
        if isinstance(checkpoint_config, Config)
        else checkpoint_config.get("save_optimizer", True)
    )
    if optimizer is not None and save_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if extra is not None:
        payload["extra"] = dict(extra)

    _atomic_torch_save(payload, destination)
    LOGGER.info("Saved checkpoint: %s", destination)

    return destination


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a project checkpoint into a model and optional optimizer.

    Args:
        path: Checkpoint path.
        model: Model to populate.
        optimizer: Optional optimizer to populate.
        map_location: Torch map location.
        strict: Whether model state keys must match exactly.

    Returns:
        Checkpoint metadata.

    Raises:
        FileNotFoundError: If the checkpoint is absent.
        KeyError: If required checkpoint keys are absent.
        RuntimeError: If state keys mismatch in strict mode.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        payload = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load checkpoint {checkpoint_path}: {error}"
        ) from error

    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise KeyError(f"Missing model_state_dict in checkpoint: {checkpoint_path}")

    incompatibility = model.load_state_dict(payload["model_state_dict"], strict=strict)
    if not strict and (incompatibility.missing_keys or incompatibility.unexpected_keys):
        LOGGER.warning(
            "Checkpoint mismatch; missing=%s unexpected=%s",
            incompatibility.missing_keys,
            incompatibility.unexpected_keys,
        )

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping")

    result = dict(metadata)
    if isinstance(payload.get("extra"), Mapping):
        result["extra"] = dict(payload["extra"])
    return result


def load_pretrained_weights(
    path: str | Path,
    model: torch.nn.Module,
    strict: bool = False,
) -> dict[str, list[str]]:
    """Load common third-party state-dict layouts into a model.

    Args:
        path: Third-party checkpoint path.
        model: Model to populate.
        strict: Whether all keys must match.

    Returns:
        Missing and unexpected model key names.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained checkpoint does not exist: {checkpoint_path}"
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict: Any = payload

    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in payload and isinstance(payload[key], Mapping):
                state_dict = payload[key]
                break

    if not isinstance(state_dict, Mapping):
        raise ValueError("Pretrained checkpoint does not contain a state dictionary")

    normalized_state = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        for prefix in ("module.", "_orig_mod."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix) :]
        normalized_state[normalized_key] = value
    incompatibility = model.load_state_dict(normalized_state, strict=strict)
    result = {
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
    }
    LOGGER.info(
        "Loaded pretrained weights; missing=%s unexpected=%s",
        result["missing_keys"],
        result["unexpected_keys"],
    )

    return result


def find_latest_checkpoint(
    directory: str | Path,
    patterns: tuple[str, ...] = ("*.pt", "*.pth"),
) -> Path | None:
    """Return the most recently modified checkpoint in a directory."""
    checkpoint_dir = Path(directory)
    if not checkpoint_dir.is_dir():
        return None

    candidates = []
    for pattern in patterns:
        candidates.extend(path for path in checkpoint_dir.glob(pattern) if path.is_file())
    candidates.sort(key=lambda path: path.stat().st_mtime)

    return candidates[-1] if candidates else None


def prune_old_checkpoints(directory: str | Path, keep_last_n: int) -> list[Path]:
    """Delete older ``.pt`` checkpoints and return deleted paths.

    Args:
        directory: Checkpoint directory.
        keep_last_n: Number of newest checkpoints to retain.

    Raises:
        ValueError: If ``keep_last_n`` is negative.
    """
    if keep_last_n < 0:
        raise ValueError("keep_last_n must be non-negative")

    checkpoint_dir = Path(directory)
    checkpoints = sorted(
        (path for path in checkpoint_dir.glob("*.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    deleted: list[Path] = []

    for checkpoint_path in checkpoints[keep_last_n:]:
        checkpoint_path.unlink()
        deleted.append(checkpoint_path)

    return deleted


def save_if_best(
    path: str | Path,
    metric: float,
    best_metric: float | None,
    model: torch.nn.Module,
    cfg: Config | Mapping[str, Any],
    *,
    epoch: int,
    metrics: Mapping[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
    maximize: bool = True,
) -> bool:
    """Save a checkpoint only when the supplied metric improves.

    Args:
        path: Destination checkpoint path.
        metric: Current validation metric.
        best_metric: Previous best metric, or ``None``.
        model: Model to save.
        cfg: Experiment configuration.
        epoch: Epoch associated with the metric.
        metrics: Metrics to store.
        optimizer: Optional optimizer.
        extra: Optional metadata.
        maximize: Whether larger metric values are better.

    Returns:
        Whether a checkpoint was saved.
    """
    improved = best_metric is None or (
        metric > best_metric if maximize else metric < best_metric
    )
    if not improved:
        return False

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=epoch,
        best_metric=metric,
        metrics=metrics,
        cfg=cfg,
        extra=extra,
    )

    return True
