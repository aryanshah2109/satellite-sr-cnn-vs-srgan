"""Create qualitative comparison figures from evaluation outputs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.nn import functional

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr_datasets.sr_dataset import SRDataset

LOGGER = logging.getLogger(__name__)


def _image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW image tensor into an HWC NumPy image."""
    return tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _select_zoom(reference: torch.Tensor, size: int, mode: str) -> tuple[int, int]:
    """Select a zoom box from the reference image."""
    height, width = reference.shape[-2:]
    if mode == "center":
        return max((height - size) // 2, 0), max((width - size) // 2, 0)
    gradient = reference[:, 1:, :] - reference[:, :-1, :]
    energy = gradient.abs().mean(dim=0)
    row, col = torch.unravel_index(torch.argmax(energy), energy.shape)
    return max(min(int(row) - size // 2, height - size), 0), max(min(int(col) - size // 2, width - size), 0)


def _figure_for_patch(
    cfg: Any,
    methods: dict[str, tuple[Any, Path | None, torch.nn.Module | None]],
    patch_name: str,
    lr: torch.Tensor,
    hr: torch.Tensor,
    metrics: dict[str, dict[str, float]],
    destination: Path,
    footer: str,
) -> None:
    """Save one qualitative comparison figure."""
    device = next(
        (next(model.parameters()).device for _, _, model in methods.values() if model is not None),
        torch.device("cpu"),
    )
    outputs: dict[str, torch.Tensor] = {"LR": functional.interpolate(lr[None], scale_factor=4, mode="nearest")[0]}
    for name, (_, _, model) in methods.items():
        if model is None:
            outputs[name] = functional.interpolate(lr[None], scale_factor=4, mode="bicubic", align_corners=False)[0]
        else:
            with torch.no_grad():
                outputs[name] = model(lr[None].to(device)).float().clamp(0.0, 1.0)[0].cpu()
    outputs["HR"] = hr
    names = list(outputs)
    box_top, box_left = _select_zoom(hr, int(cfg.visual.zoom_box_size), str(cfg.visual.zoom_select))
    columns = len(names)
    rows = 3 if bool(cfg.visual.error_map) else 2
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.atleast_2d(axes)
    for column, name in enumerate(names):
        axes[0, column].imshow(_image(outputs[name]))
        axes[0, column].set_title(name)
        axes[0, column].axis("off")
        zoom = outputs[name][..., box_top : box_top + int(cfg.visual.zoom_box_size), box_left : box_left + int(cfg.visual.zoom_box_size)]
        axes[1, column].imshow(_image(zoom))
        axes[1, column].axis("off")
        if name in metrics:
            axes[0, column].set_title(
                f"{name}\nPSNR {float(metrics[name]['psnr_y']):.2f} "
                f"LPIPS {float(metrics[name]['lpips']):.3f}"
            )
        if rows == 3:
            error = (outputs[name] - hr).abs().mean(dim=0) if name not in {"LR", "HR"} else torch.zeros_like(hr[0])
            axes[2, column].imshow(error.numpy(), cmap="magma", vmin=0.0, vmax=0.5)
            axes[2, column].axis("off")
    figure.suptitle(f"{patch_name} — {footer}")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(cfg.visual.dpi))
    plt.close(figure)


def make_comparisons(
    cfg: Any,
    methods: dict[str, tuple[Any, Path | None, torch.nn.Module | None]],
    out_dir: Path,
    per_image_csv: Path,
) -> None:
    """Create random, SRGAN-win, and SRGAN-fail comparison groups."""
    rows: list[dict[str, str]] = []
    with per_image_csv.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    by_patch: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_patch.setdefault(row["patch_name"], {})[row["method"]] = row
    rng = np.random.default_rng(int(cfg.visual.selection_seed))
    patch_names = sorted(by_patch)
    random_names = list(rng.choice(patch_names, min(int(cfg.visual.num_random), len(patch_names)), replace=False))
    reference_a, reference_b = [str(value) for value in cfg.visual.reference_methods]
    common = [name for name in patch_names if reference_a in by_patch[name] and reference_b in by_patch[name]]
    wins = sorted(common, key=lambda name: float(by_patch[name][reference_a]["lpips"]) - float(by_patch[name][reference_b]["lpips"]), reverse=True)[: int(cfg.visual.num_wins)]
    fails = sorted(common, key=lambda name: float(by_patch[name][reference_a]["psnr_y"]) - float(by_patch[name][reference_b]["psnr_y"]))[: int(cfg.visual.num_fails)]
    dataset = SRDataset(cfg, str(cfg.evaluation.split), "test")
    samples = {dataset.rows[index]["patch_name"]: dataset[index] for index in range(len(dataset)) if dataset.rows[index]["patch_name"] in set(random_names + wins + fails)}
    metric_lookup = {name: {item["patch_name"]: item for item in values} for name, values in ((method, []) for method in methods)}
    for row in rows:
        metric_lookup.setdefault(row["method"], {})[row["patch_name"]] = row
    for group, selected in (("random", random_names), ("srgan_wins", wins), ("srgan_fails", fails)):
        footer = "representative random sample" if group == "random" else "artifact-analysis group; not representative"
        for patch_name in selected:
            sample = samples[patch_name]
            metrics = {name: metric_lookup[name][patch_name] for name in methods if patch_name in metric_lookup.get(name, {})}
            destination = out_dir / group / f"{Path(patch_name).stem}.png"
            _figure_for_patch(cfg, methods, patch_name, sample["lr"], sample["hr"], metrics, destination, footer)
    LOGGER.info("Saved visual comparison groups under %s", out_dir)
