"""Evaluate super-resolution methods on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.nn import functional

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.metrics import LPIPSMetric, psnr_y, ssim_y
from inference.super_resolve import load_model_for_inference
from sr_datasets.sr_dataset import build_dataloader
from utils.checkpoint import find_latest_checkpoint
from utils.config import Config, config_hash, load_config
from utils.image_utils import get_device
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)
RESULT_COLUMNS = (
    "method",
    "split",
    "num_images",
    "mean_psnr_y",
    "std_psnr_y",
    "mean_ssim_y",
    "std_ssim_y",
    "mean_lpips",
    "std_lpips",
)


def _git_commit() -> str | None:
    """Return the current git commit when available."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or None


def _checkpoint_path(cfg: Config, method: Config) -> Path | None:
    """Resolve explicit or automatic method checkpoint paths."""
    checkpoint = str(method.get("checkpoint", ""))
    if checkpoint != "auto":
        return Path(checkpoint)
    method_type = str(method.type)
    family = "cnn" if method_type == "cnn" else "srgan"
    root = Path(cfg.paths.checkpoints_dir) / family
    run_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True) if root.is_dir() else []
    preferred = ("best.pt", "best_generator.pth", "best_gan.pt")
    for run_dir in run_dirs:
        for filename in preferred:
            candidate = run_dir / filename
            if candidate.is_file():
                return candidate
    return find_latest_checkpoint(root)


def _city_from_name(name: str, pattern: str) -> tuple[str, str]:
    """Extract city and tile identifier from a patch name."""
    stem = Path(name).stem
    match = re.match(pattern, stem.split("_y", 1)[0].split("_x", 1)[0])
    if match is None:
        return "unknown", stem
    return match.group("city"), f"{match.group('city')}{match.group('index')}"


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    """Return a stable mean and sample standard deviation."""
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float, float]:
    """Return paired mean difference, confidence interval, and win rate."""
    difference = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(resamples, len(difference)))
    boot_means = difference[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot_means, [tail, 1.0 - tail])
    if metric == "lpips":
        win_rate = float(np.mean(first < second))
    else:
        win_rate = float(np.mean(first > second))
    return float(difference.mean()), float(low), float(high), win_rate


class Evaluator:
    """Evaluate configured super-resolution methods on identical test batches."""

    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.device = get_device(args.device or str(cfg.device.preferred))
        evaluation = cfg.evaluation
        self.output_dir = Path(args.output_dir or evaluation.output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.lpips_metric = LPIPSMetric(str(evaluation.lpips_net))
        self.checkpoint_overrides = {
            value.split("=", 1)[0]: Path(value.split("=", 1)[1])
            for value in args.checkpoint
            if "=" in value
        }
        self.loader = build_dataloader(
            cfg,
            str(evaluation.split),
            batch_size=int(evaluation.batch_size),
            num_workers=int(evaluation.num_workers),
            shuffle=False,
        )
        self.methods: dict[str, tuple[Config, Path | None, torch.nn.Module | None]] = {}

    def _load_methods(self) -> None:
        """Resolve and load every available configured method."""
        requested = set(self.args.methods.split(",")) if self.args.methods else None
        for method in self.cfg.methods:
            name = str(method.name)
            if requested is not None and name not in requested:
                continue
            if str(method.type) == "bicubic":
                self.methods[name] = (method, None, None)
                continue
            checkpoint = self.checkpoint_overrides.get(name, _checkpoint_path(self.cfg, method))
            if checkpoint is None or not checkpoint.is_file():
                message = f"Skipping {name}; checkpoint is missing: {checkpoint}"
                if bool(self.cfg.evaluation.skip_missing):
                    LOGGER.warning(message)
                    continue
                raise FileNotFoundError(message)
            model = load_model_for_inference(
                str(method.type), checkpoint, str(method.model_config), self.device
            )
            self.methods[name] = (method, checkpoint, model)

    def evaluate(self) -> dict[str, list[dict[str, Any]]]:
        """Evaluate all available methods and write metric artifacts."""
        self._load_methods()
        records: dict[str, list[dict[str, Any]]] = {name: [] for name in self.methods}
        limit = self.args.max_samples
        processed = 0
        start_time = time.perf_counter()
        with torch.no_grad():
            for batch in self.loader:
                if limit is not None and processed >= limit:
                    break
                lr = batch["lr"].to(self.device)
                hr = batch["hr"].to(self.device)
                names = list(batch["name"])
                if limit is not None:
                    keep = min(len(names), limit - processed)
                    lr, hr, names = lr[:keep], hr[:keep], names[:keep]
                outputs: dict[str, torch.Tensor] = {}
                for name, (_, _, model) in self.methods.items():
                    if model is None:
                        outputs[name] = functional.interpolate(
                            lr,
                            scale_factor=4,
                            mode="bicubic",
                            align_corners=False,
                        ).clamp(0.0, 1.0)
                    else:
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.float16,
                            enabled=bool(self.cfg.evaluation.amp) and self.device.type == "cuda",
                        ):
                            outputs[name] = model(lr).float().clamp(0.0, 1.0)
                batch_metrics = {}
                for name, output in outputs.items():
                    batch_metrics[name] = {
                        "psnr_y": psnr_y(output, hr, int(self.cfg.evaluation.border))
                        .detach()
                        .cpu()
                        .tolist(),
                        "ssim_y": ssim_y(output, hr, int(self.cfg.evaluation.border))
                        .detach()
                        .cpu()
                        .tolist(),
                        "lpips": self.lpips_metric(output.float(), hr.float())
                        .detach()
                        .cpu()
                        .tolist(),
                    }
                for index, patch_name in enumerate(names):
                    for name in outputs:
                        values = {
                            metric: float(batch_metrics[name][metric][index])
                            for metric in ("psnr_y", "ssim_y", "lpips")
                        }
                        city, tile_id = _city_from_name(
                            patch_name, str(self.cfg.split.city_regex)
                        )
                        records[name].append(
                            {
                                "method": name,
                                "patch_name": patch_name,
                                "tile_id": tile_id,
                                "city": city,
                                **values,
                            }
                        )
                processed += len(names)
        self._write_metrics(records, processed, time.perf_counter() - start_time)
        if self.args.visual:
            from evaluation.visual_comparison import make_comparisons

            make_comparisons(self.cfg, self.methods, self.output_dir / "comparisons", self.metrics_dir / "per_image.csv")
        if self.args.update_readme:
            self._update_readme()
        return records

    def _write_metrics(
        self,
        records: dict[str, list[dict[str, Any]]],
        sample_count: int,
        elapsed: float,
    ) -> None:
        """Write tables, significance results, metadata, and plots."""
        rows = []
        for name, values in records.items():
            metrics = {metric: np.asarray([item[metric] for item in values]) for metric in ("psnr_y", "ssim_y", "lpips")}
            means = {metric: _mean_std(array) for metric, array in metrics.items()}
            rows.append({"method": name, "split": str(self.cfg.evaluation.split), "num_images": len(values), "mean_psnr_y": means["psnr_y"][0], "std_psnr_y": means["psnr_y"][1], "mean_ssim_y": means["ssim_y"][0], "std_ssim_y": means["ssim_y"][1], "mean_lpips": means["lpips"][0], "std_lpips": means["lpips"][1]})
        with (self.metrics_dir / "results.csv").open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        markdown = [
            "| method | split | num_images | PSNR_Y (mean +/- std) | SSIM_Y (mean +/- std) | LPIPS (mean +/- std) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            markdown.append(
                f"| {row['method']} | {row['split']} | {row['num_images']} | "
                f"{row['mean_psnr_y']:.4f} +/- {row['std_psnr_y']:.4f} | "
                f"{row['mean_ssim_y']:.4f} +/- {row['std_ssim_y']:.4f} | "
                f"{row['mean_lpips']:.4f} +/- {row['std_lpips']:.4f} |"
            )
        (self.metrics_dir / "results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
        per_image_fields = ("method", "patch_name", "tile_id", "city", "psnr_y", "ssim_y", "lpips")
        with (self.metrics_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=per_image_fields)
            writer.writeheader()
            for values in records.values():
                writer.writerows(values)
        city_rows = []
        for name, values in records.items():
            cities = sorted({item["city"] for item in values})
            for city in cities:
                selected = [item for item in values if item["city"] == city]
                city_rows.append({"method": name, "city": city, "num_images": len(selected), **{f"mean_{metric}": float(np.mean([item[metric] for item in selected])) for metric in ("psnr_y", "ssim_y", "lpips")}})
        with (self.metrics_dir / "per_city.csv").open("w", newline="", encoding="utf-8") as output_file:
            fields = ("method", "city", "num_images", "mean_psnr_y", "mean_ssim_y", "mean_lpips")
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(city_rows)
        by_method = {name: {item["patch_name"]: item for item in values} for name, values in records.items()}
        significance = []
        for pair in self.cfg.comparisons:
            first_name, second_name = str(pair[0]), str(pair[1])
            if first_name not in by_method or second_name not in by_method:
                LOGGER.warning("Skipping comparison %s vs %s; method is missing", first_name, second_name)
                continue
            common = sorted(set(by_method[first_name]) & set(by_method[second_name]))
            for metric_index, metric in enumerate(("psnr_y", "ssim_y", "lpips")):
                first = np.asarray([by_method[first_name][name][metric] for name in common])
                second = np.asarray([by_method[second_name][name][metric] for name in common])
                difference, low, high, win_rate = _bootstrap_difference(first, second, metric, int(self.cfg.bootstrap.num_resamples), float(self.cfg.bootstrap.confidence), int(self.cfg.bootstrap.seed) + metric_index)
                significance.append({"method_a": first_name, "method_b": second_name, "metric": metric, "num_images": len(common), "mean_difference_a_minus_b": difference, "ci_low": low, "ci_high": high, "win_rate": win_rate})
        with (self.metrics_dir / "significance.csv").open("w", newline="", encoding="utf-8") as output_file:
            fields = ("method_a", "method_b", "metric", "num_images", "mean_difference_a_minus_b", "ci_low", "ci_high", "win_rate")
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(significance)
        self._write_plots(rows, records)
        checkpoint_meta = {name: {"path": str(path), "size": path.stat().st_size} for name, (_, path, _) in self.methods.items() if path is not None}
        metadata = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config_hash": config_hash(self.cfg), "git_commit": _git_commit(), "device": str(self.device), "torch_version": str(torch.__version__), "sample_count": sample_count, "elapsed_seconds": elapsed, "checkpoints": checkpoint_meta, "metric_notes": "PSNR_Y and SSIM_Y use plain luma coefficients [0.299, 0.587, 0.114], border 4; LPIPS uses AlexNet."}
        (self.metrics_dir / "eval_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Evaluation complete: samples=%s elapsed=%.2fs", sample_count, elapsed)
        LOGGER.info("Summary: %s", rows)

    def _write_plots(self, rows: list[dict[str, Any]], records: dict[str, list[dict[str, Any]]]) -> None:
        """Write the requested evaluation plots."""
        names = [row["method"] for row in rows]
        positions = np.arange(len(names))
        figure, axis = plt.subplots(figsize=(12, 6))
        for offset, metric in enumerate(("mean_psnr_y", "mean_ssim_y", "mean_lpips")):
            axis.bar(positions + (offset - 1) * 0.25, [row[metric] for row in rows], 0.25, label=metric)
        axis.set_xticks(positions, names, rotation=45, ha="right")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.plots_dir / "metrics_by_method.png", dpi=150)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(8, 6))
        for name, values in records.items():
            axis.scatter([item["psnr_y"] for item in values], [item["lpips"] for item in values], label=name, alpha=0.5)
        axis.set_xlabel("PSNR_Y")
        axis.set_ylabel("LPIPS")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.plots_dir / "perception_distortion.png", dpi=150)
        plt.close(figure)

    def _update_readme(self) -> None:
        """Replace the README results block with the generated Markdown table."""
        readme = Path("README.md")
        if not readme.is_file():
            return
        text = readme.read_text(encoding="utf-8")
        start_marker = "<!-- RESULTS_TABLE_START -->"
        end_marker = "<!-- RESULTS_TABLE_END -->"
        table = (self.metrics_dir / "results.md").read_text(encoding="utf-8").strip()
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start >= 0 and end > start:
            replacement = f"{start_marker}\n{table}\n{end_marker}"
            readme.write_text(text[:start] + replacement + text[end + len(end_marker) :], encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--methods", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run configured evaluation."""
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg, "evaluation")
    Evaluator(cfg, args).evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
