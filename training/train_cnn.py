"""Train the CNN super-resolution baseline."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from evaluation.metrics import LPIPSMetric
from losses.reconstruction import PixelLoss
from models import build_model
from sr_datasets.sr_dataset import build_dataloader
from training.trainer_utils import (
    RunningLosses,
    ThroughputTimer,
    amp_context,
    build_optimizer,
    build_scheduler,
    create_writer,
    evenly_spaced_indices,
    grad_scaler,
    append_experiment_log,
    append_experiment_history,
    save_training_checkpoint,
    set_requires_grad,
    setup_run,
    to_channels_last,
    to_channels_last_input,
    validate,
)
from utils.checkpoint import load_checkpoint
from utils.config import Config, load_config
from utils.env_info import collect_environment_info
from utils.image_utils import get_device
from utils.logger import setup_logging
from utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


class CNNTrainer:
    """Train and validate the CNN generator."""

    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.device = get_device(args.device or str(cfg.device.preferred))
        set_seed(int(cfg.seed), bool(cfg.deterministic))
        self.channels_last = bool(cfg.training.get("channels_last", False))
        self.model = to_channels_last(build_model(cfg).to(self.device), self.channels_last)
        set_requires_grad(self.model, True)
        self.optimizer = build_optimizer(self.model.parameters(), cfg.training.optimizer)
        self.scheduler = build_scheduler(
            self.optimizer, cfg.training.scheduler, int(cfg.training.epochs)
        )
        self.scaler = grad_scaler(self.device, bool(cfg.training.amp))
        self.criterion = PixelLoss(str(cfg.training.loss.kind)).to(self.device)
        self.lpips_metric = LPIPSMetric(str(cfg.training.metrics.lpips_net))
        self.train_loader = build_dataloader(cfg, "train")
        val_count = args.val_subset or int(cfg.training.val_subset_size)
        val_workers = int(cfg.training.get("val_num_workers", 0))
        val_dataset = build_dataloader(cfg, "val", num_workers=val_workers).dataset
        val_indices = evenly_spaced_indices(len(val_dataset), val_count)
        self.val_loader = build_dataloader(cfg, "val", val_indices, num_workers=val_workers)
        self.checkpoint_dir = setup_run(cfg, args.run_name, "cnn")
        self.writer = create_writer(cfg, args.run_name)
        self.start_epoch = 0
        self.best_metric: float | None = None
        self.global_step = 0
        if args.resume:
            self._resume(Path(args.resume))

    def _resume(self, path: Path) -> None:
        metadata = load_checkpoint(path, self.model, self.optimizer, self.device)
        extra = metadata.get("extra", {})
        if self.scheduler is not None and extra.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(extra["scheduler_state_dict"])
        if extra.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(extra["scaler_state_dict"])
        self.start_epoch = int(metadata.get("epoch", 0))
        self.best_metric = metadata.get("best_metric")
        self.global_step = int(extra.get("stage_step", extra.get("global_step", 0)))
        LOGGER.info("Resumed CNN training from %s", path)

    def train(self) -> dict[str, float | int]:
        """Run the configured bounded or full CNN training loop."""
        epochs = int(self.cfg.training.epochs)
        max_iters = self.args.max_iters or self.cfg.training.max_iters
        max_iters = None if max_iters is None else int(max_iters)
        timer = ThroughputTimer(max(len(self.train_loader) * epochs, 1))
        final_metrics: dict[str, float] = {}
        start_time = time.perf_counter()
        completed_epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, epochs):
                self.model.train()
                losses = RunningLosses()
                wait_total = 0.0
                compute_total = 0.0
                batch_iterator = iter(self.train_loader)
                while True:
                    if max_iters is not None and self.global_step >= max_iters:
                        break
                    wait_start = time.perf_counter()
                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        break
                    wait_total += time.perf_counter() - wait_start
                    compute_start = time.perf_counter()
                    lr = to_channels_last_input(
                        batch["lr"].to(self.device, non_blocking=True), self.channels_last
                    )
                    hr = to_channels_last_input(
                        batch["hr"].to(self.device, non_blocking=True), self.channels_last
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    with amp_context(self.device, bool(self.cfg.training.amp)):
                        loss = self.criterion(self.model(lr), hr)
                    self.scaler.scale(loss).backward()
                    if self.cfg.training.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), float(self.cfg.training.grad_clip_norm)
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    compute_total += time.perf_counter() - compute_start
                    losses.update({"loss": loss})
                    self.global_step += 1
                    rate, eta = timer.update(self.global_step)
                    if self.global_step % int(self.cfg.training.log_every_n_iters) == 0:
                        values = losses.averages()
                        log_window = int(self.cfg.training.log_every_n_iters)
                        LOGGER.info(
                            "epoch=%s step=%s loss=%.6f lr=%.6g it/s=%.3f eta=%.1fs "
                            "data_wait=%.4fs compute=%.4fs",
                            epoch + 1,
                            self.global_step,
                            values["loss"],
                            self.optimizer.param_groups[0]["lr"],
                            rate,
                            eta,
                            wait_total / log_window,
                            compute_total / log_window,
                        )
                        self.writer.add_scalar("train/loss", values["loss"], self.global_step)
                        self.writer.add_scalar("train/it_per_sec", rate, self.global_step)
                        wait_total = 0.0
                        compute_total = 0.0
                completed_epoch = epoch + 1
                if self.scheduler is not None:
                    self.scheduler.step()
                if completed_epoch % int(self.cfg.training.val_every_n_epochs) == 0:
                    final_metrics = validate(
                        self.model,
                        self.val_loader,
                        self.device,
                        self.cfg.training.metrics,
                        bool(self.cfg.training.amp),
                        self.lpips_metric,
                    )
                    LOGGER.info("epoch=%s validation=%s", completed_epoch, final_metrics)
                    for name, value in final_metrics.items():
                        self.writer.add_scalar(f"val/{name}", value, completed_epoch)
                    metric = final_metrics["psnr"]
                    if self.best_metric is None or metric > self.best_metric:
                        self.best_metric = metric
                        save_training_checkpoint(
                            self.checkpoint_dir / "best.pt",
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.scaler,
                            completed_epoch,
                            self.best_metric,
                            final_metrics,
                            self.cfg,
                            {"stage": "cnn", "stage_step": self.global_step},
                        )
                if completed_epoch % int(self.cfg.training.save_every_n_epochs) == 0:
                    save_training_checkpoint(
                        self.checkpoint_dir / "last.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        completed_epoch,
                        self.best_metric,
                        final_metrics,
                        self.cfg,
                        {"stage": "cnn", "stage_step": self.global_step},
                    )
                if max_iters is not None and self.global_step >= max_iters:
                    break
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted; saving last CNN checkpoint")
            save_training_checkpoint(
                self.checkpoint_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                completed_epoch,
                self.best_metric,
                final_metrics,
                self.cfg,
                {"stage": "cnn", "stage_step": self.global_step},
            )
        finally:
            self.writer.close()
        elapsed = time.perf_counter() - start_time
        info = collect_environment_info()
        gpu_name = info["gpus"][0]["name"] if info["gpus"] else "CPU"
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        LOGGER.info(
            "CNN training complete in %.2fs; GPU=%s; peak_gpu_memory_mb=%.2f",
            elapsed,
            gpu_name,
            peak_memory,
        )
        return {**final_metrics, "epochs_done": completed_epoch, "elapsed_seconds": elapsed}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cnn.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Run CNN training."""
    args = _parse_args()
    cfg = load_config(args.config, {"seed": args.seed} if args.seed is not None else None)
    args.run_name = args.run_name or str(cfg.training.run_name)
    setup_logging(cfg, args.run_name)
    try:
        result = CNNTrainer(cfg, args).train()
        append_experiment_log(cfg, args.run_name, "cnn", result)
        append_experiment_history(cfg, args.run_name, "cnn", result)
    except Exception as error:
        LOGGER.error("CNN training failed: %s", error, exc_info=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
