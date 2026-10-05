"""Fine-tune the pretrained SRGAN generator on satellite imagery."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch import nn

from losses.adversarial import GANLoss
from losses.perceptual import VGGPerceptualLoss
from losses.reconstruction import PixelLoss
from evaluation.metrics import LPIPSMetric
from models import build_discriminator, build_model
from sr_datasets.sr_dataset import build_dataloader
from training.trainer_utils import (
    RunningLosses,
    ThroughputTimer,
    append_experiment_log,
    append_experiment_history,
    amp_context,
    build_optimizer,
    build_scheduler,
    create_writer,
    evenly_spaced_indices,
    grad_scaler,
    save_training_checkpoint,
    set_requires_grad,
    setup_run,
    to_channels_last,
    to_channels_last_input,
    validate,
)
from utils.checkpoint import load_checkpoint, load_pretrained_weights
from utils.config import Config, load_config
from utils.image_utils import get_device
from utils.logger import setup_logging
from utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


class SRGANTrainer:
    """Run pixel warmup and adversarial fine-tuning stages."""

    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.device = get_device(args.device or str(cfg.device.preferred))
        set_seed(int(cfg.seed), bool(cfg.deterministic))
        self.channels_last = bool(cfg.training.get("channels_last", False))
        self.generator = to_channels_last(build_model(cfg).to(self.device), self.channels_last)
        self.discriminator = to_channels_last(
            build_discriminator(cfg).to(self.device), self.channels_last
        )
        self.train_loader = build_dataloader(cfg, "train")
        val_count = args.val_subset or int(cfg.training.val_subset_size)
        val_workers = int(cfg.training.get("val_num_workers", 0))
        val_dataset = build_dataloader(cfg, "val", num_workers=val_workers).dataset
        self.val_loader = build_dataloader(
            cfg,
            "val",
            evenly_spaced_indices(len(val_dataset), val_count),
            num_workers=val_workers,
        )
        self.checkpoint_dir = setup_run(cfg, args.run_name, "srgan")
        self.writer = create_writer(cfg, args.run_name)
        self.global_step = 0
        self.best_metric: float | None = None
        self.pretrained_loaded = False
        self.lpips_metric = LPIPSMetric(str(cfg.training.metrics.lpips_net))
        self.completed_results: list[tuple[str, dict[str, float | int]]] = []

    def _load_pretrained(self) -> None:
        if not self.pretrained_loaded:
            load_pretrained_weights(Path(self.cfg.model.pretrained_weights), self.generator, strict=True)
            self.pretrained_loaded = True
            LOGGER.info("Generator initialization source: ImageNet pretrained weights")

    def _apply_generator_bn_freeze(self) -> None:
        """Keep generator BatchNorm statistics and affine parameters frozen."""
        if not bool(self.cfg.training.get("freeze_generator_bn", False)):
            return
        for module in self.generator.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def _load_project_generator(self, path: Path, source: str) -> dict[str, object]:
        """Load generator weights from a project-format checkpoint."""
        metadata = load_checkpoint(path, self.generator, map_location=self.device, strict=True)
        self.pretrained_loaded = True
        LOGGER.info("Generator initialization source: %s (%s)", source, path)
        return metadata

    def _initialize_gan_generator(self) -> None:
        """Select the requested GAN-stage generator initialization source."""
        init_from = getattr(self.args, "init_from", None)
        if init_from:
            self._load_project_generator(Path(init_from), "--init-from checkpoint")
            return
        warmup_path = self.checkpoint_dir / "pixel_warmup_final.pt"
        if warmup_path.is_file():
            self._load_project_generator(warmup_path, "pixel warmup checkpoint")
            return
        LOGGER.warning(
            "Pixel warmup checkpoint was skipped; falling back to ImageNet pretrained weights"
        )
        self._load_pretrained()

    def _make_optimizers(self, stage: str) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None, object, object | None]:
        stage_cfg = self.cfg.training.stages[stage]
        g_cfg = stage_cfg.optimizer if stage == "pixel_warmup" else stage_cfg.g_optimizer
        g_optimizer = build_optimizer(self.generator.parameters(), g_cfg)
        scheduler_cfg = stage_cfg.scheduler if stage == "pixel_warmup" else stage_cfg.g_scheduler
        g_scheduler = build_scheduler(g_optimizer, scheduler_cfg, int(stage_cfg.epochs))
        if stage == "pixel_warmup":
            return g_optimizer, None, g_scheduler, None
        d_optimizer = build_optimizer(self.discriminator.parameters(), stage_cfg.d_optimizer)
        d_scheduler = build_scheduler(d_optimizer, stage_cfg.d_scheduler, int(stage_cfg.epochs))
        return g_optimizer, d_optimizer, g_scheduler, d_scheduler

    def _run_stage_corrected(
        self,
        stage: str,
        resume_path: Path | None = None,
    ) -> dict[str, float | int]:
        """Run one SRGAN stage with stage-local counters and efficient updates."""
        stage_cfg = self.cfg.training.stages[stage]
        if stage == "pixel_warmup" and resume_path is None:
            self._load_pretrained()
        if stage == "gan" and resume_path is None and not self.pretrained_loaded:
            self._initialize_gan_generator()
        self.best_metric = None
        g_optimizer, d_optimizer, g_scheduler, d_scheduler = self._make_optimizers(stage)
        g_scaler = grad_scaler(self.device, bool(self.cfg.training.amp))
        d_scaler = grad_scaler(self.device, bool(self.cfg.training.amp))
        pixel_loss = PixelLoss("l1").to(self.device)
        perceptual_loss = None
        gan_loss = None
        if stage == "gan":
            perceptual_loss = VGGPerceptualLoss(
                int(self.cfg.training.perceptual.feature_layer),
                str(self.cfg.training.perceptual.kind),
                tuple(self.cfg.training.perceptual.mean),
                tuple(self.cfg.training.perceptual.std),
            ).to(self.device)
            gan_loss = GANLoss(
                str(self.cfg.training.adversarial.kind),
                float(self.cfg.training.adversarial.real_label_smoothing),
            ).to(self.device)
        start_epoch = 0
        stage_step = 0
        if resume_path is not None:
            metadata = load_checkpoint(
                resume_path,
                self.generator,
                g_optimizer,
                map_location=self.device,
                strict=True,
            )
            extra = metadata.get("extra", {})
            if d_optimizer is not None:
                discriminator_state = extra.get("discriminator_state_dict")
                if discriminator_state is not None:
                    self.discriminator.load_state_dict(discriminator_state, strict=True)
                if extra.get("d_optimizer_state_dict") is not None:
                    d_optimizer.load_state_dict(extra["d_optimizer_state_dict"])
                if d_scheduler is not None and extra.get("d_scheduler_state_dict") is not None:
                    d_scheduler.load_state_dict(extra["d_scheduler_state_dict"])
                if extra.get("d_scaler_state_dict") is not None:
                    d_scaler.load_state_dict(extra["d_scaler_state_dict"])
            if g_scheduler is not None and extra.get("scheduler_state_dict") is not None:
                g_scheduler.load_state_dict(extra["scheduler_state_dict"])
            if extra.get("scaler_state_dict") is not None:
                g_scaler.load_state_dict(extra["scaler_state_dict"])
            start_epoch = int(metadata.get("epoch", 0))
            stage_step = int(extra.get("stage_step", 0))
            self.best_metric = extra.get("best_metric", metadata.get("best_metric"))
            LOGGER.info(
                "Resumed SRGAN stage=%s from %s at epoch=%s stage_step=%s",
                stage,
                resume_path,
                start_epoch,
                stage_step,
            )
        epochs = int(stage_cfg.epochs)
        max_iters = self.args.max_iters or self.cfg.training.max_iters
        max_iters = None if max_iters is None else int(max_iters)
        warmup_iters = (
            self.args.d_warmup_iters
            if self.args.d_warmup_iters is not None
            else int(stage_cfg.get("d_warmup_iters", 0))
        )
        if stage == "gan":
            LOGGER.info("D-only warmup for %s iterations", warmup_iters)
        timer = ThroughputTimer(max(len(self.train_loader) * epochs, 1))
        final_metrics: dict[str, float] = {}
        stage_start = time.perf_counter()
        completed_epoch = start_epoch
        for epoch in range(start_epoch, epochs):
            self.generator.train()
            self.discriminator.train()
            self._apply_generator_bn_freeze()
            losses = RunningLosses()
            wait_total = 0.0
            compute_total = 0.0
            batch_iterator = iter(self.train_loader)
            while True:
                if max_iters is not None and stage_step >= max_iters:
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
                if stage == "pixel_warmup":
                    g_optimizer.zero_grad(set_to_none=True)
                    with amp_context(self.device, bool(self.cfg.training.amp)):
                        loss = pixel_loss(self.generator(lr), hr)
                    g_scaler.scale(loss).backward()
                    g_scaler.step(g_optimizer)
                    g_scaler.update()
                    losses.update({"pixel": loss, "g": loss})
                else:
                    assert d_optimizer is not None and gan_loss is not None and perceptual_loss is not None
                    do_generator_update = stage_step >= warmup_iters
                    if do_generator_update and stage_step == warmup_iters:
                        LOGGER.info("D-only warmup complete; generator updates begin")
                    set_requires_grad(self.discriminator, True)
                    d_optimizer.zero_grad(set_to_none=True)
                    with amp_context(self.device, bool(self.cfg.training.amp)):
                        if do_generator_update:
                            fake = self.generator(lr)
                        else:
                            with torch.no_grad():
                                fake = self.generator(lr)
                        real_logits = self.discriminator(hr)
                        fake_logits = self.discriminator(fake.detach())
                        d_loss = gan_loss.discriminator_loss(real_logits, fake_logits)
                    d_scaler.scale(d_loss).backward()
                    d_scaler.step(d_optimizer)
                    d_scaler.update()
                    if do_generator_update:
                        set_requires_grad(self.discriminator, False)
                        g_optimizer.zero_grad(set_to_none=True)
                        with amp_context(self.device, bool(self.cfg.training.amp)):
                            pixel_value = pixel_loss(fake.float(), hr.float())
                            perceptual_value = perceptual_loss(fake.float(), hr.float())
                            adversarial_value = gan_loss.generator_loss(self.discriminator(fake))
                            g_loss = (
                                float(stage_cfg.w_pixel) * pixel_value
                                + float(stage_cfg.w_perceptual) * perceptual_value
                                + float(stage_cfg.w_adv) * adversarial_value
                            )
                        g_scaler.scale(g_loss).backward()
                        g_scaler.step(g_optimizer)
                        g_scaler.update()
                    else:
                        g_loss = d_loss
                        pixel_value = torch.zeros_like(d_loss)
                        perceptual_value = torch.zeros_like(d_loss)
                        adversarial_value = torch.zeros_like(d_loss)
                    losses.update(
                        {
                            "d": d_loss,
                            "g": g_loss,
                            "pixel": pixel_value,
                            "perceptual": perceptual_value,
                            "adversarial": adversarial_value,
                            "d_real_prob": torch.sigmoid(real_logits).mean(),
                            "d_fake_prob": torch.sigmoid(fake_logits).mean(),
                        }
                    )
                compute_total += time.perf_counter() - compute_start
                self.global_step += 1
                stage_step += 1
                rate, eta = timer.update(stage_step)
                if self.global_step % int(self.cfg.training.log_every_n_iters) == 0:
                    values = losses.averages()
                    log_window = int(self.cfg.training.log_every_n_iters)
                    LOGGER.info(
                        "stage=%s epoch=%s step=%s values=%s it/s=%.3f eta=%.1fs "
                        "data_wait=%.4fs compute=%.4fs",
                        stage,
                        epoch + 1,
                        self.global_step,
                        values,
                        rate,
                        eta,
                        wait_total / log_window,
                        compute_total / log_window,
                    )
                    for name, value in values.items():
                        self.writer.add_scalar(f"train/{stage}/{name}", value, self.global_step)
                    wait_total = 0.0
                    compute_total = 0.0
            completed_epoch = epoch + 1
            if g_scheduler is not None:
                g_scheduler.step()
            if d_scheduler is not None:
                d_scheduler.step()
            if completed_epoch % int(self.cfg.training.val_every_n_epochs) == 0:
                final_metrics = validate(
                    self.generator,
                    self.val_loader,
                    self.device,
                    self.cfg.training.metrics,
                    bool(self.cfg.training.amp),
                    self.lpips_metric,
                )
                LOGGER.info("stage=%s epoch=%s validation=%s", stage, completed_epoch, final_metrics)
                metric_name = str(stage_cfg.best_metric)
                metric = final_metrics[metric_name]
                improved = self.best_metric is None or (
                    metric > self.best_metric if metric_name == "psnr" else metric < self.best_metric
                )
                if improved:
                    self.best_metric = metric
                    save_training_checkpoint(
                        self.checkpoint_dir / f"best_{stage}.pt",
                        self.generator,
                        g_optimizer,
                        g_scheduler,
                        g_scaler,
                        completed_epoch,
                        metric,
                        final_metrics,
                        self.cfg,
                        {"stage": stage, "stage_step": stage_step, "best_metric": self.best_metric},
                    )
                    if stage == "gan":
                        save_training_checkpoint(
                            self.checkpoint_dir / "best_generator.pth",
                            self.generator,
                            None,
                            None,
                            g_scaler,
                            completed_epoch,
                            metric,
                            final_metrics,
                            self.cfg,
                            {"stage": stage, "stage_step": stage_step, "best_metric": self.best_metric},
                        )
            if completed_epoch % int(self.cfg.training.save_every_n_epochs) == 0:
                extra: dict[str, object] = {
                    "stage": stage,
                    "stage_step": stage_step,
                    "best_metric": self.best_metric,
                }
                if d_optimizer is not None:
                    extra.update(
                        {
                            "discriminator_state_dict": self.discriminator.state_dict(),
                            "d_optimizer_state_dict": d_optimizer.state_dict(),
                            "d_scheduler_state_dict": (
                                None if d_scheduler is None else d_scheduler.state_dict()
                            ),
                            "d_scaler_state_dict": d_scaler.state_dict(),
                        }
                    )
                save_training_checkpoint(
                    self.checkpoint_dir / f"last_{stage}.pt",
                    self.generator,
                    g_optimizer,
                    g_scheduler,
                    g_scaler,
                    completed_epoch,
                    self.best_metric,
                    final_metrics,
                    self.cfg,
                    extra,
                )
            if max_iters is not None and stage_step >= max_iters:
                break
        if stage == "pixel_warmup":
            save_training_checkpoint(
                self.checkpoint_dir / "pixel_warmup_final.pt",
                self.generator,
                g_optimizer,
                g_scheduler,
                g_scaler,
                completed_epoch,
                self.best_metric,
                final_metrics,
                self.cfg,
                {"stage": stage, "stage_step": stage_step, "best_metric": self.best_metric},
            )
        elapsed = time.perf_counter() - stage_start
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )
        LOGGER.info(
            "Completed stage=%s in %.2fs; peak_gpu_memory_mb=%.2f",
            stage,
            elapsed,
            peak_memory,
        )
        return {**final_metrics, "epochs_done": completed_epoch, "elapsed_seconds": elapsed}

    def _run_stage(self, stage: str) -> dict[str, float | int]:
        stage_cfg = self.cfg.training.stages[stage]
        if stage == "pixel_warmup":
            self._load_pretrained()
        g_optimizer, d_optimizer, g_scheduler, d_scheduler = self._make_optimizers(stage)
        scaler = grad_scaler(self.device, bool(self.cfg.training.amp))
        pixel_loss = PixelLoss("l1").to(self.device)
        perceptual_loss = None
        gan_loss = None
        if stage == "gan":
            perceptual_loss = VGGPerceptualLoss(int(self.cfg.training.perceptual.feature_layer), str(self.cfg.training.perceptual.kind), tuple(self.cfg.training.perceptual.mean), tuple(self.cfg.training.perceptual.std)).to(self.device)
            gan_loss = GANLoss(str(self.cfg.training.adversarial.kind), float(self.cfg.training.adversarial.real_label_smoothing)).to(self.device)
        epochs = int(stage_cfg.epochs)
        max_iters = self.args.max_iters or self.cfg.training.max_iters
        max_iters = None if max_iters is None else int(max_iters)
        warmup_iters = self.args.d_warmup_iters if self.args.d_warmup_iters is not None else int(stage_cfg.d_warmup_iters if "d_warmup_iters" in stage_cfg else 0)
        timer = ThroughputTimer(max(len(self.train_loader) * epochs, 1))
        final_metrics: dict[str, float] = {}
        stage_start = time.perf_counter()
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            for batch in self.train_loader:
                if max_iters is not None and self.global_step >= max_iters:
                    break
                lr = batch["lr"].to(self.device, non_blocking=True)
                hr = batch["hr"].to(self.device, non_blocking=True)
                if stage == "pixel_warmup":
                    g_optimizer.zero_grad(set_to_none=True)
                    with amp_context(self.device, bool(self.cfg.training.amp)):
                        loss = pixel_loss(self.generator(lr), hr)
                    scaler.scale(loss).backward()
                    scaler.step(g_optimizer)
                    scaler.update()
                    log_values = {"pixel": float(loss.item())}
                else:
                    assert d_optimizer is not None and gan_loss is not None and perceptual_loss is not None
                    set_requires_grad(self.discriminator, True)
                    d_optimizer.zero_grad(set_to_none=True)
                    with amp_context(self.device, bool(self.cfg.training.amp)):
                        fake = self.generator(lr)
                        real_logits = self.discriminator(hr)
                        fake_logits = self.discriminator(fake.detach())
                        d_loss = gan_loss.discriminator_loss(real_logits, fake_logits)
                    scaler.scale(d_loss).backward()
                    scaler.step(d_optimizer)
                    scaler.update()
                    if self.global_step >= warmup_iters:
                        set_requires_grad(self.discriminator, False)
                        g_optimizer.zero_grad(set_to_none=True)
                        with amp_context(self.device, bool(self.cfg.training.amp)):
                            generated = self.generator(lr)
                            pixel_value = pixel_loss(generated.float(), hr.float())
                            perceptual_value = perceptual_loss(generated.float(), hr.float())
                            adversarial_value = gan_loss.generator_loss(self.discriminator(generated))
                            loss = float(stage_cfg.w_pixel) * pixel_value + float(stage_cfg.w_perceptual) * perceptual_value + float(stage_cfg.w_adv) * adversarial_value
                        scaler.scale(loss).backward()
                        scaler.step(g_optimizer)
                        scaler.update()
                    else:
                        loss = d_loss
                        pixel_value = torch.zeros((), device=self.device)
                        perceptual_value = torch.zeros((), device=self.device)
                        adversarial_value = torch.zeros((), device=self.device)
                    log_values = {"d": float(d_loss.item()), "g": float(loss.item()), "pixel": float(pixel_value.item()), "perceptual": float(perceptual_value.item()), "adversarial": float(adversarial_value.item()), "d_real_prob": float(torch.sigmoid(real_logits).mean().item()), "d_fake_prob": float(torch.sigmoid(fake_logits).mean().item())}
                self.global_step += 1
                rate, eta = timer.update(self.global_step)
                if self.global_step % int(self.cfg.training.log_every_n_iters) == 0:
                    LOGGER.info("stage=%s epoch=%s step=%s values=%s it/s=%.3f eta=%.1fs", stage, epoch + 1, self.global_step, log_values, rate, eta)
                    for name, value in log_values.items():
                        self.writer.add_scalar(f"train/{stage}/{name}", value, self.global_step)
            if g_scheduler is not None:
                g_scheduler.step()
            if d_scheduler is not None:
                d_scheduler.step()
            if (epoch + 1) % int(self.cfg.training.val_every_n_epochs) == 0:
                final_metrics = validate(self.generator, self.val_loader, self.device, self.cfg.training.metrics, bool(self.cfg.training.amp))
                LOGGER.info("stage=%s epoch=%s validation=%s", stage, epoch + 1, final_metrics)
                metric_name = str(stage_cfg.best_metric)
                metric = final_metrics[metric_name]
                improved = self.best_metric is None or (metric > self.best_metric if metric_name == "psnr" else metric < self.best_metric)
                if improved:
                    self.best_metric = metric
                    save_training_checkpoint(self.checkpoint_dir / f"best_{stage}.pt", self.generator, g_optimizer, g_scheduler, scaler, epoch + 1, metric, final_metrics, self.cfg, {"stage": stage})
                    if stage == "gan":
                        save_training_checkpoint(self.checkpoint_dir / "best_generator.pth", self.generator, None, None, scaler, epoch + 1, metric, final_metrics, self.cfg, {"stage": stage})
            if (epoch + 1) % int(self.cfg.training.save_every_n_epochs) == 0:
                extra = {"stage": stage}
                if d_optimizer is not None:
                    extra.update({"discriminator_state_dict": self.discriminator.state_dict(), "d_optimizer_state_dict": d_optimizer.state_dict(), "d_scheduler_state_dict": None if d_scheduler is None else d_scheduler.state_dict()})
                save_training_checkpoint(self.checkpoint_dir / f"last_{stage}.pt", self.generator, g_optimizer, g_scheduler, scaler, epoch + 1, self.best_metric, final_metrics, self.cfg, extra)
            if max_iters is not None and self.global_step >= max_iters:
                break
        if stage == "pixel_warmup":
            save_training_checkpoint(self.checkpoint_dir / "pixel_warmup_final.pt", self.generator, g_optimizer, g_scheduler, scaler, epochs, self.best_metric, final_metrics, self.cfg, {"stage": stage})
        LOGGER.info("Completed stage=%s in %.2fs", stage, time.perf_counter() - stage_start)
        return {**final_metrics, "epochs_done": epoch + 1, "elapsed_seconds": time.perf_counter() - stage_start}

    def train(self) -> dict[str, float | int]:
        """Run the requested SRGAN stages."""
        resume_path = Path(self.args.resume) if self.args.resume else None
        resume_stage = None
        if resume_path is not None:
            resume_stage = "gan" if "gan" in resume_path.stem else "pixel_warmup"
        if self.args.stage in {"pixel_warmup", "all"} and resume_stage != "gan":
            result = self._run_stage_corrected(
                "pixel_warmup", resume_path if resume_stage == "pixel_warmup" else None
            )
            self.completed_results.append(("pixel_warmup", result))
            resume_path = None
        if self.args.stage in {"gan", "all"}:
            result = self._run_stage_corrected("gan", resume_path)
            self.completed_results.append(("gan", result))
            return result
        return self.completed_results[-1][1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/srgan.yaml"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stage", choices=("pixel_warmup", "gan", "all"), default="all")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--d-warmup-iters", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    """Run SRGAN training."""
    args = _parse_args()
    cfg = load_config(args.config)
    args.run_name = args.run_name or str(cfg.training.run_name)
    setup_logging(cfg, args.run_name)
    try:
        trainer = SRGANTrainer(cfg, args)
        result = trainer.train()
        del result
        for stage, stage_result in trainer.completed_results:
            append_experiment_log(cfg, args.run_name, stage, stage_result)
            append_experiment_history(cfg, args.run_name, stage, stage_result)
    except KeyboardInterrupt:
        LOGGER.warning("SRGAN training interrupted")
    except Exception as error:
        LOGGER.error("SRGAN training failed: %s", error, exc_info=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
