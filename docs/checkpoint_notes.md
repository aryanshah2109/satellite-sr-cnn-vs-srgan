# Checkpoint notes

## Folder convention

Training checkpoints are stored under `checkpoints/<model>/<run_name>/` and
resolved run configurations and environments are stored under
`runs/<run_name>/`. Existing trained runs are `cnn/cnn_gpu` and
`srgan/srgan_gpu`; v2 runs use `cnn/cnn_v2` and `srgan/srgan_v2`.

## CNN files

`best.pt` is written when validation PSNR improves. `last.pt` is written at the
configured save interval and is the resume checkpoint. Both contain the CNN
`model_state_dict`, metadata, optimizer state, and `extra` state containing the
scheduler, scaler, stage, and stage step.

## SRGAN files

The pixel stage writes `best_pixel_warmup.pt`, `last_pixel_warmup.pt`, and
`pixel_warmup_final.pt`. The GAN stage writes `best_gan.pt`, `last_gan.pt`, and
`best_generator.pth`. `best_gan.pt` and `best_generator.pth` contain the
generator checkpoint; `last_gan.pt` additionally stores the discriminator,
both optimizers, both schedulers, both scalers, stage-local step, and stage
name in `extra`.

Every project checkpoint stores `epoch`, `best_metric`, `metrics`,
`config_hash`, git commit when available, timestamp, and torch version in
`metadata`. State-dict prefixes `module.` and `_orig_mod.` are normalized when
loading third-party weights.

## Loading

Use `load_checkpoint` for project-format checkpoints and
`load_pretrained_weights` for the third-party ImageNet checkpoint. Inference
uses `inference.super_resolve.load_model_for_inference`; resume uses the
trainer's `--resume` option. `best.pt` and `best_generator.pth` are intended
for inference, while `last.pt` and `last_gan.pt` are intended for resume.

## Pretrained source

`checkpoints/pretrained/SOURCE.md` identifies `SRGAN_x4-ImageNet.pth.tar` as
coming from SRGAN-PyTorch, trained on ImageNet2012 verification images, with
Apache License 2.0 attribution. The bundled source file does not state an
upstream repository URL.

## Runs

| run_name | stage | best metric | value | date |
|---|---|---:|---:|---|
| cnn_gpu | cnn | PSNR | 28.728268 | 2026-10-04 |
| srgan_gpu | all / GAN | LPIPS 0.143520 (best epoch 7) | 0.143520 | 2026-10-04 |
