# Satellite Super-Resolution: CNN vs Fine-Tuned SRGAN

## Overview

This project compares a CNN baseline with a fine-tuned SRGAN for 4× satellite
super-resolution: HR imagery is 0.5 m per pixel and the provided LR imagery is
2 m per pixel. The evaluation uses paired satellite patches and reports
PSNR_Y, SSIM_Y, and LPIPS.

## Repository structure

```text
configs/       YAML training and evaluation configuration
models/        CNN, SRGAN generator, and discriminator
training/      CNN and SRGAN trainers
evaluation/    metrics, evaluation tables, plots, and comparisons
inference/     tiled x4 inference CLI
sr_datasets/   raster reading, patch datasets, and splits
checkpoints/   trained and pretrained weights
results/       generated metrics, plots, and images
docs/          experiment history and project notes
```

## Setup on Windows

```powershell
cd "D:\Sem 7 AML Project\Main"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the CUDA-compatible PyTorch and torchvision wheels appropriate for the
machine before installing the remaining requirements when necessary.

## Data

Raw paired GeoTIFFs are under `data/raw/satellite/HR_0.5m/HR_0.5m` and
`data/raw/satellite/LR_2m/LR_2m`. Data preparation commands are:

```powershell
python -m scripts.prepare_data inspect --config configs/data.yaml
python -m scripts.prepare_data check-alignment --config configs/data.yaml
python -m scripts.prepare_data split --config configs/data.yaml
python -m scripts.prepare_data build --config configs/data.yaml
```

See `data/README.md` for the data card and provenance details.

## Training

Original configurations:

```powershell
python -m training.train_cnn --config configs/cnn.yaml
python -m training.train_srgan --config configs/srgan.yaml --stage all
```

Optimized configurations:

```powershell
python -m training.train_cnn --config configs/cnn_v2.yaml
python -m training.train_srgan --config configs/srgan_v2.yaml --stage all
```

The v2 configurations use larger batches, worker prefetching, random smaller
crops at equal pixel exposure, channels-last tensors, and frozen generator
BatchNorm statistics for SRGAN. Validate their quality against the original
runs before adopting them.

Resume or initialize training with:

```powershell
python -m training.train_cnn --config configs/cnn_v2.yaml --resume checkpoints/cnn/cnn_v2/last.pt
python -m training.train_srgan --config configs/srgan_v2.yaml --stage gan --resume checkpoints/srgan/srgan_v2/last_gan.pt
python -m training.train_srgan --config configs/srgan_v2.yaml --stage gan --init-from checkpoints/srgan/srgan_v2/pixel_warmup_final.pt
tensorboard --logdir runs
```

## Evaluation

```powershell
python -m evaluation.evaluate --config configs/evaluation.yaml --visual --update-readme
```

For a bounded check:

```powershell
python -m evaluation.evaluate --config configs/evaluation.yaml --max-samples 64 --output-dir results/smoke --visual
```

Metrics use plain luma coefficients `[0.299, 0.587, 0.114]`, not studio-range
BT.601. A 4-pixel border is excluded from PSNR_Y and SSIM_Y.

## Inference

```powershell
python -m inference.super_resolve --model srgan --model-config configs/srgan.yaml --checkpoint checkpoints/srgan/srgan_gpu/best_generator.pth --input data/raw/satellite/LR_2m/LR_2m/austin1.tif --output results/inference
python -m inference.super_resolve --model cnn --model-config configs/cnn.yaml --checkpoint checkpoints/cnn/cnn_gpu/best.pt --input path\to\images --output results/inference
```

## Results

<!-- RESULTS_TABLE_START -->
| method | split | num_images | PSNR_Y (mean +/- std) | SSIM_Y (mean +/- std) | LPIPS (mean +/- std) |
|---|---|---:|---:|---:|---:|
| bicubic | test | 6050 | 25.7540 +/- 3.7367 | 0.6782 +/- 0.1082 | 0.5507 +/- 0.1125 |
| cnn | test | 6050 | 28.2560 +/- 3.7142 | 0.7913 +/- 0.0946 | 0.3120 +/- 0.1196 |
| srgan_pixel_only | test | 6050 | 28.0481 +/- 3.6772 | 0.7846 +/- 0.0964 | 0.3231 +/- 0.1196 |
| srgan_pretrained | test | 6050 | 26.8227 +/- 3.5537 | 0.7277 +/- 0.1032 | 0.2102 +/- 0.0690 |
| srgan_finetuned | test | 6050 | 25.8570 +/- 3.1573 | 0.7173 +/- 0.1103 | 0.1487 +/- 0.0712 |
| cnn_v2 | test | 6050 | 25.2188 +/- 3.7870 | 0.6718 +/- 0.1117 | 0.4686 +/- 0.1150 |
| srgan_v2_pixel_only | test | 6050 | 24.4288 +/- 3.9260 | 0.6529 +/- 0.1153 | 0.4324 +/- 0.1165 |
| srgan_v2_finetuned | test | 6050 | 25.5781 +/- 3.5789 | 0.6891 +/- 0.1197 | 0.1530 +/- 0.0710 |
<!-- RESULTS_TABLE_END -->

## PSNR versus perceptual quality

PSNR and SSIM reward pixel-level agreement, while LPIPS measures learned
perceptual similarity. A model can therefore improve perceptual sharpness while
losing some distortion-based fidelity; the evaluated test-set numbers and
qualitative comparisons should determine the final choice.

## Limitations

The study uses one satellite dataset, 256-pixel stored patches, and real paired
LR/HR imagery. GAN outputs can hallucinate textures, and no metric fully proves
that fine detail is faithful to the ground truth.

## License and attribution

The bundled ImageNet SRGAN checkpoint is attributed in
`checkpoints/pretrained/SOURCE.md` to the upstream SRGAN-PyTorch project and
Apache License 2.0. The upstream repository URL is not stated in the bundled
materials and remains to be filled manually.
