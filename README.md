# UC Merced 4× Image Super-Resolution

This project trains and compares SRGAN and ESRGAN to upscale UC Merced RGB images by 4×. Bicubic interpolation is included as a non-learned baseline. Images are synthetically downsampled with bicubic degradation; each 64×64 LR input is paired with its original 256×256 HR image. Evaluation reports PSNR_Y, SSIM_Y, and LPIPS.

## Repository structure

```text
configs/       data, model, training, and evaluation configuration
models/        SRGAN and ESRGAN generators and discriminator
training/      adversarial SRGAN/ESRGAN trainer
evaluation/    metrics, evaluation tables, plots, and comparisons
inference/     tiled x4 inference CLI
sr_datasets/   raster reading, patch datasets, and splits
checkpoints/   trained and pretrained weights
results/       generated metrics, plots, and images
docs/          experiment history and project notes
```

## Setup and data preparation

On Windows, use the project environment and a CUDA-enabled PyTorch build compatible with your GPU. The commands below call the environment's Python directly, so PowerShell script activation is not required.

```powershell
cd "D:\Sem 7 AML Project\Sem 7 AML Project\Main"
.\.venv312\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
.\.venv312\Scripts\python.exe -m scripts.prepare_data inspect --config configs/data.yaml
.\.venv312\Scripts\python.exe -m scripts.prepare_data split --config configs/data.yaml
.\.venv312\Scripts\python.exe -m scripts.prepare_data build --config configs/data.yaml
```

The source dataset is `../UCMerced_LandUse/Images`. Splits are class-stratified at image level (70/15/15); see `data/README.md` for details.

## Training

Train SRGAN and the larger GPU-configured ESRGAN independently:

```powershell
.\.venv312\Scripts\python.exe -m training.train_srgan --config configs/srgan_ucmerced.yaml --stage all
.\.venv312\Scripts\python.exe -m training.train_srgan --config configs/esrgan_ucmerced.yaml --stage all
```

The updated ESRGAN uses 12 RRDB blocks, 48 feature channels, 24 growth channels, batch size 4, mixed precision, 20 pixel-warmup epochs, and 40 adversarial epochs. Its new run name is `esrgan_ucmerced_v2`, so its checkpoints go to `checkpoints/esrgan/esrgan_ucmerced_v2/` and do not overwrite the earlier run. The config requests CUDA; start training only in the environment where `torch.cuda.is_available()` is true.

## Evaluation

After both models have trained, evaluate the same UC Merced test split:

```powershell
.\.venv312\Scripts\python.exe -m evaluation.evaluate --config configs/evaluation_ucmerced.yaml
```

Results are written under `results/ucmerced_v2/metrics/` and `results/ucmerced_v2/plots/`, keeping the earlier UC Merced evaluation intact. Higher PSNR_Y and SSIM_Y indicate closer pixel/structure agreement; lower LPIPS indicates greater perceptual similarity. ESRGAN's architecture does not guarantee it will beat SRGAN on every metric; compare its newly trained checkpoint on the held-out test set.

## Inference

Run x4 inference on an LR image with either model:

```powershell
.\.venv312\Scripts\python.exe -m inference.super_resolve --model esrgan --model-config configs/esrgan_ucmerced.yaml --checkpoint checkpoints/esrgan/esrgan_ucmerced_v2/best_generator.pth --input path\to\low_resolution_image.png --output results/ucmerced/inference
.\.venv312\Scripts\python.exe -m inference.super_resolve --model srgan --model-config configs/srgan_ucmerced.yaml --checkpoint checkpoints/srgan/srgan_ucmerced/best_generator.pth --input path\to\low_resolution_image.png --output results/ucmerced/inference
```

Previously recorded CNN checkpoints, metrics, and reports are retained as historical artifacts. CNN is no longer available as an active model, training command, inference option, or evaluation method.

## Limitations

UC Merced contains 2,100 images. Because the LR inputs are synthetically bicubic-downsampled, results do not establish performance on real sensor LR imagery. GAN outputs may hallucinate textures; metrics and visual comparison both matter.
