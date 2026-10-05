# Final report skeleton

## 1 Objective

Compare a CNN super-resolution baseline with a fine-tuned SRGAN for 4×
satellite image reconstruction.

## 2 Data

The processed data contains 360 valid paired tiles across 10 cities, with 36
tiles per city. HR imagery is approximately 3000×3000 at 0.5 m resolution and
LR imagery is approximately 750×750 at 2 m resolution. The split is
train/validation/test at 70/15/15 by city-stratified tile assignment. The
processed test split contains 6,050 HR/LR patches. The alignment report found
zero misaligned sampled tiles and mean zero-shift PSNR of 24.6387 dB.

## 3 Methods

The CNN uses 16 residual blocks, 64 channels, L1 loss, Adam at 2e-4, cosine
annealing, and AMP. The SRGAN generator uses the reference-compatible
16-block SRResNet architecture and a VGG-style discriminator. Pixel warm-up
uses L1 at 1e-4 for 2 epochs; GAN fine-tuning uses generator and discriminator
Adam optimizers at 1e-5 with pixel, perceptual, and adversarial weights 0.01,
1.0, and 0.001. Parameter counts are approximately 1.52M for the CNN and
1.55M for the SRGAN generator.

## 4 Training runs

The original CNN run completed 30 epochs and the original SRGAN run completed
2 pixel-warm-up epochs plus 8 GAN epochs. Their recorded validation results are
in `docs/experiment_log.md`. The v2 YAML files are optimized re-runs and have
no final checkpoints yet.

### Known deviations

The first GAN log lines had nonzero `pixel` and `perceptual` values, so the
intended D-only warm-up was skipped in the original run because the stage check
used the global step. The log shows ImageNet pretrained weights loaded before
the pixel warm-up; the GAN stage then followed the completed pixel warm-up in
the same process. The original run used batch size 2, `num_workers: 0`, BatchNorm
in train mode with small batches, no early stopping, and a cosine schedule for
the CNN. These facts are from `logs/srgan_gpu_2026-10-04_11-29-18.log` and the
original training configuration.

## 5 Evaluation protocol

Evaluation uses the held-out test patches with identical batches for all
methods, exact fp32 mode by default, a bicubic baseline, PSNR_Y, SSIM_Y, and
AlexNet LPIPS. PSNR_Y and SSIM_Y use plain luma coefficients
`[0.299, 0.587, 0.114]` and exclude a 4-pixel border. Paired significance
comparisons use seeded bootstrap confidence intervals.

## 6 Results

Full test evaluation covered 6,050 patches and took 251.12 seconds on CUDA.
Metrics use plain luma and a 4-pixel border; lower LPIPS is better.

| method | PSNR_Y | SSIM_Y | LPIPS |
|---|---:|---:|---:|
| bicubic | 25.7540 +/- 3.7367 | 0.6782 +/- 0.1082 | 0.5507 +/- 0.1125 |
| cnn | 28.2560 +/- 3.7142 | 0.7913 +/- 0.0946 | 0.3120 +/- 0.1196 |
| srgan_pixel_only | 28.0481 +/- 3.6772 | 0.7846 +/- 0.0964 | 0.3231 +/- 0.1196 |
| srgan_pretrained | 26.8227 +/- 3.5537 | 0.7277 +/- 0.1032 | 0.2102 +/- 0.0690 |
| srgan_finetuned | 25.8570 +/- 3.1573 | 0.7173 +/- 0.1103 | 0.1487 +/- 0.0712 |

Paired bootstrap mean differences (A minus B; 95% CI) were: CNN vs bicubic:
PSNR 2.5019 [2.4785, 2.5253], SSIM 0.1131 [0.1120, 0.1142], LPIPS
-0.2386 [-0.2406, -0.2366]; fine-tuned SRGAN vs bicubic: PSNR 0.1029
[0.0760, 0.1300], SSIM 0.0390 [0.0378, 0.0403], LPIPS -0.4020
[-0.4038, -0.4002]; fine-tuned SRGAN vs CNN: PSNR -2.3990 [-2.4214,
-2.3763], SSIM -0.0741 [-0.0748, -0.0733], LPIPS -0.1634 [-0.1651,
-0.1618]; fine-tuned SRGAN vs pretrained SRGAN: PSNR -0.9657 [-0.9846,
-0.9474], SSIM -0.0104 [-0.0110, -0.0098], LPIPS -0.0615 [-0.0626,
-0.0605]; fine-tuned SRGAN vs pixel-only SRGAN: PSNR -2.1911 [-2.2120,
-2.1694], SSIM -0.0673 [-0.0680, -0.0666], LPIPS -0.1745 [-0.1762,
-0.1728]; pixel-only SRGAN vs CNN: PSNR -0.2079 [-0.2135, -0.2024], SSIM
-0.0067 [-0.0069, -0.0066], LPIPS 0.0111 [0.0108, 0.0113].

Per-city ranges (minimum to maximum city mean) were CNN: PSNR 25.0683--33.1201,
SSIM 0.6742--0.8754, LPIPS 0.2200--0.4174; fine-tuned SRGAN: PSNR
22.6113--29.8448, SSIM 0.5956--0.8198, LPIPS 0.1155--0.2286. Figures are in
`results/plots/`, with qualitative groups in `results/comparisons/`.

## 7 Qualitative analysis

Figure folders: `results/comparisons/random/`,
`results/comparisons/srgan_wins/`, and `results/comparisons/srgan_fails/`.

TODO(user): fill in after viewing results/comparisons (artifacts:
oversharpening, hallucinated texture, blur, checkerboard).

## 8 Limitations

This study uses one dataset, real LR rather than synthetic degradation, and
256-pixel stored patches. Bicubic PSNR at zero shift is about 24.6 dB. GAN
outputs may hallucinate satellite texture, and metrics cannot fully measure
ground-truth fidelity of fine detail.

## 9 Conclusion

CNN is best on PSNR_Y (28.2560) and SSIM_Y (0.7913). Fine-tuned SRGAN is best
on LPIPS (0.1487). Against CNN, the fine-tuned SRGAN differences are PSNR
-2.3990 dB (95% CI [-2.4214, -2.3763]), SSIM -0.0741 ([-0.0748, -0.0733]),
and LPIPS -0.1634 ([-0.1651, -0.1618]). The final choice depends on whether
ground-truth fidelity or perceptual sharpness matters for the downstream use.

## 10 Reproducibility

Use the commands in `README.md`, the configured seed 42, the saved resolved
configs in `runs/`, the JSON experiment history, and the environment metadata.
