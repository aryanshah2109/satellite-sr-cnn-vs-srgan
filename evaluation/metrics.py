"""GPU-friendly image-quality metrics for super-resolution."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from utils.image_utils import crop_border, rgb_to_y


def _crop_y(sr: Tensor, hr: Tensor, border: int) -> tuple[Tensor, Tensor]:
    return crop_border(rgb_to_y(sr), border), crop_border(rgb_to_y(hr), border)


def psnr_y(sr: Tensor, hr: Tensor, border: int = 0, data_range: float = 1.0) -> Tensor:
    """Return per-image BT.601 Y-channel PSNR for BCHW tensors."""
    sr_y, hr_y = _crop_y(sr, hr, border)
    mse = (sr_y - hr_y).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).eps) / data_range**2)


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] @ kernel[None, :]
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim_y(
    sr: Tensor,
    hr: Tensor,
    border: int = 0,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Return per-image Gaussian-window SSIM on the BT.601 Y channel."""
    sr_y, hr_y = _crop_y(sr, hr, border)
    window = _gaussian_window(window_size, sigma, 1, sr.device, sr.dtype)
    padding = window_size // 2
    mu_sr = functional.conv2d(sr_y, window, padding=padding)
    mu_hr = functional.conv2d(hr_y, window, padding=padding)
    sigma_sr = functional.conv2d(sr_y * sr_y, window, padding=padding) - mu_sr.square()
    sigma_hr = functional.conv2d(hr_y * hr_y, window, padding=padding) - mu_hr.square()
    sigma_cross = functional.conv2d(sr_y * hr_y, window, padding=padding) - mu_sr * mu_hr
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    score = ((2 * mu_sr * mu_hr + c1) * (2 * sigma_cross + c2)) / (
        (mu_sr.square() + mu_hr.square() + c1) * (sigma_sr + sigma_hr + c2)
    )
    return score.flatten(1).mean(dim=1)


class LPIPSMetric:
    """Lazily constructed LPIPS metric returning one value per image."""

    def __init__(self, net: str = "alex") -> None:
        self.net_name = net
        self.metric: nn.Module | None = None

    def _get_metric(self, device: torch.device) -> nn.Module:
        if self.metric is None:
            try:
                import lpips

                self.metric = lpips.LPIPS(net=self.net_name).to(device).eval()
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    "Could not initialize LPIPS; install lpips and make its weights available"
                ) from error
        return self.metric.to(device)

    def __call__(self, sr: Tensor, hr: Tensor) -> Tensor:
        """Return per-image LPIPS for [0, 1] RGB BCHW tensors."""
        with torch.no_grad():
            values = self._get_metric(sr.device)(
                sr.mul(2.0).sub(1.0), hr.mul(2.0).sub(1.0)
            )
        return values.flatten(1).mean(dim=1)


def average_metric(values: Tensor) -> float:
    """Return a Python float average of per-image metric values."""
    return float(values.detach().float().mean().item())


__all__ = ["LPIPSMetric", "average_metric", "psnr_y", "ssim_y"]
