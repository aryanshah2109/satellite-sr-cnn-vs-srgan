import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from evaluation.evaluate import _city_from_name
from inference.super_resolve import load_model_for_inference
from sr_datasets.sr_dataset import build_dataloader
from utils.config import load_config
from utils.image_utils import crop_border, get_device, rgb_to_y

SOBEL = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
DEFAULT_METHODS = "bicubic,cnn_v2,srgan_pretrained,srgan_v2_pixel_only,srgan_finetuned,srgan_v2_finetuned"


def grad_mag(y):
    kx = SOBEL.to(y.device)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return (gx.square() + gy.square() + 1e-12).sqrt()


def high_freq(y):
    return y - F.avg_pool2d(y, 5, 1, 2, count_include_pad=False)


def structure_metrics(sr, hr, lr, border=4):
    sr_y = crop_border(rgb_to_y(sr), border)
    hr_y = crop_border(rgb_to_y(hr), border)
    g_sr = grad_mag(sr_y)
    g_hr = grad_mag(hr_y)
    a = g_sr.flatten(1)
    b = g_hr.flatten(1)
    a0 = a - a.mean(1, keepdim=True)
    b0 = b - b.mean(1, keepdim=True)
    corr = (a0 * b0).sum(1) / (a0.norm(dim=1) * b0.norm(dim=1)).clamp_min(1e-8)
    hf_sr = high_freq(sr_y).square().flatten(1).mean(1)
    hf_hr = high_freq(hr_y).square().flatten(1).mean(1).clamp_min(1e-10)
    threshold = torch.quantile(b, 0.9, dim=1, keepdim=True)
    e_sr = (a > threshold).float().view_as(g_sr)
    e_hr = (b > threshold).float().view_as(g_hr)
    dilate = lambda e: F.max_pool2d(e, 3, 1, 1)
    precision = (e_sr * dilate(e_hr)).flatten(1).sum(1) / e_sr.flatten(1).sum(1).clamp_min(1)
    recall = (e_hr * dilate(e_sr)).flatten(1).sum(1) / e_hr.flatten(1).sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    down = F.interpolate(sr, scale_factor=0.25, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    mse = (down - lr).square().flatten(1).mean(1).clamp_min(1e-10)
    return {
        "lr_psnr": -10 * torch.log10(mse),
        "grad_corr": corr,
        "hf_ratio": hf_sr / hf_hr,
        "edge_prec": precision,
        "edge_rec": recall,
        "edge_f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results_final"))
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config("configs/evaluation.yaml")
    device = get_device("auto")
    requested = args.methods.split(",")
    models = {}
    for method in cfg.methods:
        name = str(method.name)
        if name not in requested:
            continue
        if str(method.type) == "bicubic":
            models[name] = None
        else:
            models[name] = load_model_for_inference(
                str(method.type), str(method.checkpoint), str(method.model_config), device
            )
    loader = build_dataloader(cfg, "test", batch_size=args.batch_size, num_workers=0, shuffle=False)
    rows = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            names = list(batch["name"])
            for name, model in models.items():
                if model is None:
                    sr = F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
                else:
                    sr = model(lr).float().clamp(0, 1)
                values = {k: v.cpu().tolist() for k, v in structure_metrics(sr, hr, lr).items()}
                for i, patch in enumerate(names):
                    city, tile = _city_from_name(patch, str(cfg.split.city_regex))
                    rows.append(
                        {"method": name, "patch_name": patch, "tile_id": tile, "city": city,
                         **{k: v[i] for k, v in values.items()}}
                    )
            processed += len(names)
            if args.max_samples and processed >= args.max_samples:
                break
    df = pd.DataFrame(rows)
    out_dir = args.results / "metrics"
    df.to_csv(out_dir / "structure.csv", index=False)
    summary = df.groupby("method")[["lr_psnr", "grad_corr", "hf_ratio", "edge_prec", "edge_rec", "edge_f1"]].mean().round(4)
    summary.to_csv(args.results / "report" / "structure_summary.csv")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())