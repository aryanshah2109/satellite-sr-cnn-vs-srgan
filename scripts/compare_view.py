import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from inference.super_resolve import _input_tensor, load_model_for_inference, super_resolve_tensor
from utils.config import load_config
from utils.image_utils import get_device


def img(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=Path, required=True)
    p.add_argument("--hr", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("results_final/demo/compare.png"))
    p.add_argument("--zoom", type=int, default=32)
    args = p.parse_args()
    cfg = load_config("configs/evaluation.yaml")
    device = get_device("auto")
    methods = {str(m.name): m for m in cfg.methods}
    models = {
        n: load_model_for_inference(str(methods[n].type), str(methods[n].checkpoint), str(methods[n].model_config), device)
        for n in ("cnn_v2", "srgan_v2_finetuned")
    }
    lr = _input_tensor(args.lr, cfg)
    panels = [
        ("LR (nearest)", F.interpolate(lr[None], scale_factor=4, mode="nearest")[0]),
        ("Bicubic", F.interpolate(lr[None], scale_factor=4, mode="bicubic", align_corners=False)[0]),
    ]
    for name, model in models.items():
        out = super_resolve_tensor(model, lr, int(cfg.inference.tile_size), int(cfg.inference.overlap), device, False)
        panels.append((name, out.cpu()))
    if args.hr is not None:
        panels.append(("HR (ground truth)", _input_tensor(args.hr, cfg)))
    h, w = panels[0][1].shape[-2:]
    z = args.zoom * 4
    top, left = (h - z) // 2, (w - z) // 2
    fig, axes = plt.subplots(2, len(panels), figsize=(4 * len(panels), 8))
    for i, (title, t) in enumerate(panels):
        axes[0, i].imshow(img(t))
        axes[0, i].set_title(title)
        axes[0, i].axis("off")
        axes[1, i].imshow(img(t)[top : top + z, left : left + z])
        axes[1, i].axis("off")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())