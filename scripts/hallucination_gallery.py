import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from inference.super_resolve import load_model_for_inference
from sr_datasets.sr_dataset import SRDataset
from utils.config import load_config
from utils.image_utils import get_device


def to_img(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/ucmerced_v2"))
    parser.add_argument("--first", default="esrgan_ucmerced")
    parser.add_argument("--second", default="srgan_ucmerced")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config("configs/evaluation_ucmerced.yaml")
    device = get_device("auto")
    methods = {str(m.name): m for m in cfg.methods}
    models = {
        name: load_model_for_inference(
            str(methods[name].type), str(methods[name].checkpoint), str(methods[name].model_config), device
        )
        for name in (args.first, args.second)
    }
    per = pd.read_csv(args.results / "metrics" / "per_image.csv")
    lpips = per.pivot(index="patch_name", columns="method", values="lpips")
    chosen = (lpips[args.first] - lpips[args.second]).abs().nlargest(args.top).index.tolist()
    dataset = SRDataset(cfg, "test", "test")
    index = {row["patch_name"]: i for i, row in enumerate(dataset.rows)}
    out_dir = args.results / "report" / "hallucination_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        sample = dataset[index[name]]
        lr = sample["lr"][None].to(device)
        hr = sample["hr"]
        with torch.no_grad():
            first = models[args.first](lr).float().clamp(0, 1)[0].cpu()
            second = models[args.second](lr).float().clamp(0, 1)[0].cpu()
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        panels = [
            ("LR (nearest)", torch.nn.functional.interpolate(lr.cpu(), scale_factor=4)[0]),
            (f"{args.first}\nLPIPS {lpips.loc[name, args.first]:.3f}", first),
            (f"{args.second}\nLPIPS {lpips.loc[name, args.second]:.3f}", second),
            ("HR", hr),
        ]
        for col, (title, tensor) in enumerate(panels):
            axes[0, col].imshow(to_img(tensor))
            axes[0, col].set_title(title, fontsize=9)
            axes[0, col].axis("off")
            if col in (1, 2):
                axes[1, col].imshow((tensor - hr).abs().mean(0).numpy(), cmap="magma", vmin=0, vmax=0.4)
            else:
                axes[1, col].imshow(to_img(tensor)[96:160, 96:160])
            axes[1, col].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{Path(name).stem}.png", dpi=150)
        plt.close(fig)
    print(f"Saved {len(chosen)} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
