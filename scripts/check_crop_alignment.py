import torch
from torch.nn import functional as F
from utils.config import load_config
from sr_datasets.sr_dataset import SRDataset


def mean_psnr(ds, n=200):
    scores = []
    step = max(len(ds) // n, 1)
    for i in range(0, len(ds), step):
        s = ds[i]
        down = F.interpolate(s["hr"][None], scale_factor=0.25, mode="bicubic",
                             align_corners=False, antialias=True)[0].clamp(0, 1)
        mse = (down - s["lr"]).square().mean().item()
        scores.append(-10 * torch.log10(torch.tensor(max(mse, 1e-10))).item())
        if len(scores) >= n:
            break
    return sum(scores) / len(scores)


if __name__ == "__main__":
    cfg = load_config("configs/esrgan_ucmerced.yaml")
    print("train crops (128, aug):", mean_psnr(SRDataset(cfg, "train", "train")))
    print("val full 256 (no crop):", mean_psnr(SRDataset(cfg, "val", "val")))
