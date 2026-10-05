import torch
from torch.nn import functional as F
from utils.config import load_config
from sr_datasets.sr_dataset import SRDataset

cfg = load_config("configs/cnn_v2.yaml")
ds = SRDataset(cfg, "train", "train")

scores = []
for i in range(200):
    s = ds[i]
    down = F.interpolate(s["hr"][None], scale_factor=0.25, mode="bicubic",
                         align_corners=False, antialias=True)[0].clamp(0, 1)
    mse = (down - s["lr"]).square().mean().item()
    scores.append(-10 * torch.log10(torch.tensor(max(mse, 1e-10))).item() * -1)

print("mean PSNR(down(HR), LR):", sum(scores) / len(scores))