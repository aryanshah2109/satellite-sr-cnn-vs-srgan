import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [
    ("cnn", "bicubic"),
    ("srgan_finetuned", "bicubic"),
    ("srgan_finetuned", "cnn"),
    ("srgan_finetuned", "srgan_pretrained"),
    ("srgan_finetuned", "srgan_pixel_only"),
    ("cnn_v2", "cnn"),
    ("srgan_v2_finetuned", "srgan_finetuned"),
    ("srgan_v2_finetuned", "cnn_v2"),
    ("srgan_v2_finetuned", "srgan_v2_pixel_only"),
    ("srgan_v2_finetuned", "srgan_pretrained"),
]
METRICS = ("psnr_y", "ssim_y", "lpips")
THRESHOLD = {"psnr_y": 0.1, "ssim_y": 0.005, "lpips": 0.01}


def cluster_bootstrap(diff, tiles, resamples, seed):
    codes, _ = pd.factorize(tiles)
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    k = len(sums)
    idx = rng.integers(0, k, size=(resamples, k))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(diff.mean()), float(low), float(high), k


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results_final"))
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    metrics_dir = args.results / "metrics"
    df = pd.read_csv(metrics_dir / "per_image.csv")
    tile_map = df.drop_duplicates("patch_name").set_index("patch_name")["tile_id"]
    rows = []
    for metric in METRICS:
        wide = df.pivot(index="patch_name", columns="method", values=metric)
        tiles = tile_map.reindex(wide.index).to_numpy()
        for a, b in PAIRS:
            if a not in wide.columns or b not in wide.columns:
                continue
            diff = (wide[a] - wide[b]).to_numpy()
            mean, low, high, n_tiles = cluster_bootstrap(diff, tiles, args.resamples, 42)
            excludes_zero = bool(low > 0 or high < 0)
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "metric": metric,
                    "n_patches": len(diff),
                    "n_tiles": n_tiles,
                    "diff_a_minus_b": round(mean, 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                    "excludes_zero": excludes_zero,
                    "practically_meaningful": excludes_zero and abs(mean) >= THRESHOLD[metric],
                }
            )
    out = pd.DataFrame(rows)
    destination = args.results / "report" / "significance_cluster.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    print(out.to_string(index=False))
    print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())