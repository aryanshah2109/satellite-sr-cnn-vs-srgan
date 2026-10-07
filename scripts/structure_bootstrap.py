import argparse
from pathlib import Path

import pandas as pd

from scripts.cluster_bootstrap import cluster_bootstrap

PAIRS = [
    ("srgan_finetuned", "cnn_v2"),
    ("srgan_v2_finetuned", "cnn_v2"),
    ("srgan_v2_finetuned", "srgan_v2_pixel_only"),
    ("srgan_v2_finetuned", "srgan_finetuned"),
]
METRICS = ("lr_psnr", "grad_corr", "hf_ratio", "edge_prec", "edge_rec", "edge_f1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results_final"))
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    df = pd.read_csv(args.results / "metrics" / "structure.csv")
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
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "metric": metric,
                    "n_tiles": n_tiles,
                    "diff_a_minus_b": round(mean, 4),
                    "ci_low": round(low, 4),
                    "ci_high": round(high, 4),
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    out = pd.DataFrame(rows)
    destination = args.results / "report" / "structure_significance.csv"
    out.to_csv(destination, index=False)
    print(out.to_string(index=False))
    print(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())