import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results_final"))
    parser.add_argument("--history", type=Path, default=Path("docs/experiment_history.json"))
    args = parser.parse_args()
    metrics = args.results / "metrics"
    out = args.results / "report"
    out.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(metrics / "results.csv")
    results = results.sort_values("mean_lpips")
    results[["method", "mean_psnr_y", "mean_ssim_y", "mean_lpips"]].round(4).to_csv(
        out / "summary.csv", index=False
    )

    sig = pd.read_csv(metrics / "significance.csv")
    sig["significant"] = (sig["ci_low"] > 0) | (sig["ci_high"] < 0)
    sig.round(4).to_csv(out / "significance_flagged.csv", index=False)

    city = pd.read_csv(metrics / "per_city.csv")
    for metric in ("mean_psnr_y", "mean_lpips"):
        pivot = city.pivot(index="city", columns="method", values=metric)
        pivot.round(4).to_csv(out / f"per_city_{metric}.csv")

    per_image = pd.read_csv(metrics / "per_image.csv")
    wide = per_image.pivot(index="patch_name", columns="method", values=["psnr_y", "lpips"])
    figure, axis = plt.subplots(figsize=(8, 6))
    for method in results["method"]:
        axis.scatter(
            results.loc[results["method"] == method, "mean_psnr_y"],
            results.loc[results["method"] == method, "mean_lpips"],
            s=80,
            label=method,
        )
    axis.set_xlabel("PSNR_Y (higher is better)")
    axis.set_ylabel("LPIPS (lower is better)")
    axis.set_title("Perception-distortion tradeoff (test mean)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "tradeoff.png", dpi=200)
    plt.close(figure)

    if "srgan_v2_finetuned" in wide["psnr_y"].columns and "cnn_v2" in wide["psnr_y"].columns:
        gap = wide["psnr_y"]["cnn_v2"] - wide["psnr_y"]["srgan_v2_finetuned"]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(gap.dropna(), bins=60)
        axis.axvline(0, color="k", linewidth=1)
        axis.set_xlabel("PSNR gap: cnn_v2 minus srgan_v2_finetuned (dB)")
        axis.set_ylabel("patches")
        figure.tight_layout()
        figure.savefig(out / "psnr_gap_hist.png", dpi=200)
        plt.close(figure)

    if args.history.is_file():
        history = json.loads(args.history.read_text(encoding="utf-8"))
        rows = [
            {
                "run_name": item["run_name"],
                "stage": item["stage"],
                "epochs": item["epochs"],
                "minutes": round(float(item["elapsed_seconds"]) / 60.0, 1),
            }
            for item in history
            if item.get("elapsed_seconds") is not None
        ]
        frame = pd.DataFrame(rows)
        frame = frame[~frame["run_name"].str.startswith("smoke") & (frame["minutes"] > 1.0)]
        frame.to_csv(out / "training_time.csv", index=False)

    print(f"Report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())