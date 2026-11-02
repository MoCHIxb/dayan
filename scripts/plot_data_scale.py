from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot audited real data-scale experiment results")
    parser.add_argument("--summary", default="experiments/data_scale_summary.json")
    parser.add_argument("--output-dir", default="experiments/figures")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / args.summary).read_text(encoding="utf-8"))
    rows = report["results"]
    sizes = [row["train_records"] for row in rows]
    if sizes != [50, 100, 200, 500, 2000, 10437]:
        raise ValueError("A complete, audited six-scale summary is required")
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, metric, label in zip(axes.flat, ["bleu", "chrf", "rougeL_f", "bertscore_f1"],
                                 ["BLEU", "chrF", "ROUGE-L F1", "BERTScore F1"]):
        for condition, caption, color in [("profile", "Adapter + profile", "#2166ac"),
                                           ("no_profile", "Adapter without profile", "#b35806")]:
            if metric == "bertscore_f1":
                values = [row["bertscore"]["condition_scores"][f"adapter_{condition}"]["f1"] for row in rows]
            else:
                values = [row["conditions"][condition][metric] for row in rows]
            ax.plot(sizes, values, marker="o", color=color, label=caption)
        ax.set_xscale("log")
        ax.set_xticks(sizes, ["50", "100", "200", "500", "2,000", "10,437"], rotation=25)
        ax.set_xlabel("Training records (log scale)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Qwen3-4B / LaMP-7: controlled data-scale experiment\n"
                 "LoRA r=16, alpha=32; fixed 2 epochs; seed=20251206; full test n=1,496",
                 fontsize=12)
    fig.savefig(output / "data_scale_test_metrics.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for epoch in (0, 1):
        ax.plot(sizes, [row["dev_losses"][epoch] for row in rows], marker="o", label=f"Epoch {epoch + 1}")
    ax.set_xscale("log")
    ax.set_xticks(sizes, ["50", "100", "200", "500", "2,000", "10,437"])
    ax.set_xlabel("Training records (log scale)")
    ax.set_ylabel("Dev loss (same fixed 500 records)")
    ax.set_title("Dev loss across data scales: one training seed, fixed 2 epochs")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "data_scale_dev_loss.png", dpi=200)
    plt.close(fig)
    print(f"Saved two scientific plots to {output}")


if __name__ == "__main__":
    main()
