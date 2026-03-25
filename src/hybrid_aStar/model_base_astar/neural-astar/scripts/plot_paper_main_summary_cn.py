"""Plot a Chinese paper-style summary chart from summary_metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

PAPER_COLORS = ["#0072B2", "#E69F00", "#009E73"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Chinese summary bars for A* comparison results.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Path to summary_metrics.csv")
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG path")
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        help="Chinese font file used by matplotlib",
    )
    return parser.parse_args()


def load_metric_rows(summary_csv: Path) -> dict[str, dict[str, str]]:
    with summary_csv.open() as f:
        rows = list(csv.DictReader(f))
    return {row["metric"]: row for row in rows}


def main() -> None:
    args = parse_args()
    metric_rows = load_metric_rows(args.summary_csv)
    font_prop = font_manager.FontProperties(fname=str(args.font_path))

    labels = ["传统A*", "改进A*", "本文方法"]
    metric_keys = ["traditional", "improved", "learned"]
    expanded_nodes = [float(metric_rows["expanded_nodes_mean"][key]) for key in metric_keys]
    runtime_ms = [float(metric_rows["runtime_ms_mean"][key]) for key in metric_keys]
    colors = PAPER_COLORS

    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].bar(range(len(labels)), expanded_nodes, color=colors)
    axes[0].set_title("平均扩展节点", fontproperties=font_prop)
    axes[0].set_ylabel("节点", fontproperties=font_prop)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=8)
    for tick in axes[0].get_xticklabels():
        tick.set_fontproperties(font_prop)
    for idx, value in enumerate(expanded_nodes):
        axes[0].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(range(len(labels)), runtime_ms, color=colors)
    axes[1].set_title("平均运行时间", fontproperties=font_prop)
    axes[1].set_ylabel("时间（ms）", fontproperties=font_prop)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=8)
    for tick in axes[1].get_xticklabels():
        tick.set_fontproperties(font_prop)
    for idx, value in enumerate(runtime_ms):
        axes[1].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180)


if __name__ == "__main__":
    main()
