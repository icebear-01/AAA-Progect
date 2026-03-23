#!/usr/bin/env python3
"""Publication-style plots for frontend-only street architecture ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")

METHOD_ORDER = [
    "astar",
    "cnn_guided",
    "unet_guided",
    "v1_guided",
    "v2_guided",
    "transformer_v3",
]

METHOD_COLORS: Dict[str, str] = {
    "astar": "#4C78A8",
    "cnn_guided": "#8E6C8A",
    "unet_guided": "#72B7B2",
    "v1_guided": "#6C8EBF",
    "v2_guided": "#9C755F",
    "transformer_v3": "#D95F5F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frontend-only street ablation figures.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="street_frontend_ablation400")
    return parser.parse_args()


def load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    order = {key: idx for idx, key in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda row: order.get(row["method"], 999))
    return rows


def cjk_font():
    if not CJK_FONT_PATH.exists():
        return None
    from matplotlib import font_manager as fm

    return fm.FontProperties(fname=str(CJK_FONT_PATH))


def style_axes(ax, font_prop) -> None:
    ax.set_facecolor("#FBFBFA")
    ax.grid(True, axis="y", color="#D8DEE6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#A9B0B7")
    if font_prop is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)


def annotate_bars(ax, bars, fmt: str, fontsize: int = 10) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            height,
            format(height, fmt),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#30343B",
        )


def plot_main(summary: List[Dict[str, str]], out_path: Path) -> None:
    font_prop = cjk_font()
    labels = [row["method_label"] for row in summary]
    colors = [METHOD_COLORS[row["method"]] for row in summary]
    expanded = [float(row["expanded_nodes"]) for row in summary]
    frontend_total = [float(row["frontend_total_ms"]) for row in summary]
    raw_length = [float(row["raw_length_m"]) for row in summary]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5), facecolor="#FBFBFA")
    titles = ["平均扩展节点", "平均前端耗时", "平均前端路径长度"]
    ylabels = ["节点", "时间 [ms]", "长度 [m]"]
    values = [expanded, frontend_total, raw_length]
    formats = [".2f", ".2f", ".2f"]

    for ax, title, ylabel, series, fmt in zip(axes, titles, ylabels, values, formats):
        bars = ax.bar(x, series, color=colors, width=0.68, alpha=0.95)
        ax.set_title(title, fontproperties=font_prop, fontsize=18, pad=10)
        ax.set_ylabel(ylabel, fontproperties=font_prop, fontsize=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.tick_params(axis="both", labelsize=12)
        style_axes(ax, font_prop)
        annotate_bars(ax, bars, fmt)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_timing(summary: List[Dict[str, str]], out_path: Path) -> None:
    font_prop = cjk_font()
    labels = [row["method_label"] for row in summary]
    colors = [METHOD_COLORS[row["method"]] for row in summary]
    infer_ms = [float(row["guidance_infer_ms"]) for row in summary]
    search_ms = [float(row["frontend_search_ms"]) for row in summary]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), facecolor="#FBFBFA")
    bars_infer = ax.bar(x, infer_ms, color="#F2C14E", width=0.68, alpha=0.92, label="模型推理")
    bars_search = ax.bar(x, search_ms, bottom=infer_ms, color=colors, width=0.68, alpha=0.95, label="前端搜索")
    ax.set_title("前端耗时分解", fontproperties=font_prop, fontsize=18, pad=10)
    ax.set_ylabel("时间 [ms]", fontproperties=font_prop, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.tick_params(axis="both", labelsize=12)
    style_axes(ax, font_prop)
    legend = ax.legend(loc="upper left", frameon=True, fontsize=12)
    if font_prop is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
    for infer_bar, search_bar in zip(bars_infer, bars_search):
        total = infer_bar.get_height() + search_bar.get_height()
        ax.text(
            infer_bar.get_x() + infer_bar.get_width() * 0.5,
            total,
            f"{total:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#30343B",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = load_summary(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_main(summary, args.output_dir / f"{args.prefix}_frontend_ablation_main_pub.png")
    plot_timing(summary, args.output_dir / f"{args.prefix}_frontend_ablation_timing_pub.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
