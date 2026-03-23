#!/usr/bin/env python3
"""Generate paper-style figures from street batch benchmark case metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")

METHOD_LABELS = {
    "dijkstra": "Dijkstra",
    "astar": "A*",
    "greedy": "Greedy",
    "cnn_guided": "CNN-guided A*",
    "transformer_v3": "Transformer-guided A*",
}

METHOD_COLORS = {
    "dijkstra": "#6E7079",
    "astar": "#4C78A8",
    "greedy": "#E39C37",
    "cnn_guided": "#8F6BB3",
    "transformer_v3": "#D1495B",
}

GROUP_COLORS = {
    "all": "#4C78A8",
    "backend_success": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style figures from street batch benchmark.")
    parser.add_argument("--case-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="street_batch64_pub")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    val = float(value)
    return val


def valid_float(value: str) -> bool:
    try:
        return not math.isnan(float(value))
    except Exception:
        return False


def mean_of(rows: Sequence[Dict[str, str]], key: str) -> float:
    vals = [to_float(row[key]) for row in rows if key in row and valid_float(row[key])]
    return float(np.mean(vals)) if vals else math.nan


def setup_font():
    font_prop = None
    if CJK_FONT_PATH.exists():
        from matplotlib import font_manager as fm

        font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH))
        plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return font_prop


def style_axes(ax):
    ax.set_facecolor("#FBFBFA")
    ax.grid(True, axis="y", color="#D7DBE0", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#B0B4BB")


def apply_cjk(ax, font_prop):
    if font_prop is None:
        return
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    ax.title.set_fontproperties(font_prop)
    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)


def annotate_bars(ax, bars, fmt: str, fontsize: int = 11):
    for bar in bars:
        h = bar.get_height()
        if math.isnan(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(0.01 * max(1.0, ax.get_ylim()[1]), 0.01),
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#333333",
        )


def plot_exp1_v3_lengths(rows: Sequence[Dict[str, str]], out_path: Path, font_prop) -> Dict[str, float]:
    v3_rows = [row for row in rows if row["method"] == "transformer_v3"]
    v3_success = [row for row in v3_rows if int(row["backend_success"]) == 1]
    stages = ["raw_length_m", "seed_length_m", "smoothed_length_m"]
    stage_labels = ["前端路径", "初始优化路径", "优化路径"]
    succ_vals = [mean_of(v3_success, key) for key in stages]

    x = np.arange(len(stage_labels))
    fig, ax = plt.subplots(figsize=(8.8, 5.3), facecolor="#FBFBFA")
    style_axes(ax)
    bars_succ = ax.bar(
        x,
        succ_vals,
        width=0.56,
        color=GROUP_COLORS["backend_success"],
        label=f"后端成功场景 (n={len(v3_success)})",
    )
    ax.set_title("V3 后端成功场景下三阶段平均路径长度", fontsize=16, pad=12)
    ax.set_ylabel("平均长度 [m]", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=12)
    ax.legend(frameon=True, fontsize=12, loc="upper right")
    annotate_bars(ax, bars_succ, "{:.2f}")
    apply_cjk(ax, font_prop)
    fig.tight_layout()
    fig.savefig(out_path, dpi=350, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {
        "all_raw_length_m": mean_of(v3_rows, "raw_length_m"),
        "all_seed_length_m": mean_of(v3_rows, "seed_length_m"),
        "all_smoothed_length_m": mean_of(v3_rows, "smoothed_length_m"),
        "success_raw_length_m": succ_vals[0],
        "success_seed_length_m": succ_vals[1],
        "success_smoothed_length_m": succ_vals[2],
        "v3_cases": float(len(v3_rows)),
        "v3_backend_success_cases": float(len(v3_success)),
    }


def plot_exp2_backend_success(rows: Sequence[Dict[str, str]], out_path: Path, font_prop) -> List[Dict[str, float | str]]:
    methods = ["dijkstra", "astar", "greedy", "cnn_guided", "transformer_v3"]
    summary = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        rate = sum(int(row["backend_success"]) for row in group) / max(1, len(group))
        summary.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "backend_success_rate": rate,
            }
        )

    x = np.arange(len(summary))
    vals = [100.0 * float(row["backend_success_rate"]) for row in summary]
    colors = [METHOD_COLORS[str(row["method"])] for row in summary]
    fig, ax = plt.subplots(figsize=(9.2, 5.1), facecolor="#FBFBFA")
    style_axes(ax)
    bars = ax.bar(x, vals, color=colors, width=0.62)
    ax.set_title("不同前端方法的后端优化成功率", fontsize=16, pad=12)
    ax.set_ylabel("成功率 [%]", fontsize=14)
    ax.set_ylim(0.0, max(vals) * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["method_label"]) for row in summary], rotation=12, ha="right", fontsize=12)
    annotate_bars(ax, bars, "{:.1f}%")
    apply_cjk(ax, font_prop)
    fig.tight_layout()
    fig.savefig(out_path, dpi=350, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_exp3_v3_timing(rows: Sequence[Dict[str, str]], out_path: Path, font_prop) -> Dict[str, float]:
    v3_rows = [row for row in rows if row["method"] == "transformer_v3"]
    v3_success = [row for row in v3_rows if int(row["backend_success"]) == 1]
    metrics = [
        ("guidance_infer_ms", "模型推理"),
        ("frontend_search_ms", "前端搜索"),
        ("seed_stage_ms", "初始优化"),
        ("smooth_stage_ms", "后端优化"),
        ("pipeline_total_ms", "总耗时"),
    ]
    all_vals = [mean_of(v3_rows, key) for key, _ in metrics]
    succ_vals = [mean_of(v3_success, key) for key, _ in metrics]

    x = np.arange(len(metrics))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), facecolor="#FBFBFA")
    # Left: small-scale stages
    ax = axes[0]
    style_axes(ax)
    idx_small = np.array([0, 1, 2])
    labels_small = [metrics[i][1] for i in idx_small]
    bars_all_small = ax.bar(
        np.arange(len(idx_small)) - width / 2,
        [all_vals[i] for i in idx_small],
        width,
        color=GROUP_COLORS["all"],
        label=f"全部场景 (n={len(v3_rows)})",
    )
    bars_succ_small = ax.bar(
        np.arange(len(idx_small)) + width / 2,
        [succ_vals[i] for i in idx_small],
        width,
        color=GROUP_COLORS["backend_success"],
        label=f"后端成功场景 (n={len(v3_success)})",
    )
    ax.set_title("V3 前端与初始优化耗时", fontsize=15, pad=10)
    ax.set_ylabel("平均时间 [ms]", fontsize=13)
    ax.set_xticks(np.arange(len(idx_small)))
    ax.set_xticklabels(labels_small, fontsize=12)
    annotate_bars(ax, bars_all_small, "{:.2f}", fontsize=10)
    annotate_bars(ax, bars_succ_small, "{:.2f}", fontsize=10)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    apply_cjk(ax, font_prop)

    # Right: backend smoothing and total
    ax = axes[1]
    style_axes(ax)
    idx_large = np.array([3, 4])
    labels_large = [metrics[i][1] for i in idx_large]
    bars_all_large = ax.bar(
        np.arange(len(idx_large)) - width / 2,
        [all_vals[i] for i in idx_large],
        width,
        color=GROUP_COLORS["all"],
    )
    bars_succ_large = ax.bar(
        np.arange(len(idx_large)) + width / 2,
        [succ_vals[i] for i in idx_large],
        width,
        color=GROUP_COLORS["backend_success"],
    )
    ax.set_title("V3 后端与总耗时", fontsize=15, pad=10)
    ax.set_ylabel("平均时间 [ms]", fontsize=13)
    ax.set_xticks(np.arange(len(idx_large)))
    ax.set_xticklabels(labels_large, fontsize=12)
    annotate_bars(ax, bars_all_large, "{:.1f}", fontsize=10)
    annotate_bars(ax, bars_succ_large, "{:.1f}", fontsize=10)
    apply_cjk(ax, font_prop)

    fig.tight_layout()
    fig.savefig(out_path, dpi=350, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {
        "all_guidance_infer_ms": all_vals[0],
        "all_frontend_search_ms": all_vals[1],
        "all_seed_stage_ms": all_vals[2],
        "all_smooth_stage_ms": all_vals[3],
        "all_pipeline_total_ms": all_vals[4],
        "success_guidance_infer_ms": succ_vals[0],
        "success_frontend_search_ms": succ_vals[1],
        "success_seed_stage_ms": succ_vals[2],
        "success_smooth_stage_ms": succ_vals[3],
        "success_pipeline_total_ms": succ_vals[4],
    }


def write_tables(
    out_dir: Path,
    prefix: str,
    exp1: Dict[str, float],
    exp2: Sequence[Dict[str, float | str]],
    exp3: Dict[str, float],
) -> None:
    csv_path = out_dir / f"{prefix}_paper_metrics.csv"
    md_path = out_dir / f"{prefix}_paper_metrics.md"

    rows = []
    rows.extend(
        [
            {"experiment": "exp1_v3_lengths", "metric": key, "value": value}
            for key, value in exp1.items()
        ]
    )
    rows.extend(
        [
            {"experiment": "exp2_backend_success", "metric": row["method"], "value": row["backend_success_rate"]}
            for row in exp2
        ]
    )
    rows.extend(
        [
            {"experiment": "exp3_v3_timing", "metric": key, "value": value}
            for key, value in exp3.items()
        ]
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| 实验 | 指标 | 数值 |",
        "|---|---|---:|",
    ]
    for row in rows:
        val = float(row["value"])
        if "rate" in row["metric"]:
            pretty = f"{100.0 * val:.2f}%"
        else:
            pretty = f"{val:.4f}"
        lines.append(f"| {row['experiment']} | {row['metric']} | {pretty} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_prop = setup_font()
    rows = load_rows(args.case_metrics)

    exp1 = plot_exp1_v3_lengths(
        rows,
        args.output_dir / f"{args.prefix}_exp1_v3_length_compare_pub.png",
        font_prop,
    )
    exp2 = plot_exp2_backend_success(
        rows,
        args.output_dir / f"{args.prefix}_exp2_backend_success_rate_pub.png",
        font_prop,
    )
    exp3 = plot_exp3_v3_timing(
        rows,
        args.output_dir / f"{args.prefix}_exp3_v3_timing_breakdown_pub.png",
        font_prop,
    )
    write_tables(args.output_dir, args.prefix, exp1, exp2, exp3)
    print(f"saved_exp1={args.output_dir / (args.prefix + '_exp1_v3_length_compare_pub.png')}")
    print(f"saved_exp2={args.output_dir / (args.prefix + '_exp2_backend_success_rate_pub.png')}")
    print(f"saved_exp3={args.output_dir / (args.prefix + '_exp3_v3_timing_breakdown_pub.png')}")
    print(f"saved_metrics_csv={args.output_dir / (args.prefix + '_paper_metrics.csv')}")
    print(f"saved_metrics_md={args.output_dir / (args.prefix + '_paper_metrics.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
