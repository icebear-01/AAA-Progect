from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_CJK_FONT_PATH)
    _CJK_FONT_NAME = fm.FontProperties(fname=_CJK_FONT_PATH).get_name()
except Exception:
    _CJK_FONT_NAME = "DejaVu Sans"

plt.rcParams["font.family"] = _CJK_FONT_NAME
plt.rcParams["font.sans-serif"] = [_CJK_FONT_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


METHOD_ORDER = [
    "dijkstra",
    "traditional_astar",
    "improved_astar",
    "unet",
    "transformer_rebuild_v1",
]

DISPLAY_NAME = {
    "dijkstra": "Dijkstra",
    "traditional_astar": "A*",
    "improved_astar": "改进A*",
    "unet": "UNet",
    "transformer_rebuild_v1": "本文方法",
    "transformer_formal_v1": "旧版本文方法",
}

SCIENTIFIC_PALETTE = {
    "Dijkstra": "#0072B2",
    "A*": "#56B4E9",
    "改进A*": "#009E73",
    "UNet": "#E69F00",
    "本文方法": "#D55E00",
    "旧版本文方法": "#CC79A7",
}


def _draw_broken_axis_marks(
    ax_left: plt.Axes,
    ax_right: plt.Axes,
    *,
    color: str = "#475569",
    linewidth: float = 1.2,
    size: float = 0.012,
) -> None:
    """Draw aligned // markers on a broken x-axis."""
    kwargs = dict(color=color, clip_on=False, linewidth=linewidth, solid_capstyle="round")

    # Right spine of the left subplot.
    ax_left.plot((1 - size, 1 + size), (-size, +size), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=ax_left.transAxes, **kwargs)

    # Left spine of the right subplot.
    ax_right.plot((-size, +size), (-size, +size), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-size, +size), (1 - size, 1 + size), transform=ax_right.transAxes, **kwargs)


def _draw_broken_y_axis_marks(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    *,
    color: str = "#475569",
    linewidth: float = 1.2,
    size: float = 0.012,
) -> None:
    """Draw aligned // markers on a broken y-axis."""
    kwargs = dict(color=color, clip_on=False, linewidth=linewidth, solid_capstyle="round")

    ax_top.plot((-size, +size), (-size, +size), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - size, 1 + size), (-size, +size), transform=ax_top.transAxes, **kwargs)

    ax_bottom.plot((-size, +size), (1 - size, 1 + size), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=ax_bottom.transAxes, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成论文用栅格对比图。")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--case-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_main_metric_bar(
    output_path: Path,
    labels: List[str],
    values: List[float],
    *,
    xlabel: str,
    title: str,
    highlight_index: int,
    broken_axis: bool = False,
    value_suffix: str = "",
) -> None:
    y = np.arange(len(labels))
    colors = [SCIENTIFIC_PALETTE.get(label, "#94a3b8") for label in labels]
    colors[highlight_index] = SCIENTIFIC_PALETTE.get(labels[highlight_index], "#D55E00")

    if not broken_axis:
        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6, height=0.64)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        for yi, val in zip(y, values):
            text = f"{val:.2f}{value_suffix}" if value_suffix else f"{val:.2f}"
            ax.text(val + max(values) * 0.01, yi, text, va="center", ha="left", fontsize=10)

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    value_max = max(values)
    second_max = sorted(values)[-2]
    left_max = second_max * 1.12
    right_min = max(left_max * 2.5, value_max * 0.78)
    right_max = value_max * 1.08

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(10.2, 5.5),
        dpi=180,
        gridspec_kw={"width_ratios": [4.4, 1.5], "wspace": 0.06},
    )
    for ax in (ax_left, ax_right):
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.6, height=0.64)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    ax_left.set_xlim(0.0, left_max)
    ax_right.set_xlim(right_min, right_max)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels, fontsize=11)
    ax_left.invert_yaxis()
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_left.set_title(title, fontsize=14, weight="bold")
    fig.supxlabel(xlabel, fontsize=11)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    _draw_broken_axis_marks(ax_left, ax_right)

    left_pad = left_max * 0.015
    right_pad = (right_max - right_min) * 0.03
    for yi, val in zip(y, values):
        if val < right_min:
            text = f"{val:.2f}{value_suffix}" if value_suffix else f"{val:.2f}"
            ax_left.text(min(val + left_pad, left_max - left_pad), yi, text, va="center", ha="left", fontsize=10)
        else:
            text = f"{val:.2f}{value_suffix}" if value_suffix else f"{val:.2f}"
            ax_right.text(min(val + right_pad, right_max - right_pad), yi, text, va="center", ha="left", fontsize=10)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.9, bottom=0.13, wspace=0.06)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_vertical_main_metric_bar(
    output_path: Path,
    labels: List[str],
    values: List[float],
    *,
    ylabel: str,
    title: str,
    highlight_index: int,
    value_suffix: str = "",
) -> None:
    x = np.arange(len(labels))
    colors = [SCIENTIFIC_PALETTE.get(label, "#94a3b8") for label in labels]
    colors[highlight_index] = SCIENTIFIC_PALETTE.get(labels[highlight_index], "#D55E00")

    value_max = max(values)
    second_max = sorted(values)[-2]
    bottom_max = second_max * 1.16
    top_min = max(bottom_max * 2.5, value_max * 0.78)
    top_max = value_max * 1.08

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9.4, 7.6),
        dpi=180,
        gridspec_kw={"height_ratios": [1.45, 3.8], "hspace": 0.05},
    )
    for ax in (ax_top, ax_bottom):
        ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.7, width=0.64)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=23)

    ax_bottom.set_ylim(0.0, bottom_max)
    ax_top.set_ylim(top_min, top_max)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, fontsize=22)
    fig.supylabel(ylabel, fontsize=23)
    ax_top.set_title(title, fontsize=24, weight="bold", pad=12)
    _draw_broken_y_axis_marks(ax_top, ax_bottom)

    bottom_pad = bottom_max * 0.02
    top_pad = (top_max - top_min) * 0.04
    for xi, val in zip(x, values):
        text = f"{val:.2f}{value_suffix}" if value_suffix else f"{val:.2f}"
        if val < top_min:
            ax_bottom.text(xi, min(val + bottom_pad, bottom_max - bottom_pad), text, ha="center", va="bottom", fontsize=20)
        else:
            ax_top.text(xi, min(val + top_pad, top_max - top_pad), text, ha="center", va="bottom", fontsize=20)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.12, hspace=0.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_histogram(
    output_path: Path,
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    color: str = "#c2410c",
    zero_line: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    ax.hist(values, bins=24, color=color, edgecolor="white", alpha=0.9)
    if zero_line:
        ax.axvline(0.0, color="#0f172a", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("场景数量（个）", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_scatter(
    output_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=180)
    ax.scatter(x, y, s=18, alpha=0.75, color="#c2410c", edgecolors="none")
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#1e293b", linewidth=1.2)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_multi_method_scatter(
    output_path: Path,
    baseline: np.ndarray,
    method_series: List[tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=180)

    all_vals = [baseline]
    for _, values in method_series:
        all_vals.append(values)

    lo = min(float(np.min(vals)) for vals in all_vals)
    hi = max(float(np.max(vals)) for vals in all_vals)

    for label, values in method_series:
        ax.scatter(
            baseline,
            values,
            s=14,
            alpha=0.5,
            color=SCIENTIFIC_PALETTE.get(label, "#64748b"),
            edgecolors="none",
            label=label,
        )

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#334155", linewidth=1.1, alpha=0.9)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(linestyle="--", alpha=0.2)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_small_multiple_scatter(
    output_path: Path,
    baseline: np.ndarray,
    method_series: List[tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(1, len(method_series), figsize=(13.5, 4.5), dpi=180, sharex=True, sharey=True)
    if len(method_series) == 1:
        axes = [axes]

    all_vals = [baseline]
    for _, values in method_series:
        all_vals.append(values)
    lo = min(float(np.min(vals)) for vals in all_vals)
    hi = max(float(np.max(vals)) for vals in all_vals)

    for ax, (label, values) in zip(axes, method_series):
        ax.scatter(
            baseline,
            values,
            s=15,
            alpha=0.55,
            color=SCIENTIFIC_PALETTE.get(label, "#64748b"),
            edgecolors="none",
        )
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#334155", linewidth=1.1, alpha=0.9)
        ax.set_title(label, fontsize=20, weight="bold")
        ax.grid(linestyle="--", alpha=0.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="both", labelsize=18)

    axes[0].set_ylabel(ylabel, fontsize=20, labelpad=10)
    fig.supxlabel(xlabel, fontsize=20, y=0.045)
    fig.suptitle(title, fontsize=22, weight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.84, wspace=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_win_loss_bar(
    output_path: Path,
    better: int,
    equal: int,
    worse: int,
    *,
    title: str,
) -> None:
    labels = ["本文方法更优", "持平", "本文方法退化"]
    values = [better, equal, worse]
    colors = ["#c2410c", "#94a3b8", "#475569"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=180)
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("场景数量（个）", fontsize=11)
    ax.set_title(title, fontsize=14, weight="bold")
    for xi, val in zip(x, values):
        ax.text(xi, val + max(values) * 0.02, str(val), ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_ablation_bar(
    output_path: Path,
    labels: List[str],
    values: List[float],
    *,
    title: str,
    xlabel: str,
    value_suffix: str = "",
) -> None:
    colors = [
        SCIENTIFIC_PALETTE.get("UNet", "#94a3b8"),
        SCIENTIFIC_PALETTE.get("旧版本文方法", "#64748b"),
        SCIENTIFIC_PALETTE.get("本文方法", "#D55E00"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors[: len(labels)], height=0.62, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for yi, val in zip(y, values):
        text = f"{val:.2f}{value_suffix}" if value_suffix else f"{val:.2f}"
        ax.text(val + max(values) * 0.01, yi, text, va="center", ha="left", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _load_csv_rows(args.summary_csv)
    case_rows = _load_csv_rows(args.case_csv)
    summary_by_label = {row["label"]: row for row in summary_rows}

    main_rows = [summary_by_label[label] for label in METHOD_ORDER]
    labels = [DISPLAY_NAME[row["label"]] for row in main_rows]
    expanded = [float(row["expanded_nodes"]) for row in main_rows]
    runtime = [float(row["runtime_ms"]) for row in main_rows]
    path_length = [float(row["path_length"]) for row in main_rows]

    _save_vertical_main_metric_bar(
        args.output_dir / "01_expanded_nodes_main.png",
        labels,
        expanded,
        ylabel="平均扩展节点",
        title="扩展节点对比",
        highlight_index=labels.index("本文方法"),
        value_suffix="",
    )
    _save_vertical_main_metric_bar(
        args.output_dir / "02_runtime_main.png",
        labels,
        runtime,
        ylabel="平均搜索时间（ms）",
        title="搜索时间对比",
        highlight_index=labels.index("本文方法"),
        value_suffix="",
    )
    _save_main_metric_bar(
        args.output_dir / "03_path_length_main.png",
        labels,
        path_length,
        xlabel="路径长度（栅格单位）",
        title="路径长度对比",
        highlight_index=labels.index("本文方法"),
    )

    astar_exp = np.array([float(row["traditional_astar_expanded"]) for row in case_rows], dtype=np.float32)
    improved_exp = np.array([float(row["improved_astar_expanded"]) for row in case_rows], dtype=np.float32)
    unet_exp = np.array([float(row["unet_expanded"]) for row in case_rows], dtype=np.float32)
    v3_exp = np.array([float(row["transformer_rebuild_v1_expanded"]) for row in case_rows], dtype=np.float32)
    astar_runtime = np.array([float(row["traditional_astar_runtime_ms"]) for row in case_rows], dtype=np.float32)
    v3_runtime = np.array([float(row["transformer_rebuild_v1_runtime_ms"]) for row in case_rows], dtype=np.float32)
    astar_path = np.array([float(row["traditional_astar_path_length"]) for row in case_rows], dtype=np.float32)
    v3_path = np.array([float(row["transformer_rebuild_v1_path_length"]) for row in case_rows], dtype=np.float32)

    exp_gain = astar_exp - v3_exp
    runtime_gain = astar_runtime - v3_runtime
    path_delta = v3_path - astar_path

    _save_histogram(
        args.output_dir / "04_v3_vs_astar_expanded_gain_hist.png",
        exp_gain,
        title="本文方法相对A*的节点减少分布",
        xlabel="单个场景节点减少量 (A* - 本文方法)",
    )
    _save_histogram(
        args.output_dir / "05_v3_vs_astar_runtime_gain_hist.png",
        runtime_gain,
        title="本文方法相对A*的时间减少分布",
        xlabel="单个场景时间减少量 (ms) (A* - 本文方法)",
    )
    _save_histogram(
        args.output_dir / "06_v3_vs_astar_path_delta_hist.png",
        path_delta,
        title="本文方法相对A*的路径长度变化",
        xlabel="单个场景路径长度变化 (本文方法 - A*)",
        color="#2563eb",
    )
    _save_scatter(
        args.output_dir / "07_v3_vs_astar_expanded_scatter.png",
        astar_exp,
        v3_exp,
        title="逐场景扩展节点对比：A* 与 本文方法",
        xlabel="A* 扩展节点数",
        ylabel="本文方法扩展节点数",
    )
    _save_multi_method_scatter(
        args.output_dir / "10_astar_vs_methods_multi_scatter.png",
        astar_exp,
        [
            ("改进A*", improved_exp),
            ("UNet", unet_exp),
            ("本文方法", v3_exp),
        ],
        title="逐场景扩展节点对比：A* 与各方法",
        xlabel="A* 扩展节点数",
        ylabel="方法扩展节点数",
    )
    _save_small_multiple_scatter(
        args.output_dir / "11_astar_vs_methods_small_multiples.png",
        astar_exp,
        [
            ("改进A*", improved_exp),
            ("UNet", unet_exp),
            ("本文方法", v3_exp),
        ],
        title="相对A*的逐场景扩展节点对比",
        xlabel="A* 扩展节点数",
        ylabel="方法扩展节点数",
    )

    better = int(np.sum(v3_exp < astar_exp))
    equal = int(np.sum(v3_exp == astar_exp))
    worse = int(np.sum(v3_exp > astar_exp))
    _save_win_loss_bar(
        args.output_dir / "08_v3_win_tie_loss_vs_astar.png",
        better,
        equal,
        worse,
        title="逐场景胜平负统计：本文方法 vs A*",
    )

    ablation_rows = [
        summary_by_label["unet"],
        summary_by_label["transformer_formal_v1"],
        summary_by_label["transformer_rebuild_v1"],
    ]
    _save_ablation_bar(
        args.output_dir / "09_v3_ablation_expanded_nodes.png",
        [DISPLAY_NAME.get(row["label"], row["label"]) for row in ablation_rows],
        [float(row["expanded_nodes"]) for row in ablation_rows],
        title="消融实验：从UNet到本文方法",
        xlabel="扩展节点数（个）",
        value_suffix="个",
    )

    selected_table_path = args.output_dir / "paper_main_table.csv"
    with selected_table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "expanded_nodes", "runtime_ms", "path_length", "success_rate"],
        )
        writer.writeheader()
        for row in main_rows:
            writer.writerow(
                {
                    "method": DISPLAY_NAME[row["label"]],
                    "expanded_nodes": row["expanded_nodes"],
                    "runtime_ms": row["runtime_ms"],
                    "path_length": row["path_length"],
                    "success_rate": row["success_rate"],
                }
            )

    readme = args.output_dir / "README.md"
    baseline_summary = summary_by_label["traditional_astar"]
    v3_summary = summary_by_label["transformer_rebuild_v1"]
    reduction_pct = (
        (float(baseline_summary["expanded_nodes"]) - float(v3_summary["expanded_nodes"]))
        / float(baseline_summary["expanded_nodes"])
        * 100.0
    )
    readme.write_text(
        "\n".join(
            [
                "# 本文方法论文图",
                "",
                "数据集：`planning-datasets/data/mpd/all_064_moore_c16.npz`",
                "划分：`test`",
                "样本数：`400`",
                "评测设置：`octile`，`residual_weight=1.25`，`residual_confidence_mode=learned_spike`",
                "",
                "论文主图方法：",
                "- Dijkstra",
                "- A*",
                "- 改进A*",
                "- UNet",
                "- 本文方法（`transformer_rebuild_v1`）",
                "",
                f"关键结果：本文方法相对A*将扩展节点数降低了 {reduction_pct:.2f}%。",
                "",
                "文件说明：",
                "- `01_expanded_nodes_main.png`：扩展节点主对比图",
                "- `02_runtime_main.png`：搜索时间对比图",
                "- `03_path_length_main.png`：路径长度对比图",
                "- `04_v3_vs_astar_expanded_gain_hist.png`：逐场景节点减少分布",
                "- `05_v3_vs_astar_runtime_gain_hist.png`：逐场景时间减少分布",
                "- `06_v3_vs_astar_path_delta_hist.png`：逐场景路径长度变化分布",
                "- `07_v3_vs_astar_expanded_scatter.png`：A* 与本文方法逐场景散点图",
                "- `08_v3_win_tie_loss_vs_astar.png`：本文方法相对A*的胜平负统计",
                "- `09_v3_ablation_expanded_nodes.png`：UNet / 旧版本文方法 / 本文方法消融图",
                "- `10_astar_vs_methods_multi_scatter.png`：A* 与多方法逐场景散点图",
                "- `11_astar_vs_methods_small_multiples.png`：三联逐场景散点图",
                "- `paper_main_table.csv`：论文主表数值",
                "",
                "说明：",
                "- 仓库里没有同分布的 grid CNN checkpoint，因此这套栅格论文图未包含 CNN。",
            ]
        ),
        encoding="utf-8",
    )

    print(selected_table_path)
    print(readme)


if __name__ == "__main__":
    main()
