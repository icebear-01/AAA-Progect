#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
except ImportError as exc:
    raise SystemExit(f"matplotlib import failed: {exc}")


PATH_CASES = [
    {"name": "straight_case_multi_obs_paper", "label": "直道多障碍"},
    {"name": "turn_60_deg_arc", "label": "60°弯道"},
    {"name": "curve_turn30_feasible", "label": "30°曲线"},
    {"name": "s_curve_x2_y2_demo", "label": "S形弯道"},
]

SPEED_CASES = [
    {"name": "st_single_dynamic_fast", "label": "单动态障碍"},
    {"name": "my_st_case_replay", "label": "跟驰复现"},
    {"name": "st_multi_default_check", "label": "多障碍巡航"},
    {"name": "st_crossing_straight_demo_tuned", "label": "横穿避让"},
]


def read_summary_txt(path):
    summary = {}
    if not path.exists():
        return summary
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            summary[key.strip()] = value.strip()
    return summary


def to_float(summary, key, default=0.0):
    raw = summary.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def pick_font_family(candidates):
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def build_text_font_properties():
    font_family = pick_font_family(
        [
            "AR PL UMing CN",
            "SimSun",
            "Songti SC",
            "STSong",
            "Noto Sans CJK SC",
        ]
    )
    if not font_family:
        return None
    return font_manager.FontProperties(family=font_family)


def apply_style():
    serif_family = pick_font_family(["AR PL UMing CN", "SimSun", "Songti SC", "STSong"])
    sans_family = pick_font_family(["Noto Sans CJK SC"])
    font_family = [name for name in [serif_family, sans_family, "DejaVu Serif", "DejaVu Sans"] if name]
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.20,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_case_records(results_root, case_specs):
    records = []
    for case_spec in case_specs:
        case_dir = results_root / case_spec["name"]
        summary = read_summary_txt(case_dir / "summary.txt")
        records.append(
            {
                "name": case_spec["name"],
                "label": case_spec["label"],
                "dir": case_dir,
                "summary": summary,
            }
        )
    return records


def add_case_title(ax, record, text_font, show_time=True, show_speed_time=False):
    summary = record["summary"]
    title_lines = [record["label"]]
    obstacle_count = summary.get("obstacle_count")
    if obstacle_count:
        title_lines.append(f"障碍物 {obstacle_count}")
    if show_time:
        total_ms = to_float(summary, "planner_total_ms", default=-1.0)
        if total_ms >= 0.0:
            title_lines.append(f"总耗时 {total_ms:.1f} ms")
    if show_speed_time:
        speed_ms = to_float(summary, "speed_planning_ms", default=-1.0)
        if speed_ms >= 0.0:
            title_lines.append(f"速度规划 {speed_ms:.1f} ms")
    ax.set_title("\n".join(title_lines), fontproperties=text_font, pad=6.0)


def plot_gallery(records, image_name, output_path, text_font, show_speed_time=False):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), constrained_layout=True)
    for ax, record in zip(axes.flat, records):
        image_path = record["dir"] / image_name
        if image_path.exists():
            image = plt.imread(str(image_path))
            ax.imshow(image)
        else:
            ax.text(
                0.5,
                0.5,
                f"missing\n{image_path.name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontproperties=text_font,
            )
        ax.set_axis_off()
        add_case_title(ax, record, text_font, show_time=True, show_speed_time=show_speed_time)
    fig.savefig(output_path, dpi=320)
    plt.close(fig)


def plot_runtime_breakdown(records, output_path, text_font):
    labels = [record["label"] for record in records]
    search_ms = [to_float(record["summary"], "dp_sampling_ms") for record in records]
    path_qp_ms = [to_float(record["summary"], "qp_optimization_ms") for record in records]
    speed_ms = [to_float(record["summary"], "speed_planning_ms") for record in records]
    total_ms = [to_float(record["summary"], "planner_total_ms") for record in records]

    positions = np.arange(len(records))
    width = 0.62

    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    ax.bar(positions, search_ms, width=width, color="#7A92A3", label="前端搜索")
    ax.bar(
        positions,
        path_qp_ms,
        width=width,
        bottom=search_ms,
        color="#C48E6B",
        label="路径QP",
    )
    ax.bar(
        positions,
        speed_ms,
        width=width,
        bottom=np.array(search_ms) + np.array(path_qp_ms),
        color="#8BAA97",
        label="速度DP+QP",
    )
    ax.plot(
        positions,
        total_ms,
        color="#8C6D8F",
        linewidth=1.8,
        marker="o",
        markersize=5.0,
        label="总耗时",
    )
    for x_pos, total in zip(positions, total_ms):
        ax.text(x_pos, total + 1.2, f"{total:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("time [ms]", fontproperties=text_font)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontproperties=text_font)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if text_font is not None:
        ax.legend(frameon=True, framealpha=0.92, prop=text_font, ncol=4, loc="upper left")
    else:
        ax.legend(frameon=True, framealpha=0.92, ncol=4, loc="upper left")
    fig.savefig(output_path, dpi=320)
    plt.close(fig)


def plot_curvature_comparison(records, output_path, text_font):
    labels = [record["label"] for record in records]
    dp_mean = [to_float(record["summary"], "dp_mean_abs_kappa") for record in records]
    qp_mean = [to_float(record["summary"], "qp_mean_abs_kappa") for record in records]
    dp_max = [to_float(record["summary"], "dp_max_abs_kappa") for record in records]
    qp_max = [to_float(record["summary"], "qp_max_abs_kappa") for record in records]

    positions = np.arange(len(records))
    bar_width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    subplot_specs = [
        (axes[0], dp_mean, qp_mean, "mean |kappa|"),
        (axes[1], dp_max, qp_max, "max |kappa|"),
    ]
    for ax, dp_values, qp_values, ylabel in subplot_specs:
        ax.bar(
            positions - bar_width / 2.0,
            dp_values,
            width=bar_width,
            color="#7A92A3",
            label="决策路径",
        )
        ax.bar(
            positions + bar_width / 2.0,
            qp_values,
            width=bar_width,
            color="#C48E6B",
            label="QP优化后",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontproperties=text_font)
        ax.set_ylabel(ylabel, fontproperties=text_font)
        ax.grid(True, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels_out = axes[0].get_legend_handles_labels()
    if text_font is not None:
        fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=True, framealpha=0.92, prop=text_font)
    else:
        fig.legend(handles, labels_out, loc="upper center", ncol=2, frameon=True, framealpha=0.92)
    fig.savefig(output_path, dpi=320)
    plt.close(fig)


def write_metrics_csv(output_path, path_records, speed_records):
    fields = [
        "group",
        "case_name",
        "case_label",
        "obstacle_count",
        "planner_total_ms",
        "dp_sampling_ms",
        "qp_optimization_ms",
        "speed_planning_ms",
        "dp_mean_abs_kappa",
        "qp_mean_abs_kappa",
        "dp_max_abs_kappa",
        "qp_max_abs_kappa",
        "dp_source",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, records in [("path_gallery", path_records), ("speed_gallery", speed_records)]:
            for record in records:
                summary = record["summary"]
                writer.writerow(
                    {
                        "group": group,
                        "case_name": record["name"],
                        "case_label": record["label"],
                        "obstacle_count": summary.get("obstacle_count", ""),
                        "planner_total_ms": summary.get("planner_total_ms", ""),
                        "dp_sampling_ms": summary.get("dp_sampling_ms", ""),
                        "qp_optimization_ms": summary.get("qp_optimization_ms", ""),
                        "speed_planning_ms": summary.get("speed_planning_ms", ""),
                        "dp_mean_abs_kappa": summary.get("dp_mean_abs_kappa", ""),
                        "qp_mean_abs_kappa": summary.get("qp_mean_abs_kappa", ""),
                        "dp_max_abs_kappa": summary.get("dp_max_abs_kappa", ""),
                        "qp_max_abs_kappa": summary.get("qp_max_abs_kappa", ""),
                        "dp_source": summary.get("dp_source", ""),
                    }
                )


def main():
    script_dir = Path(__file__).resolve().parent
    emplanner_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Generate paper-ready experiment summary figures.")
    parser.add_argument(
        "--results-root",
        default=str(emplanner_dir / "benchmark_results"),
        help="Directory containing benchmark result folders",
    )
    parser.add_argument(
        "--output-dir",
        default=str(emplanner_dir / "benchmark_results_paper_res" / "experiment_figures"),
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    text_font = build_text_font_properties()

    path_records = load_case_records(results_root, PATH_CASES)
    speed_records = load_case_records(results_root, SPEED_CASES)

    plot_gallery(
        path_records,
        image_name="comparison_paper.png",
        output_path=output_dir / "path_case_gallery.png",
        text_font=text_font,
        show_speed_time=False,
    )
    plot_gallery(
        speed_records,
        image_name="st_graph_paper.png",
        output_path=output_dir / "speed_case_gallery.png",
        text_font=text_font,
        show_speed_time=True,
    )
    plot_runtime_breakdown(
        speed_records,
        output_path=output_dir / "runtime_breakdown.png",
        text_font=text_font,
    )
    plot_curvature_comparison(
        path_records,
        output_path=output_dir / "curvature_comparison.png",
        text_font=text_font,
    )
    write_metrics_csv(output_dir / "selected_case_metrics.csv", path_records, speed_records)

    print(f"saved={output_dir / 'path_case_gallery.png'}")
    print(f"saved={output_dir / 'speed_case_gallery.png'}")
    print(f"saved={output_dir / 'runtime_breakdown.png'}")
    print(f"saved={output_dir / 'curvature_comparison.png'}")
    print(f"saved={output_dir / 'selected_case_metrics.csv'}")


if __name__ == "__main__":
    main()
