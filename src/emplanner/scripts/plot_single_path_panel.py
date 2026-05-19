#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle


def pick_font():
    preferred = [
        "Noto Serif CJK SC",
        "Noto Sans CJK SC",
        "AR PL UMing CN",
        "AR PL UKai CN",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return font_manager.FontProperties(family=name)
    return None


def obstacle_polygon(obstacle):
    cx = float(obstacle["center_x"])
    cy = float(obstacle["center_y"])
    length = float(obstacle["length"])
    width = float(obstacle["width"])
    yaw = float(obstacle["yaw"])
    half_l = 0.5 * length
    half_w = 0.5 * width
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = corners @ rot.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


def vehicle_polygon(center_x, center_y, yaw, length, width):
    half_l = 0.5 * length
    half_w = 0.5 * width
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = corners @ rot.T
    world[:, 0] += center_x
    world[:, 1] += center_y
    return world


def style_axes(ax, tick_size):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_size, width=1.1, length=5.5, colors="#202124")
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color("#202124")
    ax.spines["bottom"].set_color("#202124")


def add_corner_axis_labels(ax, label_size, font_prop):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        1.01,
        -0.015,
        "x",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=label_size,
        fontproperties=font_prop,
        color="#202124",
        clip_on=False,
    )
    ax.text(
        -0.02,
        1.01,
        "y",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=label_size,
        fontproperties=font_prop,
        color="#202124",
        clip_on=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-abs-max", type=float, default=3.0)
    parser.add_argument("--start-vehicle-length", type=float, default=0.9)
    parser.add_argument("--start-vehicle-width", type=float, default=0.6)
    parser.add_argument("--tick-size", type=float, default=20.0)
    parser.add_argument("--legend-size", type=float, default=18.0)
    parser.add_argument("--axis-label-size", type=float, default=22.0)
    parser.add_argument("--hide-axis-corner-labels", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf) if args.output_pdf else None
    output_png.parent.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(input_dir / "reference_path.csv")
    dp = pd.read_csv(input_dir / "dp_path.csv")
    qp = pd.read_csv(input_dir / "qp_path.csv")
    obstacles = pd.read_csv(input_dir / "obstacles.csv")
    grid = pd.read_csv(input_dir / "dp_grid_points.csv")

    font_prop = pick_font()

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "legend.fontsize": 16,
            "xtick.labelsize": max(18, args.tick_size),
            "ytick.labelsize": max(18, args.tick_size),
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.22,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=220)
    fig.patch.set_facecolor("white")

    ax.scatter(
        grid["x"].to_numpy(),
        grid["y"].to_numpy(),
        s=30,
        facecolors="none",
        edgecolors="#8a939c",
        linewidths=1.35,
        alpha=0.95,
        label="采样栅格点",
        zorder=1,
    )
    ax.plot(
        reference["x"].to_numpy(),
        reference["y"].to_numpy(),
        color="#4c566a",
        linestyle="--",
        linewidth=1.3,
        label="全局路径",
        zorder=2,
    )
    ax.plot(dp["x"].to_numpy(), dp["y"].to_numpy(), color="#1f5aa6", linewidth=2.1, label="决策路径", zorder=3)
    ax.plot(qp["x"].to_numpy(), qp["y"].to_numpy(), color="#d04a02", linewidth=2.1, label="优化路径", zorder=4)

    for _, obstacle in obstacles.iterrows():
        poly = obstacle_polygon(obstacle)
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="#9aa0a6",
                edgecolor="#5f6368",
                linewidth=0.9,
                alpha=0.35,
                zorder=2,
            )
        )

    if not qp.empty and args.start_vehicle_length > 0 and args.start_vehicle_width > 0:
        start_poly = vehicle_polygon(
            float(qp.iloc[0]["x"]),
            float(qp.iloc[0]["y"]),
            float(qp.iloc[0]["yaw"]),
            args.start_vehicle_length,
            args.start_vehicle_width,
        )
        ax.add_patch(
            Polygon(
                start_poly,
                closed=True,
                facecolor="#f8f9fa",
                edgecolor="#202124",
                linewidth=1.2,
                alpha=0.96,
                zorder=4.8,
            )
        )

    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(-args.y_abs_max, args.y_abs_max)
    ax.set_xticks(np.arange(math.floor(args.x_min), math.ceil(args.x_max) + 1, 1))
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax, args.tick_size)
    if not args.hide_axis_corner_labels:
        add_corner_axis_labels(ax, args.axis_label_size, font_prop)

    legend_handles = [
        Line2D([], [], color="#4c566a", linestyle="--", linewidth=1.5, label="全局路径"),
        Line2D([], [], color="#1f5aa6", linewidth=2.6, label="决策路径"),
        Line2D([], [], color="#d04a02", linewidth=2.8, label="优化路径"),
        Rectangle((0, 0), 1, 0.65, facecolor="#9aa0a6", edgecolor="#5f6368", linewidth=1.0, alpha=0.35, label="障碍物"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        fontsize=args.legend_size,
        columnspacing=1.2,
        handlelength=2.0,
        borderaxespad=0.0,
        prop=font_prop,
    )
    legend.get_frame().set_edgecolor("#d0d7de")
    legend.get_frame().set_linewidth(0.9)

    fig.tight_layout(pad=0.8)
    fig.savefig(output_png, bbox_inches="tight")
    if output_pdf:
        try:
            fig.savefig(output_pdf, bbox_inches="tight")
        except Exception as exc:  # pragma: no cover - best effort export
            print(f"Warning: failed to save PDF: {exc}")
    plt.close(fig)


if __name__ == "__main__":
    main()
