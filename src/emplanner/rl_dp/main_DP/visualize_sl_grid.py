#!/usr/bin/env python3
"""
Render the s-l sampling grid together with randomly generated obstacles.

Occupied grid points are highlighted in red so the spatial relationship between
obstacles and discrete grid cells is easy to inspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from rl_env import SLPathEnv
from sl_grid import (
    DEFAULT_L_RANGE,
    DEFAULT_L_SAMPLES,
    DEFAULT_S_RANGE,
    DEFAULT_S_SAMPLES,
    GridSpec,
)


def _parse_range(value: Sequence[float]) -> Tuple[float, float]:
    if len(value) != 2:
        raise argparse.ArgumentTypeError("range must have exactly two values")
    start, end = float(value[0]), float(value[1])
    if start >= end:
        raise argparse.ArgumentTypeError("range start must be smaller than end")
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the s-l grid, obstacles, and occupied grid points. "
            "Occupied cells are marked in red."
        )
    )
    parser.add_argument(
        "--s-range",
        nargs=2,
        type=float,
        default=DEFAULT_S_RANGE,
        metavar=("S_MIN", "S_MAX"),
        help="longitudinal range of the sampling grid",
    )
    parser.add_argument(
        "--l-range",
        nargs=2,
        type=float,
        default=DEFAULT_L_RANGE,
        metavar=("L_MIN", "L_MAX"),
        help="lateral range of the sampling grid",
    )
    parser.add_argument(
        "--s-samples",
        type=int,
        default=DEFAULT_S_SAMPLES,
        help="number of longitudinal samples (columns)",
    )
    parser.add_argument(
        "--l-samples",
        type=int,
        default=DEFAULT_L_SAMPLES,
        help="number of lateral samples (rows)",
    )
    parser.add_argument(
        "--max-obstacles",
        type=int,
        default=8,
        help="maximum number of random obstacles to generate",
    )
    parser.add_argument(
        "--length-range",
        nargs=2,
        type=float,
        default=(0.6, 1.8),
        metavar=("LEN_MIN", "LEN_MAX"),
        help="uniform range for obstacle length sampling",
    )
    parser.add_argument(
        "--width-range",
        nargs=2,
        type=float,
        default=(0.4, 1.4),
        metavar=("W_MIN", "W_MAX"),
        help="uniform range for obstacle width sampling",
    )
    parser.add_argument(
        "--collision-inflation",
        type=float,
        default=None,
        help="legacy shortcut：同时设置粗筛/细筛膨胀系数",
    )
    parser.add_argument(
        "--coarse-collision-inflation",
        type=float,
        default=0.2,
        help="粗筛 AABB 的膨胀系数（默认沿用环境默认或 --collision-inflation）",
    )
    parser.add_argument(
        "--fine-collision-inflation",
        type=float,
        default=0.2,
        help="细筛精确判定的膨胀系数（默认沿用环境默认或 --collision-inflation）",
    )
    parser.add_argument(
        "--start-clear-fraction",
        type=float,
        default=None,
        help="起点清障比例（默认沿用环境默认）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for obstacle placement",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional path to save the figure instead of showing it",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="figure DPI when saving to file",
    )
    return parser.parse_args()


def build_environment(args: argparse.Namespace) -> SLPathEnv:
    spec = GridSpec(
        s_range=_parse_range(args.s_range),
        l_range=_parse_range(args.l_range),
        s_samples=int(args.s_samples),
        l_samples=int(args.l_samples),
    )
    env_kwargs = {
        "seed": args.seed,
        "max_obstacles": int(args.max_obstacles),
        "obstacle_length_range": _parse_range(args.length_range),
        "obstacle_width_range": _parse_range(args.width_range),
    }
    if args.collision_inflation is not None:
        env_kwargs["collision_inflation"] = args.collision_inflation
    if args.coarse_collision_inflation is not None:
        env_kwargs["coarse_collision_inflation"] = args.coarse_collision_inflation
    if args.fine_collision_inflation is not None:
        env_kwargs["fine_collision_inflation"] = args.fine_collision_inflation
    if args.start_clear_fraction is not None:
        env_kwargs["start_clear_fraction"] = args.start_clear_fraction

    env = SLPathEnv(spec, **env_kwargs)
    return env


def main() -> None:
    args = parse_args()
    env = build_environment(args)
    observation = env.reset()

    occupancy = observation["occupancy"].astype(bool)
    s_coords = observation["s_coords"]
    l_coords = observation["l_coords"]

    s_grid, l_grid = np.meshgrid(s_coords, l_coords, indexing="ij")
    occupied_mask = occupancy.reshape(-1)
    free_mask = ~occupied_mask

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw grid lines for reference.
    for s_value in s_coords:
        ax.axvline(s_value, color="#d0d0d0", linewidth=0.6, zorder=0)
    for l_value in l_coords:
        ax.axhline(l_value, color="#d0d0d0", linewidth=0.6, zorder=0)

    # Scatter all grid points, coloring occupied cells in red.
    ax.scatter(
        s_grid.reshape(-1)[free_mask],
        l_grid.reshape(-1)[free_mask],
        c="#1f77b4",
        marker="s",
        s=80,
        edgecolors="none",
        alpha=0.75,
        label="Free grid point",
    )
    ax.scatter(
        s_grid.reshape(-1)[occupied_mask],
        l_grid.reshape(-1)[occupied_mask],
        c="#d62728",
        marker="s",
        s=120,
        edgecolors="k",
        linewidths=0.6,
        label="Occupied grid point",
    )

    # Draw obstacle polygons.
    for obstacle in env.obstacles:
        corners = obstacle.corners()
        patch = Polygon(
            corners,
            closed=True,
            facecolor="#ff9896",
            edgecolor="#c12f23",
            linewidth=1.0,
            alpha=0.6,
            label="Obstacle",
        )
        ax.add_patch(patch)

    # Avoid duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")

    ax.set_xlabel("s (longitudinal)")
    ax.set_ylabel("l (lateral)")
    ax.set_title("SL Grid with Obstacles and Occupancy")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(s_coords) - 0.5, max(s_coords) + 0.5)
    ax.set_ylim(min(l_coords) - 0.5, max(l_coords) + 0.5)
    ax.invert_yaxis()  # Match matrix-style orientation if desired.
    ax.grid(False)

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=args.dpi)
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
