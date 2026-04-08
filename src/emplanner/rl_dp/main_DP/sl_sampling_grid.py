#!/usr/bin/env python3
"""
Generate and visualize an s-l sampling grid with optional random obstacles.

The grid always includes the origin (0, 0). By default the script samples
the shared training grid defaults: 9 points along the longitudinal `s` axis
and 23 points along the lateral `l` axis, with an exact 0.35 m lateral spacing
across -3.85 to 3.85 m. Both axis ranges and the number of samples are
configurable through the CLI flags.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from sl_grid import (
    DEFAULT_L_RANGE,
    DEFAULT_L_SAMPLES,
    DEFAULT_S_RANGE,
    DEFAULT_S_SAMPLES,
    GridSpec,
    build_grid,
)
from sl_obstacles import Obstacle, generate_random_obstacles


def plot_grid(spec: GridSpec, *, obstacles: Optional[Sequence[Obstacle]] = None) -> None:
    """Plot the grid and display it."""
    s_grid, l_grid = build_grid(spec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(s_grid, l_grid, c="tab:blue", edgecolors="white", zorder=3)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    if obstacles:
        for obstacle in obstacles:
            polygon = Polygon(
                obstacle.corners(),
                closed=True,
                facecolor="tab:orange",
                edgecolor="tab:red",
                linewidth=1.2,
                alpha=0.4,
                zorder=2,
            )
            ax.add_patch(polygon)

    ax.set_xlabel("s (longitudinal)")
    ax.set_ylabel("l (lateral)")
    title = f"s-l Sampling Grid ({spec.s_samples} × {spec.l_samples} points)"
    if obstacles:
        title += f" with {len(obstacles)} obstacle(s)"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an s-l sampling grid.")
    parser.add_argument(
        "--s-range",
        type=float,
        nargs=2,
        metavar=("S_MIN", "S_MAX"),
        default=DEFAULT_S_RANGE,
        help="longitudinal axis range; must include 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--l-range",
        type=float,
        nargs=2,
        metavar=("L_MIN", "L_MAX"),
        default=DEFAULT_L_RANGE,
        help="lateral axis range; must include 0 (default: %(default)s)",
    )
    parser.add_argument(
        "--s-samples",
        type=int,
        default=DEFAULT_S_SAMPLES,
        help="number of samples along s (default: %(default)s)",
    )
    parser.add_argument(
        "--l-samples",
        type=int,
        default=DEFAULT_L_SAMPLES,
        help="number of samples along l (default: %(default)s)",
    )
    parser.add_argument(
        "--max-obstacles",
        type=int,
        default=10,
        help="maximum number of random obstacles to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible obstacle placement (default: random)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = GridSpec(
        s_range=(args.s_range[0], args.s_range[1]),
        l_range=(args.l_range[0], args.l_range[1]),
        s_samples=args.s_samples,
        l_samples=args.l_samples,
    )
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    obstacles: Iterable[Obstacle]
    if args.max_obstacles > 0:
        obstacles = generate_random_obstacles(
            spec.s_range,
            spec.l_range,
            max_count=args.max_obstacles,
            rng=rng,
        )
    else:
        obstacles = []
    plot_grid(spec, obstacles=tuple(obstacles))


if __name__ == "__main__":
    main()
