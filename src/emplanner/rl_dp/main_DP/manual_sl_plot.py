#!/usr/bin/env python3
"""Plot a manually specified s-l path together with hand-crafted obstacles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

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
    build_grid,
)
from sl_obstacles import Obstacle

DEFAULT_CONFIG = {
    "grid": {
        "s_range": list(DEFAULT_S_RANGE),
        "l_range": list(DEFAULT_L_RANGE),
        "s_samples": DEFAULT_S_SAMPLES,
        "l_samples": DEFAULT_L_SAMPLES,
    },
    "obstacles": [
        {"center": [3.0, -1.0], "length": 1.2, "width": 1.4, "yaw": 0.1},
        {"center": [0.5, 1.5], "length": 0.8, "width": 1.2, "yaw": -0.3},
        {"center": [4.0, 1.5], "length": 0.95, "width": 0.83, "yaw": -0.8},
        {"center": [7.3, 1.0], "length": 1.36, "width": 1.22, "yaw": -0.53},
    ],
    "path": [
        [0.0, -0.4],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.45],
        [4.0, -0.0],
        [5.0, 0.0],
        [6.0, 0.0],
        [7.0, -0.4],
        [8.0, -0.4],
    ],
    "title": "Manual Path with Obstacles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot manually defined obstacles and path on the s-l grid."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file describing grid/obstacles/path."
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_CONFIG
    data = json.loads(path.read_text())
    return data


def make_obstacles(raw: Sequence[dict]) -> List[Obstacle]:
    obstacles: List[Obstacle] = []
    for entry in raw:
        center = tuple(entry["center"])
        length = float(entry["length"])
        width = float(entry["width"])
        yaw = float(entry.get("yaw", 0.0))
        obstacles.append(Obstacle(center=center, length=length, width=width, yaw=yaw))
    return obstacles


def plot_manual(
    spec: GridSpec,
    obstacles: Iterable[Obstacle],
    path_points: np.ndarray,
    title: str,
) -> None:
    env = SLPathEnv(spec, max_obstacles=0)
    obstacles = list(obstacles)
    occupancy = env._build_occupancy(obstacles)  # type: ignore[attr-defined]

    s_grid, l_grid = build_grid(spec)
    occupancy_mask = occupancy.astype(bool)
    free_mask = ~occupancy_mask

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # grid lines
    for s_val in s_grid[:, 0]:
        ax.axvline(s_val, color="#d0d0d0", linewidth=0.6, zorder=0)
    for l_val in l_grid[0, :]:
        ax.axhline(l_val, color="#d0d0d0", linewidth=0.6, zorder=0)

    ax.scatter(
        s_grid[free_mask],
        l_grid[free_mask],
        c="#1f77b4",
        marker="s",
        s=60,
        edgecolors="none",
        alpha=0.7,
        label="Free cell",
    )
    ax.scatter(
        s_grid[occupancy_mask],
        l_grid[occupancy_mask],
        c="#d62728",
        marker="s",
        s=80,
        edgecolors="k",
        linewidths=0.5,
        label="Occupied cell",
    )

    for obs in obstacles:
        patch = Polygon(
            obs.corners(),
            closed=True,
            facecolor="#ff9896",
            edgecolor="#c12f23",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.add_patch(patch)

    ax.plot(
        path_points[:, 0],
        path_points[:, 1],
        color="tab:red",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Manual path",
    )
    ax.scatter(
        path_points[0, 0],
        path_points[0, 1],
        color="tab:green",
        s=90,
        edgecolors="black",
        linewidths=1.0,
        label="Start",
        zorder=5,
    )
    ax.scatter(
        path_points[-1, 0],
        path_points[-1, 1],
        color="tab:purple",
        s=90,
        edgecolors="black",
        linewidths=1.0,
        label="End",
        zorder=5,
    )

    ax.set_xlabel("s (longitudinal)")
    ax.set_ylabel("l (lateral)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    grid_cfg = config.get("grid", {})
    spec = GridSpec(
        s_range=tuple(grid_cfg.get("s_range", DEFAULT_CONFIG["grid"]["s_range"])),
        l_range=tuple(grid_cfg.get("l_range", DEFAULT_CONFIG["grid"]["l_range"])),
        s_samples=int(grid_cfg.get("s_samples", DEFAULT_CONFIG["grid"]["s_samples"])),
        l_samples=int(grid_cfg.get("l_samples", DEFAULT_CONFIG["grid"]["l_samples"])),
    )

    obstacles = make_obstacles(config.get("obstacles", []))
    path_points = np.asarray(config.get("path", []), dtype=float)
    if path_points.ndim != 2 or path_points.shape[1] != 2:
        raise ValueError("path must be a list of [s, l] pairs")

    title = str(config.get("title", DEFAULT_CONFIG["title"]))
    plot_manual(spec, obstacles, path_points, title)


if __name__ == "__main__":
    main()
