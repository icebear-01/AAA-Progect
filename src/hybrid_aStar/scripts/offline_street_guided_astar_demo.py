#!/usr/bin/env python3
"""Offline street-map demo: model-guided A* frontend + C++ backend smoothing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib import patheffects as pe


SCRIPT_PATH = Path(__file__).resolve()
HYBRID_ASTAR_ROOT = SCRIPT_PATH.parent.parent
REPO_SRC_ROOT = HYBRID_ASTAR_ROOT.parent.parent.parent
PLOT_DPI = 600
NEURAL_ASTAR_SRC = HYBRID_ASTAR_ROOT / "model_base_astar" / "neural-astar" / "src"
PathRecord = Tuple[float, float, Optional[int]]
ShiftPoint = Tuple[int, float, float, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline street demo for transformer-guided A* frontend and HybridAStar smoother"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=HYBRID_ASTAR_ROOT
        / "model_base_astar"
        / "neural-astar"
        / "planning-datasets"
        / "data"
        / "street"
        / "mixed_064_moore_c16.npz",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=HYBRID_ASTAR_ROOT
        / "model_base_astar"
        / "neural-astar"
        / "outputs"
        / "model_guidance_street"
        / "best.pt",
    )
    parser.add_argument(
        "--smoother-cli",
        type=Path,
        default=REPO_SRC_ROOT.parent / "devel" / "lib" / "hybrid_a_star" / "smooth_path_cli",
    )
    parser.add_argument(
        "--neural-astar-src",
        type=Path,
        default=NEURAL_ASTAR_SRC,
    )
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--map-index", type=int, default=-1, help="negative means random index")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-x", type=int, default=None)
    parser.add_argument("--start-y", type=int, default=None)
    parser.add_argument("--goal-x", type=int, default=None)
    parser.add_argument("--goal-y", type=int, default=None)
    parser.add_argument("--replay-meta", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--lambda-guidance", type=float, default=1.0)
    parser.add_argument("--heuristic-mode", type=str, default="octile")
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument("--guidance-integration-mode", type=str, default="g_cost")
    parser.add_argument("--guidance-bonus-threshold", type=float, default=0.5)
    parser.add_argument("--seed-xy-box-half-extent", type=float, default=None)
    parser.add_argument("--skip-seed-collision-check", action="store_true")
    parser.add_argument("--allow-corner-cut", action="store_true")
    parser.add_argument("--invert-guidance-cost", action="store_true")
    parser.add_argument("--min-start-goal-dist", type=float, default=22.0, help="grid-cell distance")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HYBRID_ASTAR_ROOT / "offline_results" / "street_guided_demo",
    )
    parser.add_argument("--case-name", type=str, default="demo")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=None)
    return parser.parse_args()


def _load_python_frontend(neural_astar_src: Path):
    neural_astar_src = neural_astar_src.resolve()
    if not neural_astar_src.exists():
        raise FileNotFoundError(f"neural-astar src not found: {neural_astar_src}")
    if str(neural_astar_src) not in sys.path:
        sys.path.insert(0, str(neural_astar_src))
    from hybrid_astar_guided.grid_astar import astar_8conn  # type: ignore
    from neural_astar.api.guidance_infer import infer_cost_map  # type: ignore

    return astar_8conn, infer_cost_map


def _split_key(split: str) -> str:
    return {"train": "arr_0", "valid": "arr_4", "test": "arr_8"}[split]


def load_street_occ(dataset: Path, split: str, map_index: int, rng: random.Random) -> Tuple[np.ndarray, int]:
    with np.load(dataset) as data:
        key = _split_key(split)
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"{key} must be [N,H,W], got {arr.shape}")
        if map_index < 0:
            map_index = rng.randrange(arr.shape[0])
        if not (0 <= map_index < arr.shape[0]):
            raise ValueError(f"map_index {map_index} out of range [0, {arr.shape[0]})")
        # street dataset uses 1=free, 0=obstacle
        occ = 1.0 - arr[map_index]
        return occ.astype(np.float32), map_index


def random_free_xy(occ: np.ndarray, rng: random.Random) -> Tuple[int, int]:
    h, w = occ.shape
    for _ in range(10000):
        x = rng.randrange(w)
        y = rng.randrange(h)
        if occ[y, x] < 0.5:
            return x, y
    raise RuntimeError("failed to sample a free cell")


def sample_problem(
    occ: np.ndarray,
    rng: random.Random,
    min_start_goal_dist: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    collision_grid_resolution: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    for _ in range(5000):
        start_xy = random_free_xy(occ, rng)
        goal_xy = random_free_xy(occ, rng)
        if start_xy == goal_xy:
            continue
        start_world = grid_to_world(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
        goal_world = grid_to_world(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)
        if not footprint16_collision_free_world(
            occ, start_world[0], start_world[1], origin_x, origin_y, resolution, collision_grid_resolution
        ):
            continue
        if not footprint16_collision_free_world(
            occ, goal_world[0], goal_world[1], origin_x, origin_y, resolution, collision_grid_resolution
        ):
            continue
        if math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]) < min_start_goal_dist:
            continue
        if astar_8conn(occ, start_xy, goal_xy) is None:
            continue
        return start_xy, goal_xy
    raise RuntimeError("failed to sample a solvable start/goal pair")


def resolve_problem(
    occ: np.ndarray,
    astar_solver,
    rng: random.Random,
    min_start_goal_dist: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    collision_grid_resolution: float,
    start_xy: Optional[Tuple[int, int]],
    goal_xy: Optional[Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if (start_xy is None) != (goal_xy is None):
        raise ValueError("start_xy and goal_xy must be provided together")
    if start_xy is None or goal_xy is None:
        return sample_problem(
            occ,
            rng,
            min_start_goal_dist,
            origin_x,
            origin_y,
            resolution,
            collision_grid_resolution,
        )

    for name, xy in (("start", start_xy), ("goal", goal_xy)):
        x, y = xy
        if not (0 <= x < occ.shape[1] and 0 <= y < occ.shape[0]):
            raise ValueError(f"{name}_xy {xy} out of bounds for map {occ.shape[1]}x{occ.shape[0]}")
        if occ[y, x] >= 0.5:
            raise ValueError(f"{name}_xy {xy} is occupied")

    if start_xy == goal_xy:
        raise ValueError("start_xy and goal_xy must be different")
    if math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]) < min_start_goal_dist:
        raise ValueError("fixed start/goal distance is below min_start_goal_dist")

    start_world = grid_to_world(start_xy[0], start_xy[1], origin_x, origin_y, resolution)
    goal_world = grid_to_world(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution)
    if not footprint16_collision_free_world(
        occ, start_world[0], start_world[1], origin_x, origin_y, resolution, collision_grid_resolution
    ):
        raise ValueError(f"start_xy {start_xy} fails footprint collision check")
    if not footprint16_collision_free_world(
        occ, goal_world[0], goal_world[1], origin_x, origin_y, resolution, collision_grid_resolution
    ):
        raise ValueError(f"goal_xy {goal_xy} fails footprint collision check")
    if astar_solver(occ, start_xy, goal_xy) is None:
        raise ValueError(f"fixed start/goal has no feasible 8-connected path: {start_xy} -> {goal_xy}")
    return start_xy, goal_xy


def grid_to_world(
    gx: int,
    gy: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    return origin_x + (gx + 0.5) * resolution, origin_y + (gy + 0.5) * resolution


def world_to_grid(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    return (x - origin_x) / resolution - 0.5, (y - origin_y) / resolution - 0.5


def point_collision_free_world(
    occ: np.ndarray,
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> bool:
    gx_f, gy_f = world_to_grid(x, y, origin_x, origin_y, resolution)
    gx = int(round(gx_f))
    gy = int(round(gy_f))
    if gy < 0 or gy >= occ.shape[0] or gx < 0 or gx >= occ.shape[1]:
        return False
    return bool(occ[gy, gx] < 0.5)


def footprint16_collision_free_world(
    occ: np.ndarray,
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    collision_grid_resolution: float,
) -> bool:
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for dx in offsets:
        for dy in offsets:
            if not point_collision_free_world(
                occ,
                x + dx * collision_grid_resolution,
                y + dy * collision_grid_resolution,
                origin_x,
                origin_y,
                resolution,
            ):
                return False
    return True


def segment_collision_free_world(
    occ: np.ndarray,
    a: Tuple[float, float],
    b: Tuple[float, float],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> bool:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    distance = math.hypot(dx, dy)
    steps = max(2, int(math.ceil(distance / max(0.06, resolution * 0.25))))
    for i in range(steps + 1):
        ratio = i / steps
        x = a[0] + dx * ratio
        y = a[1] + dy * ratio
        if not point_collision_free_world(occ, x, y, origin_x, origin_y, resolution):
            return False
    return True


def simplify_collinear(path: Sequence[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    if len(path) < 3:
        return list(path)
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        ax, ay = simplified[-1]
        bx, by = path[i]
        cx, cy = path[i + 1]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) > eps:
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified


def shortcut_path(
    occ: np.ndarray,
    path: Sequence[Tuple[float, float]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> List[Tuple[float, float]]:
    if len(path) < 3:
        return list(path)
    anchors = [path[0]]
    index = 0
    while index < len(path) - 1:
        best = index + 1
        for candidate in range(len(path) - 1, index + 1, -1):
            if segment_collision_free_world(occ, path[index], path[candidate], origin_x, origin_y, resolution):
                best = candidate
                break
        anchors.append(path[best])
        index = best
    return anchors


def chaikin_once(path: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    for i in range(len(path) - 1):
        p0 = np.asarray(path[i], dtype=np.float64)
        p1 = np.asarray(path[i + 1], dtype=np.float64)
        q = 0.75 * p0 + 0.25 * p1
        r = 0.25 * p0 + 0.75 * p1
        out.append((float(q[0]), float(q[1])))
        out.append((float(r[0]), float(r[1])))
    out.append(path[-1])
    return out


def path_collision_free(
    occ: np.ndarray,
    path: Sequence[Tuple[float, float]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> bool:
    return all(
        segment_collision_free_world(occ, path[i], path[i + 1], origin_x, origin_y, resolution)
        for i in range(len(path) - 1)
    )


def resample_path(path: Sequence[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
    if len(path) < 2:
        return list(path)
    resampled = [path[0]]
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        dx = bx - ax
        dy = by - ay
        dist = math.hypot(dx, dy)
        segments = max(1, int(math.ceil(dist / step)))
        for seg in range(1, segments + 1):
            ratio = seg / segments
            resampled.append((ax + dx * ratio, ay + dy * ratio))
    return resampled


def cumulative_arc_length(path: Sequence[Tuple[float, float]]) -> np.ndarray:
    if len(path) == 0:
        return np.zeros((0,), dtype=np.float64)
    s = np.zeros((len(path),), dtype=np.float64)
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        s[i] = s[i - 1] + math.hypot(dx, dy)
    return s


def curvature_profile(path: Sequence[Tuple[float, float]], step: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    if len(path) < 3:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    resampled = np.asarray(resample_path(path, step), dtype=np.float64)
    if resampled.shape[0] < 3:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    s = cumulative_arc_length([tuple(p) for p in resampled])
    x = resampled[:, 0]
    y = resampled[:, 1]
    dx = np.gradient(x, s, edge_order=2)
    dy = np.gradient(y, s, edge_order=2)
    ddx = np.gradient(dx, s, edge_order=2)
    ddy = np.gradient(dy, s, edge_order=2)
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
    kappa = (dx * ddy - dy * ddx) / denom
    return s, kappa


def backend_code_curvature_profile(path: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if len(path) < 3:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    points = np.asarray(path, dtype=np.float64)
    s = cumulative_arc_length(path)
    total_length = float(s[-1])
    average_interval_length = total_length / max(1, len(path) - 1)
    if average_interval_length <= 1e-9:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    second_diff_x = points[:-2, 0] - 2.0 * points[1:-1, 0] + points[2:, 0]
    second_diff_y = points[:-2, 1] - 2.0 * points[1:-1, 1] + points[2:, 1]
    curvature_mag = np.sqrt(second_diff_x * second_diff_x + second_diff_y * second_diff_y) / (
        average_interval_length * average_interval_length
    )
    return s[1:-1], curvature_mag


def split_point_arc_lengths(
    seed_world_path: Sequence[Tuple[float, float]],
    split_world_points: Sequence[Tuple[int, float, float]],
) -> List[Tuple[int, float]]:
    if not split_world_points:
        return []
    seed_s = cumulative_arc_length(seed_world_path)
    seed_xy = np.asarray(seed_world_path, dtype=np.float64)
    positions: List[Tuple[int, float]] = []
    for order, (split_index, x, y) in enumerate(split_world_points, start=1):
        deltas = seed_xy - np.asarray([x, y], dtype=np.float64)
        nearest_index = int(np.argmin(np.sum(deltas * deltas, axis=1)))
        if 0 <= split_index < len(seed_s):
            direct_match_error = float(np.linalg.norm(seed_xy[split_index] - np.asarray([x, y], dtype=np.float64)))
            if direct_match_error <= 1e-4:
                nearest_index = split_index
        positions.append((order, float(seed_s[nearest_index])))
    return positions


def annotate_split_points_on_curvature(
    ax: plt.Axes,
    seed_world_path: Sequence[Tuple[float, float]],
    split_world_points: Optional[Sequence[Tuple[int, float, float]]],
) -> None:
    if not split_world_points:
        return
    for order, split_s in split_point_arc_lengths(seed_world_path, split_world_points):
        ax.axvline(
            split_s,
            color="#1f4e99",
            linewidth=0.9,
            linestyle=":",
            alpha=0.72,
            zorder=1,
        )
        ax.annotate(
            "P{}".format(order),
            xy=(split_s, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(3, -1),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.3,
            color="#1f4e99",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "#f4f8ff", "edgecolor": "#c4d2ea", "alpha": 0.92},
            zorder=5,
        )


def preprocess_seed_path(
    occ: np.ndarray,
    raw_world_path: Sequence[Tuple[float, float]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> List[Tuple[float, float]]:
    seed = simplify_collinear(raw_world_path)
    seed = shortcut_path(occ, seed, origin_x, origin_y, resolution)
    for _ in range(2):
        candidate = chaikin_once(seed)
        if path_collision_free(occ, candidate, origin_x, origin_y, resolution):
            seed = candidate
        else:
            break
    seed = resample_path(seed, max(0.12, resolution * 0.6))
    seed[0] = raw_world_path[0]
    seed[-1] = raw_world_path[-1]
    return seed


def write_raw_path_csv(path: Path, world_path: Sequence[Tuple[float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for x, y in world_path:
            writer.writerow([f"{x:.6f}", f"{y:.6f}"])


def read_path_records_csv(path: Path) -> List[PathRecord]:
    records: List[PathRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            direction: Optional[int] = None
            if "dir" in row and row["dir"] not in (None, ""):
                direction = int(float(row["dir"]))
            records.append((float(row["x"]), float(row["y"]), direction))
    if len(records) < 2:
        raise RuntimeError(f"path csv is empty: {path}")
    return records


def read_xy_path_csv(path: Path) -> List[Tuple[float, float]]:
    points = [(x, y) for x, y, _ in read_path_records_csv(path)]
    return points


def read_smoothed_path_csv(path: Path) -> List[Tuple[float, float]]:
    return read_xy_path_csv(path)


def extract_gear_shift_points(path_records: Sequence[PathRecord]) -> List[ShiftPoint]:
    shift_points: List[ShiftPoint] = []
    for index in range(1, len(path_records)):
        _, _, prev_dir = path_records[index - 1]
        x, y, curr_dir = path_records[index]
        if prev_dir is None or curr_dir is None or prev_dir == curr_dir:
            continue
        shift_points.append((index, x, y, prev_dir, curr_dir))
    return shift_points


def format_direction_label(direction: int) -> str:
    if direction > 0:
        return "F"
    if direction < 0:
        return "R"
    return str(direction)


def annotate_gear_shifts(
    ax: plt.Axes,
    shift_world_points: Optional[Sequence[ShiftPoint]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    shift_grid = []
    if shift_world_points:
        shift_grid = [
            (point_index, *world_to_grid(x, y, origin_x, origin_y, resolution), prev_dir, curr_dir)
            for point_index, x, y, prev_dir, curr_dir in shift_world_points
        ]

    if not shift_grid:
        ax.text(
            0.02,
            0.06,
            "No gear shifts",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#7d4d4d",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fff5f1", "edgecolor": "#d4b7af", "alpha": 0.9},
            zorder=10,
        )
        return

    for order, (point_index, xg, yg, prev_dir, curr_dir) in enumerate(shift_grid, start=1):
        ax.scatter(
            [xg],
            [yg],
            s=34,
            marker="s",
            c="#b34747",
            edgecolors="#fffef8",
            linewidths=0.8,
            zorder=10,
            label="Gear Shift" if order == 1 else None,
        )
        ax.text(
            xg + 0.42,
            yg + 0.35,
            f"S{order} {format_direction_label(prev_dir)}->{format_direction_label(curr_dir)}",
            fontsize=7.0,
            color="#8c2f2f",
            ha="left",
            va="bottom",
            zorder=11,
            path_effects=[pe.Stroke(linewidth=1.2, foreground="#fffef8"), pe.Normal()],
        )


def read_split_points_csv(path: Path) -> List[Tuple[int, float, float]]:
    points: List[Tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((int(row["index"]), float(row["x"]), float(row["y"])))
    return points


def write_smoother_yaml(
    path: Path,
    occ: np.ndarray,
    raw_world_path: Sequence[Tuple[float, float]],
    origin_x: float,
    origin_y: float,
    resolution: float,
    seed_xy_box_half_extent: float,
    skip_seed_collision_check: bool,
) -> None:
    data = {
        "map": {
            "width": int(occ.shape[1]),
            "height": int(occ.shape[0]),
            "resolution": float(resolution),
            "collision_grid_resolution": 0.125,
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "state_grid_resolution": 1.0,
            "steering_angle": 10.0,
            "steering_angle_discrete_num": 1,
            "wheel_base": 0.8,
            "segment_length": 1.6,
            "segment_length_discrete_num": 8,
            "steering_penalty": 1.05,
            "reversing_penalty": 2.0,
            "steering_change_penalty": 1.5,
            "shot_distance": 5.0,
            "seed_resample_step": 0.10,
            "seed_xy_box_half_extent": float(seed_xy_box_half_extent),
            "skip_seed_collision_check": bool(skip_seed_collision_check),
            "simplified_collision_check": True,
            "fix_endpoint_heading": False,
            "occupancy": occ.astype(int).tolist(),
        },
        "raw_path": [[float(x), float(y)] for x, y in raw_world_path],
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.6f")


def load_matrix_csv(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    return matrix


def plot_case(
    occ: np.ndarray,
    cost_map: np.ndarray,
    raw_world_path: Sequence[Tuple[float, float]],
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    origin_x: float,
    origin_y: float,
    resolution: float,
    out_path: Path,
    split_world_points: Optional[Sequence[Tuple[int, float, float]]] = None,
    shift_world_points: Optional[Sequence[ShiftPoint]] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6), facecolor="#fbfbf8")
    map_cmap = ListedColormap(["#f2f1ea", "#7b7b7f"])
    overlay_rgba = np.zeros((*cost_map.shape, 4), dtype=np.float32)
    cost_min = float(np.min(cost_map))
    cost_max = float(np.max(cost_map))
    cost_norm = (cost_map - cost_min) / max(1e-6, cost_max - cost_min)
    guidance_score = np.clip(1.0 - cost_norm, 0.0, 1.0)
    guidance_alpha = np.clip((guidance_score - 0.22) / 0.78, 0.0, 1.0) ** 1.2
    overlay_rgba[..., :3] = np.asarray(to_rgba("#d9e5c6"))[:3]
    overlay_rgba[..., 3] = guidance_alpha * 0.34

    route_outline = [pe.Stroke(linewidth=1.2, foreground="#fffef8"), pe.Normal()]
    smooth_outline = [pe.Stroke(linewidth=1.35, foreground="#fffef8"), pe.Normal()]

    for ax in axes:
        ax.imshow(occ, cmap=map_cmap, origin="upper", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f2f1ea")
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#9a988f")

    axes[0].imshow(overlay_rgba, origin="upper", interpolation="nearest")
    raw_grid = np.asarray(
        [world_to_grid(x, y, origin_x, origin_y, resolution) for x, y in raw_world_path], dtype=np.float32
    )
    seed_grid = np.asarray(
        [world_to_grid(x, y, origin_x, origin_y, resolution) for x, y in seed_world_path], dtype=np.float32
    )
    smooth_grid = np.asarray(
        [world_to_grid(x, y, origin_x, origin_y, resolution) for x, y in smoothed_world_path], dtype=np.float32
    )
    seed_overlaps_smooth = (
        seed_grid.shape == smooth_grid.shape
        and seed_grid.size > 0
        and float(np.max(np.linalg.norm(seed_grid - smooth_grid, axis=1))) < 1e-3
    )
    split_grid = []
    if split_world_points:
        split_grid = [
            (idx, *world_to_grid(x, y, origin_x, origin_y, resolution)) for idx, x, y in split_world_points
        ]

    sx, sy = start_xy
    gx, gy = goal_xy
    for ax in axes:
        ax.scatter([sx], [sy], c="#e63b2e", s=60, marker="x", linewidths=1.7, label="Start", zorder=7)
        ax.scatter([gx], [gy], c="#31c93c", s=42, marker="o", label="Goal", zorder=7)

    axes[0].plot(
        raw_grid[:, 0],
        raw_grid[:, 1],
        color="#7ecf48",
        linewidth=1.1,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Model Route",
        zorder=5,
        path_effects=route_outline,
    )
    axes[0].text(
        0.02,
        0.98,
        "(a) Transformer-Guided A*",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#222222",
    )
    axes[0].legend(
        loc="upper center",
        fontsize=9,
        framealpha=0.94,
        ncol=3,
        facecolor="#fffef9",
        edgecolor="#c5c2b8",
    )

    axes[1].plot(
        raw_grid[:, 0],
        raw_grid[:, 1],
        color="#b8b3a1",
        linewidth=1.0,
        alpha=0.95,
        linestyle="--",
        label="Model Route",
        zorder=3,
    )
    axes[1].plot(
        smooth_grid[:, 0],
        smooth_grid[:, 1],
        color="#78bf54",
        linewidth=1.1,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Smoothed Route",
        zorder=5,
        path_effects=smooth_outline,
    )
    if seed_overlaps_smooth:
        marker_step = max(1, len(seed_grid) // 18)
        axes[1].plot(
            seed_grid[:, 0],
            seed_grid[:, 1],
            color="#d89133",
            linewidth=0.0,
            linestyle="None",
            marker="o",
            markersize=3.3,
            markevery=marker_step,
            markeredgewidth=0.45,
            markeredgecolor="#fffef8",
            alpha=0.96,
            label="XY Seed Route",
            zorder=6,
        )
    else:
        axes[1].plot(
            seed_grid[:, 0],
            seed_grid[:, 1],
            color="#d89133",
            linewidth=1.1,
            alpha=0.98,
            linestyle="-.",
            label="XY Seed Route",
            zorder=4,
        )
    axes[1].text(
        0.02,
        0.98,
        "(b) Backend Smoother",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#222222",
    )
    if split_grid:
        for idx, xg, yg in split_grid:
            axes[1].scatter(
                [xg],
                [yg],
                s=20,
                c="#1f4e99",
                edgecolors="#fffef8",
                linewidths=0.6,
                zorder=8,
            )
            axes[1].text(
                xg + 0.35,
                yg - 0.35,
                str(idx + 1),
                fontsize=7,
                color="#1f4e99",
                ha="left",
                va="center",
                zorder=9,
                path_effects=[pe.Stroke(linewidth=1.2, foreground="#fffef8"), pe.Normal()],
            )
    annotate_gear_shifts(axes[1], shift_world_points, origin_x, origin_y, resolution)
    axes[1].legend(
        loc="upper center",
        fontsize=9,
        framealpha=0.94,
        ncol=5,
        facecolor="#fffef9",
        edgecolor="#c5c2b8",
    )

    fig.tight_layout(pad=1.0, w_pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_path.with_name("offline_demo_paper.png"), dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_path.with_name("offline_demo_paper.pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_split_points_debug(
    occ: np.ndarray,
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    split_world_points: Sequence[Tuple[int, float, float]],
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    origin_x: float,
    origin_y: float,
    resolution: float,
    out_path: Path,
    shift_world_points: Optional[Sequence[ShiftPoint]] = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.3), facecolor="#fbfbf8")
    map_cmap = ListedColormap(["#f2f1ea", "#7b7b7f"])
    ax.imshow(occ, cmap=map_cmap, origin="upper", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f2f1ea")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#9a988f")

    seed_grid = np.asarray(
        [world_to_grid(x, y, origin_x, origin_y, resolution) for x, y in seed_world_path], dtype=np.float32
    )
    smooth_grid = np.asarray(
        [world_to_grid(x, y, origin_x, origin_y, resolution) for x, y in smoothed_world_path], dtype=np.float32
    )
    split_grid = [
        (idx, *world_to_grid(x, y, origin_x, origin_y, resolution)) for idx, x, y in split_world_points
    ]

    sx, sy = start_xy
    gx, gy = goal_xy
    ax.scatter([sx], [sy], c="#e63b2e", s=60, marker="x", linewidths=1.7, label="Start", zorder=7)
    ax.scatter([gx], [gy], c="#31c93c", s=42, marker="o", label="Goal", zorder=7)
    ax.plot(seed_grid[:, 0], seed_grid[:, 1], color="#d9a04c", linewidth=1.0, linestyle="-.", alpha=0.9, label="Seed Route", zorder=4)
    ax.plot(smooth_grid[:, 0], smooth_grid[:, 1], color="#78bf54", linewidth=1.05, label="Final Route", zorder=5)
    for idx, xg, yg in split_grid:
        ax.scatter([xg], [yg], s=24, c="#1f4e99", edgecolors="#fffef8", linewidths=0.8, zorder=8)
        ax.text(
            xg + 0.4,
            yg - 0.4,
            str(idx + 1),
            fontsize=7.5,
            color="#1f4e99",
            ha="left",
            va="center",
            zorder=9,
            path_effects=[pe.Stroke(linewidth=1.2, foreground="#fffef8"), pe.Normal()],
        )
    annotate_gear_shifts(ax, shift_world_points, origin_x, origin_y, resolution)
    ax.legend(loc="upper center", fontsize=8.5, framealpha=0.94, ncol=4, facecolor="#fffef9", edgecolor="#c5c2b8")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_curvature_compare(
    raw_world_path: Sequence[Tuple[float, float]],
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    out_path: Path,
    include_raw_path: bool = True,
    split_world_points: Optional[Sequence[Tuple[int, float, float]]] = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.3), facecolor="#fbfbf8")
    ax.set_facecolor("#fbfbf8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#9a988f")
    ax.grid(True, color="#d8d5cc", linewidth=0.6, alpha=0.75)
    ax.axhline(
        1.0,
        color="#b34747",
        linewidth=0.95,
        linestyle=":",
        alpha=0.9,
        zorder=1,
        label="kappa = 1.0",
    )
    ax.axhline(
        -1.0,
        color="#b34747",
        linewidth=0.95,
        linestyle=":",
        alpha=0.9,
        zorder=1,
        label="kappa = -1.0",
    )

    curve_specs = [
        ("Seed Route", seed_world_path, "#d89133", "-.", 1.25),
        ("Smoothed Route", smoothed_world_path, "#78bf54", "-", 1.35),
    ]
    if include_raw_path:
        curve_specs.insert(0, ("Model Route", raw_world_path, "#b8b3a1", "--", 1.2))
    for label, path, color, linestyle, linewidth in curve_specs:
        s, kappa = curvature_profile(path)
        if s.size == 0:
            continue
        ax.plot(
            s,
            kappa,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    annotate_split_points_on_curvature(ax, seed_world_path, split_world_points)

    ax.set_xlabel("s [m]")
    ax.set_ylabel("kappa [1/m]")
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.94,
        facecolor="#fffef9",
        edgecolor="#c5c2b8",
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_backend_code_curvature_compare(
    raw_world_path: Sequence[Tuple[float, float]],
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    out_path: Path,
    include_raw_path: bool = True,
    split_world_points: Optional[Sequence[Tuple[int, float, float]]] = None,
    curvature_limit: float = 1.0,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.3), facecolor="#fbfbf8")
    ax.set_facecolor("#fbfbf8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#9a988f")
    ax.grid(True, color="#d8d5cc", linewidth=0.6, alpha=0.75)
    ax.axhline(
        curvature_limit,
        color="#b34747",
        linewidth=0.95,
        linestyle=":",
        alpha=0.9,
        zorder=1,
        label="backend kappa limit = {:.1f}".format(curvature_limit),
    )

    curve_specs = [
        ("Seed Route", seed_world_path, "#d89133", "-.", 1.25),
        ("Smoothed Route", smoothed_world_path, "#78bf54", "-", 1.35),
    ]
    if include_raw_path:
        curve_specs.insert(0, ("Model Route", raw_world_path, "#b8b3a1", "--", 1.2))
    for label, path, color, linestyle, linewidth in curve_specs:
        s, kappa = backend_code_curvature_profile(path)
        if s.size == 0:
            continue
        ax.plot(
            s,
            kappa,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    annotate_split_points_on_curvature(ax, seed_world_path, split_world_points)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("backend approx |kappa| [1/m]")
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.94,
        facecolor="#fffef9",
        edgecolor="#c5c2b8",
    )
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.replay_meta is not None:
        replay_meta = json.loads(args.replay_meta.read_text(encoding="utf-8"))
        args.dataset = Path(replay_meta["dataset"])
        args.split = replay_meta["split"]
        args.map_index = int(replay_meta["map_index"])
        args.seed = int(replay_meta["seed"])
        args.start_x, args.start_y = map(int, replay_meta["start_xy"])
        args.goal_x, args.goal_y = map(int, replay_meta["goal_xy"])
        args.lambda_guidance = float(replay_meta["lambda_guidance"])
        args.heuristic_mode = replay_meta["heuristic_mode"]
        args.heuristic_weight = float(replay_meta["heuristic_weight"])
        args.guidance_integration_mode = replay_meta["guidance_integration_mode"]
        args.guidance_bonus_threshold = float(replay_meta["guidance_bonus_threshold"])
        if args.seed_xy_box_half_extent is None:
            args.seed_xy_box_half_extent = float(replay_meta.get("seed_xy_box_half_extent", 0.10))
        if not args.skip_seed_collision_check:
            args.skip_seed_collision_check = bool(replay_meta.get("skip_seed_collision_check", False))
        args.resolution = float(replay_meta["resolution"])
        args.origin_x = float(replay_meta["origin_x"])
        args.origin_y = float(replay_meta["origin_y"])
    if args.seed_xy_box_half_extent is None:
        args.seed_xy_box_half_extent = 0.10
    if args.plot_only:
        if args.input_dir is None:
            raise RuntimeError("--plot-only requires --input-dir")
        input_dir = args.input_dir
        meta = json.loads((input_dir / "meta.json").read_text(encoding="utf-8"))
        occ = load_matrix_csv(input_dir / "occupancy.csv").astype(np.float32)
        cost_map = load_matrix_csv(input_dir / "guidance_cost.csv").astype(np.float32)
        raw_world_path = read_xy_path_csv(input_dir / "frontend_raw_path.csv")
        seed_records = read_path_records_csv(input_dir / "frontend_seed_path.csv")
        smoothed_records = read_path_records_csv(input_dir / "smoothed_path.csv")
        seed_world_path = [(x, y) for x, y, _ in seed_records]
        smoothed_world_path = [(x, y) for x, y, _ in smoothed_records]
        shift_world_points = extract_gear_shift_points(smoothed_records)
        split_points_csv = input_dir / "segment_split_points.csv"
        split_world_points = read_split_points_csv(split_points_csv) if split_points_csv.exists() else []
        start_xy = tuple(meta["start_xy"])
        goal_xy = tuple(meta["goal_xy"])
        origin_x = float(meta["origin_x"])
        origin_y = float(meta["origin_y"])
        resolution = float(meta["resolution"])
        fig_png = input_dir / "offline_demo.png"
        split_debug_png = input_dir / "offline_demo_split_points.png"
        curvature_png = input_dir / "curvature_compare.png"
        curvature_seed_smooth_png = input_dir / "curvature_compare_seed_smooth.png"
        curvature_backend_png = input_dir / "curvature_compare_backend_code.png"
        curvature_backend_seed_smooth_png = input_dir / "curvature_compare_backend_code_seed_smooth.png"
        plot_case(
            occ=occ,
            cost_map=cost_map,
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            start_xy=start_xy,
            goal_xy=goal_xy,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            out_path=fig_png,
            split_world_points=split_world_points,
            shift_world_points=shift_world_points,
        )
        plot_split_points_debug(
            occ=occ,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            split_world_points=split_world_points,
            start_xy=start_xy,
            goal_xy=goal_xy,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            out_path=split_debug_png,
            shift_world_points=shift_world_points,
        )
        plot_curvature_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=curvature_png,
            split_world_points=split_world_points,
        )
        plot_curvature_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=curvature_seed_smooth_png,
            include_raw_path=False,
            split_world_points=split_world_points,
        )
        plot_backend_code_curvature_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=curvature_backend_png,
            split_world_points=split_world_points,
        )
        plot_backend_code_curvature_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=curvature_backend_seed_smooth_png,
            include_raw_path=False,
            split_world_points=split_world_points,
        )
        print(f"saved_png={fig_png}")
        print(f"saved_split_debug_png={split_debug_png}")
        print(f"saved_curvature_png={curvature_png}")
        print(f"saved_curvature_seed_smooth_png={curvature_seed_smooth_png}")
        print(f"saved_curvature_backend_png={curvature_backend_png}")
        print(f"saved_curvature_backend_seed_smooth_png={curvature_backend_seed_smooth_png}")
        return 0

    rng = random.Random(args.seed)
    occ, map_index = load_street_occ(args.dataset, args.split, args.map_index, rng)
    collision_grid_resolution = 0.125
    astar_8conn, infer_cost_map = _load_python_frontend(args.neural_astar_src)
    start_xy_opt = None
    goal_xy_opt = None
    if None not in (args.start_x, args.start_y, args.goal_x, args.goal_y):
        start_xy_opt = (args.start_x, args.start_y)
        goal_xy_opt = (args.goal_x, args.goal_y)
    start_xy, goal_xy = resolve_problem(
        occ,
        astar_8conn,
        rng,
        args.min_start_goal_dist,
        args.origin_x,
        args.origin_y,
        args.resolution,
        collision_grid_resolution,
        start_xy_opt,
        goal_xy_opt,
    )
    cost_map = infer_cost_map(
        ckpt_path=args.ckpt,
        occ_map_numpy=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_yaw=0.0,
        goal_yaw=0.0,
        device=args.device,
        invert_guidance_cost=args.invert_guidance_cost,
    )

    path_xy = astar_8conn(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        guidance_cost=cost_map,
        lambda_guidance=args.lambda_guidance,
        allow_corner_cut=args.allow_corner_cut,
        heuristic_mode=args.heuristic_mode,
        heuristic_weight=args.heuristic_weight,
        guidance_integration_mode=args.guidance_integration_mode,
        guidance_bonus_threshold=args.guidance_bonus_threshold,
    )
    if path_xy is None or len(path_xy) < 2:
        raise RuntimeError("model-guided A* failed to produce a valid raw path")

    raw_world_path = [
        grid_to_world(gx, gy, args.origin_x, args.origin_y, args.resolution) for gx, gy in path_xy
    ]
    raw_world_path[0] = grid_to_world(start_xy[0], start_xy[1], args.origin_x, args.origin_y, args.resolution)
    raw_world_path[-1] = grid_to_world(goal_xy[0], goal_xy[1], args.origin_x, args.origin_y, args.resolution)

    out_dir = args.output_dir / args.case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "frontend_raw_path.csv"
    seed_csv = out_dir / "frontend_seed_path.csv"
    smooth_csv = out_dir / "smoothed_path.csv"
    split_points_csv = out_dir / "segment_split_points.csv"
    smooth_yaml = out_dir / "smoother_request.yaml"
    occupancy_csv = out_dir / "occupancy.csv"
    guidance_csv = out_dir / "guidance_cost.csv"
    fig_png = out_dir / "offline_demo.png"
    split_debug_png = out_dir / "offline_demo_split_points.png"
    curvature_png = out_dir / "curvature_compare.png"
    curvature_seed_smooth_png = out_dir / "curvature_compare_seed_smooth.png"
    curvature_backend_png = out_dir / "curvature_compare_backend_code.png"
    curvature_backend_seed_smooth_png = out_dir / "curvature_compare_backend_code_seed_smooth.png"
    meta_json = out_dir / "meta.json"

    write_raw_path_csv(raw_csv, raw_world_path)
    write_matrix_csv(occupancy_csv, occ.astype(np.float32))
    write_matrix_csv(guidance_csv, cost_map.astype(np.float32))
    write_smoother_yaml(
        smooth_yaml,
        occ,
        raw_world_path,
        args.origin_x,
        args.origin_y,
        args.resolution,
        args.seed_xy_box_half_extent,
        args.skip_seed_collision_check,
    )

    cmd = [
        str(args.smoother_cli),
        "--input-yaml",
        str(smooth_yaml),
        "--seed-csv",
        str(seed_csv),
        "--split-points-csv",
        str(split_points_csv),
        "--output-csv",
        str(smooth_csv),
    ]
    smoother_env = os.environ.copy()
    smoother_lib_dir = args.smoother_cli.resolve().parents[1]
    existing_ld = smoother_env.get("LD_LIBRARY_PATH", "")
    smoother_env["LD_LIBRARY_PATH"] = (
        f"{smoother_lib_dir}:{existing_ld}" if existing_ld else str(smoother_lib_dir)
    )
    subprocess.run(cmd, check=True, env=smoother_env)
    seed_records = read_path_records_csv(seed_csv)
    smoothed_records = read_path_records_csv(smooth_csv)
    seed_world_path = [(x, y) for x, y, _ in seed_records]
    smoothed_world_path = [(x, y) for x, y, _ in smoothed_records]
    shift_world_points = extract_gear_shift_points(smoothed_records)
    split_world_points = read_split_points_csv(split_points_csv)

    plot_case(
        occ=occ,
        cost_map=cost_map,
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        start_xy=start_xy,
        goal_xy=goal_xy,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        resolution=args.resolution,
        out_path=fig_png,
        split_world_points=split_world_points,
        shift_world_points=shift_world_points,
    )
    plot_split_points_debug(
        occ=occ,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        split_world_points=split_world_points,
        start_xy=start_xy,
        goal_xy=goal_xy,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        resolution=args.resolution,
        out_path=split_debug_png,
        shift_world_points=shift_world_points,
    )
    plot_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=curvature_png,
        split_world_points=split_world_points,
    )
    plot_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=curvature_seed_smooth_png,
        include_raw_path=False,
        split_world_points=split_world_points,
    )
    plot_backend_code_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=curvature_backend_png,
        split_world_points=split_world_points,
    )
    plot_backend_code_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=curvature_backend_seed_smooth_png,
        include_raw_path=False,
        split_world_points=split_world_points,
    )

    meta = {
        "dataset": str(args.dataset),
        "split": args.split,
        "map_index": map_index,
        "seed": args.seed,
        "start_xy": list(start_xy),
        "goal_xy": list(goal_xy),
        "lambda_guidance": args.lambda_guidance,
        "heuristic_mode": args.heuristic_mode,
        "heuristic_weight": args.heuristic_weight,
        "guidance_integration_mode": args.guidance_integration_mode,
        "guidance_bonus_threshold": args.guidance_bonus_threshold,
        "seed_xy_box_half_extent": args.seed_xy_box_half_extent,
        "skip_seed_collision_check": args.skip_seed_collision_check,
        "resolution": args.resolution,
        "collision_grid_resolution": collision_grid_resolution,
        "origin_x": args.origin_x,
        "origin_y": args.origin_y,
        "raw_path_points": len(raw_world_path),
        "seed_path_points": len(seed_world_path),
        "smoothed_path_points": len(smoothed_world_path),
        "segment_split_points": len(split_world_points),
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"saved_png={fig_png}")
    print(f"saved_raw_csv={raw_csv}")
    print(f"saved_seed_csv={seed_csv}")
    print(f"saved_split_points_csv={split_points_csv}")
    print(f"saved_smoothed_csv={smooth_csv}")
    print(f"saved_occupancy_csv={occupancy_csv}")
    print(f"saved_guidance_csv={guidance_csv}")
    print(f"saved_split_debug_png={split_debug_png}")
    print(f"saved_curvature_png={curvature_png}")
    print(f"saved_curvature_seed_smooth_png={curvature_seed_smooth_png}")
    print(f"saved_curvature_backend_png={curvature_backend_png}")
    print(f"saved_curvature_backend_seed_smooth_png={curvature_backend_seed_smooth_png}")
    print(f"saved_meta={meta_json}")
    print(f"map_index={map_index}")
    print(f"start_xy={start_xy}")
    print(f"goal_xy={goal_xy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
