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
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib import patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D


SCRIPT_PATH = Path(__file__).resolve()
HYBRID_ASTAR_ROOT = SCRIPT_PATH.parent.parent
REPO_SRC_ROOT = HYBRID_ASTAR_ROOT.parent.parent
PLOT_DPI = 600
NEURAL_ASTAR_SRC = HYBRID_ASTAR_ROOT / "model_base_astar" / "neural-astar" / "src"
CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")


def default_smoother_cli() -> Path:
    candidates = [
        REPO_SRC_ROOT / "build_hybrid_astar" / "devel" / "lib" / "hybrid_a_star" / "smooth_path_cli",
        REPO_SRC_ROOT / "devel" / "lib" / "hybrid_a_star" / "smooth_path_cli",
        REPO_SRC_ROOT.parent / "build_hybrid_astar" / "devel" / "lib" / "hybrid_a_star" / "smooth_path_cli",
        REPO_SRC_ROOT.parent / "devel" / "lib" / "hybrid_a_star" / "smooth_path_cli",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
        default=default_smoother_cli(),
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
    astar_solver,
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
        if astar_solver(occ, start_xy, goal_xy) is None:
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
    start_xy: Tuple[int, int] | None,
    goal_xy: Tuple[int, int] | None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if (start_xy is None) != (goal_xy is None):
        raise ValueError("start_xy and goal_xy must be provided together")
    if start_xy is None or goal_xy is None:
        return sample_problem(
            occ,
            astar_solver,
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
    edge_order = 2 if resampled.shape[0] >= 3 else 1
    dx = np.gradient(x, s, edge_order=edge_order)
    dy = np.gradient(y, s, edge_order=edge_order)
    ddx = np.gradient(dx, s, edge_order=edge_order)
    ddy = np.gradient(dy, s, edge_order=edge_order)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.where(denom < 1e-9, np.inf, denom)
    kappa = (dx * ddy - dy * ddx) / denom
    return s, kappa


def heading_profile(path: Sequence[Tuple[float, float]], step: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    if len(path) < 2:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    resampled = np.asarray(resample_path(path, step), dtype=np.float64)
    if resampled.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    s = cumulative_arc_length([tuple(p) for p in resampled])
    x = resampled[:, 0]
    y = resampled[:, 1]
    edge_order = 2 if resampled.shape[0] >= 3 else 1
    dx = np.gradient(x, s, edge_order=edge_order)
    dy = np.gradient(y, s, edge_order=edge_order)
    heading = np.unwrap(np.arctan2(dy, dx))
    return s, np.rad2deg(heading)


def project_split_points_to_arc_length(
    path: Sequence[Tuple[float, float]],
    split_world_points: Sequence[Tuple[int, float, float]],
) -> List[Tuple[int, float]]:
    if len(path) < 2 or not split_world_points:
        return []
    path_np = np.asarray(path, dtype=np.float64)
    s = cumulative_arc_length(path)
    split_s: List[Tuple[int, float]] = []
    for idx, x, y in split_world_points:
        query = np.asarray([x, y], dtype=np.float64)
        best_dist_sqr = float("inf")
        best_s = 0.0
        for seg_idx in range(len(path_np) - 1):
            a = path_np[seg_idx]
            b = path_np[seg_idx + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-12:
                t = 0.0
                proj = a
            else:
                t = float(np.clip(np.dot(query - a, ab) / denom, 0.0, 1.0))
                proj = a + t * ab
            dist_sqr = float(np.dot(query - proj, query - proj))
            if dist_sqr < best_dist_sqr:
                best_dist_sqr = dist_sqr
                best_s = float(s[seg_idx] + np.linalg.norm(proj - a))
        split_s.append((idx, best_s))
    return split_s


def interpolate_profile_to_reference(
    s_ref: np.ndarray,
    s_src: np.ndarray,
    value_src: np.ndarray,
) -> np.ndarray:
    if s_ref.size == 0 or s_src.size == 0 or value_src.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if s_src.size == 1:
        return np.full_like(s_ref, float(value_src[0]), dtype=np.float64)
    s_ref_clip = np.clip(s_ref, float(s_src[0]), float(s_src[-1]))
    return np.interp(s_ref_clip, s_src, value_src)


def wrap_deg_to_ref(angle_deg: np.ndarray, ref_deg: np.ndarray) -> np.ndarray:
    if angle_deg.size == 0:
        return angle_deg
    if ref_deg.size == 0:
        return ((angle_deg + 180.0) % 360.0) - 180.0
    wrapped = ((angle_deg - ref_deg + 180.0) % 360.0) - 180.0 + ref_deg
    return wrapped


def shift_curve_to_branch(angle_deg: np.ndarray, target_center_deg: float) -> np.ndarray:
    if angle_deg.size == 0:
        return angle_deg
    median_deg = float(np.median(angle_deg))
    k = round((target_center_deg - median_deg) / 360.0)
    return angle_deg + 360.0 * float(k)


def normalize_heading_branch(angle_deg: np.ndarray, target_center_deg: float) -> np.ndarray:
    if angle_deg.size == 0:
        return angle_deg
    normalized = np.asarray(angle_deg, dtype=np.float64).copy()
    normalized = ((normalized - target_center_deg + 180.0) % 360.0) - 180.0 + target_center_deg
    for i in range(1, normalized.size):
        delta = normalized[i] - normalized[i - 1]
        if delta > 180.0:
            normalized[i:] -= 360.0
        elif delta < -180.0:
            normalized[i:] += 360.0
    return shift_curve_to_branch(normalized, target_center_deg)


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


def read_xy_path_csv(path: Path) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            points.append((float(row["x"]), float(row["y"])))
    if len(points) < 2:
        raise RuntimeError(f"path csv is empty: {path}")
    return points


def read_smoothed_path_csv(path: Path) -> List[Tuple[float, float]]:
    return read_xy_path_csv(path)


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


def safe_save_pdf(fig: plt.Figure, out_path: Path) -> None:
    try:
        fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    except Exception as exc:
        print(f"warning: skip pdf export {out_path}: {exc}")


def add_scale_bar(
    ax: plt.Axes,
    length_data: float,
    label: str = "1 m",
    *,
    side: str = "right",
    font_prop: fm.FontProperties | None = None,
    color: str = "#2b2d31",
    pad_frac: float = 0.045,
    linewidth: float = 2.2,
) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_span = abs(x1 - x0)
    y_span = abs(y1 - y0)
    if x_span <= 1e-6 or y_span <= 1e-6:
        return

    right_edge = min(x0, x1) if ax.xaxis_inverted() else max(x0, x1)
    left_edge = max(x0, x1) if ax.xaxis_inverted() else min(x0, x1)
    bottom_edge = max(y0, y1) if ax.yaxis_inverted() else min(y0, y1)
    margin_x = pad_frac * x_span
    margin_y = 0.022 * y_span

    if side == "left":
        if ax.xaxis_inverted():
            x_start = left_edge - margin_x
            x_end = x_start - length_data
        else:
            x_start = left_edge + margin_x
            x_end = x_start + length_data
    else:
        if ax.xaxis_inverted():
            x_end = right_edge + margin_x
            x_start = x_end + length_data
        else:
            x_end = right_edge - margin_x
            x_start = x_end - length_data
    if ax.yaxis_inverted():
        y_bar = bottom_edge - margin_y
        text_y = y_bar - 0.006 * y_span
    else:
        y_bar = bottom_edge + margin_y
        text_y = y_bar + 0.006 * y_span
    tick_h = 0.012 * y_span

    line_kwargs = dict(color=color, linewidth=linewidth, solid_capstyle="butt", zorder=20)
    ax.plot([x_start, x_end], [y_bar, y_bar], **line_kwargs)
    ax.plot([x_start, x_start], [y_bar - tick_h, y_bar + tick_h], **line_kwargs)
    ax.plot([x_end, x_end], [y_bar - tick_h, y_bar + tick_h], **line_kwargs)
    text_kwargs = dict(
        ha="center",
        va="bottom",
        color=color,
        fontsize=18,
        fontweight="bold",
        zorder=21,
        path_effects=[pe.Stroke(linewidth=2.0, foreground="#fffef8"), pe.Normal()],
    )
    scale_font_prop = None
    if font_prop is not None and CJK_FONT_PATH.exists():
        scale_font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH), size=17, weight="bold")
        text_kwargs["fontproperties"] = scale_font_prop
        text_kwargs.pop("fontsize", None)
        text_kwargs.pop("fontweight", None)
    text = ax.text(0.5 * (x_start + x_end), text_y, label, **text_kwargs)
    text.set_fontsize(17)
    text.set_fontweight("bold")
    if scale_font_prop is not None:
        text.set_fontproperties(scale_font_prop)


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
    split_world_points: Sequence[Tuple[int, float, float]] | None = None,
) -> None:
    font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH)) if CJK_FONT_PATH.exists() else None
    paper_font_size = 13
    legend_font_size = 15
    fig = plt.figure(figsize=(10.4, 6.8), facecolor="#fbfbf8")
    grid_spec = fig.add_gridspec(2, 2, width_ratios=[0.82, 1.18], height_ratios=[1.0, 1.0])
    local_axes = [fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[1, 0])]
    global_ax = fig.add_subplot(grid_spec[:, 1])
    axes = [*local_axes, global_ax]
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
    local_axes[0].set_anchor("E")
    local_axes[1].set_anchor("E")
    global_ax.set_anchor("W")

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
    local_view_override_path = out_path.parent / "local_view_overrides.json"
    local_view_overrides = None
    if local_view_override_path.exists():
        with local_view_override_path.open("r", encoding="utf-8") as handle:
            local_view_overrides = json.load(handle)

    sx, sy = start_xy
    gx, gy = goal_xy
    route_colors = {
        "raw": "#4C78A8",
        "seed": "#E69F00",
        "smooth": "#009E73",
    }
    for ax in axes:
        ax.scatter([sx], [sy], c="#e63b2e", s=60, marker="x", linewidths=1.7, zorder=7)
        ax.scatter([gx], [gy], c="#31c93c", s=42, marker="o", zorder=7)

    def _plot_route_triplet(ax):
        ax.plot(
            raw_grid[:, 0],
            raw_grid[:, 1],
            color=route_colors["raw"],
            linewidth=1.25,
            linestyle="--",
            label="前端路径",
            zorder=5,
            path_effects=route_outline,
        )
        ax.plot(
            seed_grid[:, 0],
            seed_grid[:, 1],
            color=route_colors["seed"],
            linewidth=1.45,
            linestyle="-",
            alpha=0.98,
            label="初始路径",
            zorder=6,
        )
        ax.plot(
            smooth_grid[:, 0],
            smooth_grid[:, 1],
            color=route_colors["smooth"],
            linewidth=1.7,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="优化路径",
            zorder=7,
            path_effects=smooth_outline,
        )

    focus_regions: List[Tuple[int | None, float, float, int]] = []
    if seed_grid.shape == smooth_grid.shape and seed_grid.size > 0:
        diff = np.linalg.norm(seed_grid - smooth_grid, axis=1)
        used_idx_windows: List[Tuple[int, int]] = []

        def _register_focus(near_idx: int) -> None:
            used_idx_windows.append((max(0, near_idx - 24), min(len(diff), near_idx + 25)))

        def _is_used(near_idx: int) -> bool:
            return any(lo <= near_idx < hi for lo, hi in used_idx_windows)

        if split_grid:
            scored_regions = []
            for idx, xg, yg in split_grid:
                # Do not waste local detail panels on start/goal markers when real split points exist.
                if (abs(xg - sx) < 1e-3 and abs(yg - sy) < 1e-3) or (abs(xg - gx) < 1e-3 and abs(yg - gy) < 1e-3):
                    continue
                dist = np.linalg.norm(smooth_grid - np.array([xg, yg], dtype=np.float32), axis=1)
                near_idx = int(np.argmin(dist))
                lo = max(0, near_idx - 18)
                hi = min(len(diff), near_idx + 19)
                local_score = float(np.max(diff[lo:hi]))
                scored_regions.append((local_score, idx, xg, yg, near_idx))
            scored_regions.sort(key=lambda item: item[0], reverse=True)
            for _, idx, xg, yg, near_idx in scored_regions[:2]:
                focus_regions.append((idx, xg, yg, near_idx))
                _register_focus(near_idx)

        while len(focus_regions) < 2:
            focus_idx = int(np.argmax(diff))
            sorted_idx = np.argsort(diff)[::-1]
            chosen_idx = None
            for candidate_idx in sorted_idx:
                if not _is_used(int(candidate_idx)):
                    chosen_idx = int(candidate_idx)
                    break
            if chosen_idx is None:
                chosen_idx = focus_idx
            xg, yg = smooth_grid[chosen_idx]
            focus_regions.append((None, float(xg), float(yg), chosen_idx))
            _register_focus(chosen_idx)

    if len(focus_regions) == 1:
        focus_regions.append(focus_regions[0])
    if not focus_regions:
        fallback_idx = max(0, len(smooth_grid) // 2)
        xg, yg = smooth_grid[fallback_idx]
        focus_regions = [(None, float(xg), float(yg), fallback_idx)] * 2

    local_view_specs = []
    common_local_side = 0.0
    for local_plot_idx in range(len(local_axes)):
        split_idx, focus_x, focus_y, focus_idx = focus_regions[min(local_plot_idx, len(focus_regions) - 1)]
        raw_focus_idx = int(np.argmin(np.linalg.norm(raw_grid - np.array([focus_x, focus_y], dtype=np.float32), axis=1)))
        seed_lo = max(0, focus_idx - 26)
        seed_hi = min(len(seed_grid), focus_idx + 28)
        raw_lo = max(0, raw_focus_idx - 18)
        raw_hi = min(len(raw_grid), raw_focus_idx + 20)
        focus_arrays = [
            raw_grid[raw_lo:raw_hi],
            seed_grid[seed_lo:seed_hi],
            smooth_grid[seed_lo:seed_hi],
            np.array([[focus_x, focus_y]], dtype=np.float32),
        ]
        focus_stack = np.vstack(focus_arrays)
        x_min = float(np.min(focus_stack[:, 0]))
        x_max = float(np.max(focus_stack[:, 0]))
        y_min = float(np.min(focus_stack[:, 1]))
        y_max = float(np.max(focus_stack[:, 1]))
        pad = 1.15
        local_zoom_scale = 0.84
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        side = (max(x_max - x_min, y_max - y_min) + 2.0 * pad) * local_zoom_scale
        common_local_side = max(common_local_side, side)
        local_view_specs.append(
            {
                "split_idx": split_idx,
                "focus_x": float(focus_x),
                "focus_y": float(focus_y),
                "cx": float(cx),
                "cy": float(cy),
                "side": float(side),
                "goal_dist": float(math.hypot(float(focus_x) - gx, float(focus_y) - gy)),
            }
        )

    for local_plot_idx, ax in enumerate(local_axes):
        _plot_route_triplet(ax)
        view_spec = local_view_specs[min(local_plot_idx, len(local_view_specs) - 1)]
        split_idx = view_spec["split_idx"]
        focus_x = view_spec["focus_x"]
        focus_y = view_spec["focus_y"]
        side = common_local_side if common_local_side > 0.0 else view_spec["side"]
        half_side = 0.5 * side
        x0 = view_spec["cx"] - half_side
        x1 = view_spec["cx"] + half_side
        y0 = view_spec["cy"] - half_side
        y1 = view_spec["cy"] + half_side
        if view_spec["goal_dist"] <= 0.42 * side and not local_view_overrides:
            # For goal-focused local panels, anchor the goal near the lower-left
            # corner so the approach geometry remains visible to the right/upstream.
            goal_left_frac = 0.16
            goal_from_top_frac = 0.82
            x0 = gx - goal_left_frac * side
            x1 = x0 + side
            y0 = gy - goal_from_top_frac * side
            y1 = y0 + side
        if local_plot_idx == 1 and not local_view_overrides:
            shift_right = 0.12 * side
            shift_down = 0.14 * side
            x0 += shift_right
            x1 += shift_right
            y0 += shift_down
            y1 += shift_down
        if local_view_overrides and local_plot_idx < len(local_view_overrides):
            override = local_view_overrides[local_plot_idx]
            x0 = float(override["xlim"][0])
            x1 = float(override["xlim"][1])
            y0 = float(override["ylim"][0])
            y1 = float(override["ylim"][1])
            split_idx = override.get("split_idx", split_idx)
            focus_x = float(override.get("focus_x", focus_x))
            focus_y = float(override.get("focus_y", focus_y))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        else:
            ax.set_aspect("equal", adjustable="box")
        if split_idx is not None:
            ax.scatter(
                [focus_x],
                [focus_y],
                s=52,
                c="#1f4e99",
                edgecolors="#fffef8",
                linewidths=1.0,
                zorder=9,
            )
            ax.text(
                focus_x + 0.35,
                focus_y - 0.35,
                str(split_idx + 1),
                fontsize=18,
                color="#1f4e99",
                ha="left",
                va="center",
                zorder=10,
                path_effects=[pe.Stroke(linewidth=1.4, foreground="#fffef8"), pe.Normal()],
                fontproperties=font_prop,
            )
        panel_label = "(a) 局部细节1" if local_plot_idx == 0 else "(b) 局部细节2"
        panel_text = ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=paper_font_size + 4,
            fontweight="semibold",
            color="#222222",
            fontproperties=font_prop,
        )
        panel_text.set_fontsize(paper_font_size + 2)

    global_ax.plot(
        raw_grid[:, 0],
        raw_grid[:, 1],
        color=route_colors["raw"],
        linewidth=1.2,
        alpha=0.95,
        linestyle="--",
        label="前端路径",
        zorder=3,
    )
    global_ax.plot(
        smooth_grid[:, 0],
        smooth_grid[:, 1],
        color=route_colors["smooth"],
        linewidth=1.35,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="优化路径",
        zorder=5,
        path_effects=smooth_outline,
    )
    if seed_overlaps_smooth:
        marker_step = max(1, len(seed_grid) // 18)
        global_ax.plot(
            seed_grid[:, 0],
            seed_grid[:, 1],
            color=route_colors["seed"],
            linewidth=0.0,
            linestyle="None",
            marker="o",
            markersize=3.3,
            markevery=marker_step,
            markeredgewidth=0.45,
            markeredgecolor="#fffef8",
            alpha=0.96,
            label="初始路径",
            zorder=6,
        )
    else:
        global_ax.plot(
            seed_grid[:, 0],
            seed_grid[:, 1],
            color=route_colors["seed"],
            linewidth=1.2,
            alpha=0.98,
            linestyle="-",
            label="初始路径",
            zorder=4,
        )

    global_text = global_ax.text(
        0.02,
        0.98,
        "(c) 全局路径结果",
        transform=global_ax.transAxes,
        ha="left",
        va="top",
        fontsize=paper_font_size + 2,
        fontweight="semibold",
        color="#222222",
        fontproperties=font_prop,
    )
    global_text.set_fontsize(paper_font_size + 4)

    if split_grid:
        for idx, xg, yg in split_grid:
            global_ax.scatter(
                [xg],
                [yg],
                s=20,
                c="#1f4e99",
                edgecolors="#fffef8",
                linewidths=0.6,
                zorder=8,
            )
            global_ax.text(
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

    add_scale_bar(global_ax, 1.0 / resolution, side="left", font_prop=font_prop)

    legend_handles = [
        Line2D([0], [0], marker="x", color="none", markeredgecolor="#e63b2e", markeredgewidth=1.7, markersize=9, label="起点"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#31c93c", markeredgecolor="#31c93c", markersize=7, label="终点"),
        Line2D([0], [0], color=route_colors["raw"], linestyle="--", linewidth=1.8, label="前端路径"),
        Line2D([0], [0], color=route_colors["seed"], linestyle="-", linewidth=1.8, label="初始路径"),
        Line2D([0], [0], color=route_colors["smooth"], linestyle="-", linewidth=2.0, label="优化路径"),
    ]
    legend_kwargs = dict(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=5,
        framealpha=0.96,
        facecolor="#fffef9",
        edgecolor="#c5c2b8",
        handlelength=1.9,
        columnspacing=0.9,
        borderpad=0.38,
    )
    if font_prop is not None:
        legend_kwargs["prop"] = fm.FontProperties(fname=str(CJK_FONT_PATH), size=legend_font_size)
    else:
        legend_kwargs["fontsize"] = legend_font_size
    legend = fig.legend(**legend_kwargs)

    fig.subplots_adjust(left=0.028, right=0.992, bottom=0.03, top=0.91, wspace=0.002, hspace=0.008)
    fig.canvas.draw()
    global_pos = global_ax.get_position()
    local_top_pos = local_axes[0].get_position()
    gap = 0.012
    local_total_height = global_pos.height
    each_height = 0.5 * (local_total_height - gap)
    left_x0 = local_top_pos.x0
    max_width = max(0.0, global_pos.x0 - left_x0 - 0.012)
    each_width = min(each_height, max_width)
    left_x = global_pos.x0 - 0.012 - each_width
    local_axes[0].set_position([left_x, global_pos.y0 + each_height + gap, each_width, each_height])
    local_axes[1].set_position([left_x, global_pos.y0, each_width, each_height])
    group_left = min(local_axes[0].get_position().x0, local_axes[1].get_position().x0, global_ax.get_position().x0)
    group_right = max(local_axes[0].get_position().x1, local_axes[1].get_position().x1, global_ax.get_position().x1)
    group_center = 0.5 * (group_left + group_right)
    legend.set_bbox_to_anchor((group_center, 0.992), transform=fig.transFigure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_path.with_name("offline_demo_paper.png"), dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    safe_save_pdf(fig, out_path.with_name("offline_demo_paper.pdf"))
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
    ax.legend(loc="upper center", fontsize=8.5, framealpha=0.94, ncol=4, facecolor="#fffef9", edgecolor="#c5c2b8")
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    safe_save_pdf(fig, out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_curvature_compare(
    raw_world_path: Sequence[Tuple[float, float]],
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    out_path: Path,
    include_raw_path: bool = True,
    split_world_points: Sequence[Tuple[int, float, float]] = (),
    show_legend: bool = False,
    curvature_limit: float = 1.0,
) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH)) if CJK_FONT_PATH.exists() else None
    paper_font_size = 17
    tick_font_size = 24
    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#a9adb3")
    ax.set_axisbelow(True)
    ax.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.85)

    curve_specs = [
        ("初始路径", seed_world_path, "#E69F00", "-", 1.7),
        ("优化路径", smoothed_world_path, "#009E73", "-", 1.9),
    ]
    if include_raw_path:
        curve_specs.insert(0, ("前端路径", raw_world_path, "#4C78A8", "--", 1.6))
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

    for idx, ref_y in enumerate((-curvature_limit, curvature_limit)):
        ax.axhline(
            ref_y,
            color="#D62728",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
            label="最大曲率" if idx == 0 else None,
        )
    split_s = project_split_points_to_arc_length(seed_world_path, split_world_points)
    if split_s:
        ymin, ymax = ax.get_ylim()
        y_text = ymax - 0.04 * (ymax - ymin)
        for idx, split_arc in split_s:
            ax.axvline(split_arc, color="#5B8FF9", linestyle=":", linewidth=0.95, alpha=0.95, zorder=1)
            ax.text(
                split_arc,
                y_text,
                str(idx + 1),
                fontsize=8.5,
                color="#1f4e99",
                ha="center",
                va="bottom",
                zorder=6,
                path_effects=[pe.Stroke(linewidth=1.2, foreground="#fffef8"), pe.Normal()],
            )

    ax.tick_params(axis="both", labelsize=tick_font_size, colors="#3d4148")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if font_prop is not None:
            label.set_fontproperties(font_prop)
        label.set_fontsize(tick_font_size)

    label_kwargs = {"fontsize": paper_font_size, "color": "#22252b"}
    if font_prop is not None:
        label_kwargs["fontproperties"] = font_prop
    ax.set_xlabel("纵向位移 [m]", **label_kwargs)
    ax.set_ylabel("曲率 [1/m]", **label_kwargs)
    ax.xaxis.label.set_fontsize(paper_font_size)
    ax.yaxis.label.set_fontsize(paper_font_size)

    if show_legend:
        legend_font_size = 18
        legend_kwargs = dict(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=legend_font_size,
            framealpha=0.96,
            facecolor="#ffffff",
            edgecolor="#cfd6de",
            borderpad=0.35,
            labelspacing=0.35,
            handlelength=1.9,
            handletextpad=0.5,
            borderaxespad=0.0,
            ncol=1,
        )
        if font_prop is not None:
            legend_kwargs["prop"] = fm.FontProperties(fname=str(CJK_FONT_PATH), size=legend_font_size)
            legend_kwargs.pop("fontsize", None)
        legend = ax.legend(**legend_kwargs)
        if font_prop is not None:
            for text in legend.get_texts():
                text.set_fontproperties(font_prop)

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.95)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    safe_save_pdf(fig, out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_heading_compare(
    raw_world_path: Sequence[Tuple[float, float]],
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    out_path: Path,
    split_world_points: Sequence[Tuple[int, float, float]] = (),
    target_center_deg: float = -200.0,
) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH)) if CJK_FONT_PATH.exists() else None
    paper_font_size = 14
    tick_font_size = 18
    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#a9adb3")
    ax.set_axisbelow(True)
    ax.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.85)

    raw_s_ref, raw_heading_deg = heading_profile(raw_world_path)
    raw_heading_wrapped = normalize_heading_branch(raw_heading_deg, target_center_deg)
    plotted_headings: List[np.ndarray] = []
    curve_specs = [
        ("前端路径", raw_s_ref, raw_heading_wrapped, "#4C78A8", "--", 1.6),
        ("初始路径", *heading_profile(seed_world_path), "#E69F00", "-", 1.7),
        ("优化路径", *heading_profile(smoothed_world_path), "#009E73", "-", 1.9),
    ]
    for label, s_src, heading_src, color, linestyle, linewidth in curve_specs:
        if s_src.size == 0:
            continue
        if raw_s_ref.size > 0 and label != "前端路径":
            plot_s = raw_s_ref
            plot_heading = interpolate_profile_to_reference(raw_s_ref, s_src, heading_src)
            plot_heading = normalize_heading_branch(plot_heading, target_center_deg)
        else:
            plot_s = s_src
            plot_heading = normalize_heading_branch(heading_src, target_center_deg)
        ax.plot(
            plot_s,
            plot_heading,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        plotted_headings.append(np.asarray(plot_heading, dtype=np.float64))

    if plotted_headings:
        max_dev = max(float(np.max(np.abs(h - target_center_deg))) for h in plotted_headings if h.size > 0)
        half_range = max(35.0, math.ceil((max_dev + 8.0) / 10.0) * 10.0)
        ax.set_ylim(target_center_deg - half_range, target_center_deg + half_range)

    split_s = project_split_points_to_arc_length(raw_world_path, split_world_points)
    if split_s:
        ymin, ymax = ax.get_ylim()
        y_text = ymax - 0.04 * (ymax - ymin)
        for idx, split_arc in split_s:
            ax.axvline(split_arc, color="#5B8FF9", linestyle=":", linewidth=0.95, alpha=0.95, zorder=1)
            ax.text(
                split_arc,
                y_text,
                str(idx + 1),
                fontsize=8.5,
                color="#1f4e99",
                ha="center",
                va="bottom",
                zorder=6,
                path_effects=[pe.Stroke(linewidth=1.2, foreground="#fffef8"), pe.Normal()],
            )

    ax.tick_params(axis="both", labelsize=tick_font_size, colors="#3d4148")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if font_prop is not None:
            label.set_fontproperties(font_prop)
        label.set_fontsize(tick_font_size)

    label_kwargs = {"fontsize": paper_font_size, "color": "#22252b"}
    if font_prop is not None:
        label_kwargs["fontproperties"] = font_prop
    ax.set_xlabel("纵向位移 [m]", **label_kwargs)
    ax.set_ylabel("航向角 [deg]", **label_kwargs)

    legend_font_size = 16
    legend_kwargs = dict(
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        fontsize=legend_font_size,
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#cfd6de",
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.9,
        handletextpad=0.5,
        borderaxespad=0.0,
        ncol=1,
    )
    if font_prop is not None:
        legend_kwargs["prop"] = fm.FontProperties(fname=str(CJK_FONT_PATH), size=legend_font_size)
        legend_kwargs.pop("fontsize", None)
    legend = ax.legend(**legend_kwargs)
    if font_prop is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
            text.set_fontsize(legend_font_size)

    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.95)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    safe_save_pdf(fig, out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_xy_trajectory_compare(
    raw_world_path: Sequence[Tuple[float, float]],
    seed_world_path: Sequence[Tuple[float, float]],
    smoothed_world_path: Sequence[Tuple[float, float]],
    out_path: Path,
) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH)) if CJK_FONT_PATH.exists() else None
    paper_font_size = 12
    tick_font_size = 15
    legend_font_size = 14
    legend_font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH), size=legend_font_size) if CJK_FONT_PATH.exists() else None
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.0), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#a9adb3")
    ax.set_axisbelow(True)
    ax.grid(True, color="#d9dde3", linewidth=0.7, alpha=0.85)

    curve_specs = [
        ("前端路径", raw_world_path, "#4C78A8", "--", 1.7),
        ("初始路径", seed_world_path, "#E69F00", "-", 1.8),
        ("优化路径", smoothed_world_path, "#009E73", "-", 2.0),
    ]
    for label, path, color, linestyle, linewidth in curve_specs:
        arr = np.asarray(path, dtype=np.float32)
        if arr.size == 0:
            continue
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=tick_font_size, colors="#3d4148")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if font_prop is not None:
            label.set_fontproperties(font_prop)
        label.set_fontsize(tick_font_size)

    label_kwargs = {"fontsize": paper_font_size, "color": "#22252b"}
    if font_prop is not None:
        label_kwargs["fontproperties"] = font_prop
    ax.set_xlabel("X 坐标 [m]", **label_kwargs)
    ax.set_ylabel("Y 坐标 [m]", **label_kwargs)

    legend_kwargs = dict(
        loc="lower left",
        bbox_to_anchor=(0.04, 0.04),
        fontsize=legend_font_size,
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#cfd6de",
        borderpad=0.7,
        labelspacing=0.6,
        handlelength=2.6,
        borderaxespad=0.0,
    )
    if legend_font_prop is not None:
        legend_kwargs["prop"] = legend_font_prop
        legend_kwargs.pop("fontsize", None)
    legend = ax.legend(**legend_kwargs)
    if legend_font_prop is not None:
        for text in legend.get_texts():
            text.set_fontproperties(legend_font_prop)

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.13, top=0.97)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    safe_save_pdf(fig, out_path.with_suffix(".pdf"))
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
        seed_world_path = read_xy_path_csv(input_dir / "frontend_seed_path.csv")
        smoothed_world_path = read_xy_path_csv(input_dir / "smoothed_path.csv")
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
        heading_png = input_dir / "heading_compare.png"
        trajectory_xy_png = input_dir / "trajectory_xy_compare.png"
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
        )
        plot_curvature_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=curvature_png,
            split_world_points=split_world_points,
            curvature_limit=0.8,
        )
        plot_curvature_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=curvature_seed_smooth_png,
            include_raw_path=False,
            split_world_points=split_world_points,
            curvature_limit=0.8,
        )
        plot_heading_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=heading_png,
            split_world_points=split_world_points,
        )
        plot_xy_trajectory_compare(
            raw_world_path=raw_world_path,
            seed_world_path=seed_world_path,
            smoothed_world_path=smoothed_world_path,
            out_path=trajectory_xy_png,
        )
        print(f"saved_png={fig_png}")
        print(f"saved_split_debug_png={split_debug_png}")
        print(f"saved_curvature_png={curvature_png}")
        print(f"saved_curvature_seed_smooth_png={curvature_seed_smooth_png}")
        print(f"saved_heading_png={heading_png}")
        print(f"saved_trajectory_xy_png={trajectory_xy_png}")
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
    heading_png = out_dir / "heading_compare.png"
    trajectory_xy_png = out_dir / "trajectory_xy_compare.png"
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
    seed_world_path = read_smoothed_path_csv(seed_csv)
    smoothed_world_path = read_smoothed_path_csv(smooth_csv)
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
    )
    plot_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=curvature_png,
        split_world_points=split_world_points,
        curvature_limit=0.8,
    )
    plot_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=curvature_seed_smooth_png,
        include_raw_path=False,
        split_world_points=split_world_points,
        curvature_limit=0.8,
    )
    plot_heading_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=heading_png,
        split_world_points=split_world_points,
    )
    plot_xy_trajectory_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=trajectory_xy_png,
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
    print(f"saved_heading_png={heading_png}")
    print(f"saved_trajectory_xy_png={trajectory_xy_png}")
    print(f"saved_meta={meta_json}")
    print(f"map_index={map_index}")
    print(f"start_xy={start_xy}")
    print(f"goal_xy={goal_xy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
