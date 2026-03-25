#!/usr/bin/env python3
"""Run transformer-guided 2D A* and export a coarse world-frame path.

Input JSON schema:
{
  "width": int,
  "height": int,
  "resolution": float,
  "origin_x": float,
  "origin_y": float,
  "start_world": [x, y],
  "goal_world": [x, y],
  "start_yaw": float,
  "goal_yaw": float,
  "occupancy": [[0|1, ...], ...]  # [H][W], 1=obstacle
}

Output CSV:
  x,y
  ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from hybrid_astar_guided.grid_astar import astar_8conn
from neural_astar.api.guidance_infer import infer_cost_map


def _load_request(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {
        "width",
        "height",
        "resolution",
        "origin_x",
        "origin_y",
        "start_world",
        "goal_world",
        "occupancy",
    }
    missing = sorted(required.difference(data.keys()))
    if missing:
        raise ValueError(f"request missing keys: {missing}")
    return data


def _world_to_grid(x: float, y: float, origin_x: float, origin_y: float, resolution: float, width: int, height: int) -> Tuple[int, int]:
    gx = int(math.floor((x - origin_x) / resolution))
    gy = int(math.floor((y - origin_y) / resolution))
    gx = max(0, min(width - 1, gx))
    gy = max(0, min(height - 1, gy))
    return gx, gy


def _grid_to_world(gx: int, gy: int, origin_x: float, origin_y: float, resolution: float) -> Tuple[float, float]:
    return (
        float(origin_x + (gx + 0.5) * resolution),
        float(origin_y + (gy + 0.5) * resolution),
    )


def _write_path_csv(path: Path, world_path: List[Tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        for x, y in world_path:
            writer.writerow([f"{x:.6f}", f"{y:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transformer-guided 2D A* frontend.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lambda-guidance", type=float, default=1.0)
    parser.add_argument("--heuristic-mode", type=str, default="octile")
    parser.add_argument("--heuristic-weight", type=float, default=1.0)
    parser.add_argument("--guidance-integration-mode", type=str, default="g_cost")
    parser.add_argument("--guidance-bonus-threshold", type=float, default=0.5)
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    parser.add_argument("--clearance-safe-distance", type=float, default=0.0)
    parser.add_argument("--clearance-power", type=float, default=2.0)
    parser.add_argument(
        "--clearance-integration-mode",
        type=str,
        default="g_cost",
        choices=["g_cost", "heuristic_bias", "priority_tie_break"],
    )
    parser.add_argument("--allow-corner-cut", action="store_true")
    parser.add_argument("--invert-guidance-cost", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    req = _load_request(args.input_json)

    width = int(req["width"])
    height = int(req["height"])
    resolution = float(req["resolution"])
    origin_x = float(req["origin_x"])
    origin_y = float(req["origin_y"])
    start_world = tuple(float(v) for v in req["start_world"])
    goal_world = tuple(float(v) for v in req["goal_world"])
    start_yaw = float(req.get("start_yaw", 0.0))
    goal_yaw = float(req.get("goal_yaw", 0.0))

    occ = np.asarray(req["occupancy"], dtype=np.float32)
    if occ.shape != (height, width):
        raise ValueError(f"occupancy shape mismatch: {occ.shape} vs ({height}, {width})")

    start_xy = _world_to_grid(start_world[0], start_world[1], origin_x, origin_y, resolution, width, height)
    goal_xy = _world_to_grid(goal_world[0], goal_world[1], origin_x, origin_y, resolution, width, height)

    cost_map = infer_cost_map(
        ckpt_path=args.ckpt,
        occ_map_numpy=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_yaw=start_yaw,
        goal_yaw=goal_yaw,
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
        clearance_weight=args.clearance_weight,
        clearance_safe_distance=args.clearance_safe_distance,
        clearance_power=args.clearance_power,
        clearance_integration_mode=args.clearance_integration_mode,
    )
    if path_xy is None or len(path_xy) < 2:
        raise RuntimeError("guided grid A* failed to find a valid path")

    world_path = [_grid_to_world(gx, gy, origin_x, origin_y, resolution) for gx, gy in path_xy]
    world_path[0] = (start_world[0], start_world[1])
    world_path[-1] = (goal_world[0], goal_world[1])
    _write_path_csv(args.output_csv, world_path)
    print(f"saved_path_csv={args.output_csv}")
    print(f"path_points={len(world_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
