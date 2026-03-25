"""Generate temporary parking guidance dataset with 2D expert A*.

Saved schema per sample (.npz):
- occ_map: [H, W], float32, 1=obstacle, 0=free
- start_map: [1, H, W], float32 one-hot
- goal_map: [1, H, W], float32 one-hot
- opt_traj: [1, H, W], float32 corridor mask
- target_cost: [1, H, W], float32 dense distance-field target
- path_poses: [N, 3], float32 pseudo-pose path with estimated yaw
- opt_traj_orient: [K, H, W], optional orientation-aware corridor
- target_cost_orient: [K, H, W], optional orientation-aware dense target
- start_pose: [3], float32, zero-yaw pose
- goal_pose: [3], float32, zero-yaw pose

Coordinate convention:
- world node is (x, y)
- numpy index is [y, x]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from hybrid_astar_guided.grid_astar import astar_8conn
from neural_astar.utils.coords import make_one_hot_xy
from neural_astar.utils.guidance_targets import (
    build_orientation_target_maps,
    build_target_cost_map,
)


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=np.float32)
    ys, xs = np.where(mask > 0.5)
    for y, x in zip(ys, xs):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        out[y0:y1, x0:x1] = 1.0
    return out


def random_free_xy(occ_map: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    h, w = occ_map.shape
    for _ in range(10000):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if occ_map[y, x] < 0.5:
            return x, y
    raise RuntimeError("Could not sample free cell")


def path_to_mask(path_xy: List[Tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    for x, y in path_xy:
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = 1.0
    return mask


def path_xy_to_poses(path_xy: List[Tuple[int, int]]) -> np.ndarray:
    if not path_xy:
        return np.zeros((0, 3), dtype=np.float32)
    poses = []
    last_yaw = 0.0
    for i, (x, y) in enumerate(path_xy):
        if len(path_xy) == 1:
            yaw = 0.0
        elif i + 1 < len(path_xy):
            nx, ny = path_xy[i + 1]
            dx = float(nx - x)
            dy = float(ny - y)
            if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
                yaw = last_yaw
            else:
                yaw = float(np.arctan2(dy, dx))
        else:
            px, py = path_xy[i - 1]
            dx = float(x - px)
            dy = float(y - py)
            if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
                yaw = last_yaw
            else:
                yaw = float(np.arctan2(dy, dx))
        last_yaw = yaw
        poses.append((float(x), float(y), float(yaw)))
    return np.asarray(poses, dtype=np.float32)


def build_sample(
    size: int,
    obstacle_prob: float,
    dilation_radius: int,
    rng: np.random.Generator,
    max_trials: int = 200,
):
    h = size
    w = size
    for _ in range(max_trials):
        occ = (rng.random((h, w)) < obstacle_prob).astype(np.float32)

        start_xy = random_free_xy(occ, rng)
        goal_xy = random_free_xy(occ, rng)
        if start_xy == goal_xy:
            continue

        path = astar_8conn(occ, start_xy, goal_xy)
        if path is None or len(path) < 2:
            continue

        start_map = make_one_hot_xy(start_xy[0], start_xy[1], w, h)[None, ...]
        goal_map = make_one_hot_xy(goal_xy[0], goal_xy[1], w, h)[None, ...]
        opt = path_to_mask(path, h, w)
        opt = dilate_binary(opt, dilation_radius)[None, ...]

        return occ, start_map, goal_map, opt, start_xy, goal_xy, path

    raise RuntimeError("Failed to generate valid sample with path")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dummy parking guidance .npz files")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--obstacle-prob", type=float, default=0.22)
    p.add_argument("--dilation-radius", type=int, default=1)
    p.add_argument(
        "--guidance-orientation-bins",
        type=int,
        default=1,
        help="Optional number of yaw bins used to precompute orientation-aware guidance labels.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.num_samples):
        occ, start_map, goal_map, opt, start_xy, goal_xy, path_xy = build_sample(
            size=args.size,
            obstacle_prob=args.obstacle_prob,
            dilation_radius=args.dilation_radius,
            rng=rng,
        )
        path_poses = path_xy_to_poses(path_xy)
        target_cost = build_target_cost_map(occ, opt[0])[None, ...].astype(np.float32)
        payload = {
            "occ_map": occ.astype(np.float32),
            "start_map": start_map.astype(np.float32),
            "goal_map": goal_map.astype(np.float32),
            "opt_traj": opt.astype(np.float32),
            "target_cost": target_cost,
            "path_poses": path_poses,
            "start_xy": np.array(start_xy, dtype=np.int32),
            "goal_xy": np.array(goal_xy, dtype=np.int32),
            "start_pose": np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float32),
            "goal_pose": np.array([goal_xy[0], goal_xy[1], 0.0], dtype=np.float32),
        }
        if int(args.guidance_orientation_bins) > 1:
            opt_traj_orient, target_cost_orient = build_orientation_target_maps(
                occ_map=occ,
                opt_traj=opt[0],
                path_poses=path_poses,
                yaw_bins=int(args.guidance_orientation_bins),
            )
            payload["opt_traj_orient"] = opt_traj_orient.astype(np.float32)
            payload["target_cost_orient"] = target_cost_orient.astype(np.float32)
        out_file = args.out_dir / f"sample_{i:05d}.npz"
        np.savez_compressed(out_file, **payload)

    print(f"saved {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
