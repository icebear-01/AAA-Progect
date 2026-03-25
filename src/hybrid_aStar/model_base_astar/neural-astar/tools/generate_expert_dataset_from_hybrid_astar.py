"""Generate yaw-aware expert guidance dataset using traditional Hybrid A*.

This script supports two source modes:
- Random maps generated on the fly.
- Existing map banks loaded from ``.npz`` files.

Saved schema per sample (.npz):
- occ_map: [H, W], float32, 1=obstacle, 0=free
- start_map: [1, H, W], float32 one-hot
- goal_map: [1, H, W], float32 one-hot
- opt_traj: [1, H, W], float32 dilated expert corridor
- target_cost: [1, H, W], float32 dense distance-field target
- expanded_trace_map: [1, H, W], float32 normalized projection of expanded Hybrid A* rollout cells
- opt_traj_orient: [K, H, W], optional orientation-aware corridor
- target_cost_orient: [K, H, W], optional orientation-aware dense target
- start_pose: [3], float32, (x, y, yaw_rad)
- goal_pose: [3], float32, (x, y, yaw_rad)
- path_poses: [N, 3], float32, raw expert Hybrid A* path
- expanded_trace_xy: [M, 2], optional float32 raw expanded rollout points
- source_map_index: scalar int32, map index in source bank (-1 for random maps)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from hybrid_astar_guided import GuidedHybridAstar
from hybrid_astar_guided.grid_astar import astar_8conn
from neural_astar.utils.coords import make_one_hot_xy
from neural_astar.utils.guidance_targets import (
    build_expanded_trace_map,
    build_orientation_target_maps,
    build_target_cost_map,
)


PoseF = Tuple[float, float, float]


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


def path_to_mask(path: Sequence[PoseF], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    for x, y, _ in path:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            mask[yi, xi] = 1.0
    return mask


def random_free_xy(occ: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    h, w = occ.shape
    for _ in range(10000):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if occ[y, x] < 0.5:
            return x, y
    raise RuntimeError("Could not sample free cell")


def _pick_occ_key(keys: Sequence[str]) -> str:
    for key in ("occ_map", "occupancy_map", "occupancy", "map", "arr_0", "arr_4", "arr_8"):
        if key in keys:
            return key
    raise ValueError("Could not find occupancy key in source npz")


def _as_map_bank(arr: np.ndarray, key: str) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2:
        return x[None, ...]
    if x.ndim == 3:
        return x
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[:, :, :, 0]
    raise ValueError(f"Cannot convert key={key} with shape={x.shape} to [N,H,W]")


def _resolve_occ_semantics(key: str, occ_semantics: str) -> str:
    if occ_semantics != "auto":
        return occ_semantics
    if key.startswith("arr_"):
        return "passable1"
    return "obstacle1"


def load_occ_map_bank(
    npz_path: Path,
    split: str,
    occ_semantics: str,
) -> Tuple[np.ndarray, str, str]:
    with np.load(npz_path) as data:
        keys = list(data.files)
        if "arr_0" in data.files:
            key = {"train": "arr_0", "valid": "arr_4", "test": "arr_8"}[split]
        else:
            key = _pick_occ_key(keys)
        maps = _as_map_bank(np.asarray(data[key], dtype=np.float32), key=key)

    resolved_mode = _resolve_occ_semantics(key, occ_semantics)
    if resolved_mode == "obstacle1":
        occ_maps = maps
    elif resolved_mode == "passable1":
        occ_maps = 1.0 - maps
    else:
        raise ValueError(f"Unknown occ semantics: {resolved_mode}")
    return occ_maps.astype(np.float32), key, resolved_mode


def build_traditional_hybrid_astar(args: argparse.Namespace) -> GuidedHybridAstar:
    return GuidedHybridAstar(
        yaw_bins=args.yaw_bins,
        n_steer=args.n_steer,
        motion_step=args.motion_step,
        primitive_length=args.primitive_length,
        wheel_base=args.wheel_base,
        max_steer=args.max_steer,
        allow_reverse=args.allow_reverse,
        strict_goal_pose=args.strict_goal_pose,
        use_rs_shot=args.use_rs_shot,
        goal_tolerance_yaw_deg=args.goal_tolerance_yaw_deg,
        steer_penalty=args.steer_penalty,
        reverse_penalty=args.reverse_penalty,
        steer_change_penalty=args.steer_change_penalty,
        direction_change_penalty=args.direction_change_penalty,
    )


def run_hybrid_astar(
    planner: GuidedHybridAstar,
    occ_map: np.ndarray,
    start_pose: PoseF,
    goal_pose: PoseF,
    max_expansions: int,
) -> Optional[object]:
    zero_cost = np.zeros_like(occ_map, dtype=np.float32)
    res = planner.plan(
        occ_map=occ_map,
        start_pose=start_pose,
        goal_pose=goal_pose,
        cost_map=zero_cost,
        lambda_guidance=0.0,
        max_expansions=max_expansions,
    )
    if not res.success or len(res.path) < 2:
        return None
    return res


def sample_yaw(
    rng: np.random.Generator,
    mode: str,
    yaw_deg: Optional[float],
) -> float:
    if yaw_deg is not None:
        return math.radians(float(yaw_deg))
    if mode == "zero":
        return 0.0
    if mode == "random":
        return float(rng.uniform(0.0, 2.0 * math.pi))
    raise ValueError(f"Unknown yaw mode: {mode}")


def build_random_occ_map(
    size: int,
    obstacle_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random((size, size)) < obstacle_prob).astype(np.float32)


def sample_problem_on_map(
    occ: np.ndarray,
    rng: np.random.Generator,
    min_start_goal_dist: float,
    start_yaw_mode: str,
    goal_yaw_mode: str,
    start_yaw_deg: Optional[float],
    goal_yaw_deg: Optional[float],
    max_trials: int,
) -> Tuple[Tuple[int, int], Tuple[int, int], PoseF, PoseF]:
    h, w = occ.shape
    for _ in range(max_trials):
        start_xy = random_free_xy(occ, rng)
        goal_xy = random_free_xy(occ, rng)
        if start_xy == goal_xy:
            continue
        if math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]) < min_start_goal_dist:
            continue
        if astar_8conn(occ, start_xy, goal_xy) is None:
            continue

        start_pose = (
            float(start_xy[0]),
            float(start_xy[1]),
            sample_yaw(rng, start_yaw_mode, start_yaw_deg),
        )
        goal_pose = (
            float(goal_xy[0]),
            float(goal_xy[1]),
            sample_yaw(rng, goal_yaw_mode, goal_yaw_deg),
        )
        if not (0 <= int(round(start_pose[0])) < w and 0 <= int(round(start_pose[1])) < h):
            continue
        if not (0 <= int(round(goal_pose[0])) < w and 0 <= int(round(goal_pose[1])) < h):
            continue
        return start_xy, goal_xy, start_pose, goal_pose

    raise RuntimeError("Failed to sample a feasible start/goal pair on map")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate expert dataset from traditional Hybrid A*")
    p.add_argument("--occ-npz", type=Path, default=None, help="Optional existing map bank (.npz)")
    p.add_argument(
        "--source-split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Split used when source npz is planning-datasets format.",
    )
    p.add_argument(
        "--occ-semantics",
        type=str,
        default="auto",
        choices=["auto", "obstacle1", "passable1"],
        help="Semantics of source occupancy arrays.",
    )
    p.add_argument("--map-index", type=int, default=0, help="Start index when reusing existing maps.")
    p.add_argument("--random-map-index", action="store_true", help="Sample source map index randomly.")

    p.add_argument("--num-maps", type=int, default=50)
    p.add_argument("--pairs-per-map", type=int, default=4)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--obstacle-prob", type=float, default=0.22)
    p.add_argument("--min-start-goal-dist", type=float, default=0.0)
    p.add_argument("--max-trials-per-pair", type=int, default=300)
    p.add_argument("--dilation-radius", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--yaw-bins", type=int, default=72)
    p.add_argument("--n-steer", type=int, default=5)
    p.add_argument("--motion-step", type=float, default=0.5)
    p.add_argument("--primitive-length", type=float, default=2.5)
    p.add_argument("--wheel-base", type=float, default=2.7)
    p.add_argument("--max-steer", type=float, default=0.60)
    p.add_argument("--allow-reverse", dest="allow_reverse", action="store_true")
    p.add_argument("--no-allow-reverse", dest="allow_reverse", action="store_false")
    p.add_argument("--strict-goal-pose", action="store_true")
    p.add_argument("--use-rs-shot", action="store_true")
    p.add_argument("--goal-tolerance-yaw-deg", type=float, default=5.0)
    p.add_argument("--max-expansions", type=int, default=40000)
    p.add_argument("--steer-penalty", type=float, default=0.2)
    p.add_argument("--reverse-penalty", type=float, default=2.0)
    p.add_argument("--steer-change-penalty", type=float, default=0.2)
    p.add_argument("--direction-change-penalty", type=float, default=1.0)

    p.add_argument(
        "--start-yaw-mode",
        type=str,
        default="random",
        choices=["random", "zero"],
        help="How to sample start yaw.",
    )
    p.add_argument(
        "--goal-yaw-mode",
        type=str,
        default="random",
        choices=["random", "zero"],
        help="How to sample goal yaw.",
    )
    p.add_argument("--start-yaw-deg", type=float, default=None)
    p.add_argument("--goal-yaw-deg", type=float, default=None)
    p.add_argument(
        "--guidance-orientation-bins",
        type=int,
        default=1,
        help="Optional number of yaw bins used to precompute orientation-aware guidance labels.",
    )
    p.add_argument(
        "--save-expanded-trace-xy",
        action="store_true",
        help="Also save raw expanded_trace_xy arrays. Disabled by default to keep dataset size smaller.",
    )
    p.set_defaults(allow_reverse=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    planner = build_traditional_hybrid_astar(args)

    occ_bank = None
    source_key = "random"
    resolved_mode = "obstacle1"
    if args.occ_npz is not None:
        occ_bank, source_key, resolved_mode = load_occ_map_bank(
            npz_path=args.occ_npz,
            split=args.source_split,
            occ_semantics=args.occ_semantics,
        )
        print(
            f"loaded source maps: {args.occ_npz} key={source_key} "
            f"semantics={resolved_mode} count={occ_bank.shape[0]}"
        )

    saved = 0
    failed = 0

    for map_slot in range(args.num_maps):
        if occ_bank is None:
            source_map_index = -1
            occ = build_random_occ_map(args.size, args.obstacle_prob, rng)
        else:
            if args.random_map_index:
                source_map_index = int(rng.integers(0, occ_bank.shape[0]))
            else:
                source_map_index = int(args.map_index + map_slot)
                if source_map_index >= occ_bank.shape[0]:
                    break
            occ = occ_bank[source_map_index].astype(np.float32)

        h, w = occ.shape
        for pair_idx in range(args.pairs_per_map):
            ok = False
            for _ in range(args.max_trials_per_pair):
                try:
                    start_xy, goal_xy, start_pose, goal_pose = sample_problem_on_map(
                        occ=occ,
                        rng=rng,
                        min_start_goal_dist=float(args.min_start_goal_dist),
                        start_yaw_mode=args.start_yaw_mode,
                        goal_yaw_mode=args.goal_yaw_mode,
                        start_yaw_deg=args.start_yaw_deg,
                        goal_yaw_deg=args.goal_yaw_deg,
                        max_trials=1,
                    )
                except RuntimeError:
                    continue

                plan_res = run_hybrid_astar(
                    planner=planner,
                    occ_map=occ,
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    max_expansions=int(args.max_expansions),
                )
                if plan_res is None:
                    continue
                path = [(float(x), float(y), float(th)) for x, y, th in plan_res.path]
                expanded_trace_xy = np.asarray(plan_res.expanded_trace_xy, dtype=np.float32).reshape(-1, 2)

                opt = path_to_mask(path, h, w)
                opt = dilate_binary(opt, radius=args.dilation_radius)
                target_cost = build_target_cost_map(occ, opt)[None, ...].astype(np.float32)
                expanded_trace_map = build_expanded_trace_map(
                    occ_map=occ,
                    expanded_trace_xy=expanded_trace_xy,
                )[None, ...].astype(np.float32)
                orient_traj = None
                orient_target = None
                if int(args.guidance_orientation_bins) > 1:
                    orient_traj_arr, orient_target_arr = build_orientation_target_maps(
                        occ_map=occ,
                        opt_traj=opt,
                        path_poses=np.asarray(path, dtype=np.float32),
                        yaw_bins=int(args.guidance_orientation_bins),
                    )
                    orient_traj = orient_traj_arr.astype(np.float32)
                    orient_target = orient_target_arr.astype(np.float32)

                sample_id = f"sample_{saved:05d}_map_{source_map_index:04d}_pair_{pair_idx:03d}"
                out_file = args.out_dir / f"{sample_id}.npz"
                payload = {
                    "occ_map": occ.astype(np.float32),
                    "start_map": make_one_hot_xy(start_xy[0], start_xy[1], w, h)[None, ...].astype(
                        np.float32
                    ),
                    "goal_map": make_one_hot_xy(goal_xy[0], goal_xy[1], w, h)[None, ...].astype(
                        np.float32
                    ),
                    "opt_traj": opt[None, ...].astype(np.float32),
                    "target_cost": target_cost,
                    "expanded_trace_map": expanded_trace_map,
                    "start_pose": np.asarray(start_pose, dtype=np.float32),
                    "goal_pose": np.asarray(goal_pose, dtype=np.float32),
                    "path_poses": np.asarray(path, dtype=np.float32),
                    "source_map_index": np.asarray(source_map_index, dtype=np.int32),
                }
                if args.save_expanded_trace_xy:
                    payload["expanded_trace_xy"] = expanded_trace_xy.astype(np.float32)
                if orient_traj is not None and orient_target is not None:
                    payload["opt_traj_orient"] = orient_traj
                    payload["target_cost_orient"] = orient_target
                np.savez_compressed(out_file, **payload)
                saved += 1
                ok = True
                break

            if not ok:
                failed += 1

    print(f"saved_samples={saved}")
    print(f"failed_pairs={failed}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
