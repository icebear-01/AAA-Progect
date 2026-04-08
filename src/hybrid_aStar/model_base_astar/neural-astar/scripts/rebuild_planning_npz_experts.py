"""Rebuild planning-datasets expert labels while preserving maps and goals.

This script rewrites only the expert-side arrays:
- opt_policy
- opt_dist

It keeps:
- map_design
- goal_map
- split boundaries

The rebuilt expert follows the current grid-planner semantics more closely than
the original planning-datasets utilities by supporting:
- geometric diagonal cost
- optional clearance-aware cost shaping
- optional corner-cut control
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from neural_astar.utils.guidance_targets import build_clearance_penalty_map


ACTION_TO_MOVE_4: List[Tuple[int, int, int]] = [
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
    (0, +1, 0),
]

ACTION_TO_MOVE_8: List[Tuple[int, int, int]] = [
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
    (0, +1, 0),
    (0, -1, +1),
    (0, -1, -1),
    (0, +1, +1),
    (0, +1, -1),
]

XY = Tuple[int, int]


@dataclass
class RebuildStats:
    total_samples: int = 0
    changed_policy_samples: int = 0
    mean_old_geo_len: float = 0.0
    mean_new_geo_len: float = 0.0


def _split_layout() -> Dict[str, Tuple[int, int, int, int]]:
    return {
        "train": (0, 1, 2, 3),
        "valid": (4, 5, 6, 7),
        "test": (8, 9, 10, 11),
    }


def _action_table(mechanism: str) -> List[Tuple[int, int, int]]:
    if mechanism == "news":
        return ACTION_TO_MOVE_4
    if mechanism == "moore":
        return ACTION_TO_MOVE_8
    raise ValueError(f"Unsupported mechanism: {mechanism}")


def _neighbors(
    x: int,
    y: int,
    w: int,
    h: int,
    mechanism: str,
) -> List[Tuple[int, int, int]]:
    actions = ACTION_TO_MOVE_4 if mechanism == "news" else ACTION_TO_MOVE_8
    out: List[Tuple[int, int, int]] = []
    for action_idx, (_, dy, dx) in enumerate(actions):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            out.append((action_idx, nx, ny))
    return out


def _legal_transition(
    occ_map: np.ndarray,
    x: int,
    y: int,
    nx: int,
    ny: int,
    *,
    allow_corner_cut: bool,
) -> bool:
    h, w = occ_map.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    if occ_map[y, x] > 0.5:
        return False
    if nx < 0 or nx >= w or ny < 0 or ny >= h:
        return False
    if occ_map[ny, nx] > 0.5:
        return False
    dx = int(nx - x)
    dy = int(ny - y)
    is_diagonal = (dx != 0) and (dy != 0)
    if not is_diagonal:
        return True
    side_block_x = occ_map[y, nx] > 0.5
    side_block_y = occ_map[ny, x] > 0.5
    if side_block_x and side_block_y:
        return False
    if (not allow_corner_cut) and (side_block_x or side_block_y):
        return False
    return True


def _step_cost(x: int, y: int, nx: int, ny: int, diagonal_cost: float) -> float:
    return float(diagonal_cost) if (int(nx - x) != 0 and int(ny - y) != 0) else 1.0


def _reverse_dijkstra_cost_to_goal(
    occ_map: np.ndarray,
    goal_xy: XY,
    *,
    mechanism: str,
    diagonal_cost: float,
    allow_corner_cut: bool,
    clearance_weight: float,
    clearance_safe_distance: float,
    clearance_power: float,
) -> np.ndarray:
    h, w = occ_map.shape
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    # Keep the reverse-search distance map in float64 so heap values and the
    # stored best cost use the same precision. With float32, valid nodes can be
    # skipped as "stale" due to ~1e-6 rounding drift, which silently breaks
    # connectivity in the rebuilt expert labels.
    dist = np.full((h, w), np.inf, dtype=np.float64)
    if occ_map[gy, gx] > 0.5:
        return dist

    if float(clearance_weight) > 0.0 and float(clearance_safe_distance) > 0.0:
        clearance_penalty = build_clearance_penalty_map(
            occ_map=occ_map,
            safe_distance=float(clearance_safe_distance),
            power=float(clearance_power),
        ).astype(np.float64)
        clearance_penalty[occ_map > 0.5] = 0.0
    else:
        clearance_penalty = np.zeros_like(occ_map, dtype=np.float64)

    dist[gy, gx] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, gx, gy)]

    while heap:
        cur_cost, x, y = heapq.heappop(heap)
        if cur_cost > float(dist[y, x]) + 1e-6:
            continue
        for _, px, py in _neighbors(x, y, w, h, mechanism):
            if not _legal_transition(
                occ_map,
                px,
                py,
                x,
                y,
                allow_corner_cut=allow_corner_cut,
            ):
                continue
            step = _step_cost(px, py, x, y, diagonal_cost)
            edge_cost = step + float(clearance_weight) * float(clearance_penalty[y, x])
            cand = float(cur_cost) + edge_cost
            if cand + 1e-6 < float(dist[py, px]):
                dist[py, px] = cand
                heapq.heappush(heap, (cand, px, py))
    return dist.astype(np.float32)


def _build_policy_from_cost(
    occ_map: np.ndarray,
    goal_xy: XY,
    cost_to_goal: np.ndarray,
    *,
    mechanism: str,
    diagonal_cost: float,
    allow_corner_cut: bool,
    clearance_weight: float,
    clearance_safe_distance: float,
    clearance_power: float,
    policy_tie_break: str,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = occ_map.shape
    actions = _action_table(mechanism)
    policy = np.zeros((len(actions), 1, h, w), dtype=np.float32)

    if policy_tie_break in {"clearance", "clearance_then_first"} and float(clearance_safe_distance) > 0.0:
        clearance_map = build_clearance_penalty_map(
            occ_map=occ_map,
            safe_distance=float(clearance_safe_distance),
            power=float(clearance_power),
        ).astype(np.float32)
        clearance_map[occ_map > 0.5] = 1.0
    else:
        clearance_map = np.zeros_like(occ_map, dtype=np.float32)

    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    for y in range(h):
        for x in range(w):
            if occ_map[y, x] > 0.5:
                continue
            if x == gx and y == gy:
                continue

            candidates: List[Tuple[int, float, float]] = []
            for action_idx, nx, ny in _neighbors(x, y, w, h, mechanism):
                if not _legal_transition(
                    occ_map,
                    x,
                    y,
                    nx,
                    ny,
                    allow_corner_cut=allow_corner_cut,
                ):
                    continue
                nxt_cost = float(cost_to_goal[ny, nx])
                if not np.isfinite(nxt_cost):
                    continue
                step = _step_cost(x, y, nx, ny, diagonal_cost)
                edge_cost = step + float(clearance_weight) * float(clearance_map[ny, nx])
                total_cost = edge_cost + nxt_cost
                candidates.append((action_idx, total_cost, float(clearance_map[ny, nx])))

            if not candidates:
                continue

            min_cost = min(item[1] for item in candidates)
            best = [item for item in candidates if abs(item[1] - min_cost) <= 1e-6]

            if len(best) > 1 and policy_tie_break in {"clearance", "clearance_then_first"}:
                min_clearance_pen = min(item[2] for item in best)
                best = [item for item in best if abs(item[2] - min_clearance_pen) <= 1e-6]
            if len(best) > 1 and policy_tie_break == "random":
                chosen = best[int(rng.integers(0, len(best)))]
            else:
                chosen = min(best, key=lambda item: int(item[0]))
            policy[int(chosen[0]), 0, y, x] = 1.0
    return policy.astype(np.float32)


def _path_length(
    path_xy: Sequence[XY],
    *,
    diagonal_cost: float,
) -> float:
    if len(path_xy) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path_xy)):
        x0, y0 = path_xy[i - 1]
        x1, y1 = path_xy[i]
        total += float(diagonal_cost) if (abs(x1 - x0) == 1 and abs(y1 - y0) == 1) else 1.0
    return float(total)


def _trace_path_from_policy(
    policy: np.ndarray,
    goal_map: np.ndarray,
    cost_to_goal_neg: np.ndarray,
) -> List[XY]:
    goal_y, goal_x = np.unravel_index(int(np.argmax(goal_map[0])), goal_map.shape[-2:])
    flat = cost_to_goal_neg[0].reshape(-1)
    min_v = float(flat.min())
    valid = np.where((flat > min_v) & (flat < 0.0))[0]
    if valid.size == 0:
        return [(int(goal_x), int(goal_y))]
    start_idx = int(valid[np.argmin(flat[valid])])
    sy, sx = np.unravel_index(start_idx, cost_to_goal_neg[0].shape)

    cur = (0, int(sy), int(sx))
    out: List[XY] = [(int(sx), int(sy))]
    max_steps = int(goal_map.shape[-2] * goal_map.shape[-1] * 6)
    for _ in range(max_steps):
        if cur[1] == goal_y and cur[2] == goal_x:
            break
        action = int(np.argmax(policy[:, cur[0], cur[1], cur[2]]))
        _, dy, dx = _action_table("moore" if policy.shape[0] == 8 else "news")[action]
        nxt = (cur[0], cur[1] + dy, cur[2] + dx)
        if nxt == cur:
            break
        if nxt[1] < 0 or nxt[1] >= goal_map.shape[-2] or nxt[2] < 0 or nxt[2] >= goal_map.shape[-1]:
            break
        out.append((int(nxt[2]), int(nxt[1])))
        if nxt == cur:
            break
        cur = nxt
    return out


def _infer_mechanism(npz_path: Path, action_dim: int) -> str:
    name = npz_path.stem.lower()
    if "moore" in name:
        return "moore"
    if "news" in name:
        return "news"
    if int(action_dim) == 8:
        return "moore"
    if int(action_dim) == 4:
        return "news"
    raise ValueError(f"Cannot infer mechanism from file name/action_dim: {npz_path} / {action_dim}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild planning NPZ expert labels with updated planning cost.")
    p.add_argument("--input-npz", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--mechanism", type=str, default=None, choices=["news", "moore"])
    p.add_argument("--diagonal-cost", type=float, default=math.sqrt(2.0))
    p.add_argument("--clearance-weight", type=float, default=0.0)
    p.add_argument("--clearance-safe-distance", type=float, default=0.0)
    p.add_argument("--clearance-power", type=float, default=2.0)
    p.add_argument("--allow-corner-cut", action="store_true", default=True)
    p.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    p.add_argument(
        "--policy-tie-break",
        type=str,
        default="clearance_then_first",
        choices=["first", "random", "clearance", "clearance_then_first"],
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit-per-split", type=int, default=0)
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.diagonal_cost <= 0.0:
        raise ValueError(f"--diagonal-cost must be positive, got {args.diagonal_cost}")
    if args.clearance_weight < 0.0:
        raise ValueError(f"--clearance-weight must be non-negative, got {args.clearance_weight}")
    if args.clearance_safe_distance < 0.0:
        raise ValueError(
            f"--clearance-safe-distance must be non-negative, got {args.clearance_safe_distance}"
        )
    if args.clearance_power <= 0.0:
        raise ValueError(f"--clearance-power must be positive, got {args.clearance_power}")
    if not args.input_npz.exists():
        raise FileNotFoundError(args.input_npz)

    rng = np.random.default_rng(args.seed)
    summary_rows: List[Dict[str, object]] = []
    stats = RebuildStats()

    with np.load(args.input_npz) as data:
        arrays = {name: np.array(data[name]) for name in data.files}

    sample_policy = arrays["arr_2"]
    if sample_policy.ndim != 5 or int(sample_policy.shape[2]) != 1:
        raise ValueError(
            "This script currently supports orientation-free datasets with policy shape [N,A,1,H,W], "
            f"got {sample_policy.shape}"
        )
    action_dim = int(sample_policy.shape[1])
    mechanism = args.mechanism or _infer_mechanism(args.input_npz, action_dim=action_dim)
    expected_actions = 8 if mechanism == "moore" else 4
    if action_dim != expected_actions:
        raise ValueError(
            f"Input action dim {action_dim} does not match mechanism {mechanism} ({expected_actions})"
        )

    unreachable_cost = float(expected_actions * sample_policy.shape[-2] * sample_policy.shape[-1])

    for split_name, (map_idx, goal_idx, policy_idx, dist_idx) in _split_layout().items():
        map_designs = np.asarray(arrays[f"arr_{map_idx}"], dtype=np.float32)
        goal_maps = np.asarray(arrays[f"arr_{goal_idx}"], dtype=np.float32)
        old_policies = np.asarray(arrays[f"arr_{policy_idx}"], dtype=np.float32)
        old_dists = np.asarray(arrays[f"arr_{dist_idx}"], dtype=np.float32)

        rebuild_count = int(map_designs.shape[0])
        if int(args.limit_per_split) > 0:
            rebuild_count = min(rebuild_count, int(args.limit_per_split))

        new_policies = old_policies.copy()
        new_dists = old_dists.copy()

        for local_idx in range(rebuild_count):
            map_design = map_designs[local_idx]
            goal_map = goal_maps[local_idx]
            occ_map = (1.0 - map_design).astype(np.float32)
            goal_y, goal_x = np.unravel_index(int(goal_map[0].argmax()), goal_map.shape[-2:])

            cost_to_goal = _reverse_dijkstra_cost_to_goal(
                occ_map=occ_map,
                goal_xy=(int(goal_x), int(goal_y)),
                mechanism=mechanism,
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.clearance_weight),
                clearance_safe_distance=float(args.clearance_safe_distance),
                clearance_power=float(args.clearance_power),
            )

            cost_to_goal_neg = np.where(
                np.isfinite(cost_to_goal),
                -cost_to_goal,
                -float(unreachable_cost),
            ).astype(np.float32)[None, ...]

            policy = _build_policy_from_cost(
                occ_map=occ_map,
                goal_xy=(int(goal_x), int(goal_y)),
                cost_to_goal=cost_to_goal,
                mechanism=mechanism,
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.clearance_weight),
                clearance_safe_distance=float(args.clearance_safe_distance),
                clearance_power=float(args.clearance_power),
                policy_tie_break=str(args.policy_tie_break),
                rng=rng,
            )

            old_policy = old_policies[local_idx]
            new_policies[local_idx] = policy
            new_dists[local_idx] = cost_to_goal_neg.astype(old_dists.dtype)
            changed = not np.array_equal(old_policy, policy)
            if changed:
                stats.changed_policy_samples += 1

            old_trace = _trace_path_from_policy(old_policy, goal_map, old_dists[local_idx])
            new_trace = _trace_path_from_policy(policy, goal_map, cost_to_goal_neg)
            old_geo_len = _path_length(old_trace, diagonal_cost=float(args.diagonal_cost))
            new_geo_len = _path_length(new_trace, diagonal_cost=float(args.diagonal_cost))
            stats.total_samples += 1
            stats.mean_old_geo_len += old_geo_len
            stats.mean_new_geo_len += new_geo_len

            summary_rows.append(
                {
                    "split": split_name,
                    "local_idx": int(local_idx),
                    "goal_x": int(goal_x),
                    "goal_y": int(goal_y),
                    "old_geo_len_probe": float(old_geo_len),
                    "new_geo_len_probe": float(new_geo_len),
                    "policy_changed": int(changed),
                }
            )

            if (local_idx + 1) % 100 == 0 or (local_idx + 1) == rebuild_count:
                print(
                    f"split={split_name} rebuilt={local_idx + 1}/{rebuild_count} "
                    f"changed={stats.changed_policy_samples}"
                )

        arrays[f"arr_{policy_idx}"] = new_policies
        arrays[f"arr_{dist_idx}"] = new_dists

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, *(arrays[f"arr_{i}"] for i in range(12)))

    if args.summary_csv is not None:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "local_idx",
                    "goal_x",
                    "goal_y",
                    "old_geo_len_probe",
                    "new_geo_len_probe",
                    "policy_changed",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    denom = max(1, stats.total_samples)
    print(f"saved_output_npz={args.output_npz}")
    if args.summary_csv is not None:
        print(f"saved_summary_csv={args.summary_csv}")
    print(f"mechanism={mechanism}")
    print(f"total_samples={stats.total_samples}")
    print(f"changed_policy_samples={stats.changed_policy_samples}")
    print(f"mean_old_geo_len_probe={stats.mean_old_geo_len / denom:.4f}")
    print(f"mean_new_geo_len_probe={stats.mean_new_geo_len / denom:.4f}")


if __name__ == "__main__":
    main()
