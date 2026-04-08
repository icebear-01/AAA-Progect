"""Visualize how expert-path objectives change under modified planning rules.

This script compares:
1. stored NPZ expert path traced from opt_policy
2. geometric-distance 8-connected A*
3. geometric-distance + clearance-biased 8-connected A*

It picks the cases with the largest path-shape change and saves:
- a summary PNG
- a CSV with per-case statistics
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn


ACTION_TO_MOVE = [
    (0, -1, 0),   # up
    (0, 0, +1),   # right
    (0, 0, -1),   # left
    (0, +1, 0),   # down
    (0, -1, +1),  # up-right
    (0, -1, -1),  # up-left
    (0, +1, +1),  # down-right
    (0, +1, -1),  # down-left
]

XY = Tuple[int, int]
OYX = Tuple[int, int, int]


@dataclass
class CaseCompare:
    idx: int
    start_xy: XY
    goal_xy: XY
    expert_path: List[XY]
    distance_path: List[XY]
    clearance_path: List[XY]
    distance_expanded: int
    clearance_expanded: int
    expert_geo_len: float
    distance_geo_len: float
    clearance_geo_len: float
    expert_unit_len: float
    distance_unit_len: float
    clearance_unit_len: float
    distance_change_pixels: int
    clearance_change_pixels: int


def _split_base(split: str) -> int:
    return {"train": 0, "valid": 4, "test": 8}[split]


def _sample_start(
    opt_dist: np.ndarray,
    goal_map: np.ndarray,
    free_mask: np.ndarray,
    rng: np.random.Generator,
    pcts: np.ndarray,
) -> OYX:
    od_vct = opt_dist.flatten()
    min_v = float(od_vct.min())
    od_vals = od_vct[od_vct > min_v]

    free_3d = np.repeat(free_mask[None, ...], opt_dist.shape[0], axis=0)
    goal_3d = goal_map > 0.5
    valid_mask = free_3d & (~goal_3d)
    valid_idx = np.where(valid_mask.flatten())[0]
    if valid_idx.size == 0:
        raise RuntimeError("No valid start cells available.")

    if od_vals.size > 0:
        od_th = np.percentile(od_vals, 100.0 * (1.0 - pcts))
        r = int(rng.integers(0, len(od_th) - 1))
        candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
        candidate_idx = np.where(candidate & valid_mask.flatten())[0]
        idx = int(rng.choice(candidate_idx if candidate_idx.size > 0 else valid_idx))
    else:
        idx = int(rng.choice(valid_idx))

    o, y, x = np.unravel_index(idx, opt_dist.shape)
    return int(o), int(y), int(x)


def _trace_optimal_path(
    opt_policy: np.ndarray,  # [A,O,H,W]
    goal_map: np.ndarray,    # [O,H,W]
    start_oyx: OYX,
) -> List[OYX]:
    policy = np.transpose(opt_policy, (1, 2, 3, 0))  # [O,H,W,A]
    goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
    if len(goal_loc) != 3:
        raise ValueError(f"goal_map must contain one goal in [O,H,W], got goal_loc={goal_loc}")

    cur = (int(start_oyx[0]), int(start_oyx[1]), int(start_oyx[2]))
    path_oyx: List[OYX] = [cur]

    max_steps = int(goal_map.shape[1] * goal_map.shape[2] * 6)
    for _ in range(max_steps):
        if cur == goal_loc:
            break
        action = int(np.argmax(policy[cur]))
        d_o, d_y, d_x = ACTION_TO_MOVE[action]
        nxt = (cur[0] + d_o, cur[1] + d_y, cur[2] + d_x)
        if (
            nxt[0] < 0
            or nxt[0] >= goal_map.shape[0]
            or nxt[1] < 0
            or nxt[1] >= goal_map.shape[1]
            or nxt[2] < 0
            or nxt[2] >= goal_map.shape[2]
        ):
            raise RuntimeError("Optimal policy stepped out of bounds.")
        if nxt in path_oyx:
            raise RuntimeError("Loop detected while tracing optimal policy.")
        cur = nxt
        path_oyx.append(cur)
    else:
        raise RuntimeError("Exceeded max steps while tracing optimal policy.")
    return path_oyx


def _path_oyx_to_xy(path_oyx: Sequence[OYX]) -> List[XY]:
    return [(int(x), int(y)) for _, y, x in path_oyx]


def _path_to_mask(path_xy: Sequence[XY], shape_hw: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    for x, y in path_xy:
        if 0 <= x < shape_hw[1] and 0 <= y < shape_hw[0]:
            mask[y, x] = True
    return mask


def _path_change_pixels(path_a: Sequence[XY], path_b: Sequence[XY], shape_hw: Tuple[int, int]) -> int:
    mask_a = _path_to_mask(path_a, shape_hw)
    mask_b = _path_to_mask(path_b, shape_hw)
    return int(np.logical_xor(mask_a, mask_b).sum())


def _score_case(distance_change: int, clearance_change: int, distance_len: float, clearance_len: float) -> float:
    return float(0.4 * distance_change + 0.6 * clearance_change + 0.1 * abs(clearance_len - distance_len))


def _plot_panel(
    ax: plt.Axes,
    occ_map: np.ndarray,
    path_xy: Sequence[XY],
    start_xy: XY,
    goal_xy: XY,
    title: str,
    color: str,
) -> None:
    ax.imshow(1.0 - occ_map, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if len(path_xy) > 1:
        xs = [p[0] for p in path_xy]
        ys = [p[1] for p in path_xy]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.95)
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=28, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=34, marker="x")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def _save_summary_png(cases: Sequence[CaseCompare], occ_maps: Sequence[np.ndarray], out_png: Path) -> None:
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 3, figsize=(12.6, 4.0 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    for row, (case, occ_map) in enumerate(zip(cases, occ_maps)):
        titles = [
            (
                "Stored Expert\n"
                f"unit={case.expert_unit_len:.1f} geo={case.expert_geo_len:.1f}"
            ),
            (
                "Geometric Distance\n"
                f"expanded={case.distance_expanded} geo={case.distance_geo_len:.1f}\n"
                f"delta_px={case.distance_change_pixels}"
            ),
            (
                "Distance + Clearance\n"
                f"expanded={case.clearance_expanded} geo={case.clearance_geo_len:.1f}\n"
                f"delta_px={case.clearance_change_pixels}"
            ),
        ]
        paths = [case.expert_path, case.distance_path, case.clearance_path]
        colors = ["deepskyblue", "gold", "chartreuse"]

        for col in range(3):
            _plot_panel(
                axes[row, col],
                occ_map=occ_map,
                path_xy=paths[col],
                start_xy=case.start_xy,
                goal_xy=case.goal_xy,
                title=titles[col],
                color=colors[col],
            )
        axes[row, 0].text(
            -0.06,
            0.5,
            f"case={case.idx}",
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=10,
        )

    fig.suptitle("Expert Objective Change: stored NPZ vs modified planners", fontsize=14)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize expert-path changes under modified planning rules.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--distance-diagonal-cost", type=float, default=math.sqrt(2.0))
    p.add_argument("--clearance-weight", type=float, default=0.10)
    p.add_argument("--clearance-safe-distance", type=float, default=4.0)
    p.add_argument("--clearance-power", type=float, default=2.0)
    p.add_argument(
        "--clearance-integration-mode",
        type=str,
        default="g_cost",
        choices=["g_cost", "heuristic_bias", "priority_tie_break"],
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/expert_objective_change"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    rng = np.random.default_rng(args.seed)
    base = _split_base(args.split)

    with np.load(args.dataset) as data:
        map_designs = np.asarray(data[f"arr_{base}"], dtype=np.float32)
        goal_maps = np.asarray(data[f"arr_{base+1}"], dtype=np.float32)
        opt_policies = np.asarray(data[f"arr_{base+2}"], dtype=np.float32)
        opt_dists = np.asarray(data[f"arr_{base+3}"], dtype=np.float32)

    eval_count = min(int(args.max_samples), int(map_designs.shape[0]))
    all_cases: List[Tuple[float, CaseCompare, np.ndarray]] = []

    for idx in range(eval_count):
        map_design = map_designs[idx]
        free_mask = map_design > 0.5
        occ_map = (1.0 - map_design).astype(np.float32)
        goal_map = goal_maps[idx]
        opt_policy = opt_policies[idx]
        opt_dist = opt_dists[idx]

        try:
            start_oyx = _sample_start(
                opt_dist=opt_dist,
                goal_map=goal_map,
                free_mask=free_mask,
                rng=rng,
                pcts=np.array([0.55, 0.70, 0.85, 1.0], dtype=np.float32),
            )
            expert_path_oyx = _trace_optimal_path(opt_policy, goal_map, start_oyx)
        except RuntimeError:
            continue

        goal_oyx = tuple(np.array(np.nonzero(goal_map)).squeeze())
        start_xy = (int(start_oyx[2]), int(start_oyx[1]))
        goal_xy = (int(goal_oyx[2]), int(goal_oyx[1]))
        expert_path = _path_oyx_to_xy(expert_path_oyx)

        distance_stats = astar_8conn_stats(
            occ_map=occ_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            diagonal_cost=float(args.distance_diagonal_cost),
            heuristic_mode="octile",
            allow_corner_cut=True,
        )
        clearance_stats = astar_8conn_stats(
            occ_map=occ_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            diagonal_cost=float(args.distance_diagonal_cost),
            heuristic_mode="octile",
            allow_corner_cut=True,
            clearance_weight=float(args.clearance_weight),
            clearance_safe_distance=float(args.clearance_safe_distance),
            clearance_power=float(args.clearance_power),
            clearance_integration_mode=str(args.clearance_integration_mode),
        )
        if not distance_stats.success or distance_stats.path is None:
            continue
        if not clearance_stats.success or clearance_stats.path is None:
            continue

        distance_path = list(distance_stats.path)
        clearance_path = list(clearance_stats.path)
        shape_hw = occ_map.shape

        distance_change = _path_change_pixels(expert_path, distance_path, shape_hw)
        clearance_change = _path_change_pixels(expert_path, clearance_path, shape_hw)
        expert_unit_len = float(path_length_8conn(expert_path, diagonal_cost=1.0))
        distance_unit_len = float(path_length_8conn(distance_path, diagonal_cost=1.0))
        clearance_unit_len = float(path_length_8conn(clearance_path, diagonal_cost=1.0))
        expert_geo_len = float(path_length_8conn(expert_path, diagonal_cost=float(args.distance_diagonal_cost)))
        distance_geo_len = float(path_length_8conn(distance_path, diagonal_cost=float(args.distance_diagonal_cost)))
        clearance_geo_len = float(path_length_8conn(clearance_path, diagonal_cost=float(args.distance_diagonal_cost)))

        case = CaseCompare(
            idx=int(idx),
            start_xy=start_xy,
            goal_xy=goal_xy,
            expert_path=expert_path,
            distance_path=distance_path,
            clearance_path=clearance_path,
            distance_expanded=int(distance_stats.expanded_nodes),
            clearance_expanded=int(clearance_stats.expanded_nodes),
            expert_geo_len=expert_geo_len,
            distance_geo_len=distance_geo_len,
            clearance_geo_len=clearance_geo_len,
            expert_unit_len=expert_unit_len,
            distance_unit_len=distance_unit_len,
            clearance_unit_len=clearance_unit_len,
            distance_change_pixels=int(distance_change),
            clearance_change_pixels=int(clearance_change),
        )
        score = _score_case(distance_change, clearance_change, distance_geo_len, clearance_geo_len)
        all_cases.append((score, case, occ_map))

    if not all_cases:
        raise RuntimeError("No successful comparison cases found.")

    all_cases.sort(key=lambda item: item[0], reverse=True)
    selected = all_cases[: max(1, int(args.top_k))]
    cases = [item[1] for item in selected]
    occ_maps = [item[2] for item in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"expert_objective_change_{args.split}.png"
    csv_path = args.output_dir / f"expert_objective_change_{args.split}.csv"

    _save_summary_png(cases, occ_maps, png_path)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "case_idx,start_x,start_y,goal_x,goal_y,"
            "expert_unit_len,expert_geo_len,distance_unit_len,distance_geo_len,"
            "clearance_unit_len,clearance_geo_len,distance_expanded,clearance_expanded,"
            "distance_change_pixels,clearance_change_pixels\n"
        )
        for case in cases:
            f.write(
                f"{case.idx},{case.start_xy[0]},{case.start_xy[1]},{case.goal_xy[0]},{case.goal_xy[1]},"
                f"{case.expert_unit_len:.4f},{case.expert_geo_len:.4f},"
                f"{case.distance_unit_len:.4f},{case.distance_geo_len:.4f},"
                f"{case.clearance_unit_len:.4f},{case.clearance_geo_len:.4f},"
                f"{case.distance_expanded},{case.clearance_expanded},"
                f"{case.distance_change_pixels},{case.clearance_change_pixels}\n"
            )

    print(f"saved_png={png_path}")
    print(f"saved_csv={csv_path}")
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
