from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn
from neural_astar.utils.guidance_targets import build_clearance_penalty_map
from rebuild_planning_npz_experts import _build_policy_from_cost, _reverse_dijkstra_cost_to_goal


XY = Tuple[int, int]
ACTION_TO_MOVE = [
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
    (0, +1, 0),
    (0, -1, +1),
    (0, -1, -1),
    (0, +1, +1),
    (0, +1, -1),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit and spot-check corrected expert paths.")
    p.add_argument("--data-npz", type=Path, required=True)
    p.add_argument("--old-expert-npz", type=Path, required=True)
    p.add_argument("--case-csv", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--sample-seed", type=int, default=20260330)
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    p.add_argument("--clearance-weight", type=float, default=0.35)
    p.add_argument("--clearance-safe-distance", type=float, default=5.0)
    p.add_argument("--clearance-power", type=float, default=2.0)
    p.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    p.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    p.add_argument("--policy-tie-break", type=str, default="clearance_then_first")
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(allow_corner_cut=True)
    return p.parse_args()


def _load_split_arrays(npz_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_base = {"train": 0, "valid": 4, "test": 8}[split]
    with np.load(npz_path) as data:
        maps = np.asarray(data[f"arr_{idx_base}"], dtype=np.float32)
        goals = np.asarray(data[f"arr_{idx_base+1}"], dtype=np.float32)
        policies = np.asarray(data[f"arr_{idx_base+2}"], dtype=np.float32)
    return maps, goals, policies


def _load_case_rows(path: Path) -> Dict[int, Tuple[int, int, int, int]]:
    out: Dict[int, Tuple[int, int, int, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[int(row["idx"])] = (
                int(row["start_x"]),
                int(row["start_y"]),
                int(row["goal_x"]),
                int(row["goal_y"]),
            )
    return out


def _trace_policy(policy: np.ndarray, sx: int, sy: int, gx: int, gy: int) -> Tuple[List[XY], bool, str]:
    pol = np.transpose(policy, (1, 2, 3, 0))
    cur = (0, int(sy), int(sx))
    goal = (0, int(gy), int(gx))
    seen = {cur}
    path: List[XY] = [(int(sx), int(sy))]
    max_steps = int(policy.shape[-1] * policy.shape[-2] * 4)
    for _ in range(max_steps):
        if cur == goal:
            return path, True, "goal"
        a = int(np.argmax(pol[cur]))
        mv = ACTION_TO_MOVE[a]
        nxt = (cur[0] + mv[0], cur[1] + mv[1], cur[2] + mv[2])
        if nxt[1] < 0 or nxt[1] >= policy.shape[-2] or nxt[2] < 0 or nxt[2] >= policy.shape[-1]:
            return path, False, "oob"
        if nxt in seen:
            return path, False, "loop"
        seen.add(nxt)
        cur = nxt
        path.append((int(cur[2]), int(cur[1])))
    return path, False, "max_steps"


def _objective_cost(path_xy: Sequence[XY], clearance_penalty: np.ndarray, diagonal_cost: float, clearance_weight: float) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path_xy[:-1], path_xy[1:]):
        step = float(diagonal_cost) if (abs(x1 - x0) == 1 and abs(y1 - y0) == 1) else 1.0
        total += step + float(clearance_weight) * float(clearance_penalty[y1, x1])
    return float(total)


def _plot_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    path_xy: Sequence[XY],
    title: str,
    color: str,
    start_xy: XY,
    goal_xy: XY,
) -> None:
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if len(path_xy) > 1:
        xs = [p[0] for p in path_xy]
        ys = [p[1] for p in path_xy]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.98)
    ax.scatter([start_xy[0]], [start_xy[1]], c="#22c55e", s=30, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#ef4444", s=34, marker="x")
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_axis_off()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    maps, _, _ = _load_split_arrays(args.data_npz, args.split)
    _, _, old_policies = _load_split_arrays(args.old_expert_npz, args.split)
    rows = _load_case_rows(args.case_csv)
    rng = np.random.default_rng(int(args.sample_seed))

    audit_rows: List[List[object]] = []
    if not bool(args.skip_audit):
        match_count = 0
        old_invalid_count = 0
        old_worse_count = 0
        max_gap = (-np.inf, -1)

        for idx in sorted(rows.keys()):
            sx, sy, gx, gy = rows[idx]
            occ = (1.0 - maps[idx]).astype(np.float32)
            clr = build_clearance_penalty_map(
                occ_map=occ,
                safe_distance=float(args.clearance_safe_distance),
                power=float(args.clearance_power),
            ).astype(np.float32)
            clr[occ > 0.5] = 0.0

            old_path, old_valid, old_reason = _trace_policy(old_policies[idx], sx, sy, gx, gy)
            old_geo = float(path_length_8conn(old_path, diagonal_cost=float(args.diagonal_cost))) if len(old_path) > 1 else 0.0
            old_obj = _objective_cost(old_path, clr, float(args.diagonal_cost), float(args.clearance_weight)) if len(old_path) > 1 else float("inf")
            if not old_valid:
                old_invalid_count += 1

            cost_to_goal = _reverse_dijkstra_cost_to_goal(
                occ_map=occ,
                goal_xy=(gx, gy),
                mechanism="moore",
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.clearance_weight),
                clearance_safe_distance=float(args.clearance_safe_distance),
                clearance_power=float(args.clearance_power),
            )
            corrected_policy = _build_policy_from_cost(
                occ_map=occ,
                goal_xy=(gx, gy),
                cost_to_goal=cost_to_goal,
                mechanism="moore",
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.clearance_weight),
                clearance_safe_distance=float(args.clearance_safe_distance),
                clearance_power=float(args.clearance_power),
                policy_tie_break=str(args.policy_tie_break),
                rng=np.random.default_rng(int(args.sample_seed) + int(idx)),
            )
            corrected_path, corrected_valid, corrected_reason = _trace_policy(corrected_policy, sx, sy, gx, gy)
            corrected_geo = float(path_length_8conn(corrected_path, diagonal_cost=float(args.diagonal_cost)))
            corrected_obj = _objective_cost(corrected_path, clr, float(args.diagonal_cost), float(args.clearance_weight))

            stats = astar_8conn_stats(
                occ_map=occ,
                start_xy=(sx, sy),
                goal_xy=(gx, gy),
                heuristic_mode="octile",
                heuristic_weight=1.0,
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.clearance_weight),
                clearance_safe_distance=float(args.clearance_safe_distance),
                clearance_power=float(args.clearance_power),
                clearance_integration_mode="g_cost",
            )
            planner_geo = float(path_length_8conn(stats.path, diagonal_cost=float(args.diagonal_cost))) if stats.path else 0.0
            planner_obj = _objective_cost(stats.path, clr, float(args.diagonal_cost), float(args.clearance_weight)) if stats.path else float("inf")

            corrected_matches_planner = abs(corrected_obj - planner_obj) <= 1e-5
            if corrected_matches_planner:
                match_count += 1
            if old_obj > planner_obj + 1e-5:
                old_worse_count += 1
            gap = old_obj - corrected_obj
            if gap > max_gap[0]:
                max_gap = (gap, idx)

            audit_rows.append(
                [
                    idx,
                    sx,
                    sy,
                    gx,
                    gy,
                    int(old_valid),
                    old_reason,
                    old_geo,
                    old_obj,
                    int(corrected_valid),
                    corrected_reason,
                    corrected_geo,
                    corrected_obj,
                    planner_geo,
                    planner_obj,
                    int(corrected_matches_planner),
                    old_obj - corrected_obj,
                ]
            )

        with (args.output_dir / "audit_test400.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "idx",
                    "start_x",
                    "start_y",
                    "goal_x",
                    "goal_y",
                    "old_valid",
                    "old_reason",
                    "old_geo",
                    "old_obj",
                    "corrected_valid",
                    "corrected_reason",
                    "corrected_geo",
                    "corrected_obj",
                    "planner_geo",
                    "planner_obj",
                    "corrected_matches_planner",
                    "old_minus_corrected_obj",
                ]
            )
            writer.writerows(audit_rows)

        with (args.output_dir / "audit_summary.txt").open("w", encoding="utf-8") as f:
            f.write(f"total_cases={len(audit_rows)}\n")
            f.write(f"corrected_matches_planner={match_count}\n")
            f.write(f"old_invalid={old_invalid_count}\n")
            f.write(f"old_worse_than_planner={old_worse_count}\n")
            f.write(f"max_old_minus_corrected_gap={max_gap[0]}\n")
            f.write(f"max_gap_case_idx={max_gap[1]}\n")

    chosen = sorted(rng.choice(np.array(sorted(rows.keys()), dtype=np.int32), size=min(int(args.num_samples), len(rows)), replace=False).tolist())
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11.2, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = np.asarray([axes])

    for row_i, idx in enumerate(chosen):
        sx, sy, gx, gy = rows[idx]
        occ = (1.0 - maps[idx]).astype(np.float32)

        old_path, _, _ = _trace_policy(old_policies[idx], sx, sy, gx, gy)

        cost_to_goal = _reverse_dijkstra_cost_to_goal(
            occ_map=occ,
            goal_xy=(gx, gy),
            mechanism="moore",
            diagonal_cost=float(args.diagonal_cost),
            allow_corner_cut=bool(args.allow_corner_cut),
            clearance_weight=float(args.clearance_weight),
            clearance_safe_distance=float(args.clearance_safe_distance),
            clearance_power=float(args.clearance_power),
        )
        corrected_policy = _build_policy_from_cost(
            occ_map=occ,
            goal_xy=(gx, gy),
            cost_to_goal=cost_to_goal,
            mechanism="moore",
            diagonal_cost=float(args.diagonal_cost),
            allow_corner_cut=bool(args.allow_corner_cut),
            clearance_weight=float(args.clearance_weight),
            clearance_safe_distance=float(args.clearance_safe_distance),
            clearance_power=float(args.clearance_power),
            policy_tie_break=str(args.policy_tie_break),
            rng=np.random.default_rng(int(args.sample_seed) + int(idx)),
        )
        corrected_path, _, _ = _trace_policy(corrected_policy, sx, sy, gx, gy)
        planner = astar_8conn_stats(
            occ_map=occ,
            start_xy=(sx, sy),
            goal_xy=(gx, gy),
            heuristic_mode="octile",
            heuristic_weight=1.0,
            diagonal_cost=float(args.diagonal_cost),
            allow_corner_cut=bool(args.allow_corner_cut),
            clearance_weight=float(args.clearance_weight),
            clearance_safe_distance=float(args.clearance_safe_distance),
            clearance_power=float(args.clearance_power),
            clearance_integration_mode="g_cost",
        )
        clr = build_clearance_penalty_map(
            occ_map=occ,
            safe_distance=float(args.clearance_safe_distance),
            power=float(args.clearance_power),
        ).astype(np.float32)
        clr[occ > 0.5] = 0.0
        old_obj = _objective_cost(old_path, clr, float(args.diagonal_cost), float(args.clearance_weight))
        corrected_obj = _objective_cost(corrected_path, clr, float(args.diagonal_cost), float(args.clearance_weight))
        planner_obj = _objective_cost(planner.path, clr, float(args.diagonal_cost), float(args.clearance_weight))

        _plot_panel(axes[row_i, 0], occ, old_path, f"Old v1 Expert idx={idx}\nobj={old_obj:.2f}", "#94a3b8", (sx, sy), (gx, gy))
        _plot_panel(axes[row_i, 1], occ, corrected_path, f"Corrected Expert\nobj={corrected_obj:.2f}", "#22d3ee", (sx, sy), (gx, gy))
        _plot_panel(axes[row_i, 2], occ, planner.path, f"Clearance A*\nobj={planner_obj:.2f}", "#16a34a", (sx, sy), (gx, gy))

    fig.suptitle("Spot Check: Old Rebuilt Expert vs Corrected Expert vs Clearance-aware A*", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.output_dir / "spotcheck_random.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
