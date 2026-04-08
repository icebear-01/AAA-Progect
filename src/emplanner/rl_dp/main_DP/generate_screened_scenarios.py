#!/usr/bin/env python3
"""Generate an offline dataset of DP-screened s-l obstacle scenarios."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from rl_env import SLPathEnv
from sl_grid import build_grid, default_training_grid_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DP-screened obstacle scenarios for training or evaluation."
    )
    parser.add_argument("--count", type=int, default=1000, help="number of screened scenarios to keep")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("main_DP/scenario_sets/dp_screened_scenarios_1000.json"),
        help="output JSON file",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--min-obstacles", type=int, default=2, help="minimum obstacle count per scene")
    parser.add_argument("--max-obstacles", type=int, default=10, help="maximum obstacle count per scene")
    parser.add_argument(
        "--scenario-pool-size",
        type=int,
        default=32,
        help="candidate scene count sampled per reset before DP screening",
    )
    parser.add_argument(
        "--scenario-top-k",
        type=int,
        default=8,
        help="randomly choose one scene from the best K screened candidates",
    )
    parser.add_argument(
        "--scenario-max-avg-cost",
        type=float,
        default=None,
        help="optional hard ceiling on DP average transition cost",
    )
    parser.add_argument(
        "--scenario-max-attempts",
        type=int,
        default=128,
        help="max random attempts used to assemble each screened pool",
    )
    parser.add_argument(
        "--lateral-move-limit",
        type=int,
        default=3,
        help="maximum lateral index jump allowed by the DP feasibility check",
    )
    parser.add_argument(
        "--start-clear-fraction",
        type=float,
        default=0.2,
        help="fraction of the start region kept obstacle-free",
    )
    parser.add_argument(
        "--avoid-obstacle-overlap",
        action="store_true",
        help="use a coarse fast overlap rejection when sampling obstacles",
    )
    parser.add_argument(
        "--obstacle-overlap-clearance",
        type=float,
        default=0.05,
        help="extra clearance added to the coarse obstacle overlap rejection",
    )
    parser.add_argument(
        "--obstacle-sampling-attempts-per-obstacle",
        type=int,
        default=24,
        help="maximum retries used to place each obstacle before giving up",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="print progress every N kept scenarios",
    )
    return parser.parse_args()


def obstacle_to_dict(obstacle) -> Dict[str, object]:
    center_s, center_l = obstacle.center
    return {
        "center": [float(center_s), float(center_l)],
        "length": float(obstacle.length),
        "width": float(obstacle.width),
        "yaw": float(obstacle.yaw),
    }


def build_scenario_record(env: SLPathEnv, scenario_index: int) -> Dict[str, object]:
    dp_result = env.last_scenario_dp_result
    if dp_result is None or not dp_result.feasible:
        raise RuntimeError("Expected a feasible DP-screened scenario.")

    s_grid, l_grid = build_grid(env.grid_spec)
    s_coords = s_grid[:, 0]
    l_coords = l_grid[0, :]
    path_indices = [int(idx) for idx in dp_result.path_indices]
    path_points = [
        [float(s_coords[s_idx]), float(l_coords[l_idx])]
        for s_idx, l_idx in enumerate(path_indices)
    ]

    return {
        "scenario_index": int(scenario_index),
        "obstacle_count": len(env.obstacles),
        "start_l": float(env.start_l),
        "dp_total_cost": float(dp_result.total_cost),
        "dp_avg_cost": float(dp_result.avg_cost),
        "path_indices": path_indices,
        "path": path_points,
        "obstacles": [obstacle_to_dict(obstacle) for obstacle in env.obstacles],
    }


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.min_obstacles < 0 or args.max_obstacles < args.min_obstacles:
        raise ValueError("obstacle count range is invalid")

    spec = default_training_grid_spec()
    env = SLPathEnv(
        spec,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
        lateral_move_limit=args.lateral_move_limit,
        start_clear_fraction=args.start_clear_fraction,
        avoid_obstacle_overlap=args.avoid_obstacle_overlap,
        obstacle_overlap_clearance=args.obstacle_overlap_clearance,
        obstacle_sampling_attempts_per_obstacle=args.obstacle_sampling_attempts_per_obstacle,
        scenario_pool_size=args.scenario_pool_size,
        scenario_top_k=args.scenario_top_k,
        scenario_min_obstacles=args.min_obstacles,
        scenario_max_avg_cost=args.scenario_max_avg_cost,
        scenario_max_attempts=args.scenario_max_attempts,
        seed=args.seed,
    )

    scenarios: List[Dict[str, object]] = []
    failed_resets = 0
    max_failed_resets = max(args.count, 100)

    while len(scenarios) < args.count:
        env.reset()
        dp_result = env.last_scenario_dp_result
        if dp_result is None or not dp_result.feasible:
            failed_resets += 1
            if failed_resets > max_failed_resets:
                raise RuntimeError(
                    f"Unable to collect {args.count} feasible scenarios; "
                    f"failed_resets={failed_resets}"
                )
            continue
        failed_resets = 0
        scenarios.append(build_scenario_record(env, len(scenarios)))
        if args.progress_interval > 0 and len(scenarios) % args.progress_interval == 0:
            print(f"Collected {len(scenarios)}/{args.count} scenarios")

    dp_avg_costs = np.asarray([entry["dp_avg_cost"] for entry in scenarios], dtype=np.float64)
    obstacle_counts = np.asarray([entry["obstacle_count"] for entry in scenarios], dtype=np.float64)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "grid": {
            "s_range": list(spec.s_range),
            "l_range": list(spec.l_range),
            "s_samples": int(spec.s_samples),
            "l_samples": int(spec.l_samples),
        },
        "generation": {
            "seed": int(args.seed),
            "count": int(args.count),
            "min_obstacles": int(args.min_obstacles),
            "max_obstacles": int(args.max_obstacles),
            "scenario_pool_size": int(args.scenario_pool_size),
            "scenario_top_k": int(args.scenario_top_k),
            "scenario_max_avg_cost": args.scenario_max_avg_cost,
            "scenario_max_attempts": int(args.scenario_max_attempts),
            "lateral_move_limit": int(args.lateral_move_limit),
            "start_clear_fraction": float(args.start_clear_fraction),
            "avoid_obstacle_overlap": bool(args.avoid_obstacle_overlap),
            "obstacle_overlap_clearance": float(args.obstacle_overlap_clearance),
            "obstacle_sampling_attempts_per_obstacle": int(
                args.obstacle_sampling_attempts_per_obstacle
            ),
        },
        "summary": {
            "dp_avg_cost_mean": float(dp_avg_costs.mean()),
            "dp_avg_cost_min": float(dp_avg_costs.min()),
            "dp_avg_cost_max": float(dp_avg_costs.max()),
            "obstacle_count_mean": float(obstacle_counts.mean()),
            "obstacle_count_min": int(obstacle_counts.min()),
            "obstacle_count_max": int(obstacle_counts.max()),
        },
        "scenarios": scenarios,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(scenarios)} screened scenarios to {args.output}")
    print(
        "Summary | "
        f"avg_cost_mean={payload['summary']['dp_avg_cost_mean']:.3f} "
        f"avg_cost_min={payload['summary']['dp_avg_cost_min']:.3f} "
        f"avg_cost_max={payload['summary']['dp_avg_cost_max']:.3f} "
        f"obstacles_mean={payload['summary']['obstacle_count_mean']:.2f}"
    )


if __name__ == "__main__":
    main()
