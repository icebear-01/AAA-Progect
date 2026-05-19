#!/usr/bin/env python3
"""Benchmark the original EMPlanner-style DP logic from emplanner.cpp."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from sl_grid import DEFAULT_L_RANGE, DEFAULT_S_RANGE
from sl_obstacles import Obstacle, generate_random_obstacles


LINE_COLORS = [
    "#1f4e79",
    "#d97904",
    "#2e8b57",
    "#b24745",
    "#6b5b95",
]


@dataclass(frozen=True)
class ClassicObstacle:
    corners: np.ndarray
    min_s: float
    max_s: float
    min_l: float
    max_l: float


def _pick_font_family(candidates: Sequence[str]) -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def apply_paper_style() -> None:
    serif = _pick_font_family(["Times New Roman", "Liberation Serif", "DejaVu Serif"])
    sans = _pick_font_family(["Arial", "Liberation Sans", "DejaVu Sans"])
    families = [name for name in [serif, sans] if name]
    plt.rcParams.update(
        {
            "font.family": families or ["DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "axes.facecolor": "#fcfcfc",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#c9cfd6",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the original EMPlanner-style DP logic."
    )
    parser.add_argument("--col-node-num", type=int, default=9, help="number of DP columns")
    parser.add_argument("--row-node-min", type=int, default=9, help="minimum DP row count")
    parser.add_argument("--row-node-max", type=int, default=30, help="maximum DP row count")
    parser.add_argument(
        "--obstacle-counts",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
        help="obstacle counts to benchmark",
    )
    parser.add_argument("--trials", type=int, default=20, help="trial count per configuration")
    parser.add_argument("--seed", type=int, default=20260408, help="random seed")
    parser.add_argument("--sample-s", type=float, default=0.8, help="longitudinal step")
    parser.add_argument("--sample-l", type=float, default=0.35, help="lateral step")
    parser.add_argument(
        "--sample-s-num",
        type=float,
        default=10.0,
        help="sample points per meter along each DP edge",
    )
    parser.add_argument("--car-width", type=float, default=0.75, help="car width from emplanner")
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=100.0,
        help="runtime threshold used for visualization",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("temp/emplanner_classic_dp_benchmark_20260408"),
        help="output directory",
    )
    return parser.parse_args()


def convert_obstacles(raw_obstacles: Sequence[Obstacle]) -> List[ClassicObstacle]:
    converted: List[ClassicObstacle] = []
    for obstacle in raw_obstacles:
        corners = np.asarray(obstacle.corners(), dtype=np.float64)
        converted.append(
            ClassicObstacle(
                corners=corners,
                min_s=float(corners[:, 0].min()),
                max_s=float(corners[:, 0].max()),
                min_l=float(corners[:, 1].min()),
                max_l=float(corners[:, 1].max()),
            )
        )
    return converted


def calculate_three_degree_polynomial_coefficients(
    start_l: float,
    start_dl: float,
    end_l: float,
    end_dl: float,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    start_s_sq = start_s * start_s
    end_s_sq = end_s * end_s
    matrix = np.array(
        [
            [1.0, start_s, start_s_sq, start_s_sq * start_s],
            [0.0, 1.0, 2.0 * start_s, 3.0 * start_s_sq],
            [1.0, end_s, end_s_sq, end_s_sq * end_s],
            [0.0, 1.0, 2.0 * end_s, 3.0 * end_s_sq],
        ],
        dtype=np.float64,
    )
    rhs = np.array([start_l, start_dl, end_l, end_dl], dtype=np.float64)
    return np.linalg.solve(matrix, rhs)


def point_inside_quadrilateral(corners: np.ndarray, point: np.ndarray) -> bool:
    prev_sign = 0
    for idx in range(4):
        a = corners[idx]
        b = corners[(idx + 1) % 4]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) <= 1e-9:
            continue
        sign = 1 if cross > 0 else -1
        if prev_sign == 0:
            prev_sign = sign
        elif sign != prev_sign:
            return False
    return True


def point_to_segment_dist_sq(point: np.ndarray, seg0: np.ndarray, seg1: np.ndarray) -> float:
    segment = seg1 - seg0
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1e-12:
        diff = point - seg0
        return float(np.dot(diff, diff))
    t = float(np.dot(point - seg0, segment) / seg_len_sq)
    t = max(0.0, min(1.0, t))
    proj = seg0 + t * segment
    diff = point - proj
    return float(np.dot(diff, diff))


def calc_obstacle_cost(
    obstacle: ClassicObstacle,
    aim_s: float,
    aim_l: float,
    host_start_s: float,
    dp_min_collision_distance_sq: float,
) -> tuple[float, bool]:
    abs_s = host_start_s + aim_s
    if obstacle.min_s < abs_s < obstacle.max_s and obstacle.min_l < aim_l < obstacle.max_l:
        return 1e8, True

    point = np.array([aim_s, aim_l], dtype=np.float64)
    corners = obstacle.corners.copy()
    corners[:, 0] -= host_start_s

    if point_inside_quadrilateral(corners, point):
        return 1e8, True

    min_distance_sq = point_to_segment_dist_sq(point, corners[0], corners[1])
    min_distance_sq = min(min_distance_sq, point_to_segment_dist_sq(point, corners[1], corners[2]))
    min_distance_sq = min(min_distance_sq, point_to_segment_dist_sq(point, corners[2], corners[3]))
    min_distance_sq = min(min_distance_sq, point_to_segment_dist_sq(point, corners[3], corners[0]))
    if min_distance_sq >= dp_min_collision_distance_sq:
        return 0.0, False
    return 1e8, True


def calculate_path_cost(
    *,
    pre_node_s: float,
    pre_node_l: float,
    current_node_s: float,
    current_node_l: float,
    host_start_s: float,
    sample_s: float,
    sample_s_num: float,
    w_cost_smooth_dl: float,
    w_cost_smooth_ddl: float,
    w_cost_smooth_dddl: float,
    w_cost_smooth_total: float,
    w_cost_ref: float,
    w_cost_collision: float,
    obstacles: Sequence[ClassicObstacle],
    col_node_num: int,
    row_node_num: int,
    sample_l: float,
    dp_min_collision_distance_sq: float,
) -> float:
    coeff = calculate_three_degree_polynomial_coefficients(
        pre_node_l,
        0.0,
        current_node_l,
        0.0,
        pre_node_s,
        current_node_s,
    )
    a0, a1, a2, a3 = coeff.tolist()
    points_num = max(2, int(math.floor(sample_s * sample_s_num)))
    ds = np.linspace(pre_node_s, pre_node_s + sample_s, points_num, dtype=np.float64)
    ds_pow2 = ds * ds
    ds_pow3 = ds_pow2 * ds
    l = a0 + a1 * ds + a2 * ds_pow2 + a3 * ds_pow3
    dl = a1 + 2.0 * a2 * ds + 3.0 * a3 * ds_pow2
    ddl = 2.0 * a2 + 6.0 * a3 * ds
    dddl = np.full_like(ds, 6.0 * a3)

    cost_smooth = w_cost_smooth_total * (
        w_cost_smooth_dl * float(np.dot(dl, dl))
        + w_cost_smooth_ddl * float(np.dot(ddl, ddl))
        + w_cost_smooth_dddl * float(np.dot(dddl, dddl))
    ) / ds.size
    if float(np.max(np.abs(dl))) > 1.4:
        cost_smooth += 1e6

    cost_ref = w_cost_ref * float(np.dot(l, l))
    cost_collision = 0.0
    lateral_limit = sample_l * row_node_num
    forward_limit = host_start_s + sample_s * col_node_num
    for obstacle in obstacles:
        if obstacle.max_s < host_start_s or obstacle.min_s > forward_limit:
            continue
        if obstacle.max_l > lateral_limit or obstacle.min_l < -lateral_limit:
            continue
        collision = False
        for sample_s_value, sample_l_value in zip(ds, l):
            obstacle_cost, collision = calc_obstacle_cost(
                obstacle,
                float(sample_s_value),
                float(sample_l_value),
                host_start_s,
                dp_min_collision_distance_sq,
            )
            cost_collision += obstacle_cost
            if collision:
                break
        if collision:
            break
    return cost_smooth + cost_ref + cost_collision * w_cost_collision


def run_classic_dp(
    *,
    col_node_num: int,
    row_node_num: int,
    sample_s: float,
    sample_l: float,
    sample_s_num: float,
    obstacles: Sequence[ClassicObstacle],
    car_width: float,
) -> tuple[float, bool]:
    costs = np.full((col_node_num, row_node_num), np.inf, dtype=np.float64)
    parents = np.full((col_node_num, row_node_num), -1, dtype=np.int32)
    center = (row_node_num - 1) / 2.0
    host_start_s = 0.0

    for row in range(row_node_num):
        node_s = sample_s
        node_l = (center - row) * sample_l
        costs[0, row] = calculate_path_cost(
            pre_node_s=0.0,
            pre_node_l=0.0,
            current_node_s=node_s,
            current_node_l=node_l,
            host_start_s=host_start_s,
            sample_s=sample_s,
            sample_s_num=sample_s_num,
            w_cost_smooth_dl=2.0,
            w_cost_smooth_ddl=1.0,
            w_cost_smooth_dddl=2.0,
            w_cost_smooth_total=20.0,
            w_cost_ref=2000.0,
            w_cost_collision=1.0,
            obstacles=obstacles,
            col_node_num=col_node_num,
            row_node_num=row_node_num,
            sample_l=sample_l,
            dp_min_collision_distance_sq=(car_width / 2.0) ** 2,
        )
        parents[0, row] = 0

    for col in range(1, col_node_num):
        node_s = (col + 1) * sample_s
        for row in range(row_node_num):
            node_l = (center - row) * sample_l
            best_cost = math.inf
            best_parent = -1
            for prev_row in range(row_node_num):
                if abs((prev_row - row) * sample_l / sample_s) > 1.2:
                    continue
                prev_cost = float(costs[col - 1, prev_row])
                if not math.isfinite(prev_cost):
                    continue
                prev_s = col * sample_s
                prev_l = (center - prev_row) * sample_l
                edge_cost = calculate_path_cost(
                    pre_node_s=prev_s,
                    pre_node_l=prev_l,
                    current_node_s=node_s,
                    current_node_l=node_l,
                    host_start_s=host_start_s,
                    sample_s=sample_s,
                    sample_s_num=sample_s_num,
                    w_cost_smooth_dl=2.0,
                    w_cost_smooth_ddl=1.0,
                    w_cost_smooth_dddl=2.0,
                    w_cost_smooth_total=20.0,
                    w_cost_ref=2000.0,
                    w_cost_collision=1.0,
                    obstacles=obstacles,
                    col_node_num=col_node_num,
                    row_node_num=row_node_num,
                    sample_l=sample_l,
                    dp_min_collision_distance_sq=(car_width / 2.0) ** 2,
                )
                total_cost = prev_cost + edge_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_parent = prev_row
            costs[col, row] = best_cost
            parents[col, row] = best_parent

    last_row = costs[col_node_num - 1]
    best_last_row = int(np.argmin(last_row))
    best_cost = float(last_row[best_last_row])
    if not math.isfinite(best_cost) or best_cost >= 1e8:
        return best_cost, False
    return best_cost, True


def summarize_ms(values_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(arr.max()),
    }


def save_csv(records: List[Dict[str, object]], csv_path: Path) -> None:
    header = [
        "col_node_num",
        "row_node_num",
        "obstacle_count",
        "trials",
        "feasible_rate",
        "mean_ms",
        "std_ms",
        "min_ms",
        "p50_ms",
        "p90_ms",
        "max_ms",
    ]
    lines = [",".join(header)]
    for record in records:
        lines.append(",".join(str(record[key]) for key in header))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    *,
    records: List[Dict[str, object]],
    row_values: List[int],
    obstacle_counts: List[int],
    threshold_ms: float,
    output_path: Path,
) -> None:
    apply_paper_style()
    heatmap = np.full((len(obstacle_counts), len(row_values)), np.nan, dtype=np.float64)
    for row_idx, obstacle_count in enumerate(obstacle_counts):
        for col_idx, row_node_num in enumerate(row_values):
            match = next(
                (
                    record
                    for record in records
                    if record["obstacle_count"] == obstacle_count
                    and record["row_node_num"] == row_node_num
                ),
                None,
            )
            if match is not None:
                heatmap[row_idx, col_idx] = float(match["mean_ms"])

    clipped_heatmap = np.minimum(heatmap, threshold_ms)
    threshold_mask = np.where(heatmap > threshold_ms, 1.0, 0.0)

    fig, (ax_heatmap, ax_mask) = plt.subplots(1, 2, figsize=(14.5, 5.6), constrained_layout=True)
    im = ax_heatmap.imshow(
        clipped_heatmap,
        aspect="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=threshold_ms,
    )
    ax_heatmap.set_title(f"Runtime Heatmap (clipped at {threshold_ms:.0f} ms)")
    ax_heatmap.set_xlabel("Grid Width (9 x L)")
    ax_heatmap.set_ylabel("Obstacle Count")
    ax_heatmap.set_xticks(range(len(row_values)))
    ax_heatmap.set_xticklabels(row_values, rotation=45)
    ax_heatmap.set_yticks(range(len(obstacle_counts)))
    ax_heatmap.set_yticklabels(obstacle_counts)
    colorbar = fig.colorbar(im, ax=ax_heatmap)
    colorbar.set_label("Mean Runtime (ms)")
    over_y, over_x = np.where(heatmap > threshold_ms)
    if over_y.size:
        ax_heatmap.scatter(
            over_x,
            over_y,
            marker="x",
            s=24,
            linewidths=0.9,
            color="#f6f8fa",
            label=f">{threshold_ms:.0f} ms",
        )
        ax_heatmap.legend(frameon=False, loc="lower right")
    for spine in ax_heatmap.spines.values():
        spine.set_color("#4a4f55")
        spine.set_linewidth(0.8)

    mask_im = ax_mask.imshow(
        threshold_mask,
        aspect="auto",
        cmap=plt.get_cmap("Greys", 2),
        vmin=0.0,
        vmax=1.0,
    )
    ax_mask.set_title(f"Threshold Exceedance Map ({threshold_ms:.0f} ms)")
    ax_mask.set_xlabel("Grid Width (9 x L)")
    ax_mask.set_ylabel("Obstacle Count")
    ax_mask.set_xticks(range(len(row_values)))
    ax_mask.set_xticklabels(row_values, rotation=45)
    ax_mask.set_yticks(range(len(obstacle_counts)))
    ax_mask.set_yticklabels(obstacle_counts)
    mask_cbar = fig.colorbar(mask_im, ax=ax_mask, ticks=[0, 1])
    mask_cbar.ax.set_yticklabels(["<= threshold", "> threshold"])
    for spine in ax_mask.spines.values():
        spine.set_color("#4a4f55")
        spine.set_linewidth(0.8)

    fig.suptitle(
        "Original EMPlanner DP Runtime Under Different Grid Widths and Obstacle Counts",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.row_node_min < 3 or args.row_node_max < args.row_node_min:
        raise ValueError("Invalid row-node range")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    row_values = list(range(args.row_node_min, args.row_node_max + 1))
    obstacle_counts = [int(value) for value in args.obstacle_counts]
    records: List[Dict[str, object]] = []

    total_jobs = len(row_values) * len(obstacle_counts)
    job_index = 0
    for obstacle_count in obstacle_counts:
        for row_node_num in row_values:
            job_index += 1
            rng_seed = args.seed + obstacle_count * 1000 + row_node_num
            rng = np.random.default_rng(rng_seed)
            times_ms: List[float] = []
            feasible_count = 0
            for _ in range(args.trials):
                raw_obstacles = generate_random_obstacles(
                    DEFAULT_S_RANGE,
                    DEFAULT_L_RANGE,
                    min_count=obstacle_count,
                    max_count=obstacle_count,
                    length_range=(0.6, 1.8),
                    width_range=(0.4, 1.4),
                    rng=rng,
                )
                obstacles = convert_obstacles(raw_obstacles)
                start_time = perf_counter()
                _, feasible = run_classic_dp(
                    col_node_num=args.col_node_num,
                    row_node_num=row_node_num,
                    sample_s=args.sample_s,
                    sample_l=args.sample_l,
                    sample_s_num=args.sample_s_num,
                    obstacles=obstacles,
                    car_width=args.car_width,
                )
                times_ms.append((perf_counter() - start_time) * 1000.0)
                if feasible:
                    feasible_count += 1

            metrics = summarize_ms(times_ms)
            metrics.update(
                {
                    "col_node_num": int(args.col_node_num),
                    "row_node_num": int(row_node_num),
                    "obstacle_count": int(obstacle_count),
                    "trials": int(args.trials),
                    "feasible_rate": float(feasible_count / args.trials),
                }
            )
            records.append(metrics)
            print(
                f"[{job_index}/{total_jobs}] {args.col_node_num}x{row_node_num} | "
                f"obstacles={obstacle_count} | mean={metrics['mean_ms']:.3f} ms | "
                f"p90={metrics['p90_ms']:.3f} ms | feasible_rate={metrics['feasible_rate']:.2f}"
            )

    json_path = output_dir / "emplanner_classic_dp_benchmark.json"
    csv_path = output_dir / "emplanner_classic_dp_benchmark.csv"
    png_path = output_dir / "emplanner_classic_dp_benchmark.png"
    payload = {
        "config": {
            "col_node_num": int(args.col_node_num),
            "row_node_min": int(args.row_node_min),
            "row_node_max": int(args.row_node_max),
            "obstacle_counts": obstacle_counts,
            "trials": int(args.trials),
            "seed": int(args.seed),
            "sample_s": float(args.sample_s),
            "sample_l": float(args.sample_l),
            "sample_s_num": float(args.sample_s_num),
            "car_width": float(args.car_width),
            "threshold_ms": float(args.threshold_ms),
            "s_range": list(DEFAULT_S_RANGE),
            "l_range": list(DEFAULT_L_RANGE),
        },
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(records, csv_path)
    plot_results(
        records=records,
        row_values=row_values,
        obstacle_counts=obstacle_counts,
        threshold_ms=float(args.threshold_ms),
        output_path=png_path,
    )

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
