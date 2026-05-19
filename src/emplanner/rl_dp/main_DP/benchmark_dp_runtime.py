#!/usr/bin/env python3
"""Benchmark traditional DP runtime under different grid widths and obstacle counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from rl_env import SLPathEnv
from sl_grid import DEFAULT_L_RANGE, DEFAULT_S_RANGE, GridSpec


LINE_COLORS = [
    "#1f4e79",
    "#d97904",
    "#2e8b57",
    "#b24745",
    "#6b5b95",
]


def _pick_font_family(candidates: List[str]) -> str | None:
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
        description="Benchmark traditional DP runtime over grid sizes and obstacle counts."
    )
    parser.add_argument(
        "--l-samples-min",
        type=int,
        default=9,
        help="minimum lateral grid sample count",
    )
    parser.add_argument(
        "--l-samples-max",
        type=int,
        default=30,
        help="maximum lateral grid sample count",
    )
    parser.add_argument(
        "--obstacle-counts",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
        help="list of obstacle counts to benchmark",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="number of random scenes per configuration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260408,
        help="base random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("temp/dp_runtime_benchmark_20260408"),
        help="directory for json/csv/png outputs",
    )
    parser.add_argument(
        "--vehicle-length",
        type=float,
        default=0.0,
        help="vehicle length used for OBB collision checking; 0 disables footprint mode",
    )
    parser.add_argument(
        "--vehicle-width",
        type=float,
        default=0.0,
        help="vehicle width used for OBB collision checking; 0 disables footprint mode",
    )
    parser.add_argument(
        "--interpolation-points",
        type=int,
        default=3,
        help="interpolation points checked on each DP transition",
    )
    return parser.parse_args()


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


def build_env(
    spec: GridSpec,
    obstacle_count: int,
    seed: int,
    *,
    vehicle_length: float,
    vehicle_width: float,
    interpolation_points: int,
) -> SLPathEnv:
    return SLPathEnv(
        spec,
        min_obstacles=obstacle_count,
        max_obstacles=obstacle_count,
        lateral_move_limit=3,
        start_clear_fraction=0.2,
        scenario_pool_size=1,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        interpolation_points=interpolation_points,
        seed=seed,
    )


def benchmark_configuration(
    *,
    spec: GridSpec,
    obstacle_count: int,
    trials: int,
    seed: int,
    vehicle_length: float,
    vehicle_width: float,
    interpolation_points: int,
) -> Dict[str, object]:
    env = build_env(
        spec,
        obstacle_count,
        seed,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        interpolation_points=interpolation_points,
    )
    times_ms: List[float] = []
    feasible_count = 0

    for _ in range(trials):
        observation = env.reset()
        start_index = int(np.asarray(observation["path_indices"], dtype=np.int32)[0])
        start_time = perf_counter()
        dp_result = env._evaluate_scenario_with_dp(start_index)
        elapsed_ms = (perf_counter() - start_time) * 1000.0
        times_ms.append(elapsed_ms)
        if dp_result.feasible:
            feasible_count += 1

    metrics = summarize_ms(times_ms)
    metrics.update(
        {
            "s_samples": int(spec.s_samples),
            "l_samples": int(spec.l_samples),
            "obstacle_count": int(obstacle_count),
            "trials": int(trials),
            "feasible_rate": float(feasible_count / trials),
        }
    )
    return metrics


def save_csv(records: List[Dict[str, object]], csv_path: Path) -> None:
    header = [
        "s_samples",
        "l_samples",
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
    l_samples_values: List[int],
    obstacle_counts: List[int],
    output_path: Path,
) -> None:
    apply_paper_style()
    heatmap = np.full((len(obstacle_counts), len(l_samples_values)), np.nan, dtype=np.float64)
    for row_idx, obstacle_count in enumerate(obstacle_counts):
        for col_idx, l_samples in enumerate(l_samples_values):
            match = next(
                (
                    record
                    for record in records
                    if record["obstacle_count"] == obstacle_count
                    and record["l_samples"] == l_samples
                ),
                None,
            )
            if match is not None:
                heatmap[row_idx, col_idx] = float(match["mean_ms"])

    fig, (ax_heatmap, ax_lines) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.4),
        constrained_layout=True,
    )

    im = ax_heatmap.imshow(heatmap, aspect="auto", cmap="cividis")
    ax_heatmap.set_title("DP Runtime Heatmap")
    ax_heatmap.set_xlabel("Grid Width (9 x L)")
    ax_heatmap.set_ylabel("Obstacle Count")
    ax_heatmap.set_xticks(range(len(l_samples_values)))
    ax_heatmap.set_xticklabels(l_samples_values, rotation=45)
    ax_heatmap.set_yticks(range(len(obstacle_counts)))
    ax_heatmap.set_yticklabels(obstacle_counts)
    colorbar = fig.colorbar(im, ax=ax_heatmap)
    colorbar.set_label("Mean Runtime (ms)")
    for spine in ax_heatmap.spines.values():
        spine.set_color("#4a4f55")
        spine.set_linewidth(0.8)

    for line_idx, obstacle_count in enumerate(obstacle_counts):
        xs = []
        ys = []
        for l_samples in l_samples_values:
            match = next(
                (
                    record
                    for record in records
                    if record["obstacle_count"] == obstacle_count
                    and record["l_samples"] == l_samples
                ),
                None,
            )
            if match is None:
                continue
            xs.append(int(l_samples))
            ys.append(float(match["mean_ms"]))
        ax_lines.plot(
            xs,
            ys,
            marker="o",
            markersize=4.2,
            linewidth=2.0,
            color=LINE_COLORS[line_idx % len(LINE_COLORS)],
            label=f"{obstacle_count} obstacles",
        )

    ax_lines.set_title("DP Runtime Curves")
    ax_lines.set_xlabel("Grid Width L Samples")
    ax_lines.set_ylabel("Mean Runtime (ms)")
    ax_lines.grid(True)
    ax_lines.legend(frameon=False, loc="upper left")
    for spine in ax_lines.spines.values():
        spine.set_color("#4a4f55")
        spine.set_linewidth(0.8)

    fig.suptitle(
        "Traditional DP Planning Time Under Different Grid Widths and Obstacle Densities",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.l_samples_min < 2 or args.l_samples_max < args.l_samples_min:
        raise ValueError("Invalid l-samples range")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if not args.obstacle_counts:
        raise ValueError("--obstacle-counts must not be empty")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    l_samples_values = list(range(args.l_samples_min, args.l_samples_max + 1))
    obstacle_counts = [int(value) for value in args.obstacle_counts]
    records: List[Dict[str, object]] = []

    total_jobs = len(l_samples_values) * len(obstacle_counts)
    job_index = 0
    for obstacle_count in obstacle_counts:
        for l_samples in l_samples_values:
            job_index += 1
            spec = GridSpec(
                s_range=DEFAULT_S_RANGE,
                l_range=DEFAULT_L_RANGE,
                s_samples=9,
                l_samples=int(l_samples),
            )
            metrics = benchmark_configuration(
                spec=spec,
                obstacle_count=obstacle_count,
                trials=args.trials,
                seed=args.seed + obstacle_count * 1000 + l_samples,
                vehicle_length=float(args.vehicle_length),
                vehicle_width=float(args.vehicle_width),
                interpolation_points=int(args.interpolation_points),
            )
            records.append(metrics)
            print(
                f"[{job_index}/{total_jobs}] 9x{l_samples} | obstacles={obstacle_count} | "
                f"mean={metrics['mean_ms']:.3f} ms | p90={metrics['p90_ms']:.3f} ms | "
                f"feasible_rate={metrics['feasible_rate']:.2f}"
            )

    json_path = output_dir / "dp_runtime_benchmark.json"
    csv_path = output_dir / "dp_runtime_benchmark.csv"
    png_path = output_dir / "dp_runtime_benchmark.png"

    payload = {
        "config": {
            "s_samples": 9,
            "l_samples_min": int(args.l_samples_min),
            "l_samples_max": int(args.l_samples_max),
            "obstacle_counts": obstacle_counts,
            "trials": int(args.trials),
            "seed": int(args.seed),
            "vehicle_length": float(args.vehicle_length),
            "vehicle_width": float(args.vehicle_width),
            "interpolation_points": int(args.interpolation_points),
            "s_range": list(DEFAULT_S_RANGE),
            "l_range": list(DEFAULT_L_RANGE),
        },
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(records, csv_path)
    plot_results(
        records=records,
        l_samples_values=l_samples_values,
        obstacle_counts=obstacle_counts,
        output_path=png_path,
    )

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {png_path}")


if __name__ == "__main__":
    main()
