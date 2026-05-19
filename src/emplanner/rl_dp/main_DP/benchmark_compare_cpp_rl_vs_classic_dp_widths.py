#!/usr/bin/env python3
"""Compare classic DP runtime sweep (9x9..9x23) against C++ RL-DP runtime."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

from benchmark_emplanner_classic_dp import convert_obstacles, run_classic_dp
from sl_obstacles import Obstacle, generate_random_obstacles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classic DP width sweep against C++ RL-DP runtime."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--row-node-min", type=int, default=9)
    parser.add_argument("--row-node-max", type=int, default=23)
    parser.add_argument("--obstacle-min", type=int, default=10)
    parser.add_argument("--obstacle-max", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sample-s-num", type=float, default=5.0)
    parser.add_argument("--car-width", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def _font() -> FontProperties:
    return FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")


def _apply_plot_style() -> FontProperties:
    font = _font()
    plt.rcParams.update(
        {
            "font.family": font.get_name(),
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#fcfcfc",
            "axes.edgecolor": "#4a4f55",
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def _compile_cpp_benchmark(binary_path: Path, force_rebuild: bool = False) -> None:
    source_dir = Path("/home/wmd/elevetor_demo0317/AAA-Progect/src/emplanner/RL_DP")
    source_files = [
        source_dir / "benchmark_rl_dp_runtime.cpp",
        source_dir / "rl_dp.cpp",
        source_dir / "dp_planner.cpp",
        source_dir / "dp_policy.cpp",
    ]
    if not force_rebuild and binary_path.exists():
        latest_src = max(path.stat().st_mtime for path in source_files)
        if binary_path.stat().st_mtime >= latest_src:
            return

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-I/usr/local/onnxruntime/include",
        f"-I{source_dir}",
        *[str(path) for path in source_files],
        "-L/usr/local/onnxruntime/lib",
        "-lonnxruntime",
        "-Wl,-rpath,/usr/local/onnxruntime/lib",
        "-o",
        str(binary_path),
    ]
    subprocess.run(cmd, check=True)


def _generate_shared_scenarios(
    *,
    output_csv: Path,
    obstacle_counts: Sequence[int],
    trials: int,
    seed: int,
    s_range: Sequence[float],
    l_range: Sequence[float],
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    scenarios: List[Dict[str, object]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "scenario_id",
                "obstacle_count",
                "start_l",
                "center_s",
                "center_l",
                "length",
                "width",
                "yaw",
            ]
        )
        scenario_id = 0
        for obstacle_count in obstacle_counts:
            for _ in range(int(trials)):
                obstacles = generate_random_obstacles(
                    s_range,
                    l_range,
                    min_count=obstacle_count,
                    max_count=obstacle_count,
                    length_range=(0.6, 1.8),
                    width_range=(0.4, 1.4),
                    rng=rng,
                )
                start_l = 0.0
                scenarios.append(
                    {
                        "scenario_id": int(scenario_id),
                        "obstacle_count": int(obstacle_count),
                        "start_l": float(start_l),
                        "obstacles": obstacles,
                    }
                )
                for obstacle in obstacles:
                    writer.writerow(
                        [
                            scenario_id,
                            obstacle_count,
                            f"{start_l:.6f}",
                            f"{obstacle.center[0]:.6f}",
                            f"{obstacle.center[1]:.6f}",
                            f"{obstacle.length:.6f}",
                            f"{obstacle.width:.6f}",
                            f"{obstacle.yaw:.6f}",
                        ]
                    )
                scenario_id += 1
    return scenarios


def _run_cpp_rl_benchmark(
    *,
    binary_path: Path,
    onnx_path: Path,
    scenario_csv: Path,
    output_json: Path,
    s_samples: int,
    l_samples: int,
    s_range: Sequence[float],
    l_range: Sequence[float],
) -> Dict[str, object]:
    cmd = [
        str(binary_path),
        "--model",
        str(onnx_path),
        "--scenario-csv",
        str(scenario_csv),
        "--output-json",
        str(output_json),
        "--s-samples",
        str(int(s_samples)),
        "--l-samples",
        str(int(l_samples)),
        "--s-min",
        str(float(s_range[0])),
        "--s-max",
        str(float(s_range[1])),
        "--l-min",
        str(float(l_range[0])),
        "--l-max",
        str(float(l_range[1])),
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return json.loads(output_json.read_text(encoding="utf-8"))


def _benchmark_classic_dp(
    *,
    scenarios: Sequence[Dict[str, object]],
    row_values: Sequence[int],
    col_node_num: int,
    sample_s_num: float,
    car_width: float,
    s_range: Sequence[float],
    l_range: Sequence[float],
) -> Dict[int, Dict[int, Dict[str, float]]]:
    sample_s = float(s_range[1] - s_range[0]) / max(int(col_node_num) - 1, 1)
    summaries: Dict[int, Dict[int, List[float]]] = {
        int(obstacle_count): {int(row): [] for row in row_values}
        for obstacle_count in sorted({int(s["obstacle_count"]) for s in scenarios})
    }

    total_jobs = len(row_values) * len(scenarios)
    job_index = 0
    for row_node_num in row_values:
        sample_l = float(l_range[1] - l_range[0]) / max(int(row_node_num) - 1, 1)
        for scenario in scenarios:
            job_index += 1
            obstacle_count = int(scenario["obstacle_count"])
            classic_obstacles = convert_obstacles(scenario["obstacles"])
            t0 = perf_counter()
            run_classic_dp(
                col_node_num=int(col_node_num),
                row_node_num=int(row_node_num),
                sample_s=float(sample_s),
                sample_l=float(sample_l),
                sample_s_num=float(sample_s_num),
                obstacles=classic_obstacles,
                car_width=float(car_width),
            )
            elapsed_ms = (perf_counter() - t0) * 1000.0
            summaries[obstacle_count][int(row_node_num)].append(float(elapsed_ms))
            if job_index % max(1, len(scenarios)) == 0:
                print(
                    f"[classic-dp] row={row_node_num} progress={job_index}/{total_jobs} "
                    f"latest={elapsed_ms:.3f} ms"
                )

    return {
        obstacle_count: {
            row_node_num: _stats(values)
            for row_node_num, values in row_map.items()
        }
        for obstacle_count, row_map in summaries.items()
    }


def _plot_heatmap(
    *,
    obstacle_counts: Sequence[int],
    row_values: Sequence[int],
    dp_matrix: np.ndarray,
    rl_cpp_column: np.ndarray,
    output_path: Path,
    vmax_ms: float | None = None,
) -> None:
    font = _apply_plot_style()
    cmap = LinearSegmentedColormap.from_list(
        "paper_runtime",
        ["#163b66", "#2a6f97", "#5aa6a6", "#d9c27d", "#d97b29", "#b33f2f"],
        N=256,
    )

    fig, (ax_dp, ax_rl) = plt.subplots(
        1,
        2,
        figsize=(11.8, 8.6),
        gridspec_kw={"width_ratios": [5.2, 1.1]},
    )

    dp_display = np.asarray(dp_matrix, dtype=np.float64)
    rl_display = np.asarray(rl_cpp_column, dtype=np.float64)
    imshow_kwargs = {"aspect": "auto", "origin": "lower", "cmap": cmap}
    if vmax_ms is not None:
        vmax = float(vmax_ms)
        dp_display = np.clip(dp_display, 0.0, vmax)
        rl_display = np.clip(rl_display, 0.0, vmax)
        imshow_kwargs["vmin"] = 0.0
        imshow_kwargs["vmax"] = vmax

    im_dp = ax_dp.imshow(dp_display, **imshow_kwargs)
    ax_dp.set_title("传统DP规划耗时", fontproperties=font, pad=8)
    ax_dp.set_xlabel("L 的数量", fontproperties=font)
    ax_dp.set_ylabel("障碍物数量", fontproperties=font)
    ax_dp.set_xticks(np.arange(len(row_values)))
    ax_dp.set_xticklabels([str(row) for row in row_values], rotation=45, ha="right", fontproperties=font)
    ax_dp.set_yticks(np.arange(0, len(obstacle_counts), 2))
    ax_dp.set_yticklabels([str(v) for v in obstacle_counts[::2]], fontproperties=font)

    im_rl = ax_rl.imshow(rl_display[:, None], **imshow_kwargs)
    ax_rl.set_title("RL-C++", fontproperties=font, pad=8)
    ax_rl.set_xlabel("L 的数量", fontproperties=font)
    ax_rl.set_xticks([0])
    ax_rl.set_xticklabels(["23"], fontproperties=font)
    ax_rl.set_yticks(np.arange(0, len(obstacle_counts), 2))
    ax_rl.set_yticklabels([str(v) for v in obstacle_counts[::2]], fontproperties=font)

    cbar_dp = fig.colorbar(im_dp, ax=ax_dp, fraction=0.035, pad=0.02)
    cbar_dp.set_label("时间（ms）", fontproperties=font)
    cbar_rl = fig.colorbar(im_rl, ax=ax_rl, fraction=0.12, pad=0.08)
    cbar_rl.set_label("时间（ms）", fontproperties=font)

    fig.suptitle("L=5 至 23 时传统DP与 RL-C++ 规划耗时对比", fontproperties=font, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(
    *,
    row_values: Sequence[int],
    dp_mean_curve: np.ndarray,
    rl_cpp_mean: float,
    output_path: Path,
) -> None:
    font = _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(
        row_values,
        dp_mean_curve,
        color="#d04a02",
        marker="o",
        linewidth=2.4,
        markersize=5.8,
        label="传统DP",
    )
    ax.axhline(
        rl_cpp_mean,
        color="#1f5aa6",
        linestyle="--",
        linewidth=2.2,
        label="RL-C++（L=23）",
    )
    ax.set_xlabel("L 的数量", fontproperties=font)
    ax.set_ylabel("平均规划时间（ms）", fontproperties=font)
    ax.set_xticks(list(row_values))
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_title("传统DP采样增密与 RL-C++ 规划耗时对比", fontproperties=font, pad=10)
    ax.legend(prop=font, loc="upper left", frameon=True, fancybox=False, edgecolor="#80868b")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu")
    grid_spec = payload["grid_spec"]
    s_samples = int(grid_spec["s_samples"])
    l_samples = int(grid_spec["l_samples"])
    s_range = list(grid_spec["s_range"])
    l_range = list(grid_spec["l_range"])

    row_values = list(range(int(args.row_node_min), int(args.row_node_max) + 1))
    obstacle_counts = list(range(int(args.obstacle_min), int(args.obstacle_max) + 1))

    scenario_csv = output_dir / "shared_scenarios.csv"
    cpp_json = output_dir / "cpp_rl_runtime.json"
    merged_json = output_dir / "compare_cpp_rl_vs_classic_dp_widths.json"
    heatmap_path = output_dir / "compare_cpp_rl_vs_classic_dp_widths_heatmap.png"
    curve_path = output_dir / "compare_cpp_rl_vs_classic_dp_widths_curve.png"
    binary_path = output_dir / "build" / "benchmark_rl_dp_runtime"

    scenarios = _generate_shared_scenarios(
        output_csv=scenario_csv,
        obstacle_counts=obstacle_counts,
        trials=int(args.trials),
        seed=int(args.seed),
        s_range=s_range,
        l_range=l_range,
    )
    print(f"Saved shared scenarios to {scenario_csv}")

    _compile_cpp_benchmark(binary_path, force_rebuild=bool(args.force_rebuild))
    print(f"Compiled C++ benchmark: {binary_path}")

    cpp_payload = _run_cpp_rl_benchmark(
        binary_path=binary_path,
        onnx_path=args.onnx,
        scenario_csv=scenario_csv,
        output_json=cpp_json,
        s_samples=s_samples,
        l_samples=l_samples,
        s_range=s_range,
        l_range=l_range,
    )
    print(f"Saved C++ RL JSON to {cpp_json}")

    classic_summary = _benchmark_classic_dp(
        scenarios=scenarios,
        row_values=row_values,
        col_node_num=s_samples,
        sample_s_num=float(args.sample_s_num),
        car_width=float(args.car_width),
        s_range=s_range,
        l_range=l_range,
    )

    rl_summary = {
        int(record["obstacle_count"]): {
            "mean_ms": float(record["mean_ms"]),
            "p50_ms": float(record["p50_ms"]),
            "p95_ms": float(record["p95_ms"]),
            "max_ms": float(record["max_ms"]),
        }
        for record in cpp_payload["summary"]
    }

    dp_matrix = np.array(
        [
            [classic_summary[count][row]["mean_ms"] for row in row_values]
            for count in obstacle_counts
        ],
        dtype=np.float64,
    )
    rl_cpp_column = np.array([rl_summary[count]["mean_ms"] for count in obstacle_counts], dtype=np.float64)

    merged = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "onnx": str(args.onnx),
            "row_values": row_values,
            "obstacle_counts": obstacle_counts,
            "trials": int(args.trials),
            "sample_s_num": float(args.sample_s_num),
            "car_width": float(args.car_width),
            "seed": int(args.seed),
            "grid_spec": grid_spec,
            "scenario_csv": str(scenario_csv),
            "cpp_binary": str(binary_path),
        },
        "rl_cpp_summary": rl_summary,
        "classic_dp_summary": classic_summary,
    }
    merged_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    _plot_heatmap(
        obstacle_counts=obstacle_counts,
        row_values=row_values,
        dp_matrix=dp_matrix,
        rl_cpp_column=rl_cpp_column,
        output_path=heatmap_path,
    )
    _plot_curve(
        row_values=row_values,
        dp_mean_curve=dp_matrix.mean(axis=0),
        rl_cpp_mean=float(rl_cpp_column.mean()),
        output_path=curve_path,
    )

    print(f"Saved merged JSON to {merged_json}")
    print(f"Saved heatmap to {heatmap_path}")
    print(f"Saved curve to {curve_path}")
    print(
        "Summary | RL-C++ mean={:.3f} ms | classic DP 9x9 mean={:.3f} ms | classic DP 9x23 mean={:.3f} ms".format(
            float(rl_cpp_column.mean()),
            float(dp_matrix[:, 0].mean()),
            float(dp_matrix[:, -1].mean()),
        )
    )


if __name__ == "__main__":
    main()
