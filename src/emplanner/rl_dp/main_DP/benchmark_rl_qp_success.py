#!/usr/bin/env python3
"""Batch-evaluate RL+QP planning success on an offline scenario dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RL+QP success rate on a screened scenario dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=Path("/home/wmd/elevetor_demo0317/AAA-Progect/devel/lib/emplanner/emplanner_compare_benchmark"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/wmd/elevetor_demo0317/AAA-Progect/src/emplanner/rl_dp/main/onnx/"
            "ppo_policy_dataset20k_mixed_complex_v2_20260408_update_20000.onnx"
        ),
    )
    parser.add_argument(
        "--benchmark-results-root",
        type=Path,
        default=Path("/home/wmd/elevetor_demo0317/AAA-Progect/src/emplanner/benchmark_results"),
    )
    parser.add_argument(
        "--scenario-name",
        default="tmp_rl_qp_dataset_batch",
        help="benchmark result directory name reused for each scenario",
    )
    parser.add_argument(
        "--obstacle-csv",
        type=Path,
        default=Path("/tmp/rl_qp_dataset_batch_obstacles.csv"),
        help="temporary obstacle csv reused for each scenario",
    )
    parser.add_argument(
        "--node-name",
        default="emplanner_compare_benchmark_dataset_batch",
        help="ROS node name used for the benchmark process",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument(
        "--planner-mode",
        choices=["rl_dp", "classic"],
        default="rl_dp",
        help="planner backend to evaluate",
    )
    parser.add_argument(
        "--success-mode",
        choices=["rl_qp", "dp_only"],
        default="rl_qp",
        help="success criterion: dp_only ignores QP status, rl_qp requires QP normal too",
    )
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--sample-s", type=float, default=1.0)
    parser.add_argument("--sample-l", type=float, default=0.35)
    parser.add_argument("--sample-s-num", type=float, default=10.0)
    parser.add_argument("--sample-s-per-meters", type=float, default=20.0)
    parser.add_argument("--col-node-num", type=int, default=9)
    parser.add_argument("--row-node-num", type=int, default=23)
    parser.add_argument("--rl-dp-s-samples", type=int, default=9)
    parser.add_argument("--rl-dp-l-samples", type=int, default=23)
    parser.add_argument("--rl-dp-s-min", type=float, default=0.0)
    parser.add_argument("--rl-dp-s-max", type=float, default=8.0)
    parser.add_argument("--rl-dp-l-min", type=float, default=-3.85)
    parser.add_argument("--rl-dp-l-max", type=float, default=3.85)
    parser.add_argument("--w-qp-l", type=float, default=800.0)
    parser.add_argument("--w-qp-dl", type=float, default=2000.0)
    parser.add_argument("--w-qp-ddl", type=float, default=10000.0)
    parser.add_argument("--w-qp-ref-dp", type=float, default=50.0)
    parser.add_argument("--straight-start-x", type=float, default=0.0)
    parser.add_argument("--straight-start-y", type=float, default=0.0)
    parser.add_argument("--straight-length", type=float, default=10.0)
    parser.add_argument("--straight-step", type=float, default=0.1)
    parser.add_argument("--plan-iterations", type=int, default=0)
    return parser.parse_args()


def write_obstacles_csv(csv_path: Path, obstacles: Iterable[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "center_x", "center_y", "length", "width", "yaw", "x_vel", "y_vel"])
        for i, obstacle in enumerate(obstacles):
            center_s, center_l = obstacle["center"]
            writer.writerow(
                [
                    i,
                    float(center_s),
                    float(center_l),
                    float(obstacle["length"]),
                    float(obstacle["width"]),
                    float(obstacle["yaw"]),
                    0.0,
                    0.0,
                ]
            )


def load_summary(summary_path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not summary_path.exists():
        return result
    for line in summary_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def safe_float(summary: Dict[str, str], key: str) -> float | None:
    value = summary.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(summary: Dict[str, str], key: str) -> int | None:
    value = summary.get(key)
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")
    if not args.benchmark_binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {args.benchmark_binary}")
    if not args.model.exists():
        raise FileNotFoundError(f"onnx model not found: {args.model}")

    dataset = json.loads(args.dataset.read_text())
    scenarios = dataset["scenarios"]
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    result_dir = args.benchmark_results_root / args.scenario_name
    summary_path = result_dir / "summary.txt"

    rows: List[Dict[str, object]] = []
    grouped_rows: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    overall_success = 0
    start_time = time.time()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fieldnames = [
        "scenario_index",
        "obstacle_count",
        "start_l",
        "source",
        "success",
        "returncode",
        "dp_source",
        "qp_running_normally",
        "dp_points",
        "dp_path_length",
        "planner_total_ms",
        "dp_sampling_ms",
        "qp_optimization_ms",
    ]
    csv_file = args.output_csv.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    writer.writeheader()

    try:
        for index, scenario in enumerate(scenarios):
            if result_dir.exists():
                shutil.rmtree(result_dir)

            write_obstacles_csv(args.obstacle_csv, scenario["obstacles"])
            start_l = float(scenario["start_l"])
            use_rl_dp = args.planner_mode == "rl_dp"
            expected_dp_source = "RL_DP" if use_rl_dp else "classic"
            cmd = [
                str(args.benchmark_binary),
                f"__name:={args.node_name}",
                f"_scenario_name:={args.scenario_name}",
                "_use_straight_trajectory:=true",
                f"_straight_start_x:={args.straight_start_x}",
                f"_straight_start_y:={args.straight_start_y}",
                f"_straight_length:={args.straight_length}",
                f"_straight_step:={args.straight_step}",
                f"_use_rl_dp:={'true' if use_rl_dp else 'false'}",
                f"_sample_s:={args.sample_s}",
                f"_sample_l:={args.sample_l}",
                f"_sample_s_num:={args.sample_s_num}",
                f"_sample_s_per_meters:={args.sample_s_per_meters}",
                f"_col_node_num:={args.col_node_num}",
                f"_row_node_num:={args.row_node_num}",
                f"_rl_dp_s_samples:={args.rl_dp_s_samples}",
                f"_rl_dp_l_samples:={args.rl_dp_l_samples}",
                f"_rl_dp_s_min:={args.rl_dp_s_min}",
                f"_rl_dp_s_max:={args.rl_dp_s_max}",
                f"_rl_dp_l_min:={args.rl_dp_l_min}",
                f"_rl_dp_l_max:={args.rl_dp_l_max}",
                f"_w_qp_l:={args.w_qp_l}",
                f"_w_qp_dl:={args.w_qp_dl}",
                f"_w_qp_ddl:={args.w_qp_ddl}",
                f"_w_qp_ref_dp:={args.w_qp_ref_dp}",
                f"_plan_iterations:={args.plan_iterations}",
                f"_start_x:={args.start_x}",
                f"_start_y:={start_l}",
                f"_start_yaw:={args.start_yaw}",
                f"_obstacle_csv:={args.obstacle_csv}",
            ]
            if use_rl_dp:
                cmd.append(f"_rl_dp_model_path:={args.model}")
            completed = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            summary = load_summary(summary_path)
            dp_source = summary.get("dp_source", "")
            qp_running_normally = summary.get("qp_running_normally", "").lower() == "true"
            dp_points = safe_int(summary, "dp_points")
            dp_path_length = safe_float(summary, "dp_path_length")
            dp_success = (
                completed.returncode == 0
                and summary_path.exists()
                and dp_source == expected_dp_source
                and (dp_points or 0) > 0
                and (dp_path_length or 0.0) > 0.0
            )
            success = dp_success if args.success_mode == "dp_only" else (dp_success and qp_running_normally)
            planner_total_ms = safe_float(summary, "planner_total_ms")
            dp_sampling_ms = safe_float(summary, "dp_sampling_ms")
            qp_optimization_ms = safe_float(summary, "qp_optimization_ms")

            row = {
                "scenario_index": int(scenario["scenario_index"]),
                "obstacle_count": int(scenario["obstacle_count"]),
                "start_l": start_l,
                "source": scenario.get("source", ""),
                "success": bool(success),
                "returncode": int(completed.returncode),
                "dp_source": dp_source,
                "qp_running_normally": qp_running_normally,
                "dp_points": dp_points,
                "dp_path_length": dp_path_length,
                "planner_total_ms": planner_total_ms,
                "dp_sampling_ms": dp_sampling_ms,
                "qp_optimization_ms": qp_optimization_ms,
            }
            rows.append(row)
            grouped_rows[int(scenario["obstacle_count"])].append(row)
            writer.writerow(row)
            csv_file.flush()
            if success:
                overall_success += 1

            if args.progress_interval > 0 and (index + 1) % args.progress_interval == 0:
                elapsed = time.time() - start_time
                rate = overall_success / float(index + 1)
                print(
                    f"progress {index + 1}/{len(scenarios)} "
                    f"success={overall_success} rate={rate:.4f} elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    finally:
        csv_file.close()

    def mean_of(key: str, entries: List[Dict[str, object]]) -> float | None:
        values = [entry[key] for entry in entries if isinstance(entry[key], (int, float))]
        if not values:
            return None
        return float(sum(values) / len(values))

    by_obstacle_count: Dict[str, Dict[str, object]] = {}
    for obstacle_count in sorted(grouped_rows):
        entries = grouped_rows[obstacle_count]
        success_count = sum(1 for entry in entries if entry["success"])
        by_obstacle_count[str(obstacle_count)] = {
            "success": int(success_count),
            "total": int(len(entries)),
            "success_rate": float(success_count / max(1, len(entries))),
            "planner_total_ms_mean": mean_of("planner_total_ms", entries),
            "dp_sampling_ms_mean": mean_of("dp_sampling_ms", entries),
            "qp_optimization_ms_mean": mean_of("qp_optimization_ms", entries),
        }

    result = {
        "dataset": str(args.dataset),
        "planner_mode": args.planner_mode,
        "model": str(args.model) if args.planner_mode == "rl_dp" else "",
        "success_mode": args.success_mode,
        "scenario_count": int(len(scenarios)),
        "overall_success": int(overall_success),
        "overall_success_rate": float(overall_success / max(1, len(scenarios))),
        "planner_total_ms_mean": mean_of("planner_total_ms", rows),
        "dp_sampling_ms_mean": mean_of("dp_sampling_ms", rows),
        "qp_optimization_ms_mean": mean_of("qp_optimization_ms", rows),
        "by_obstacle_count": by_obstacle_count,
        "sample_failures": [row for row in rows if not row["success"]][:30],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "scenario_count": result["scenario_count"],
                "overall_success": result["overall_success"],
                "overall_success_rate": result["overall_success_rate"],
                "planner_total_ms_mean": result["planner_total_ms_mean"],
                "dp_sampling_ms_mean": result["dp_sampling_ms_mean"],
                "qp_optimization_ms_mean": result["qp_optimization_ms_mean"],
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
