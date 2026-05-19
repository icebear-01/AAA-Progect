#!/usr/bin/env python3
"""Generate global-reference replanning frames with DP fallback.

The global straight reference line and static obstacles remain unchanged.
Each frame updates only the ego start pose. If the current frame's QP stage is
reported as unsuccessful, the next frame uses the DP path pose at s=advance_s
instead of the QP path pose.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-reference replanning frames with DP fallback."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-binary", type=Path, required=True)
    parser.add_argument("--plot-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario-prefix", type=str, default="fixed_ref_replan")
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--advance-s", type=float, default=2.0)
    parser.add_argument("--path-x-min", type=float, default=0.0)
    parser.add_argument("--path-x-max", type=float, default=18.0)
    parser.add_argument("--path-y-abs-max", type=float, default=6.0)
    parser.add_argument("--sample-s", type=float, default=1.0)
    parser.add_argument("--sample-l", type=float, default=0.286364)
    parser.add_argument("--sample-s-num", type=int, default=10)
    parser.add_argument("--sample-s-per-meters", type=int, default=20)
    parser.add_argument("--col-node-num", type=int, default=9)
    parser.add_argument("--row-node-num", type=int, default=23)
    parser.add_argument("--rl-dp-s-samples", type=int, default=9)
    parser.add_argument("--rl-dp-l-samples", type=int, default=23)
    parser.add_argument("--rl-dp-s-min", type=float, default=0.0)
    parser.add_argument("--rl-dp-s-max", type=float, default=8.0)
    parser.add_argument("--rl-dp-l-min", type=float, default=-3.15)
    parser.add_argument("--rl-dp-l-max", type=float, default=3.15)
    parser.add_argument("--w-qp-l", type=float, default=800.0)
    parser.add_argument("--w-qp-dl", type=float, default=2000.0)
    parser.add_argument("--w-qp-ddl", type=float, default=10000.0)
    parser.add_argument("--w-qp-ref-dp", type=float, default=50.0)
    parser.add_argument("--straight-length", type=float, default=20.0)
    parser.add_argument("--straight-step", type=float, default=0.1)
    parser.add_argument("--safe-distance", type=float, default=0.2)
    parser.add_argument("--safe-distance-wall", type=float, default=0.15)
    parser.add_argument("--relax-qp-start-dl", action="store_true")
    parser.add_argument("--qp-start-dl-slack", type=float, default=0.0)
    parser.add_argument(
        "--chain-source",
        choices=("auto", "qp", "dp"),
        default="auto",
        help="How to choose the next-frame start pose: successful QP only, DP only, or auto fallback.",
    )
    return parser.parse_args()


def read_path_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "index": float(row["index"]),
                    "s": float(row["s"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw": float(row["yaw"]),
                    "kappa": float(row["kappa"]),
                }
            )
    if not rows:
        raise RuntimeError(f"Empty path csv: {path}")
    return rows


def path_is_finite(rows: List[Dict[str, float]]) -> bool:
    for row in rows:
        for key, value in row.items():
            if key == "index":
                continue
            if not math.isfinite(value):
                return False
    return True


def pick_pose_at_s(rows: List[Dict[str, float]], target_s: float) -> Dict[str, float]:
    return min(rows, key=lambda row: abs(row["s"] - target_s))


def load_summary(path: Path) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        summary[key.strip()] = value.strip()
    return summary


def run_command(cmd: List[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def main() -> int:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    benchmark_binary = args.benchmark_binary.resolve()
    plot_script = args.plot_script.resolve()
    model_path = args.model.resolve()
    obstacle_csv = (source_dir / "obstacles.csv").resolve()
    source_qp_csv = (source_dir / "qp_path.csv").resolve()

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not benchmark_binary.exists():
        raise FileNotFoundError(benchmark_binary)
    if not plot_script.exists():
        raise FileNotFoundError(plot_script)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not obstacle_csv.exists():
        raise FileNotFoundError(obstacle_csv)
    if not source_qp_csv.exists():
        raise FileNotFoundError(source_qp_csv)

    pkg_dir = Path(__file__).resolve().parents[1]
    benchmark_results_root = pkg_dir / "benchmark_results"
    benchmark_results_root.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_pose = pick_pose_at_s(read_path_csv(source_qp_csv), 0.0)
    current_abs_s = current_pose["s"]
    manifest_rows: List[Dict[str, object]] = []

    for frame_idx in range(args.frame_count):
        cumulative_s = frame_idx * args.advance_s
        scenario_name = f"{args.scenario_prefix}_frame_{frame_idx:02d}_s{int(round(cumulative_s)):02d}"
        result_dir = benchmark_results_root / scenario_name
        frame_dir = output_dir / scenario_name
        if result_dir.exists():
            shutil.rmtree(result_dir)
        if frame_dir.exists():
            shutil.rmtree(frame_dir)

        run_command(
            [
                str(benchmark_binary),
                "__name:=emplanner_compare_benchmark",
                f"_scenario_name:={scenario_name}",
                "_use_straight_trajectory:=true",
                "_straight_start_x:=0.0",
                "_straight_start_y:=0.0",
                f"_straight_length:={args.straight_length}",
                f"_straight_step:={args.straight_step}",
                "_straight_turn_x:=1000000.0",
                "_straight_turn_angle_deg:=0.0",
                "_straight_turn_arc_length:=0.0",
                "_turn_shape_case:=single_arc",
                "_second_turn_gap:=0.8",
                "_second_turn_angle_deg:=0.0",
                "_second_turn_arc_length:=0.0",
                "_use_rl_dp:=true",
                f"_rl_dp_model_path:={model_path}",
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
                f"_safe_distance:={args.safe_distance}",
                f"_safe_distance_wall:={args.safe_distance_wall}",
                f"_relax_qp_start_dl:={'true' if args.relax_qp_start_dl else 'false'}",
                f"_qp_start_dl_slack:={args.qp_start_dl_slack}",
                "_plan_iterations:=0",
                f"_start_x:={current_pose['x']}",
                f"_start_y:={current_pose['y']}",
                f"_start_yaw:={current_pose['yaw']}",
                f"_obstacle_csv:={obstacle_csv}",
            ]
        )

        run_command(
            [
                sys.executable,
                str(plot_script),
                "--input-dir",
                str(result_dir),
                "--paper",
                "--export-pdf",
                "--right-panel",
                "curvature",
                "--path-x-min",
                str(args.path_x_min),
                "--path-x-max",
                str(args.path_x_max),
                "--path-y-abs-max",
                str(args.path_y_abs_max),
            ]
        )

        shutil.copytree(result_dir, frame_dir)
        png_src = frame_dir / "comparison_paper.png"
        png_dst = output_dir / f"frame_{frame_idx:02d}_global_fixed_ref.png"
        shutil.copy2(png_src, png_dst)

        summary = load_summary(frame_dir / "summary.txt")
        qp_ok = summary.get("qp_running_normally", "").lower() == "true"
        if args.chain_source == "qp":
            next_source = "qp"
        elif args.chain_source == "dp":
            next_source = "dp"
        else:
            next_source = "qp" if qp_ok else "dp"
        next_csv = frame_dir / f"{next_source}_path.csv"
        next_rows = read_path_csv(next_csv)
        if not path_is_finite(next_rows):
            raise RuntimeError(f"Non-finite {next_source}_path at frame {frame_idx}: {next_csv}")
        next_abs_s = current_abs_s + args.advance_s
        current_pose = pick_pose_at_s(next_rows, next_abs_s)
        current_abs_s = current_pose["s"]

        manifest_rows.append(
            {
                "frame_index": frame_idx,
                "cumulative_s": cumulative_s,
                "next_abs_s": current_abs_s,
                "chain_source_mode": args.chain_source,
                "relax_qp_start_dl": args.relax_qp_start_dl,
                "qp_start_dl_slack": args.qp_start_dl_slack,
                "next_pose_source": next_source,
                "qp_running_normally": qp_ok,
                "start_x": current_pose["x"],
                "start_y": current_pose["y"],
                "start_yaw": current_pose["yaw"],
                "next_local_s": current_pose["s"],
                "scenario_dir": str(frame_dir),
                "png_path": str(png_dst),
            }
        )

    manifest_path = output_dir / "frame_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
