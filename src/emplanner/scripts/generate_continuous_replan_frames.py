#!/usr/bin/env python3
"""Generate continuous local replanning frames for a fixed benchmark scenario.

Frame 0 replans the original scenario.
Each following frame takes the previous frame's optimized path pose at local
``s=advance_s`` as the new ego pose, transforms both the optimized path and the
static obstacles into that new ego-local frame, and replans again.

This matches a real-time local planner rollout where the vehicle advances along
the previous optimized path and static obstacles appear to move backward in the
ego frame.
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
        description="Generate sequential RL-DP+QP replanning frames."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-binary", type=Path, required=True)
    parser.add_argument("--plot-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scenario-prefix", type=str, default="continuous_replan")
    parser.add_argument("--frame-count", type=int, default=6)
    parser.add_argument("--advance-s", type=float, default=2.0)
    parser.add_argument("--extension-length", type=float, default=10.0)
    parser.add_argument("--trajectory-step", type=float, default=0.1)
    parser.add_argument("--path-x-min", type=float, default=0.0)
    parser.add_argument("--path-x-max", type=float, default=10.0)
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


def read_obstacles_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "id": int(row["id"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "length": float(row["length"]),
                    "width": float(row["width"]),
                    "yaw": float(row["yaw"]),
                    "x_vel": float(row.get("x_vel", 0.0)),
                    "y_vel": float(row.get("y_vel", 0.0)),
                }
            )
    if not rows:
        raise RuntimeError(f"Empty obstacles csv: {path}")
    return rows


def pick_pose_at_s(path_rows: List[Dict[str, float]], target_s: float) -> Dict[str, float]:
    return min(path_rows, key=lambda row: abs(row["s"] - target_s))


def transform_to_local(x: float, y: float, origin_x: float, origin_y: float, origin_yaw: float) -> Dict[str, float]:
    dx = x - origin_x
    dy = y - origin_y
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    return {
        "x": cos_yaw * dx + sin_yaw * dy,
        "y": -sin_yaw * dx + cos_yaw * dy,
    }


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def transform_path_rows(
    path_rows: List[Dict[str, float]],
    start_s: float,
    extension_length: float,
    step: float,
) -> List[Dict[str, float]]:
    anchor = pick_pose_at_s(path_rows, start_s)
    start_index = path_rows.index(anchor)
    kept_rows = path_rows[start_index:]
    if not kept_rows:
        kept_rows = [path_rows[-1]]

    transformed: List[Dict[str, float]] = []
    for idx, row in enumerate(kept_rows):
        local = transform_to_local(row["x"], row["y"], anchor["x"], anchor["y"], anchor["yaw"])
        transformed.append(
            {
                "index": float(idx),
                "s": max(0.0, row["s"] - anchor["s"]),
                "x": local["x"],
                "y": local["y"],
                "yaw": normalize_angle(row["yaw"] - anchor["yaw"]),
                "kappa": row["kappa"],
            }
        )

    last = transformed[-1]
    heading = last["yaw"]
    x = last["x"]
    y = last["y"]
    current_s = last["s"]
    count = max(1, int(math.ceil(extension_length / max(step, 1e-3))))
    for i in range(count):
        x += math.cos(heading) * step
        y += math.sin(heading) * step
        current_s += step
        transformed.append(
            {
                "index": float(len(transformed)),
                "s": current_s,
                "x": x,
                "y": y,
                "yaw": heading,
                "kappa": 0.0 if i == 0 else transformed[-1]["kappa"],
            }
        )
    return transformed


def transform_obstacles(
    obstacles: List[Dict[str, float]],
    anchor_pose: Dict[str, float],
) -> List[Dict[str, float]]:
    transformed: List[Dict[str, float]] = []
    for obstacle in obstacles:
        local = transform_to_local(
            obstacle["center_x"],
            obstacle["center_y"],
            anchor_pose["x"],
            anchor_pose["y"],
            anchor_pose["yaw"],
        )
        transformed.append(
            {
                "id": obstacle["id"],
                "center_x": local["x"],
                "center_y": local["y"],
                "length": obstacle["length"],
                "width": obstacle["width"],
                "yaw": normalize_angle(obstacle["yaw"] - anchor_pose["yaw"]),
                "x_vel": obstacle["x_vel"],
                "y_vel": obstacle["y_vel"],
            }
        )
    return transformed


def write_trajectory_txt(path_rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in path_rows:
            f.write(f"{row['x']:.6f} {row['y']:.6f}\n")


def write_obstacles_csv(obstacles: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "center_x",
                "center_y",
                "length",
                "width",
                "yaw",
                "x_vel",
                "y_vel",
            ],
        )
        writer.writeheader()
        writer.writerows(obstacles)


def run_command(cmd: List[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def path_is_finite(path_rows: List[Dict[str, float]]) -> bool:
    for row in path_rows:
        for key, value in row.items():
            if key == "index":
                continue
            if not math.isfinite(value):
                return False
    return True


def main() -> int:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    benchmark_binary = args.benchmark_binary.resolve()
    plot_script = args.plot_script.resolve()
    model_path = args.model.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if not benchmark_binary.exists():
        raise FileNotFoundError(benchmark_binary)
    if not plot_script.exists():
        raise FileNotFoundError(plot_script)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    obstacle_csv = source_dir / "obstacles.csv"
    reference_csv = source_dir / "reference_path.csv"
    if not obstacle_csv.exists():
        raise FileNotFoundError(obstacle_csv)
    if not reference_csv.exists():
        raise FileNotFoundError(reference_csv)

    pkg_dir = Path(__file__).resolve().parents[1]
    benchmark_results_root = pkg_dir / "benchmark_results"
    benchmark_results_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "frame_manifest.csv"
    current_reference_rows = read_path_csv(reference_csv)
    current_obstacles = read_obstacles_csv(obstacle_csv)

    manifest_rows: List[Dict[str, object]] = []

    for frame_idx in range(args.frame_count):
        cumulative_s = frame_idx * args.advance_s
        scenario_name = f"{args.scenario_prefix}_frame_{frame_idx:02d}_s{int(round(cumulative_s)):02d}"
        benchmark_dir = benchmark_results_root / scenario_name
        frame_dir = output_dir / scenario_name

        if benchmark_dir.exists():
            shutil.rmtree(benchmark_dir)
        if frame_dir.exists():
            shutil.rmtree(frame_dir)

        trajectory_file = output_dir / f"frame_{frame_idx:02d}_trajectory.txt"
        obstacle_file = output_dir / f"frame_{frame_idx:02d}_obstacles.csv"
        write_trajectory_txt(current_reference_rows, trajectory_file)
        write_obstacles_csv(current_obstacles, obstacle_file)

        cmd = [
            str(benchmark_binary),
            "__name:=emplanner_compare_benchmark",
            f"_scenario_name:={scenario_name}",
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
            "_plan_iterations:=0",
            "_start_x:=0.0",
            "_start_y:=0.0",
            "_start_yaw:=0.0",
            f"_obstacle_csv:={obstacle_file}",
            "_use_straight_trajectory:=false",
            f"_trajectory_file:={trajectory_file}",
        ]
        run_command(cmd)

        run_command(
            [
                sys.executable,
                str(plot_script),
                "--input-dir",
                str(benchmark_dir),
                "--paper",
                "--export-pdf",
                "--right-panel",
                "curvature",
                "--path-x-min",
                str(args.path_x_min),
                "--path-x-max",
                str(args.path_x_max),
            ]
        )

        shutil.copytree(benchmark_dir, frame_dir)
        png_src = frame_dir / "comparison_paper.png"
        png_dst = output_dir / f"frame_{frame_idx:02d}_s{int(round(cumulative_s)):02d}.png"
        shutil.copy2(png_src, png_dst)

        qp_rows = read_path_csv(frame_dir / "qp_path.csv")
        anchor_pose = pick_pose_at_s(qp_rows, args.advance_s)
        if not path_is_finite(qp_rows):
            raise RuntimeError(f"Non-finite qp_path at frame {frame_idx}: {frame_dir / 'qp_path.csv'}")
        summary_txt = (frame_dir / "summary.txt").read_text()
        qp_ok = "qp_running_normally: true" in summary_txt
        manifest_rows.append(
            {
                "frame_index": frame_idx,
                "cumulative_s": cumulative_s,
                "start_x": 0.0,
                "start_y": 0.0,
                "start_yaw": 0.0,
                "qp_running_normally": qp_ok,
                "next_local_s": anchor_pose["s"],
                "next_x": anchor_pose["x"],
                "next_y": anchor_pose["y"],
                "next_yaw": anchor_pose["yaw"],
                "scenario_dir": str(frame_dir),
                "png_path": str(png_dst),
            }
        )
        current_reference_rows = transform_path_rows(
            qp_rows,
            args.advance_s,
            args.extension_length,
            args.trajectory_step,
        )
        current_obstacles = transform_obstacles(current_obstacles, anchor_pose)

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "cumulative_s",
                "start_x",
                "start_y",
                "start_yaw",
                "qp_running_normally",
                "next_local_s",
                "next_x",
                "next_y",
                "next_yaw",
                "scenario_dir",
                "png_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
