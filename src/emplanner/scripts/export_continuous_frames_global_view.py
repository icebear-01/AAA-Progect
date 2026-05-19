#!/usr/bin/env python3
"""Convert ego-local continuous replanning frames into a unified world view."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export continuous replanning frame results in one world coordinate frame."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-script", type=Path, required=True)
    parser.add_argument("--path-x-min", type=float, default=0.0)
    parser.add_argument("--path-x-max", type=float, default=-1.0)
    parser.add_argument("--path-y-abs-max", type=float, default=-1.0)
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compose_pose(base: Tuple[float, float, float], local: Tuple[float, float, float]) -> Tuple[float, float, float]:
    bx, by, byaw = base
    lx, ly, lyaw = local
    cos_yaw = math.cos(byaw)
    sin_yaw = math.sin(byaw)
    wx = bx + cos_yaw * lx - sin_yaw * ly
    wy = by + sin_yaw * lx + cos_yaw * ly
    wyaw = normalize_angle(byaw + lyaw)
    return wx, wy, wyaw


def transform_xy(x: float, y: float, pose: Tuple[float, float, float]) -> Tuple[float, float]:
    px, py, pyaw = pose
    cos_yaw = math.cos(pyaw)
    sin_yaw = math.sin(pyaw)
    wx = px + cos_yaw * x - sin_yaw * y
    wy = py + sin_yaw * x + cos_yaw * y
    return wx, wy


def transform_yaw(yaw: float, pose: Tuple[float, float, float]) -> float:
    return normalize_angle(pose[2] + yaw)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transform_path_csv(src: Path, dst: Path, pose: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    fieldnames, rows = read_csv_rows(src)
    x_min = float("inf")
    x_max = float("-inf")
    y_abs_max = 0.0
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        yaw = float(row["yaw"])
        wx, wy = transform_xy(x, y, pose)
        wyaw = transform_yaw(yaw, pose)
        row["x"] = f"{wx:.6f}"
        row["y"] = f"{wy:.6f}"
        row["yaw"] = f"{wyaw:.6f}"
        x_min = min(x_min, wx)
        x_max = max(x_max, wx)
        y_abs_max = max(y_abs_max, abs(wy))
    write_csv_rows(dst, fieldnames, rows)
    return x_min, x_max, y_abs_max, y_abs_max


def transform_grid_csv(src: Path, dst: Path, pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    fieldnames, rows = read_csv_rows(src)
    x_min = float("inf")
    x_max = float("-inf")
    y_abs_max = 0.0
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        yaw = float(row["yaw"])
        wx, wy = transform_xy(x, y, pose)
        wyaw = transform_yaw(yaw, pose)
        row["x"] = f"{wx:.6f}"
        row["y"] = f"{wy:.6f}"
        row["yaw"] = f"{wyaw:.6f}"
        x_min = min(x_min, wx)
        x_max = max(x_max, wx)
        y_abs_max = max(y_abs_max, abs(wy))
    write_csv_rows(dst, fieldnames, rows)
    return x_min, x_max, y_abs_max


def transform_obstacles_csv(src: Path, dst: Path, pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    fieldnames, rows = read_csv_rows(src)
    x_min = float("inf")
    x_max = float("-inf")
    y_abs_max = 0.0
    for row in rows:
        x = float(row["center_x"])
        y = float(row["center_y"])
        yaw = float(row["yaw"])
        wx, wy = transform_xy(x, y, pose)
        wyaw = transform_yaw(yaw, pose)
        row["center_x"] = f"{wx:.6f}"
        row["center_y"] = f"{wy:.6f}"
        row["yaw"] = f"{wyaw:.6f}"
        x_min = min(x_min, wx)
        x_max = max(x_max, wx)
        y_abs_max = max(y_abs_max, abs(wy))
    write_csv_rows(dst, fieldnames, rows)
    return x_min, x_max, y_abs_max


def transform_trajectory_txt(src: Path, dst: Path, pose: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x_min = float("inf")
    x_max = float("-inf")
    y_abs_max = 0.0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split()[:2]
            wx, wy = transform_xy(float(x_str), float(y_str), pose)
            fout.write(f"{wx:.6f} {wy:.6f}\n")
            x_min = min(x_min, wx)
            x_max = max(x_max, wx)
            y_abs_max = max(y_abs_max, abs(wy))
    return x_min, x_max, y_abs_max


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    plot_script = args.plot_script.resolve()

    manifest = read_manifest(input_dir / "frame_manifest.csv")
    frame_dirs = [Path(row["scenario_dir"]) for row in manifest]
    frame_pngs = [Path(row["png_path"]) for row in manifest]
    output_dir.mkdir(parents=True, exist_ok=True)

    world_poses: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for i in range(1, len(manifest)):
        prev = manifest[i - 1]
        local_pose = (
            float(prev["next_x"]),
            float(prev["next_y"]),
            float(prev["next_yaw"]),
        )
        world_poses.append(compose_pose(world_poses[-1], local_pose))

    global_x_min = float("inf")
    global_x_max = float("-inf")
    global_y_abs_max = 0.0

    transformed_dirs: List[Path] = []
    for idx, frame_dir in enumerate(frame_dirs):
        dst_dir = output_dir / frame_dir.name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(frame_dir, dst_dir)
        pose = world_poses[idx]

        for filename in ["dp_path.csv", "qp_path.csv", "reference_path.csv"]:
            x_min, x_max, y_abs_max, _ = transform_path_csv(dst_dir / filename, dst_dir / filename, pose)
            global_x_min = min(global_x_min, x_min)
            global_x_max = max(global_x_max, x_max)
            global_y_abs_max = max(global_y_abs_max, y_abs_max)

        if (dst_dir / "dp_grid_points.csv").exists():
            x_min, x_max, y_abs_max = transform_grid_csv(dst_dir / "dp_grid_points.csv", dst_dir / "dp_grid_points.csv", pose)
            global_x_min = min(global_x_min, x_min)
            global_x_max = max(global_x_max, x_max)
            global_y_abs_max = max(global_y_abs_max, y_abs_max)

        if (dst_dir / "obstacles.csv").exists():
            x_min, x_max, y_abs_max = transform_obstacles_csv(dst_dir / "obstacles.csv", dst_dir / "obstacles.csv", pose)
            global_x_min = min(global_x_min, x_min)
            global_x_max = max(global_x_max, x_max)
            global_y_abs_max = max(global_y_abs_max, y_abs_max)

        if (dst_dir / "straight_trajectory.txt").exists():
            x_min, x_max, y_abs_max = transform_trajectory_txt(dst_dir / "straight_trajectory.txt", dst_dir / "straight_trajectory.txt", pose)
            global_x_min = min(global_x_min, x_min)
            global_x_max = max(global_x_max, x_max)
            global_y_abs_max = max(global_y_abs_max, y_abs_max)

        transformed_dirs.append(dst_dir)

    path_x_min = args.path_x_min if args.path_x_min >= 0.0 else math.floor(global_x_min)
    path_x_max = args.path_x_max if args.path_x_max > 0.0 else math.ceil(global_x_max + 0.5)
    path_y_abs_max = args.path_y_abs_max if args.path_y_abs_max > 0.0 else math.ceil(global_y_abs_max + 0.5)

    for idx, dst_dir in enumerate(transformed_dirs):
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--input-dir",
                str(dst_dir),
                "--paper",
                "--export-pdf",
                "--right-panel",
                "curvature",
                "--path-x-min",
                str(path_x_min),
                "--path-x-max",
                str(path_x_max),
                "--path-y-abs-max",
                str(path_y_abs_max),
            ],
            check=True,
        )
        png_src = dst_dir / "comparison_paper.png"
        png_dst = output_dir / f"frame_{idx:02d}_global.png"
        shutil.copy2(png_src, png_dst)

    shutil.copy2(input_dir / "frame_manifest.csv", output_dir / "frame_manifest.csv")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
