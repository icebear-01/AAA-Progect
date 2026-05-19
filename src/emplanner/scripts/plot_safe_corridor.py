#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def pick_font_family(candidates):
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def read_path_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "index": [int(row["index"]) for row in rows],
        "s": [float(row["s"]) for row in rows],
        "x": [float(row["x"]) for row in rows],
        "y": [float(row["y"]) for row in rows],
        "yaw": [float(row["yaw"]) for row in rows],
        "kappa": [float(row["kappa"]) for row in rows],
    }


def read_optional_reference_path(path):
    path = Path(path)
    if not path.exists():
        return None
    return read_path_csv(path)


def read_obstacles_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    obstacles = []
    for row in rows:
        center_x = float(row["center_x"])
        center_y = float(row["center_y"])
        length = float(row["length"])
        width = float(row["width"])
        yaw = float(row["yaw"])
        half_l = 0.5 * length
        half_w = 0.5 * width
        corners = []
        for local_x, local_y in [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]:
            world_x = center_x + local_x * math.cos(yaw) - local_y * math.sin(yaw)
            world_y = center_y + local_x * math.sin(yaw) + local_y * math.cos(yaw)
            corners.append((world_x, world_y))
        obstacles.append(
            {
                "id": int(row["id"]),
                "s": center_x,
                "l": center_y,
                "min_s": min(x for x, _ in corners),
                "max_s": max(x for x, _ in corners),
                "min_l": min(y for _, y in corners),
                "max_l": max(y for _, y in corners),
            }
        )
    return obstacles


def compute_safe_corridor(dp_path, obstacles, min_l, max_l, car_width, safe_distance):
    s_vals = dp_path["s"]
    l_vals = dp_path["y"]
    l_min = [min_l] * len(s_vals)
    l_max = [max_l] * len(s_vals)

    for obstacle in obstacles:
        if obstacle["min_s"] > s_vals[-1] or obstacle["max_s"] < s_vals[0]:
            continue

        index_min = 0
        for j, s_value in enumerate(s_vals):
            if obstacle["min_s"] <= s_value:
                index_min = j
                break
            if j == len(s_vals) - 1:
                index_min = j

        index_middle = index_min
        for k in range(index_min, len(s_vals)):
            if obstacle["s"] <= s_vals[k]:
                index_middle = k
                break
            if k == len(s_vals) - 1:
                index_middle = k

        index_max = index_middle
        for k in range(index_middle, len(s_vals)):
            if obstacle["max_s"] <= s_vals[k]:
                index_max = k
                break
            if k == len(s_vals) - 1:
                index_max = k

        obstacle["index_s_min"] = index_min
        obstacle["index_s_max"] = index_max
        obstacle["is_left_avoid"] = l_vals[index_middle] >= obstacle["l"]

    for obstacle in obstacles:
        if obstacle["min_s"] > s_vals[-1] or obstacle["max_s"] < s_vals[0]:
            continue

        if obstacle["is_left_avoid"]:
            distance = obstacle["max_l"] + 0.5 * car_width + safe_distance
            for j in range(obstacle["index_s_min"], obstacle["index_s_max"] + 1):
                if obstacle["max_l"] > l_min[j] and distance < l_max[j]:
                    l_min[j] = distance
        else:
            distance = obstacle["min_l"] - 0.5 * car_width - safe_distance
            for j in range(obstacle["index_s_min"], obstacle["index_s_max"] + 1):
                if obstacle["min_l"] < l_max[j] and distance > l_min[j]:
                    l_max[j] = distance

    start_l = l_vals[0]
    delta_s = s_vals[1] - s_vals[0] if len(s_vals) > 1 else 0.05
    if start_l < l_min[0]:
        l_min[0] = start_l - 0.01
        for i in range(1, len(l_min)):
            target = l_min[0] + (i - 2) * delta_s * 0.8
            if l_min[i] > target:
                l_min[i] = target
            else:
                break
    elif start_l > l_max[0]:
        l_max[0] = start_l + 0.01
        for i in range(1, len(l_max)):
            target = l_max[0] - (i - 2) * delta_s * 0.8
            if l_max[i] < target:
                l_max[i] = target
            else:
                break

    return l_min, l_max


def save_corridor_csv(path, dp_path, l_min, l_max):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "s", "dp_l", "l_min", "l_max", "corridor_width"])
        for index, s_value, l_value, low, high in zip(dp_path["index"], dp_path["s"], dp_path["y"], l_min, l_max):
            writer.writerow([index, s_value, l_value, low, high, high - low])


def plot_corridor(output_path, dp_path, qp_path, reference_path, obstacles, l_min, l_max, x_min, x_max, y_abs_max):
    serif_family = pick_font_family(["AR PL UMing CN", "SimSun", "Songti SC", "STSong"])
    sans_family = pick_font_family(["Noto Sans CJK SC"])
    font_family = [name for name in [serif_family, sans_family, "DejaVu Serif", "DejaVu Sans"] if name]
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.fill_between(dp_path["s"], l_min, l_max, color="#c6d9f1", alpha=0.75, zorder=1, label="安全走廊")
    ax.plot(dp_path["s"], l_min, color="#1f4e79", linewidth=1.6, zorder=2)
    ax.plot(dp_path["s"], l_max, color="#1f4e79", linewidth=1.6, zorder=2)
    if reference_path is not None:
        ax.plot(
            reference_path["s"],
            reference_path["y"],
            color="#4c566a",
            linestyle="--",
            linewidth=2.0,
            zorder=3.6,
            label="全局路径规划",
        )
    obstacle_labeled = False
    for obstacle in obstacles:
        left = obstacle["min_s"]
        width = obstacle["max_s"] - obstacle["min_s"]
        bottom = obstacle["min_l"]
        height = obstacle["max_l"] - obstacle["min_l"]
        ax.add_patch(
            plt.Rectangle(
                (left, bottom),
                width,
                height,
                facecolor="#9aa0a6",
                edgecolor="#5f6368",
                linewidth=0.9,
                alpha=0.30,
                zorder=3,
                label="障碍物" if not obstacle_labeled else None,
            )
        )
        obstacle_labeled = True
    ax.plot(qp_path["s"], qp_path["y"], color="#d04a02", linewidth=3.4, zorder=5, label="优化路径")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_abs_max, y_abs_max)
    ax.set_xlabel("s [m]")
    ax.set_ylabel("l [m]")
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    fig.savefig(output_path, dpi=330)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot safe corridor for one frame.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--min-l", type=float, default=-3.5)
    parser.add_argument("--max-l", type=float, default=3.5)
    parser.add_argument("--car-width", type=float, default=0.75)
    parser.add_argument("--safe-distance", type=float, default=0.2)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--y-abs-max", type=float, default=3.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    dp_path = read_path_csv(input_dir / "dp_path.csv")
    qp_path = read_path_csv(input_dir / "qp_path.csv")
    reference_path = read_optional_reference_path(input_dir / "reference_path.csv")
    obstacles = read_obstacles_csv(input_dir / "obstacles.csv")
    l_min, l_max = compute_safe_corridor(
        dp_path,
        obstacles,
        min_l=args.min_l,
        max_l=args.max_l,
        car_width=args.car_width,
        safe_distance=args.safe_distance,
    )
    save_corridor_csv(Path(args.csv_output), dp_path, l_min, l_max)
    plot_corridor(
        Path(args.output),
        dp_path,
        qp_path,
        reference_path,
        obstacles,
        l_min,
        l_max,
        args.x_min,
        args.x_max,
        args.y_abs_max,
    )
    print(f"saved_png={args.output}")
    print(f"saved_csv={args.csv_output}")


if __name__ == "__main__":
    main()
