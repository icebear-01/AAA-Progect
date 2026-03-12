#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MultipleLocator
except ImportError as exc:
    print(f"matplotlib import failed: {exc}", file=sys.stderr)
    sys.exit(2)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_path_csv(path):
    rows = read_csv_rows(path)
    return {
        "s": [float(row["s"]) for row in rows],
        "x": [float(row["x"]) for row in rows],
        "y": [float(row["y"]) for row in rows],
        "kappa": [float(row["kappa"]) for row in rows],
    }


def read_summary_txt(path):
    summary = {}
    if not os.path.exists(path):
        return summary
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            summary[key.strip()] = value.strip()
    return summary


def read_obstacles_csv(path):
    rows = read_csv_rows(path)
    obstacles = []
    for row in rows:
        obstacles.append(
            {
                "id": int(row["id"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "length": float(row["length"]),
                "width": float(row["width"]),
                "yaw": float(row["yaw"]),
                "x_vel": float(row.get("x_vel", 0.0) or 0.0),
                "y_vel": float(row.get("y_vel", 0.0) or 0.0),
            }
        )
    return obstacles


def read_speed_profile_csv(path):
    rows = read_csv_rows(path)
    return {
        "t": [float(row["t"]) for row in rows],
        "s": [float(row.get("s", 0.0) or 0.0) for row in rows],
        "v": [float(row.get("v", 0.0) or 0.0) for row in rows],
        "a": [float(row.get("a", 0.0) or 0.0) for row in rows],
    }


def read_st_dp_path_csv(path):
    rows = read_csv_rows(path)
    return {
        "t": [float(row["t"]) for row in rows],
        "s": [float(row["s"]) for row in rows],
        "is_feasible": [int(row.get("is_feasible", "1")) != 0 for row in rows],
    }


def read_st_lattice_csv(path):
    rows = read_csv_rows(path)
    return {
        "t": [float(row["t"]) for row in rows],
        "s": [float(row["s"]) for row in rows],
        "col": [int(row["col"]) for row in rows] if rows and "col" in rows[0] else [],
        "row": [int(row["row"]) for row in rows] if rows and "row" in rows[0] else [],
    }


def weighted_moving_average(values, weights=None):
    if not values:
        return []
    if weights is None:
        weights = [1.0, 2.0, 3.0, 2.0, 1.0]
    radius = len(weights) // 2
    smoothed = []
    for index in range(len(values)):
        weighted_sum = 0.0
        weight_sum = 0.0
        for offset, weight in enumerate(weights):
            sample_index = index + offset - radius
            sample_index = min(max(sample_index, 0), len(values) - 1)
            weighted_sum += weight * values[sample_index]
            weight_sum += weight
        smoothed.append(weighted_sum / weight_sum if weight_sum > 1e-9 else values[index])
    return smoothed


def integrate_speed(t_values, v_values, s0):
    if not t_values:
        return []
    s_values = [s0]
    for index in range(1, len(t_values)):
        dt = t_values[index] - t_values[index - 1]
        s_values.append(s_values[-1] + 0.5 * (v_values[index - 1] + v_values[index]) * dt)
    return s_values


def build_smoothed_speed_profile(speed_profile):
    if speed_profile is None or len(speed_profile.get("t", [])) < 2:
        return speed_profile
    v_smooth = weighted_moving_average(speed_profile["v"])
    v_smooth[0] = speed_profile["v"][0]
    s_smooth = integrate_speed(speed_profile["t"], v_smooth, speed_profile["s"][0])
    profile = dict(speed_profile)
    profile["s"] = s_smooth
    profile["v"] = v_smooth
    return profile


def read_st_obstacles_csv(path):
    rows = read_csv_rows(path)
    obstacles = []
    for row in rows:
        corners = []
        for idx in range(4):
            s_key = f"corner{idx}_s"
            l_key = f"corner{idx}_l"
            if s_key in row and l_key in row and row[s_key] != "" and row[l_key] != "":
                corners.append((float(row[s_key]), float(row[l_key])))
        obstacles.append(
            {
                "id": int(row["id"]),
                "is_consider": int(row["is_consider"]) != 0,
                "is_dynamic": int(row.get("is_dynamic", 0) or 0) != 0,
                "min_s": float(row["min_s"]),
                "max_s": float(row["max_s"]),
                "min_l": float(row.get("min_l", 0.0) or 0.0),
                "max_l": float(row.get("max_l", 0.0) or 0.0),
                "s_vel": float(row["s_vel"]),
                "l_vel": float(row.get("l_vel", 0.0) or 0.0),
                "t_in": float(row["t_in"]),
                "t_out": float(row["t_out"]),
                "s_in": float(row.get("s_in", row["min_s"])),
                "s_out": float(row.get("s_out", row["max_s"])),
                "corners": corners,
            }
        )
    return obstacles


def read_grid_csv(path):
    rows = read_csv_rows(path)
    return {
        "x": [float(row["x"]) for row in rows],
        "y": [float(row["y"]) for row in rows],
        "s": [float(row["s"]) for row in rows],
        "l": [float(row["l"]) for row in rows],
        "col": [int(row["col"]) for row in rows],
        "row": [int(row["row"]) for row in rows],
    }


def thin_st_lattice_points(st_lattice, row_stride=1, col_stride=1, keep_points=None):
    if st_lattice is None or not st_lattice.get("t"):
        return st_lattice
    if row_stride <= 1 and col_stride <= 1:
        return st_lattice

    rows = st_lattice.get("row") or []
    cols = st_lattice.get("col") or []
    if len(rows) != len(st_lattice["t"]) or len(cols) != len(st_lattice["t"]):
        return st_lattice

    keep_set = set()
    if keep_points is not None and keep_points.get("t") and keep_points.get("s"):
        keep_set = {
            (round(t_value, 6), round(s_value, 6))
            for t_value, s_value in zip(keep_points["t"], keep_points["s"])
        }

    filtered = {"t": [], "s": [], "row": [], "col": []}
    for t_value, s_value, row_value, col_value in zip(
        st_lattice["t"], st_lattice["s"], rows, cols
    ):
        point_key = (round(t_value, 6), round(s_value, 6))
        if point_key in keep_set:
            filtered["t"].append(t_value)
            filtered["s"].append(s_value)
            filtered["row"].append(row_value)
            filtered["col"].append(col_value)
            continue
        if row_stride > 1 and row_value % row_stride != 0:
            continue
        if col_stride > 1 and col_value % col_stride != 0:
            continue
        filtered["t"].append(t_value)
        filtered["s"].append(s_value)
        filtered["row"].append(row_value)
        filtered["col"].append(col_value)
    return filtered


def exclude_st_keep_points(st_lattice, keep_points=None):
    if st_lattice is None or not st_lattice.get("t"):
        return st_lattice
    if keep_points is None or not keep_points.get("t") or not keep_points.get("s"):
        return st_lattice

    keep_set = {
        (round(t_value, 6), round(s_value, 6))
        for t_value, s_value in zip(keep_points["t"], keep_points["s"])
    }
    filtered = {"t": [], "s": [], "row": [], "col": []}
    rows = st_lattice.get("row") or [0] * len(st_lattice["t"])
    cols = st_lattice.get("col") or [0] * len(st_lattice["t"])
    for t_value, s_value, row_value, col_value in zip(st_lattice["t"], st_lattice["s"], rows, cols):
        point_key = (round(t_value, 6), round(s_value, 6))
        if point_key in keep_set:
            continue
        filtered["t"].append(t_value)
        filtered["s"].append(s_value)
        filtered["row"].append(row_value)
        filtered["col"].append(col_value)
    return filtered


def split_feasible_dp_path(st_dp_path):
    feasible_t = []
    feasible_s = []
    fallback_t = []
    fallback_s = []
    flags = st_dp_path.get("is_feasible", [True] * len(st_dp_path.get("t", [])))
    for t_value, s_value, is_feasible in zip(st_dp_path.get("t", []), st_dp_path.get("s", []), flags):
        if is_feasible:
            feasible_t.append(t_value)
            feasible_s.append(s_value)
        else:
            fallback_t.append(t_value)
            fallback_s.append(s_value)
    if feasible_t and fallback_t:
        fallback_t = [feasible_t[-1]] + fallback_t
        fallback_s = [feasible_s[-1]] + fallback_s
    return feasible_t, feasible_s, fallback_t, fallback_s


def compute_st_row_stride(st_s_min_step, paper):
    if st_s_min_step <= 1e-9:
        return 1
    target_display_step = 0.20 if paper else 0.10
    return max(1, int(round(target_display_step / st_s_min_step)))


def obstacle_polygon(obstacle):
    cx = obstacle["center_x"]
    cy = obstacle["center_y"]
    length = obstacle["length"]
    width = obstacle["width"]
    yaw = obstacle["yaw"]
    half_l = 0.5 * length
    half_w = 0.5 * width
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    dx = half_l * cos_yaw
    dy = half_l * sin_yaw
    wx = -half_w * sin_yaw
    wy = half_w * cos_yaw
    corners = [
        (cx + dx + wx, cy + dy + wy),
        (cx + dx - wx, cy + dy - wy),
        (cx - dx - wx, cy - dy - wy),
        (cx - dx + wx, cy - dy + wy),
    ]
    corners.append(corners[0])
    return corners


def offset_reference_path(reference, offset):
    if reference is None or not reference.get("x") or len(reference["x"]) < 2:
        return None
    xs = reference["x"]
    ys = reference["y"]
    out_x = []
    out_y = []
    last_index = len(xs) - 1
    for index in range(len(xs)):
        prev_index = max(0, index - 1)
        next_index = min(last_index, index + 1)
        dx = xs[next_index] - xs[prev_index]
        dy = ys[next_index] - ys[prev_index]
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            out_x.append(xs[index])
            out_y.append(ys[index])
            continue
        nx = -dy / norm
        ny = dx / norm
        out_x.append(xs[index] + nx * offset)
        out_y.append(ys[index] + ny * offset)
    return {"x": out_x, "y": out_y}


def st_obstacle_polygon(obstacle):
    t_in = obstacle["t_in"]
    t_out = obstacle["t_out"]
    if t_out < t_in:
        t_in, t_out = t_out, t_in
    lower = min(obstacle["s_in"], obstacle["s_out"])
    upper = max(obstacle["s_in"], obstacle["s_out"])
    return [(t_in, lower), (t_out, lower), (t_out, upper), (t_in, upper)]


def unique_sorted(values, tol=1e-6):
    ordered = sorted(values)
    unique = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > tol:
            unique.append(value)
    return unique


def obstacle_sl_corners(obstacle):
    if obstacle.get("corners"):
        return list(obstacle["corners"])
    return [
        (obstacle["min_s"], obstacle["min_l"]),
        (obstacle["max_s"], obstacle["min_l"]),
        (obstacle["max_s"], obstacle["max_l"]),
        (obstacle["min_s"], obstacle["max_l"]),
    ]


def interpolate_at_l(point_a, point_b, target_l):
    ds = point_b[0] - point_a[0]
    dl = point_b[1] - point_a[1]
    if abs(dl) < 1e-9:
        return (point_a[0] + 0.5 * ds, target_l)
    ratio = (target_l - point_a[1]) / dl
    return (point_a[0] + ratio * ds, target_l)


def clip_polygon_lower_l(polygon, lower_l):
    if not polygon:
        return []
    output = []
    prev = polygon[-1]
    prev_inside = prev[1] >= lower_l - 1e-9
    for curr in polygon:
        curr_inside = curr[1] >= lower_l - 1e-9
        if curr_inside != prev_inside:
            output.append(interpolate_at_l(prev, curr, lower_l))
        if curr_inside:
            output.append(curr)
        prev = curr
        prev_inside = curr_inside
    return output


def clip_polygon_upper_l(polygon, upper_l):
    if not polygon:
        return []
    output = []
    prev = polygon[-1]
    prev_inside = prev[1] <= upper_l + 1e-9
    for curr in polygon:
        curr_inside = curr[1] <= upper_l + 1e-9
        if curr_inside != prev_inside:
            output.append(interpolate_at_l(prev, curr, upper_l))
        if curr_inside:
            output.append(curr)
        prev = curr
        prev_inside = curr_inside
    return output


def clip_polygon_to_lateral_band(polygon, lateral_limit):
    clipped = clip_polygon_lower_l(polygon, -lateral_limit)
    clipped = clip_polygon_upper_l(clipped, lateral_limit)
    return clipped


def translated_obstacle_polygon(obstacle, time_s):
    translated = []
    for corner_s, corner_l in obstacle_sl_corners(obstacle):
        translated.append(
            (corner_s + obstacle["s_vel"] * time_s, corner_l + obstacle["l_vel"] * time_s)
        )
    return translated


def obstacle_full_s_range_at_time(obstacle, time_s):
    polygon = translated_obstacle_polygon(obstacle, time_s)
    s_values = [point[0] for point in polygon]
    return (min(s_values), max(s_values))


def obstacle_active_time_window(obstacle, lateral_limit, plan_time):
    if not obstacle.get("is_dynamic", False):
        return None
    corners = obstacle_sl_corners(obstacle)
    min_l = min(point[1] for point in corners)
    max_l = max(point[1] for point in corners)
    l_vel = obstacle["l_vel"]
    if abs(l_vel) < 1e-9:
        if max_l < -lateral_limit or min_l > lateral_limit:
            return None
        return (0.0, plan_time)
    t_first = (lateral_limit - min_l) / l_vel
    t_last = (-lateral_limit - max_l) / l_vel
    t_in = max(0.0, min(t_first, t_last))
    t_out = min(plan_time, max(t_first, t_last))
    if t_out < t_in - 1e-9:
        return None
    return (t_in, t_out)


def obstacle_s_range_at_time(obstacle, time_s, lateral_limit):
    clipped = clip_polygon_to_lateral_band(translated_obstacle_polygon(obstacle, time_s), lateral_limit)
    if not clipped:
        return None
    s_values = [point[0] for point in clipped]
    return (min(s_values), max(s_values))


def build_st_occupancy_polygon(obstacle, lateral_limit, plan_time):
    active_window = obstacle_active_time_window(obstacle, lateral_limit, plan_time)
    if active_window is None:
        return None
    corners = obstacle_sl_corners(obstacle)
    if abs(obstacle["l_vel"]) < 1e-9:
        t_in, t_out = active_window
        s_range_in = obstacle_full_s_range_at_time(obstacle, t_in)
        s_range_out = obstacle_full_s_range_at_time(obstacle, t_out)
        return [
            (t_in, s_range_in[1]),
            (t_out, s_range_out[1]),
            (t_out, s_range_out[0]),
            (t_in, s_range_in[0]),
        ]

    if obstacle["l_vel"] > 0.0:
        entry_boundary = -lateral_limit
        exit_boundary = lateral_limit
        entry_corners = sorted(corners, key=lambda point: point[1], reverse=True)[:2]
        exit_corners = sorted(corners, key=lambda point: point[1])[:2]
    else:
        entry_boundary = lateral_limit
        exit_boundary = -lateral_limit
        entry_corners = sorted(corners, key=lambda point: point[1])[:2]
        exit_corners = sorted(corners, key=lambda point: point[1], reverse=True)[:2]

    def crossing_point(corner, boundary_l):
        raw_t = (boundary_l - corner[1]) / obstacle["l_vel"]
        clipped_t = min(max(raw_t, 0.0), plan_time)
        return (clipped_t, corner[0] + obstacle["s_vel"] * clipped_t)

    entry_points = [crossing_point(corner, entry_boundary) for corner in entry_corners]
    exit_points = [crossing_point(corner, exit_boundary) for corner in exit_corners]
    entry_points.sort(key=lambda point: point[1], reverse=True)
    exit_points.sort(key=lambda point: point[1], reverse=True)
    return [
        entry_points[0],
        exit_points[0],
        exit_points[1],
        entry_points[1],
    ]


def max_abs(values):
    return max((abs(v) for v in values), default=0.0)


def mean_abs(values):
    finite = [abs(v) for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else 0.0


def pick_font_family(candidates):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def apply_paper_style():
    serif_family = pick_font_family(["AR PL UMing CN", "SimSun", "Songti SC", "STSong"])
    sans_family = pick_font_family(["Noto Sans CJK SC"])
    font_family = [name for name in [serif_family, sans_family, "DejaVu Serif", "DejaVu Sans"] if name]
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def build_text_font_properties():
    font_family = pick_font_family(
        [
            "AR PL UMing CN",
            "SimSun",
            "Songti SC",
            "STSong",
            "Noto Sans CJK SC",
        ]
    )
    if not font_family:
        return None
    return font_manager.FontProperties(family=font_family)


def add_panel_label(ax, text):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )


def style_axes(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def find_available_scenarios(input_dir):
    benchmark_root = Path(input_dir).resolve().parent
    if not benchmark_root.is_dir():
        return []
    scenario_names = []
    for child in sorted(benchmark_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "dp_path.csv").exists() and (child / "qp_path.csv").exists():
            scenario_names.append(child.name)
    return scenario_names


def ensure_required_inputs(input_dir, dp_path, qp_path):
    missing_files = [path for path in [dp_path, qp_path] if not os.path.exists(path)]
    if not missing_files:
        return

    print("plot_compare.py input error:", file=sys.stderr)
    print(f"  input-dir: {input_dir}", file=sys.stderr)
    for path in missing_files:
        print(f"  missing: {path}", file=sys.stderr)

    scenarios = find_available_scenarios(input_dir)
    if scenarios:
        print("  available scenarios:", file=sys.stderr)
        for name in scenarios[:12]:
            print(f"    - {name}", file=sys.stderr)
    else:
        print("  no valid benchmark result directories found nearby.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot EMPlanner DP/QP comparison.")
    parser.add_argument("--input-dir", required=True, help="Directory containing dp_path.csv and qp_path.csv")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--paper", action="store_true", help="Use publication-style layout")
    parser.add_argument("--export-pdf", action="store_true", help="Also export a PDF with the same stem")
    parser.add_argument("--show-grid", dest="show_grid", action="store_true", help="Overlay DP lattice points on the path subplot")
    parser.add_argument("--hide-grid", dest="show_grid", action="store_false", help="Hide DP lattice points on the path subplot")
    parser.add_argument("--right-panel", choices=["curvature", "st"], default="curvature", help="Right subplot content")
    parser.add_argument("--path-x-min", type=float, default=0.0, help="Path subplot x-axis minimum")
    parser.add_argument("--path-x-max", type=float, default=9.0, help="Path subplot x-axis maximum")
    parser.add_argument("--path-y-abs-max", type=float, default=3.5, help="Absolute y-axis limit for the path subplot")
    parser.set_defaults(show_grid=True)
    args = parser.parse_args()

    dp_path = os.path.join(args.input_dir, "dp_path.csv")
    qp_path = os.path.join(args.input_dir, "qp_path.csv")
    obstacles_path = os.path.join(args.input_dir, "obstacles.csv")
    reference_path = os.path.join(args.input_dir, "reference_path.csv")
    grid_path = os.path.join(args.input_dir, "dp_grid_points.csv")
    speed_profile_path = os.path.join(args.input_dir, "speed_profile.csv")
    st_dp_path_path = os.path.join(args.input_dir, "st_dp_path.csv")
    st_lattice_path = os.path.join(args.input_dir, "st_lattice.csv")
    st_obstacles_path = os.path.join(args.input_dir, "st_obstacles.csv")
    summary_path = os.path.join(args.input_dir, "summary.txt")
    output_path = args.output or os.path.join(
        args.input_dir, "comparison_paper.png" if args.paper else "comparison.png"
    )
    ensure_required_inputs(args.input_dir, dp_path, qp_path)

    dp = read_path_csv(dp_path)
    qp = read_path_csv(qp_path)
    reference = read_path_csv(reference_path) if os.path.exists(reference_path) else None
    grid = read_grid_csv(grid_path) if args.show_grid and os.path.exists(grid_path) else None
    obstacles = read_obstacles_csv(obstacles_path) if os.path.exists(obstacles_path) else []
    speed_profile = read_speed_profile_csv(speed_profile_path) if os.path.exists(speed_profile_path) else None
    speed_profile_smoothed = build_smoothed_speed_profile(speed_profile) if speed_profile is not None else None
    st_dp_path = read_st_dp_path_csv(st_dp_path_path) if os.path.exists(st_dp_path_path) else None
    st_lattice = read_st_lattice_csv(st_lattice_path) if args.show_grid and os.path.exists(st_lattice_path) else None
    st_obstacles = read_st_obstacles_csv(st_obstacles_path) if os.path.exists(st_obstacles_path) else []
    summary = read_summary_txt(summary_path)
    st_time_step = float(summary.get("speed_plan_t_dt", 0.25) or 0.25)
    st_lateral_limit = float(summary.get("st_lateral_limit", 1.0) or 1.0)
    st_s_min_step = float(summary.get("st_s_min_step", 0.05) or 0.05)
    speed_reference = float(summary.get("speed_reference", 0.0) or 0.0)
    st_plan_time = 0.0
    if speed_profile is not None and speed_profile["t"]:
        st_plan_time = max(speed_profile["t"])
    st_obstacles_by_id = {obstacle["id"]: obstacle for obstacle in st_obstacles}
    st_lattice_plot = thin_st_lattice_points(
        st_lattice,
        row_stride=compute_st_row_stride(st_s_min_step, args.paper),
        col_stride=1,
        keep_points=st_dp_path,
    ) if st_lattice is not None else None
    if args.right_panel == "st" and speed_profile is None:
        print("plot_compare.py input error:", file=sys.stderr)
        print(f"  input-dir: {args.input_dir}", file=sys.stderr)
        print(f"  missing: {speed_profile_path}", file=sys.stderr)
        print("  hint: rerun benchmark with plan_iterations:=1.", file=sys.stderr)
        sys.exit(1)

    if args.paper:
        apply_paper_style()
    text_font = build_text_font_properties()

    dp_source = summary.get("dp_source", "classic")
    dp_label = "RL-DP路径" if dp_source == "RL_DP" else "决策路径"  
    dp_curvature_label = "决策路径曲率" if dp_source == "RL_DP" else "决策路径曲率"
    qp_label = "优化路径"
    qp_curvature_label = "优化后曲率"   
    reference_label = "参考线"
    intrusion_label = None
    if args.right_panel == "st":
        dp_label = None
        qp_label = "局部路径"
        reference_label = None
        intrusion_label = "侵入参考线"

    fig_width = 7.2 if args.paper else 14.0
    fig_height = 3.2 if args.paper else 6.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=args.paper)

    ax_path = axes[0]
    if args.right_panel != "st" and grid is not None and grid["x"]:
        grid_marker_size = 24 if args.paper else 32
        grid_marker_linewidth = 1.25 if args.paper else 1.5
        ax_path.scatter(
            grid["x"],
            grid["y"],
            s=grid_marker_size,
            facecolors="none",
            edgecolors="#8a939c",
            linewidths=grid_marker_linewidth,
            alpha=0.95,
            label="采样栅格点",
            zorder=1,
        )
    if reference is not None:
        ax_path.plot(
            reference["x"],
            reference["y"],
            label=reference_label,
            linewidth=1.3 if args.paper else 1.6,
            color="#4c566a",
            linestyle="--",
        )
    if args.right_panel == "st" and reference is not None and st_lateral_limit > 1e-6:
        upper_intrusion = offset_reference_path(reference, st_lateral_limit)
        lower_intrusion = offset_reference_path(reference, -st_lateral_limit)
        if upper_intrusion is not None:
            ax_path.plot(
                upper_intrusion["x"],
                upper_intrusion["y"],
                linewidth=1.0 if args.paper else 1.2,
                color="#7b8794",
                linestyle=(0, (4, 3)),
                alpha=0.9,
                label=intrusion_label,
                zorder=1.5,
            )
        if lower_intrusion is not None:
            ax_path.plot(
                lower_intrusion["x"],
                lower_intrusion["y"],
                linewidth=1.0 if args.paper else 1.2,
                color="#7b8794",
                linestyle=(0, (4, 3)),
                alpha=0.9,
                zorder=1.5,
            )
    if dp_label is not None:
        ax_path.plot(dp["x"], dp["y"], label=dp_label, linewidth=2.1, color="#1f5aa6", zorder=3)
    ax_path.plot(qp["x"], qp["y"], label=qp_label, linewidth=2.1, color="#d04a02", zorder=4)
    obstacle_motion_horizon = 0.0
    if speed_profile is not None and speed_profile["t"]:
        obstacle_motion_horizon = max(speed_profile["t"])
    obstacle_body_label_used = False
    for obstacle in obstacles:
        poly = obstacle_polygon(obstacle)
        xs = [point[0] for point in poly]
        ys = [point[1] for point in poly]
        ax_path.fill(
            xs,
            ys,
            color="#9aa0a6",
            alpha=0.35,
            zorder=2,
            label="障碍物" if args.right_panel == "st" and not obstacle_body_label_used else None,
        )
        ax_path.plot(xs, ys, color="#5f6368", linewidth=0.9, zorder=2)
        if args.right_panel == "st" and not obstacle_body_label_used:
            obstacle_body_label_used = True
        if obstacle_motion_horizon > 0.0 and (
            abs(obstacle["x_vel"]) > 1e-6 or abs(obstacle["y_vel"]) > 1e-6
        ):
            arrow_start_x = obstacle["center_x"]
            arrow_start_y = obstacle["center_y"]
            arrow_alpha = 0.85
            arrow_time = obstacle_motion_horizon
            if args.right_panel == "st":
                st_obstacle = st_obstacles_by_id.get(obstacle["id"])
                if st_obstacle is not None and not st_obstacle["is_consider"]:
                    arrow_alpha = 0.30
                # ST论文图里只保留“当前障碍物 + 运动方向”示意，不再画 t_in->t_out 的断开箭头。
                # 箭头长度适度缩放，避免跨图过长。
                arrow_time = min(obstacle_motion_horizon, 2.0)
            arrow_end_x = obstacle["center_x"] + obstacle["x_vel"] * arrow_time
            arrow_end_y = obstacle["center_y"] + obstacle["y_vel"] * arrow_time
            ax_path.annotate(
                "",
                xy=(arrow_end_x, arrow_end_y),
                xytext=(arrow_start_x, arrow_start_y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#5f6368",
                    "linewidth": 1.25 if args.paper else 1.5,
                    "mutation_scale": 16 if args.paper else 20,
                    "alpha": arrow_alpha,
                },
                zorder=2,
            )
    ax_path.set_xlabel("x [m]", fontproperties=text_font)
    ax_path.set_ylabel("y [m]", fontproperties=text_font)
    ax_path.set_xlim(args.path_x_min, args.path_x_max)
    ax_path.set_ylim(-args.path_y_abs_max, args.path_y_abs_max)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.autoscale(enable=False)
    style_axes(ax_path)
    path_legend_kwargs = {"frameon": True, "framealpha": 0.92}
    if args.paper:
        path_legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.10),
                "ncol": 2,
                "columnspacing": 1.2,
                "handlelength": 2.0,
                "borderaxespad": 0.0,
            }
        )
    else:
        path_legend_kwargs["loc"] = "upper right"
    if text_font is not None:
        path_legend_kwargs["prop"] = text_font
    if args.right_panel == "st":
        path_legend_handles = [
            Line2D([], [], color="#7b8794", linewidth=1.0 if args.paper else 1.2, linestyle=(0, (4, 3)), label="侵入参考线"),
            Line2D([], [], color="#d04a02", linewidth=2.1, label="局部路径"),
            Rectangle(
                (0.0, 0.0),
                1.0,
                0.65,
                facecolor="#9aa0a6",
                edgecolor="#5f6368",
                linewidth=0.9,
                alpha=0.35,
                label="障碍物",
            ),
        ]
        ax_path.legend(handles=path_legend_handles, **path_legend_kwargs)
    else:
        ax_path.legend(**path_legend_kwargs)
    # if args.paper:
    #     add_panel_label(ax_path, "(a)")

    ax_kappa = axes[1]
    if args.right_panel == "st":
        if st_lattice_plot is not None and st_lattice_plot["t"]:
            ax_kappa.scatter(
                st_lattice_plot["t"],
                st_lattice_plot["s"],
                s=14 if args.paper else 24,
                facecolors="none",
                edgecolors="#8a939c",
                linewidths=0.9 if args.paper else 1.1,
                alpha=0.85,
                label="速度采样栅格",
                zorder=1,
            )
        obstacle_label_used = False
        for obstacle in st_obstacles:
            if not obstacle["is_dynamic"]:
                continue
            poly = build_st_occupancy_polygon(obstacle, st_lateral_limit, st_plan_time)
            if poly is None or len(poly) < 3:
                continue
            xs = [point[0] for point in poly] + [poly[0][0]]
            ys = [point[1] for point in poly] + [poly[0][1]]
            ax_kappa.fill(xs, ys, color="#9aa0a6", alpha=0.30, zorder=2)
            ax_kappa.plot(
                xs,
                ys,
                color="#4f545a",
                linewidth=1.2,
                zorder=2,
                label="障碍物速度占据" if not obstacle_label_used else None,
            )
            obstacle_label_used = True
        if st_dp_path is not None and st_dp_path["t"]:
            feasible_t, feasible_s, fallback_t, fallback_s = split_feasible_dp_path(st_dp_path)
            if feasible_t:
                ax_kappa.plot(
                    feasible_t,
                    feasible_s,
                    label="速度决策线",
                    linewidth=1.8,
                    color="#1f5aa6",
                    linestyle="--",
                    marker="o",
                    markersize=3.2 if args.paper else 4.2,
                    markerfacecolor="white",
                    markeredgecolor="#1f5aa6",
                    markeredgewidth=0.9 if args.paper else 1.0,
                    zorder=3,
                )
            if fallback_t:
                ax_kappa.plot(
                    fallback_t,
                    fallback_s,
                    linewidth=1.3,
                    color="#1f5aa6",
                    linestyle=":",
                    alpha=0.55,
                    zorder=2.8,
                )
        ax_kappa.plot(
            speed_profile_smoothed["t"],
            speed_profile_smoothed["s"],
            label="优化速度曲线",
            linewidth=2.1,
            color="#d04a02",
            zorder=4,
        )
        if speed_reference > 1e-6 and speed_profile_smoothed["t"]:
            ax_kappa.plot(
                speed_profile_smoothed["t"],
                [speed_reference * t_value for t_value in speed_profile_smoothed["t"]],
                label="参考速度线",
                linewidth=1.9,
                color="#2e8b57",
                linestyle="-.",
                alpha=0.95,
                zorder=5,
            )
        ax_kappa.set_xlabel("t [s]", fontproperties=text_font)
        ax_kappa.set_ylabel("s [m]", fontproperties=text_font)
        ax_kappa.xaxis.set_major_locator(MultipleLocator(st_time_step))
        for tick in ax_kappa.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        if args.paper:
            ax_kappa.tick_params(axis="x", labelsize=8)
    else:
        ax_kappa.plot(dp["s"], dp["kappa"], label=dp_curvature_label, linewidth=2.0, color="#1f5aa6")
        ax_kappa.plot(qp["s"], qp["kappa"], label=qp_curvature_label, linewidth=2.0, color="#d04a02")
        ax_kappa.set_xlabel("s [m]", fontproperties=text_font)
        ax_kappa.set_ylabel("kappa [1/m]", fontproperties=text_font)
    style_axes(ax_kappa)
    kappa_legend_kwargs = {"frameon": True, "framealpha": 0.92}
    if args.paper:
        kappa_legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.10),
                "ncol": 3 if args.right_panel == "st" else 2,
                "columnspacing": 1.2,
                "handlelength": 2.0,
                "borderaxespad": 0.0,
            }
        )
    else:
        kappa_legend_kwargs["loc"] = "upper right"
    if text_font is not None:
        kappa_legend_kwargs["prop"] = text_font
    if args.right_panel == "st":
        obstacle_legend_handle = Rectangle(
            (0.0, 0.0),
            1.0,
            0.52,
            facecolor="#9aa0a6",
            edgecolor="#4f545a",
            linewidth=1.2,
            alpha=0.30,
        )
        kappa_legend_handles = [
            obstacle_legend_handle,
            Line2D([], [], color="#1f5aa6", linewidth=1.8, linestyle="--", marker="o", markersize=3.2 if args.paper else 4.2, markerfacecolor="white", markeredgecolor="#1f5aa6", markeredgewidth=0.9 if args.paper else 1.0, label="速度决策线"),
            Line2D([], [], color="#d04a02", linewidth=2.1, label="优化速度曲线"),
            Line2D([], [], color="#2e8b57", linewidth=1.9, linestyle="-.", alpha=0.95, label="参考速度线"),
            Line2D([], [], linestyle="none", marker="o", markersize=5.0 if args.paper else 6.0, markerfacecolor="none", markeredgecolor="#8a939c", markeredgewidth=0.9 if args.paper else 1.1, alpha=0.85, label="速度采样栅格"),
        ]
        kappa_legend_labels = ["障碍物速度占据", "速度决策线", "优化速度曲线", "参考速度线", "速度采样栅格"]
        ax_kappa.legend(
            kappa_legend_handles,
            kappa_legend_labels,
            **kappa_legend_kwargs,
        )
    else:
        ax_kappa.legend(**kappa_legend_kwargs)
    # if args.paper:
    #     add_panel_label(ax_kappa, "(b)")
    if not args.paper:
        fig.tight_layout()
    fig.savefig(output_path, dpi=330 if args.paper else 330)  ##默认330dpi，论文图可以考虑更高一些
    pdf_path = None
    if args.export_pdf:
        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(pdf_path)
    print(f"saved_png={output_path}")
    if pdf_path is not None:
        print(f"saved_pdf={pdf_path}")


if __name__ == "__main__":
    main()
