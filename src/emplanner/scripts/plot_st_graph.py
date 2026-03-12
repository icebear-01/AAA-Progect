#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys

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

try:
    from scipy.interpolate import Akima1DInterpolator
except ImportError:
    Akima1DInterpolator = None


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def read_speed_profile_csv(path):
    rows = read_csv_rows(path)
    return {
        "t": [float(row["t"]) for row in rows],
        "s": [float(row["s"]) for row in rows],
        "v": [float(row["v"]) for row in rows],
        "a": [float(row["a"]) for row in rows],
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
        "col": [int(row["col"]) for row in rows],
        "row": [int(row["row"]) for row in rows],
    }


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


def thin_st_lattice_points(st_lattice, row_stride=1, col_stride=1, keep_points=None):
    if st_lattice is None or not st_lattice.get("t"):
        return st_lattice
    if row_stride <= 1 and col_stride <= 1:
        return st_lattice

    keep_set = set()
    if keep_points is not None and keep_points.get("t") and keep_points.get("s"):
        keep_set = {
            (round(t_value, 6), round(s_value, 6))
            for t_value, s_value in zip(keep_points["t"], keep_points["s"])
        }

    filtered = {"t": [], "s": [], "col": [], "row": []}
    for t_value, s_value, col_value, row_value in zip(
        st_lattice["t"], st_lattice["s"], st_lattice["col"], st_lattice["row"]
    ):
        point_key = (round(t_value, 6), round(s_value, 6))
        if point_key in keep_set:
            filtered["t"].append(t_value)
            filtered["s"].append(s_value)
            filtered["col"].append(col_value)
            filtered["row"].append(row_value)
            continue
        if row_stride > 1 and row_value % row_stride != 0:
            continue
        if col_stride > 1 and col_value % col_stride != 0:
            continue
        filtered["t"].append(t_value)
        filtered["s"].append(s_value)
        filtered["col"].append(col_value)
        filtered["row"].append(row_value)
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
    filtered = {"t": [], "s": [], "col": [], "row": []}
    cols = st_lattice.get("col") or [0] * len(st_lattice["t"])
    rows = st_lattice.get("row") or [0] * len(st_lattice["t"])
    for t_value, s_value, col_value, row_value in zip(st_lattice["t"], st_lattice["s"], cols, rows):
        point_key = (round(t_value, 6), round(s_value, 6))
        if point_key in keep_set:
            continue
        filtered["t"].append(t_value)
        filtered["s"].append(s_value)
        filtered["col"].append(col_value)
        filtered["row"].append(row_value)
    return filtered


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
                "s_in": float(row["s_in"]),
                "s_out": float(row["s_out"]),
                "corners": corners,
            }
        )
    return obstacles


def compute_st_row_stride(st_s_min_step, paper):
    if st_s_min_step <= 1e-9:
        return 1
    target_display_step = 0.20 if paper else 0.10
    return max(1, int(round(target_display_step / st_s_min_step)))


def weighted_moving_average(values, weights=None, passes=1):
    if not values:
        return []
    if weights is None:
        weights = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
    output = list(values)
    radius = len(weights) // 2
    for _ in range(max(1, passes)):
        smoothed = []
        for index in range(len(output)):
            weighted_sum = 0.0
            weight_sum = 0.0
            for offset, weight in enumerate(weights):
                sample_index = index + offset - radius
                sample_index = min(max(sample_index, 0), len(output) - 1)
                weighted_sum += weight * output[sample_index]
                weight_sum += weight
            smoothed.append(weighted_sum / weight_sum if weight_sum > 1e-9 else output[index])
        output = smoothed
    return output


def derivative(values, t_values):
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    output = []
    for index in range(len(values)):
        if index == 0:
            dt = t_values[1] - t_values[0]
            output.append((values[1] - values[0]) / dt if abs(dt) > 1e-9 else 0.0)
        elif index == len(values) - 1:
            dt = t_values[-1] - t_values[-2]
            output.append((values[-1] - values[-2]) / dt if abs(dt) > 1e-9 else 0.0)
        else:
            dt = t_values[index + 1] - t_values[index - 1]
            output.append((values[index + 1] - values[index - 1]) / dt if abs(dt) > 1e-9 else 0.0)
    return output


def integrate_speed(t_values, v_values, s0):
    if not t_values:
        return []
    s_values = [s0]
    for index in range(1, len(t_values)):
        dt = t_values[index] - t_values[index - 1]
        s_values.append(s_values[-1] + 0.5 * (v_values[index - 1] + v_values[index]) * dt)
    return s_values


def monotone_cubic_curve(x_values, y_values, samples_per_segment=24):
    if not x_values or len(x_values) != len(y_values):
        return x_values, y_values
    if len(x_values) < 3:
        return x_values, y_values

    n = len(x_values)
    h = [x_values[i + 1] - x_values[i] for i in range(n - 1)]
    if any(abs(step) < 1e-9 for step in h):
        return x_values, y_values
    delta = [(y_values[i + 1] - y_values[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0] = delta[0]
    m[-1] = delta[-1]

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    smooth_x = [x_values[0]]
    smooth_y = [y_values[0]]
    for i in range(n - 1):
        x0 = x_values[i]
        x1 = x_values[i + 1]
        y0 = y_values[i]
        y1 = y_values[i + 1]
        step = x1 - x0
        for sample_index in range(1, samples_per_segment + 1):
            tau = sample_index / samples_per_segment
            h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
            h10 = tau**3 - 2.0 * tau**2 + tau
            h01 = -2.0 * tau**3 + 3.0 * tau**2
            h11 = tau**3 - tau**2
            smooth_x.append(x0 + tau * step)
            smooth_y.append(
                h00 * y0
                + h10 * step * m[i]
                + h01 * y1
                + h11 * step * m[i + 1]
            )
    return smooth_x, smooth_y


def smooth_curve_for_display(x_values, y_values, samples_per_segment=48):
    if not x_values or len(x_values) != len(y_values):
        return x_values, y_values
    if len(x_values) < 3:
        return x_values, y_values

    sample_count = max(2, (len(x_values) - 1) * max(1, samples_per_segment) + 1)
    if Akima1DInterpolator is not None and len(x_values) >= 5:
        x_start = x_values[0]
        x_end = x_values[-1]
        step = (x_end - x_start) / (sample_count - 1)
        dense_x = [x_start + step * index for index in range(sample_count)]
        dense_y = Akima1DInterpolator(x_values, y_values)(dense_x)
        return dense_x, [float(value) for value in dense_y]

    return monotone_cubic_curve(x_values, y_values, samples_per_segment=samples_per_segment)


def build_smoothed_speed_profile(speed_profile):
    if speed_profile is None or len(speed_profile.get("t", [])) < 2:
        return speed_profile
    t_values = list(speed_profile["t"])
    v_raw = list(speed_profile["v"])
    a_raw = list(speed_profile["a"])
    s_raw = list(speed_profile["s"])

    v_smooth = weighted_moving_average(v_raw, passes=2)
    v_smooth[0] = v_raw[0]
    v_smooth[-1] = v_raw[-1]
    s_smooth = integrate_speed(t_values, v_smooth, s_raw[0])
    a_from_v = derivative(v_smooth, t_values)
    a_smooth = weighted_moving_average(a_from_v, passes=2)
    a_smooth[0] = a_from_v[0]
    a_smooth[-1] = a_from_v[-1]
    jerk = derivative(a_smooth, t_values)
    return {
        "t": t_values,
        "s": s_smooth,
        "v": v_smooth,
        "a": a_smooth,
        "jerk": jerk,
        "s_raw": s_raw,
        "v_raw": v_raw,
        "a_raw": a_raw,
    }


def build_dp_speed_profile(st_dp_path, initial_speed=None):
    if st_dp_path is None or len(st_dp_path.get("t", [])) < 2:
        return None
    t_values = list(st_dp_path["t"])
    s_values = list(st_dp_path["s"])
    v_values = derivative(s_values, t_values)
    if initial_speed is not None and v_values:
        # The DP speed plot is reconstructed from discrete ST points; anchor the
        # first sample to the same initial condition used by the planner/QP curve.
        v_values[0] = float(initial_speed)
    return {
        "t": t_values,
        "s": s_values,
        "v": v_values,
        "is_feasible": list(st_dp_path.get("is_feasible", [True] * len(t_values))),
    }


def write_smoothed_speed_profile_csv(path, smoothed_profile):
    if smoothed_profile is None or not smoothed_profile.get("t"):
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "t",
                "s_raw",
                "s_smooth",
                "v_raw",
                "v_smooth",
                "a_raw",
                "a_smooth",
                "jerk",
            ]
        )
        for index, (t_value, s_raw, s_smooth, v_raw, v_smooth, a_raw, a_smooth, jerk) in enumerate(
            zip(
                smoothed_profile["t"],
                smoothed_profile["s_raw"],
                smoothed_profile["s"],
                smoothed_profile["v_raw"],
                smoothed_profile["v"],
                smoothed_profile["a_raw"],
                smoothed_profile["a"],
                smoothed_profile["jerk"],
            )
        ):
            writer.writerow(
                [
                    index,
                    f"{t_value:.6f}",
                    f"{s_raw:.6f}",
                    f"{s_smooth:.6f}",
                    f"{v_raw:.6f}",
                    f"{v_smooth:.6f}",
                    f"{a_raw:.6f}",
                    f"{a_smooth:.6f}",
                    f"{jerk:.6f}",
                ]
            )


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


def style_axes(ax):
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def obstacle_polygon(obstacle):
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


def ensure_required_inputs(input_dir, speed_profile_path):
    if os.path.exists(speed_profile_path):
        return
    print("plot_st_graph.py input error:", file=sys.stderr)
    print(f"  input-dir: {input_dir}", file=sys.stderr)
    print(f"  missing: {speed_profile_path}", file=sys.stderr)
    print("  hint: rerun benchmark with plan_iterations:=1 to export speed planning data.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot EMPlanner ST speed-planning results.")
    parser.add_argument("--input-dir", required=True, help="Directory containing speed_profile.csv")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--paper", action="store_true", help="Use publication-style layout")
    parser.add_argument("--export-pdf", action="store_true", help="Also export a PDF with the same stem")
    parser.add_argument("--show-grid", dest="show_grid", action="store_true", help="Overlay ST lattice points")
    parser.add_argument("--hide-grid", dest="show_grid", action="store_false", help="Hide ST lattice points")
    parser.set_defaults(show_grid=True)
    args = parser.parse_args()

    speed_profile_path = os.path.join(args.input_dir, "speed_profile.csv")
    st_dp_path_path = os.path.join(args.input_dir, "st_dp_path.csv")
    st_lattice_path = os.path.join(args.input_dir, "st_lattice.csv")
    st_obstacles_path = os.path.join(args.input_dir, "st_obstacles.csv")
    summary_path = os.path.join(args.input_dir, "summary.txt")
    output_path = args.output or os.path.join(
        args.input_dir, "st_graph_paper.png" if args.paper else "st_graph.png"
    )

    ensure_required_inputs(args.input_dir, speed_profile_path)

    speed_profile = read_speed_profile_csv(speed_profile_path)
    speed_colors = {
        "qp_speed": "#A45F3A",
        "dp_speed": "#5F83A8",
        "ref_speed": "#6F9880",
        "acceleration": "#A97893",
    }
    smoothed_speed_profile = build_smoothed_speed_profile(speed_profile)
    smooth_t_dense, smooth_v_dense = smooth_curve_for_display(
        smoothed_speed_profile["t"],
        smoothed_speed_profile["v"],
        samples_per_segment=56 if args.paper else 40,
    )
    _, smooth_a_dense = smooth_curve_for_display(
        smoothed_speed_profile["t"],
        smoothed_speed_profile["a"],
        samples_per_segment=56 if args.paper else 40,
    )
    st_dp_path = read_st_dp_path_csv(st_dp_path_path) if os.path.exists(st_dp_path_path) else None
    dp_speed_profile = build_dp_speed_profile(
        st_dp_path,
        initial_speed=speed_profile["v"][0] if speed_profile.get("v") else None,
    )
    st_lattice = read_st_lattice_csv(st_lattice_path) if args.show_grid and os.path.exists(st_lattice_path) else None
    st_obstacles = read_st_obstacles_csv(st_obstacles_path) if os.path.exists(st_obstacles_path) else []
    summary = read_summary_txt(summary_path)
    st_time_step = float(summary.get("speed_plan_t_dt", 0.25) or 0.25)
    st_lateral_limit = float(summary.get("st_lateral_limit", 1.0) or 1.0)
    st_s_min_step = float(summary.get("st_s_min_step", 0.05) or 0.05)
    speed_reference = float(summary.get("speed_reference", 0.0) or 0.0)
    st_plan_time = max(speed_profile["t"]) if speed_profile["t"] else 0.0
    st_lattice_plot = thin_st_lattice_points(
        st_lattice,
        row_stride=compute_st_row_stride(st_s_min_step, args.paper),
        col_stride=1,
        keep_points=st_dp_path,
    ) if st_lattice is not None else None
    if args.paper:
        apply_paper_style()
    text_font = build_text_font_properties()

    fig_width = 7.2 if args.paper else 14.0
    fig_height = 3.2 if args.paper else 6.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), constrained_layout=args.paper)

    ax_st = axes[0]
    if st_lattice_plot is not None and st_lattice_plot["t"]:
        ax_st.scatter(
            st_lattice_plot["t"],
            st_lattice_plot["s"],
            s=14 if args.paper else 24,
            facecolors="none",
            edgecolors="#8a939c",
            linewidths=0.9 if args.paper else 1.1,
            alpha=0.85,
            label="ST采样栅格",
            zorder=1,
        )
    obstacle_label_used = False
    for obstacle in st_obstacles:
        if not obstacle["is_dynamic"]:
            continue
        polygon = build_st_occupancy_polygon(obstacle, st_lateral_limit, st_plan_time)
        if polygon is None or len(polygon) < 3:
            continue
        xs = [point[0] for point in polygon] + [polygon[0][0]]
        ys = [point[1] for point in polygon] + [polygon[0][1]]
        ax_st.fill(xs, ys, color="#9aa0a6", alpha=0.30, zorder=2)
        ax_st.plot(
            xs,
            ys,
            color="#4f545a",
            linewidth=1.2,
            zorder=2,
            label="ST障碍物" if not obstacle_label_used else None,
        )
        obstacle_label_used = True
    if st_dp_path is not None and st_dp_path["t"]:
        feasible_t, feasible_s, fallback_t, fallback_s = split_feasible_dp_path(st_dp_path)
        if feasible_t:
            ax_st.plot(
                feasible_t,
                feasible_s,
                label="DP速度决策",
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
            ax_st.plot(
                fallback_t,
                fallback_s,
                linewidth=1.3,
                color="#1f5aa6",
                linestyle=":",
                alpha=0.55,
                zorder=2.8,
            )
    ax_st.plot(
        smoothed_speed_profile["t"],
        smoothed_speed_profile["s"],
        label="QP速度曲线",
        linewidth=2.1,
        color="#d04a02",
        zorder=4,
    )
    if speed_reference > 1e-6 and smoothed_speed_profile["t"]:
        ax_st.plot(
            smoothed_speed_profile["t"],
            [speed_reference * t_value for t_value in smoothed_speed_profile["t"]],
            label="参考速度线",
            linewidth=1.9,
            color="#2e8b57",
            linestyle="-.",
            alpha=0.95,
            zorder=5,
        )
    ax_st.set_xlabel("t [s]", fontproperties=text_font)
    ax_st.set_ylabel("s [m]", fontproperties=text_font)
    ax_st.xaxis.set_major_locator(MultipleLocator(st_time_step))
    for tick in ax_st.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    if args.paper:
        ax_st.tick_params(axis="x", labelsize=8)
    style_axes(ax_st)
    st_legend_kwargs = {"frameon": True, "framealpha": 0.92}
    if args.paper:
        st_legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.10),
                "ncol": 3,
                "columnspacing": 1.2,
                "handlelength": 2.0,
                "borderaxespad": 0.0,
            }
        )
    else:
        st_legend_kwargs["loc"] = "upper right"
    if text_font is not None:
        st_legend_kwargs["prop"] = text_font
    obstacle_legend_handle = Rectangle(
        (0.0, 0.0),
        1.0,
        0.52,
        facecolor="#9aa0a6",
        edgecolor="#4f545a",
        linewidth=1.2,
        alpha=0.30,
    )
    st_legend_handles = [
        obstacle_legend_handle,
        Line2D([], [], color="#1f5aa6", linewidth=1.8, linestyle="--", marker="o", markersize=3.2 if args.paper else 4.2, markerfacecolor="white", markeredgecolor="#1f5aa6", markeredgewidth=0.9 if args.paper else 1.0, label="DP速度决策"),
        Line2D([], [], color="#d04a02", linewidth=2.1, label="QP速度曲线"),
        Line2D([], [], color="#2e8b57", linewidth=1.9, linestyle="-.", alpha=0.95, label="参考速度线"),
        Line2D([], [], linestyle="none", marker="o", markersize=5.0 if args.paper else 6.0, markerfacecolor="none", markeredgecolor="#8a939c", markeredgewidth=0.9 if args.paper else 1.1, alpha=0.85, label="ST采样栅格"),
    ]
    st_legend_labels = ["障碍物速度占据", "DP速度决策", "QP速度曲线", "参考速度线", "ST采样栅格"]
    ax_st.legend(
        st_legend_handles,
        st_legend_labels,
        **st_legend_kwargs,
    )

    ax_speed = axes[1]
    speed_line = ax_speed.plot(
        smooth_t_dense,
        smooth_v_dense,
        color=speed_colors["qp_speed"],
        linewidth=2.1,
        label="速度",
        solid_joinstyle="round",
        solid_capstyle="round",
    )[0]
    ref_speed_line = None
    dp_speed_line = None
    if speed_reference > 1e-6 and smoothed_speed_profile["t"]:
        ref_speed_line = ax_speed.axhline(
            speed_reference,
            color=speed_colors["ref_speed"],
            linewidth=1.9,
            linestyle="-.",
            alpha=0.95,
            label="参考速度",
        )
    if dp_speed_profile is not None and dp_speed_profile["t"]:
        dp_t_dense, dp_v_dense = monotone_cubic_curve(
            dp_speed_profile["t"],
            dp_speed_profile["v"],
            samples_per_segment=20 if args.paper else 14,
        )
        dp_speed_line = ax_speed.plot(
            dp_t_dense,
            dp_v_dense,
            color=speed_colors["dp_speed"],
            linewidth=1.7,
            linestyle="-.",
            label="决策速度",
            solid_joinstyle="round",
            solid_capstyle="round",
            zorder=3.5,
        )[0]
        ax_speed.plot(
            dp_speed_profile["t"],
            dp_speed_profile["v"],
            linestyle="none",
            marker="o",
            markersize=2.6 if args.paper else 3.4,
            markerfacecolor="white",
            markeredgecolor=speed_colors["dp_speed"],
            markeredgewidth=0.8 if args.paper else 0.9,
            alpha=0.95,
            zorder=3.6,
            label="_nolegend_",
        )
    ax_speed.set_xlabel("t [s]", fontproperties=text_font)
    ax_speed.set_ylabel("v [m/s]", fontproperties=text_font)
    ax_speed.xaxis.set_major_locator(MultipleLocator(0.25))
    for tick in ax_speed.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    if args.paper:
        ax_speed.tick_params(axis="x", labelsize=8)
    style_axes(ax_speed)
    ax_acc = ax_speed.twinx()
    acc_line = ax_acc.plot(
        smooth_t_dense,
        smooth_a_dense,
        color=speed_colors["acceleration"],
        linewidth=1.8,
        linestyle="--",
        label="加速度",
        dash_joinstyle="round",
        dash_capstyle="round",
    )[0]
    ax_acc.set_ylabel("a [m/s^2]", fontproperties=text_font)
    ax_acc.spines["top"].set_visible(False)

    speed_legend_lines = [speed_line]
    if dp_speed_line is not None:
        speed_legend_lines.append(dp_speed_line)
    if ref_speed_line is not None:
        speed_legend_lines.append(ref_speed_line)
    speed_legend_lines.append(acc_line)
    speed_legend_labels = [line.get_label() for line in speed_legend_lines]
    speed_legend_kwargs = {"frameon": True, "framealpha": 0.92}
    if args.paper:
        speed_legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.10),
                "ncol": 2 if dp_speed_line is not None else 3,
                "columnspacing": 1.0,
                "handlelength": 2.0,
                "borderaxespad": 0.0,
            }
        )
    else:
        speed_legend_kwargs["loc"] = "upper right"
    if text_font is not None:
        speed_legend_kwargs["prop"] = text_font
    ax_speed.legend(speed_legend_lines, speed_legend_labels, **speed_legend_kwargs)

    if not args.paper:
        fig.tight_layout()
    fig.savefig(output_path, dpi=330)
    pdf_path = None
    if args.export_pdf:
        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(pdf_path)
    print(f"saved_png={output_path}")
    if pdf_path is not None:
        print(f"saved_pdf={pdf_path}")


if __name__ == "__main__":
    main()
