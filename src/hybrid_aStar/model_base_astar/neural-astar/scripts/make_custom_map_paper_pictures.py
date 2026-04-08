from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = ROOT / "map"
OUT_DIR = MAP_DIR / "paper_pictures"
FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")
PLOT_DPI = 500


def _font(size: float, weight: str | None = None):
    if FONT_PATH.exists():
        return fm.FontProperties(fname=str(FONT_PATH), size=size, weight=weight)
    return None


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_path_csv(path: Path, x_key: str, y_key: str):
    pts = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row[x_key]), float(row[y_key])))
    return np.asarray(pts, dtype=np.float64)


def _world_to_grid(path_xy: np.ndarray, origin_x: float, origin_y: float, resolution: float):
    if path_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    gx = (path_xy[:, 0] - origin_x) / resolution
    gy = (path_xy[:, 1] - origin_y) / resolution
    return np.column_stack([gx, gy])


def _style_axes(ax, tick_size: int = 16):
    ax.set_facecolor("#fbfbfa")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#a7adb5")
    ax.tick_params(axis="both", labelsize=tick_size, colors="#2d3138")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        fp = _font(tick_size)
        if fp is not None:
            label.set_fontproperties(fp)


def make_astar_vs_ours_planning():
    req = _read_json(MAP_DIR / "map_frontend_request_auto.json")
    occ = np.asarray(req["occupancy"], dtype=np.uint8)
    origin_x = float(req["origin_x"])
    origin_y = float(req["origin_y"])
    resolution = float(req["resolution"])
    start_world = np.asarray(req["start_world"], dtype=np.float64)
    goal_world = np.asarray(req["goal_world"], dtype=np.float64)
    start_grid = (
        (start_world[0] - origin_x) / resolution,
        (start_world[1] - origin_y) / resolution,
    )
    goal_grid = (
        (goal_world[0] - origin_x) / resolution,
        (goal_world[1] - origin_y) / resolution,
    )

    astar = _read_path_csv(MAP_DIR / "map_frontend_request_auto_path_astar_baseline.csv", "world_x", "world_y")
    ours = _read_path_csv(
        MAP_DIR / "map_frontend_request_auto_path_v3_tuned_w015_learned_resize64_clearance.csv",
        "world_x",
        "world_y",
    )
    astar_g = _world_to_grid(astar, origin_x, origin_y, resolution)
    ours_g = _world_to_grid(ours, origin_x, origin_y, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor="#fbfbfa")
    titles = ["A* 基线", "本文方法（64×64）"]
    paths = [astar_g, ours_g]
    colors = ["#4C78A8", "#E66100"]
    for ax, title, path, color in zip(axes, titles, paths, colors):
        ax.imshow(occ, cmap="gray_r", origin="lower", interpolation="nearest")
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.8, solid_capstyle="round")
        ax.scatter([start_grid[0]], [start_grid[1]], s=90, color="#1a9850", edgecolors="none", zorder=5)
        ax.scatter([goal_grid[0]], [goal_grid[1]], s=90, marker="x", linewidths=2.0, color="#d73027", zorder=6)
        _style_axes(ax, tick_size=16)
        title_fp = _font(20, weight="bold")
        if title_fp is not None:
            ax.set_title(title, fontproperties=title_fp, pad=10)
        else:
            ax.set_title(title, fontsize=20, pad=10)
        label_fp = _font(17)
        if label_fp is not None:
            ax.set_xlabel("栅格 X", fontproperties=label_fp)
            ax.set_ylabel("栅格 Y", fontproperties=label_fp)
        else:
            ax.set_xlabel("栅格 X", fontsize=17)
            ax.set_ylabel("栅格 Y", fontsize=17)
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.88, wspace=0.08)
    out = OUT_DIR / "01_astar_vs_ours_planning.png"
    fig.savefig(out, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def make_speed_compare():
    data = _read_json(MAP_DIR / "map_frontend_request_auto_v3_resize64_benchmark.json")
    labels = ["A*", "V3 原尺度", "本文方法\n64×64"]
    infer = np.array(
        [
            0.0,
            float(data["native_fullres"]["infer_ms_avg"]),
            float(data["resize64_then_upsample"]["infer_ms_avg"]),
        ],
        dtype=np.float64,
    )
    search = np.array(
        [
            float(data["baseline_astar"]["search_time_ms"]),
            float(data["native_fullres"]["search_time_ms"]),
            float(data["resize64_then_upsample"]["search_time_ms"]),
        ],
        dtype=np.float64,
    )
    total = infer + search

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.8), facecolor="#fbfbfa")
    _style_axes(ax, tick_size=18)
    x = np.arange(len(labels))
    width = 0.56
    infer_color = "#A6CEE3"
    search_colors = ["#4C78A8", "#1B9E77", "#E66100"]

    infer_bars = ax.bar(x, infer, width=width, color=infer_color, edgecolor="none", label="模型推理")
    search_bars = ax.bar(
        x,
        search,
        width=width,
        bottom=infer,
        color=search_colors,
        edgecolor="none",
        label="搜索阶段",
    )
    ax.set_xticks(x)
    fp_tick = _font(18)
    if fp_tick is not None:
        ax.set_xticklabels(labels, fontproperties=fp_tick)
    else:
        ax.set_xticklabels(labels, fontsize=18)
    fp_label = _font(18)
    if fp_label is not None:
        ax.set_ylabel("平均时间（ms）", fontproperties=fp_label)
    else:
        ax.set_ylabel("平均时间（ms）", fontsize=18)
    fp_title = _font(21, weight="bold")
    if fp_title is not None:
        ax.set_title("推理速度对比", fontproperties=fp_title, pad=10)
    else:
        ax.set_title("推理速度对比", fontsize=21, pad=10)
    ax.grid(True, axis="y", color="#d6dbe2", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    for idx, val in enumerate(total):
        ax.text(
            x[idx],
            val + max(total) * 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="#22252b",
            fontproperties=_font(16),
        )

    legend = ax.legend(
        loc="upper right",
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#cfd6de",
        prop=_font(15),
    )
    if legend is not None:
        for text in legend.get_texts():
            fp = _font(15)
            if fp is not None:
                text.set_fontproperties(fp)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.15, top=0.87)
    out = OUT_DIR / "06_speed_compare.png"
    fig.savefig(out, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "A*": {
            "infer_ms": 0.0,
            "search_ms": float(search[0]),
            "total_ms": float(total[0]),
        },
        "V3_原尺度": {
            "infer_ms": float(infer[1]),
            "search_ms": float(search[1]),
            "total_ms": float(total[1]),
        },
        "本文方法_64x64": {
            "infer_ms": float(infer[2]),
            "search_ms": float(search[2]),
            "total_ms": float(total[2]),
        },
    }
    with (OUT_DIR / "06_speed_compare_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def copy_existing():
    mapping = {
        "map_frontend_request_auto_v3_clearance_backend_compare.png": "02_model_frontend_backend_planning.png",
        "map_frontend_request_auto_expansion_compare.png": "03_model_guided_expansion_demo.png",
        "map_frontend_request_auto_v3_clearance_heading_compare.png": "04_heading_curve_compare.png",
        "map_frontend_request_auto_v3_clearance_curvature_compare.png": "05_curvature_curve_compare.png",
    }
    for src_name, dst_name in mapping.items():
        shutil.copy2(MAP_DIR / src_name, OUT_DIR / dst_name)


def write_readme():
    content = """# Custom Map Paper Figures

1. `01_astar_vs_ours_planning.png`
   A* 基线与本文方法（64×64）规划路径对比。

2. `02_model_frontend_backend_planning.png`
   本文方法前端路径、后端 seed、后端平滑路径对比。

3. `03_model_guided_expansion_demo.png`
   模型引导节点扩展示意图。

4. `04_heading_curve_compare.png`
   前端路径、初始路径、优化路径的航向角曲线。

5. `05_curvature_curve_compare.png`
   前端路径、初始路径、优化路径的曲率曲线。

6. `06_speed_compare.png`
   A*、V3 原尺度、本文方法 64×64 的时间对比。
"""
    (OUT_DIR / "README.md").write_text(content, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_astar_vs_ours_planning()
    copy_existing()
    make_speed_compare()
    write_readme()
    print(f"saved_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
