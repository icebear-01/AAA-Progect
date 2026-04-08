from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
NEURAL_ASTAR_ROOT = SCRIPT_PATH.parent.parent
HYBRID_ASTAR_ROOT = NEURAL_ASTAR_ROOT.parents[1]
MAP_DIR = NEURAL_ASTAR_ROOT / "map"

if str(NEURAL_ASTAR_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(NEURAL_ASTAR_ROOT / "src"))
if str(HYBRID_ASTAR_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(HYBRID_ASTAR_ROOT / "scripts"))

from benchmark_custom_map_v3_resize64 import _decode_prediction  # type: ignore
from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.coords import make_one_hot_xy
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from offline_street_guided_astar_demo import (  # type: ignore
    default_smoother_cli,
    plot_curvature_compare,
    plot_heading_compare,
    plot_xy_trajectory_compare,
    read_smoothed_path_csv,
    read_split_points_csv,
    write_raw_path_csv,
    write_smoother_yaml,
)
import make_custom_map_paper_pictures as paper_figs  # type: ignore


DEFAULT_CKPT = (
    NEURAL_ASTAR_ROOT
    / "outputs"
    / "model_guidance_grid_mpd_unet_transformer_v3_rebuiltexpert_w035_sd5_p2_v2_policyfix_gpu_v1"
    / "best_eval_snapshot_epoch1.pt"
)
PLOT_DPI = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate custom-map paper figures for a new start/goal.")
    p.add_argument("--base-request-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start-grid-x", type=int, default=None)
    p.add_argument("--start-grid-y", type=int, default=None)
    p.add_argument("--goal-grid-x", type=int, required=True)
    p.add_argument("--goal-grid-y", type=int, required=True)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--target-height", type=int, default=64)
    p.add_argument("--target-width", type=int, default=64)
    p.add_argument("--residual-weight", type=float, default=0.15)
    p.add_argument("--confidence-mode", type=str, default="learned")
    p.add_argument("--search-clearance-weight", type=float, default=0.2)
    p.add_argument("--search-clearance-safe-distance", type=float, default=3.0)
    p.add_argument("--search-clearance-power", type=float, default=2.0)
    return p.parse_args()


def grid_to_world(gx: int, gy: int, origin_x: float, origin_y: float, resolution: float) -> tuple[float, float]:
    return origin_x + (gx + 0.5) * resolution, origin_y + (gy + 0.5) * resolution


def world_to_grid_xy(
    pts: Sequence[tuple[float, float]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    gx = (arr[:, 0] - origin_x) / resolution - 0.5
    gy = (arr[:, 1] - origin_y) / resolution - 0.5
    return np.column_stack([gx, gy])


def load_request(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_guidance_native(
    *,
    model: torch.nn.Module,
    device: str,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    confidence_mode: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    h, w = occ.shape
    start = make_one_hot_xy(start_xy[0], start_xy[1], w, h)
    goal = make_one_hot_xy(goal_xy[0], goal_xy[1], w, h)
    clearance_input = build_clearance_input_map(
        occ_map=occ,
        clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
    )[None, None].astype(np.float32)

    occ_t = torch.from_numpy(occ[None, None]).to(device)
    start_t = torch.from_numpy(start[None, None]).to(device)
    goal_t = torch.from_numpy(goal[None, None]).to(device)
    extra_input_t = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        extra_input_t = torch.from_numpy(clearance_input).to(device)

    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(
            occ_t,
            start_t,
            goal_t,
            start_yaw=torch.zeros(1, device=device, dtype=occ_t.dtype),
            goal_yaw=torch.zeros(1, device=device, dtype=occ_t.dtype),
            extra_input_maps=extra_input_t,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

    residual, learned_conf = _decode_prediction(model, out)
    conf = resolve_residual_confidence_map(
        mode=str(confidence_mode),
        occ_map=occ,
        residual_map=residual,
        learned_confidence_map=learned_conf,
    )
    return infer_ms, residual.astype(np.float32), conf.astype(np.float32)


def infer_guidance_resize64(
    *,
    model: torch.nn.Module,
    device: str,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    target_h: int,
    target_w: int,
    confidence_mode: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    h, w = occ.shape
    sx = int(np.clip(round(start_xy[0] * (target_w - 1) / max(w - 1, 1)), 0, target_w - 1))
    sy = int(np.clip(round(start_xy[1] * (target_h - 1) / max(h - 1, 1)), 0, target_h - 1))
    gx = int(np.clip(round(goal_xy[0] * (target_w - 1) / max(w - 1, 1)), 0, target_w - 1))
    gy = int(np.clip(round(goal_xy[1] * (target_h - 1) / max(h - 1, 1)), 0, target_h - 1))
    start = make_one_hot_xy(sx, sy, target_w, target_h)
    goal = make_one_hot_xy(gx, gy, target_w, target_h)

    occ_t = torch.from_numpy(occ[None, None]).to(device)
    occ_small = F.interpolate(occ_t, size=(target_h, target_w), mode="nearest")
    clearance_small = build_clearance_input_map(
        occ_map=occ_small[0, 0].detach().cpu().numpy().astype(np.float32),
        clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
    )[None, None].astype(np.float32)
    start_t = torch.from_numpy(start[None, None]).to(device)
    goal_t = torch.from_numpy(goal[None, None]).to(device)
    extra_input_t = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        extra_input_t = torch.from_numpy(clearance_small).to(device)

    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(
            occ_small,
            start_t,
            goal_t,
            start_yaw=torch.zeros(1, device=device, dtype=occ_small.dtype),
            goal_yaw=torch.zeros(1, device=device, dtype=occ_small.dtype),
            extra_input_maps=extra_input_t,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

    residual_small, learned_conf_small = _decode_prediction(model, out)
    occ_small_np = occ_small[0, 0].detach().cpu().numpy().astype(np.float32)
    conf_small = resolve_residual_confidence_map(
        mode=str(confidence_mode),
        occ_map=occ_small_np,
        residual_map=residual_small,
        learned_confidence_map=learned_conf_small,
    )
    residual = (
        F.interpolate(
            torch.from_numpy(residual_small[None, None]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    conf = (
        F.interpolate(
            torch.from_numpy(conf_small[None, None]),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    residual[occ > 0.5] = 0.0
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
    conf[occ > 0.5] = 0.0
    return infer_ms, residual.astype(np.float32), conf.astype(np.float32)


def path_to_world(path_xy: Sequence[tuple[int, int]], origin_x: float, origin_y: float, resolution: float) -> list[tuple[float, float]]:
    return [grid_to_world(x, y, origin_x, origin_y, resolution) for x, y in path_xy]


def write_grid_world_csv(
    out_csv: Path,
    path_xy: Sequence[tuple[int, int]],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_x", "grid_y", "world_x", "world_y"])
        for x, y in path_xy:
            wx, wy = grid_to_world(x, y, origin_x, origin_y, resolution)
            writer.writerow([x, y, f"{wx:.6f}", f"{wy:.6f}"])


def remove_pdf(png_path: Path) -> None:
    pdf_path = png_path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()


def plot_astar_vs_ours(
    *,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    astar_path: Sequence[tuple[int, int]],
    ours_path: Sequence[tuple[int, int]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor="#fbfbfa")
    titles = ["A* 基线", "本文方法（64×64）"]
    colors = ["#4C78A8", "#E66100"]
    paths = [astar_path, ours_path]
    for ax, title, color, path in zip(axes, titles, colors, paths):
        ax.imshow(occ, cmap="gray_r", origin="lower", interpolation="nearest")
        if path:
            ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=2.8, solid_capstyle="round")
        ax.scatter([start_xy[0]], [start_xy[1]], s=90, color="#1a9850", edgecolors="none", zorder=5)
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=90, marker="x", linewidths=2.0, color="#d73027", zorder=6)
        paper_figs._style_axes(ax, tick_size=16)
        title_fp = paper_figs._font(20, weight="bold")
        if title_fp is not None:
            ax.set_title(title, fontproperties=title_fp, pad=10)
        else:
            ax.set_title(title, fontsize=20, pad=10)
        label_fp = paper_figs._font(17)
        if label_fp is not None:
            ax.set_xlabel("栅格 X", fontproperties=label_fp)
            ax.set_ylabel("栅格 Y", fontproperties=label_fp)
        else:
            ax.set_xlabel("栅格 X", fontsize=17)
            ax.set_ylabel("栅格 Y", fontsize=17)
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.88, wspace=0.08)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_expansion_compare(
    *,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    cases: Sequence[dict],
    resolution: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(17.2, 5.3), facecolor="#fbfbfa")
    if len(cases) == 1:
        axes = [axes]
    title_fp = paper_figs._font(17, weight="bold")
    small_fp = paper_figs._font(13)
    for ax, case in zip(axes, cases):
        ax.imshow(occ, cmap="gray_r", origin="lower", interpolation="nearest")
        heat = np.zeros_like(occ, dtype=np.float32)
        for x, y in case["expanded_xy"]:
            if 0 <= y < heat.shape[0] and 0 <= x < heat.shape[1]:
                heat[y, x] += 1.0
        if np.any(heat > 0):
            masked = np.ma.masked_where(heat <= 0, heat)
            ax.imshow(masked, cmap="cividis", origin="lower", interpolation="nearest", alpha=0.65)
        path = case["path"]
        if path:
            ax.plot([p[0] for p in path], [p[1] for p in path], color=case["path_color"], linewidth=2.6)
        ax.scatter([start_xy[0]], [start_xy[1]], s=65, color="#1a9850", zorder=5)
        ax.scatter([goal_xy[0]], [goal_xy[1]], s=72, marker="x", linewidths=2.0, color="#d73027", zorder=6)
        paper_figs._style_axes(ax, tick_size=12)
        if title_fp is not None:
            ax.set_title(case["title"], fontproperties=title_fp, pad=8)
        else:
            ax.set_title(case["title"], fontsize=17, pad=8)
        txt = (
            f"扩展节点={case['expanded_nodes']}  路径长度={case['path_length_m']:.2f} m\n"
            f"搜索时间={case['runtime_ms']:.2f} ms"
        )
        if small_fp is not None:
            ax.text(
                0.5,
                1.03,
                txt,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontproperties=small_fp,
                fontsize=13,
                color="#22252b",
            )
        else:
            ax.text(0.5, 1.03, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=13, color="#22252b")
        ax.set_xlabel("栅格 X", fontproperties=paper_figs._font(13))
        ax.set_ylabel("栅格 Y", fontproperties=paper_figs._font(13))
        ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.06, top=0.88, wspace=0.05)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_overlay_on_map(
    *,
    occ: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    raw_world_path: Sequence[tuple[float, float]],
    seed_world_path: Sequence[tuple[float, float]],
    smoothed_world_path: Sequence[tuple[float, float]],
    out_path: Path,
) -> None:
    raw_g = world_to_grid_xy(raw_world_path, origin_x, origin_y, resolution)
    seed_g = world_to_grid_xy(seed_world_path, origin_x, origin_y, resolution)
    smooth_g = world_to_grid_xy(smoothed_world_path, origin_x, origin_y, resolution)
    fig, ax = plt.subplots(1, 1, figsize=(9.4, 5.0), facecolor="#fcfcfb")
    ax.imshow(occ, cmap="gray_r", origin="lower", interpolation="nearest", extent=[0, occ.shape[1], 0, occ.shape[0]])
    if raw_g.size:
        ax.plot(raw_g[:, 0], raw_g[:, 1], color="#4C78A8", linestyle="--", linewidth=2.4)
    if seed_g.size:
        ax.plot(seed_g[:, 0], seed_g[:, 1], color="#E69F00", linewidth=2.6)
    if smooth_g.size:
        ax.plot(smooth_g[:, 0], smooth_g[:, 1], color="#009E73", linewidth=2.9)
    paper_figs._style_axes(ax, tick_size=20)
    label_fp = paper_figs._font(20)
    if label_fp is not None:
        ax.set_xlabel("栅格 X", fontproperties=label_fp)
        ax.set_ylabel("栅格 Y", fontproperties=label_fp)
    else:
        ax.set_xlabel("栅格 X", fontsize=20)
        ax.set_ylabel("栅格 Y", fontsize=20)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.97)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_map_only(*, occ: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), facecolor="#fcfcfb")
    ax.imshow(
        occ,
        cmap="gray_r",
        origin="lower",
        interpolation="nearest",
        extent=[0, occ.shape[1], 0, occ.shape[0]],
    )
    paper_figs._style_axes(ax, tick_size=22)
    label_fp = paper_figs._font(21)
    if label_fp is not None:
        ax.set_xlabel("栅格 X", fontproperties=label_fp)
        ax.set_ylabel("栅格 Y", fontproperties=label_fp)
    else:
        ax.set_xlabel("栅格 X", fontsize=21)
        ax.set_ylabel("栅格 Y", fontsize=21)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.98)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_speed_compare(benchmark_json: Path, out_path: Path, metrics_out: Path) -> None:
    data = json.loads(benchmark_json.read_text(encoding="utf-8"))
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
    paper_figs._style_axes(ax, tick_size=18)
    x = np.arange(len(labels))
    width = 0.56
    infer_color = "#A6CEE3"
    search_colors = ["#4C78A8", "#1B9E77", "#E66100"]
    ax.bar(x, infer, width=width, color=infer_color, edgecolor="none", label="模型推理")
    ax.bar(x, search, width=width, bottom=infer, color=search_colors, edgecolor="none", label="搜索阶段")
    fp_tick = paper_figs._font(18)
    if fp_tick is not None:
        ax.set_xticks(x, labels, fontproperties=fp_tick)
    else:
        ax.set_xticks(x, labels, fontsize=18)
    fp_label = paper_figs._font(18)
    if fp_label is not None:
        ax.set_ylabel("平均时间（ms）", fontproperties=fp_label)
    else:
        ax.set_ylabel("平均时间（ms）", fontsize=18)
    fp_title = paper_figs._font(21, weight="bold")
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
            fontproperties=paper_figs._font(16),
        )
    legend = ax.legend(
        loc="upper right",
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#cfd6de",
        prop=paper_figs._font(15),
    )
    if legend is not None:
        for text in legend.get_texts():
            fp = paper_figs._font(15)
            if fp is not None:
                text.set_fontproperties(fp)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.15, top=0.87)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "A*": {"infer_ms": 0.0, "search_ms": float(search[0]), "total_ms": float(total[0])},
        "V3_原尺度": {"infer_ms": float(infer[1]), "search_ms": float(search[1]), "total_ms": float(total[1])},
        "本文方法_64x64": {"infer_ms": float(infer[2]), "search_ms": float(search[2]), "total_ms": float(total[2])},
    }
    metrics_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def benchmark_speed(
    *,
    request_json: Path,
    ckpt: Path,
    output_json: Path,
    target_h: int,
    target_w: int,
    residual_weight: float,
    confidence_mode: str,
    warmup: int,
    repeat: int,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH.parent / "benchmark_custom_map_v3_resize64.py"),
        "--map-json",
        str(request_json),
        "--ckpt",
        str(ckpt),
        "--output-json",
        str(output_json),
        "--target-height",
        str(target_h),
        "--target-width",
        str(target_w),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--residual-weight",
        str(residual_weight),
        "--confidence-mode",
        str(confidence_mode),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else str(args.device))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    request = load_request(args.base_request_json)
    occ = np.asarray(request["occupancy"], dtype=np.float32)
    origin_x = float(request["origin_x"])
    origin_y = float(request["origin_y"])
    resolution = float(request["resolution"])
    width = int(request["width"])
    height = int(request["height"])

    if args.start_grid_x is None or args.start_grid_y is None:
        start_world = tuple(request["start_world"])
        start_xy = (
            int(round((start_world[0] - origin_x) / resolution - 0.5)),
            int(round((start_world[1] - origin_y) / resolution - 0.5)),
        )
    else:
        start_xy = (int(args.start_grid_x), int(args.start_grid_y))
    goal_xy = (int(args.goal_grid_x), int(args.goal_grid_y))

    if not (0 <= start_xy[0] < width and 0 <= start_xy[1] < height):
        raise ValueError(f"start_xy out of bounds: {start_xy}")
    if not (0 <= goal_xy[0] < width and 0 <= goal_xy[1] < height):
        raise ValueError(f"goal_xy out of bounds: {goal_xy}")
    if occ[start_xy[1], start_xy[0]] > 0.5:
        raise ValueError(f"start cell occupied: {start_xy}")
    if occ[goal_xy[1], goal_xy[0]] > 0.5:
        raise ValueError(f"goal cell occupied: {goal_xy}")

    request["start_world"] = list(grid_to_world(start_xy[0], start_xy[1], origin_x, origin_y, resolution))
    request["goal_world"] = list(grid_to_world(goal_xy[0], goal_xy[1], origin_x, origin_y, resolution))
    request_json = data_dir / "map_request_goal_shifted.json"
    request_json.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_t0 = time.perf_counter()
    baseline = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
    )
    baseline_ms = (time.perf_counter() - baseline_t0) * 1000.0
    if not baseline.success or baseline.path is None:
        raise RuntimeError(f"A* baseline failed for {start_xy} -> {goal_xy}")

    model = load_guidance_encoder(args.ckpt, device=device)
    native_infer_ms, native_residual, native_conf = infer_guidance_native(
        model=model,
        device=device,
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        confidence_mode=args.confidence_mode,
    )
    native_t0 = time.perf_counter()
    native = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        heuristic_residual_map=native_residual,
        residual_confidence_map=native_conf,
        residual_weight=float(args.residual_weight),
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
    )
    native_search_ms = (time.perf_counter() - native_t0) * 1000.0
    if not native.success or native.path is None:
        raise RuntimeError("Native V3 failed.")

    resize_infer_ms, resize_residual, resize_conf = infer_guidance_resize64(
        model=model,
        device=device,
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        target_h=int(args.target_height),
        target_w=int(args.target_width),
        confidence_mode=args.confidence_mode,
    )
    resize_t0 = time.perf_counter()
    resize = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        heuristic_residual_map=resize_residual,
        residual_confidence_map=resize_conf,
        residual_weight=float(args.residual_weight),
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
    )
    resize_search_ms = (time.perf_counter() - resize_t0) * 1000.0
    if not resize.success or resize.path is None:
        raise RuntimeError("Resize64 V3 failed.")

    clearance_t0 = time.perf_counter()
    clearance = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        heuristic_residual_map=resize_residual,
        residual_confidence_map=resize_conf,
        residual_weight=float(args.residual_weight),
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
        clearance_weight=float(args.search_clearance_weight),
        clearance_safe_distance=float(args.search_clearance_safe_distance),
        clearance_power=float(args.search_clearance_power),
        clearance_integration_mode="g_cost",
    )
    clearance_search_ms = (time.perf_counter() - clearance_t0) * 1000.0
    if not clearance.success or clearance.path is None:
        raise RuntimeError("Resize64 clearance V3 failed.")

    astar_path_csv = data_dir / "astar_baseline_path.csv"
    native_path_csv = data_dir / "v3_native_fullres_path.csv"
    resize_path_csv = data_dir / "v3_resize64_path.csv"
    clearance_path_csv = data_dir / "v3_resize64_clearance_path.csv"
    write_grid_world_csv(astar_path_csv, baseline.path, origin_x, origin_y, resolution)
    write_grid_world_csv(native_path_csv, native.path, origin_x, origin_y, resolution)
    write_grid_world_csv(resize_path_csv, resize.path, origin_x, origin_y, resolution)
    write_grid_world_csv(clearance_path_csv, clearance.path, origin_x, origin_y, resolution)

    raw_world_path = path_to_world(clearance.path, origin_x, origin_y, resolution)
    raw_csv = data_dir / "frontend_raw_path.csv"
    seed_csv = data_dir / "frontend_seed_path.csv"
    smooth_csv = data_dir / "smoothed_path.csv"
    split_csv = data_dir / "segment_split_points.csv"
    smoother_yaml = data_dir / "smoother_request.yaml"
    write_raw_path_csv(raw_csv, raw_world_path)
    write_smoother_yaml(
        smoother_yaml,
        occ=occ,
        raw_world_path=raw_world_path,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
        seed_xy_box_half_extent=0.10,
        skip_seed_collision_check=False,
    )
    smoother_cli = default_smoother_cli()
    smoother_env = os.environ.copy()
    smoother_lib_dir = smoother_cli.resolve().parents[1]
    existing_ld = smoother_env.get("LD_LIBRARY_PATH", "")
    smoother_env["LD_LIBRARY_PATH"] = f"{smoother_lib_dir}:{existing_ld}" if existing_ld else str(smoother_lib_dir)
    subprocess.run(
        [
            str(smoother_cli),
            "--input-yaml",
            str(smoother_yaml),
            "--seed-csv",
            str(seed_csv),
            "--split-points-csv",
            str(split_csv),
            "--output-csv",
            str(smooth_csv),
        ],
        check=True,
        env=smoother_env,
    )
    seed_world_path = read_smoothed_path_csv(seed_csv)
    smoothed_world_path = read_smoothed_path_csv(smooth_csv)
    split_world_points = read_split_points_csv(split_csv)

    all_heading = []
    for path in (raw_world_path, seed_world_path, smoothed_world_path):
        arr = np.asarray(path, dtype=np.float64)
        if arr.shape[0] >= 2:
            delta = np.diff(arr, axis=0)
            heading = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
            if heading.size > 0:
                all_heading.append(heading)
    if all_heading:
        heading_all = np.concatenate(all_heading)
        heading_center = float(0.5 * (np.min(heading_all) + np.max(heading_all)))
    else:
        heading_center = 0.0

    plot_astar_vs_ours(
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        astar_path=baseline.path,
        ours_path=clearance.path,
        out_path=output_dir / "01_astar_vs_ours_planning.png",
    )

    plot_xy_trajectory_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=output_dir / "02_model_frontend_backend_planning.png",
        show_legend=False,
        y_grid_cells=5,
        equal_aspect=False,
        figsize=(9.5, 5.1),
        linewidth_scale=1.45,
        tick_font_size_override=20,
    )
    remove_pdf(output_dir / "02_model_frontend_backend_planning.png")

    plot_expansion_compare(
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        cases=[
            {
                "title": "A*",
                "path": baseline.path,
                "expanded_xy": baseline.expanded_xy,
                "expanded_nodes": int(baseline.expanded_nodes),
                "path_length_m": float(path_length_8conn(baseline.path, diagonal_cost=math.sqrt(2.0)) * resolution),
                "runtime_ms": float(baseline_ms),
                "path_color": "#4C78A8",
            },
            {
                "title": "V3 原尺度",
                "path": native.path,
                "expanded_xy": native.expanded_xy,
                "expanded_nodes": int(native.expanded_nodes),
                "path_length_m": float(path_length_8conn(native.path, diagonal_cost=math.sqrt(2.0)) * resolution),
                "runtime_ms": float(native_infer_ms + native_search_ms),
                "path_color": "#1B9E77",
            },
            {
                "title": "本文方法64×64",
                "path": resize.path,
                "expanded_xy": resize.expanded_xy,
                "expanded_nodes": int(resize.expanded_nodes),
                "path_length_m": float(path_length_8conn(resize.path, diagonal_cost=math.sqrt(2.0)) * resolution),
                "runtime_ms": float(resize_infer_ms + resize_search_ms),
                "path_color": "#E66100",
            },
        ],
        resolution=resolution,
        out_path=output_dir / "03_model_guided_expansion_demo.png",
    )

    plot_heading_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=output_dir / "04_heading_curve_compare.png",
        split_world_points=split_world_points,
        target_center_deg=heading_center,
        show_legend=False,
        plot_smooth_window_m=0.8,
        align_to_raw_reference_s=False,
        tick_font_size_override=24,
    )
    remove_pdf(output_dir / "04_heading_curve_compare.png")

    plot_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=output_dir / "05_curvature_curve_compare.png",
        include_raw_path=False,
        split_world_points=split_world_points,
        curvature_limit=0.8,
        plot_smooth_window_m=0.0,
    )
    remove_pdf(output_dir / "05_curvature_curve_compare.png")
    plot_curvature_compare(
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=output_dir / "05_curvature_curve_compare_light_smooth.png",
        include_raw_path=False,
        split_world_points=split_world_points,
        curvature_limit=0.8,
        plot_smooth_window_m=0.35,
    )
    remove_pdf(output_dir / "05_curvature_curve_compare_light_smooth.png")

    speed_json = data_dir / "speed_benchmark.json"
    benchmark_speed(
        request_json=request_json,
        ckpt=args.ckpt,
        output_json=speed_json,
        target_h=int(args.target_height),
        target_w=int(args.target_width),
        residual_weight=float(args.residual_weight),
        confidence_mode=str(args.confidence_mode),
        warmup=int(args.warmup),
        repeat=int(args.repeat),
    )
    plot_speed_compare(speed_json, output_dir / "06_speed_compare.png", output_dir / "06_speed_compare_metrics.json")

    plot_overlay_on_map(
        occ=occ,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
        raw_world_path=raw_world_path,
        seed_world_path=seed_world_path,
        smoothed_world_path=smoothed_world_path,
        out_path=output_dir / "07_frontend_backend_overlay_on_map.png",
    )

    plot_map_only(occ=occ, out_path=output_dir / "08_map_only.png")

    summary = {
        "request_json": str(request_json),
        "start_grid": list(start_xy),
        "goal_grid": list(goal_xy),
        "astar": {
            "expanded_nodes": int(baseline.expanded_nodes),
            "runtime_ms": float(baseline_ms),
            "path_length_m": float(path_length_8conn(baseline.path, diagonal_cost=math.sqrt(2.0)) * resolution),
        },
        "v3_native": {
            "expanded_nodes": int(native.expanded_nodes),
            "infer_ms": float(native_infer_ms),
            "search_ms": float(native_search_ms),
        },
        "v3_resize64": {
            "expanded_nodes": int(resize.expanded_nodes),
            "infer_ms": float(resize_infer_ms),
            "search_ms": float(resize_search_ms),
        },
        "v3_resize64_clearance": {
            "expanded_nodes": int(clearance.expanded_nodes),
            "infer_ms": float(resize_infer_ms),
            "search_ms": float(clearance_search_ms),
        },
    }
    (data_dir / "case_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
