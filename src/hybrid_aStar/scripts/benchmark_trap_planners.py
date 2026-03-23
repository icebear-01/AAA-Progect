#!/usr/bin/env python3
"""Synthetic trap benchmark for exposing greedy best-first weaknesses.

This script builds a separate frontend-only benchmark composed of synthetic
64x64 trap maps. The goal is not to replace the street benchmark, but to
stress-test search strategies on maps with:

- dead-end corridors toward the goal
- bugtrap-like pockets
- offset narrow entrances
- comb-style misleading branches

The benchmark reuses the same planner implementations and learned guidance
checkpoints as ``benchmark_street_planners.py`` so the comparison protocol
stays aligned with the main project benchmark.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
HYBRID_ASTAR_ROOT = SCRIPT_PATH.parent.parent
BENCHMARK_STREET_PATH = SCRIPT_PATH.parent / "benchmark_street_planners.py"
DEFAULT_OUTPUT_DIR = HYBRID_ASTAR_ROOT / "offline_results" / "paper_demo" / "trap_frontend_benchmark_20260322"
CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")

XY = Tuple[int, int]


def load_street_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_street_planners", BENCHMARK_STREET_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load benchmark module from {BENCHMARK_STREET_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_street_benchmark_module()


@dataclass
class TrapCase:
    occ: np.ndarray
    start_xy: XY
    goal_xy: XY
    trap_type: str
    transform_id: int


@dataclass
class PlotMethod:
    key: str
    label: str
    color: str


METHODS: List[PlotMethod] = [
    PlotMethod("dijkstra", "Dijkstra", "#7A7A7A"),
    PlotMethod("astar", "A*", "#4C78A8"),
    PlotMethod("greedy", "Greedy", "#E3B23C"),
    PlotMethod("weighted_astar", "Weighted A*", "#72B7B2"),
    PlotMethod("cnn_guided", "Legacy-CNN-guided A*", "#8E6C8A"),
    PlotMethod("unet_guided", "U-Net-guided A*", "#54A24B"),
    PlotMethod("v1_guided", "U-Net+Transformer-guided A*", "#B279A2"),
    PlotMethod("v2_guided", "U-Net+MSWA+GoalCA-guided A*", "#F58518"),
    PlotMethod("transformer_v3", "U-Net+MSWA+GoalCA+GatedSkip-guided A*", "#E45756"),
]
METHODS_BY_KEY: Dict[str, PlotMethod] = {m.key: m for m in METHODS}

TRAP_TYPES = (
    "dead_end",
    "bugtrap",
    "offset_gate",
    "comb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic trap benchmark for street frontend planners.")
    parser.add_argument("--num-cases", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260322)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--corridor-half-width", type=int, default=2)
    parser.add_argument(
        "--trap-types",
        nargs="+",
        default=list(TRAP_TYPES),
        choices=list(TRAP_TYPES),
        help="Subset of trap templates to include.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--heuristic-mode", type=str, default="octile")
    parser.add_argument("--weighted-heuristic-weight", type=float, default=1.5)
    parser.add_argument("--guidance-integration-mode", type=str, default="g_cost")
    parser.add_argument("--guidance-bonus-threshold", type=float, default=0.5)
    parser.add_argument("--allow-corner-cut", action="store_true")
    parser.add_argument("--cnn-ckpt", type=Path, default=BENCH.DEFAULT_CNN_CKPT)
    parser.add_argument("--unet-ckpt", type=Path, default=BENCH.DEFAULT_UNET_CKPT)
    parser.add_argument("--v1-ckpt", type=Path, default=BENCH.DEFAULT_V1_CKPT)
    parser.add_argument("--v2-ckpt", type=Path, default=BENCH.DEFAULT_V2_CKPT)
    parser.add_argument("--v3-ckpt", type=Path, default=BENCH.DEFAULT_V3_CKPT)
    parser.add_argument("--cnn-lambda-guidance", type=float, default=0.4)
    parser.add_argument("--unet-lambda-guidance", type=float, default=0.4)
    parser.add_argument("--cnn-residual-weight", type=float, default=1.25)
    parser.add_argument("--unet-residual-weight", type=float, default=1.25)
    parser.add_argument("--v1-residual-weight", type=float, default=1.25)
    parser.add_argument("--v2-residual-weight", type=float, default=1.25)
    parser.add_argument("--transformer-residual-weight", type=float, default=1.25)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--seed-xy-box-half-extent", type=float, default=0.15)
    parser.add_argument("--skip-seed-collision-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-prefix", type=str, default="trap_frontend_benchmark")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[m.key for m in METHODS],
        choices=[m.key for m in METHODS],
        help="Subset of planners to benchmark.",
    )
    return parser.parse_args()


def _fill_rect(occ: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: float) -> None:
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa = max(0, xa)
    ya = max(0, ya)
    xb = min(occ.shape[1] - 1, xb)
    yb = min(occ.shape[0] - 1, yb)
    occ[ya : yb + 1, xa : xb + 1] = float(value)


def _draw_disk(occ: np.ndarray, cx: float, cy: float, radius: float, value: float) -> None:
    x0 = max(0, int(math.floor(cx - radius - 1)))
    x1 = min(occ.shape[1] - 1, int(math.ceil(cx + radius + 1)))
    y0 = max(0, int(math.floor(cy - radius - 1)))
    y1 = min(occ.shape[0] - 1, int(math.ceil(cy + radius + 1)))
    rr = radius * radius
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= rr:
                occ[y, x] = float(value)


def _draw_segment(occ: np.ndarray, p0: XY, p1: XY, radius: float, value: float) -> None:
    x0, y0 = p0
    x1, y1 = p1
    steps = max(abs(x1 - x0), abs(y1 - y0), 1) * 4
    for t in np.linspace(0.0, 1.0, steps + 1):
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        _draw_disk(occ, x, y, radius, value)


def _carve_polyline(occ: np.ndarray, points: Sequence[XY], radius: float) -> None:
    for idx in range(len(points) - 1):
        _draw_segment(occ, points[idx], points[idx + 1], radius, 0.0)
    for pt in points:
        _draw_disk(occ, pt[0], pt[1], radius, 0.0)


def _add_border(occ: np.ndarray) -> None:
    occ[0, :] = 1.0
    occ[-1, :] = 1.0
    occ[:, 0] = 1.0
    occ[:, -1] = 1.0


def _transform_xy(xy: XY, size: int, rot_k: int, flip_lr: bool) -> XY:
    x, y = int(xy[0]), int(xy[1])
    for _ in range(rot_k % 4):
        x, y = size - 1 - y, x
    if flip_lr:
        x = size - 1 - x
    return x, y


def _apply_transform(occ: np.ndarray, start_xy: XY, goal_xy: XY, rot_k: int, flip_lr: bool) -> Tuple[np.ndarray, XY, XY]:
    out = np.rot90(occ, k=rot_k)
    if flip_lr:
        out = np.fliplr(out)
    size = occ.shape[0]
    start_t = _transform_xy(start_xy, size=size, rot_k=rot_k, flip_lr=flip_lr)
    goal_t = _transform_xy(goal_xy, size=size, rot_k=rot_k, flip_lr=flip_lr)
    return np.ascontiguousarray(out, dtype=np.float32), start_t, goal_t


def _build_dead_end_case(size: int, radius: float, rng: random.Random) -> Tuple[np.ndarray, XY, XY]:
    occ = np.ones((size, size), dtype=np.float32)
    y = 32 + rng.randint(-4, 4)
    detour_y = 50 + rng.randint(-2, 3)
    fake_x = 49 + rng.randint(-2, 2)
    fake_top = 14 + rng.randint(-2, 2)
    start = (6, y)
    goal = (58, y)
    true_pts = [start, (20, y), (20, detour_y), (54, detour_y), (54, y), goal]
    fake_pts = [(20, y), (fake_x, y), (fake_x, fake_top)]
    _carve_polyline(occ, true_pts, radius)
    _carve_polyline(occ, fake_pts, radius)
    for tx, ty in [(30, y - 6), (38, y - 8), (45, y - 10)]:
        _carve_polyline(occ, [(tx, y), (tx, max(ty, 6))], max(1.5, radius - 0.4))
    _draw_disk(occ, fake_x, fake_top, radius + 1.5, 0.0)
    _add_border(occ)
    return occ, start, goal


def _build_bugtrap_case(size: int, radius: float, rng: random.Random) -> Tuple[np.ndarray, XY, XY]:
    occ = np.ones((size, size), dtype=np.float32)
    pocket_x = 48 + rng.randint(-2, 1)
    pocket_top = 12 + rng.randint(-2, 2)
    bottom_y = 50 + rng.randint(-2, 2)
    start = (6, pocket_top + 2)
    goal = (pocket_x + 6, pocket_top)
    # outer approach that must go around and enter from below
    true_pts = [start, (20, pocket_top + 2), (20, bottom_y), (pocket_x + 6, bottom_y), (pocket_x + 6, pocket_top)]
    fake_pts = [(20, pocket_top + 2), (pocket_x - 2, pocket_top + 2), (pocket_x - 2, 30 + rng.randint(-2, 2))]
    _carve_polyline(occ, true_pts, radius)
    _carve_polyline(occ, fake_pts, radius)
    # carve pocket volume but leave opening only from below
    _fill_rect(occ, pocket_x, pocket_top - 1, pocket_x + 10, pocket_top + 18, 0.0)
    _fill_rect(occ, pocket_x, pocket_top - 1, pocket_x + 2, pocket_top + 18, 1.0)
    _fill_rect(occ, pocket_x + 8, pocket_top - 1, pocket_x + 10, pocket_top + 18, 1.0)
    _fill_rect(occ, pocket_x, pocket_top - 1, pocket_x + 10, pocket_top + 2, 1.0)
    _fill_rect(occ, pocket_x + 3, pocket_top + 16, pocket_x + 7, pocket_top + 18, 0.0)
    _draw_disk(occ, goal[0], goal[1], radius + 0.8, 0.0)
    _add_border(occ)
    return occ, start, goal


def _build_offset_gate_case(size: int, radius: float, rng: random.Random) -> Tuple[np.ndarray, XY, XY]:
    occ = np.ones((size, size), dtype=np.float32)
    y = 20 + rng.randint(-4, 4)
    gate_y = 52 + rng.randint(-2, 2)
    wall_x = 40 + rng.randint(-2, 2)
    lure_x = 52 + rng.randint(-2, 2)
    start = (6, y)
    goal = (58, y)
    true_pts = [start, (24, y), (24, gate_y), (56, gate_y), (56, y), goal]
    fake_pts = [(24, y), (lure_x, y), (lure_x, min(size - 8, y + 16))]
    _carve_polyline(occ, true_pts, radius)
    _carve_polyline(occ, fake_pts, radius)
    _fill_rect(occ, wall_x, 4, wall_x + 6, gate_y - 4, 1.0)
    _draw_disk(occ, lure_x, min(size - 8, y + 16), radius + 1.2, 0.0)
    _add_border(occ)
    return occ, start, goal


def _build_comb_case(size: int, radius: float, rng: random.Random) -> Tuple[np.ndarray, XY, XY]:
    occ = np.ones((size, size), dtype=np.float32)
    start_y = 52 + rng.randint(-2, 2)
    top_y = 16 + rng.randint(-2, 2)
    junction_x = 18 + rng.randint(-1, 2)
    trunk_end_x = 54 + rng.randint(-2, 1)
    start = (6, start_y)
    goal = (58, top_y)
    true_pts = [start, (junction_x, start_y), (junction_x, top_y), goal]
    fake_pts = [(junction_x, start_y), (trunk_end_x, start_y)]
    _carve_polyline(occ, true_pts, radius)
    _carve_polyline(occ, fake_pts, radius)
    tooth_specs = [(28, 18), (36, 22), (44, 26), (52, 30)]
    for tx, ty in tooth_specs:
        tx += rng.randint(-1, 1)
        ty += rng.randint(-1, 1)
        _carve_polyline(occ, [(tx, start_y), (tx, ty)], max(1.5, radius - 0.4))
        _draw_disk(occ, tx, ty, radius + 0.6, 0.0)
    _add_border(occ)
    return occ, start, goal


def build_trap_case(trap_type: str, size: int, corridor_half_width: float, rng: random.Random) -> TrapCase:
    builders = {
        "dead_end": _build_dead_end_case,
        "bugtrap": _build_bugtrap_case,
        "offset_gate": _build_offset_gate_case,
        "comb": _build_comb_case,
    }
    if trap_type not in builders:
        raise ValueError(f"unknown trap type: {trap_type}")
    occ, start_xy, goal_xy = builders[trap_type](size, corridor_half_width, rng)
    transform_id = rng.randrange(8)
    rot_k = transform_id % 4
    flip_lr = transform_id >= 4
    occ_t, start_t, goal_t = _apply_transform(occ, start_xy, goal_xy, rot_k=rot_k, flip_lr=flip_lr)
    return TrapCase(occ=occ_t, start_xy=start_t, goal_xy=goal_t, trap_type=trap_type, transform_id=transform_id)


def render_example_cases(cases: Sequence[TrapCase], out_path: Path, max_cases: int = 8) -> None:
    if not cases:
        return
    show = list(cases[:max_cases])
    cols = min(4, len(show))
    rows = int(math.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), facecolor="white")
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, case in zip(axes.flat, show):
        ax.imshow(case.occ, cmap="gray_r", origin="upper", vmin=0.0, vmax=1.0)
        ax.scatter([case.start_xy[0]], [case.start_xy[1]], c="#d73027", marker="x", s=80, linewidths=2.0)
        ax.scatter([case.goal_xy[0]], [case.goal_xy[1]], c="#2ca02c", marker="o", s=70)
        ax.set_title(f"{case.trap_type} | tf={case.transform_id}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_trap_summary(summary: Sequence[Dict[str, float | str]], out_path: Path) -> None:
    labels = [str(row["method_label"]) for row in summary]
    colors = [METHODS_BY_KEY[str(row["method"])].color for row in summary]
    expanded = [float(row["expanded_nodes"]) for row in summary]
    search_ms = [float(row["frontend_search_ms"]) for row in summary]
    raw_length = [float(row["raw_length_m"]) for row in summary]
    success_rate = [100.0 * float(row["success_rate"]) for row in summary]
    x = np.arange(len(labels))

    font_prop = None
    if CJK_FONT_PATH.exists():
        from matplotlib import font_manager as fm

        font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.6), facecolor="#fcfcfb")
    axes = axes.reshape(-1)
    metrics = [
        (expanded, "平均扩展节点", "nodes"),
        (search_ms, "平均搜索时间", "ms"),
        (raw_length, "平均路径长度", "m"),
        (success_rate, "前端成功率", "%"),
    ]
    for ax, (vals, title, ylabel) in zip(axes, metrics):
        ax.bar(x, vals, color=colors, alpha=0.92)
        ax.set_title(title, fontproperties=font_prop)
        ax.set_ylabel(ylabel, fontproperties=font_prop)
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color="#d9dde3", linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#a9adb3")
        for i, v in enumerate(vals):
            if title == "前端成功率":
                txt = f"{v:.1f}%"
            elif v < 100:
                txt = f"{v:.2f}"
            else:
                txt = f"{v:.1f}"
            ax.text(i, v + (max(vals) * 0.03 if max(vals) > 0 else 0.1), txt, ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if font_prop is not None:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_traptype_greedy_gap(rows: Sequence[Dict[str, float | int | str]], out_path: Path) -> None:
    per_type: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for row in rows:
        trap_type = str(row["trap_type"])
        method = str(row["method"])
        metrics = per_type.setdefault(trap_type, {}).setdefault(method, {"expanded": [], "raw_length": []})
        metrics["expanded"].append(float(row["expanded_nodes"]))
        metrics["raw_length"].append(float(row["raw_length_m"]))

    trap_types = [tt for tt in TRAP_TYPES if tt in per_type]
    astar_expanded = [float(np.mean(per_type[tt].get("astar", {"expanded": [math.nan]})["expanded"])) for tt in trap_types]
    greedy_expanded = [float(np.mean(per_type[tt].get("greedy", {"expanded": [math.nan]})["expanded"])) for tt in trap_types]
    astar_length = [float(np.mean(per_type[tt].get("astar", {"raw_length": [math.nan]})["raw_length"])) for tt in trap_types]
    greedy_length = [float(np.mean(per_type[tt].get("greedy", {"raw_length": [math.nan]})["raw_length"])) for tt in trap_types]
    x = np.arange(len(trap_types))
    width = 0.34

    font_prop = None
    if CJK_FONT_PATH.exists():
        from matplotlib import font_manager as fm

        font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), facecolor="#fcfcfb")
    panels = [
        (axes[0], astar_expanded, greedy_expanded, "各陷阱类型下 Greedy 与 A* 的扩展节点对比", "nodes"),
        (axes[1], astar_length, greedy_length, "各陷阱类型下 Greedy 与 A* 的路径长度对比", "m"),
    ]
    for ax, astar_vals, greedy_vals, title, ylabel in panels:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color="#d9dde3", linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        ax.bar(x - width / 2, astar_vals, width=width, color="#4C78A8", label="A*")
        ax.bar(x + width / 2, greedy_vals, width=width, color="#E3B23C", label="Greedy")
        ax.set_title(title, fontproperties=font_prop)
        ax.set_ylabel(ylabel, fontproperties=font_prop)
        ax.set_xticks(x)
        ax.set_xticklabels(trap_types)
        leg = ax.legend(prop=font_prop)
        if font_prop is not None:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)
            for t in leg.get_texts():
                t.set_fontproperties(font_prop)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # benchmark_case expects these fields to exist even in frontend-only mode
    args.skip_smoother = True
    args.smoother_cli = BENCH.DEMO.default_smoother_cli()

    selected_methods = [PlotMethod(m.key, m.label, m.color) for m in METHODS if m.key in args.methods]

    print(f"loading_cnn_model={args.cnn_ckpt}")
    cnn_model = BENCH.load_guidance_encoder(args.cnn_ckpt, device=args.device)
    print(f"loading_unet_model={args.unet_ckpt}")
    unet_model = BENCH.load_guidance_encoder(args.unet_ckpt, device=args.device)
    print(f"loading_v1_model={args.v1_ckpt}")
    v1_model = BENCH.load_guidance_encoder(args.v1_ckpt, device=args.device)
    print(f"loading_v2_model={args.v2_ckpt}")
    v2_model = BENCH.load_guidance_encoder(args.v2_ckpt, device=args.device)
    print(f"loading_v3_model={args.v3_ckpt}")
    v3_model = BENCH.load_guidance_encoder(args.v3_ckpt, device=args.device)

    warm_occ = np.zeros((args.grid_size, args.grid_size), dtype=np.float32)
    warm_start = (4, 4)
    warm_goal = (args.grid_size - 5, args.grid_size - 5)
    _ = BENCH.infer_model_output_loaded(cnn_model, warm_occ, warm_start, warm_goal, args.device)
    _ = BENCH.infer_model_output_loaded(unet_model, warm_occ, warm_start, warm_goal, args.device)
    _ = BENCH.infer_model_output_loaded(v1_model, warm_occ, warm_start, warm_goal, args.device)
    _ = BENCH.infer_model_output_loaded(v2_model, warm_occ, warm_start, warm_goal, args.device)
    _ = BENCH.infer_model_output_loaded(v3_model, warm_occ, warm_start, warm_goal, args.device)
    BENCH.maybe_cuda_sync(args.device)

    rows: List[Dict[str, float | int | str]] = []
    case_meta: List[Dict[str, int | str]] = []
    cases: List[TrapCase] = []
    rng = random.Random(args.seed)

    attempts = 0
    case_id = 0
    while case_id < args.num_cases:
        attempts += 1
        if attempts > args.num_cases * 20:
            raise RuntimeError("failed to generate enough valid trap cases")
        trap_type = args.trap_types[case_id % len(args.trap_types)]
        case = build_trap_case(
            trap_type=trap_type,
            size=args.grid_size,
            corridor_half_width=float(args.corridor_half_width),
            rng=rng,
        )
        # validate with A* to ensure solvable under benchmark settings
        astar_check = BENCH.astar_8conn_stats(
            occ_map=case.occ,
            start_xy=case.start_xy,
            goal_xy=case.goal_xy,
            lambda_guidance=0.0,
            heuristic_mode=args.heuristic_mode,
            heuristic_weight=1.0,
            allow_corner_cut=args.allow_corner_cut,
        )
        if not astar_check.success:
            continue

        case_dir = args.output_dir / "cases" / f"case_{case_id:03d}_{case.trap_type}_tf{case.transform_id}"
        case_rows = BENCH.benchmark_case(
            case_id=case_id,
            map_index=-1,
            occ=case.occ,
            start_xy=case.start_xy,
            goal_xy=case.goal_xy,
            cnn_model=cnn_model,
            unet_model=unet_model,
            v1_model=v1_model,
            v2_model=v2_model,
            v3_model=v3_model,
            args=args,
            case_dir=case_dir,
            methods=selected_methods,
        )
        for row in case_rows:
            row["trap_type"] = case.trap_type
            row["transform_id"] = case.transform_id
        rows.extend(case_rows)
        case_meta.append(
            {
                "case_id": case_id,
                "trap_type": case.trap_type,
                "transform_id": case.transform_id,
                "start_x": case.start_xy[0],
                "start_y": case.start_xy[1],
                "goal_x": case.goal_xy[0],
                "goal_y": case.goal_xy[1],
            }
        )
        cases.append(case)
        print(
            f"completed_case={case_id} trap={case.trap_type} tf={case.transform_id} start={case.start_xy} goal={case.goal_xy}",
            flush=True,
        )
        case_id += 1

    summary = BENCH.aggregate(rows, selected_methods)

    raw_csv = args.output_dir / f"{args.paper_prefix}_case_metrics.csv"
    summary_csv = args.output_dir / f"{args.paper_prefix}_summary.csv"
    summary_md = args.output_dir / f"{args.paper_prefix}_summary.md"
    cases_csv = args.output_dir / f"{args.paper_prefix}_cases.csv"
    summary_png = args.output_dir / f"{args.paper_prefix}_summary.png"
    traptype_png = args.output_dir / f"{args.paper_prefix}_greedy_gap_by_trap.png"
    gallery_png = args.output_dir / f"{args.paper_prefix}_case_gallery.png"

    BENCH.write_csv(raw_csv, rows)
    BENCH.write_csv(summary_csv, summary)
    BENCH.write_csv(cases_csv, case_meta)
    BENCH.write_summary_md(summary_md, summary)
    plot_trap_summary(summary, summary_png)
    plot_traptype_greedy_gap(rows, traptype_png)
    render_example_cases(cases, gallery_png, max_cases=min(8, len(cases)))

    print(f"saved_case_metrics={raw_csv}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_summary_md={summary_md}")
    print(f"saved_cases_csv={cases_csv}")
    print(f"saved_summary_png={summary_png}")
    print(f"saved_greedy_gap_png={traptype_png}")
    print(f"saved_case_gallery_png={gallery_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
