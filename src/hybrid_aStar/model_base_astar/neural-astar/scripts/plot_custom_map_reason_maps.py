from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark_custom_map_v3_resize64 import _decode_prediction  # type: ignore
from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.coords import make_one_hot_xy
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map


DEFAULT_CKPT = (
    ROOT
    / "outputs"
    / "model_guidance_grid_mpd_unet_transformer_v3_rebuiltexpert_w035_sd5_p2_v2_policyfix_gpu_v1"
    / "best_eval_snapshot_epoch1.pt"
)
SCENE_COLORS = {
    "raw": "#4C78A8",
    "final": "#D55E00",
    "start": "#22c55e",
    "goal": "#ef4444",
}
FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot custom-map inference / confidence / cost / effective-guidance figures.")
    p.add_argument("--request-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--target-height", type=int, default=64)
    p.add_argument("--target-width", type=int, default=64)
    p.add_argument("--residual-weight", type=float, default=0.15)
    p.add_argument("--confidence-mode", type=str, default="learned")
    p.add_argument("--confidence-kernel", type=int, default=3)
    p.add_argument("--confidence-strength", type=float, default=0.75)
    p.add_argument("--confidence-min", type=float, default=0.1)
    p.add_argument("--search-clearance-weight", type=float, default=0.2)
    p.add_argument("--search-clearance-safe-distance", type=float, default=3.0)
    p.add_argument("--search-clearance-power", type=float, default=2.0)
    p.add_argument("--dpi", type=int, default=260)
    p.add_argument("--also-save-64x64", action="store_true")
    return p.parse_args()


def _font(size: float, weight: str | None = None):
    if FONT_PATH.exists():
        return fm.FontProperties(fname=str(FONT_PATH), size=size, weight=weight)
    return None


def _load_request(path: Path) -> tuple[np.ndarray, float, float, float, tuple[int, int], tuple[int, int]]:
    req = json.loads(path.read_text(encoding="utf-8"))
    occ = np.asarray(req["occupancy"], dtype=np.float32)
    origin_x = float(req["origin_x"])
    origin_y = float(req["origin_y"])
    resolution = float(req["resolution"])
    start_world = tuple(req["start_world"])
    goal_world = tuple(req["goal_world"])
    start_xy = (
        int(round((start_world[0] - origin_x) / resolution - 0.5)),
        int(round((start_world[1] - origin_y) / resolution - 0.5)),
    )
    goal_xy = (
        int(round((goal_world[0] - origin_x) / resolution - 0.5)),
        int(round((goal_world[1] - origin_y) / resolution - 0.5)),
    )
    return occ, origin_x, origin_y, resolution, start_xy, goal_xy


def _infer_resize64(
    *,
    model: torch.nn.Module,
    device: str,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    target_h: int,
    target_w: int,
    confidence_mode: str,
    confidence_kernel: int,
    confidence_strength: float,
    confidence_min: float,
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
        kernel_size=int(confidence_kernel),
        strength=float(confidence_strength),
        min_confidence=float(confidence_min),
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
    return infer_ms, residual, conf


def _infer_resize64_dual(
    *,
    model: torch.nn.Module,
    device: str,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    target_h: int,
    target_w: int,
    confidence_mode: str,
    confidence_kernel: int,
    confidence_strength: float,
    confidence_min: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], np.ndarray]:
    h, w = occ.shape
    sx = int(np.clip(round(start_xy[0] * (target_w - 1) / max(w - 1, 1)), 0, target_w - 1))
    sy = int(np.clip(round(start_xy[1] * (target_h - 1) / max(h - 1, 1)), 0, target_h - 1))
    gx = int(np.clip(round(goal_xy[0] * (target_w - 1) / max(w - 1, 1)), 0, target_w - 1))
    gy = int(np.clip(round(goal_xy[1] * (target_h - 1) / max(h - 1, 1)), 0, target_h - 1))
    start = make_one_hot_xy(sx, sy, target_w, target_h)
    goal = make_one_hot_xy(gx, gy, target_w, target_h)

    occ_t = torch.from_numpy(occ[None, None]).to(device)
    occ_small_t = F.interpolate(occ_t, size=(target_h, target_w), mode="nearest")
    occ_small = occ_small_t[0, 0].detach().cpu().numpy().astype(np.float32)
    clearance_small = build_clearance_input_map(
        occ_map=occ_small,
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
            occ_small_t,
            start_t,
            goal_t,
            start_yaw=torch.zeros(1, device=device, dtype=occ_small_t.dtype),
            goal_yaw=torch.zeros(1, device=device, dtype=occ_small_t.dtype),
            extra_input_maps=extra_input_t,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

    residual_small, learned_conf_small = _decode_prediction(model, out)
    conf_small = resolve_residual_confidence_map(
        mode=str(confidence_mode),
        occ_map=occ_small,
        residual_map=residual_small,
        learned_confidence_map=learned_conf_small,
        kernel_size=int(confidence_kernel),
        strength=float(confidence_strength),
        min_confidence=float(confidence_min),
    )
    residual_small[occ_small > 0.5] = 0.0
    conf_small = np.clip(conf_small, 0.0, 1.0).astype(np.float32)
    conf_small[occ_small > 0.5] = 0.0

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
    return infer_ms, residual_small, conf_small, residual, conf, (sx, sy), (gx, gy), occ_small


def _base_scene(ax: plt.Axes, occ: np.ndarray, start_xy: tuple[int, int], goal_xy: tuple[int, int]) -> None:
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.scatter([start_xy[0]], [start_xy[1]], c=SCENE_COLORS["start"], s=52, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=SCENE_COLORS["goal"], s=60, marker="x")
    ax.set_axis_off()


def _plot_path_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    path: list[tuple[int, int]] | None,
    color: str,
    title: str,
) -> None:
    _base_scene(ax, occ, start_xy, goal_xy)
    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, linewidth=2.3, alpha=0.98)
    title_fp = _font(11)
    if title_fp is not None:
        ax.set_title(title, fontproperties=title_fp, pad=8)
    else:
        ax.set_title(title, fontsize=11, pad=8)


def _plot_map_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    data: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    path: list[tuple[int, int]] | None,
    color: str,
) -> None:
    _base_scene(ax, occ, start_xy, goal_xy)
    masked = np.ma.masked_where(occ > 0.5, data)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.90, interpolation="nearest")
    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.96)
    title_fp = _font(11)
    if title_fp is not None:
        ax.set_title(title, fontproperties=title_fp, pad=8)
    else:
        ax.set_title(title, fontsize=11, pad=8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)


def _save_metrics(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    occ, origin_x, origin_y, resolution, start_xy, goal_xy = _load_request(args.request_json)
    model = load_guidance_encoder(args.ckpt, device=device)
    infer_ms, residual_map, confidence_map = _infer_resize64(
        model=model,
        device=device,
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        target_h=int(args.target_height),
        target_w=int(args.target_width),
        confidence_mode=str(args.confidence_mode),
        confidence_kernel=int(args.confidence_kernel),
        confidence_strength=float(args.confidence_strength),
        confidence_min=float(args.confidence_min),
    )
    effective_map = float(args.residual_weight) * residual_map * confidence_map

    residual_small = None
    confidence_small = None
    effective_small = None
    occ_small = None
    start_small = None
    goal_small = None
    if args.also_save_64x64:
        (
            _infer_ms_small,
            residual_small,
            confidence_small,
            _residual_full2,
            _confidence_full2,
            start_small,
            goal_small,
            occ_small,
        ) = _infer_resize64_dual(
            model=model,
            device=device,
            occ=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            target_h=int(args.target_height),
            target_w=int(args.target_width),
            confidence_mode=str(args.confidence_mode),
            confidence_kernel=int(args.confidence_kernel),
            confidence_strength=float(args.confidence_strength),
            confidence_min=float(args.confidence_min),
        )
        effective_small = float(args.residual_weight) * residual_small * confidence_small

    raw_t0 = time.perf_counter()
    raw_stats = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        heuristic_residual_map=residual_map,
        residual_confidence_map=confidence_map,
        residual_weight=float(args.residual_weight),
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
    )
    raw_ms = (time.perf_counter() - raw_t0) * 1000.0

    final_t0 = time.perf_counter()
    final_stats = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        heuristic_residual_map=residual_map,
        residual_confidence_map=confidence_map,
        residual_weight=float(args.residual_weight),
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
        clearance_weight=float(args.search_clearance_weight),
        clearance_safe_distance=float(args.search_clearance_safe_distance),
        clearance_power=float(args.search_clearance_power),
        clearance_integration_mode="g_cost",
    )
    final_ms = (time.perf_counter() - final_t0) * 1000.0

    if not raw_stats.success or raw_stats.path is None:
        raise RuntimeError("raw V3 inference search failed")
    if not final_stats.success or final_stats.path is None:
        raise RuntimeError("final V3 search failed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_raw = args.output_dir / "09_raw_inference_planning.png"
    out_conf = args.output_dir / "10_confidence_map.png"
    out_cost = args.output_dir / "11_cost_map.png"
    out_effective = args.output_dir / "12_effective_cost_path.png"

    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.8))
    _plot_path_panel(
        ax,
        occ,
        start_xy,
        goal_xy,
        raw_stats.path,
        SCENE_COLORS["raw"],
        f"原始推理结果\nexpanded={raw_stats.expanded_nodes}  path={path_length_8conn(raw_stats.path) * resolution:.2f} m  time={infer_ms + raw_ms:.2f} ms",
    )
    fig.tight_layout()
    fig.savefig(out_raw, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.8))
    _plot_map_panel(
        ax,
        occ,
        start_xy,
        goal_xy,
        confidence_map,
        "置信度图",
        "viridis",
        0.0,
        1.0,
        raw_stats.path,
        SCENE_COLORS["raw"],
    )
    fig.tight_layout()
    fig.savefig(out_conf, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    pos_res = residual_map[residual_map > 0.0]
    res_vmax = float(np.percentile(pos_res, 98)) if pos_res.size > 0 else 1.0
    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.8))
    _plot_map_panel(
        ax,
        occ,
        start_xy,
        goal_xy,
        residual_map,
        "代价图",
        "cividis",
        0.0,
        max(res_vmax, 1e-6),
        raw_stats.path,
        SCENE_COLORS["raw"],
    )
    fig.tight_layout()
    fig.savefig(out_cost, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    pos_eff = effective_map[effective_map > 0.0]
    eff_vmax = float(np.percentile(pos_eff, 98)) if pos_eff.size > 0 else 1.0
    fig, ax = plt.subplots(1, 1, figsize=(8.3, 4.8))
    _plot_map_panel(
        ax,
        occ,
        start_xy,
        goal_xy,
        effective_map,
        "最终代价路径图",
        "magma",
        0.0,
        max(eff_vmax, 1e-6),
        final_stats.path,
        SCENE_COLORS["final"],
    )
    fig.tight_layout()
    fig.savefig(out_effective, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    _save_metrics(
        args.output_dir / "09_12_reason_map_metrics.json",
        {
            "request_json": str(args.request_json.resolve()),
            "ckpt": str(args.ckpt.resolve()),
            "start_grid": list(start_xy),
            "goal_grid": list(goal_xy),
            "infer_ms": float(infer_ms),
            "raw": {
                "expanded_nodes": int(raw_stats.expanded_nodes),
                "search_time_ms": float(raw_ms),
                "path_length_m": float(path_length_8conn(raw_stats.path, diagonal_cost=math.sqrt(2.0)) * resolution),
            },
            "final": {
                "expanded_nodes": int(final_stats.expanded_nodes),
                "search_time_ms": float(final_ms),
                "path_length_m": float(path_length_8conn(final_stats.path, diagonal_cost=math.sqrt(2.0)) * resolution),
            },
            "residual_max": float(np.max(residual_map)),
            "confidence_mean": float(np.mean(confidence_map[occ <= 0.5])),
            "effective_max": float(np.max(effective_map)),
        },
    )

    if args.also_save_64x64 and occ_small is not None and residual_small is not None and confidence_small is not None and effective_small is not None and start_small is not None and goal_small is not None:
        raw_small_t0 = time.perf_counter()
        raw_small_stats = astar_8conn_stats(
            occ_map=occ_small,
            start_xy=start_small,
            goal_xy=goal_small,
            heuristic_mode="octile",
            heuristic_weight=1.0,
            heuristic_residual_map=residual_small,
            residual_confidence_map=confidence_small,
            residual_weight=float(args.residual_weight),
            diagonal_cost=math.sqrt(2.0),
            allow_corner_cut=False,
        )
        raw_small_ms = (time.perf_counter() - raw_small_t0) * 1000.0

        final_small_t0 = time.perf_counter()
        final_small_stats = astar_8conn_stats(
            occ_map=occ_small,
            start_xy=start_small,
            goal_xy=goal_small,
            heuristic_mode="octile",
            heuristic_weight=1.0,
            heuristic_residual_map=residual_small,
            residual_confidence_map=confidence_small,
            residual_weight=float(args.residual_weight),
            diagonal_cost=math.sqrt(2.0),
            allow_corner_cut=False,
            clearance_weight=float(args.search_clearance_weight),
            clearance_safe_distance=float(args.search_clearance_safe_distance),
            clearance_power=float(args.search_clearance_power),
            clearance_integration_mode="g_cost",
        )
        final_small_ms = (time.perf_counter() - final_small_t0) * 1000.0

        if raw_small_stats.success and raw_small_stats.path is not None:
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
            _plot_path_panel(
                ax,
                occ_small,
                start_small,
                goal_small,
                raw_small_stats.path,
                SCENE_COLORS["raw"],
                f"64×64 原始推理\nexpanded={raw_small_stats.expanded_nodes}  time={infer_ms + raw_small_ms:.2f} ms",
            )
            fig.tight_layout()
            fig.savefig(args.output_dir / "13_raw_inference_planning_64x64.png", dpi=int(args.dpi), bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
        _plot_map_panel(
            ax,
            occ_small,
            start_small,
            goal_small,
            confidence_small,
            "64×64 置信度图",
            "viridis",
            0.0,
            1.0,
            raw_small_stats.path if raw_small_stats.success else None,
            SCENE_COLORS["raw"],
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / "14_confidence_map_64x64.png", dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

        pos_res_small = residual_small[residual_small > 0.0]
        res_small_vmax = float(np.percentile(pos_res_small, 98)) if pos_res_small.size > 0 else 1.0
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
        _plot_map_panel(
            ax,
            occ_small,
            start_small,
            goal_small,
            residual_small,
            "64×64 代价图",
            "cividis",
            0.0,
            max(res_small_vmax, 1e-6),
            raw_small_stats.path if raw_small_stats.success else None,
            SCENE_COLORS["raw"],
        )
        fig.tight_layout()
        fig.savefig(args.output_dir / "15_cost_map_64x64.png", dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)

        if final_small_stats.success and final_small_stats.path is not None:
            pos_eff_small = effective_small[effective_small > 0.0]
            eff_small_vmax = float(np.percentile(pos_eff_small, 98)) if pos_eff_small.size > 0 else 1.0
            fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))
            _plot_map_panel(
                ax,
                occ_small,
                start_small,
                goal_small,
                effective_small,
                "64×64 最终代价路径图",
                "magma",
                0.0,
                max(eff_small_vmax, 1e-6),
                final_small_stats.path,
                SCENE_COLORS["final"],
            )
            fig.tight_layout()
            fig.savefig(args.output_dir / "16_effective_cost_path_64x64.png", dpi=int(args.dpi), bbox_inches="tight")
            plt.close(fig)

        _save_metrics(
            args.output_dir / "13_16_reason_map_metrics_64x64.json",
            {
                "request_json": str(args.request_json.resolve()),
                "start_grid_64x64": list(start_small),
                "goal_grid_64x64": list(goal_small),
                "infer_ms": float(infer_ms),
                "raw_64x64": {
                    "expanded_nodes": int(raw_small_stats.expanded_nodes),
                    "search_time_ms": float(raw_small_ms),
                },
                "final_64x64": {
                    "expanded_nodes": int(final_small_stats.expanded_nodes),
                    "search_time_ms": float(final_small_ms),
                },
                "residual_max_64x64": float(np.max(residual_small)),
                "confidence_mean_64x64": float(np.mean(confidence_small[occ_small <= 0.5])),
                "effective_max_64x64": float(np.max(effective_small)),
            },
        )


if __name__ == "__main__":
    main()
