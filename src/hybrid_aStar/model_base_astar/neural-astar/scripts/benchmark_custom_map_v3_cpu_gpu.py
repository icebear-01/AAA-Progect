from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.coords import make_one_hot_xy
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark tuned V3 on a custom map with CPU/GPU.")
    p.add_argument("--map-json", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-overlay", type=Path, required=True)
    p.add_argument("--output-cpu-csv", type=Path, required=True)
    p.add_argument("--output-gpu-csv", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--residual-weight", type=float, default=0.15)
    p.add_argument("--confidence-mode", type=str, default="learned")
    return p.parse_args()


def _load_map_request(path: Path) -> tuple[np.ndarray, float, float, float, tuple[int, int], tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        req = json.load(f)
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


def _infer_once(
    *,
    model: torch.nn.Module,
    device: str,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray | None]:
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

    pred = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32) if out.scale_map is not None else None
    learned_conf = None
    if out.confidence_map is not None:
        learned_conf = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
        learned_conf = learned_conf[0] if learned_conf.shape[0] == 1 else np.min(learned_conf, axis=0).astype(np.float32)

    pred = pred[0] if pred.shape[0] == 1 else np.min(pred, axis=0).astype(np.float32)
    pred = decode_residual_prediction_np(pred, transform=str(getattr(model, "residual_target_transform", "none")))
    if scale is not None:
        scale = scale[0] if scale.shape[0] == 1 else np.min(scale, axis=0).astype(np.float32)
    pred = apply_residual_scale_np(pred, scale)
    residual = np.maximum(pred, 0.0).astype(np.float32)
    return infer_ms, residual, learned_conf


def _benchmark_device(
    *,
    ckpt: Path,
    device: str,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    residual_weight: float,
    confidence_mode: str,
    warmup: int,
    repeat: int,
    resolution: float,
) -> dict:
    model = load_guidance_encoder(ckpt, device=device)
    for _ in range(max(warmup, 0)):
        _infer_once(model=model, device=device, occ=occ, start_xy=start_xy, goal_xy=goal_xy)

    infer_times: list[float] = []
    residual = None
    learned_conf = None
    for _ in range(max(repeat, 1)):
        infer_ms, residual, learned_conf = _infer_once(
            model=model,
            device=device,
            occ=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
        )
        infer_times.append(infer_ms)

    confidence = resolve_residual_confidence_map(
        mode=confidence_mode,
        occ_map=occ,
        residual_map=residual,
        learned_confidence_map=learned_conf,
    )

    st0 = time.perf_counter()
    stats = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        heuristic_residual_map=residual,
        residual_confidence_map=confidence,
        residual_weight=float(residual_weight),
        diagonal_cost=math.sqrt(2.0),
        allow_corner_cut=False,
    )
    search_ms = (time.perf_counter() - st0) * 1000.0
    path_length = path_length_8conn(stats.path, diagonal_cost=math.sqrt(2.0)) if stats.path else float("inf")
    return {
        "device": device,
        "infer_ms_avg": float(np.mean(infer_times)),
        "infer_ms_std": float(np.std(infer_times)),
        "infer_ms_min": float(np.min(infer_times)),
        "infer_ms_max": float(np.max(infer_times)),
        "search_time_ms": float(search_ms),
        "end_to_end_ms": float(np.mean(infer_times) + search_ms),
        "expanded_nodes": int(stats.expanded_nodes),
        "path_length_m": float(path_length * resolution),
        "path_points": len(stats.path) if stats.path else 0,
        "path": stats.path,
    }


def _write_path_csv(path: list[tuple[int, int]], out_csv: Path, origin_x: float, origin_y: float, resolution: float) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_x", "grid_y", "world_x", "world_y"])
        for x, y in path:
            wx = origin_x + (x + 0.5) * resolution
            wy = origin_y + (y + 0.5) * resolution
            writer.writerow([x, y, f"{wx:.6f}", f"{wy:.6f}"])


def _plot_overlay(
    *,
    occ: np.ndarray,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    baseline_path: list[tuple[int, int]] | None,
    cpu_path: list[tuple[int, int]] | None,
    gpu_path: list[tuple[int, int]] | None,
    baseline_expanded: int,
    cpu_expanded: int,
    gpu_expanded: int | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=220)
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if baseline_path:
        ax.plot([p[0] for p in baseline_path], [p[1] for p in baseline_path], color="#38bdf8", linewidth=2.5, label=f"A* ({baseline_expanded})")
    if cpu_path:
        ax.plot([p[0] for p in cpu_path], [p[1] for p in cpu_path], color="#f97316", linewidth=2.1, label=f"V3 CPU ({cpu_expanded})")
    if gpu_path and gpu_expanded is not None:
        ax.plot([p[0] for p in gpu_path], [p[1] for p in gpu_path], color="#a855f7", linewidth=1.8, linestyle="--", label=f"V3 GPU ({gpu_expanded})")
    ax.scatter([start_xy[0]], [start_xy[1]], c="#22c55e", s=48, marker="o", label="start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#ef4444", s=60, marker="x", label="goal")
    ax.set_title("Custom Map: A* vs Tuned V3 on CPU/GPU", fontsize=14)
    ax.legend(loc="lower right", frameon=True)
    ax.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    occ, origin_x, origin_y, resolution, start_xy, goal_xy = _load_map_request(args.map_json)

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
    baseline_len = path_length_8conn(baseline.path, diagonal_cost=math.sqrt(2.0)) if baseline.path else float("inf")

    cpu_res = _benchmark_device(
        ckpt=args.ckpt,
        device="cpu",
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        residual_weight=float(args.residual_weight),
        confidence_mode=str(args.confidence_mode),
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        resolution=float(resolution),
    )

    gpu_res = None
    if torch.cuda.is_available():
        gpu_res = _benchmark_device(
            ckpt=args.ckpt,
            device="cuda",
            occ=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            residual_weight=float(args.residual_weight),
            confidence_mode=str(args.confidence_mode),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            resolution=float(resolution),
        )

    _write_path_csv(cpu_res["path"], args.output_cpu_csv, origin_x, origin_y, resolution)
    if gpu_res is not None:
        _write_path_csv(gpu_res["path"], args.output_gpu_csv, origin_x, origin_y, resolution)

    _plot_overlay(
        occ=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        baseline_path=baseline.path,
        cpu_path=cpu_res["path"],
        gpu_path=(gpu_res["path"] if gpu_res is not None else None),
        baseline_expanded=int(baseline.expanded_nodes),
        cpu_expanded=int(cpu_res["expanded_nodes"]),
        gpu_expanded=(int(gpu_res["expanded_nodes"]) if gpu_res is not None else None),
        output_path=args.output_overlay,
    )

    summary = {
        "map_json": str(args.map_json.resolve()),
        "ckpt": str(args.ckpt.resolve()),
        "start_grid": list(start_xy),
        "goal_grid": list(goal_xy),
        "residual_weight": float(args.residual_weight),
        "confidence_mode": str(args.confidence_mode),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "baseline_astar": {
            "expanded_nodes": int(baseline.expanded_nodes),
            "search_time_ms": float(baseline_ms),
            "path_length_m": float(baseline_len * resolution),
            "path_points": len(baseline.path) if baseline.path else 0,
        },
        "v3_cpu": {k: v for k, v in cpu_res.items() if k != "path"},
        "v3_gpu": ({k: v for k, v in gpu_res.items() if k != "path"} if gpu_res is not None else None),
        "gpu_speedup_vs_cpu_infer": (
            float(cpu_res["infer_ms_avg"] / gpu_res["infer_ms_avg"])
            if gpu_res is not None and gpu_res["infer_ms_avg"] > 0.0
            else None
        ),
        "overlay": str(args.output_overlay.resolve()),
        "cpu_csv": str(args.output_cpu_csv.resolve()),
        "gpu_csv": (str(args.output_gpu_csv.resolve()) if gpu_res is not None else None),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
