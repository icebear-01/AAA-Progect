from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    p = argparse.ArgumentParser(description="Benchmark native V3 inference vs resize-to-64x64 inference.")
    p.add_argument("--map-json", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--target-height", type=int, default=64)
    p.add_argument("--target-width", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--residual-weight", type=float, default=0.15)
    p.add_argument("--confidence-mode", type=str, default="learned")
    return p.parse_args()


def _load_request(path: Path) -> tuple[np.ndarray, float, float, float, tuple[int, int], tuple[int, int]]:
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


def _decode_prediction(model: torch.nn.Module, out) -> tuple[np.ndarray, np.ndarray | None]:
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
    return residual, learned_conf


def main() -> None:
    args = parse_args()

    occ, _, _, resolution, start_xy, goal_xy = _load_request(args.map_json)
    h, w = occ.shape
    target_h = int(args.target_height)
    target_w = int(args.target_width)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_guidance_encoder(args.ckpt, device=device)

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

    def infer_native() -> tuple[float, np.ndarray, np.ndarray | None]:
        start = make_one_hot_xy(start_xy[0], start_xy[1], w, h)
        goal = make_one_hot_xy(goal_xy[0], goal_xy[1], w, h)
        clearance_input = build_clearance_input_map(
            occ_map=occ,
            clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
        )[None, None].astype(np.float32)
        occ_t = torch.from_numpy(occ[None, None]).to(device)
        start_t = torch.from_numpy(start[None, None]).to(device)
        goal_t = torch.from_numpy(goal[None, None]).to(device)
        extra_input_t = torch.from_numpy(clearance_input).to(device) if int(getattr(model, "extra_input_channels", 0)) > 0 else None
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
            mode=str(args.confidence_mode),
            occ_map=occ,
            residual_map=residual,
            learned_confidence_map=learned_conf,
        )
        return infer_ms, residual, conf

    def infer_resize() -> tuple[float, np.ndarray, np.ndarray | None]:
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
        extra_input_t = torch.from_numpy(clearance_small).to(device) if int(getattr(model, "extra_input_channels", 0)) > 0 else None

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
            mode=str(args.confidence_mode),
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
        conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
        residual[occ > 0.5] = 0.0
        conf[occ > 0.5] = 0.0
        return infer_ms, residual, conf

    def bench(fn) -> dict:
        for _ in range(max(int(args.warmup), 0)):
            fn()
        infer_times: list[float] = []
        residual = None
        conf = None
        for _ in range(max(int(args.repeat), 1)):
            infer_ms, residual, conf = fn()
            infer_times.append(infer_ms)
        st0 = time.perf_counter()
        stats = astar_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            heuristic_mode="octile",
            heuristic_weight=1.0,
            heuristic_residual_map=residual,
            residual_confidence_map=conf,
            residual_weight=float(args.residual_weight),
            diagonal_cost=math.sqrt(2.0),
            allow_corner_cut=False,
        )
        search_ms = (time.perf_counter() - st0) * 1000.0
        path_len = path_length_8conn(stats.path, diagonal_cost=math.sqrt(2.0)) if stats.path else float("inf")
        return {
            "infer_ms_avg": float(np.mean(infer_times)),
            "infer_ms_std": float(np.std(infer_times)),
            "search_time_ms": float(search_ms),
            "end_to_end_ms": float(np.mean(infer_times) + search_ms),
            "expanded_nodes": int(stats.expanded_nodes),
            "path_length_m": float(path_len * resolution),
            "path_points": len(stats.path) if stats.path else 0,
        }

    native = bench(infer_native)
    resize64 = bench(infer_resize)

    summary = {
        "device": device,
        "map_json": str(args.map_json.resolve()),
        "ckpt": str(args.ckpt.resolve()),
        "start_grid": list(start_xy),
        "goal_grid": list(goal_xy),
        "target_size": [target_h, target_w],
        "baseline_astar": {
            "expanded_nodes": int(baseline.expanded_nodes),
            "search_time_ms": float(baseline_ms),
            "path_length_m": float(baseline_len * resolution),
            "path_points": len(baseline.path) if baseline.path else 0,
        },
        "native_fullres": native,
        "resize64_then_upsample": resize64,
        "delta_resize64_vs_native": {
            "expanded_nodes": int(resize64["expanded_nodes"] - native["expanded_nodes"]),
            "path_length_m": float(resize64["path_length_m"] - native["path_length_m"]),
            "infer_ms_avg": float(resize64["infer_ms_avg"] - native["infer_ms_avg"]),
            "end_to_end_ms": float(resize64["end_to_end_ms"] - native["end_to_end_ms"]),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
