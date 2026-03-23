#!/usr/bin/env python3
"""Batch benchmark for street frontend planners + shared backend smoother."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
HYBRID_ASTAR_ROOT = SCRIPT_PATH.parent.parent
NEURAL_ASTAR_SRC = HYBRID_ASTAR_ROOT / "model_base_astar" / "neural-astar" / "src"
DEFAULT_DATASET = (
    HYBRID_ASTAR_ROOT
    / "model_base_astar"
    / "neural-astar"
    / "planning-datasets"
    / "data"
    / "street"
    / "mixed_064_moore_c16.npz"
)
DEFAULT_CNN_CKPT = (
    HYBRID_ASTAR_ROOT
    / "model_base_astar"
    / "neural-astar"
    / "outputs"
    / "model_guidance_street"
    / "best.pt"
)
DEFAULT_UNET_CKPT = (
    HYBRID_ASTAR_ROOT
    / "model_base_astar"
    / "neural-astar"
    / "outputs"
    / "model_guidance_grid_street_formal"
    / "best.pt"
)
DEFAULT_V1_CKPT = (
    HYBRID_ASTAR_ROOT
    / "model_base_astar"
    / "neural-astar"
    / "outputs"
    / "model_guidance_street_unet_transformer_v1_finetune_v1"
    / "best.pt"
)
DEFAULT_V2_CKPT = (
    HYBRID_ASTAR_ROOT
    / "model_base_astar"
    / "neural-astar"
    / "outputs"
    / "ablation_residual_weighted_warmup_v2"
    / "best.pt"
)
DEFAULT_V3_CKPT = (
    HYBRID_ASTAR_ROOT
    / "model_base_astar"
    / "neural-astar"
    / "outputs"
    / "model_guidance_street_unet_transformer_v3_finetune_v1_logged"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = HYBRID_ASTAR_ROOT / "offline_results" / "paper_benchmark_street_methods_20260321"

CJK_FONT_PATH = Path("/usr/share/fonts/truetype/arphic/uming.ttc")

XY = Tuple[int, int]


def load_street_demo_module():
    demo_path = HYBRID_ASTAR_ROOT / "scripts" / "offline_street_guided_astar_demo.py"
    spec = importlib.util.spec_from_file_location("offline_street_guided_astar_demo", demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEMO = load_street_demo_module()


if str(NEURAL_ASTAR_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_ASTAR_SRC))

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn, astar_8conn_stats  # noqa: E402
from neural_astar.api.guidance_infer import load_guidance_encoder  # noqa: E402
from neural_astar.utils.coords import clip_cost_map_with_obstacles, make_one_hot_xy  # noqa: E402
from neural_astar.utils.guidance_targets import build_clearance_input_map  # noqa: E402
from neural_astar.utils.residual_prediction import apply_residual_scale_np, decode_residual_prediction_np  # noqa: E402


@dataclass
class PlannerMethod:
    key: str
    label: str
    color: str


METHODS: List[PlannerMethod] = [
    PlannerMethod("dijkstra", "Dijkstra", "#808080"),
    PlannerMethod("astar", "A*", "#4C78A8"),
    PlannerMethod("greedy", "Greedy", "#F58518"),
    PlannerMethod("weighted_astar", "Weighted A*", "#54A24B"),
    PlannerMethod("cnn_guided", "Legacy-CNN-guided A*", "#B279A2"),
    PlannerMethod("unet_guided", "U-Net-guided A*", "#72B7B2"),
    PlannerMethod("v1_guided", "Transformer-v1-guided A*", "#6C8EBF"),
    PlannerMethod("v2_guided", "Residual-v2-guided A*", "#9C755F"),
    PlannerMethod("transformer_v3", "Transformer-guided A*", "#E45756"),
]
METHODS_BY_KEY: Dict[str, PlannerMethod] = {method.key: method for method in METHODS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch benchmark street planners with shared backend smoothing.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260321)
    parser.add_argument("--cnn-ckpt", type=Path, default=DEFAULT_CNN_CKPT)
    parser.add_argument("--unet-ckpt", type=Path, default=DEFAULT_UNET_CKPT)
    parser.add_argument("--v1-ckpt", type=Path, default=DEFAULT_V1_CKPT)
    parser.add_argument("--v2-ckpt", type=Path, default=DEFAULT_V2_CKPT)
    parser.add_argument("--v3-ckpt", type=Path, default=DEFAULT_V3_CKPT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--cnn-lambda-guidance", type=float, default=0.4)
    parser.add_argument("--unet-lambda-guidance", type=float, default=0.4)
    parser.add_argument("--cnn-residual-weight", type=float, default=1.25)
    parser.add_argument("--unet-residual-weight", type=float, default=1.25)
    parser.add_argument("--v1-residual-weight", type=float, default=1.25)
    parser.add_argument("--v2-residual-weight", type=float, default=1.25)
    parser.add_argument("--transformer-residual-weight", type=float, default=1.25)
    parser.add_argument("--heuristic-mode", type=str, default="octile")
    parser.add_argument("--weighted-heuristic-weight", type=float, default=1.5)
    parser.add_argument("--guidance-integration-mode", type=str, default="g_cost")
    parser.add_argument("--guidance-bonus-threshold", type=float, default=0.5)
    parser.add_argument("--allow-corner-cut", action="store_true")
    parser.add_argument("--min-start-goal-dist", type=float, default=22.0)
    parser.add_argument("--seed-xy-box-half-extent", type=float, default=0.15)
    parser.add_argument("--skip-seed-collision-check", action="store_true")
    parser.add_argument("--skip-smoother", action="store_true")
    parser.add_argument("--smoother-cli", type=Path, default=DEMO.default_smoother_cli())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-prefix", type=str, default="street_planner_compare")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[method.key for method in METHODS],
        choices=[method.key for method in METHODS],
        help="Subset of planners to benchmark.",
    )
    return parser.parse_args()


def load_maps(dataset: Path, split: str) -> np.ndarray:
    key = DEMO._split_key(split)
    with np.load(dataset) as data:
        arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{key} must be [N,H,W], got {arr.shape}")
    return arr


def occ_from_map_design(map_design: np.ndarray) -> np.ndarray:
    return (1.0 - np.asarray(map_design, dtype=np.float32)).astype(np.float32)


def infer_model_output_loaded(
    model: torch.nn.Module,
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    device: str,
) -> Dict[str, Optional[np.ndarray] | str]:
    h, w = occ.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    start = make_one_hot_xy(sx, sy, w, h)
    goal = make_one_hot_xy(gx, gy, w, h)
    occ_t = torch.from_numpy(occ[None, None]).to(device)
    start_t = torch.from_numpy(start[None, None]).to(device)
    goal_t = torch.from_numpy(goal[None, None]).to(device)
    extra_input_t = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        clearance_input = build_clearance_input_map(
            occ_map=occ,
            clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
        )[None, None].astype(np.float32)
        extra_input_t = torch.from_numpy(clearance_input).to(device)
    start_yaw_t = torch.tensor([0.0], dtype=torch.float32, device=device)
    goal_yaw_t = torch.tensor([0.0], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(
            occ_t,
            start_t,
            goal_t,
            start_yaw=start_yaw_t,
            goal_yaw=goal_yaw_t,
            extra_input_maps=extra_input_t,
        )
    cost = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    scale = None
    if out.scale_map is not None:
        scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32)
    confidence = None
    if out.confidence_map is not None:
        confidence = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
        confidence = confidence[0] if confidence.shape[0] == 1 else np.min(confidence, axis=0).astype(np.float32)
    if cost.ndim != 3:
        raise ValueError(f"expected cost volume [K,H,W], got {cost.shape}")
    mode = str(getattr(model, "output_mode", "cost_map"))
    if mode == "residual_heuristic":
        residual = cost[0] if cost.shape[0] == 1 else np.min(cost, axis=0).astype(np.float32)
        residual = decode_residual_prediction_np(
            residual,
            transform=str(getattr(model, "residual_target_transform", "none")),
        )
        if scale is not None:
            scale = scale[0] if scale.shape[0] == 1 else np.min(scale, axis=0).astype(np.float32)
        residual = apply_residual_scale_np(residual, scale)
        return {
            "mode": mode,
            "guidance_cost": None,
            "heuristic_residual_map": residual.astype(np.float32),
            "residual_confidence_map": confidence.astype(np.float32) if confidence is not None else None,
        }
    cost_2d = cost[0] if cost.shape[0] == 1 else np.min(cost, axis=0).astype(np.float32)
    return {
        "mode": mode,
        "guidance_cost": clip_cost_map_with_obstacles(cost_2d, occ, obstacle_cost=1.0).astype(np.float32),
        "heuristic_residual_map": None,
        "residual_confidence_map": None,
    }


def maybe_cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _neighbors_8(x: int, y: int, w: int, h: int) -> Iterable[Tuple[int, int, int, int]]:
    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny, dx, dy


def _heuristic(
    x: int,
    y: int,
    gx: int,
    gy: int,
    diagonal_cost: float,
    mode: str,
) -> float:
    dx = abs(gx - x)
    dy = abs(gy - y)
    if mode == "euclidean":
        return float(math.hypot(dx, dy))
    if mode == "manhattan":
        return float(dx + dy)
    if mode == "chebyshev":
        return float(max(dx, dy))
    if mode == "octile":
        dmin = float(min(dx, dy))
        dmax = float(max(dx, dy))
        return dmax + (float(diagonal_cost) - 1.0) * dmin
    raise ValueError(f"unknown heuristic mode: {mode}")


def reconstruct_path(came_from: Dict[XY, Optional[XY]], current: XY) -> List[XY]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def greedy_best_first_stats(
    occ_map: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    diagonal_cost: float = math.sqrt(2.0),
    allow_corner_cut: bool = True,
    heuristic_mode: str = "octile",
) -> Astar8ConnStats:
    occ = np.asarray(occ_map, dtype=np.float32)
    h, w = occ.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if occ[sy, sx] > 0.5 or occ[gy, gx] > 0.5:
        return Astar8ConnStats(path=None, expanded_nodes=0, success=False, expanded_xy=[])

    import heapq

    start = (sx, sy)
    goal = (gx, gy)
    open_heap: List[Tuple[float, int, XY]] = []
    push_id = 0
    heapq.heappush(open_heap, (_heuristic(sx, sy, gx, gy, diagonal_cost, heuristic_mode), push_id, start))
    came_from: Dict[XY, Optional[XY]] = {start: None}
    discovered: set[XY] = {start}
    closed: set[XY] = set()
    expanded_nodes = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded_nodes += 1
        if current == goal:
            return Astar8ConnStats(
                path=reconstruct_path(came_from, current),
                expanded_nodes=expanded_nodes,
                success=True,
                expanded_xy=list(closed),
            )
        cx, cy = current
        for nx, ny, dx, dy in _neighbors_8(cx, cy, w, h):
            if occ[ny, nx] > 0.5:
                continue
            if dx != 0 and dy != 0 and (not allow_corner_cut):
                if occ[cy, nx] > 0.5 or occ[ny, cx] > 0.5:
                    continue
            neighbor = (nx, ny)
            if neighbor in discovered:
                continue
            discovered.add(neighbor)
            came_from[neighbor] = current
            push_id += 1
            heapq.heappush(
                open_heap,
                (_heuristic(nx, ny, gx, gy, diagonal_cost, heuristic_mode), push_id, neighbor),
            )

    return Astar8ConnStats(path=None, expanded_nodes=expanded_nodes, success=False, expanded_xy=list(closed))


def grid_path_to_world(
    path_xy: Sequence[XY],
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> List[Tuple[float, float]]:
    return [DEMO.grid_to_world(x, y, origin_x, origin_y, resolution) for x, y in path_xy]


def path_length_world(path_xy_world: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(path_xy_world)):
        total += math.hypot(
            path_xy_world[i][0] - path_xy_world[i - 1][0],
            path_xy_world[i][1] - path_xy_world[i - 1][1],
        )
    return total


def run_smoother(
    smoother_cli: Path,
    occ: np.ndarray,
    raw_world_path: Sequence[Tuple[float, float]],
    out_dir: Path,
    origin_x: float,
    origin_y: float,
    resolution: float,
    seed_xy_box_half_extent: float,
    skip_seed_collision_check: bool,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_csv = out_dir / "frontend_seed_path.csv"
    split_csv = out_dir / "segment_split_points.csv"
    smooth_csv = out_dir / "smoothed_path.csv"
    metrics_json = out_dir / "smoother_metrics.json"
    request_yaml = out_dir / "smoother_request.yaml"

    DEMO.write_smoother_yaml(
        request_yaml,
        occ,
        raw_world_path,
        origin_x,
        origin_y,
        resolution,
        seed_xy_box_half_extent,
        skip_seed_collision_check,
    )
    cmd = [
        str(smoother_cli),
        "--input-yaml",
        str(request_yaml),
        "--seed-csv",
        str(seed_csv),
        "--split-points-csv",
        str(split_csv),
        "--output-csv",
        str(smooth_csv),
        "--metrics-json",
        str(metrics_json),
    ]
    env = os.environ.copy()
    smoother_lib_dir = smoother_cli.resolve().parents[1]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{smoother_lib_dir}:{existing_ld}" if existing_ld else str(smoother_lib_dir)
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with metrics_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def benchmark_case(
    *,
    case_id: int,
    map_index: int,
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    cnn_model: torch.nn.Module,
    unet_model: torch.nn.Module,
    v1_model: torch.nn.Module,
    v2_model: torch.nn.Module,
    v3_model: torch.nn.Module,
    args: argparse.Namespace,
    case_dir: Path,
    methods: Sequence[PlannerMethod],
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []

    def run_search(method_key: str) -> Tuple[Astar8ConnStats, float, float, Optional[np.ndarray]]:
        infer_ms = 0.0
        guidance = None
        t0 = time.perf_counter()
        if method_key == "dijkstra":
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                lambda_guidance=0.0,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=0.0,
                allow_corner_cut=args.allow_corner_cut,
            )
        elif method_key == "astar":
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                lambda_guidance=0.0,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=1.0,
                allow_corner_cut=args.allow_corner_cut,
            )
        elif method_key == "greedy":
            result = greedy_best_first_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                allow_corner_cut=args.allow_corner_cut,
                heuristic_mode=args.heuristic_mode,
            )
        elif method_key == "weighted_astar":
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                lambda_guidance=0.0,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=args.weighted_heuristic_weight,
                allow_corner_cut=args.allow_corner_cut,
            )
        elif method_key == "cnn_guided":
            maybe_cuda_sync(args.device)
            infer_begin = time.perf_counter()
            model_out = infer_model_output_loaded(cnn_model, occ, start_xy, goal_xy, args.device)
            maybe_cuda_sync(args.device)
            infer_ms = (time.perf_counter() - infer_begin) * 1000.0
            guidance = model_out["guidance_cost"]
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                guidance_cost=guidance if isinstance(guidance, np.ndarray) else None,
                heuristic_residual_map=(
                    model_out["heuristic_residual_map"]
                    if isinstance(model_out["heuristic_residual_map"], np.ndarray)
                    else None
                ),
                residual_confidence_map=(
                    model_out["residual_confidence_map"]
                    if isinstance(model_out["residual_confidence_map"], np.ndarray)
                    else None
                ),
                lambda_guidance=(
                    args.cnn_lambda_guidance
                    if isinstance(guidance, np.ndarray)
                    else 0.0
                ),
                residual_weight=args.cnn_residual_weight,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=1.0,
                allow_corner_cut=args.allow_corner_cut,
                guidance_integration_mode=args.guidance_integration_mode,
                guidance_bonus_threshold=args.guidance_bonus_threshold,
            )
        elif method_key == "unet_guided":
            maybe_cuda_sync(args.device)
            infer_begin = time.perf_counter()
            model_out = infer_model_output_loaded(unet_model, occ, start_xy, goal_xy, args.device)
            maybe_cuda_sync(args.device)
            infer_ms = (time.perf_counter() - infer_begin) * 1000.0
            guidance = model_out["guidance_cost"]
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                guidance_cost=guidance if isinstance(guidance, np.ndarray) else None,
                heuristic_residual_map=(
                    model_out["heuristic_residual_map"]
                    if isinstance(model_out["heuristic_residual_map"], np.ndarray)
                    else None
                ),
                residual_confidence_map=(
                    model_out["residual_confidence_map"]
                    if isinstance(model_out["residual_confidence_map"], np.ndarray)
                    else None
                ),
                lambda_guidance=(
                    args.unet_lambda_guidance
                    if isinstance(guidance, np.ndarray)
                    else 0.0
                ),
                residual_weight=args.unet_residual_weight,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=1.0,
                allow_corner_cut=args.allow_corner_cut,
                guidance_integration_mode=args.guidance_integration_mode,
                guidance_bonus_threshold=args.guidance_bonus_threshold,
            )
        elif method_key == "transformer_v3":
            maybe_cuda_sync(args.device)
            infer_begin = time.perf_counter()
            model_out = infer_model_output_loaded(v3_model, occ, start_xy, goal_xy, args.device)
            maybe_cuda_sync(args.device)
            infer_ms = (time.perf_counter() - infer_begin) * 1000.0
            guidance = model_out["guidance_cost"]
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                guidance_cost=guidance if isinstance(guidance, np.ndarray) else None,
                heuristic_residual_map=(
                    model_out["heuristic_residual_map"]
                    if isinstance(model_out["heuristic_residual_map"], np.ndarray)
                    else None
                ),
                residual_confidence_map=(
                    model_out["residual_confidence_map"]
                    if isinstance(model_out["residual_confidence_map"], np.ndarray)
                    else None
                ),
                lambda_guidance=0.0,
                residual_weight=args.transformer_residual_weight,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=1.0,
                allow_corner_cut=args.allow_corner_cut,
                guidance_integration_mode=args.guidance_integration_mode,
                guidance_bonus_threshold=args.guidance_bonus_threshold,
            )
        elif method_key == "v1_guided":
            maybe_cuda_sync(args.device)
            infer_begin = time.perf_counter()
            model_out = infer_model_output_loaded(v1_model, occ, start_xy, goal_xy, args.device)
            maybe_cuda_sync(args.device)
            infer_ms = (time.perf_counter() - infer_begin) * 1000.0
            guidance = model_out["guidance_cost"]
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                guidance_cost=guidance if isinstance(guidance, np.ndarray) else None,
                heuristic_residual_map=(
                    model_out["heuristic_residual_map"]
                    if isinstance(model_out["heuristic_residual_map"], np.ndarray)
                    else None
                ),
                residual_confidence_map=(
                    model_out["residual_confidence_map"]
                    if isinstance(model_out["residual_confidence_map"], np.ndarray)
                    else None
                ),
                lambda_guidance=0.0,
                residual_weight=args.v1_residual_weight,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=1.0,
                allow_corner_cut=args.allow_corner_cut,
                guidance_integration_mode=args.guidance_integration_mode,
                guidance_bonus_threshold=args.guidance_bonus_threshold,
            )
        elif method_key == "v2_guided":
            maybe_cuda_sync(args.device)
            infer_begin = time.perf_counter()
            model_out = infer_model_output_loaded(v2_model, occ, start_xy, goal_xy, args.device)
            maybe_cuda_sync(args.device)
            infer_ms = (time.perf_counter() - infer_begin) * 1000.0
            guidance = model_out["guidance_cost"]
            result = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                guidance_cost=guidance if isinstance(guidance, np.ndarray) else None,
                heuristic_residual_map=(
                    model_out["heuristic_residual_map"]
                    if isinstance(model_out["heuristic_residual_map"], np.ndarray)
                    else None
                ),
                residual_confidence_map=(
                    model_out["residual_confidence_map"]
                    if isinstance(model_out["residual_confidence_map"], np.ndarray)
                    else None
                ),
                lambda_guidance=0.0,
                residual_weight=args.v2_residual_weight,
                heuristic_mode=args.heuristic_mode,
                heuristic_weight=1.0,
                allow_corner_cut=args.allow_corner_cut,
                guidance_integration_mode=args.guidance_integration_mode,
                guidance_bonus_threshold=args.guidance_bonus_threshold,
            )
        else:
            raise ValueError(f"unknown method: {method_key}")
        search_ms = (time.perf_counter() - t0) * 1000.0 - infer_ms
        return result, infer_ms, max(0.0, search_ms), guidance

    for method in methods:
        result, infer_ms, search_ms, _ = run_search(method.key)
        row: Dict[str, float | int | str] = {
            "case_id": case_id,
            "map_index": map_index,
            "start_x": int(start_xy[0]),
            "start_y": int(start_xy[1]),
            "goal_x": int(goal_xy[0]),
            "goal_y": int(goal_xy[1]),
            "method": method.key,
            "method_label": method.label,
            "frontend_success": int(result.success),
            "backend_success": 0,
            "guidance_infer_ms": infer_ms,
            "frontend_search_ms": search_ms,
            "frontend_total_ms": infer_ms + search_ms,
            "expanded_nodes": int(result.expanded_nodes),
            "raw_points": 0,
            "raw_length_m": 0.0,
            "seed_stage_ms": math.nan,
            "smooth_stage_ms": math.nan,
            "smoother_total_ms": math.nan,
            "seed_points": 0,
            "seed_length_m": math.nan,
            "smoothed_points": 0,
            "smoothed_length_m": math.nan,
            "segment_split_points": math.nan,
            "pipeline_total_ms": math.nan,
        }
        if result.success and result.path is not None and len(result.path) >= 2:
            raw_world_path = grid_path_to_world(result.path, args.origin_x, args.origin_y, args.resolution)
            row["raw_points"] = len(raw_world_path)
            row["raw_length_m"] = path_length_world(raw_world_path)
            if args.skip_smoother:
                row["pipeline_total_ms"] = float(row["frontend_total_ms"])
            else:
                smoother_metrics = run_smoother(
                    args.smoother_cli,
                    occ,
                    raw_world_path,
                    case_dir / method.key,
                    args.origin_x,
                    args.origin_y,
                    args.resolution,
                    args.seed_xy_box_half_extent,
                    args.skip_seed_collision_check,
                )
                row["seed_stage_ms"] = float(smoother_metrics["seed_stage_ms"])
                row["smooth_stage_ms"] = float(smoother_metrics["smooth_stage_ms"])
                row["smoother_total_ms"] = float(smoother_metrics["total_cli_ms"])
                row["seed_points"] = int(smoother_metrics["seed_points"])
                row["seed_length_m"] = float(smoother_metrics["seed_length_m"])
                row["smoothed_points"] = int(smoother_metrics["smoothed_points"])
                row["smoothed_length_m"] = float(smoother_metrics["smoothed_length_m"])
                row["segment_split_points"] = int(smoother_metrics["segment_split_points"])
                row["backend_success"] = int(bool(smoother_metrics.get("backend_success", True)))
                row["pipeline_total_ms"] = float(row["frontend_total_ms"]) + float(row["smoother_total_ms"])
        rows.append(row)
    return rows


def aggregate(
    rows: Sequence[Dict[str, float | int | str]],
    methods: Sequence[PlannerMethod],
) -> List[Dict[str, float | str]]:
    by_method: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    astar_rows = by_method.get("astar", [])
    astar_expanded = np.mean([float(r["expanded_nodes"]) for r in astar_rows]) if astar_rows else math.nan

    summary: List[Dict[str, float | str]] = []
    for method in methods:
        group = by_method.get(method.key, [])
        if not group:
            continue
        success_mask = [int(r["frontend_success"]) > 0 for r in group]
        backend_success_mask = [int(r.get("backend_success", 0)) > 0 for r in group]
        denom = max(1, len(group))
        successful = [r for r in group if int(r["frontend_success"]) > 0]
        def mean_of(key: str) -> float:
            values = [float(r[key]) for r in successful if not math.isnan(float(r[key]))]
            return float(np.mean(values)) if values else math.nan
        avg_expanded = float(np.mean([float(r["expanded_nodes"]) for r in group]))
        summary.append(
            {
                "method": method.key,
                "method_label": method.label,
                "success_rate": sum(success_mask) / denom,
                "backend_success_rate": sum(backend_success_mask) / denom,
                "guidance_infer_ms": mean_of("guidance_infer_ms"),
                "frontend_search_ms": mean_of("frontend_search_ms"),
                "frontend_total_ms": mean_of("frontend_total_ms"),
                "expanded_nodes": avg_expanded,
                "raw_length_m": mean_of("raw_length_m"),
                "seed_stage_ms": mean_of("seed_stage_ms"),
                "smooth_stage_ms": mean_of("smooth_stage_ms"),
                "smoother_total_ms": mean_of("smoother_total_ms"),
                "seed_length_m": mean_of("seed_length_m"),
                "smoothed_length_m": mean_of("smoothed_length_m"),
                "pipeline_total_ms": mean_of("pipeline_total_ms"),
                "segment_split_points": mean_of("segment_split_points"),
                "seed_vs_raw_pct": (
                    100.0 * (mean_of("seed_length_m") - mean_of("raw_length_m")) / max(mean_of("raw_length_m"), 1e-9)
                    if not math.isnan(mean_of("seed_length_m")) and not math.isnan(mean_of("raw_length_m"))
                    else math.nan
                ),
                "smooth_vs_seed_pct": (
                    100.0 * (mean_of("smoothed_length_m") - mean_of("seed_length_m")) / max(mean_of("seed_length_m"), 1e-9)
                    if not math.isnan(mean_of("smoothed_length_m")) and not math.isnan(mean_of("seed_length_m"))
                    else math.nan
                ),
                "smooth_vs_raw_pct": (
                    100.0 * (mean_of("smoothed_length_m") - mean_of("raw_length_m")) / max(mean_of("raw_length_m"), 1e-9)
                    if not math.isnan(mean_of("smoothed_length_m")) and not math.isnan(mean_of("raw_length_m"))
                    else math.nan
                ),
                "expanded_vs_astar_pct": (
                    100.0 * (avg_expanded - astar_expanded) / max(astar_expanded, 1e-9)
                    if not math.isnan(astar_expanded)
                    else math.nan
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_md(path: Path, summary: Sequence[Dict[str, float | str]]) -> None:
    lines = [
        "| 方法 | 前端成功率 | 后端成功率 | 扩展节点 | 前端总耗时(ms) | seed耗时(ms) | 优化耗时(ms) | 总规划耗时(ms) | 原始长度(m) | 初始长度(m) | 优化长度(m) | 相对A*扩展节点变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method_label} | {success_rate:.2%} | {backend_success_rate:.2%} | {expanded_nodes:.2f} | {frontend_total_ms:.2f} | {seed_stage_ms:.2f} | {smooth_stage_ms:.2f} | {pipeline_total_ms:.2f} | {raw_length_m:.3f} | {seed_length_m:.3f} | {smoothed_length_m:.3f} | {expanded_vs_astar_pct:+.2f}% |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: Sequence[Dict[str, float | str]], out_path: Path) -> None:
    font_prop = None
    if CJK_FONT_PATH.exists():
        from matplotlib import font_manager as fm

        font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH))

    labels = [str(row["method_label"]) for row in summary]
    colors = [METHODS_BY_KEY[str(row["method"])].color for row in summary]
    expanded = [float(row["expanded_nodes"]) for row in summary]
    frontend = [float(row["frontend_total_ms"]) for row in summary]
    seed_ms = [float(row["seed_stage_ms"]) for row in summary]
    smooth_ms = [float(row["smooth_stage_ms"]) for row in summary]
    lengths = [float(row["smoothed_length_m"]) for row in summary]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), facecolor="#fcfcfb")
    axes = axes.reshape(-1)
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color="#d9dde3", linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#a9adb3")

    axes[0].bar(x, expanded, color=colors, alpha=0.92)
    axes[0].set_title("平均扩展节点数", fontproperties=font_prop)
    axes[0].set_ylabel("nodes", fontproperties=font_prop)

    axes[1].bar(x, frontend, color=colors, alpha=0.92)
    axes[1].set_title("平均前端规划耗时", fontproperties=font_prop)
    axes[1].set_ylabel("ms", fontproperties=font_prop)

    axes[2].bar(x, seed_ms, color="#E69F00", alpha=0.85, label="初始路径")
    axes[2].bar(x, smooth_ms, bottom=seed_ms, color="#009E73", alpha=0.85, label="优化路径")
    axes[2].set_title("后端阶段耗时", fontproperties=font_prop)
    axes[2].set_ylabel("ms", fontproperties=font_prop)
    axes[2].legend(prop=font_prop)

    axes[3].bar(x, lengths, color=colors, alpha=0.92)
    axes[3].set_title("平均优化后路径长度", fontproperties=font_prop)
    axes[3].set_ylabel("m", fontproperties=font_prop)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if font_prop is not None:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_relative(summary: Sequence[Dict[str, float | str]], out_path: Path) -> None:
    font_prop = None
    if CJK_FONT_PATH.exists():
        from matplotlib import font_manager as fm

        font_prop = fm.FontProperties(fname=str(CJK_FONT_PATH))

    labels = [str(row["method_label"]) for row in summary]
    colors = [METHODS_BY_KEY[str(row["method"])].color for row in summary]
    expanded_pct = [float(row["expanded_vs_astar_pct"]) for row in summary]
    smooth_vs_raw = [float(row["smooth_vs_raw_pct"]) for row in summary]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), facecolor="#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color="#d9dde3", linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_edgecolor("#a9adb3")

    axes[0].bar(x, expanded_pct, color=colors, alpha=0.92)
    axes[0].axhline(0.0, color="#666666", linewidth=1.0)
    axes[0].set_title("相对 A* 的扩展节点变化", fontproperties=font_prop)
    axes[0].set_ylabel("%", fontproperties=font_prop)

    axes[1].bar(x, smooth_vs_raw, color=colors, alpha=0.92)
    axes[1].axhline(0.0, color="#666666", linewidth=1.0)
    axes[1].set_title("优化路径相对原始路径长度变化", fontproperties=font_prop)
    axes[1].set_ylabel("%", fontproperties=font_prop)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        if font_prop is not None:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font_prop)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = [METHODS_BY_KEY[key] for key in args.methods]

    maps = load_maps(args.dataset, args.split)
    rng = random.Random(args.sample_seed)

    print(f"loading_cnn_model={args.cnn_ckpt}")
    cnn_model = load_guidance_encoder(args.cnn_ckpt, device=args.device)
    print(f"loading_unet_model={args.unet_ckpt}")
    unet_model = load_guidance_encoder(args.unet_ckpt, device=args.device)
    print(f"loading_v1_model={args.v1_ckpt}")
    v1_model = load_guidance_encoder(args.v1_ckpt, device=args.device)
    print(f"loading_v2_model={args.v2_ckpt}")
    v2_model = load_guidance_encoder(args.v2_ckpt, device=args.device)
    print(f"loading_v3_model={args.v3_ckpt}")
    v3_model = load_guidance_encoder(args.v3_ckpt, device=args.device)
    warm_occ = np.zeros((64, 64), dtype=np.float32)
    warm_start = (4, 4)
    warm_goal = (59, 59)
    _ = infer_model_output_loaded(cnn_model, warm_occ, warm_start, warm_goal, args.device)
    _ = infer_model_output_loaded(unet_model, warm_occ, warm_start, warm_goal, args.device)
    _ = infer_model_output_loaded(v1_model, warm_occ, warm_start, warm_goal, args.device)
    _ = infer_model_output_loaded(v2_model, warm_occ, warm_start, warm_goal, args.device)
    _ = infer_model_output_loaded(v3_model, warm_occ, warm_start, warm_goal, args.device)
    maybe_cuda_sync(args.device)

    rows: List[Dict[str, float | int | str]] = []
    sampled_cases: List[Dict[str, int]] = []

    case_id = 0
    attempts = 0
    while case_id < args.max_cases:
        attempts += 1
        if attempts > args.max_cases * 50:
            raise RuntimeError("failed to sample enough valid street cases")
        map_index = rng.randrange(maps.shape[0])
        occ = occ_from_map_design(maps[map_index])
        solver = lambda occ_map, sxy, gxy: astar_8conn(
            occ_map=occ_map,
            start_xy=sxy,
            goal_xy=gxy,
            lambda_guidance=0.0,
            heuristic_mode=args.heuristic_mode,
            heuristic_weight=1.0,
            allow_corner_cut=args.allow_corner_cut,
        )
        try:
            start_xy, goal_xy = DEMO.sample_problem(
                occ,
                solver,
                rng,
                args.min_start_goal_dist,
                args.origin_x,
                args.origin_y,
                args.resolution,
                0.125,
            )
        except Exception:
            continue

        case_dir = args.output_dir / "cases" / f"case_{case_id:03d}_m{map_index}_s{args.sample_seed + case_id}"
        case_rows = benchmark_case(
            case_id=case_id,
            map_index=map_index,
            occ=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            cnn_model=cnn_model,
            unet_model=unet_model,
            v1_model=v1_model,
            v2_model=v2_model,
            v3_model=v3_model,
            args=args,
            case_dir=case_dir,
            methods=selected_methods,
        )
        rows.extend(case_rows)
        sampled_cases.append(
            {
                "case_id": case_id,
                "map_index": map_index,
                "start_x": int(start_xy[0]),
                "start_y": int(start_xy[1]),
                "goal_x": int(goal_xy[0]),
                "goal_y": int(goal_xy[1]),
            }
        )
        print(f"completed_case={case_id} map_index={map_index} start={start_xy} goal={goal_xy}", flush=True)
        case_id += 1

    summary = aggregate(rows, selected_methods)

    raw_csv = args.output_dir / f"{args.paper_prefix}_case_metrics.csv"
    summary_csv = args.output_dir / f"{args.paper_prefix}_summary.csv"
    summary_md = args.output_dir / f"{args.paper_prefix}_summary.md"
    cases_csv = args.output_dir / f"{args.paper_prefix}_sampled_cases.csv"
    summary_png = args.output_dir / f"{args.paper_prefix}_summary.png"
    relative_png = args.output_dir / f"{args.paper_prefix}_relative.png"

    write_csv(raw_csv, rows)
    write_csv(summary_csv, summary)
    write_csv(cases_csv, sampled_cases)
    write_summary_md(summary_md, summary)
    plot_summary(summary, summary_png)
    plot_relative(summary, relative_png)

    print(f"saved_case_metrics={raw_csv}")
    print(f"saved_summary_csv={summary_csv}")
    print(f"saved_summary_md={summary_md}")
    print(f"saved_sampled_cases={cases_csv}")
    print(f"saved_summary_png={summary_png}")
    print(f"saved_relative_png={relative_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
