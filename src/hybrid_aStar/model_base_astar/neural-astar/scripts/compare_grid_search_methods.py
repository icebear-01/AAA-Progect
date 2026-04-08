"""Compare classic 8-connected search baselines and learned residual models.

This script evaluates a unified set of planners on the same planning-datasets
NPZ split and writes both aggregate metrics and per-case metrics.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import PlanningNPZGuidanceDataset
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import (
    apply_confidence_safety_gate,
    resolve_residual_confidence_map,
)
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)


XY = Tuple[int, int]


@dataclass
class Stat:
    count: int = 0
    success: int = 0
    expanded_nodes: float = 0.0
    runtime_ms: float = 0.0
    path_length: float = 0.0

    def update(self, *, success: bool, expanded_nodes: int, runtime_ms: float, path_length: float) -> None:
        self.count += 1
        self.success += int(bool(success))
        self.expanded_nodes += float(expanded_nodes)
        self.runtime_ms += float(runtime_ms)
        self.path_length += float(path_length)

    def summary(self) -> Dict[str, float]:
        denom = max(self.count, 1)
        return {
            "success_rate": self.success / denom,
            "expanded_nodes": self.expanded_nodes / denom,
            "runtime_ms": self.runtime_ms / denom,
            "path_length": self.path_length / denom,
        }


@dataclass
class MethodSpec:
    label: str
    kind: str
    ckpt: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare classic search baselines and learned residual models.")
    p.add_argument("--data-npz", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--residual-weight", type=float, default=1.25)
    p.add_argument(
        "--residual-confidence-mode",
        type=str,
        default="learned_spike",
        choices=["none", "spike_suppression", "learned", "learned_spike"],
    )
    p.add_argument("--residual-confidence-kernel", type=int, default=5)
    p.add_argument("--residual-confidence-strength", type=float, default=0.75)
    p.add_argument("--residual-confidence-min", type=float, default=0.25)
    p.add_argument("--safety-gate-threshold", type=float, default=0.0)
    p.add_argument("--safety-gate-kernel", type=int, default=1)
    p.add_argument("--safety-gate-low-scale", type=float, default=0.0)
    p.add_argument("--safety-gate-residual-min", type=float, default=0.0)
    p.add_argument("--heuristic-mode", type=str, default="octile", choices=["euclidean", "manhattan", "chebyshev", "octile"])
    p.add_argument("--heuristic-weight", type=float, default=1.0)
    p.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    p.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    p.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    p.add_argument(
        "--classic-methods",
        nargs="*",
        default=["traditional_astar", "improved_astar", "dijkstra", "greedy_best_first"],
        choices=["traditional_astar", "improved_astar", "dijkstra", "greedy_best_first"],
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec in the form label=/abs/path/to/best.pt",
    )
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--case-csv", type=Path, required=True)
    p.set_defaults(allow_corner_cut=True)
    return p.parse_args()


def _onehot_xy(one_hot_1hw: torch.Tensor) -> XY:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y)


def _heuristic(x: int, y: int, gx: int, gy: int, diagonal_cost: float, mode: str) -> float:
    dx = abs(int(gx) - int(x))
    dy = abs(int(gy) - int(y))
    if mode == "euclidean":
        return float(np.hypot(dx, dy))
    if mode == "manhattan":
        return float(dx + dy)
    if mode == "chebyshev":
        return float(max(dx, dy))
    if mode == "octile":
        dmin = float(min(dx, dy))
        dmax = float(max(dx, dy))
        return dmax + (float(diagonal_cost) - 1.0) * dmin
    raise ValueError(f"Unknown heuristic mode: {mode}")


def _neighbors_8(x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
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
            out.append((nx, ny, dx, dy))
    return out


def _reconstruct_path(came_from: Dict[XY, Optional[XY]], current: XY) -> List[XY]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def greedy_best_first_8conn_stats(
    occ_map: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    *,
    diagonal_cost: float,
    allow_corner_cut: bool,
    heuristic_mode: str,
) -> Astar8ConnStats:
    occ = np.asarray(occ_map, dtype=np.float32)
    h, w = occ.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return Astar8ConnStats(path=None, expanded_nodes=0, success=False, expanded_xy=[])
    if occ[sy, sx] > 0.5 or occ[gy, gx] > 0.5:
        return Astar8ConnStats(path=None, expanded_nodes=0, success=False, expanded_xy=[])

    start = (sx, sy)
    goal = (gx, gy)
    open_heap: List[Tuple[float, float, int, XY]] = []
    push_id = 0
    heapq.heappush(
        open_heap,
        (_heuristic(sx, sy, gx, gy, diagonal_cost=diagonal_cost, mode=heuristic_mode), 0.0, push_id, start),
    )
    came_from: Dict[XY, Optional[XY]] = {start: None}
    g_score: Dict[XY, float] = {start: 0.0}
    closed: set[XY] = set()
    expanded_nodes = 0

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded_nodes += 1
        if current == goal:
            return Astar8ConnStats(
                path=_reconstruct_path(came_from, current),
                expanded_nodes=expanded_nodes,
                success=True,
                expanded_xy=list(closed),
            )

        cx, cy = current
        for nx, ny, dx, dy in _neighbors_8(cx, cy, w, h):
            if occ[ny, nx] > 0.5:
                continue
            is_diagonal = (dx != 0) and (dy != 0)
            if is_diagonal:
                side_block_x = occ[cy, nx] > 0.5
                side_block_y = occ[ny, cx] > 0.5
                if side_block_x and side_block_y:
                    continue
                if (not allow_corner_cut) and (side_block_x or side_block_y):
                    continue
            move_cost = float(diagonal_cost) if is_diagonal else 1.0
            neighbor = (nx, ny)
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                push_id += 1
                heapq.heappush(
                    open_heap,
                    (
                        _heuristic(nx, ny, gx, gy, diagonal_cost=diagonal_cost, mode=heuristic_mode),
                        tentative_g,
                        push_id,
                        neighbor,
                    ),
                )

    return Astar8ConnStats(path=None, expanded_nodes=expanded_nodes, success=False, expanded_xy=list(closed))


def _evaluate_classic(
    label: str,
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    *,
    diagonal_cost: float,
    allow_corner_cut: bool,
) -> Tuple[bool, int, float, float]:
    t0 = time.perf_counter()
    if label == "traditional_astar":
        result = astar_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            diagonal_cost=diagonal_cost,
            allow_corner_cut=allow_corner_cut,
            heuristic_mode="euclidean",
            heuristic_weight=1.0,
        )
    elif label == "improved_astar":
        result = astar_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            diagonal_cost=diagonal_cost,
            allow_corner_cut=allow_corner_cut,
            heuristic_mode="octile",
            heuristic_weight=1.0,
        )
    elif label == "dijkstra":
        result = astar_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            diagonal_cost=diagonal_cost,
            allow_corner_cut=allow_corner_cut,
            heuristic_mode="octile",
            heuristic_weight=0.0,
        )
    elif label == "greedy_best_first":
        result = greedy_best_first_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            diagonal_cost=diagonal_cost,
            allow_corner_cut=allow_corner_cut,
            heuristic_mode="octile",
        )
    else:
        raise ValueError(f"Unknown classic method: {label}")
    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    path_len = 0.0 if result.path is None else path_length_8conn(result.path, diagonal_cost=diagonal_cost)
    return result.success, result.expanded_nodes, runtime_ms, path_len


def _infer_residual_prediction_2d(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    occ = sample["occ_map"].unsqueeze(0).to(device)
    start = sample["start_map"].unsqueeze(0).to(device)
    goal = sample["goal_map"].unsqueeze(0).to(device)
    start_pose = sample.get("start_pose")
    goal_pose = sample.get("goal_pose")
    start_yaw = None
    goal_yaw = None
    if start_pose is not None:
        start_yaw = start_pose[2].view(1).to(device=device, dtype=occ.dtype)
    if goal_pose is not None:
        goal_yaw = goal_pose[2].view(1).to(device=device, dtype=occ.dtype)
    extra_input_maps = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        clearance_input = sample.get("clearance_input_map")
        if clearance_input is None:
            clearance_input_np = build_clearance_input_map(
                occ_map=sample["occ_map"].numpy()[0].astype(np.float32),
                clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
            )[None, ...].astype(np.float32)
            clearance_input = torch.from_numpy(clearance_input_np)
        extra_input_maps = clearance_input.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            occ,
            start,
            goal,
            start_yaw=start_yaw,
            goal_yaw=goal_yaw,
            extra_input_maps=extra_input_maps,
        )
    pred = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    scale = None
    if out.scale_map is not None:
        scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32)
    confidence = None
    if out.confidence_map is not None:
        confidence = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
        confidence = confidence[0] if confidence.shape[0] == 1 else np.min(confidence, axis=0).astype(np.float32)
    if pred.ndim != 3:
        raise ValueError(f"Expected guidance volume [K,H,W], got {pred.shape}")
    pred = pred[0] if pred.shape[0] == 1 else np.min(pred, axis=0).astype(np.float32)
    if getattr(model, "output_mode", "cost_map") == "residual_heuristic":
        pred = decode_residual_prediction_np(
            pred,
            transform=str(getattr(model, "residual_target_transform", "none")),
        )
        if scale is not None:
            scale = scale[0] if scale.shape[0] == 1 else np.min(scale, axis=0).astype(np.float32)
        pred = apply_residual_scale_np(pred, scale)
    return pred.astype(np.float32), confidence


def _evaluate_model(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    *,
    device: str,
    residual_weight: float,
    residual_confidence_mode: str,
    residual_confidence_kernel: int,
    residual_confidence_strength: float,
    residual_confidence_min: float,
    safety_gate_threshold: float,
    safety_gate_kernel: int,
    safety_gate_low_scale: float,
    safety_gate_residual_min: float,
    diagonal_cost: float,
    allow_corner_cut: bool,
    heuristic_mode: str,
    heuristic_weight: float,
) -> Tuple[bool, int, float, float]:
    occ = sample["occ_map"].numpy()[0].astype(np.float32)
    start_xy = _onehot_xy(sample["start_map"])
    goal_xy = _onehot_xy(sample["goal_map"])
    pred_residual, learned_confidence = _infer_residual_prediction_2d(model, sample, device=device)
    residual_confidence_map = None
    if residual_confidence_mode != "none":
        residual_confidence_map = resolve_residual_confidence_map(
            mode=residual_confidence_mode,
            occ_map=occ,
            residual_map=pred_residual,
            learned_confidence_map=learned_confidence,
            kernel_size=residual_confidence_kernel,
            strength=residual_confidence_strength,
            min_confidence=residual_confidence_min,
        )
        residual_confidence_map = apply_confidence_safety_gate(
            residual_confidence_map,
            occ_map=occ,
            gate_threshold=safety_gate_threshold,
            gate_kernel=safety_gate_kernel,
            low_scale=safety_gate_low_scale,
            residual_map=pred_residual,
            residual_min=safety_gate_residual_min,
        )
    t0 = time.perf_counter()
    result = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_residual_map=pred_residual,
        residual_confidence_map=residual_confidence_map,
        residual_weight=float(residual_weight),
        diagonal_cost=diagonal_cost,
        allow_corner_cut=allow_corner_cut,
        heuristic_mode=heuristic_mode,
        heuristic_weight=heuristic_weight,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    path_len = 0.0 if result.path is None else path_length_8conn(result.path, diagonal_cost=diagonal_cost)
    return result.success, result.expanded_nodes, runtime_ms, path_len


def _sanitize_label(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip()).strip("_").lower()
    return slug or "method"


def _parse_model_specs(items: Sequence[str]) -> List[MethodSpec]:
    specs: List[MethodSpec] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model must be label=/path/to/best.pt, got {item}")
        label, path_str = item.split("=", 1)
        ckpt = Path(path_str).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"model checkpoint not found: {ckpt}")
        specs.append(MethodSpec(label=label.strip(), kind="model", ckpt=ckpt))
    return specs


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_specs = _parse_model_specs(args.model)
    all_specs = [MethodSpec(label=name, kind="classic") for name in args.classic_methods] + model_specs
    models: Dict[str, torch.nn.Module] = {}
    clearance_input_clip_distance = 0.0
    for spec in model_specs:
        model = load_guidance_encoder(spec.ckpt, device=device)
        models[spec.label] = model
        clearance_input_clip_distance = max(
            clearance_input_clip_distance,
            float(getattr(model, "clearance_input_clip_distance", 0.0)),
        )

    ds = PlanningNPZGuidanceDataset(
        npz_path=args.data_npz,
        split=args.split,
        orientation_bins=1,
        clearance_input_clip_distance=clearance_input_clip_distance,
    )
    eval_count = len(ds) if int(args.max_samples) <= 0 else min(len(ds), int(args.max_samples))

    stats = {spec.label: Stat() for spec in all_specs}
    case_rows: List[Dict[str, float | int | str]] = []

    for idx in range(eval_count):
        sample = ds[idx]
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])
        row: Dict[str, float | int | str] = {
            "idx": idx,
            "start_x": start_xy[0],
            "start_y": start_xy[1],
            "goal_x": goal_xy[0],
            "goal_y": goal_xy[1],
        }
        for spec in all_specs:
            if spec.kind == "classic":
                ok, expanded, runtime_ms, path_len = _evaluate_classic(
                    spec.label,
                    occ,
                    start_xy,
                    goal_xy,
                    diagonal_cost=args.diagonal_cost,
                    allow_corner_cut=args.allow_corner_cut,
                )
            else:
                ok, expanded, runtime_ms, path_len = _evaluate_model(
                    models[spec.label],
                    sample,
                    device=device,
                    residual_weight=args.residual_weight,
                    residual_confidence_mode=args.residual_confidence_mode,
                    residual_confidence_kernel=args.residual_confidence_kernel,
                    residual_confidence_strength=args.residual_confidence_strength,
                    residual_confidence_min=args.residual_confidence_min,
                    safety_gate_threshold=args.safety_gate_threshold,
                    safety_gate_kernel=args.safety_gate_kernel,
                    safety_gate_low_scale=args.safety_gate_low_scale,
                    safety_gate_residual_min=args.safety_gate_residual_min,
                    diagonal_cost=args.diagonal_cost,
                    allow_corner_cut=args.allow_corner_cut,
                    heuristic_mode=args.heuristic_mode,
                    heuristic_weight=args.heuristic_weight,
                )
            stats[spec.label].update(
                success=ok,
                expanded_nodes=expanded,
                runtime_ms=runtime_ms,
                path_length=path_len,
            )
            prefix = _sanitize_label(spec.label)
            row[f"{prefix}_success"] = int(ok)
            row[f"{prefix}_expanded"] = expanded
            row[f"{prefix}_runtime_ms"] = runtime_ms
            row[f"{prefix}_path_length"] = path_len
        case_rows.append(row)
        print(f"processed={idx + 1}/{eval_count}", flush=True)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.case_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "kind",
                "ckpt",
                "success_rate",
                "expanded_nodes",
                "runtime_ms",
                "path_length",
            ],
        )
        writer.writeheader()
        for spec in all_specs:
            row = {
                "label": spec.label,
                "kind": spec.kind,
                "ckpt": "" if spec.ckpt is None else str(spec.ckpt),
            }
            row.update(stats[spec.label].summary())
            writer.writerow(row)

    with args.case_csv.open("w", newline="") as f:
        fieldnames: List[str] = []
        for row in case_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    print(f"saved_summary_csv={args.summary_csv}")
    print(f"saved_case_csv={args.case_csv}")


if __name__ == "__main__":
    main()
