"""Evaluate improved-heuristic 2D A* baseline vs learned cost/residual models."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import ParkingGuidanceDataset, PlanningNPZGuidanceDataset
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)


XY = Tuple[int, int]


@dataclass
class Stat:
    total: int = 0
    success: int = 0
    expanded_nodes: float = 0.0
    runtime_ms: float = 0.0
    path_length: float = 0.0

    def update(self, *, success: bool, expanded_nodes: int, runtime_ms: float, path_length: float) -> None:
        self.total += 1
        self.success += int(success)
        self.expanded_nodes += float(expanded_nodes)
        self.runtime_ms += float(runtime_ms)
        self.path_length += float(path_length)

    def summary(self) -> Dict[str, float]:
        denom = max(1, self.total)
        return {
            "success_rate": self.success / denom,
            "expanded_nodes": self.expanded_nodes / denom,
            "runtime_ms": self.runtime_ms / denom,
            "path_length": self.path_length / denom,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate improved-heuristic 2D A* vs learned guidance.")
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path, default=None)
    data_group.add_argument("--data-npz", type=Path, default=None)
    p.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Split used when evaluating planning-datasets .npz.",
    )
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument(
        "--guidance-source",
        type=str,
        default="ckpt",
        choices=["ckpt", "target_cost", "astar_expanded_map", "opt_traj", "residual_heuristic_map"],
        help="Source used for the guided/residual run.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--lambda-guidance", type=float, default=0.4)
    p.add_argument("--lambda-sweep", type=float, nargs="*", default=None)
    p.add_argument("--residual-weight", type=float, default=1.0)
    p.add_argument("--residual-weight-sweep", type=float, nargs="*", default=None)
    p.add_argument(
        "--heuristic-mode",
        type=str,
        default="octile",
        choices=["euclidean", "manhattan", "chebyshev", "octile"],
    )
    p.add_argument("--heuristic-weight", type=float, default=1.0)
    p.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    p.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    p.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    p.add_argument(
        "--guidance-integration-mode",
        type=str,
        default="heuristic_bonus",
        choices=["g_cost", "heuristic_bias", "heuristic_bonus"],
    )
    p.add_argument("--guidance-bonus-threshold", type=float, default=0.6)
    p.add_argument("--invert-guidance-cost", action="store_true")
    p.add_argument(
        "--clearance-weight",
        type=float,
        default=0.0,
        help="Planner-side obstacle-clearance penalty weight added to g-cost.",
    )
    p.add_argument(
        "--clearance-safe-distance",
        type=float,
        default=0.0,
        help="Clearance radius in grid cells used by planner-side obstacle penalty.",
    )
    p.add_argument(
        "--clearance-power",
        type=float,
        default=2.0,
        help="Power used by planner-side obstacle-clearance penalty.",
    )
    p.add_argument(
        "--clearance-integration-mode",
        type=str,
        default="g_cost",
        choices=["g_cost", "heuristic_bias", "priority_tie_break"],
        help="How clearance affects search priority.",
    )
    p.add_argument(
        "--residual-confidence-mode",
        type=str,
        default="none",
        choices=["none", "spike_suppression", "learned", "learned_spike"],
        help="Residual confidence source for learned residual checkpoints.",
    )
    p.add_argument("--residual-confidence-kernel", type=int, default=5)
    p.add_argument("--residual-confidence-strength", type=float, default=0.75)
    p.add_argument("--residual-confidence-min", type=float, default=0.25)
    p.add_argument("--csv-out", type=Path, default=None)
    p.set_defaults(allow_corner_cut=True)
    return p.parse_args()


def _onehot_xy(one_hot_1hw: torch.Tensor) -> XY:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y)


def _infer_guidance_2d(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: str,
    invert_guidance_cost: bool,
) -> np.ndarray:
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
    cost = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    if invert_guidance_cost:
        cost = 1.0 - cost
    if cost.ndim != 3:
        raise ValueError(f"Expected guidance volume [K,H,W], got {cost.shape}")
    return cost[0] if cost.shape[0] == 1 else np.min(cost, axis=0).astype(np.float32)


def _infer_residual_prediction_2d(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: str,
    invert_guidance_cost: bool,
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
    if invert_guidance_cost:
        pred = 1.0 - pred
    return pred.astype(np.float32), confidence


def _resolve_guidance(
    sample: Dict[str, torch.Tensor],
    model: Optional[torch.nn.Module],
    device: str,
    source: str,
    invert_guidance_cost: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    if source == "ckpt":
        if model is None:
            raise ValueError("guidance-source=ckpt requires --ckpt")
        pred, confidence = _infer_residual_prediction_2d(
            model=model,
            sample=sample,
            device=device,
            invert_guidance_cost=invert_guidance_cost,
        )
        if getattr(model, "output_mode", "cost_map") == "residual_heuristic":
            return None, pred, confidence, "residual"
        return pred, None, None, "guidance"
    if source == "target_cost":
        return sample["target_cost"].numpy()[0].astype(np.float32), None, None, "guidance"
    if source == "astar_expanded_map":
        return sample["astar_expanded_map"].numpy()[0].astype(np.float32), None, None, "guidance"
    if source == "opt_traj":
        return (1.0 - sample["opt_traj"].numpy()[0]).astype(np.float32), None, None, "guidance"
    if source == "residual_heuristic_map":
        return None, sample["residual_heuristic_map"].numpy()[0].astype(np.float32), None, "residual"
    raise ValueError(f"Unknown guidance source: {source}")


def _evaluate_one(
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    guidance_cost: Optional[np.ndarray],
    heuristic_residual_map: Optional[np.ndarray],
    residual_confidence_map: Optional[np.ndarray],
    *,
    lambda_guidance: float,
    residual_weight: float,
    heuristic_mode: str,
    heuristic_weight: float,
    diagonal_cost: float,
    allow_corner_cut: bool,
    guidance_integration_mode: str,
    guidance_bonus_threshold: float,
    clearance_weight: float,
    clearance_safe_distance: float,
    clearance_power: float,
    clearance_integration_mode: str,
) -> Tuple[bool, int, float, float]:
    t0 = time.perf_counter()
    result = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        guidance_cost=guidance_cost,
        heuristic_residual_map=heuristic_residual_map,
        residual_confidence_map=residual_confidence_map,
        lambda_guidance=lambda_guidance,
        residual_weight=residual_weight,
        diagonal_cost=diagonal_cost,
        allow_corner_cut=allow_corner_cut,
        heuristic_mode=heuristic_mode,
        heuristic_weight=heuristic_weight,
        guidance_integration_mode=guidance_integration_mode,
        guidance_bonus_threshold=guidance_bonus_threshold,
        clearance_weight=clearance_weight,
        clearance_safe_distance=clearance_safe_distance,
        clearance_power=clearance_power,
        clearance_integration_mode=clearance_integration_mode,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    path_length = 0.0 if result.path is None else path_length_8conn(result.path, diagonal_cost=diagonal_cost)
    return result.success, result.expanded_nodes, runtime_ms, path_length


def _iter_lambdas(args: argparse.Namespace) -> Iterable[float]:
    if args.lambda_sweep:
        return [float(v) for v in args.lambda_sweep]
    return [float(args.lambda_guidance)]


def _iter_residual_weights(args: argparse.Namespace) -> Iterable[float]:
    if args.residual_weight_sweep:
        return [float(v) for v in args.residual_weight_sweep]
    return [float(args.residual_weight)]


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = None
    if args.guidance_source == "ckpt":
        if args.ckpt is None:
            raise ValueError("--guidance-source=ckpt requires --ckpt")
        model = load_guidance_encoder(args.ckpt, device=device)
    clearance_input_clip_distance = float(
        getattr(model, "clearance_input_clip_distance", 0.0) if model is not None else 0.0
    )
    if args.data_dir is not None:
        ds = ParkingGuidanceDataset(
            args.data_dir,
            orientation_bins=1,
            clearance_input_clip_distance=clearance_input_clip_distance,
        )
    else:
        ds = PlanningNPZGuidanceDataset(
            npz_path=args.data_npz,
            split=args.split,
            orientation_bins=1,
            clearance_input_clip_distance=clearance_input_clip_distance,
        )
    eval_count = len(ds) if int(args.max_samples) <= 0 else min(len(ds), int(args.max_samples))

    lambdas = list(_iter_lambdas(args))
    residual_weights = list(_iter_residual_weights(args))
    baseline = Stat()
    eval_mode = (
        "residual"
        if args.guidance_source == "residual_heuristic_map"
        else ("residual" if (model is not None and getattr(model, "output_mode", "cost_map") == "residual_heuristic") else "guidance")
    )
    if eval_mode == "residual":
        variant_stats = {float(w): Stat() for w in residual_weights}
    else:
        variant_stats = {float(lam): Stat() for lam in lambdas}

    rows: List[Dict[str, float]] = []
    for idx in range(eval_count):
        sample = ds[idx]
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])
        guidance_cost, heuristic_residual_map, learned_confidence_map, resolved_mode = _resolve_guidance(
            sample=sample,
            model=model,
            device=device,
            source=args.guidance_source,
            invert_guidance_cost=bool(args.invert_guidance_cost),
        )
        if resolved_mode != eval_mode:
            raise RuntimeError(f"Inconsistent evaluation mode: expected {eval_mode}, got {resolved_mode}")
        residual_confidence_map = None
        if (
            resolved_mode == "residual"
            and heuristic_residual_map is not None
            and args.guidance_source == "ckpt"
            and args.residual_confidence_mode != "none"
        ):
            residual_confidence_map = resolve_residual_confidence_map(
                mode=args.residual_confidence_mode,
                occ_map=occ,
                residual_map=heuristic_residual_map,
                learned_confidence_map=learned_confidence_map,
                kernel_size=args.residual_confidence_kernel,
                strength=args.residual_confidence_strength,
                min_confidence=args.residual_confidence_min,
            )

        ok, expanded, runtime_ms, path_len = _evaluate_one(
            occ,
            start_xy,
            goal_xy,
            None,
            None,
            None,
            lambda_guidance=0.0,
            residual_weight=0.0,
            heuristic_mode=args.heuristic_mode,
            heuristic_weight=args.heuristic_weight,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
            guidance_integration_mode=args.guidance_integration_mode,
            guidance_bonus_threshold=args.guidance_bonus_threshold,
            clearance_weight=args.clearance_weight,
            clearance_safe_distance=args.clearance_safe_distance,
            clearance_power=args.clearance_power,
            clearance_integration_mode=args.clearance_integration_mode,
        )
        baseline.update(
            success=ok,
            expanded_nodes=expanded,
            runtime_ms=runtime_ms,
            path_length=path_len,
        )

        if eval_mode == "residual":
            for weight in residual_weights:
                ok, expanded, runtime_ms, path_len = _evaluate_one(
                    occ,
                    start_xy,
                    goal_xy,
                    None,
                    heuristic_residual_map,
                    residual_confidence_map,
                    lambda_guidance=0.0,
                    residual_weight=float(weight),
                    heuristic_mode=args.heuristic_mode,
                    heuristic_weight=args.heuristic_weight,
                    diagonal_cost=args.diagonal_cost,
                    allow_corner_cut=args.allow_corner_cut,
                    guidance_integration_mode=args.guidance_integration_mode,
                    guidance_bonus_threshold=args.guidance_bonus_threshold,
                    clearance_weight=args.clearance_weight,
                    clearance_safe_distance=args.clearance_safe_distance,
                    clearance_power=args.clearance_power,
                    clearance_integration_mode=args.clearance_integration_mode,
                )
                variant_stats[float(weight)].update(
                    success=ok,
                    expanded_nodes=expanded,
                    runtime_ms=runtime_ms,
                    path_length=path_len,
                )
        else:
            for lam in lambdas:
                ok, expanded, runtime_ms, path_len = _evaluate_one(
                    occ,
                    start_xy,
                    goal_xy,
                    guidance_cost,
                    None,
                    None,
                    lambda_guidance=float(lam),
                    residual_weight=0.0,
                    heuristic_mode=args.heuristic_mode,
                    heuristic_weight=args.heuristic_weight,
                    diagonal_cost=args.diagonal_cost,
                    allow_corner_cut=args.allow_corner_cut,
                    guidance_integration_mode=args.guidance_integration_mode,
                    guidance_bonus_threshold=args.guidance_bonus_threshold,
                    clearance_weight=args.clearance_weight,
                    clearance_safe_distance=args.clearance_safe_distance,
                    clearance_power=args.clearance_power,
                    clearance_integration_mode=args.clearance_integration_mode,
                )
                variant_stats[float(lam)].update(
                    success=ok,
                    expanded_nodes=expanded,
                    runtime_ms=runtime_ms,
                    path_length=path_len,
                )

        if idx % 32 == 0 or idx + 1 == eval_count:
            print(f"processed={idx + 1}/{eval_count}")

    base_row = {
        "label": "baseline",
        "lambda_guidance": 0.0,
        "residual_weight": 0.0,
        "heuristic_mode": args.heuristic_mode,
        "heuristic_weight": float(args.heuristic_weight),
        "guidance_source": args.guidance_source,
        "clearance_weight": float(args.clearance_weight),
        "clearance_safe_distance": float(args.clearance_safe_distance),
        "clearance_power": float(args.clearance_power),
        "clearance_integration_mode": args.clearance_integration_mode,
    }
    base_row.update(baseline.summary())
    rows.append(base_row)

    if eval_mode == "residual":
        for weight in residual_weights:
            row = {
                "label": "residual_guided",
                "lambda_guidance": 0.0,
                "residual_weight": float(weight),
                "heuristic_mode": args.heuristic_mode,
                "heuristic_weight": float(args.heuristic_weight),
                "guidance_source": args.guidance_source,
                "clearance_weight": float(args.clearance_weight),
                "clearance_safe_distance": float(args.clearance_safe_distance),
                "clearance_power": float(args.clearance_power),
                "clearance_integration_mode": args.clearance_integration_mode,
            }
            row.update(variant_stats[float(weight)].summary())
            rows.append(row)
    else:
        for lam in lambdas:
            row = {
                "label": "guided",
                "lambda_guidance": float(lam),
                "residual_weight": 0.0,
                "heuristic_mode": args.heuristic_mode,
                "heuristic_weight": float(args.heuristic_weight),
                "guidance_source": args.guidance_source,
                "clearance_weight": float(args.clearance_weight),
                "clearance_safe_distance": float(args.clearance_safe_distance),
                "clearance_power": float(args.clearance_power),
                "clearance_integration_mode": args.clearance_integration_mode,
            }
            row.update(variant_stats[float(lam)].summary())
            rows.append(row)

    print(
        "baseline: "
        + ", ".join(f"{k}={v:.3f}" for k, v in baseline.summary().items())
    )
    if eval_mode == "residual":
        for weight in residual_weights:
            print(
                f"residual(weight={weight:.3f}): "
                + ", ".join(f"{k}={v:.3f}" for k, v in variant_stats[float(weight)].summary().items())
            )
    else:
        for lam in lambdas:
            print(
                f"guided(lambda={lam:.3f}): "
                + ", ".join(f"{k}={v:.3f}" for k, v in variant_stats[float(lam)].summary().items())
            )

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label",
            "lambda_guidance",
            "residual_weight",
            "guidance_source",
            "heuristic_mode",
            "heuristic_weight",
            "clearance_weight",
            "clearance_safe_distance",
            "clearance_power",
            "clearance_integration_mode",
            "success_rate",
            "expanded_nodes",
            "runtime_ms",
            "path_length",
        ]
        with args.csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"saved_csv={args.csv_out}")


if __name__ == "__main__":
    main()
