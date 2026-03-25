"""Inference API for Route A guidance cost maps.

Coordinate convention:
- Input node coordinates are world (x, y)
- Map indexing is [y, x]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from neural_astar.models import GuidanceEncoder
from neural_astar.utils.coords import clip_cost_map_with_obstacles, make_one_hot_xy
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)


def _load_occ_from_npz(npz_path: Path, sample_index: int = 0) -> np.ndarray:
    with np.load(npz_path) as data:
        if "occ_map" in data.files:
            occ = np.asarray(data["occ_map"], dtype=np.float32)
            if occ.ndim == 3 and occ.shape[0] == 1:
                occ = occ[0]
            return occ

        if "arr_0" in data.files:
            # Existing dataset uses 1=free map_design, convert to 1=obstacle occupancy.
            arr = np.asarray(data["arr_0"], dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(f"arr_0 must be [N,H,W], got {arr.shape}")
            map_design = arr[sample_index]
            return 1.0 - map_design

    raise ValueError(f"Could not find occ_map or arr_0 in {npz_path}")


def load_guidance_encoder(
    ckpt_path: str | Path,
    device: str = "cpu",
) -> GuidanceEncoder:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model_cfg = ckpt.get("model_cfg", {})
    state_dict = ckpt.get("model_state_dict", ckpt)

    arch = str(model_cfg.get("arch", "unet"))
    if "arch" not in model_cfg:
        if any(str(k).startswith("net.") for k in state_dict.keys()):
            arch = "legacy_cnn"
    orientation_bins = int(model_cfg.get("orientation_bins", 0))
    if orientation_bins <= 0:
        if "head.weight" in state_dict:
            orientation_bins = int(state_dict["head.weight"].shape[0])
        elif "net.8.weight" in state_dict:
            orientation_bins = int(state_dict["net.8.weight"].shape[0])
        elif "net.7.weight" in state_dict:
            orientation_bins = int(state_dict["net.7.weight"].shape[0])
        elif "net.6.weight" in state_dict:
            orientation_bins = int(state_dict["net.6.weight"].shape[0])
        else:
            orientation_bins = 1
    use_pose_yaw_cond = bool(model_cfg.get("use_pose_yaw_cond", int(model_cfg.get("in_channels", 3)) > 3))
    in_channels = int(model_cfg.get("in_channels", 3))
    base_channels_expected = 3 + (4 if use_pose_yaw_cond else 0)
    extra_input_channels = int(
        model_cfg.get("extra_input_channels", max(0, in_channels - base_channels_expected))
    )
    model = GuidanceEncoder(
        in_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        obstacle_cost=float(model_cfg.get("obstacle_cost", 1.0)),
        arch=arch,
        use_pose_yaw_cond=use_pose_yaw_cond,
        orientation_bins=orientation_bins,
        output_mode=str(model_cfg.get("output_mode", "cost_map")),
        residual_target_transform=str(model_cfg.get("residual_target_transform", "none")),
        predict_confidence=bool(model_cfg.get("predict_confidence", False)),
        confidence_head_kernel=int(model_cfg.get("confidence_head_kernel", 1)),
        predict_residual_scale=bool(model_cfg.get("predict_residual_scale", False)),
        residual_scale_max=float(model_cfg.get("residual_scale_max", 2.0)),
        transformer_depth=int(model_cfg.get("transformer_depth", 2)),
        transformer_heads=int(model_cfg.get("transformer_heads", 8)),
        transformer_mlp_ratio=float(model_cfg.get("transformer_mlp_ratio", 4.0)),
        extra_input_channels=extra_input_channels,
        clearance_input_clip_distance=float(model_cfg.get("clearance_input_clip_distance", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_cost_volume(
    ckpt_path: str | Path,
    occ_map_numpy: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    start_yaw: float = 0.0,
    goal_yaw: float = 0.0,
    device: str = "cpu",
    invert_guidance_cost: bool = False,
) -> np.ndarray:
    """Infer a guidance cost volume [K, H, W] in [0, 1]."""
    occ = np.asarray(occ_map_numpy, dtype=np.float32)
    if occ.ndim != 2:
        raise ValueError(f"occ_map_numpy must be [H,W], got {occ.shape}")

    h, w = occ.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])

    start = make_one_hot_xy(sx, sy, w, h)
    goal = make_one_hot_xy(gx, gy, w, h)

    model = load_guidance_encoder(ckpt_path, device=device)

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
    start_yaw_t = torch.tensor([float(start_yaw)], dtype=torch.float32, device=device)
    goal_yaw_t = torch.tensor([float(goal_yaw)], dtype=torch.float32, device=device)

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
        scale = (
            None
            if out.scale_map is None
            else out.scale_map[0].detach().cpu().numpy().astype(np.float32)
        )

    if getattr(model, "output_mode", "cost_map") == "residual_heuristic":
        cost = decode_residual_prediction_np(
            cost,
            transform=str(getattr(model, "residual_target_transform", "none")),
        )
        cost = apply_residual_scale_np(cost, scale)
    elif invert_guidance_cost:
        cost = 1.0 - cost
    if cost.ndim != 3:
        raise ValueError(f"Expected cost volume [K,H,W], got {cost.shape}")
    return np.stack(
        [clip_cost_map_with_obstacles(cost[idx], occ, obstacle_cost=1.0) for idx in range(cost.shape[0])],
        axis=0,
    ).astype(np.float32)


def infer_cost_map(
    ckpt_path: str | Path,
    occ_map_numpy: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    start_yaw: float = 0.0,
    goal_yaw: float = 0.0,
    device: str = "cpu",
    invert_guidance_cost: bool = False,
) -> np.ndarray:
    """Infer a 2D guidance map [H, W] by reducing a predicted cost volume."""
    cost_volume = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ_map_numpy,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_yaw=start_yaw,
        goal_yaw=goal_yaw,
        device=device,
        invert_guidance_cost=invert_guidance_cost,
    )
    if cost_volume.shape[0] == 1:
        return cost_volume[0]
    return np.min(cost_volume, axis=0).astype(np.float32)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer guidance cost map from occupancy map")
    p.add_argument("--ckpt", type=Path, required=True, help="Path to encoder checkpoint")
    p.add_argument("--occ_npz", type=Path, required=True, help="NPZ containing occ_map or arr_0")
    p.add_argument("--index", type=int, default=0, help="Sample index if using arr_0")
    p.add_argument("--start", type=int, nargs=2, metavar=("X", "Y"), required=True)
    p.add_argument("--goal", type=int, nargs=2, metavar=("X", "Y"), required=True)
    p.add_argument("--start-yaw-deg", type=float, default=0.0)
    p.add_argument("--goal-yaw-deg", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--invert-guidance-cost",
        action="store_true",
        help="Invert inferred guidance cost as (1-cost). Useful for legacy checkpoints.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output .npy file path")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    occ = _load_occ_from_npz(args.occ_npz, sample_index=args.index)
    cost = infer_cost_map(
        ckpt_path=args.ckpt,
        occ_map_numpy=occ,
        start_xy=(args.start[0], args.start[1]),
        goal_xy=(args.goal[0], args.goal[1]),
        start_yaw=math.radians(float(args.start_yaw_deg)),
        goal_yaw=math.radians(float(args.goal_yaw_deg)),
        device=args.device,
        invert_guidance_cost=args.invert_guidance_cost,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, cost.astype(np.float32))
    print(f"saved: {args.out}")
    print(f"shape: {cost.shape}, min={cost.min():.4f}, max={cost.max():.4f}")


if __name__ == "__main__":
    main()
