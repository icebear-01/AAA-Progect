#!/usr/bin/env python3
"""Export GuidanceEncoder checkpoint to a simplified ONNX inference graph.

The exported graph matches the C++ frontend integration:
  inputs:
    occ_map   [1,1,H,W]
    start_map [1,1,H,W]
    goal_map  [1,1,H,W]
    start_yaw [1]
    goal_yaw  [1]
  outputs:
    cost_map  [1,H,W]

The wrapper applies the same post-processing as ``infer_cost_map`` so the C++
side only needs to consume a final 2D guidance map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.models.guidance_encoder import GoalCrossAttentionBlock
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_torch,
)


class GuidanceEncoderCostMapWrapper(nn.Module):
    def __init__(self, model: nn.Module, invert_guidance_cost: bool = False) -> None:
        super().__init__()
        self.model = model
        self.invert_guidance_cost = bool(invert_guidance_cost)

    def forward(  # type: ignore[override]
        self,
        occ_map: torch.Tensor,
        start_map: torch.Tensor,
        goal_map: torch.Tensor,
        start_yaw: torch.Tensor,
        goal_yaw: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            occ_map,
            start_map,
            goal_map,
            start_yaw=start_yaw,
            goal_yaw=goal_yaw,
        )
        cost_map = out.cost_map
        if getattr(self.model, "output_mode", "cost_map") == "residual_heuristic":
            transform = str(getattr(self.model, "residual_target_transform", "none"))
            if transform == "log1p":
                cost_map = torch.exp(cost_map) - 1.0
            elif transform != "none":
                raise ValueError(f"Unknown residual_target_transform: {transform}")
            cost_map = apply_residual_scale_torch(cost_map, out.scale_map)
        elif self.invert_guidance_cost:
            cost_map = 1.0 - cost_map

        if cost_map.shape[1] == 1:
            return cost_map[:, 0]
        return torch.min(cost_map, dim=1).values


def make_export_friendly_guidance_encoder() -> None:
    """Patch dynamic pooling branches to ONNX-friendly fixed-kernel ops.

    The original model uses adaptive pooling with tensor-derived output sizes and
    a dynamic kernel selection branch. PyTorch 1.12 cannot export that path to
    ONNX reliably. For export we replace it with semantically close static
    operations. Runtime resizing is handled on the C++ side.
    """

    @staticmethod
    def _context_from_mask_export(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled_mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode="nearest")
        pooled_mask = pooled_mask.expand(-1, x.shape[1], -1, -1)
        denom = pooled_mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return (x * pooled_mask).sum(dim=(2, 3)) / denom

    @staticmethod
    def _dense_context_tokens_export(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_mask = F.interpolate(mask, size=(x.shape[-2], x.shape[-1]), mode="nearest")
        pooled_mask = F.avg_pool2d(pooled_mask, kernel_size=5, stride=1, padding=2)
        pooled_mask = pooled_mask / pooled_mask.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled_mask = pooled_mask.clamp(0.0, 1.0)

        weighted = x * pooled_mask.expand(-1, x.shape[1], -1, -1)
        dense_tokens = weighted.flatten(2).transpose(1, 2)
        global_ctx = GoalCrossAttentionBlock._context_from_mask(x, mask)
        dense_tokens = dense_tokens + pooled_mask.flatten(2).transpose(1, 2) * global_ctx.unsqueeze(1)
        return dense_tokens, global_ctx

    GoalCrossAttentionBlock._context_from_mask = _context_from_mask_export
    GoalCrossAttentionBlock._dense_context_tokens = _dense_context_tokens_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GuidanceEncoder to ONNX.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--out", type=Path, required=True, help="Output .onnx path.")
    parser.add_argument("--height", type=int, default=64, help="Dummy export height.")
    parser.add_argument("--width", type=int, default=64, help="Dummy export width.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--invert-guidance-cost",
        action="store_true",
        help="Export wrapper with 1-cost guidance post-processing.",
    )
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Enable dynamic H/W axes. Disabled by default for compatibility with torch 1.12 export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    make_export_friendly_guidance_encoder()
    model = load_guidance_encoder(args.ckpt, device=str(device))
    wrapper = GuidanceEncoderCostMapWrapper(
        model=model,
        invert_guidance_cost=args.invert_guidance_cost,
    ).to(device)
    wrapper.eval()

    dummy_occ = torch.zeros((1, 1, args.height, args.width), dtype=torch.float32, device=device)
    dummy_start = torch.zeros_like(dummy_occ)
    dummy_goal = torch.zeros_like(dummy_occ)
    dummy_start[..., 0, 0] = 1.0
    dummy_goal[..., args.height - 1, args.width - 1] = 1.0
    dummy_start_yaw = torch.zeros((1,), dtype=torch.float32, device=device)
    dummy_goal_yaw = torch.zeros((1,), dtype=torch.float32, device=device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "opset_version": args.opset,
        "do_constant_folding": True,
        "input_names": ["occ_map", "start_map", "goal_map", "start_yaw", "goal_yaw"],
        "output_names": ["cost_map"],
    }
    if args.dynamic_axes:
        export_kwargs["dynamic_axes"] = {
            "occ_map": {2: "height", 3: "width"},
            "start_map": {2: "height", 3: "width"},
            "goal_map": {2: "height", 3: "width"},
            "cost_map": {1: "height", 2: "width"},
        }

    torch.onnx.export(
        wrapper,
        (dummy_occ, dummy_start, dummy_goal, dummy_start_yaw, dummy_goal_yaw),
        str(args.out),
        **export_kwargs,
    )
    print(f"exported_onnx={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
