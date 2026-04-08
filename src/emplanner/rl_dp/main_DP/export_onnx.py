#!/usr/bin/env python3
"""
Export a trained PPO ActorCritic policy checkpoint to ONNX for C++/inference use.

The export infers the input layout from checkpoint weights (grid channels from
the first conv layer and extra feature size from the trunk), so it works across
different encoder variants. It always exports the raw logits/value heads; action
masking should be applied externally (same as 训练/推理流程).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from ppo import ActorCritic, PPOConfig
from sl_grid import GridSpec


def _infer_model_dims(
    state_dict: dict, spec: GridSpec, config: PPOConfig
) -> Tuple[int, int, int, int]:
    conv_weight = state_dict.get("conv_trunk.0.weight")
    if conv_weight is None:
        raise ValueError("Missing conv_trunk.0.weight in checkpoint state_dict.")
    grid_channels = int(conv_weight.shape[1])

    trunk_weight = state_dict.get("trunk.0.weight")
    if trunk_weight is None:
        raise ValueError("Missing trunk.0.weight in checkpoint state_dict.")
    trunk_in = int(trunk_weight.shape[1])
    hidden_dim = int(trunk_weight.shape[0])

    extra_dim = trunk_in - 64
    if extra_dim < 0:
        raise ValueError(
            f"Invalid trunk input dim {trunk_in}; expected at least 64."
        )
    if hidden_dim != config.hidden_dim:
        print(
            f"Warning: checkpoint hidden_dim={hidden_dim} overrides config hidden_dim={config.hidden_dim}."
        )

    grid_size = int(spec.s_samples * spec.l_samples)
    feature_dim = grid_size * grid_channels + extra_dim
    return grid_channels, extra_dim, hidden_dim, feature_dim


def _load_policy(checkpoint_path: Path, device: torch.device) -> Tuple[ActorCritic, int]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = PPOConfig(**payload["config"])
    spec = GridSpec(**payload["grid_spec"])

    state_dict = payload["model_state"]
    grid_channels, extra_dim, hidden_dim, feature_dim = _infer_model_dims(
        state_dict, spec, config
    )

    action_dim = spec.l_samples
    occupancy_shape = (spec.s_samples, spec.l_samples)

    policy = ActorCritic(
        occupancy_shape=occupancy_shape,
        grid_channels=grid_channels,
        extra_dim=extra_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, feature_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PPO checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint (*.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path (default: same name as checkpoint with .onnx).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Export device cpu/cuda (default: cuda if available else cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    output_path = args.output or checkpoint_path.with_suffix(".onnx")

    policy, feature_dim = _load_policy(checkpoint_path, device)

    dummy_input = torch.zeros(1, feature_dim, dtype=torch.float32, device=device)
    input_names = ["state"]
    output_names = ["logits", "value"]
    dynamic_axes = {
        "state": {0: "batch"},
        "logits": {0: "batch"},
        "value": {0: "batch"},
    }

    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path} (feature_dim={feature_dim}, device={device.type})")


if __name__ == "__main__":
    main()
