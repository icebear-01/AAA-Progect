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
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
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


def _load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    payload = torch.load(checkpoint_path, map_location=device)
    return payload


def _export_metadata(
    *,
    checkpoint_path: Path,
    output_path: Path,
    metadata_path: Path,
    payload: Dict[str, object],
    feature_dim: int,
) -> None:
    config = PPOConfig(**payload["config"])
    spec = GridSpec(**payload["grid_spec"])
    state_dict = payload["model_state"]
    grid_channels, extra_dim, hidden_dim, _ = _infer_model_dims(state_dict, spec, config)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path),
        "feature_dim": int(feature_dim),
        "action_dim": int(spec.l_samples),
        "hidden_dim": int(hidden_dim),
        "grid_channels": int(grid_channels),
        "extra_dim": int(extra_dim),
        "occupancy_shape": [int(spec.s_samples), int(spec.l_samples)],
        "grid_spec": {
            "s_range": list(spec.s_range),
            "l_range": list(spec.l_range),
            "s_samples": int(spec.s_samples),
            "l_samples": int(spec.l_samples),
        },
        "input_names": ["state"],
        "output_names": ["logits", "value"],
        "notes": {
            "action_masking": "Apply action mask externally after reading logits.",
            "state_encoding": "Input must match encode_observation() output layout used during training.",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _validate_export(
    *,
    policy: ActorCritic,
    output_path: Path,
    feature_dim: int,
    device: torch.device,
    num_samples: int,
    atol: float,
    rtol: float,
) -> Dict[str, float]:
    ort_session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    max_abs_logits = 0.0
    max_abs_value = 0.0
    rng = np.random.default_rng(20260409)

    for _ in range(num_samples):
        sample = rng.standard_normal((1, feature_dim), dtype=np.float32)
        torch_input = torch.from_numpy(sample).to(device)
        with torch.no_grad():
            torch_logits, torch_value = policy(torch_input)
        ort_logits, ort_value = ort_session.run(None, {"state": sample})

        logits_diff = np.max(np.abs(torch_logits.detach().cpu().numpy() - ort_logits))
        value_diff = np.max(
            np.abs(
                np.asarray(torch_value.detach().cpu().numpy())
                - np.asarray(ort_value)
            )
        )
        max_abs_logits = max(max_abs_logits, float(logits_diff))
        max_abs_value = max(max_abs_value, float(value_diff))

    logits_ok = max_abs_logits <= atol + rtol
    value_ok = max_abs_value <= atol + rtol
    if not logits_ok or not value_ok:
        raise RuntimeError(
            "ONNX validation failed: "
            f"max_abs_logits={max_abs_logits:.6e}, max_abs_value={max_abs_value:.6e}, "
            f"atol={atol:.6e}, rtol={rtol:.6e}"
        )
    return {
        "max_abs_logits": max_abs_logits,
        "max_abs_value": max_abs_value,
        "samples": int(num_samples),
    }


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
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON path (default: same name as ONNX with .json).",
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
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip ONNX Runtime output validation after export.",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=5,
        help="Number of random samples used for ONNX/PyTorch consistency validation.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for ONNX validation.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for ONNX validation.",
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
    metadata_path = args.metadata or output_path.with_suffix(".json")

    payload = _load_checkpoint_payload(checkpoint_path, device)
    policy, feature_dim = _load_policy(checkpoint_path, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

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
    _export_metadata(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        metadata_path=metadata_path,
        payload=payload,
        feature_dim=feature_dim,
    )

    print(
        f"Exported ONNX model to {output_path} "
        f"(feature_dim={feature_dim}, device={device.type})"
    )
    print(f"Saved metadata to {metadata_path}")

    if not args.skip_validate:
        result = _validate_export(
            policy=policy,
            output_path=output_path,
            feature_dim=feature_dim,
            device=device,
            num_samples=max(1, int(args.validate_samples)),
            atol=float(args.atol),
            rtol=float(args.rtol),
        )
        print(
            "Validation passed | "
            f"samples={result['samples']} | "
            f"max_abs_logits={result['max_abs_logits']:.6e} | "
            f"max_abs_value={result['max_abs_value']:.6e}"
        )


if __name__ == "__main__":
    main()
