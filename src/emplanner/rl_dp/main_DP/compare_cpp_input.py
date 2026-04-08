#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from ppo import ActorCritic, PPOConfig, encode_observation
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
        raise ValueError(f"Invalid trunk input dim {trunk_in}; expected at least 64.")

    grid_size = int(spec.s_samples * spec.l_samples)
    feature_dim = grid_size * grid_channels + extra_dim
    return grid_channels, extra_dim, hidden_dim, feature_dim


def _action_mask_row(l_samples: int, last_l_index: int, lateral_move_limit: int) -> np.ndarray:
    if lateral_move_limit < 0:
        return np.ones(l_samples, dtype=bool)
    idxs = np.arange(l_samples, dtype=int)
    return np.abs(idxs - last_l_index) <= lateral_move_limit


def _log_mask(mask: np.ndarray) -> None:
    valid = np.nonzero(mask)[0]
    count = int(valid.size)
    if count == 0:
        print(f"Action mask: valid=0/{mask.size}")
        return
    print(f"Action mask: valid={count}/{mask.size}, idx_range=[{valid[0]},{valid[-1]}]")


def _build_observation(
    spec: GridSpec,
    *,
    s_index: int,
    last_l_index: int,
    lateral_move_limit: int,
) -> dict:
    s_min, s_max = spec.s_range
    l_min, l_max = spec.l_range
    s_coords = np.linspace(s_min, s_max, spec.s_samples, dtype=np.float32)
    l_coords = np.linspace(l_min, l_max, spec.l_samples, dtype=np.float32)
    occupancy = np.zeros((spec.s_samples, spec.l_samples), dtype=np.float32)
    action_mask = _action_mask_row(spec.l_samples, last_l_index, lateral_move_limit)
    obstacle_corners = np.zeros((0, 4, 2), dtype=np.float32)
    start_l_value = float(l_coords[last_l_index])
    return {
        "s_index": np.array(s_index, dtype=np.int32),
        "path_indices": np.array([last_l_index], dtype=np.int32),
        "occupancy": occupancy,
        "s_coords": s_coords,
        "l_coords": l_coords,
        "start_l": np.array(start_l_value, dtype=np.float32),
        "action_mask": action_mask,
        "obstacle_corners": obstacle_corners,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare C++ input encoding with Python.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PPO checkpoint (*.pt).",
    )
    parser.add_argument("--s-index", type=int, default=4, help="s_index used for encoding.")
    parser.add_argument("--last-l-index", type=int, default=9, help="last l index used for encoding.")
    parser.add_argument(
        "--lateral-move-limit", type=int, default=3, help="lateral move limit for action mask."
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--topk", type=int, default=5, help="top-k logits to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    config = PPOConfig(**payload["config"])
    spec = GridSpec(**payload["grid_spec"])

    state_dict = payload["model_state"]
    grid_channels, extra_dim, hidden_dim, feature_dim = _infer_model_dims(
        state_dict, spec, config
    )
    grid_size = spec.s_samples * spec.l_samples

    policy = ActorCritic(
        occupancy_shape=(spec.s_samples, spec.l_samples),
        grid_channels=grid_channels,
        extra_dim=extra_dim,
        action_dim=spec.l_samples,
        hidden_dim=hidden_dim,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    obs = _build_observation(
        spec,
        s_index=args.s_index,
        last_l_index=args.last_l_index,
        lateral_move_limit=args.lateral_move_limit,
    )
    encoded = encode_observation(obs, spec, include_action_mask=True)
    if encoded.shape[0] != feature_dim:
        raise ValueError(
            f"Encoded feature_dim mismatch: got {encoded.shape[0]}, expected {feature_dim}"
        )

    s_min, s_max = spec.s_range
    l_min, l_max = spec.l_range
    s_step = (s_max - s_min) / max(1, spec.s_samples - 1)
    l_step = (l_max - l_min) / max(1, spec.l_samples - 1)
    s_coord = s_min + s_step * args.s_index
    l_coord = l_min + l_step * args.last_l_index
    s_norm = args.s_index / max(1, spec.s_samples - 1)
    l_norm = (l_coord - l_min) / max(l_max - l_min, 1e-6)
    start_l_value = l_coord
    start_l_norm = (start_l_value - l_min) / max(l_max - l_min, 1e-6)

    print(
        f"feature_dim={feature_dim}, grid_size={grid_size}, grid_channels={grid_channels}, extra_dim={extra_dim}"
    )
    print(
        f"Grid: s_range={spec.s_range}, s_samples={spec.s_samples}, s_step={s_step} | "
        f"l_range={spec.l_range}, l_samples={spec.l_samples}, l_step={l_step}"
    )
    print(
        f"s_index={args.s_index}, s_coord={s_coord}, s_norm={s_norm} | "
        f"l_index={args.last_l_index}, l_coord={l_coord}, l_norm={l_norm} | "
        f"start_l={start_l_value}, start_l_norm={start_l_norm}"
    )
    mask = obs["action_mask"].astype(bool)
    _log_mask(mask)

    with torch.no_grad():
        logits, value = policy(torch.as_tensor(encoded, device=device))
    logits_np = logits.detach().cpu().numpy().reshape(-1)
    topk = min(max(1, args.topk), logits_np.size)
    top_idx = np.argsort(logits_np)[-topk:][::-1]
    print("Top logits:", [(int(i), float(logits_np[i])) for i in top_idx])

    masked_logits = np.where(mask, logits_np, -1e9)
    masked_top = int(np.argmax(masked_logits))
    print("Masked argmax:", masked_top, "logit", float(masked_logits[masked_top]))
    print("Value:", float(value.detach().cpu().item()))


if __name__ == "__main__":
    main()
