#!/usr/bin/env python3
"""
Run one deterministic inference episode with an exported ONNX policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch

from ppo import encode_observation
from ppo_inference import plot_episode_path
from rl_env import SLPathEnv
from sl_grid import GridSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic inference with an exported ONNX policy."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint used to reconstruct env/grid metadata.",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to exported ONNX policy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Use the same paper-style plotting as ppo_inference.py.",
    )
    return parser.parse_args()


def _load_env(checkpoint_path: Path) -> SLPathEnv:
    payload = torch.load(checkpoint_path, map_location="cpu")
    spec = GridSpec(**payload["grid_spec"])
    env_kwargs = payload.get("environment", {})
    return SLPathEnv(spec, **env_kwargs)


def _masked_argmax(logits: np.ndarray, action_mask: Optional[np.ndarray]) -> int:
    if action_mask is None:
        return int(np.argmax(logits))
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape[0] != logits.shape[0]:
        return int(np.argmax(logits))
    masked_logits = np.where(mask, logits, -1e9)
    return int(np.argmax(masked_logits))


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    env = _load_env(checkpoint_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    observation = env.reset()
    l_coords = np.asarray(observation["l_coords"], dtype=np.float32)
    start_idx = int(np.asarray(observation["path_indices"]).reshape(-1)[-1])
    path_s = [0.0]
    path_l = [float(l_coords[start_idx])]

    step_details = []
    total_reward = 0.0
    forward_ms_total = 0.0
    terminal_reason = None
    done = False
    step = 0

    while not done:
        encoded = encode_observation(
            observation,
            env.grid_spec,
            include_action_mask=True,
        ).astype(np.float32)[None, :]
        start = perf_counter()
        logits, value = session.run(None, {"state": encoded})
        forward_ms_total += (perf_counter() - start) * 1000.0

        action = _masked_argmax(
            np.asarray(logits[0], dtype=np.float32),
            observation.get("action_mask"),
        )
        result = env.step(action)
        observation = result.observation
        total_reward += float(result.reward)
        done = bool(result.done)
        terminal_reason = result.info.get("reason")

        s_index = min(
            max(int(np.asarray(observation["s_index"]).reshape(-1)[0]) - 1, 0),
            env.grid_spec.s_samples - 1,
        )
        s_value = float(np.asarray(observation["s_coords"])[s_index])
        l_value = float(np.asarray(observation["l_coords"])[action])
        path_s.append(s_value)
        path_l.append(l_value)
        step_details.append(
            {
                "step": step + 1,
                "action": int(action),
                "reward": float(result.reward),
                "value": float(np.asarray(value).reshape(-1)[0]),
                "reason": terminal_reason,
                "s": s_value,
                "l": l_value,
            }
        )
        step += 1

    path_s_arr = np.asarray(path_s, dtype=np.float32)
    path_l_arr = np.asarray(path_l, dtype=np.float32)
    dp_path_s = None
    dp_path_l = None
    dp_result = env.last_scenario_dp_result
    if dp_result is not None and dp_result.feasible and dp_result.path_indices:
        full_s = np.asarray(observation["s_coords"], dtype=np.float32)
        full_l = np.asarray(observation["l_coords"], dtype=np.float32)
        dp_path_s = full_s[: len(dp_result.path_indices)]
        dp_path_l = np.asarray(
            [full_l[idx] for idx in dp_result.path_indices],
            dtype=np.float32,
        )

    saved_path = plot_episode_path(
        env.grid_spec,
        path_s_arr,
        path_l_arr,
        env.obstacles,
        env.occupancy,
        total_reward,
        dp_path_s=dp_path_s,
        dp_path_l=dp_path_l,
        output_path=output_path,
        show=False,
        paper_style=bool(args.paper_style),
    )

    print(f"ONNX path: {onnx_path}")
    print(f"Providers: {session.get_providers()}")
    print(f"Episode return: {total_reward:.3f}")
    print(f"Steps: {step}")
    print(f"Forward total: {forward_ms_total:.3f} ms")
    print(f"Forward avg: {forward_ms_total / max(step, 1):.3f} ms")
    print(f"Terminal reason: {terminal_reason}")
    print(f"Saved visualization to {saved_path}")
    print("Step details:", json.dumps(step_details, ensure_ascii=False))


if __name__ == "__main__":
    main()
