#!/usr/bin/env python3
"""
Benchmark end-to-end single-step inference with an exported ONNX policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch

from ppo import encode_observation
from rl_env import SLPathEnv
from sl_grid import GridSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX policy inference including encode + forward + env.step."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    spec = GridSpec(**payload["grid_spec"])
    env_kwargs = payload.get("environment", {})
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    episodes = max(1, int(args.episodes))
    encode_ms: list[float] = []
    forward_ms: list[float] = []
    env_step_ms: list[float] = []
    end_to_end_ms: list[float] = []
    returns: list[float] = []
    reasons: list[str | None] = []
    steps_per_episode: list[int] = []

    for _ in range(episodes):
        env = SLPathEnv(spec, **env_kwargs)
        observation = env.reset()
        done = False
        episode_return = 0.0
        episode_steps = 0

        while not done:
            t0 = perf_counter()
            encoded = encode_observation(
                observation,
                spec,
                include_action_mask=True,
            ).astype(np.float32)[None, :]
            t1 = perf_counter()

            logits, _ = session.run(None, {"state": encoded})
            t2 = perf_counter()

            logits = np.asarray(logits[0], dtype=np.float32)
            mask = np.asarray(observation.get("action_mask"), dtype=bool)
            if mask.shape[0] != logits.shape[0]:
                mask = np.ones_like(logits, dtype=bool)
            masked_logits = np.where(mask, logits, -1e9)
            action = int(np.argmax(masked_logits))

            result = env.step(action)
            t3 = perf_counter()

            encode_ms.append((t1 - t0) * 1000.0)
            forward_ms.append((t2 - t1) * 1000.0)
            env_step_ms.append((t3 - t2) * 1000.0)
            end_to_end_ms.append((t3 - t0) * 1000.0)

            observation = result.observation
            done = bool(result.done)
            episode_return += float(result.reward)
            episode_steps += 1

            if done:
                reasons.append(result.info.get("reason"))

        returns.append(episode_return)
        steps_per_episode.append(episode_steps)

    summary = {
        "episodes": episodes,
        "steps_total": int(len(end_to_end_ms)),
        "episode_steps_mean": float(np.mean(steps_per_episode)),
        "return_mean": float(np.mean(returns)),
        "success_rate": float(
            sum(1 for reason in reasons if reason == "goal_reached") / len(reasons)
        ),
        "encode_ms": _stats(encode_ms),
        "forward_ms": _stats(forward_ms),
        "env_step_ms": _stats(env_step_ms),
        "end_to_end_ms": _stats(end_to_end_ms),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
