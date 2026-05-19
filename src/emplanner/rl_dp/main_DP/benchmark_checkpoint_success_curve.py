#!/usr/bin/env python3
"""Evaluate RL-DP success rate over training checkpoints on a fixed validation subset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo import ActorCritic, PPOConfig, encode_observation
from rl_env import DPCandidateResult, SLPathEnv
from sl_grid import GridSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RL-DP success rate over checkpoints.")
    parser.add_argument("--dataset", type=Path, required=True, help="screened scenario dataset json")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="directory containing training checkpoints")
    parser.add_argument("--checkpoint-prefix", type=str, required=True, help="checkpoint filename prefix stem")
    parser.add_argument("--output-json", type=Path, required=True, help="output json path")
    parser.add_argument("--output-png", type=Path, required=True, help="output png path")
    parser.add_argument("--eval-count", type=int, default=500, help="number of fixed validation scenarios")
    parser.add_argument("--step-interval", type=int, default=2000, help="checkpoint update interval to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="evaluation device")
    return parser.parse_args()


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, min(int(window), int(arr.size)))
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"_update_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _load_checkpoint_policy_env(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[ActorCritic, SLPathEnv, PPOConfig]:
    payload = torch.load(checkpoint_path, map_location=device)
    config_payload = dict(payload["config"])
    config_payload.setdefault(
        "include_action_mask_in_state",
        PPOConfig.__dataclass_fields__["include_action_mask_in_state"].default,
    )
    config_payload.setdefault(
        "apply_action_mask",
        PPOConfig.__dataclass_fields__["apply_action_mask"].default,
    )
    config = PPOConfig(**config_payload)
    spec = GridSpec(**payload["grid_spec"])

    env_kwargs: Dict[str, object] = dict(payload.get("environment", {}))
    env_kwargs["scenario_dataset_path"] = None
    env = SLPathEnv(spec, **env_kwargs)

    sample_obs = env.reset(start_l=0.0)
    feature_dim = encode_observation(
        sample_obs,
        spec,
        include_action_mask=config.include_action_mask_in_state,
    ).shape[0]
    action_dim = spec.l_samples

    occupancy_shape = (spec.s_samples, spec.l_samples)
    grid_spatial_size = occupancy_shape[0] * occupancy_shape[1]
    extra_dim = feature_dim % grid_spatial_size
    grid_channels = (feature_dim - extra_dim) // grid_spatial_size
    if grid_channels < 1:
        raise ValueError("Encoded observation must contain at least one grid plane.")

    policy = ActorCritic(
        occupancy_shape=occupancy_shape,
        grid_channels=grid_channels,
        extra_dim=extra_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    policy.load_state_dict(payload["model_state"])
    policy.eval()
    setattr(policy, "include_action_mask_input", bool(config.include_action_mask_in_state))
    setattr(policy, "mask_logits_with_action_mask", bool(config.apply_action_mask))
    return policy, env, config


def _set_fixed_scenario(env: SLPathEnv, scenario_record: Dict[str, object]) -> Dict[str, np.ndarray]:
    obstacles = tuple(env._obstacle_from_record(entry) for entry in scenario_record["obstacles"])  # type: ignore[attr-defined]
    occupancy = env._build_occupancy(obstacles)  # type: ignore[attr-defined]
    runtime_cache = env._build_runtime_feature_cache(obstacles, occupancy)  # type: ignore[attr-defined]
    env._obstacles = obstacles  # type: ignore[attr-defined]
    env._apply_runtime_feature_cache(runtime_cache, occupancy)  # type: ignore[attr-defined]
    scenario_index = int(scenario_record.get("scenario_index", 0))
    env._active_scenario_index = scenario_index  # type: ignore[attr-defined]
    env._active_transition_cache = env._build_transition_cache(scenario_index, obstacles)  # type: ignore[attr-defined]
    start_l = float(scenario_record.get("start_l", 0.0))
    initial_l = int(np.argmin(np.abs(env._l_coords - start_l)))  # type: ignore[attr-defined]
    env._start_l = start_l  # type: ignore[attr-defined]
    env._path_indices = [initial_l]  # type: ignore[attr-defined]
    env._s_index = 1 if env.grid_spec.s_samples > 1 else 0  # type: ignore[attr-defined]

    path_indices = scenario_record.get("path_indices")
    if isinstance(path_indices, list) and path_indices:
        env._last_scenario_dp_result = DPCandidateResult(  # type: ignore[attr-defined]
            feasible=True,
            total_cost=float(scenario_record.get("dp_total_cost", 0.0)),
            avg_cost=float(scenario_record.get("dp_avg_cost", 0.0)),
            path_indices=tuple(int(v) for v in path_indices),
        )
    else:
        env._last_scenario_dp_result = None  # type: ignore[attr-defined]
    return env._build_observation()  # type: ignore[attr-defined]


def _choose_action(
    policy: ActorCritic,
    observation: Dict[str, np.ndarray],
    spec: GridSpec,
    device: torch.device,
) -> int:
    include_mask = bool(getattr(policy, "include_action_mask_input", True))
    apply_mask_logits = bool(getattr(policy, "mask_logits_with_action_mask", True))
    encoded = encode_observation(
        observation,
        spec,
        include_action_mask=include_mask,
    )
    state = torch.as_tensor(encoded, device=device)
    with torch.no_grad():
        logits, _ = policy(state)
    if apply_mask_logits:
        mask_tensor = torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
        if bool(mask_tensor.any().item()):
            logits = logits.masked_fill(~mask_tensor, -1e9)
    return int(torch.argmax(logits).item())


def evaluate_checkpoint(
    checkpoint_path: Path,
    scenario_records: Sequence[Dict[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    policy, env, _ = _load_checkpoint_policy_env(checkpoint_path, device)
    success_count = 0
    reasons: Dict[str, int] = {}

    try:
        for record in scenario_records:
            observation = _set_fixed_scenario(env, record)
            done = False
            last_reason = "unknown"
            step_budget = max(1, env.grid_spec.s_samples + 2)
            step_count = 0
            while not done and step_count < step_budget:
                action = _choose_action(policy, observation, env.grid_spec, device)
                result = env.step(action)
                observation = result.observation
                done = result.done
                last_reason = str(result.info.get("reason", "unknown"))
                step_count += 1
            reasons[last_reason] = reasons.get(last_reason, 0) + 1
            if last_reason == "goal_reached":
                success_count += 1
    finally:
        env.reset()

    total = len(scenario_records)
    return {
        "checkpoint": str(checkpoint_path),
        "update": _checkpoint_sort_key(checkpoint_path),
        "success": int(success_count),
        "total": int(total),
        "success_rate": float(success_count / max(1, total)),
        "reasons": reasons,
    }


def plot_curve(records: Sequence[Dict[str, object]], output_path: Path) -> None:
    updates = np.asarray([int(item["update"]) for item in records], dtype=np.int64)
    success_rates = np.asarray([float(item["success_rate"]) for item in records], dtype=np.float64)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(
        updates,
        success_rates,
        color="#1f5aa6",
        linewidth=2.5,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    ax.fill_between(updates, success_rates, 0.0, color="#9ecae1", alpha=0.18)
    ax.set_xlabel("Update")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(int(updates.min()), int(updates.max()))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_payload = json.loads(args.dataset.read_text())
    scenarios = dataset_payload["scenarios"]
    if not scenarios:
        raise RuntimeError("dataset has no scenarios")

    eval_count = min(max(1, int(args.eval_count)), len(scenarios))
    if eval_count == len(scenarios):
        selected_indices = list(range(len(scenarios)))
    else:
        selected_indices = np.linspace(0, len(scenarios) - 1, eval_count, dtype=int).tolist()
    scenario_records = [scenarios[idx] for idx in selected_indices]

    pattern = f"{args.checkpoint_prefix}_update_*.pt"
    checkpoints = sorted(args.checkpoint_dir.glob(pattern), key=_checkpoint_sort_key)
    checkpoints = [path for path in checkpoints if _checkpoint_sort_key(path) > 0]
    if args.step_interval > 1:
        checkpoints = [path for path in checkpoints if _checkpoint_sort_key(path) % args.step_interval == 0]
    if not checkpoints:
        raise RuntimeError(f"no checkpoints matched pattern {pattern}")

    device = torch.device(args.device)
    results: List[Dict[str, object]] = []
    for checkpoint_path in checkpoints:
        result = evaluate_checkpoint(checkpoint_path, scenario_records, device)
        results.append(result)
        print(
            f"update={result['update']:>5d} | "
            f"success={result['success']}/{result['total']} | "
            f"rate={result['success_rate']:.4f}"
        )

    payload = {
        "dataset": str(args.dataset),
        "eval_count": int(eval_count),
        "selected_indices": selected_indices,
        "records": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    plot_curve(results, args.output_png)
    print(f"Saved json to {args.output_json}")
    print(f"Saved plot to {args.output_png}")


if __name__ == "__main__":
    main()
