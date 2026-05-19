#!/usr/bin/env python3
"""Plot dense inferred success-rate curves for two PPO runs.

The script combines:
1. Dense reward observations from TensorBoard scalar logs.
2. Sparse measured success-rate nodes from either JSON benchmark outputs or
   TensorBoard scalar logs.

It then infers a dense success-rate curve by interpolating between measured
nodes in reward space, which is more stable than plain linear interpolation on
update indices when reward and success improve at different speeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def load_tensorboard_scalar_series(log_dir: Path, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not log_dir.is_dir():
        raise NotADirectoryError(f"TensorBoard log dir not found: {log_dir}")

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError(f"No TensorBoard event files found in {log_dir}")

    step_to_value: Dict[int, float] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            step_to_value[int(event.step)] = float(event.value)

    if not step_to_value:
        raise RuntimeError(f"Missing scalar tag '{tag}' in {log_dir}")

    steps = np.asarray(sorted(step_to_value.keys()), dtype=np.int64)
    values = np.asarray([step_to_value[int(step)] for step in steps], dtype=np.float64)
    return steps, values


def load_success_nodes_from_json(json_path: Path, key: str | None) -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if key:
        records = payload.get(key, [])
    else:
        records = payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No success-rate records found in {json_path} (key={key!r})")
    updates = np.asarray([int(record["update"]) for record in records], dtype=np.int64)
    rates = np.asarray([float(record["success_rate"]) for record in records], dtype=np.float64)
    return updates, rates


def load_success_nodes(
    *,
    json_path: Path | None,
    json_key: str | None,
    log_dir: Path | None,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if json_path is not None:
        return load_success_nodes_from_json(json_path, json_key)
    if log_dir is not None:
        return load_tensorboard_scalar_series(log_dir, tag)
    raise ValueError("Either json_path or log_dir must be provided for success nodes.")


def filter_max_step(
    steps: np.ndarray,
    values: np.ndarray,
    max_update: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_update is None:
        return steps, values
    mask = steps <= int(max_update)
    if not np.any(mask):
        raise RuntimeError(f"No samples remain after applying max_update={max_update}")
    return steps[mask], values[mask]


def add_anchor_zero(
    steps: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if steps.size == 0 or int(steps[0]) == 0:
        return steps, values
    return (
        np.concatenate([np.asarray([0], dtype=np.int64), steps]),
        np.concatenate([np.asarray([0.0], dtype=np.float64), values]),
    )


def infer_success_curve(
    reward_steps: np.ndarray,
    reward_values: np.ndarray,
    node_steps: np.ndarray,
    node_rates: np.ndarray,
) -> np.ndarray:
    anchor_rewards = np.interp(
        node_steps,
        reward_steps,
        reward_values,
        left=float(reward_values[0]),
        right=float(reward_values[-1]),
    )
    inferred = np.empty_like(reward_values)

    for index in range(len(node_steps) - 1):
        left_step = int(node_steps[index])
        right_step = int(node_steps[index + 1])
        left_rate = float(node_rates[index])
        right_rate = float(node_rates[index + 1])
        left_reward = float(anchor_rewards[index])
        right_reward = float(anchor_rewards[index + 1])

        mask = (reward_steps >= left_step) & (reward_steps <= right_step)
        if not np.any(mask):
            continue

        segment_steps = reward_steps[mask].astype(np.float64)
        segment_rewards = reward_values[mask].astype(np.float64)

        if abs(right_reward - left_reward) < 1e-9:
            alpha = (segment_steps - float(left_step)) / max(1.0, float(right_step - left_step))
        else:
            alpha = (segment_rewards - left_reward) / (right_reward - left_reward)
        alpha = np.clip(alpha, 0.0, 1.0)
        inferred[mask] = left_rate + alpha * (right_rate - left_rate)

    before_mask = reward_steps < node_steps[0]
    if np.any(before_mask):
        inferred[before_mask] = float(node_rates[0])
    after_mask = reward_steps > node_steps[-1]
    if np.any(after_mask):
        inferred[after_mask] = float(node_rates[-1])

    return np.clip(inferred, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dense inferred success curves for two ablation runs."
    )
    parser.add_argument("--full-label", type=str, default="Full safety constraint")
    parser.add_argument("--full-reward-log-dir", type=Path, required=True)
    parser.add_argument("--full-success-json", type=Path, required=True)
    parser.add_argument("--full-success-key", type=str, default=None)

    parser.add_argument("--nomask-label", type=str, default="No safety constraint")
    parser.add_argument("--nomask-reward-log-dir", type=Path, required=True)
    parser.add_argument("--nomask-success-json", type=Path, default=None)
    parser.add_argument("--nomask-success-key", type=str, default=None)
    parser.add_argument("--nomask-success-log-dir", type=Path, default=None)
    parser.add_argument("--nomask-success-tag", type=str, default="eval/success_rate")

    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=31)
    parser.add_argument(
        "--curve-smooth-window",
        type=int,
        default=1,
        help="Optional moving-average window applied to the inferred success curve itself.",
    )
    parser.add_argument("--max-update", type=int, default=None)
    parser.add_argument(
        "--anchor-zero",
        action="store_true",
        help="Add a synthetic (0, 0) success-rate anchor before inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_reward_steps, full_reward_values = load_tensorboard_scalar_series(
        args.full_reward_log_dir,
        "train/episode_return",
    )
    full_reward_steps, full_reward_values = filter_max_step(
        full_reward_steps,
        full_reward_values,
        args.max_update,
    )
    full_reward_smooth = moving_average(full_reward_values, args.smooth_window)
    full_node_steps, full_node_rates = load_success_nodes(
        json_path=args.full_success_json,
        json_key=args.full_success_key,
        log_dir=None,
        tag="eval/success_rate",
    )
    full_node_steps, full_node_rates = filter_max_step(full_node_steps, full_node_rates, args.max_update)

    nomask_reward_steps, nomask_reward_values = load_tensorboard_scalar_series(
        args.nomask_reward_log_dir,
        "train/episode_return",
    )
    nomask_reward_steps, nomask_reward_values = filter_max_step(
        nomask_reward_steps,
        nomask_reward_values,
        args.max_update,
    )
    nomask_reward_smooth = moving_average(nomask_reward_values, args.smooth_window)
    nomask_node_steps, nomask_node_rates = load_success_nodes(
        json_path=args.nomask_success_json,
        json_key=args.nomask_success_key,
        log_dir=args.nomask_success_log_dir or args.nomask_reward_log_dir,
        tag=args.nomask_success_tag,
    )
    nomask_node_steps, nomask_node_rates = filter_max_step(
        nomask_node_steps,
        nomask_node_rates,
        args.max_update,
    )

    if args.anchor_zero:
        full_node_steps, full_node_rates = add_anchor_zero(full_node_steps, full_node_rates)
        nomask_node_steps, nomask_node_rates = add_anchor_zero(nomask_node_steps, nomask_node_rates)

    full_inferred = infer_success_curve(
        full_reward_steps,
        full_reward_smooth,
        full_node_steps,
        full_node_rates,
    )
    nomask_inferred = infer_success_curve(
        nomask_reward_steps,
        nomask_reward_smooth,
        nomask_node_steps,
        nomask_node_rates,
    )
    full_inferred = moving_average(full_inferred, args.curve_smooth_window)
    nomask_inferred = moving_average(nomask_inferred, args.curve_smooth_window)

    all_updates = sorted(
        set(full_reward_steps.tolist()) | set(nomask_reward_steps.tolist())
    )
    full_step_to_reward = {
        int(step): float(value) for step, value in zip(full_reward_steps.tolist(), full_reward_smooth.tolist())
    }
    full_step_to_rate = {
        int(step): float(value) for step, value in zip(full_reward_steps.tolist(), full_inferred.tolist())
    }
    nomask_step_to_reward = {
        int(step): float(value) for step, value in zip(nomask_reward_steps.tolist(), nomask_reward_smooth.tolist())
    }
    nomask_step_to_rate = {
        int(step): float(value) for step, value in zip(nomask_reward_steps.tolist(), nomask_inferred.tolist())
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "update",
                "full_reward_smoothed",
                "full_success_rate_inferred",
                "nomask_reward_smoothed",
                "nomask_success_rate_inferred",
            ]
        )
        for update in all_updates:
            writer.writerow(
                [
                    update,
                    f"{full_step_to_reward.get(update, np.nan):.6f}",
                    f"{full_step_to_rate.get(update, np.nan):.6f}",
                    f"{nomask_step_to_reward.get(update, np.nan):.6f}",
                    f"{nomask_step_to_rate.get(update, np.nan):.6f}",
                ]
            )

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

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(
        full_reward_steps,
        full_inferred * 100.0,
        color="#1f5aa6",
        linewidth=2.7,
        label=args.full_label,
    )
    ax.scatter(
        full_node_steps,
        full_node_rates * 100.0,
        color="#1f5aa6",
        s=16,
        alpha=0.75,
        zorder=5,
    )
    ax.plot(
        nomask_reward_steps,
        nomask_inferred * 100.0,
        color="#d04a02",
        linewidth=2.7,
        label=args.nomask_label,
    )
    ax.scatter(
        nomask_node_steps,
        nomask_node_rates * 100.0,
        color="#d04a02",
        s=16,
        alpha=0.75,
        zorder=5,
    )

    ax.set_xlabel("Update")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 100.2)
    min_visible_update = int(min(full_reward_steps.min(), nomask_reward_steps.min()))
    if args.max_update is not None:
        ax.set_xlim(min_visible_update, int(args.max_update))
    else:
        ax.set_xlim(min_visible_update, int(max(full_reward_steps.max(), nomask_reward_steps.max())))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=260)
    plt.close(fig)

    print(f"Saved dense inferred success curve to {args.output_png}")
    print(f"Saved dense inferred samples to {args.output_csv}")


if __name__ == "__main__":
    main()
