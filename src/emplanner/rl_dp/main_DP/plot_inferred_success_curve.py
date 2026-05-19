#!/usr/bin/env python3
"""Plot a reward-guided inferred success-rate curve anchored by measured checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Sequence, Tuple

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


def load_reward_curve(log_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(
        r"Update\s+(\d+)\s+\|\s+avg_return=([-\d.]+)\s+\|\s+policy_loss=([-\d.]+)\s+\|\s+value_loss=([-\d.]+)\s+\|\s+kl=([-\d.]+)"
    )
    steps = []
    rewards = []
    for line in log_file.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        steps.append(int(match.group(1)))
        rewards.append(float(match.group(2)))
    if not steps:
        raise RuntimeError(f"No reward records found in {log_file}")
    return np.asarray(steps, dtype=np.int64), np.asarray(rewards, dtype=np.float64)


def load_success_nodes(json_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(json_file.read_text())
    records = payload.get("records", [])
    if not records:
        raise RuntimeError(f"No success-rate records found in {json_file}")
    updates = np.asarray([int(record["update"]) for record in records], dtype=np.int64)
    rates = np.asarray([float(record["success_rate"]) for record in records], dtype=np.float64)
    return updates, rates


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
    parser = argparse.ArgumentParser(description="Plot inferred success curve from reward and sparse success nodes.")
    parser.add_argument("--log-file", type=Path, required=True, help="Training stdout log file with avg_return records")
    parser.add_argument("--success-json", type=Path, required=True, help="Measured checkpoint success-rate JSON")
    parser.add_argument("--output-png", type=Path, required=True, help="Output figure path")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--smooth-window", type=int, default=21, help="Reward moving-average window")
    parser.add_argument("--anchor-zero", action="store_true", help="Add a synthetic (0, 0) success-rate anchor")
    parser.add_argument("--monotonic-up", action="store_true", help="Force node success rates to be nondecreasing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reward_steps, reward_values = load_reward_curve(args.log_file)
    reward_smooth = moving_average(reward_values, args.smooth_window)
    node_steps, node_rates = load_success_nodes(args.success_json)
    if args.anchor_zero:
        node_steps = np.concatenate([np.asarray([0], dtype=np.int64), node_steps])
        node_rates = np.concatenate([np.asarray([0.0], dtype=np.float64), node_rates])
    if args.monotonic_up:
        node_rates = np.maximum.accumulate(node_rates)
    inferred_rates = infer_success_curve(reward_steps, reward_smooth, node_steps, node_rates)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["update", "reward_smoothed", "success_rate_inferred"])
        for step, reward, rate in zip(reward_steps.tolist(), reward_smooth.tolist(), inferred_rates.tolist()):
            writer.writerow([step, f"{reward:.6f}", f"{rate:.6f}"])

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
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(reward_steps, inferred_rates * 100.0, color="#1f5aa6", linewidth=2.6, label="Inferred Success Rate")
    ax.scatter(node_steps, node_rates * 100.0, color="#d04a02", s=38, zorder=5, label="Measured Nodes")
    ax.set_xlabel("Update")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(96.5, 100.2)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=260)
    plt.close(fig)
    print(f"Saved inferred success curve to {args.output_png}")
    print(f"Saved inferred success samples to {args.output_csv}")


if __name__ == "__main__":
    main()
