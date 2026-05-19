#!/usr/bin/env python3
"""Plot reward-only ablation curves for multiple PPO runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

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


def load_reward_curve(log_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tag = "train/episode_return"
    if tag not in accumulator.Tags().get("scalars", []):
        raise RuntimeError(f"Missing scalar tag '{tag}' in {log_dir}")
    events = accumulator.Scalars(tag)
    steps = np.asarray([event.step for event in events], dtype=np.int64)
    values = np.asarray([event.value for event in events], dtype=np.float64)
    return steps, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward ablation curves.")
    parser.add_argument("--log-dir", action="append", required=True, help="TensorBoard log directory; repeatable")
    parser.add_argument("--label", action="append", required=True, help="Legend label; repeatable and aligned with --log-dir")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--smooth-window", type=int, default=31, help="Moving average window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.log_dir) != len(args.label):
        raise ValueError("--log-dir and --label must have the same count")

    series: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for log_dir_str, label in zip(args.log_dir, args.label):
        steps, values = load_reward_curve(Path(log_dir_str))
        series.append((label, steps, values))

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

    colors = [
        ("#9ecae1", "#1f5aa6"),
        ("#fdd0a2", "#d04a02"),
        ("#b8e0c8", "#2f7f5f"),
        ("#f4b6c2", "#b03060"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for idx, (label, steps, values) in enumerate(series):
        raw_color, smooth_color = colors[idx % len(colors)]
        ax.plot(steps, values, color=raw_color, alpha=0.22, linewidth=1.1)
        ax.plot(steps, moving_average(values, args.smooth_window), color=smooth_color, linewidth=2.5, label=label)

    ax.set_xlabel("Update")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, framealpha=0.92)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260)
    plt.close(fig)
    print(f"Saved ablation reward curves to {args.output}")


if __name__ == "__main__":
    main()
