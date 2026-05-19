#!/usr/bin/env python3
"""Plot PPO training curves with paper-style scientific colors."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO training curves from TensorBoard or log text.")
    parser.add_argument("--log-dir", type=Path, required=True, help="TensorBoard log directory")
    parser.add_argument("--log-file", type=Path, default=None, help="optional stdout log fallback")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    parser.add_argument("--smooth-window", type=int, default=31, help="moving average window")
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


def load_from_tensorboard(log_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    required = [
        "train/episode_return",
        "train/policy_loss",
        "train/value_loss",
        "train/kl_divergence",
    ]
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag in required:
        if tag not in tags:
            continue
        events = accumulator.Scalars(tag)
        steps = np.asarray([event.step for event in events], dtype=np.int64)
        values = np.asarray([event.value for event in events], dtype=np.float64)
        result[tag] = (steps, values)
    return result


def load_from_log(log_file: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pattern = re.compile(
        r"Update\s+(\d+)\s+\|\s+avg_return=([-\d.]+)\s+\|\s+policy_loss=([-\d.]+)\s+\|\s+value_loss=([-\d.]+)\s+\|\s+kl=([-\d.]+)"
    )
    updates: List[int] = []
    avg_return: List[float] = []
    policy_loss: List[float] = []
    value_loss: List[float] = []
    kl: List[float] = []
    for line in log_file.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        updates.append(int(match.group(1)))
        avg_return.append(float(match.group(2)))
        policy_loss.append(float(match.group(3)))
        value_loss.append(float(match.group(4)))
        kl.append(float(match.group(5)))
    steps = np.asarray(updates, dtype=np.int64)
    return {
        "train/episode_return": (steps, np.asarray(avg_return, dtype=np.float64)),
        "train/policy_loss": (steps, np.asarray(policy_loss, dtype=np.float64)),
        "train/value_loss": (steps, np.asarray(value_loss, dtype=np.float64)),
        "train/kl_divergence": (steps, np.asarray(kl, dtype=np.float64)),
    }


def plot_series(ax, steps: np.ndarray, values: np.ndarray, title: str, color_raw: str, color_smooth: str, window: int) -> None:
    ax.plot(steps, values, color=color_raw, alpha=0.25, linewidth=1.2)
    ax.plot(steps, moving_average(values, window), color=color_smooth, linewidth=2.4)
    ax.set_title(title, fontsize=15, fontweight="semibold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=12)


def main() -> None:
    args = parse_args()
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if args.log_dir.exists():
        try:
            curves = load_from_tensorboard(args.log_dir)
        except Exception:
            curves = {}
    if not curves:
        if args.log_file is None or not args.log_file.exists():
            raise RuntimeError("No TensorBoard scalars found and no valid --log-file fallback provided.")
        curves = load_from_log(args.log_file)

    tags = [
        ("train/episode_return", "Reward", "#9ecae1", "#1f5aa6"),
        ("train/policy_loss", "Policy Loss", "#fdd0a2", "#d04a02"),
        ("train/value_loss", "Value Loss", "#b8e0c8", "#2f7f5f"),
        ("train/kl_divergence", "KL Divergence", "#f4b6c2", "#b03060"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    axes = axes.ravel()

    for ax, (tag, title, color_raw, color_smooth) in zip(axes, tags):
        if tag not in curves:
            ax.set_visible(False)
            continue
        steps, values = curves[tag]
        plot_series(ax, steps, values, title, color_raw, color_smooth, args.smooth_window)
        ax.set_xlabel("Update")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {args.output}")


if __name__ == "__main__":
    main()
