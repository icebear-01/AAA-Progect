#!/usr/bin/env python3
"""Plot real-time success-rate comparison curves from TensorBoard logs."""

from __future__ import annotations

import argparse
import csv
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


def filter_max_update(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a real-time success-rate comparison from TensorBoard scalars."
    )
    parser.add_argument("--full-log-dir", type=Path, required=True)
    parser.add_argument("--nomask-log-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="eval/success_rate")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=51)
    parser.add_argument("--full-label", type=str, default="Full safety constraint")
    parser.add_argument("--nomask-label", type=str, default="No safety constraint")
    parser.add_argument("--title", type=str, default="Real-Time Success Rate Comparison")
    parser.add_argument("--max-update", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_steps, full_values = load_tensorboard_scalar_series(args.full_log_dir, args.tag)
    nomask_steps, nomask_values = load_tensorboard_scalar_series(args.nomask_log_dir, args.tag)

    full_steps, full_values = filter_max_update(full_steps, full_values, args.max_update)
    nomask_steps, nomask_values = filter_max_update(nomask_steps, nomask_values, args.max_update)

    full_smooth = moving_average(full_values, args.smooth_window)
    nomask_smooth = moving_average(nomask_values, args.smooth_window)

    all_updates = sorted(set(full_steps.tolist()) | set(nomask_steps.tolist()))
    full_raw_map = {int(step): float(value) for step, value in zip(full_steps.tolist(), full_values.tolist())}
    full_smooth_map = {int(step): float(value) for step, value in zip(full_steps.tolist(), full_smooth.tolist())}
    nomask_raw_map = {
        int(step): float(value) for step, value in zip(nomask_steps.tolist(), nomask_values.tolist())
    }
    nomask_smooth_map = {
        int(step): float(value) for step, value in zip(nomask_steps.tolist(), nomask_smooth.tolist())
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "update",
                "full_success_rate_raw",
                "full_success_rate_smoothed",
                "nomask_success_rate_raw",
                "nomask_success_rate_smoothed",
            ]
        )
        for update in all_updates:
            writer.writerow(
                [
                    update,
                    f"{full_raw_map.get(update, np.nan):.6f}",
                    f"{full_smooth_map.get(update, np.nan):.6f}",
                    f"{nomask_raw_map.get(update, np.nan):.6f}",
                    f"{nomask_smooth_map.get(update, np.nan):.6f}",
                ]
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    ax.plot(full_steps, full_values * 100.0, color="#9ecae1", linewidth=1.1, alpha=0.65)
    ax.plot(
        full_steps,
        full_smooth * 100.0,
        color="#1f5aa6",
        linewidth=2.6,
        label=args.full_label,
    )
    ax.plot(nomask_steps, nomask_values * 100.0, color="#fdd0a2", linewidth=1.1, alpha=0.65)
    ax.plot(
        nomask_steps,
        nomask_smooth * 100.0,
        color="#d04a02",
        linewidth=2.6,
        label=args.nomask_label,
    )

    ax.set_title(args.title)
    ax.set_xlabel("Update")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 100.2)
    ax.set_xlim(
        int(min(full_steps.min(), nomask_steps.min())),
        int(max(full_steps.max(), nomask_steps.max())),
    )
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=260)
    plt.close(fig)

    print(f"Saved real-time success comparison to {args.output_png}")
    print(f"Saved real-time comparison samples to {args.output_csv}")


if __name__ == "__main__":
    main()
