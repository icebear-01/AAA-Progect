#!/usr/bin/env python3
"""Plot dense real-time success-rate curves from TensorBoard logs."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a real-time success-rate curve from TensorBoard scalars."
    )
    parser.add_argument("--log-dir", type=Path, required=True, help="TensorBoard log directory")
    parser.add_argument(
        "--tag",
        type=str,
        default="eval/success_rate",
        help="Scalar tag to plot",
    )
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=31,
        help="Moving-average window for the smooth overlay",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Real-Time Success Rate",
        help="Plot title",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Success rate",
        help="Legend label",
    )
    parser.add_argument(
        "--max-update",
        type=int,
        default=None,
        help="Optional maximum update to keep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps, values = load_tensorboard_scalar_series(args.log_dir, args.tag)
    if args.max_update is not None:
        mask = steps <= int(args.max_update)
        if not np.any(mask):
            raise RuntimeError(f"No samples remain after applying max_update={args.max_update}")
        steps = steps[mask]
        values = values[mask]

    smooth_values = moving_average(values, args.smooth_window)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["update", "success_rate_raw", "success_rate_smoothed"])
        for step, raw_value, smooth_value in zip(steps.tolist(), values.tolist(), smooth_values.tolist()):
            writer.writerow(
                [
                    int(step),
                    f"{float(raw_value):.6f}",
                    f"{float(smooth_value):.6f}",
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

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(
        steps,
        values * 100.0,
        color="#9ecae1",
        linewidth=1.1,
        alpha=0.8,
        label=f"{args.label} (raw)",
    )
    ax.scatter(
        steps,
        values * 100.0,
        color="#9ecae1",
        s=8,
        alpha=0.35,
    )
    ax.plot(
        steps,
        smooth_values * 100.0,
        color="#1f5aa6",
        linewidth=2.6,
        label=f"{args.label} (smoothed)",
    )

    ax.set_title(args.title)
    ax.set_xlabel("Update")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0.0, 100.2)
    ax.set_xlim(int(steps.min()), int(steps.max()))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=260)
    plt.close(fig)

    print(f"Saved real-time success curve to {args.output_png}")
    print(f"Saved real-time success samples to {args.output_csv}")


if __name__ == "__main__":
    main()
