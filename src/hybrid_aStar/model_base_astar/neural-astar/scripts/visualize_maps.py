"""Visualize map datasets used by Neural A*.

Supported inputs:
- .npz files created by planning-datasets (arr_0..arr_11 format)
- Directory containing <split>_maps.npy (WarCraft-style data)
- A standalone .npy map tensor
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_maps(
    dataset_path: Path, split: str
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Load maps and optional goal maps for visualization."""
    if dataset_path.is_file() and dataset_path.suffix == ".npz":
        split_to_idx = {"train": 0, "valid": 4, "test": 8}
        idx = split_to_idx[split]
        with np.load(dataset_path) as data:
            maps = data[f"arr_{idx}"]
            goals = data[f"arr_{idx + 1}"]
        return maps, goals, "npz"

    if dataset_path.is_dir():
        maps_file = dataset_path / f"{split}_maps.npy"
        if maps_file.exists():
            maps = np.load(maps_file)
            return maps, None, "warcraft_dir"

    if dataset_path.is_file() and dataset_path.suffix == ".npy":
        maps = np.load(dataset_path)
        return maps, None, "npy"

    raise ValueError(
        f"Unsupported dataset path: {dataset_path}. "
        "Use a .npz file, a .npy file, or a directory with <split>_maps.npy."
    )


def to_plot_image(sample: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    """Convert map sample to an image and optional colormap."""
    if sample.ndim == 2:
        return sample, "gray"

    if sample.ndim == 3:
        # CHW -> HWC
        if sample.shape[0] in (1, 3, 4):
            sample = np.transpose(sample, (1, 2, 0))

        if sample.shape[-1] == 1:
            return sample[..., 0], "gray"

        if sample.dtype != np.uint8:
            if sample.max() > 1.0:
                sample = np.clip(sample / 255.0, 0.0, 1.0)
            else:
                sample = np.clip(sample, 0.0, 1.0)
        return sample, None

    raise ValueError(f"Unsupported sample shape: {sample.shape}")


def to_goal_map(goal_sample: np.ndarray) -> np.ndarray:
    """Convert goal map sample to HxW."""
    if goal_sample.ndim == 2:
        return goal_sample
    if goal_sample.ndim == 3 and goal_sample.shape[0] == 1:
        return goal_sample[0]
    if goal_sample.ndim == 3 and goal_sample.shape[-1] == 1:
        return goal_sample[..., 0]
    raise ValueError(f"Unsupported goal map shape: {goal_sample.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize map datasets and save a grid preview image."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to .npz/.npy dataset, or directory with <split>_maps.npy.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="train",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="How many maps to plot.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the output grid.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample maps randomly (without replacement).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used with --random.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index when not using --random.",
    )
    parser.add_argument(
        "--show-goal",
        action="store_true",
        help="Overlay goal location (only for .npz datasets with goal maps).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output png path. Default: outputs/map_preview_<name>_<split>.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image dpi for saved figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    maps, goals, source_kind = load_maps(args.dataset, args.split)
    total = int(maps.shape[0])
    if total == 0:
        raise ValueError("Dataset split is empty.")

    count = min(args.count, total)
    if count <= 0:
        raise ValueError("--count must be positive.")
    if args.cols <= 0:
        raise ValueError("--cols must be positive.")

    if args.random:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(total, size=count, replace=False)
    else:
        start = max(args.start_index, 0)
        end = min(start + count, total)
        indices = np.arange(start, end)
        if len(indices) == 0:
            raise ValueError(
                f"--start-index={args.start_index} is out of range for total={total}."
            )
        count = len(indices)

    rows = int(math.ceil(count / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 2.6, rows * 2.6))
    axes = np.array(axes).reshape(rows, args.cols)

    for plot_i, data_i in enumerate(indices):
        r = plot_i // args.cols
        c = plot_i % args.cols
        ax = axes[r, c]

        image, cmap = to_plot_image(maps[data_i])
        ax.imshow(image, cmap=cmap, vmin=0.0 if cmap == "gray" else None, vmax=1.0 if cmap == "gray" else None)

        if args.show_goal and goals is not None:
            goal = to_goal_map(goals[data_i])
            ys, xs = np.where(goal > 0.5)
            if len(xs) > 0:
                ax.scatter(xs, ys, s=14, c="red", marker="x", linewidths=0.8)

        ax.set_title(f"idx={int(data_i)}", fontsize=8)
        ax.axis("off")

    # Hide unused subplots.
    for plot_i in range(count, rows * args.cols):
        r = plot_i // args.cols
        c = plot_i % args.cols
        axes[r, c].axis("off")

    name = args.dataset.stem if args.dataset.is_file() else args.dataset.name
    output_path = (
        args.output
        if args.output is not None
        else Path("outputs") / f"map_preview_{name}_{args.split}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.suptitle(
        f"{name} | split={args.split} | source={source_kind} | count={count}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"saved: {output_path}")
    print(f"total maps in split: {total}")
    print(f"indices: {indices.tolist()}")


if __name__ == "__main__":
    main()
