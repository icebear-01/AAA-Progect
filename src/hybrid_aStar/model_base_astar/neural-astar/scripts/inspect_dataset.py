"""Inspect one parking guidance dataset sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_astar.datasets import ParkingGuidanceDataset
from neural_astar.utils.coords import rc_to_xy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize one sample from parking guidance dataset")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def one_hot_to_xy(one_hot_1hw: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(one_hot_1hw, dtype=np.float32)[0]
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return rc_to_xy(y, x, width=arr.shape[1], height=arr.shape[0])


def main() -> None:
    args = parse_args()
    ds = ParkingGuidanceDataset(args.data_dir)
    idx = max(0, min(args.index, len(ds) - 1))
    sample = ds[idx]

    occ = sample["occ_map"].numpy()[0]
    start = sample["start_map"].numpy()
    goal = sample["goal_map"].numpy()
    opt = sample["opt_traj"].numpy()[0]

    print(f"dataset size: {len(ds)}")
    print(f"index: {idx}")
    print(f"occ_map shape: {sample['occ_map'].shape}")
    print(f"start_map shape: {sample['start_map'].shape}")
    print(f"goal_map shape: {sample['goal_map'].shape}")
    print(f"opt_traj shape: {sample['opt_traj'].shape}")
    if "target_cost" in sample:
        print(f"target_cost shape: {sample['target_cost'].shape}")
    print(f"start_xy: {one_hot_to_xy(start)}")
    print(f"goal_xy: {one_hot_to_xy(goal)}")
    if "start_pose" in sample:
        print(f"start_pose: {sample['start_pose'].numpy().tolist()}")
    if "goal_pose" in sample:
        print(f"goal_pose: {sample['goal_pose'].numpy().tolist()}")

    show_target = "target_cost" in sample
    ncols = 5 if show_target else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    axes[0].imshow(occ, cmap="gray")
    axes[0].set_title("occ_map (1=obstacle)")
    axes[1].imshow(start[0], cmap="viridis")
    axes[1].set_title("start_map")
    axes[2].imshow(goal[0], cmap="viridis")
    axes[2].set_title("goal_map")
    axes[3].imshow(opt, cmap="magma")
    axes[3].set_title("opt_traj")
    if show_target:
        axes[4].imshow(sample["target_cost"].numpy()[0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[4].set_title("target_cost")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
