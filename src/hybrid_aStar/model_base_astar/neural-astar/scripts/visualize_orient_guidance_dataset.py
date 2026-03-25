"""Visualize orientation-aware guidance dataset samples."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize one orientation-aware guidance dataset sample")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--top-k-bins", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _yaw_bin_center_deg(bin_idx: int, yaw_bins: int) -> float:
    return -180.0 + (360.0 * float(bin_idx) / float(yaw_bins))


def _load_sample(file_path: Path) -> dict:
    with np.load(file_path) as data:
        sample = {key: np.asarray(data[key]) for key in data.files}
    return sample


def _orientation_assignment(opt_traj_orient: np.ndarray) -> np.ndarray:
    active_mask = np.sum(opt_traj_orient, axis=0) > 0.5
    assign = np.argmax(opt_traj_orient, axis=0).astype(np.float32)
    assign[~active_mask] = np.nan
    return assign


def _plot_path(ax: plt.Axes, path_poses: np.ndarray, color: str, linewidth: float = 1.4) -> None:
    if path_poses.ndim != 2 or path_poses.shape[0] == 0:
        return
    ax.plot(path_poses[:, 0], path_poses[:, 1], color=color, linewidth=linewidth, alpha=0.95)


def main() -> None:
    args = parse_args()
    files: List[Path] = sorted(args.data_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {args.data_dir}")

    idx = max(0, min(int(args.index), len(files) - 1))
    file_path = files[idx]
    sample = _load_sample(file_path)

    occ = np.asarray(sample["occ_map"], dtype=np.float32)
    if occ.ndim == 3 and occ.shape[0] == 1:
        occ = occ[0]
    start_pose = np.asarray(sample.get("start_pose", np.array([0.0, 0.0, 0.0], dtype=np.float32)))
    goal_pose = np.asarray(sample.get("goal_pose", np.array([0.0, 0.0, 0.0], dtype=np.float32)))
    opt_traj = np.asarray(sample["opt_traj"], dtype=np.float32)[0]
    target_cost = np.asarray(sample["target_cost"], dtype=np.float32)[0]
    path_poses = np.asarray(sample.get("path_poses", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)

    opt_traj_orient = sample.get("opt_traj_orient")
    target_cost_orient = sample.get("target_cost_orient")
    if opt_traj_orient is not None:
        opt_traj_orient = np.asarray(opt_traj_orient, dtype=np.float32)
    if target_cost_orient is not None:
        target_cost_orient = np.asarray(target_cost_orient, dtype=np.float32)

    top_k = max(1, int(args.top_k_bins))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].imshow(opt_traj, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    _plot_path(axes[0, 0], path_poses, color="deepskyblue", linewidth=1.2)
    axes[0, 0].scatter([start_pose[0]], [start_pose[1]], c="lime", s=55, marker="o", label="start")
    axes[0, 0].scatter([goal_pose[0]], [goal_pose[1]], c="red", s=55, marker="x", label="goal")
    axes[0, 0].set_title("Map + Corridor + path_poses")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].set_axis_off()

    im_cost = axes[0, 1].imshow(target_cost, cmap="viridis", vmin=0.0, vmax=1.0)
    _plot_path(axes[0, 1], path_poses, color="white", linewidth=1.0)
    axes[0, 1].scatter([start_pose[0]], [start_pose[1]], c="lime", s=45, marker="o")
    axes[0, 1].scatter([goal_pose[0]], [goal_pose[1]], c="red", s=45, marker="x")
    axes[0, 1].set_title("target_cost")
    axes[0, 1].set_axis_off()
    cbar_cost = fig.colorbar(im_cost, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar_cost.set_label("cost")

    if opt_traj_orient is not None:
        assign = _orientation_assignment(opt_traj_orient)
        im_assign = axes[0, 2].imshow(assign, cmap="hsv", vmin=0.0, vmax=float(opt_traj_orient.shape[0]))
        axes[0, 2].imshow(occ, cmap="gray_r", alpha=0.18, vmin=0.0, vmax=1.0)
        _plot_path(axes[0, 2], path_poses, color="white", linewidth=0.8)
        axes[0, 2].set_title("corridor yaw-bin assignment")
        axes[0, 2].set_axis_off()
        cbar_assign = fig.colorbar(im_assign, ax=axes[0, 2], fraction=0.046, pad=0.04)
        cbar_assign.set_label("yaw bin")
        bin_mass = np.sum(opt_traj_orient, axis=(1, 2))
        active_bins = np.where(bin_mass > 0.5)[0].tolist()
        ranked_bins = list(np.argsort(-bin_mass))
    else:
        active_bins = []
        ranked_bins = []
        axes[0, 2].imshow(opt_traj, cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, 2].set_title("No orientation channels")
        axes[0, 2].set_axis_off()

    for panel_id in range(3):
        ax = axes[1, panel_id]
        if panel_id < len(ranked_bins) and target_cost_orient is not None and opt_traj_orient is not None:
            bin_idx = int(ranked_bins[panel_id])
            if np.sum(opt_traj_orient[bin_idx]) > 0.5:
                im = ax.imshow(target_cost_orient[bin_idx], cmap="viridis", vmin=0.0, vmax=1.0)
                ax.imshow(opt_traj_orient[bin_idx], cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
                ax.set_title(
                    f"bin={bin_idx} center={_yaw_bin_center_deg(bin_idx, opt_traj_orient.shape[0]):.1f} deg "
                    f"cells={int(np.sum(opt_traj_orient[bin_idx] > 0.5))}"
                )
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                continue
        ax.axis("off")
        ax.set_title("unused")

    path_len = 0.0
    if path_poses.shape[0] >= 2:
        diffs = np.diff(path_poses[:, :2], axis=0)
        path_len = float(np.sqrt(np.sum(diffs * diffs, axis=1)).sum())

    fig.suptitle(
        f"{file_path.name} | idx={idx} | start=({start_pose[0]:.1f},{start_pose[1]:.1f},{math.degrees(float(start_pose[2])):.1f}deg) "
        f"| goal=({goal_pose[0]:.1f},{goal_pose[1]:.1f},{math.degrees(float(goal_pose[2])):.1f}deg) "
        f"| path_points={path_poses.shape[0]} | path_len={path_len:.2f} | active_bins={active_bins}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)

    print(f"saved: {args.output}")
    print(f"file: {file_path}")
    print(f"path_points={path_poses.shape[0]} path_len={path_len:.3f}")
    if opt_traj_orient is not None:
        print(f"orientation_bins={opt_traj_orient.shape[0]} active_bins={active_bins}")


if __name__ == "__main__":
    main()
