"""Render paper-style Neural A* figure: search overlay + guidance map.

Layout:
- Left column: Neural A* search result (history in green, path in red)
- Right column: encoder guidance map

Coordinate convention:
- world point: (x, y)
- numpy indexing: [y, x]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import MazeDataset
from neural_astar.utils.training import load_from_ptl_checkpoint, set_global_seeds


def _normalize_2d(x: np.ndarray, mask_free: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    if mask_free.any():
        lo = float(np.nanmin(y[mask_free]))
        hi = float(np.nanmax(y[mask_free]))
    else:
        lo = float(np.nanmin(y))
        hi = float(np.nanmax(y))
    if hi - lo < 1e-8:
        return np.zeros_like(y, dtype=np.float32)
    y = (y - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _pick_indices(
    n: int,
    explicit: Sequence[int] | None,
    num_rows: int,
    random_indices: bool,
    seed: int,
) -> List[int]:
    if explicit:
        out = [int(i) for i in explicit]
        for i in out:
            if i < 0 or i >= n:
                raise ValueError(f"index out of range: {i} not in [0, {n})")
        return out

    k = max(1, min(int(num_rows), n))
    if random_indices:
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.choice(n, size=k, replace=False)]
    return list(range(k))


def _to_xy(one_hot_3d: np.ndarray) -> Tuple[int, int]:
    idx = int(np.argmax(one_hot_3d))
    _, y, x = np.unravel_index(idx, one_hot_3d.shape)
    return int(x), int(y)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style Neural A* and guidance map visualization")
    p.add_argument(
        "--dataset",
        type=str,
        default="planning-datasets/data/mpd/mazes_032_moore_c8",
        help="Dataset path prefix; .npz suffix is optional.",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit sample indices")
    p.add_argument("--num-rows", type=int, default=3, help="Used when --indices is not given")
    p.add_argument("--random-indices", action="store_true")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing lightning .ckpt. Default: model/<dataset_name>",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--encoder-input", type=str, default="m+")
    p.add_argument("--encoder-arch", type=str, default="CNN")
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--g-ratio", type=float, default=0.5)
    p.add_argument(
        "--invert-guidance-display",
        action="store_true",
        help="Invert guidance map for display only.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/paper_style_neural_astar.png"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    dataset_prefix = args.dataset[:-4] if args.dataset.endswith(".npz") else args.dataset
    dataset_npz = dataset_prefix + ".npz"
    if not Path(dataset_npz).exists():
        raise FileNotFoundError(dataset_npz)

    data_name = os.path.basename(dataset_prefix)
    ckpt_dir = args.checkpoint_dir or f"model/{data_name}"
    if not Path(ckpt_dir).exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {ckpt_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    dataset = MazeDataset(dataset_npz, args.split, num_starts=1)
    indices = _pick_indices(
        n=len(dataset),
        explicit=args.indices,
        num_rows=args.num_rows,
        random_indices=args.random_indices,
        seed=args.seed,
    )

    maps: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    goals: List[np.ndarray] = []
    for idx in indices:
        map_design, start_map, goal_map, _ = dataset[idx]
        maps.append(map_design.astype(np.float32))
        starts.append(start_map.astype(np.float32))
        goals.append(goal_map.astype(np.float32))

    map_batch = torch.from_numpy(np.stack(maps, axis=0)).float().to(device)
    start_batch = torch.from_numpy(np.stack(starts, axis=0)).float().to(device)
    goal_batch = torch.from_numpy(np.stack(goals, axis=0)).float().to(device)

    planner = NeuralAstar(
        g_ratio=float(args.g_ratio),
        Tmax=1.0,
        encoder_input=args.encoder_input,
        encoder_arch=args.encoder_arch,
        encoder_depth=int(args.encoder_depth),
        learn_obstacles=False,
        use_differentiable_astar=True,
    )
    planner.load_state_dict(load_from_ptl_checkpoint(ckpt_dir, map_location="cpu"))
    planner.eval().to(device)

    with torch.no_grad():
        outputs = planner(map_batch, start_batch, goal_batch)
        guidance = planner.encode(map_batch, start_batch, goal_batch)

    histories = outputs.histories.detach().cpu().numpy()
    paths = outputs.paths.detach().cpu().numpy()
    guides = guidance.detach().cpu().numpy()
    maps_np = map_batch.detach().cpu().numpy()
    starts_np = start_batch.detach().cpu().numpy()
    goals_np = goal_batch.detach().cpu().numpy()

    rows = len(indices)
    fig, axes = plt.subplots(rows, 2, figsize=(6.0, 3.0 * rows), squeeze=False)

    for r, idx in enumerate(indices):
        ax_l = axes[r, 0]
        ax_r = axes[r, 1]

        map_2d = maps_np[r, 0]
        free_mask = map_2d > 0.5
        hist_2d = np.max(histories[r], axis=0)
        path_2d = np.max(paths[r], axis=0)

        sx, sy = _to_xy(starts_np[r])
        gx, gy = _to_xy(goals_np[r])

        # Left: Neural A* overlay.
        ax_l.imshow(map_2d, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        hist_only = np.where((hist_2d > 0.5) & (path_2d < 0.5), 1.0, np.nan)
        ax_l.imshow(hist_only, cmap="Greens", vmin=0.0, vmax=1.0, alpha=0.65, interpolation="nearest")
        path_mask = np.where(path_2d > 0.5, 1.0, np.nan)
        ax_l.imshow(path_mask, cmap="Reds", vmin=0.0, vmax=1.0, alpha=0.95, interpolation="nearest")
        ax_l.scatter([sx], [sy], c="lime", s=20, marker="o")
        ax_l.scatter([gx], [gy], c="red", s=20, marker="x")
        ax_l.text(
            0.02,
            0.98,
            f"idx={idx}",
            transform=ax_l.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 1.5, "edgecolor": "none"},
        )
        if r == 0:
            ax_l.set_title("Neural A*")
        ax_l.set_axis_off()

        # Right: guidance map.
        guide_2d = guides[r, 0]
        guide_norm = _normalize_2d(guide_2d, free_mask)
        if args.invert_guidance_display:
            guide_norm = 1.0 - guide_norm
        ax_r.imshow(guide_norm, cmap="Greens", vmin=0.0, vmax=1.0, interpolation="nearest")
        obs = np.where(free_mask, np.nan, 1.0)
        ax_r.imshow(obs, cmap="gray_r", vmin=0.0, vmax=1.0, alpha=0.32, interpolation="nearest")
        ax_r.contour((~free_mask).astype(np.float32), levels=[0.5], colors="k", linewidths=0.35, alpha=0.5)
        if r == 0:
            ax_r.set_title("Guidance map")
        ax_r.set_axis_off()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)

    print(f"saved: {args.out}")
    print(f"dataset={dataset_npz}, split={args.split}, indices={indices}")
    print(f"checkpoint_dir={ckpt_dir}, device={device}")


if __name__ == "__main__":
    main()
