"""Visualize predicted guidance cost map.

Color semantics in this script:
- Lower cost (easier to traverse): darker
- Higher cost (harder to traverse): lighter

Coordinate convention:
- world point is (x, y)
- numpy indexing is [y, x]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from neural_astar.api.guidance_infer import infer_cost_map


def _select_2d(arr: np.ndarray, index: int, key: str) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim == 3:
        if x.shape[0] == 1:
            return x[0].astype(np.float32)
        if index < 0 or index >= x.shape[0]:
            raise ValueError(f"index out of range for {key}: {index}")
        return x[index].astype(np.float32)
    if x.ndim == 4:
        if x.shape[1] == 1:
            if index < 0 or index >= x.shape[0]:
                raise ValueError(f"index out of range for {key}: {index}")
            return x[index, 0].astype(np.float32)
        if x.shape[-1] == 1:
            if index < 0 or index >= x.shape[0]:
                raise ValueError(f"index out of range for {key}: {index}")
            return x[index, :, :, 0].astype(np.float32)
    raise ValueError(f"Cannot convert key={key} shape={x.shape} to 2D")


def _argmax_xy(one_hot_2d: np.ndarray) -> Tuple[int, int]:
    idx = int(np.argmax(one_hot_2d))
    y, x = np.unravel_index(idx, one_hot_2d.shape)
    return int(x), int(y)


def _load_occ_and_goal_hint(
    npz_path: Path,
    index: int,
    occ_semantics: str,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]], str]:
    with np.load(npz_path) as data:
        keys = set(data.files)

        occ_key = None
        for k in ("occ_map", "occupancy_map", "occupancy", "map", "arr_0"):
            if k in keys:
                occ_key = k
                break
        if occ_key is None:
            raise ValueError(f"No occupancy key found in {npz_path}")

        occ_raw = _select_2d(data[occ_key], index=index, key=occ_key)
        mode = occ_semantics
        if mode == "auto":
            mode = "passable1" if occ_key == "arr_0" else "obstacle1"
        if mode == "obstacle1":
            occ = occ_raw
        elif mode == "passable1":
            occ = 1.0 - occ_raw
        else:
            raise ValueError(f"Unknown occ semantics: {mode}")

        goal_xy = None
        if "goal_map" in keys:
            gm = _select_2d(data["goal_map"], index=index, key="goal_map")
            goal_xy = _argmax_xy(gm)
        elif "arr_1" in keys:
            gm = _select_2d(data["arr_1"], index=index, key="arr_1")
            goal_xy = _argmax_xy(gm)

    return occ.astype(np.float32), goal_xy, mode


def _random_free_xy(occ: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    h, w = occ.shape
    for _ in range(10000):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if occ[y, x] < 0.5:
            return x, y
    raise RuntimeError("Failed to sample a free cell")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize neural guidance cost map")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--occ-npz", type=Path, required=True)
    p.add_argument("--index", type=int, default=0, help="Map index in npz")
    p.add_argument(
        "--occ-semantics",
        type=str,
        default="auto",
        choices=["auto", "obstacle1", "passable1"],
        help="Map semantics. obstacle1: 1=obstacle, passable1: 1=free.",
    )
    p.add_argument("--start", type=int, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--goal", type=int, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--start-yaw-deg", type=float, default=0.0)
    p.add_argument("--goal-yaw-deg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--invert-guidance-cost", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/guidance_cost_map.png"),
        help="Output png path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    occ, goal_hint, resolved_sem = _load_occ_and_goal_hint(
        npz_path=args.occ_npz,
        index=args.index,
        occ_semantics=args.occ_semantics,
    )
    h, w = occ.shape
    rng = np.random.default_rng(args.seed)

    if args.start is not None:
        start_xy = (int(args.start[0]), int(args.start[1]))
    else:
        start_xy = _random_free_xy(occ, rng)

    if args.goal is not None:
        goal_xy = (int(args.goal[0]), int(args.goal[1]))
    elif goal_hint is not None:
        goal_xy = goal_hint
    else:
        goal_xy = _random_free_xy(occ, rng)
        while goal_xy == start_xy:
            goal_xy = _random_free_xy(occ, rng)

    for name, (x, y) in (("start", start_xy), ("goal", goal_xy)):
        if x < 0 or x >= w or y < 0 or y >= h:
            raise ValueError(f"{name} out of bounds: {(x, y)} for map {(h, w)}")
        if occ[y, x] > 0.5:
            raise ValueError(f"{name} on obstacle: {(x, y)}")

    cost = infer_cost_map(
        ckpt_path=args.ckpt,
        occ_map_numpy=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_yaw=math.radians(float(args.start_yaw_deg)),
        goal_yaw=math.radians(float(args.goal_yaw_deg)),
        device=args.device,
        invert_guidance_cost=args.invert_guidance_cost,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].scatter([start_xy[0]], [start_xy[1]], c="lime", s=60, marker="o", label="start")
    axes[0].scatter([goal_xy[0]], [goal_xy[1]], c="red", s=60, marker="x", label="goal")
    axes[0].set_title("Occupancy (white=free, black=obstacle)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_axis_off()

    # grayscale: dark=low cost(easy), light=high cost(hard)
    im = axes[1].imshow(cost, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].scatter([start_xy[0]], [start_xy[1]], c="lime", s=50, marker="o")
    axes[1].scatter([goal_xy[0]], [goal_xy[1]], c="red", s=50, marker="x")
    axes[1].set_title("Guidance Cost (dark=easier, light=harder)")
    axes[1].set_axis_off()
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("cost (0 low -> 1 high)")

    fig.suptitle(
        f"map_index={args.index}, semantics={resolved_sem}, "
        f"start={start_xy}, goal={goal_xy}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)

    print(f"saved: {args.out}")
    print(f"cost_range: min={cost.min():.4f}, max={cost.max():.4f}")
    print(f"start={start_xy}, goal={goal_xy}, map_shape={occ.shape}")


if __name__ == "__main__":
    main()
