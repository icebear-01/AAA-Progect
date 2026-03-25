"""Visualize expert shortest path, optimal action field, and distance field.

Supported dataset format: planning-datasets .npz (arr_0..arr_11).

Coordinate convention:
- world point: (x, y)
- numpy indexing: [y, x]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hybrid_astar_guided.grid_astar import astar_8conn, path_length_8conn


ACTION_TO_MOVE = [
    (0, -1, 0),   # up
    (0, 0, +1),   # right
    (0, 0, -1),   # left
    (0, +1, 0),   # down
    (0, -1, +1),  # up-right
    (0, -1, -1),  # up-left
    (0, +1, +1),  # down-right
    (0, +1, -1),  # down-left
]


def _split_base(split: str) -> int:
    return {"train": 0, "valid": 4, "test": 8}[split]


def _sample_start(
    opt_dist: np.ndarray,
    goal_map: np.ndarray,
    free_mask: np.ndarray,
    rng: np.random.Generator,
    pcts: np.ndarray,
) -> Tuple[int, int, int]:
    od_vct = opt_dist.flatten()
    min_v = float(od_vct.min())
    od_vals = od_vct[od_vct > min_v]

    free_3d = np.repeat(free_mask[None, ...], opt_dist.shape[0], axis=0)
    goal_3d = goal_map > 0.5
    valid_mask = free_3d & (~goal_3d)
    valid_idx = np.where(valid_mask.flatten())[0]
    if valid_idx.size == 0:
        raise RuntimeError("No valid start cells available.")

    if od_vals.size > 0:
        od_th = np.percentile(od_vals, 100.0 * (1.0 - pcts))
        r = int(rng.integers(0, len(od_th) - 1))
        candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
        candidate_idx = np.where(candidate & valid_mask.flatten())[0]
        if candidate_idx.size > 0:
            idx = int(rng.choice(candidate_idx))
        else:
            idx = int(rng.choice(valid_idx))
    else:
        idx = int(rng.choice(valid_idx))

    o, y, x = np.unravel_index(idx, opt_dist.shape)
    return int(o), int(y), int(x)


def _trace_optimal_path(
    opt_policy: np.ndarray,  # [A,O,H,W]
    goal_map: np.ndarray,    # [O,H,W]
    start_oyx: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    policy = np.transpose(opt_policy, (1, 2, 3, 0))  # [O,H,W,A]
    goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
    if len(goal_loc) != 3:
        raise ValueError(f"goal_map must contain one goal in [O,H,W], got goal_loc={goal_loc}")

    cur = (int(start_oyx[0]), int(start_oyx[1]), int(start_oyx[2]))
    path_oyx: List[Tuple[int, int, int]] = [cur]

    max_steps = int(goal_map.shape[1] * goal_map.shape[2] * 6)
    for _ in range(max_steps):
        if cur == goal_loc:
            break
        one_hot_action = policy[cur]
        a = int(np.argmax(one_hot_action))
        d_o, d_y, d_x = ACTION_TO_MOVE[a]
        nxt = (cur[0] + d_o, cur[1] + d_y, cur[2] + d_x)
        if (
            nxt[0] < 0
            or nxt[0] >= goal_map.shape[0]
            or nxt[1] < 0
            or nxt[1] >= goal_map.shape[1]
            or nxt[2] < 0
            or nxt[2] >= goal_map.shape[2]
        ):
            raise RuntimeError("Optimal policy stepped out of bounds.")
        if nxt in path_oyx:
            raise RuntimeError("Loop detected while tracing optimal policy.")
        cur = nxt
        path_oyx.append(cur)
    else:
        raise RuntimeError("Exceeded max steps while tracing optimal policy.")

    return path_oyx


def _path_oyx_to_mask(path_oyx: List[Tuple[int, int, int]], shape_oyx: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape_oyx, dtype=np.float32)
    for o, y, x in path_oyx:
        out[o, y, x] = 1.0
    return out


def _path_oyx_to_xy(path_oyx: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
    return [(int(x), int(y)) for _, y, x in path_oyx]


def _dist_for_plot(dist_2d: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    d = np.asarray(dist_2d, dtype=np.float32).copy()
    # Many planning-datasets store non-positive distance values.
    if float(np.nanmax(d)) <= 0.0:
        d = -d
    d[~free_mask] = np.nan
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize expert supervision from planning-datasets npz")
    p.add_argument("--dataset", type=Path, required=True, help="Path to planning-datasets .npz")
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--index", type=int, default=0, help="Sample index within split")
    p.add_argument("--start", type=int, nargs=2, default=None, metavar=("X", "Y"))
    p.add_argument("--start-orient", type=int, default=0, help="Orientation index for --start")
    p.add_argument("--policy-orient", type=int, default=0, help="Orientation slice for action-field arrows")
    p.add_argument("--arrow-step", type=int, default=3, help="Subsample step for quiver arrows")
    p.add_argument(
        "--astar-diagonal-cost",
        type=float,
        default=math.sqrt(2.0),
        help="Reference A* diagonal move cost. 1.0=unit-step shortest; sqrt(2)=geometric shortest.",
    )
    p.add_argument(
        "--astar-no-corner-cut",
        action="store_true",
        help="If set, reference A* forbids diagonal corner cutting.",
    )
    p.add_argument(
        "--no-overlay-astar-reference",
        action="store_true",
        help="Disable overlay of the 2D reference A* shortest path.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: outputs/expert_supervision_<name>_<split>_idx<k>.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    rng = np.random.default_rng(args.seed)
    base = _split_base(args.split)

    with np.load(args.dataset) as data:
        map_designs = np.asarray(data[f"arr_{base}"], dtype=np.float32)      # [N,H,W], 1=free
        goal_maps = np.asarray(data[f"arr_{base+1}"], dtype=np.float32)      # [N,O,H,W]
        opt_policies = np.asarray(data[f"arr_{base+2}"], dtype=np.float32)   # [N,A,O,H,W]
        opt_dists = np.asarray(data[f"arr_{base+3}"], dtype=np.float32)      # [N,O,H,W]

    n = int(map_designs.shape[0])
    if args.index < 0 or args.index >= n:
        raise ValueError(f"index out of range: {args.index} not in [0, {n})")

    map_design = map_designs[args.index]           # [H,W], 1=free
    free_mask = map_design > 0.5
    goal_map = goal_maps[args.index]               # [O,H,W]
    opt_policy = opt_policies[args.index]          # [A,O,H,W]
    opt_dist = opt_dists[args.index]               # [O,H,W]

    o_dim, h, w = goal_map.shape
    if args.policy_orient < 0 or args.policy_orient >= o_dim:
        raise ValueError(f"policy-orient out of range: {args.policy_orient} not in [0, {o_dim})")

    if args.start is not None:
        sx, sy = int(args.start[0]), int(args.start[1])
        so = int(args.start_orient)
        if so < 0 or so >= o_dim:
            raise ValueError(f"start-orient out of range: {so} not in [0, {o_dim})")
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            raise ValueError(f"start out of bounds: {(sx, sy)} for map {(h, w)}")
        if not free_mask[sy, sx]:
            raise ValueError(f"start on obstacle: {(sx, sy)}")
        start_oyx = (so, sy, sx)
    else:
        start_oyx = _sample_start(
            opt_dist=opt_dist,
            goal_map=goal_map,
            free_mask=free_mask,
            rng=rng,
            pcts=np.array([0.55, 0.70, 0.85, 1.0], dtype=np.float32),
        )

    path_oyx = _trace_optimal_path(opt_policy, goal_map, start_oyx)
    path_3d = _path_oyx_to_mask(path_oyx, goal_map.shape)
    path_2d = (path_3d.max(axis=0) > 0.5).astype(np.float32)
    expert_path_xy = _path_oyx_to_xy(path_oyx)

    goal_oyx = tuple(np.array(np.nonzero(goal_map)).squeeze())
    gx, gy = int(goal_oyx[2]), int(goal_oyx[1])
    sx, sy = int(start_oyx[2]), int(start_oyx[1])

    occ_map = (~free_mask).astype(np.float32)
    ref_path_xy = astar_8conn(
        occ_map=occ_map,
        start_xy=(sx, sy),
        goal_xy=(gx, gy),
        diagonal_cost=float(args.astar_diagonal_cost),
        allow_corner_cut=not bool(args.astar_no_corner_cut),
    )

    # Action field for one orientation slice.
    action_idx = np.argmax(opt_policy[:, args.policy_orient], axis=0)  # [H,W]
    dy_lut = np.array([m[1] for m in ACTION_TO_MOVE], dtype=np.float32)
    dx_lut = np.array([m[2] for m in ACTION_TO_MOVE], dtype=np.float32)
    ay = dy_lut[action_idx]
    ax = dx_lut[action_idx]
    ay[~free_mask] = 0.0
    ax[~free_mask] = 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    step = max(1, int(args.arrow_step))
    sel = (yy % step == 0) & (xx % step == 0) & free_mask

    dist_vis = _dist_for_plot(opt_dist[args.policy_orient], free_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: map + expert path (+ optional reference shortest path).
    axes[0].imshow(free_mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    py, px = np.where(path_2d > 0.5)
    axes[0].scatter(px, py, c="deepskyblue", s=8, marker=".", label="expert path")
    if (not args.no_overlay_astar_reference) and (ref_path_xy is not None):
        rx = [int(p[0]) for p in ref_path_xy]
        ry = [int(p[1]) for p in ref_path_xy]
        axes[0].plot(rx, ry, color="magenta", linewidth=1.4, alpha=0.95, label="reference A*")
    axes[0].scatter([sx], [sy], c="lime", s=55, marker="o", label="start")
    axes[0].scatter([gx], [gy], c="red", s=55, marker="x", label="goal")
    axes[0].set_title("Map + Expert Path (white=free, black=obstacle)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_axis_off()

    # Panel 2: optimal action field.
    axes[1].imshow(free_mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].quiver(
        xx[sel],
        yy[sel],
        ax[sel],
        ay[sel],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="tab:orange",
        width=0.0025,
    )
    axes[1].scatter([sx], [sy], c="lime", s=45, marker="o")
    axes[1].scatter([gx], [gy], c="red", s=45, marker="x")
    axes[1].set_title(f"Optimal Action Field (orient={args.policy_orient})")
    axes[1].set_axis_off()

    # Panel 3: distance field.
    im = axes[2].imshow(dist_vis, cmap="viridis")
    axes[2].scatter([sx], [sy], c="lime", s=45, marker="o")
    axes[2].scatter([gx], [gy], c="red", s=45, marker="x")
    axes[2].set_title("Optimal Distance Field")
    axes[2].set_axis_off()
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("distance-to-goal")

    fig.suptitle(
        f"{args.dataset.name} | split={args.split} | idx={args.index} | "
        f"start=({sx},{sy},o={start_oyx[0]}) goal=({gx},{gy},o={goal_oyx[0]})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = (
        args.output
        if args.output is not None
        else Path("outputs")
        / f"expert_supervision_{args.dataset.stem}_{args.split}_idx{args.index:04d}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)

    expert_steps = len(expert_path_xy) - 1
    expert_step_len = path_length_8conn(expert_path_xy, diagonal_cost=1.0)
    expert_geom_len = path_length_8conn(expert_path_xy, diagonal_cost=math.sqrt(2.0))
    ref_len = None
    if ref_path_xy is not None:
        ref_len = path_length_8conn(ref_path_xy, diagonal_cost=float(args.astar_diagonal_cost))

    print(f"saved: {out}")
    print(
        f"sample={args.index}, start=(x={sx}, y={sy}, o={start_oyx[0]}), "
        f"goal=(x={gx}, y={gy}, o={goal_oyx[0]}), expert_steps={expert_steps}, "
        f"expert_step_len={expert_step_len:.3f}, expert_geom_len={expert_geom_len:.3f}"
    )
    if ref_len is None:
        print("reference_astar: no path")
    else:
        print(
            "reference_astar: "
            f"diag_cost={float(args.astar_diagonal_cost):.6f}, "
            f"allow_corner_cut={not bool(args.astar_no_corner_cut)}, "
            f"path_len={ref_len:.3f}, path_steps={len(ref_path_xy) - 1}"
        )


if __name__ == "__main__":
    main()
