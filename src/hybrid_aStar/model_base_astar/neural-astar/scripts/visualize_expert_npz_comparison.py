"""Compare stored expert paths between two planning NPZ files.

The two NPZ files must share the same maps/goals/splits and differ only in
expert labels such as opt_policy / opt_dist.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hybrid_astar_guided.grid_astar import path_length_8conn


ACTION_TO_MOVE = [
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
    (0, +1, 0),
    (0, -1, +1),
    (0, -1, -1),
    (0, +1, +1),
    (0, +1, -1),
]

XY = Tuple[int, int]
OYX = Tuple[int, int, int]


@dataclass
class CaseCompare:
    idx: int
    start_xy: XY
    goal_xy: XY
    old_path: List[XY]
    new_path: List[XY]
    old_unit_len: float
    old_geo_len: float
    new_unit_len: float
    new_geo_len: float
    change_pixels: int


def _split_base(split: str) -> int:
    return {"train": 0, "valid": 4, "test": 8}[split]


def _sample_start(
    opt_dist: np.ndarray,
    goal_map: np.ndarray,
    free_mask: np.ndarray,
    rng: np.random.Generator,
    pcts: np.ndarray,
) -> OYX:
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
        idx = int(rng.choice(candidate_idx if candidate_idx.size > 0 else valid_idx))
    else:
        idx = int(rng.choice(valid_idx))

    o, y, x = np.unravel_index(idx, opt_dist.shape)
    return int(o), int(y), int(x)


def _trace_optimal_path(
    opt_policy: np.ndarray,
    goal_map: np.ndarray,
    start_oyx: OYX,
) -> List[OYX]:
    policy = np.transpose(opt_policy, (1, 2, 3, 0))
    goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
    if len(goal_loc) != 3:
        raise ValueError(f"goal_map must contain one goal in [O,H,W], got {goal_loc}")

    cur = (int(start_oyx[0]), int(start_oyx[1]), int(start_oyx[2]))
    out: List[OYX] = [cur]
    max_steps = int(goal_map.shape[1] * goal_map.shape[2] * 6)
    for _ in range(max_steps):
        if cur == goal_loc:
            break
        action = int(np.argmax(policy[cur]))
        d_o, d_y, d_x = ACTION_TO_MOVE[action]
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
        if nxt in out:
            raise RuntimeError("Loop detected while tracing optimal policy.")
        cur = nxt
        out.append(cur)
    else:
        raise RuntimeError("Exceeded max steps while tracing optimal policy.")
    return out


def _path_oyx_to_xy(path_oyx: Sequence[OYX]) -> List[XY]:
    return [(int(x), int(y)) for _, y, x in path_oyx]


def _path_to_mask(path_xy: Sequence[XY], shape_hw: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    for x, y in path_xy:
        if 0 <= x < shape_hw[1] and 0 <= y < shape_hw[0]:
            mask[y, x] = True
    return mask


def _path_change_pixels(path_a: Sequence[XY], path_b: Sequence[XY], shape_hw: Tuple[int, int]) -> int:
    return int(np.logical_xor(_path_to_mask(path_a, shape_hw), _path_to_mask(path_b, shape_hw)).sum())


def _score_case(change_pixels: int, old_geo_len: float, new_geo_len: float) -> float:
    return float(change_pixels + 0.25 * abs(old_geo_len - new_geo_len))


def _plot_panel(
    ax: plt.Axes,
    occ_map: np.ndarray,
    path_xy: Sequence[XY],
    start_xy: XY,
    goal_xy: XY,
    title: str,
    color: str,
) -> None:
    ax.imshow(1.0 - occ_map, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if len(path_xy) > 1:
        xs = [p[0] for p in path_xy]
        ys = [p[1] for p in path_xy]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.95)
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=28, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=34, marker="x")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def _save_summary_png(cases: Sequence[CaseCompare], occ_maps: Sequence[np.ndarray], out_png: Path) -> None:
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.8, 4.0 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    for row, (case, occ_map) in enumerate(zip(cases, occ_maps)):
        titles = [
            f"Old Expert\nunit={case.old_unit_len:.1f} geo={case.old_geo_len:.1f}",
            (
                "New Expert\n"
                f"unit={case.new_unit_len:.1f} geo={case.new_geo_len:.1f}\n"
                f"delta_px={case.change_pixels}"
            ),
        ]
        paths = [case.old_path, case.new_path]
        colors = ["deepskyblue", "chartreuse"]
        for col in range(2):
            _plot_panel(
                axes[row, col],
                occ_map=occ_map,
                path_xy=paths[col],
                start_xy=case.start_xy,
                goal_xy=case.goal_xy,
                title=titles[col],
                color=colors[col],
            )
        axes[row, 0].text(
            -0.06,
            0.5,
            f"case={case.idx}",
            transform=axes[row, 0].transAxes,
            va="center",
            ha="right",
            fontsize=10,
        )

    fig.suptitle("Old Expert NPZ vs Rebuilt Expert NPZ", fontsize=14)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize path differences between two expert NPZ files.")
    p.add_argument("--old-npz", type=Path, required=True)
    p.add_argument("--new-npz", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument(
        "--sample-mode",
        type=str,
        default="ranked",
        choices=["ranked", "random", "first"],
        help="How to choose the displayed cases.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--diagonal-cost", type=float, default=math.sqrt(2.0))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/expert_npz_comparison"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.old_npz.exists():
        raise FileNotFoundError(args.old_npz)
    if not args.new_npz.exists():
        raise FileNotFoundError(args.new_npz)

    rng = np.random.default_rng(args.seed)
    base = _split_base(args.split)
    with np.load(args.old_npz) as old_data, np.load(args.new_npz) as new_data:
        old_maps = np.asarray(old_data[f"arr_{base}"], dtype=np.float32)
        new_maps = np.asarray(new_data[f"arr_{base}"], dtype=np.float32)
        old_goals = np.asarray(old_data[f"arr_{base+1}"], dtype=np.float32)
        new_goals = np.asarray(new_data[f"arr_{base+1}"], dtype=np.float32)
        old_policies = np.asarray(old_data[f"arr_{base+2}"], dtype=np.float32)
        new_policies = np.asarray(new_data[f"arr_{base+2}"], dtype=np.float32)
        old_dists = np.asarray(old_data[f"arr_{base+3}"], dtype=np.float32)
        new_dists = np.asarray(new_data[f"arr_{base+3}"], dtype=np.float32)

    if old_maps.shape != new_maps.shape or old_goals.shape != new_goals.shape:
        raise ValueError("The two NPZ files do not share the same split structure.")
    if not np.array_equal(old_maps, new_maps):
        raise ValueError("The two NPZ files have different map layouts for this split.")
    if not np.array_equal(old_goals, new_goals):
        raise ValueError("The two NPZ files have different goal maps for this split.")

    eval_count = min(int(args.max_samples), int(old_maps.shape[0]))
    ranked: List[Tuple[float, CaseCompare, np.ndarray]] = []

    for idx in range(eval_count):
        map_design = old_maps[idx]
        free_mask = map_design > 0.5
        occ_map = (1.0 - map_design).astype(np.float32)
        goal_map = old_goals[idx]
        old_policy = old_policies[idx]
        new_policy = new_policies[idx]
        old_dist = old_dists[idx]

        try:
            start_oyx = _sample_start(
                opt_dist=old_dist,
                goal_map=goal_map,
                free_mask=free_mask,
                rng=rng,
                pcts=np.array([0.55, 0.70, 0.85, 1.0], dtype=np.float32),
            )
            old_path_oyx = _trace_optimal_path(old_policy, goal_map, start_oyx)
            new_path_oyx = _trace_optimal_path(new_policy, goal_map, start_oyx)
        except RuntimeError:
            continue

        goal_oyx = tuple(np.array(np.nonzero(goal_map)).squeeze())
        start_xy = (int(start_oyx[2]), int(start_oyx[1]))
        goal_xy = (int(goal_oyx[2]), int(goal_oyx[1]))
        old_path = _path_oyx_to_xy(old_path_oyx)
        new_path = _path_oyx_to_xy(new_path_oyx)
        change_pixels = _path_change_pixels(old_path, new_path, occ_map.shape)
        old_unit = float(path_length_8conn(old_path, diagonal_cost=1.0))
        new_unit = float(path_length_8conn(new_path, diagonal_cost=1.0))
        old_geo = float(path_length_8conn(old_path, diagonal_cost=float(args.diagonal_cost)))
        new_geo = float(path_length_8conn(new_path, diagonal_cost=float(args.diagonal_cost)))
        case = CaseCompare(
            idx=int(idx),
            start_xy=start_xy,
            goal_xy=goal_xy,
            old_path=old_path,
            new_path=new_path,
            old_unit_len=old_unit,
            old_geo_len=old_geo,
            new_unit_len=new_unit,
            new_geo_len=new_geo,
            change_pixels=int(change_pixels),
        )
        ranked.append((_score_case(change_pixels, old_geo, new_geo), case, occ_map))

    if not ranked:
        raise RuntimeError("No valid comparison cases found.")

    if args.sample_mode == "ranked":
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[: max(1, int(args.top_k))]
    elif args.sample_mode == "first":
        selected = ranked[: max(1, int(args.top_k))]
    else:
        choose_n = min(max(1, int(args.top_k)), len(ranked))
        pick_ids = rng.choice(len(ranked), size=choose_n, replace=False)
        selected = [ranked[int(i)] for i in np.atleast_1d(pick_ids).tolist()]
        selected.sort(key=lambda item: int(item[1].idx))

    cases = [item[1] for item in selected]
    occ_maps = [item[2] for item in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"expert_npz_comparison_{args.split}.png"
    csv_path = args.output_dir / f"expert_npz_comparison_{args.split}.csv"
    _save_summary_png(cases, occ_maps, png_path)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "case_idx,start_x,start_y,goal_x,goal_y,"
            "old_unit_len,old_geo_len,new_unit_len,new_geo_len,change_pixels\n"
        )
        for case in cases:
            f.write(
                f"{case.idx},{case.start_xy[0]},{case.start_xy[1]},{case.goal_xy[0]},{case.goal_xy[1]},"
                f"{case.old_unit_len:.4f},{case.old_geo_len:.4f},"
                f"{case.new_unit_len:.4f},{case.new_geo_len:.4f},"
                f"{case.change_pixels}\n"
            )

    print(f"saved_png={png_path}")
    print(f"saved_csv={csv_path}")
    print(f"cases={len(cases)}")


if __name__ == "__main__":
    main()
