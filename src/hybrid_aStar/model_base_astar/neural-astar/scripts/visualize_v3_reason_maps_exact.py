from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)
try:
    from scripts.rebuild_planning_npz_experts import (
        _build_policy_from_cost,
        _reverse_dijkstra_cost_to_goal,
    )
except ModuleNotFoundError:
    from rebuild_planning_npz_experts import (  # type: ignore[no-redef]
        _build_policy_from_cost,
        _reverse_dijkstra_cost_to_goal,
    )


XY = Tuple[int, int]

SCENE_COLORS = {
    "expert": "#22d3ee",
    "astar": "#56B4E9",
    "improved": "#009E73",
    "v3": "#D55E00",
    "start": "#22c55e",
    "goal": "#ef4444",
}


@dataclass
class PlannerView:
    label: str
    stats: Astar8ConnStats
    runtime_ms: float

    @property
    def path_length(self) -> float:
        if self.stats.path is None:
            return 0.0
        return float(path_length_8conn(self.stats.path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize exact confidence/value maps for selected V3 regression cases.")
    p.add_argument("--data-npz", type=Path, required=True)
    p.add_argument("--expert-npz", type=Path, required=True)
    p.add_argument("--expert-rebuild-on-the-fly", action="store_true")
    p.add_argument("--expert-clearance-weight", type=float, default=0.35)
    p.add_argument("--expert-clearance-safe-distance", type=float, default=5.0)
    p.add_argument("--expert-clearance-power", type=float, default=2.0)
    p.add_argument(
        "--expert-policy-tie-break",
        type=str,
        default="clearance_then_first",
        choices=["first", "random", "clearance", "clearance_then_first"],
    )
    p.add_argument("--expert-seed", type=int, default=1234)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--case-csv", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--case-indices", type=int, nargs="+", required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--residual-weight", type=float, default=1.25)
    p.add_argument(
        "--residual-confidence-mode",
        type=str,
        default="learned_spike",
        choices=["none", "spike_suppression", "learned", "learned_spike"],
    )
    p.add_argument("--residual-confidence-kernel", type=int, default=3)
    p.add_argument("--residual-confidence-strength", type=float, default=0.75)
    p.add_argument("--residual-confidence-min", type=float, default=0.1)
    p.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    p.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    p.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(allow_corner_cut=True)
    return p.parse_args()


def _load_rows(path: Path) -> Dict[int, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["idx"]): r for r in rows}


def _load_split_arrays(npz_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_base = {"train": 0, "valid": 4, "test": 8}[split]
    with np.load(npz_path) as data:
        maps = np.asarray(data[f"arr_{idx_base}"], dtype=np.float32)
        goals = np.asarray(data[f"arr_{idx_base+1}"], dtype=np.float32)
        policies = np.asarray(data[f"arr_{idx_base+2}"], dtype=np.float32)
    return maps, goals, policies


def _one_hot_xy(width: int, height: int, x: int, y: int) -> np.ndarray:
    arr = np.zeros((1, height, width), dtype=np.float32)
    arr[0, int(y), int(x)] = 1.0
    return arr


def _follow_opt_policy(start_map: np.ndarray, goal_map: np.ndarray, opt_policy: np.ndarray) -> Tuple[np.ndarray, List[XY]]:
    action_to_move = [
        (0, -1, 0),
        (0, 0, +1),
        (0, 0, -1),
        (0, +1, 0),
        (0, -1, +1),
        (0, -1, -1),
        (0, +1, +1),
        (0, +1, -1),
    ]
    opt_traj = np.zeros_like(start_map, dtype=np.float32)
    policy = np.transpose(opt_policy, (1, 2, 3, 0))  # [O,H,W,A]
    current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
    goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
    max_steps = int(start_map.shape[-1] * start_map.shape[-2] * 4)
    steps = 0
    ordered_path: List[XY] = []
    while goal_loc != current_loc:
        opt_traj[current_loc] = 1.0
        ordered_path.append((int(current_loc[2]), int(current_loc[1])))
        move = action_to_move[int(np.argmax(policy[current_loc]))]
        next_loc = (
            int(current_loc[0] + move[0]),
            int(current_loc[1] + move[1]),
            int(current_loc[2] + move[2]),
        )
        if (
            next_loc[0] < 0
            or next_loc[0] >= opt_traj.shape[0]
            or next_loc[1] < 0
            or next_loc[1] >= opt_traj.shape[1]
            or next_loc[2] < 0
            or next_loc[2] >= opt_traj.shape[2]
        ):
            break
        if opt_traj[next_loc] > 0.5:
            break
        current_loc = next_loc
        steps += 1
        if steps > max_steps:
            break
    ordered_path.append((int(goal_loc[2]), int(goal_loc[1])))
    return opt_traj.astype(np.float32), ordered_path


def _infer_residual_prediction_2d(model: torch.nn.Module, sample: Dict[str, torch.Tensor], device: str) -> Tuple[np.ndarray, np.ndarray | None]:
    occ = sample["occ_map"].unsqueeze(0).to(device)
    start = sample["start_map"].unsqueeze(0).to(device)
    goal = sample["goal_map"].unsqueeze(0).to(device)
    extra_input_maps = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        clearance_input = sample.get("clearance_input_map")
        if clearance_input is not None:
            extra_input_maps = clearance_input.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(
            occ,
            start,
            goal,
            start_yaw=torch.zeros(1, device=device, dtype=occ.dtype),
            goal_yaw=torch.zeros(1, device=device, dtype=occ.dtype),
            extra_input_maps=extra_input_maps,
        )
    pred = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    scale = None
    if out.scale_map is not None:
        scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32)
    confidence = None
    if out.confidence_map is not None:
        confidence = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
        confidence = confidence[0] if confidence.shape[0] == 1 else np.min(confidence, axis=0).astype(np.float32)
    pred = pred[0] if pred.shape[0] == 1 else np.min(pred, axis=0).astype(np.float32)
    pred = decode_residual_prediction_np(pred, transform=str(getattr(model, "residual_target_transform", "none")))
    if scale is not None:
        scale = scale[0] if scale.shape[0] == 1 else np.min(scale, axis=0).astype(np.float32)
    pred = apply_residual_scale_np(pred, scale)
    return np.maximum(pred, 0.0).astype(np.float32), confidence


def _run_astar(
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    *,
    label: str,
    heuristic_mode: str,
    heuristic_weight: float = 1.0,
    heuristic_residual_map: np.ndarray | None = None,
    residual_confidence_map: np.ndarray | None = None,
    residual_weight: float = 0.0,
    diagonal_cost: float,
    allow_corner_cut: bool,
) -> PlannerView:
    t0 = time.perf_counter()
    stats = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode=heuristic_mode,
        heuristic_weight=float(heuristic_weight),
        heuristic_residual_map=heuristic_residual_map,
        residual_confidence_map=residual_confidence_map,
        residual_weight=float(residual_weight),
        diagonal_cost=float(diagonal_cost),
        allow_corner_cut=bool(allow_corner_cut),
    )
    return PlannerView(label=label, stats=stats, runtime_ms=1000.0 * (time.perf_counter() - t0))


def _expanded_heatmap(stats: Astar8ConnStats, h: int, w: int) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    for x, y in stats.expanded_xy:
        if 0 <= x < w and 0 <= y < h:
            heat[y, x] += 1.0
    if float(heat.max()) > 0.0:
        heat /= float(heat.max())
    return heat


def _base_scene(
    ax: plt.Axes,
    occ: np.ndarray,
    expert_traj: np.ndarray,
    expert_path_xy: List[XY],
    start_xy: XY,
    goal_xy: XY,
) -> None:
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if expert_traj.ndim == 3:
        expert_2d = expert_traj[0]
    else:
        expert_2d = expert_traj
    ax.imshow(expert_2d, cmap="Blues", alpha=0.12, vmin=0.0, vmax=1.0, interpolation="nearest")
    if len(expert_path_xy) > 1:
        xs = [p[0] for p in expert_path_xy]
        ys = [p[1] for p in expert_path_xy]
        ax.plot(xs, ys, color=SCENE_COLORS["expert"], linewidth=2.0, alpha=0.95, linestyle="--")
    ax.scatter([start_xy[0]], [start_xy[1]], c=SCENE_COLORS["start"], s=48, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=SCENE_COLORS["goal"], s=56, marker="x")
    ax.set_axis_off()


def _plot_search_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    expert_traj: np.ndarray,
    expert_path_xy: List[XY],
    start_xy: XY,
    goal_xy: XY,
    view: PlannerView,
    color: str,
) -> None:
    _base_scene(ax, occ, expert_traj, expert_path_xy, start_xy, goal_xy)
    heat = _expanded_heatmap(view.stats, occ.shape[0], occ.shape[1])
    if float(heat.max()) > 0.0:
        ax.imshow(heat, cmap="magma", alpha=0.58, vmin=0.0, vmax=1.0, interpolation="nearest")
    if view.stats.path is not None and len(view.stats.path) > 1:
        xs = [p[0] for p in view.stats.path]
        ys = [p[1] for p in view.stats.path]
        ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.98)
    ax.set_title(
        f"{view.label}\nexpanded={view.stats.expanded_nodes}  path={view.path_length:.1f}\ntime={view.runtime_ms:.2f} ms",
        fontsize=10,
        pad=8,
    )


def _plot_map_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    expert_traj: np.ndarray,
    expert_path_xy: List[XY],
    start_xy: XY,
    goal_xy: XY,
    data: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    v3_path: List[XY] | None,
) -> None:
    _base_scene(ax, occ, expert_traj, expert_path_xy, start_xy, goal_xy)
    masked = np.ma.masked_where(occ > 0.5, data)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.90, interpolation="nearest")
    if v3_path is not None and len(v3_path) > 1:
        xs = [p[0] for p in v3_path]
        ys = [p[1] for p in v3_path]
        ax.plot(xs, ys, color=SCENE_COLORS["v3"], linewidth=1.9, alpha=0.96)
    ax.set_title(title, fontsize=10, pad=8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)


def _save_case_figure(
    idx: int,
    occ: np.ndarray,
    expert_traj: np.ndarray,
    expert_path_xy: List[XY],
    start_xy: XY,
    goal_xy: XY,
    astar_view: PlannerView,
    improved_view: PlannerView,
    v3_view: PlannerView,
    residual_map: np.ndarray,
    resolved_conf: np.ndarray,
    effective_map: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))

    _plot_search_panel(axes[0, 0], occ, expert_traj, expert_path_xy, start_xy, goal_xy, improved_view, SCENE_COLORS["improved"])
    _plot_search_panel(axes[0, 1], occ, expert_traj, expert_path_xy, start_xy, goal_xy, v3_view, SCENE_COLORS["v3"])
    _base_scene(axes[0, 2], occ, expert_traj, expert_path_xy, start_xy, goal_xy)
    axes[0, 2].set_title("Improved Expert Path", fontsize=10, pad=8)

    pos_res = residual_map[residual_map > 0.0]
    pos_eff = effective_map[effective_map > 0.0]
    res_vmax = float(np.percentile(pos_res, 98)) if pos_res.size > 0 else 1.0
    eff_vmax = float(np.percentile(pos_eff, 98)) if pos_eff.size > 0 else 1.0

    v3_path = v3_view.stats.path
    _plot_map_panel(
        axes[1, 0],
        occ,
        expert_traj,
        expert_path_xy,
        start_xy,
        goal_xy,
        residual_map,
        "Predicted Residual Value Map",
        "cividis",
        0.0,
        max(res_vmax, 1e-6),
        v3_path,
    )
    _plot_map_panel(
        axes[1, 1],
        occ,
        expert_traj,
        expert_path_xy,
        start_xy,
        goal_xy,
        resolved_conf,
        "Planner Confidence Map",
        "viridis",
        0.0,
        1.0,
        v3_path,
    )
    _plot_map_panel(
        axes[1, 2],
        occ,
        expert_traj,
        expert_path_xy,
        start_xy,
        goal_xy,
        effective_map,
        "Effective Guidance Fed to A*",
        "magma",
        0.0,
        max(eff_vmax, 1e-6),
        v3_path,
    )

    fig.suptitle(
        f"Exact Regression Case idx={idx} | V3-A*={v3_view.stats.expanded_nodes - astar_view.stats.expanded_nodes:+d} | V3-Improved={v3_view.stats.expanded_nodes - improved_view.stats.expanded_nodes:+d}",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    case_rows = _load_rows(args.case_csv)
    maps, goals, _ = _load_split_arrays(args.data_npz, args.split)
    expert_goals = None
    expert_policies = None
    if not bool(args.expert_rebuild_on_the_fly):
        _, expert_goals, expert_policies = _load_split_arrays(args.expert_npz, args.split)
    model = load_guidance_encoder(args.ckpt, device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[List[object]] = []
    for idx in args.case_indices:
        row = case_rows[int(idx)]
        sx, sy = int(row["start_x"]), int(row["start_y"])
        gx, gy = int(row["goal_x"]), int(row["goal_y"])
        occ = (1.0 - maps[int(idx)]).astype(np.float32)
        h, w = occ.shape
        goal_dataset_xy = np.unravel_index(int(np.argmax(goals[int(idx), 0])), goals[int(idx), 0].shape)[::-1]
        if bool(args.expert_rebuild_on_the_fly):
            if (gx, gy) != (int(goal_dataset_xy[0]), int(goal_dataset_xy[1])):
                raise ValueError(f"goal mismatch for idx={idx}")
        else:
            assert expert_goals is not None and expert_policies is not None
            expert_goal_xy = np.unravel_index(
                int(np.argmax(expert_goals[int(idx), 0])),
                expert_goals[int(idx), 0].shape,
            )[::-1]
            if (gx, gy) != (int(goal_dataset_xy[0]), int(goal_dataset_xy[1])) or (gx, gy) != (
                int(expert_goal_xy[0]),
                int(expert_goal_xy[1]),
            ):
                raise ValueError(f"goal mismatch for idx={idx}")

        start_map = _one_hot_xy(w, h, sx, sy)
        goal_map = _one_hot_xy(w, h, gx, gy)
        if bool(args.expert_rebuild_on_the_fly):
            expert_cost_to_goal = _reverse_dijkstra_cost_to_goal(
                occ_map=occ,
                goal_xy=(gx, gy),
                mechanism="moore",
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.expert_clearance_weight),
                clearance_safe_distance=float(args.expert_clearance_safe_distance),
                clearance_power=float(args.expert_clearance_power),
            )
            expert_policy = _build_policy_from_cost(
                occ_map=occ,
                goal_xy=(gx, gy),
                cost_to_goal=expert_cost_to_goal,
                mechanism="moore",
                diagonal_cost=float(args.diagonal_cost),
                allow_corner_cut=bool(args.allow_corner_cut),
                clearance_weight=float(args.expert_clearance_weight),
                clearance_safe_distance=float(args.expert_clearance_safe_distance),
                clearance_power=float(args.expert_clearance_power),
                policy_tie_break=str(args.expert_policy_tie_break),
                rng=np.random.default_rng(int(args.expert_seed) + int(idx)),
            )
            expert_traj, expert_path_xy = _follow_opt_policy(start_map, goal_map, expert_policy)
        else:
            assert expert_policies is not None
            expert_traj, expert_path_xy = _follow_opt_policy(start_map, goal_map, expert_policies[int(idx)])

        sample = {
            "occ_map": torch.from_numpy(occ[None, ...].astype(np.float32)),
            "start_map": torch.from_numpy(start_map.astype(np.float32)),
            "goal_map": torch.from_numpy(goal_map.astype(np.float32)),
            "clearance_input_map": torch.from_numpy(
                build_clearance_input_map(
                    occ_map=occ,
                    clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
                )[None, ...].astype(np.float32)
            ),
        }
        residual_map, learned_confidence_map = _infer_residual_prediction_2d(model, sample, device=device)
        if args.residual_confidence_mode == "none":
            resolved_conf = np.ones_like(residual_map, dtype=np.float32)
        else:
            resolved_conf = resolve_residual_confidence_map(
                mode=args.residual_confidence_mode,
                occ_map=occ,
                residual_map=residual_map,
                learned_confidence_map=learned_confidence_map,
                kernel_size=args.residual_confidence_kernel,
                strength=args.residual_confidence_strength,
                min_confidence=args.residual_confidence_min,
            )
        effective_map = float(args.residual_weight) * residual_map * resolved_conf

        astar_view = _run_astar(
            occ, (sx, sy), (gx, gy),
            label="A*",
            heuristic_mode="euclidean",
            heuristic_weight=1.0,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
        )
        improved_view = _run_astar(
            occ, (sx, sy), (gx, gy),
            label="Improved A*",
            heuristic_mode="octile",
            heuristic_weight=1.0,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
        )
        v3_view = _run_astar(
            occ, (sx, sy), (gx, gy),
            label="V3",
            heuristic_mode="octile",
            heuristic_weight=1.0,
            heuristic_residual_map=residual_map,
            residual_confidence_map=resolved_conf,
            residual_weight=float(args.residual_weight),
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
        )

        _save_case_figure(
            idx=int(idx),
            occ=occ,
            expert_traj=expert_traj,
            expert_path_xy=expert_path_xy,
            start_xy=(sx, sy),
            goal_xy=(gx, gy),
            astar_view=astar_view,
            improved_view=improved_view,
            v3_view=v3_view,
            residual_map=residual_map,
            resolved_conf=resolved_conf,
            effective_map=effective_map,
            out_path=args.output_dir / f"reason_case_idx{idx:04d}.png",
            dpi=int(args.dpi),
        )

        top_res_mask = residual_map >= float(np.percentile(residual_map, 95))
        if not np.any(top_res_mask):
            top_res_mask = residual_map > 0.0
        conf_top = float(resolved_conf[top_res_mask].mean()) if np.any(top_res_mask) else 0.0
        eff_top = float(effective_map[top_res_mask].mean()) if np.any(top_res_mask) else 0.0

        metrics_rows.append(
            [
                idx,
                sx,
                sy,
                gx,
                gy,
                astar_view.stats.expanded_nodes,
                improved_view.stats.expanded_nodes,
                v3_view.stats.expanded_nodes,
                v3_view.stats.expanded_nodes - astar_view.stats.expanded_nodes,
                v3_view.stats.expanded_nodes - improved_view.stats.expanded_nodes,
                astar_view.path_length,
                improved_view.path_length,
                v3_view.path_length,
                float(residual_map.max()),
                float(np.percentile(residual_map, 95)),
                float(resolved_conf.min()),
                float(resolved_conf.mean()),
                conf_top,
                eff_top,
            ]
        )

    with (args.output_dir / "reason_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
                "astar_expanded",
                "improved_expanded",
                "v3_expanded",
                "v3_minus_astar",
                "v3_minus_improved",
                "astar_path_length",
                "improved_path_length",
                "v3_path_length",
                "residual_max",
                "residual_p95",
                "conf_min",
                "conf_mean",
                "conf_mean_top_residual",
                "effective_mean_top_residual",
            ]
        )
        w.writerows(metrics_rows)


if __name__ == "__main__":
    main()
