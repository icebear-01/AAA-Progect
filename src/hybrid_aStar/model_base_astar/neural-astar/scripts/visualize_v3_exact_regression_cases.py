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

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.guidance_targets import build_clearance_input_map
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)


XY = Tuple[int, int]

COLORS = {
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


@dataclass
class CaseView:
    idx: int
    start_xy: XY
    goal_xy: XY
    occ: np.ndarray
    astar: PlannerView
    improved: PlannerView
    v3: PlannerView


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize exact regression cases from saved case CSV.")
    p.add_argument("--data-npz", type=Path, required=True)
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
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=220)
    p.set_defaults(allow_corner_cut=True)
    return p.parse_args()


def _load_split_arrays(npz_path: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    idx_base = {"train": 0, "valid": 4, "test": 8}[split]
    with np.load(npz_path) as data:
        map_designs = np.asarray(data[f"arr_{idx_base}"], dtype=np.float32)
        goal_maps = np.asarray(data[f"arr_{idx_base+1}"], dtype=np.float32)
    return map_designs, goal_maps


def _load_case_rows(case_csv: Path) -> Dict[int, Dict[str, str]]:
    with case_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["idx"]): r for r in rows}


def _one_hot_xy(width: int, height: int, x: int, y: int) -> np.ndarray:
    arr = np.zeros((1, height, width), dtype=np.float32)
    arr[0, int(y), int(x)] = 1.0
    return arr


def _follow_opt_policy(start_map: np.ndarray, goal_map: np.ndarray, opt_policy: np.ndarray) -> np.ndarray:
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
    while goal_loc != current_loc:
        opt_traj[current_loc] = 1.0
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
    return opt_traj.astype(np.float32)


def _infer_residual_prediction_2d(model: torch.nn.Module, sample: Dict[str, torch.Tensor], device: str) -> Tuple[np.ndarray, np.ndarray | None]:
    occ = sample["occ_map"].unsqueeze(0).to(device)
    start = sample["start_map"].unsqueeze(0).to(device)
    goal = sample["goal_map"].unsqueeze(0).to(device)
    extra_input_maps = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        clearance_input = sample.get("clearance_input_map")
        if clearance_input is None:
            clearance_input_np = build_clearance_input_map(
                occ_map=sample["occ_map"].numpy()[0].astype(np.float32),
                clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
            )[None, ...].astype(np.float32)
            clearance_input = torch.from_numpy(clearance_input_np)
        extra_input_maps = clearance_input.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(occ, start, goal, start_yaw=torch.zeros(1, device=device, dtype=occ.dtype), goal_yaw=torch.zeros(1, device=device, dtype=occ.dtype), extra_input_maps=extra_input_maps)
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
    heuristic_mode: str,
    heuristic_weight: float = 1.0,
    heuristic_residual_map: np.ndarray | None = None,
    residual_confidence_map: np.ndarray | None = None,
    residual_weight: float = 0.0,
    diagonal_cost: float,
    allow_corner_cut: bool,
    label: str,
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


def _panel_title(view: PlannerView) -> str:
    return (
        f"{view.label}\n"
        f"expanded={view.stats.expanded_nodes}  path={view.path_length:.1f}\n"
        f"time={view.runtime_ms:.2f} ms"
    )


def _plot_case_panel(ax: plt.Axes, occ: np.ndarray, opt_traj: np.ndarray, start_xy: XY, goal_xy: XY, view: PlannerView) -> None:
    h, w = occ.shape
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    traj2d = opt_traj[0] if opt_traj.ndim == 3 else opt_traj
    ax.imshow(traj2d, cmap="Blues", alpha=0.14, vmin=0.0, vmax=1.0, interpolation="nearest")
    heat = _expanded_heatmap(view.stats, h, w)
    if float(heat.max()) > 0.0:
        ax.imshow(heat, cmap="magma", alpha=0.58, vmin=0.0, vmax=1.0, interpolation="nearest")
    if view.stats.path is not None and len(view.stats.path) > 1:
        xs = [p[0] for p in view.stats.path]
        ys = [p[1] for p in view.stats.path]
        ax.plot(xs, ys, color=COLORS["v3"], linewidth=2.0, alpha=0.98)
    ax.scatter([start_xy[0]], [start_xy[1]], c=COLORS["start"], s=48, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=COLORS["goal"], s=56, marker="x")
    ax.set_title(_panel_title(view), fontsize=10, pad=8)
    ax.set_axis_off()


def _save_case_figure(case: CaseView, opt_traj: np.ndarray, out_path: Path, dpi: int) -> None:
    planners = [case.astar, case.improved, case.v3]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4))
    for ax, view in zip(np.asarray(axes), planners):
        _plot_case_panel(ax, case.occ, opt_traj, case.start_xy, case.goal_xy, view)
    delta_vs_astar = case.v3.stats.expanded_nodes - case.astar.stats.expanded_nodes
    delta_vs_improved = case.v3.stats.expanded_nodes - case.improved.stats.expanded_nodes
    fig.suptitle(
        f"Regression Case idx={case.idx} | V3-A*={delta_vs_astar:+d} | V3-Improved A*={delta_vs_improved:+d}",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_overview(cases: List[CaseView], out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(len(cases), 1, figsize=(10.0, 3.3 * len(cases)))
    if len(cases) == 1:
        axes = [axes]
    labels = [f"idx {c.idx}" for c in cases]
    astar = [c.astar.stats.expanded_nodes for c in cases]
    improved = [c.improved.stats.expanded_nodes for c in cases]
    v3 = [c.v3.stats.expanded_nodes for c in cases]
    for ax, label, a, i, v in zip(axes, labels, astar, improved, v3):
        ax.bar(["A*", "Improved A*", "V3"], [a, i, v], color=[COLORS["astar"], COLORS["improved"], COLORS["v3"]], edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=13, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        for x, val in enumerate([a, i, v]):
            ax.text(x, val + max(a, i, v) * 0.015, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Three Clear Regression Cases of V3 (Exact Start/Goal Replay)", fontsize=16, weight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    case_rows = _load_case_rows(args.case_csv)
    map_designs, goal_maps = _load_split_arrays(args.data_npz, args.split)
    idx_base = {"train": 0, "valid": 4, "test": 8}[args.split]
    with np.load(args.data_npz) as data:
        opt_policies = np.asarray(data[f"arr_{idx_base+2}"], dtype=np.float32)

    model = load_guidance_encoder(args.ckpt, device=device)
    cases: List[CaseView] = []
    exact_metrics_rows: List[List[object]] = []

    for idx in args.case_indices:
        row = case_rows[int(idx)]
        sx, sy = int(row["start_x"]), int(row["start_y"])
        gx, gy = int(row["goal_x"]), int(row["goal_y"])

        map_design = map_designs[int(idx)]
        occ = (1.0 - map_design).astype(np.float32)
        h, w = occ.shape
        start_map = _one_hot_xy(w, h, sx, sy)
        goal_map = _one_hot_xy(w, h, gx, gy)

        dataset_goal_xy = np.unravel_index(int(np.argmax(goal_maps[int(idx), 0])), goal_maps[int(idx), 0].shape)[::-1]
        if (gx, gy) != (int(dataset_goal_xy[0]), int(dataset_goal_xy[1])):
            raise ValueError(f"goal mismatch for idx={idx}: case_csv={(gx, gy)} dataset={dataset_goal_xy}")

        opt_traj = _follow_opt_policy(start_map, goal_map, opt_policies[int(idx)])
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
        pred_residual, learned_confidence = _infer_residual_prediction_2d(model, sample, device=device)
        residual_confidence_map = None
        if args.residual_confidence_mode != "none":
            residual_confidence_map = resolve_residual_confidence_map(
                mode=args.residual_confidence_mode,
                occ_map=occ,
                residual_map=pred_residual,
                learned_confidence_map=learned_confidence,
                kernel_size=args.residual_confidence_kernel,
                strength=args.residual_confidence_strength,
                min_confidence=args.residual_confidence_min,
            )

        astar_view = _run_astar(
            occ, (sx, sy), (gx, gy),
            heuristic_mode="euclidean",
            heuristic_weight=1.0,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
            label="A*",
        )
        improved_view = _run_astar(
            occ, (sx, sy), (gx, gy),
            heuristic_mode="octile",
            heuristic_weight=1.0,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
            label="Improved A*",
        )
        v3_view = _run_astar(
            occ, (sx, sy), (gx, gy),
            heuristic_mode="octile",
            heuristic_weight=1.0,
            heuristic_residual_map=pred_residual,
            residual_confidence_map=residual_confidence_map,
            residual_weight=args.residual_weight,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
            label="V3",
        )

        case = CaseView(
            idx=int(idx),
            start_xy=(sx, sy),
            goal_xy=(gx, gy),
            occ=occ,
            astar=astar_view,
            improved=improved_view,
            v3=v3_view,
        )
        cases.append(case)
        _save_case_figure(case, opt_traj, args.output_dir / f"exact_regression_idx{idx:04d}.png", int(args.dpi))

        exact_metrics_rows.append(
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
            ]
        )

    _save_overview(cases, args.output_dir / "exact_regression_overview.png", int(args.dpi))
    with (args.output_dir / "exact_case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
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
        ])
        w.writerows(exact_metrics_rows)


if __name__ == "__main__":
    main()
