"""Plot a single planning case in Chinese with three methods only."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import ParkingGuidanceDataset, PlanningNPZGuidanceDataset
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)


XY = Tuple[int, int]

PAPER_COLORS = {
    "traditional": "#0072B2",
    "improved": "#E69F00",
    "proposed": "#009E73",
    "goal": "#D55E00",
}


@dataclass
class PlannerStats:
    stats: Astar8ConnStats
    runtime_ms: float

    @property
    def path_length(self) -> float:
        if self.stats.path is None:
            return 0.0
        return float(path_length_8conn(self.stats.path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one Chinese high-resolution comparison case.")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path, default=None)
    data_group.add_argument("--data-npz", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--case-idx", type=int, required=True)
    parser.add_argument("--case-label", type=str, default="案例")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--residual-weight", type=float, default=1.25)
    parser.add_argument(
        "--residual-confidence-mode",
        type=str,
        default="learned_spike",
        choices=["none", "spike_suppression", "learned", "learned_spike"],
    )
    parser.add_argument("--residual-confidence-kernel", type=int, default=5)
    parser.add_argument("--residual-confidence-strength", type=float, default=0.75)
    parser.add_argument("--residual-confidence-min", type=float, default=0.25)
    parser.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    parser.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    parser.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.set_defaults(allow_corner_cut=True)
    return parser.parse_args()


def _onehot_xy(one_hot_1hw: torch.Tensor) -> XY:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y)


def _infer_residual_map(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    occ = sample["occ_map"].unsqueeze(0).to(device)
    start = sample["start_map"].unsqueeze(0).to(device)
    goal = sample["goal_map"].unsqueeze(0).to(device)
    start_pose = sample.get("start_pose")
    goal_pose = sample.get("goal_pose")
    start_yaw = None
    goal_yaw = None
    if start_pose is not None:
        start_yaw = start_pose[2].view(1).to(device=device, dtype=occ.dtype)
    if goal_pose is not None:
        goal_yaw = goal_pose[2].view(1).to(device=device, dtype=occ.dtype)

    with torch.no_grad():
        out = model(occ, start, goal, start_yaw=start_yaw, goal_yaw=goal_yaw)
    pred = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    scale = None
    if out.scale_map is not None:
        scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32)
    confidence = None
    if out.confidence_map is not None:
        confidence = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
        confidence = confidence[0] if confidence.shape[0] == 1 else np.min(confidence, axis=0).astype(np.float32)
    if pred.ndim != 3:
        raise ValueError(f"Expected output [K,H,W], got {pred.shape}")
    pred = pred[0] if pred.shape[0] == 1 else np.min(pred, axis=0).astype(np.float32)
    if getattr(model, "output_mode", "cost_map") != "residual_heuristic":
        raise ValueError("Checkpoint is not a residual-heuristic model.")
    pred = decode_residual_prediction_np(
        pred,
        transform=str(getattr(model, "residual_target_transform", "none")),
    )
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
    heuristic_residual_map: np.ndarray | None = None,
    residual_confidence_map: np.ndarray | None = None,
    residual_weight: float = 0.0,
    diagonal_cost: float,
    allow_corner_cut: bool,
) -> PlannerStats:
    t0 = time.perf_counter()
    stats = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode=heuristic_mode,
        heuristic_residual_map=heuristic_residual_map,
        residual_confidence_map=residual_confidence_map,
        residual_weight=float(residual_weight),
        diagonal_cost=float(diagonal_cost),
        allow_corner_cut=bool(allow_corner_cut),
        lambda_guidance=0.0,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    return PlannerStats(stats=stats, runtime_ms=runtime_ms)


def _expanded_heatmap(stats: Astar8ConnStats, height: int, width: int) -> np.ndarray:
    heat = np.zeros((height, width), dtype=np.float32)
    for x, y in stats.expanded_xy:
        if 0 <= x < width and 0 <= y < height:
            heat[y, x] += 1.0
    max_value = float(heat.max())
    if max_value > 0.0:
        heat /= max_value
    return heat


def _panel_title(label: str, planner: PlannerStats) -> str:
    return (
        f"{label}\n"
        f"成功={int(planner.stats.success)}  扩展节点={planner.stats.expanded_nodes}\n"
        f"路径长度={planner.path_length:.1f}  时间={planner.runtime_ms:.2f} ms"
    )


def _plot_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    opt_traj: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    planner: PlannerStats,
    title: str,
    font_prop: font_manager.FontProperties,
) -> None:
    height, width = occ.shape
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.imshow(opt_traj, cmap="Blues", alpha=0.16, vmin=0.0, vmax=1.0, interpolation="nearest")
    heat = _expanded_heatmap(planner.stats, height, width)
    if float(heat.max()) > 0.0:
        ax.imshow(heat, cmap="magma", alpha=0.58, vmin=0.0, vmax=1.0, interpolation="nearest")
    if planner.stats.path is not None and len(planner.stats.path) > 1:
        xs = [pt[0] for pt in planner.stats.path]
        ys = [pt[1] for pt in planner.stats.path]
        ax.plot(xs, ys, color=PAPER_COLORS["proposed"], linewidth=2.4, alpha=0.98)
    ax.scatter([start_xy[0]], [start_xy[1]], c=PAPER_COLORS["traditional"], s=64, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=PAPER_COLORS["goal"], s=72, marker="x")
    ax.set_title(title, fontsize=15, fontproperties=font_prop, pad=12)
    ax.set_axis_off()


def _load_dataset(args: argparse.Namespace):
    if args.data_dir is not None:
        return ParkingGuidanceDataset(args.data_dir, orientation_bins=1)
    return PlanningNPZGuidanceDataset(npz_path=args.data_npz, split=args.split, orientation_bins=1)


def _load_sample_by_case_idx(dataset, case_idx: int):
    """Replay dataset sampling from index 0 so the sampled start matches batch evaluation order."""
    sample = None
    for idx in range(case_idx + 1):
        sample = dataset[idx]
    if sample is None:
        raise RuntimeError(f"failed to load sample for case_idx={case_idx}")
    return sample


def main() -> None:
    args = parse_args()
    dataset = _load_dataset(args)
    if args.case_idx < 0 or args.case_idx >= len(dataset):
        raise IndexError(f"case_idx out of range: {args.case_idx} not in [0, {len(dataset) - 1}]")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_guidance_encoder(args.ckpt, device=device)
    sample = _load_sample_by_case_idx(dataset, args.case_idx)
    occ = sample["occ_map"].numpy()[0].astype(np.float32)
    opt_traj = sample["opt_traj"].numpy()[0].astype(np.float32)
    start_xy = _onehot_xy(sample["start_map"])
    goal_xy = _onehot_xy(sample["goal_map"])

    pred_residual, learned_confidence_map = _infer_residual_map(model, sample, device=device)
    learned_conf = None
    if args.residual_confidence_mode != "none":
        learned_conf = resolve_residual_confidence_map(
            mode=args.residual_confidence_mode,
            occ_map=occ,
            residual_map=pred_residual,
            learned_confidence_map=learned_confidence_map,
            kernel_size=args.residual_confidence_kernel,
            strength=args.residual_confidence_strength,
            min_confidence=args.residual_confidence_min,
        )

    traditional = _run_astar(
        occ,
        start_xy,
        goal_xy,
        heuristic_mode="euclidean",
        diagonal_cost=args.diagonal_cost,
        allow_corner_cut=args.allow_corner_cut,
    )
    improved = _run_astar(
        occ,
        start_xy,
        goal_xy,
        heuristic_mode="octile",
        diagonal_cost=args.diagonal_cost,
        allow_corner_cut=args.allow_corner_cut,
    )
    learned = _run_astar(
        occ,
        start_xy,
        goal_xy,
        heuristic_mode="octile",
        heuristic_residual_map=pred_residual,
        residual_confidence_map=learned_conf,
        residual_weight=float(args.residual_weight),
        diagonal_cost=args.diagonal_cost,
        allow_corner_cut=args.allow_corner_cut,
    )

    font_prop = font_manager.FontProperties(fname=str(args.font_path))
    plt.rcParams["axes.unicode_minus"] = False

    planners = [
        ("传统A*", traditional),
        ("改进A*", improved),
        ("本文方法", learned),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))
    for ax, (label, planner) in zip(np.asarray(axes), planners):
        _plot_panel(ax, occ, opt_traj, start_xy, goal_xy, planner, _panel_title(label, planner), font_prop)

    learned_delta = learned.stats.expanded_nodes - improved.stats.expanded_nodes
    fig.suptitle(
        f"{args.case_label} | 编号={args.case_idx} | 本文方法相对改进A*节点变化={learned_delta:+d}",
        fontsize=17,
        fontproperties=font_prop,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(args.output_png)


if __name__ == "__main__":
    main()
