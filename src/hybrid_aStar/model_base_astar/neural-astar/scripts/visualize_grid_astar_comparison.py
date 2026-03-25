"""Visualize traditional 2D A* baseline vs guidance-conditioned comparisons."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import ParkingGuidanceDataset


XY = Tuple[int, int]


@dataclass
class CaseEval:
    idx: int
    baseline: Astar8ConnStats
    secondary: Optional[Astar8ConnStats]
    primary: Astar8ConnStats
    baseline_cost: Optional[np.ndarray]
    secondary_cost: Optional[np.ndarray]
    primary_cost: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize traditional A* comparison on guidance dataset")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--primary-ckpt", type=Path, required=True)
    p.add_argument("--secondary-ckpt", type=Path, default=None)
    p.add_argument("--primary-label", type=str, default="expandproxy")
    p.add_argument("--secondary-label", type=str, default="baseline_guidance")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--orientation-bins", type=int, default=8)
    p.add_argument("--lambda-primary", type=float, default=0.4)
    p.add_argument("--lambda-secondary", type=float, default=0.4)
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grid_astar_comparison"),
    )
    return p.parse_args()


def _onehot_xy(one_hot_1hw: torch.Tensor) -> XY:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y)


def _infer_guidance_2d(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: str,
) -> np.ndarray:
    occ = sample["occ_map"].unsqueeze(0).to(device)
    start = sample["start_map"].unsqueeze(0).to(device)
    goal = sample["goal_map"].unsqueeze(0).to(device)
    start_yaw = sample["start_pose"][2].view(1).to(device=device, dtype=occ.dtype)
    goal_yaw = sample["goal_pose"][2].view(1).to(device=device, dtype=occ.dtype)
    with torch.no_grad():
        out = model(occ, start, goal, start_yaw=start_yaw, goal_yaw=goal_yaw)
    cost = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    if cost.ndim != 3:
        raise ValueError(f"Expected guidance volume [K,H,W], got {cost.shape}")
    return np.min(cost, axis=0).astype(np.float32)


def _expanded_heatmap(stats: Astar8ConnStats, h: int, w: int) -> np.ndarray:
    heat = np.zeros((h, w), dtype=np.float32)
    for x, y in stats.expanded_xy:
        if 0 <= x < w and 0 <= y < h:
            heat[y, x] += 1.0
    if float(heat.max()) > 0.0:
        heat /= float(heat.max())
    return heat


def _select_case_indices(cases: List[CaseEval]) -> List[int]:
    deltas = np.array(
        [float(c.primary.expanded_nodes - c.baseline.expanded_nodes) for c in cases],
        dtype=np.float32,
    )
    best_idx = int(np.argmin(deltas))
    worst_idx = int(np.argmax(deltas))
    median_value = float(np.median(deltas))
    median_rank = np.argsort(np.abs(deltas - median_value))
    median_idx = int(next(i for i in median_rank if i not in {best_idx, worst_idx}))
    picks = [best_idx, median_idx, worst_idx]
    return picks


def _stats_text(label: str, stats: Astar8ConnStats) -> str:
    path_len = 0.0 if stats.path is None else path_length_8conn(stats.path)
    return (
        f"{label}\n"
        f"success={int(stats.success)} expanded={stats.expanded_nodes}\n"
        f"path_len={path_len:.1f}"
    )


def _plot_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    opt_traj: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    stats: Astar8ConnStats,
    title: str,
) -> None:
    h, w = occ.shape
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.imshow(opt_traj, cmap="Blues", alpha=0.18, vmin=0.0, vmax=1.0, interpolation="nearest")
    heat = _expanded_heatmap(stats, h, w)
    if float(heat.max()) > 0.0:
        ax.imshow(heat, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0, interpolation="nearest")
    if stats.path is not None and len(stats.path) > 1:
        xs = [p[0] for p in stats.path]
        ys = [p[1] for p in stats.path]
        ax.plot(xs, ys, color="chartreuse", linewidth=1.6, alpha=0.95)
    ax.scatter([start_xy[0]], [start_xy[1]], c="lime", s=32, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="red", s=36, marker="x")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def _save_case_grid(
    cases: List[CaseEval],
    samples: List[Dict[str, torch.Tensor]],
    secondary_label: Optional[str],
    primary_label: str,
    out_path: Path,
) -> None:
    selected = _select_case_indices(cases)
    ncols = 3 if secondary_label is not None else 2
    fig, axes = plt.subplots(len(selected), ncols, figsize=(4.2 * ncols, 4.0 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    if ncols == 1:
        axes = axes[:, None]

    row_labels = ["Most Improved", "Typical", "Most Regressed"]
    for row_id, case_pos in enumerate(selected):
        case = cases[case_pos]
        sample = samples[case.idx]
        occ = sample["occ_map"].numpy()[0]
        opt_traj = sample["opt_traj"].numpy()[0]
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])

        titles = [_stats_text("baseline", case.baseline)]
        stats_list = [case.baseline]
        if secondary_label is not None and case.secondary is not None:
            titles.append(_stats_text(secondary_label, case.secondary))
            stats_list.append(case.secondary)
        titles.append(_stats_text(primary_label, case.primary))
        stats_list.append(case.primary)

        for col_id, (title, stats) in enumerate(zip(titles, stats_list)):
            ax = axes[row_id, col_id]
            _plot_panel(ax, occ, opt_traj, start_xy, goal_xy, stats, title=title)
            if col_id == 0:
                delta = case.primary.expanded_nodes - case.baseline.expanded_nodes
                ax.text(
                    -0.02,
                    0.5,
                    f"{row_labels[row_id]}\ncase={case.idx}\nprimary_delta={delta:+d}",
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

    fig.suptitle("Traditional 2D A*: expansion heatmap + path comparison", fontsize=13)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _save_summary_plot(
    cases: List[CaseEval],
    secondary_label: Optional[str],
    primary_label: str,
    out_path: Path,
) -> None:
    baseline_exp = np.array([c.baseline.expanded_nodes for c in cases], dtype=np.float32)
    primary_exp = np.array([c.primary.expanded_nodes for c in cases], dtype=np.float32)
    secondary_exp = (
        np.array([c.secondary.expanded_nodes for c in cases], dtype=np.float32)
        if secondary_label is not None
        else None
    )
    order = np.argsort(primary_exp - baseline_exp)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(primary_exp[order] - baseline_exp[order], label=f"{primary_label} - baseline", color="tab:green")
    if secondary_exp is not None:
        axes[0].plot(
            secondary_exp[order] - baseline_exp[order],
            label=f"{secondary_label} - baseline",
            color="tab:orange",
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    axes[0].set_title("Sorted Delta Expanded Nodes")
    axes[0].set_xlabel("case rank")
    axes[0].set_ylabel("guided - baseline")
    axes[0].legend(fontsize=8)

    mean_vals = [float(baseline_exp.mean())]
    labels = ["baseline"]
    colors = ["tab:blue"]
    if secondary_exp is not None:
        mean_vals.append(float(secondary_exp.mean()))
        labels.append(secondary_label)
        colors.append("tab:orange")
    mean_vals.append(float(primary_exp.mean()))
    labels.append(primary_label)
    colors.append("tab:green")
    axes[1].bar(labels, mean_vals, color=colors)
    axes[1].set_title("Mean Expanded Nodes")
    axes[1].set_ylabel("expanded nodes")
    for i, v in enumerate(mean_vals):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Traditional 2D A* summary over {len(cases)} cases",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ds = ParkingGuidanceDataset(args.data_dir, orientation_bins=args.orientation_bins)
    eval_count = min(int(args.max_samples), len(ds))
    samples = [ds[i] for i in range(eval_count)]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    primary_model = load_guidance_encoder(args.primary_ckpt, device=device)
    secondary_model = (
        load_guidance_encoder(args.secondary_ckpt, device=device)
        if args.secondary_ckpt is not None
        else None
    )

    cases: List[CaseEval] = []
    for idx, sample in enumerate(samples):
        occ = sample["occ_map"].numpy()[0]
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])

        baseline = astar_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            guidance_cost=None,
            lambda_guidance=0.0,
        )
        secondary_cost = None
        secondary_stats = None
        if secondary_model is not None:
            secondary_cost = _infer_guidance_2d(secondary_model, sample, device=device)
            secondary_stats = astar_8conn_stats(
                occ_map=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                guidance_cost=secondary_cost,
                lambda_guidance=float(args.lambda_secondary),
            )

        primary_cost = _infer_guidance_2d(primary_model, sample, device=device)
        primary_stats = astar_8conn_stats(
            occ_map=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            guidance_cost=primary_cost,
            lambda_guidance=float(args.lambda_primary),
        )
        cases.append(
            CaseEval(
                idx=idx,
                baseline=baseline,
                secondary=secondary_stats,
                primary=primary_stats,
                baseline_cost=None,
                secondary_cost=secondary_cost,
                primary_cost=primary_cost,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_grid = args.output_dir / "grid_astar_case_grid.png"
    summary_png = args.output_dir / "grid_astar_summary.png"
    _save_case_grid(
        cases=cases,
        samples=samples,
        secondary_label=args.secondary_label if secondary_model is not None else None,
        primary_label=args.primary_label,
        out_path=case_grid,
    )
    _save_summary_plot(
        cases=cases,
        secondary_label=args.secondary_label if secondary_model is not None else None,
        primary_label=args.primary_label,
        out_path=summary_png,
    )

    baseline_exp = np.mean([c.baseline.expanded_nodes for c in cases])
    primary_exp = np.mean([c.primary.expanded_nodes for c in cases])
    print(f"saved_case_grid={case_grid}")
    print(f"saved_summary={summary_png}")
    print(f"eval_cases={len(cases)} baseline_expanded_mean={baseline_exp:.3f}")
    if secondary_model is not None:
        secondary_exp = np.mean([c.secondary.expanded_nodes for c in cases if c.secondary is not None])
        print(f"secondary_expanded_mean={secondary_exp:.3f}")
    print(f"primary_expanded_mean={primary_exp:.3f}")


if __name__ == "__main__":
    main()
