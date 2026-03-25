"""Visualize traditional/improved/residual-guided 2D A* planning cases."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    label: str
    stats: Astar8ConnStats
    runtime_ms: float

    @property
    def path_length(self) -> float:
        if self.stats.path is None:
            return 0.0
        return float(path_length_8conn(self.stats.path))


@dataclass
class CaseEval:
    idx: int
    traditional: PlannerStats
    improved: PlannerStats
    learned: PlannerStats
    oracle: PlannerStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize residual-guided grid A* planning cases.")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path, default=None)
    data_group.add_argument("--data-npz", type=Path, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Split used when evaluating planning-datasets .npz.",
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--residual-weight", type=float, default=1.25)
    parser.add_argument(
        "--residual-confidence-mode",
        type=str,
        default="none",
        choices=["none", "spike_suppression", "learned", "learned_spike"],
    )
    parser.add_argument("--residual-confidence-kernel", type=int, default=5)
    parser.add_argument("--residual-confidence-strength", type=float, default=0.75)
    parser.add_argument("--residual-confidence-min", type=float, default=0.25)
    parser.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    parser.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    parser.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for case images and summary outputs.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        help="Chinese font file used by matplotlib.",
    )
    parser.add_argument("--dpi", type=int, default=240)
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
    return PlannerStats(label=heuristic_mode, stats=stats, runtime_ms=runtime_ms)


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
        ax.plot(xs, ys, color=PAPER_COLORS["proposed"], linewidth=2.0, alpha=0.98)
    ax.scatter([start_xy[0]], [start_xy[1]], c=PAPER_COLORS["traditional"], s=54, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=PAPER_COLORS["goal"], s=60, marker="x")
    ax.set_title(title, fontsize=12, fontproperties=font_prop, pad=10)
    ax.set_axis_off()


def _select_case_indices(cases: Sequence[CaseEval]) -> List[int]:
    deltas = np.array(
        [float(case.learned.stats.expanded_nodes - case.improved.stats.expanded_nodes) for case in cases],
        dtype=np.float32,
    )
    best_idx = int(np.argmin(deltas))
    worst_idx = int(np.argmax(deltas))
    median_value = float(np.median(deltas))
    median_rank = np.argsort(np.abs(deltas - median_value))
    typical_idx = int(next(i for i in median_rank if i not in {best_idx, worst_idx}))
    return [best_idx, typical_idx, worst_idx]


def _save_case_figure(
    row_label: str,
    case: CaseEval,
    sample: Dict[str, torch.Tensor],
    out_path: Path,
    font_prop: font_manager.FontProperties,
    dpi: int,
) -> None:
    occ = sample["occ_map"].numpy()[0].astype(np.float32)
    opt_traj = sample["opt_traj"].numpy()[0].astype(np.float32)
    start_xy = _onehot_xy(sample["start_map"])
    goal_xy = _onehot_xy(sample["goal_map"])

    planners = [
        ("传统A*", case.traditional),
        ("改进A*", case.improved),
        ("本文方法", case.learned),
    ]

    fig, axes = plt.subplots(1, len(planners), figsize=(5.4 * len(planners), 5.7))
    for ax, (label, planner) in zip(np.asarray(axes), planners):
        _plot_panel(ax, occ, opt_traj, start_xy, goal_xy, planner, _panel_title(label, planner), font_prop)

    learned_delta = case.learned.stats.expanded_nodes - case.improved.stats.expanded_nodes
    fig.suptitle(
        f"{row_label} | 编号={case.idx} | 本文方法相对改进A*节点变化={learned_delta:+d}",
        fontsize=16,
        fontproperties=font_prop,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_summary_plot(
    cases: Sequence[CaseEval],
    out_path: Path,
    font_prop: font_manager.FontProperties,
    dpi: int,
) -> None:
    improved_exp = np.array([c.improved.stats.expanded_nodes for c in cases], dtype=np.float32)
    learned_exp = np.array([c.learned.stats.expanded_nodes for c in cases], dtype=np.float32)
    traditional_exp = np.array([c.traditional.stats.expanded_nodes for c in cases], dtype=np.float32)
    order = np.argsort(learned_exp - improved_exp)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].plot(
        learned_exp[order] - improved_exp[order],
        label="本文方法 - 改进A*",
        color=PAPER_COLORS["proposed"],
        linewidth=2.2,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    axes[0].set_title("排序后的节点变化", fontproperties=font_prop, fontsize=16)
    axes[0].set_xlabel("案例序号", fontproperties=font_prop, fontsize=13)
    axes[0].set_ylabel("相对改进A*的节点变化", fontproperties=font_prop, fontsize=13)
    axes[0].tick_params(labelsize=11)
    legend = axes[0].legend(fontsize=11)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

    labels = ["传统A*", "改进A*", "本文方法"]
    mean_vals = [
        float(traditional_exp.mean()),
        float(improved_exp.mean()),
        float(learned_exp.mean()),
    ]
    colors = [
        PAPER_COLORS["traditional"],
        PAPER_COLORS["improved"],
        PAPER_COLORS["proposed"],
    ]
    axes[1].bar(labels, mean_vals, color=colors)
    axes[1].set_title("平均扩展节点", fontproperties=font_prop, fontsize=16)
    axes[1].set_ylabel("节点", fontproperties=font_prop, fontsize=13)
    axes[1].tick_params(labelsize=11)
    for tick in axes[1].get_xticklabels():
        tick.set_fontproperties(font_prop)
    for idx, val in enumerate(mean_vals):
        axes[1].text(idx, val, f"{val:.1f}", ha="center", va="bottom", fontsize=12)

    fig.suptitle(f"基于Transformer引导的A*结果汇总（共{len(cases)}个案例）", fontsize=18, fontproperties=font_prop)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_summary_metrics(cases: Sequence[CaseEval], out_path: Path) -> None:
    def _mean(attr: str, group: str) -> float:
        values = []
        for case in cases:
            planner = getattr(case, group)
            values.append(float(getattr(planner, attr)))
        return float(np.mean(values))

    learned_minus_improved = np.array(
        [float(case.learned.stats.expanded_nodes - case.improved.stats.expanded_nodes) for case in cases],
        dtype=np.float32,
    )
    oracle_minus_improved = np.array(
        [float(case.oracle.stats.expanded_nodes - case.improved.stats.expanded_nodes) for case in cases],
        dtype=np.float32,
    )
    rows = [
        ["metric", "traditional", "improved", "learned", "oracle"],
        [
            "expanded_nodes_mean",
            np.mean([c.traditional.stats.expanded_nodes for c in cases]),
            np.mean([c.improved.stats.expanded_nodes for c in cases]),
            np.mean([c.learned.stats.expanded_nodes for c in cases]),
            np.mean([c.oracle.stats.expanded_nodes for c in cases]),
        ],
        [
            "runtime_ms_mean",
            np.mean([c.traditional.runtime_ms for c in cases]),
            np.mean([c.improved.runtime_ms for c in cases]),
            np.mean([c.learned.runtime_ms for c in cases]),
            np.mean([c.oracle.runtime_ms for c in cases]),
        ],
        [
            "path_length_mean",
            np.mean([c.traditional.path_length for c in cases]),
            np.mean([c.improved.path_length for c in cases]),
            np.mean([c.learned.path_length for c in cases]),
            np.mean([c.oracle.path_length for c in cases]),
        ],
        ["learned_minus_improved_mean", "", "", float(learned_minus_improved.mean()), ""],
        ["learned_minus_improved_median", "", "", float(np.median(learned_minus_improved)), ""],
        ["oracle_minus_improved_mean", "", "", "", float(oracle_minus_improved.mean())],
        ["oracle_minus_improved_median", "", "", "", float(np.median(oracle_minus_improved))],
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _save_case_metrics(cases: Sequence[CaseEval], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "traditional_expanded",
                "improved_expanded",
                "learned_expanded",
                "oracle_expanded",
                "learned_minus_improved",
                "oracle_minus_improved",
                "traditional_path_length",
                "improved_path_length",
                "learned_path_length",
                "oracle_path_length",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.idx,
                    case.traditional.stats.expanded_nodes,
                    case.improved.stats.expanded_nodes,
                    case.learned.stats.expanded_nodes,
                    case.oracle.stats.expanded_nodes,
                    case.learned.stats.expanded_nodes - case.improved.stats.expanded_nodes,
                    case.oracle.stats.expanded_nodes - case.improved.stats.expanded_nodes,
                    case.traditional.path_length,
                    case.improved.path_length,
                    case.learned.path_length,
                    case.oracle.path_length,
                ]
            )


def main() -> None:
    args = parse_args()
    font_prop = font_manager.FontProperties(fname=str(args.font_path))
    plt.rcParams["axes.unicode_minus"] = False
    if args.data_dir is not None:
        dataset = ParkingGuidanceDataset(args.data_dir, orientation_bins=1)
    else:
        dataset = PlanningNPZGuidanceDataset(
            npz_path=args.data_npz,
            split=args.split,
            orientation_bins=1,
        )

    eval_count = len(dataset) if int(args.max_samples) <= 0 else min(len(dataset), int(args.max_samples))
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = load_guidance_encoder(args.ckpt, device=device)
    if getattr(model, "output_mode", "cost_map") != "residual_heuristic":
        raise ValueError("Expected residual-heuristic checkpoint for visualization.")

    samples: List[Dict[str, torch.Tensor]] = []
    cases: List[CaseEval] = []
    for idx in range(eval_count):
        sample = dataset[idx]
        samples.append(sample)
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])

        pred_residual, learned_confidence_map = _infer_residual_map(model, sample, device=device)
        oracle_residual = sample["residual_heuristic_map"].numpy()[0].astype(np.float32)
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
        oracle = _run_astar(
            occ,
            start_xy,
            goal_xy,
            heuristic_mode="octile",
            heuristic_residual_map=oracle_residual,
            residual_weight=1.0,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
        )
        cases.append(
            CaseEval(
                idx=idx,
                traditional=PlannerStats(label="traditional", stats=traditional.stats, runtime_ms=traditional.runtime_ms),
                improved=PlannerStats(label="improved", stats=improved.stats, runtime_ms=improved.runtime_ms),
                learned=PlannerStats(label="learned", stats=learned.stats, runtime_ms=learned.runtime_ms),
                oracle=PlannerStats(label="oracle", stats=oracle.stats, runtime_ms=oracle.runtime_ms),
            )
        )
        if idx % 32 == 0 or idx + 1 == eval_count:
            print(f"processed={idx + 1}/{eval_count}")

    selected = _select_case_indices(cases)
    row_labels = ["改善最明显", "典型案例", "退化最明显"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row_label, pos in zip(row_labels, selected):
        case = cases[pos]
        out_stem = {
            "改善最明显": "most_improved",
            "典型案例": "typical",
            "退化最明显": "most_regressed",
        }[row_label]
        out_path = args.output_dir / f"{out_stem}_idx{case.idx:04d}.png"
        _save_case_figure(row_label, case, samples[case.idx], out_path, font_prop, int(args.dpi))

    _save_summary_plot(cases, args.output_dir / "summary.png", font_prop, int(args.dpi))
    _save_summary_metrics(cases, args.output_dir / "summary_metrics.csv")
    _save_case_metrics(cases, args.output_dir / "case_metrics.csv")

    print(f"saved_dir={args.output_dir}")


if __name__ == "__main__":
    main()
