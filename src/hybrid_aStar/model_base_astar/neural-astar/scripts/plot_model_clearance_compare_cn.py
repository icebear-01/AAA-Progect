"""Compare two transformer-guided grid A* checkpoints on efficiency and clearance."""

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
from scipy.ndimage import distance_transform_edt

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import PlanningNPZGuidanceDataset
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from neural_astar.utils.residual_prediction import apply_residual_scale_np, decode_residual_prediction_np


XY = Tuple[int, int]
PLOT_COLORS = {
    "old": "#0072B2",
    "new": "#009E73",
    "start": "#56B4E9",
    "goal": "#D55E00",
}


@dataclass
class PlannerRun:
    stats: Astar8ConnStats
    runtime_ms: float
    path_length: float
    clearance_mean: float
    clearance_min: float


@dataclass
class CaseResult:
    idx: int
    occ: np.ndarray
    start_xy: XY
    goal_xy: XY
    old: PlannerRun
    new: PlannerRun


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two transformer-guided checkpoints on clearance.")
    p.add_argument("--data-npz", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--old-ckpt", type=Path, required=True)
    p.add_argument("--new-ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument("--residual-weight", type=float, default=1.25)
    p.add_argument("--residual-confidence-mode", type=str, default="learned_spike")
    p.add_argument("--residual-confidence-kernel", type=int, default=5)
    p.add_argument("--residual-confidence-strength", type=float, default=0.75)
    p.add_argument("--residual-confidence-min", type=float, default=0.25)
    p.add_argument("--num-cases", type=int, default=4)
    p.add_argument("--font-path", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _onehot_xy(one_hot_1hw: torch.Tensor) -> XY:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y)


def _load_font(font_path: Path | None) -> font_manager.FontProperties:
    if font_path is not None and font_path.exists():
        return font_manager.FontProperties(fname=str(font_path))
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties()


def _infer_residual_map(
    model: torch.nn.Module,
    sample: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[np.ndarray, np.ndarray | None]:
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
    extra_input_maps = None
    if int(getattr(model, "extra_input_channels", 0)) > 0 and "clearance_input_map" in sample:
        extra_input_maps = sample["clearance_input_map"].unsqueeze(0).to(device=device, dtype=occ.dtype)
    with torch.no_grad():
        out = model(
            occ,
            start,
            goal,
            start_yaw=start_yaw,
            goal_yaw=goal_yaw,
            extra_input_maps=extra_input_maps,
        )
    pred = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    pred = pred[0] if pred.shape[0] == 1 else np.min(pred, axis=0).astype(np.float32)
    pred = decode_residual_prediction_np(
        pred,
        transform=str(getattr(model, "residual_target_transform", "none")),
    )
    scale = None
    if out.scale_map is not None:
        scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32)
        scale = scale[0] if scale.shape[0] == 1 else np.min(scale, axis=0).astype(np.float32)
    pred = apply_residual_scale_np(pred, scale)
    confidence = None
    if out.confidence_map is not None:
        confidence = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
        confidence = confidence[0] if confidence.shape[0] == 1 else np.min(confidence, axis=0).astype(np.float32)
    return pred.astype(np.float32), confidence


def _run(
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    residual_map: np.ndarray,
    residual_confidence_map: np.ndarray | None,
    *,
    residual_weight: float,
) -> PlannerRun:
    dist_map = distance_transform_edt(occ < 0.5).astype(np.float32)
    t0 = time.perf_counter()
    stats = astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        allow_corner_cut=True,
        heuristic_residual_map=residual_map,
        residual_confidence_map=residual_confidence_map,
        residual_weight=float(residual_weight),
        guidance_integration_mode="heuristic_bonus",
        guidance_bonus_threshold=0.6,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - t0)
    path_length = 0.0 if stats.path is None else float(path_length_8conn(stats.path))
    if stats.path:
        values = np.asarray([dist_map[y, x] for x, y in stats.path], dtype=np.float32)
        clearance_mean = float(values.mean())
        clearance_min = float(values.min())
    else:
        clearance_mean = 0.0
        clearance_min = 0.0
    return PlannerRun(
        stats=stats,
        runtime_ms=runtime_ms,
        path_length=path_length,
        clearance_mean=clearance_mean,
        clearance_min=clearance_min,
    )


def _plot_path(
    ax: plt.Axes,
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    run: PlannerRun,
    title: str,
    color: str,
    font_prop: font_manager.FontProperties,
) -> None:
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if run.stats.path is not None and len(run.stats.path) > 1:
        xs = [pt[0] for pt in run.stats.path]
        ys = [pt[1] for pt in run.stats.path]
        ax.plot(xs, ys, color=color, linewidth=2.6, alpha=0.98)
    ax.scatter([start_xy[0]], [start_xy[1]], c=PLOT_COLORS["start"], s=60, marker="o", edgecolors="white", linewidths=0.8)
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=PLOT_COLORS["goal"], s=64, marker="x", linewidths=1.4)
    ax.set_title(title, fontsize=14, fontproperties=font_prop, pad=8)
    ax.set_axis_off()


def _save_case_figure(case: CaseResult, out_path: Path, font_prop: font_manager.FontProperties, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.1))
    title_left = (
        "旧 Transformer 引导\n"
        f"扩展节点={case.old.stats.expanded_nodes}  时间={case.old.runtime_ms:.2f} ms\n"
        f"平均净空={case.old.clearance_mean:.2f}  最小净空={case.old.clearance_min:.2f}"
    )
    title_right = (
        "新模型（距离变换输入）\n"
        f"扩展节点={case.new.stats.expanded_nodes}  时间={case.new.runtime_ms:.2f} ms\n"
        f"平均净空={case.new.clearance_mean:.2f}  最小净空={case.new.clearance_min:.2f}"
    )
    _plot_path(axes[0], case.occ, case.start_xy, case.goal_xy, case.old, title_left, PLOT_COLORS["old"], font_prop)
    _plot_path(axes[1], case.occ, case.start_xy, case.goal_xy, case.new, title_right, PLOT_COLORS["new"], font_prop)
    delta_mean = case.new.clearance_mean - case.old.clearance_mean
    delta_min = case.new.clearance_min - case.old.clearance_min
    delta_exp = case.new.stats.expanded_nodes - case.old.stats.expanded_nodes
    fig.suptitle(
        f"案例 {case.idx} | 平均净空变化={delta_mean:+.2f} | 最小净空变化={delta_min:+.2f} | 扩展节点变化={delta_exp:+d}",
        fontsize=16,
        fontproperties=font_prop,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_summary(cases: Sequence[CaseResult], out_path: Path, font_prop: font_manager.FontProperties, dpi: int) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(10.8, 4.0 * len(cases)))
    if len(cases) == 1:
        axes = np.asarray([axes])
    for row_idx, case in enumerate(cases):
        axes_row = axes[row_idx]
        _plot_path(
            axes_row[0],
            case.occ,
            case.start_xy,
            case.goal_xy,
            case.old,
            f"案例{case.idx} 旧模型",
            PLOT_COLORS["old"],
            font_prop,
        )
        _plot_path(
            axes_row[1],
            case.occ,
            case.start_xy,
            case.goal_xy,
            case.new,
            f"案例{case.idx} 新模型",
            PLOT_COLORS["new"],
            font_prop,
        )
    fig.suptitle("旧 Transformer 与距离变换输入新模型对比", fontsize=18, fontproperties=font_prop, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_case_metrics(cases: Sequence[CaseResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "idx",
                "old_expanded",
                "new_expanded",
                "old_runtime_ms",
                "new_runtime_ms",
                "old_path_length",
                "new_path_length",
                "old_clearance_mean",
                "new_clearance_mean",
                "old_clearance_min",
                "new_clearance_min",
                "expanded_delta",
                "clearance_mean_delta",
                "clearance_min_delta",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.idx,
                    case.old.stats.expanded_nodes,
                    case.new.stats.expanded_nodes,
                    case.old.runtime_ms,
                    case.new.runtime_ms,
                    case.old.path_length,
                    case.new.path_length,
                    case.old.clearance_mean,
                    case.new.clearance_mean,
                    case.old.clearance_min,
                    case.new.clearance_min,
                    case.new.stats.expanded_nodes - case.old.stats.expanded_nodes,
                    case.new.clearance_mean - case.old.clearance_mean,
                    case.new.clearance_min - case.old.clearance_min,
                ]
            )


def _write_summary(all_cases: Sequence[CaseResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0

    old_expanded = _mean([c.old.stats.expanded_nodes for c in all_cases])
    new_expanded = _mean([c.new.stats.expanded_nodes for c in all_cases])
    old_runtime = _mean([c.old.runtime_ms for c in all_cases])
    new_runtime = _mean([c.new.runtime_ms for c in all_cases])
    old_length = _mean([c.old.path_length for c in all_cases])
    new_length = _mean([c.new.path_length for c in all_cases])
    old_clear_mean = _mean([c.old.clearance_mean for c in all_cases])
    new_clear_mean = _mean([c.new.clearance_mean for c in all_cases])
    old_clear_min = _mean([c.old.clearance_min for c in all_cases])
    new_clear_min = _mean([c.new.clearance_min for c in all_cases])

    rows = [
        ("expanded_nodes", old_expanded, new_expanded),
        ("runtime_ms", old_runtime, new_runtime),
        ("path_length", old_length, new_length),
        ("clearance_mean", old_clear_mean, new_clear_mean),
        ("clearance_min", old_clear_min, new_clear_min),
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "old_model", "new_model", "absolute_delta", "relative_delta_pct"])
        for metric, old_val, new_val in rows:
            delta = new_val - old_val
            rel = 0.0 if abs(old_val) < 1e-8 else 100.0 * delta / old_val
            writer.writerow([metric, old_val, new_val, delta, rel])


def main() -> None:
    args = parse_args()
    font_prop = _load_font(args.font_path)
    plt.rcParams["axes.unicode_minus"] = False

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dataset = PlanningNPZGuidanceDataset(
        npz_path=args.data_npz,
        split=args.split,
        orientation_bins=1,
        seed=int(args.seed),
        clearance_input_clip_distance=4.0,
    )
    eval_count = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    old_model = load_guidance_encoder(args.old_ckpt, device=device)
    new_model = load_guidance_encoder(args.new_ckpt, device=device)
    old_model.eval()
    new_model.eval()

    cases: List[CaseResult] = []
    for idx in range(eval_count):
        sample = dataset[idx]
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])

        old_residual_map, old_confidence_map = _infer_residual_map(old_model, sample, device=device)
        old_residual_confidence_map = resolve_residual_confidence_map(
            mode=args.residual_confidence_mode,
            occ_map=occ,
            residual_map=old_residual_map,
            learned_confidence_map=old_confidence_map,
            kernel_size=args.residual_confidence_kernel,
            strength=args.residual_confidence_strength,
            min_confidence=args.residual_confidence_min,
        )
        old_run = _run(
            occ,
            start_xy,
            goal_xy,
            old_residual_map,
            old_residual_confidence_map,
            residual_weight=args.residual_weight,
        )

        new_residual_map, new_confidence_map = _infer_residual_map(new_model, sample, device=device)
        new_residual_confidence_map = resolve_residual_confidence_map(
            mode=args.residual_confidence_mode,
            occ_map=occ,
            residual_map=new_residual_map,
            learned_confidence_map=new_confidence_map,
            kernel_size=args.residual_confidence_kernel,
            strength=args.residual_confidence_strength,
            min_confidence=args.residual_confidence_min,
        )
        new_run = _run(
            occ,
            start_xy,
            goal_xy,
            new_residual_map,
            new_residual_confidence_map,
            residual_weight=args.residual_weight,
        )

        if old_run.stats.success and new_run.stats.success:
            cases.append(
                CaseResult(
                    idx=idx,
                    occ=occ,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    old=old_run,
                    new=new_run,
                )
            )
        if (idx + 1) % 50 == 0 or idx + 1 == eval_count:
            print(f"processed {idx + 1}/{eval_count}")

    _write_summary(cases, args.output_dir / "summary_metrics.csv")

    ranked = sorted(
        cases,
        key=lambda c: (
            c.new.clearance_mean - c.old.clearance_mean,
            -(c.new.stats.expanded_nodes - c.old.stats.expanded_nodes),
            c.new.clearance_min - c.old.clearance_min,
        ),
        reverse=True,
    )
    selected = ranked[: max(1, args.num_cases)]
    _write_case_metrics(selected, args.output_dir / "selected_case_metrics.csv")
    _save_summary(selected, args.output_dir / "model_clearance_summary_cn.png", font_prop, args.dpi)
    for rank, case in enumerate(selected, start=1):
        _save_case_figure(case, args.output_dir / f"model_clearance_rank{rank:02d}_idx{case.idx:04d}_cn.png", font_prop, args.dpi)


if __name__ == "__main__":
    main()
