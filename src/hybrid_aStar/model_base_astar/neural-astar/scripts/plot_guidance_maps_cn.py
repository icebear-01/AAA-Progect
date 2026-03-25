"""Visualize transformer guidance maps for multiple cases in Chinese."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors, font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from plot_case_compare_cn import (
    PAPER_COLORS,
    _infer_residual_map,
    _load_dataset,
    _load_sample_by_case_idx,
    _onehot_xy,
    _panel_title,
    _plot_panel,
    _run_astar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Chinese guidance maps for selected cases.")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=Path, default=None)
    data_group.add_argument("--data-npz", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--case-idxs", type=int, nargs="+", required=True)
    parser.add_argument("--row-labels", type=str, nargs="*", default=None)
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
    parser.add_argument("--dpi", type=int, default=320)
    parser.set_defaults(allow_corner_cut=True)
    return parser.parse_args()


def _masked_overlay(ax: plt.Axes, occ: np.ndarray, data: np.ndarray, cmap, norm, title: str, font_prop, start_xy, goal_xy):
    scene_cmap = colors.ListedColormap(["#161616", "#D8D8D8"])
    ax.imshow(1.0 - occ, cmap=scene_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    masked = np.ma.masked_where(occ > 0.5, data)
    ax.imshow(masked, cmap=cmap, norm=norm, alpha=0.92, interpolation="nearest")
    ax.scatter([start_xy[0]], [start_xy[1]], c=PAPER_COLORS["traditional"], s=46, marker="o")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c=PAPER_COLORS["goal"], s=52, marker="x")
    ax.set_title(title, fontsize=15, fontproperties=font_prop, pad=10)
    ax.set_axis_off()


def _add_axis_colorbar(ax: plt.Axes, norm, cmap, label: str, font_prop):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.2%", pad=0.06)
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.set_ylabel(label, fontproperties=font_prop, fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    return cbar


def main() -> None:
    args = parse_args()
    if args.row_labels and len(args.row_labels) not in {0, len(args.case_idxs)}:
        raise ValueError("--row-labels length must match --case-idxs length")
    row_labels = list(args.row_labels) if args.row_labels else [f"案例{i + 1}" for i in range(len(args.case_idxs))]

    base_dataset = _load_dataset(args)
    for case_idx in args.case_idxs:
        if case_idx < 0 or case_idx >= len(base_dataset):
            raise IndexError(f"case_idx out of range: {case_idx} not in [0, {len(base_dataset) - 1}]")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_guidance_encoder(args.ckpt, device=device)
    font_prop = font_manager.FontProperties(fname=str(args.font_path))
    plt.rcParams["axes.unicode_minus"] = False

    residual_rows = []
    conf_rows = []
    effective_rows = []
    meta_rows = []

    for row_label, case_idx in zip(row_labels, args.case_idxs):
        dataset = _load_dataset(args)
        sample = _load_sample_by_case_idx(dataset, case_idx)
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])

        pred_residual, learned_confidence_map = _infer_residual_map(model, sample, device=device)
        resolved_conf = np.ones_like(pred_residual, dtype=np.float32)
        if args.residual_confidence_mode != "none":
            resolved_conf = resolve_residual_confidence_map(
                mode=args.residual_confidence_mode,
                occ_map=occ,
                residual_map=pred_residual,
                learned_confidence_map=learned_confidence_map,
                kernel_size=args.residual_confidence_kernel,
                strength=args.residual_confidence_strength,
                min_confidence=args.residual_confidence_min,
            )
        resolved_conf = np.clip(np.asarray(resolved_conf, dtype=np.float32), 0.0, 1.0)
        effective_guidance = float(args.residual_weight) * pred_residual * resolved_conf

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
            residual_confidence_map=resolved_conf,
            residual_weight=float(args.residual_weight),
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
        )

        residual_rows.append(pred_residual)
        conf_rows.append(resolved_conf)
        effective_rows.append(effective_guidance)
        meta_rows.append(
            {
                "row_label": row_label,
                "case_idx": case_idx,
                "occ": occ,
                "opt_traj": sample["opt_traj"].numpy()[0].astype(np.float32),
                "start_xy": start_xy,
                "goal_xy": goal_xy,
                "delta": int(learned.stats.expanded_nodes - improved.stats.expanded_nodes),
                "learned_planner": learned,
            }
        )

    residual_vmax = max(float(np.percentile(x[x > 0.0], 98)) if np.any(x > 0.0) else 1.0 for x in residual_rows)
    effective_vmax = max(float(np.percentile(x[x > 0.0], 98)) if np.any(x > 0.0) else 1.0 for x in effective_rows)
    residual_norm = colors.Normalize(vmin=0.0, vmax=max(residual_vmax, 1e-6))
    conf_norm = colors.Normalize(vmin=0.0, vmax=1.0)
    effective_norm = colors.Normalize(vmin=0.0, vmax=max(effective_vmax, 1e-6))

    residual_cmap = plt.colormaps["cividis"].copy()
    residual_cmap.set_bad("#111111")
    conf_cmap = plt.colormaps["viridis"].copy()
    conf_cmap.set_bad("#111111")

    nrows = len(meta_rows)
    fig, axes = plt.subplots(nrows, 4, figsize=(22, 4.8 * nrows))
    axes = np.atleast_2d(axes)

    for row_idx, meta in enumerate(meta_rows):
        occ = meta["occ"]
        opt_traj = meta["opt_traj"]
        start_xy = meta["start_xy"]
        goal_xy = meta["goal_xy"]
        learned_planner = meta["learned_planner"]

        _plot_panel(
            axes[row_idx, 0],
            occ,
            opt_traj,
            start_xy,
            goal_xy,
            learned_planner,
            _panel_title("本文方法", learned_planner),
            font_prop,
        )

        _masked_overlay(
            axes[row_idx, 1],
            occ,
            residual_rows[row_idx],
            residual_cmap,
            residual_norm,
            "预测引导残差图",
            font_prop,
            start_xy,
            goal_xy,
        )
        _add_axis_colorbar(axes[row_idx, 1], residual_norm, residual_cmap, "残差值", font_prop)
        _masked_overlay(
            axes[row_idx, 2],
            occ,
            conf_rows[row_idx],
            conf_cmap,
            conf_norm,
            "置信度图",
            font_prop,
            start_xy,
            goal_xy,
        )
        _add_axis_colorbar(axes[row_idx, 2], conf_norm, conf_cmap, "置信度", font_prop)
        _masked_overlay(
            axes[row_idx, 3],
            occ,
            effective_rows[row_idx],
            residual_cmap,
            effective_norm,
            "有效引导图（送入A*）",
            font_prop,
            start_xy,
            goal_xy,
        )
        _add_axis_colorbar(axes[row_idx, 3], effective_norm, residual_cmap, "有效引导值", font_prop)

        if nrows > 1:
            axes[row_idx, 1].text(
                0.5,
                1.08,
                f"{meta['row_label']} | 编号={meta['case_idx']} | 节点变化={meta['delta']:+d}",
                transform=axes[row_idx, 1].transAxes,
                ha="center",
                va="bottom",
                fontproperties=font_prop,
                fontsize=18,
            )

    if nrows == 1:
        meta = meta_rows[0]
        fig.suptitle(
            f"基于Transformer的引导图可视化 | {meta['row_label']} | 编号={meta['case_idx']} | 节点变化={meta['delta']:+d}",
            fontsize=24,
            fontproperties=font_prop,
            y=0.985,
        )
        fig.subplots_adjust(top=0.86, bottom=0.04, left=0.03, right=0.98, wspace=0.34)
    else:
        fig.suptitle("基于Transformer的引导图可视化", fontsize=24, fontproperties=font_prop, y=0.985)
        fig.subplots_adjust(top=0.92, bottom=0.02, left=0.03, right=0.98, hspace=0.24, wspace=0.34)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(args.output_png)


if __name__ == "__main__":
    main()
