"""Plot multiple Chinese planning cases in a single paper-style figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from plot_case_compare_cn import (
    _infer_residual_map,
    _load_dataset,
    _load_sample_by_case_idx,
    _onehot_xy,
    _panel_title,
    _plot_panel,
    _run_astar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a multi-case Chinese comparison figure.")
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
    parser.add_argument("--dpi", type=int, default=350)
    parser.set_defaults(allow_corner_cut=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.row_labels is not None and len(args.row_labels) not in {0, len(args.case_idxs)}:
        raise ValueError("--row-labels length must match --case-idxs length")

    if args.row_labels:
        row_labels = list(args.row_labels)
    else:
        row_labels = [f"场景{i + 1}" for i in range(len(args.case_idxs))]

    dataset = _load_dataset(args)
    for case_idx in args.case_idxs:
        if case_idx < 0 or case_idx >= len(dataset):
            raise IndexError(f"case_idx out of range: {case_idx} not in [0, {len(dataset) - 1}]")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_guidance_encoder(args.ckpt, device=device)
    font_prop = font_manager.FontProperties(fname=str(args.font_path))
    plt.rcParams["axes.unicode_minus"] = False

    nrows = len(args.case_idxs)
    fig, axes = plt.subplots(nrows, 3, figsize=(17.8, 5.7 * nrows))
    axes = np.atleast_2d(axes)

    planners_meta = []
    for row_idx, case_idx in enumerate(args.case_idxs):
        row_dataset = _load_dataset(args)
        sample = _load_sample_by_case_idx(row_dataset, case_idx)
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

        row_planners = [
            ("传统A*", traditional),
            ("改进A*", improved),
            ("本文方法", learned),
        ]
        for col_idx, (label, planner) in enumerate(row_planners):
            _plot_panel(
                axes[row_idx, col_idx],
                occ,
                opt_traj,
                start_xy,
                goal_xy,
                planner,
                _panel_title(label, planner),
                font_prop,
            )

        learned_delta = learned.stats.expanded_nodes - improved.stats.expanded_nodes
        row_title = f"{row_labels[row_idx]} | 编号={case_idx} | 本文方法相对改进A*节点变化={learned_delta:+d}"
        axes[row_idx, 1].text(
            0.5,
            1.18,
            row_title,
            ha="center",
            va="bottom",
            transform=axes[row_idx, 1].transAxes,
            fontproperties=font_prop,
            fontsize=17,
        )
        planners_meta.append((case_idx, learned_delta))

    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.03, right=0.99, hspace=0.24, wspace=0.18)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(args.output_png)


if __name__ == "__main__":
    main()
