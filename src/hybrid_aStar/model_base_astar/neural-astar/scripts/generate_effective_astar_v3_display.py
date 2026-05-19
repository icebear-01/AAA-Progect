"""Generate paper display panels: A* expansion, V3 expansion, effective guidance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors, font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageDraw, ImageFont

from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import PlanningNPZGuidanceDataset
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from plot_case_compare_cn import (
    _expanded_heatmap,
    _infer_residual_map,
    _load_sample_by_case_idx,
    _onehot_xy,
    _run_astar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--case-idxs", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--map-resolution", type=float, default=1.0, help="Real-world distance per grid cell.")
    parser.add_argument("--length-unit", type=str, default="cell", help="Unit label for displayed path length.")
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--residual-confidence-kernel", type=int, default=3)
    parser.add_argument("--residual-confidence-strength", type=float, default=0.75)
    parser.add_argument("--residual-confidence-min", type=float, default=0.1)
    parser.add_argument("--v3-clearance-weight", type=float, default=0.0)
    parser.add_argument("--v3-clearance-safe-distance", type=float, default=0.0)
    parser.add_argument("--v3-clearance-power", type=float, default=2.0)
    parser.add_argument("--v3-turn-weight", type=float, default=0.0)
    parser.add_argument(
        "--v3-turn-integration-mode",
        type=str,
        default="state_g_cost",
        choices=["state_g_cost", "local_g_cost", "priority_bias"],
    )
    parser.add_argument(
        "--v3-clearance-integration-mode",
        type=str,
        default="g_cost",
        choices=["g_cost", "heuristic_bias", "priority_tie_break"],
    )
    parser.add_argument("--dpi", type=int, default=430)
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    return parser.parse_args()


def _add_colorbar(ax: plt.Axes, norm, cmap, label: str, font_prop) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.4%", pad=0.055)
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.set_ylabel(label, fontproperties=font_prop, fontsize=13)
    cbar.ax.tick_params(labelsize=10)


def _draw_expansion_panel(
    ax: plt.Axes,
    occ: np.ndarray,
    heat: np.ndarray,
    planner,
    start_xy: tuple[int, int],
    goal_xy: tuple[int, int],
    title: str,
    path_color: str,
    font_prop,
    map_resolution: float,
    length_unit: str,
) -> None:
    ax.imshow(1.0 - occ, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if float(heat.max()) > 0.0:
        ax.imshow(heat, cmap="magma", alpha=0.58, vmin=0.0, vmax=1.0, interpolation="nearest")
    if planner.stats.path is not None and len(planner.stats.path) > 1:
        xs = [pt[0] for pt in planner.stats.path]
        ys = [pt[1] for pt in planner.stats.path]
        ax.plot(xs, ys, color=path_color, linewidth=2.25, alpha=0.98)
    ax.scatter([start_xy[0]], [start_xy[1]], c="#22c55e", s=52, marker="o", edgecolors="white", linewidths=0.7)
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="#D55E00", s=62, marker="x", linewidths=1.5)
    ax.set_title(
        f"{title}\n扩展节点={planner.stats.expanded_nodes}  路径长度={planner.path_length * map_resolution:.2f} {length_unit}",
        fontproperties=font_prop,
        fontsize=15,
        pad=9,
    )
    ax.set_axis_off()


def _load_samples_by_case_idx(dataset: PlanningNPZGuidanceDataset, case_idxs: list[int]) -> dict[int, dict]:
    """Replay stochastic dataset sampling once and keep the requested cases."""
    targets = sorted(set(int(idx) for idx in case_idxs))
    if not targets:
        return {}
    if targets[0] < 0 or targets[-1] >= len(dataset):
        raise IndexError(f"case_idx out of range: requested {targets}, dataset length={len(dataset)}")

    samples = {}
    target_pos = 0
    for idx in range(targets[-1] + 1):
        sample = dataset[idx]
        if idx == targets[target_pos]:
            samples[idx] = sample
            target_pos += 1
            if target_pos >= len(targets):
                break
    missing = [idx for idx in targets if idx not in samples]
    if missing:
        raise RuntimeError(f"failed to load samples for case_idx={missing}")
    return samples


def main() -> None:
    args = parse_args()
    if float(args.map_resolution) <= 0.0:
        raise ValueError(f"--map-resolution must be positive, got {args.map_resolution}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_prop = font_manager.FontProperties(fname=str(args.font_path))
    plt.rcParams["axes.unicode_minus"] = False

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = load_guidance_encoder(args.ckpt, device=device)
    clip = float(getattr(model, "clearance_input_clip_distance", 0.0))
    scene_cmap = colors.ListedColormap(["#161616", "#D8D8D8"])
    eff_cmap = plt.colormaps["cividis"].copy()
    eff_cmap.set_bad("#111111")
    summary_rows = []
    created = []

    # The dataset samples starts from an RNG. Replay once from index 0 so figures
    # match the batch-evaluation CSV exactly, without redoing the replay per case.
    dataset = PlanningNPZGuidanceDataset(
        npz_path=args.data_npz,
        split=args.split,
        orientation_bins=1,
        clearance_input_clip_distance=clip,
    )
    samples = _load_samples_by_case_idx(dataset, list(args.case_idxs))

    for case_idx in args.case_idxs:
        sample = samples[int(case_idx)]
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])
        pred_residual, learned_confidence_map = _infer_residual_map(model, sample, device=device)
        conf = resolve_residual_confidence_map(
            mode="learned_spike",
            occ_map=occ,
            residual_map=pred_residual,
            learned_confidence_map=learned_confidence_map,
            kernel_size=int(args.residual_confidence_kernel),
            strength=float(args.residual_confidence_strength),
            min_confidence=float(args.residual_confidence_min),
        )
        effective = (float(args.residual_weight) * pred_residual * np.clip(conf, 0.0, 1.0)).astype(np.float32)

        astar = _run_astar(
            occ,
            start_xy,
            goal_xy,
            heuristic_mode="euclidean",
            diagonal_cost=float(np.sqrt(2.0)),
            allow_corner_cut=True,
        )
        v3 = _run_astar(
            occ,
            start_xy,
            goal_xy,
            heuristic_mode="octile",
            heuristic_residual_map=pred_residual,
            residual_confidence_map=conf,
            residual_weight=float(args.residual_weight),
            clearance_weight=float(args.v3_clearance_weight),
            clearance_safe_distance=float(args.v3_clearance_safe_distance),
            clearance_power=float(args.v3_clearance_power),
            clearance_integration_mode=str(args.v3_clearance_integration_mode),
            turn_weight=float(args.v3_turn_weight),
            turn_integration_mode=str(args.v3_turn_integration_mode),
            diagonal_cost=float(np.sqrt(2.0)),
            allow_corner_cut=True,
        )

        height, width = occ.shape
        astar_heat = _expanded_heatmap(astar.stats, height, width)
        v3_heat = _expanded_heatmap(v3.stats, height, width)
        eff_mask = np.ma.masked_where(occ > 0.5, effective)
        eff_vmax = float(np.percentile(effective[effective > 0.0], 98)) if np.any(effective > 0.0) else 1.0
        eff_norm = colors.Normalize(vmin=0.0, vmax=max(eff_vmax, 1e-6))

        fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))
        _draw_expansion_panel(
            axes[0],
            occ,
            astar_heat,
            astar,
            start_xy,
            goal_xy,
            "原始A*扩展图",
            "#0072B2",
            font_prop,
            float(args.map_resolution),
            str(args.length_unit),
        )
        _draw_expansion_panel(
            axes[1],
            occ,
            v3_heat,
            v3,
            start_xy,
            goal_xy,
            "V3改进后扩展图",
            "#009E73",
            font_prop,
            float(args.map_resolution),
            str(args.length_unit),
        )

        axes[2].imshow(1.0 - occ, cmap=scene_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[2].imshow(eff_mask, cmap=eff_cmap, norm=eff_norm, alpha=0.92, interpolation="nearest")
        axes[2].scatter([start_xy[0]], [start_xy[1]], c="#22c55e", s=52, marker="o", edgecolors="white", linewidths=0.7)
        axes[2].scatter([goal_xy[0]], [goal_xy[1]], c="#D55E00", s=62, marker="x", linewidths=1.5)
        axes[2].set_title("有效引导图（送入A*）", fontproperties=font_prop, fontsize=15, pad=9)
        axes[2].set_axis_off()
        _add_colorbar(axes[2], eff_norm, eff_cmap, "有效引导值", font_prop)

        exp_red = (astar.stats.expanded_nodes - v3.stats.expanded_nodes) / astar.stats.expanded_nodes * 100.0
        len_delta = v3.path_length - astar.path_length
        astar_real_length = astar.path_length * float(args.map_resolution)
        v3_real_length = v3.path_length * float(args.map_resolution)
        real_len_delta = len_delta * float(args.map_resolution)
        fig.suptitle(
            f"编号={case_idx:04d} | A*={astar.stats.expanded_nodes} -> V3={v3.stats.expanded_nodes} | "
            f"节点减少={exp_red:.1f}% | 长度变化={real_len_delta:+.2f} {args.length_unit}",
            fontproperties=font_prop,
            fontsize=17,
            y=0.99,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        out_path = args.output_dir / f"effective_astar_v3_idx{case_idx:04d}_w100_cn.png"
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        created.append(out_path)
        row = {
            "idx": case_idx,
            "astar_expanded": astar.stats.expanded_nodes,
            "v3_expanded": v3.stats.expanded_nodes,
            "expanded_reduction_pct": exp_red,
            "astar_path_length": astar.path_length,
            "v3_path_length": v3.path_length,
            "length_delta": len_delta,
            "map_resolution": float(args.map_resolution),
            "length_unit": str(args.length_unit),
            "astar_real_length": astar_real_length,
            "v3_real_length": v3_real_length,
            "real_length_delta": real_len_delta,
            "image": str(out_path),
        }
        summary_rows.append(row)
        print(out_path, flush=True)

    summary_path = args.output_dir / "effective_astar_v3_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    thumb_w = 1100
    label_h = 52
    pad = 16
    try:
        pil_font = ImageFont.truetype(str(args.font_path), 25)
    except Exception:
        pil_font = ImageFont.load_default()

    thumbs = []
    for image_path, row in zip(created, summary_rows):
        image = Image.open(image_path).convert("RGB")
        scale = thumb_w / image.width
        thumb_h = int(image.height * scale)
        image = image.resize((thumb_w, thumb_h), Image.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        text = (
            f"idx{int(row['idx']):04d}  A*: {int(row['astar_expanded'])} -> "
            f"V3: {int(row['v3_expanded'])}  减少 {float(row['expanded_reduction_pct']):.1f}%  "
            f"长度变化 {float(row['real_length_delta']):+.2f} {row['length_unit']}"
        )
        draw.text((10, 10), text, fill=(20, 20, 20), font=pil_font)
        canvas.paste(image, (0, label_h))
        thumbs.append(canvas)

    cell_h = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (thumb_w + 2 * pad, len(thumbs) * cell_h + (len(thumbs) + 1) * pad), "white")
    for i, image in enumerate(thumbs):
        sheet.paste(image, (pad, pad + i * (cell_h + pad)))
    sheet_path = args.output_dir / "effective_astar_v3_contact_sheet_w100_cn.png"
    sheet.save(sheet_path)
    print(f"summary {summary_path}")
    print(f"contact_sheet {sheet_path}")


if __name__ == "__main__":
    main()
