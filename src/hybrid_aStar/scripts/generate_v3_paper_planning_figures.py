#!/usr/bin/env python3
"""Generate paper planning visualizations with the PyTorch V3 frontend model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager as fm


HYBRID_ASTAR_ROOT = Path(__file__).resolve().parents[1]
NEURAL_ASTAR_ROOT = HYBRID_ASTAR_ROOT / "model_base_astar" / "neural-astar"
NEURAL_ASTAR_SRC = NEURAL_ASTAR_ROOT / "src"
if str(NEURAL_ASTAR_SRC) not in sys.path:
    sys.path.insert(0, str(NEURAL_ASTAR_SRC))
if str(HYBRID_ASTAR_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(HYBRID_ASTAR_ROOT / "scripts"))

from hybrid_astar_guided.grid_astar import Astar8ConnStats, astar_8conn_stats  # noqa: E402
from neural_astar.api.guidance_infer import load_guidance_encoder  # noqa: E402
from neural_astar.utils.coords import clip_cost_map_with_obstacles, make_one_hot_xy  # noqa: E402
from neural_astar.utils.guidance_targets import build_clearance_input_map  # noqa: E402
from neural_astar.utils.residual_prediction import apply_residual_scale_np, decode_residual_prediction_np  # noqa: E402

import benchmark_trap_planners as TRAP  # noqa: E402


DEFAULT_DATASET = NEURAL_ASTAR_ROOT / "planning-datasets" / "data" / "street" / "mixed_064_moore_c16.npz"
DEFAULT_V3_CKPT = (
    NEURAL_ASTAR_ROOT
    / "outputs"
    / "model_guidance_street_unet_transformer_v3_finetune_v1_logged"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = HYBRID_ASTAR_ROOT / "offline_results" / "paper_v3_planning_figures_20260514"

XY = Tuple[int, int]


@dataclass
class CaseResult:
    group: str
    scene: str
    map_index: int
    start_xy: XY
    goal_xy: XY
    occ: np.ndarray
    astar: Astar8ConnStats
    v3: Astar8ConnStats


def _set_font() -> None:
    for path in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    ):
        if path.exists():
            fm.fontManager.addfont(str(path))
            name = fm.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _split_key(split: str) -> str:
    if split == "train":
        return "arr_0"
    if split == "valid":
        return "arr_4"
    if split == "test":
        return "arr_8"
    raise ValueError(f"unknown split: {split}")


def _load_street_maps(dataset: Path, split: str) -> np.ndarray:
    with np.load(dataset) as data:
        key = _split_key(split)
        if key not in data.files:
            raise KeyError(f"{dataset} does not contain {key}; available={data.files}")
        maps = np.asarray(data[key], dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"expected street maps [N,H,W], got {maps.shape}")
    return maps


def _occ_from_design(map_design: np.ndarray) -> np.ndarray:
    return (1.0 - np.asarray(map_design, dtype=np.float32)).astype(np.float32)


def _free_cells(occ: np.ndarray) -> np.ndarray:
    yy, xx = np.where(occ <= 0.5)
    return np.stack([xx, yy], axis=1)


def _sample_start_goal(occ: np.ndarray, rng: random.Random, min_dist: float) -> Tuple[XY, XY]:
    cells = _free_cells(occ)
    if len(cells) < 2:
        raise RuntimeError("map has fewer than two free cells")
    for _ in range(2000):
        s = cells[rng.randrange(len(cells))]
        g = cells[rng.randrange(len(cells))]
        if float(np.hypot(float(s[0] - g[0]), float(s[1] - g[1]))) >= min_dist:
            return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))
    raise RuntimeError("failed to sample a distant start/goal pair")


def _maybe_cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _infer_v3_maps(model: torch.nn.Module, occ: np.ndarray, start_xy: XY, goal_xy: XY, device: str) -> Dict[str, Optional[np.ndarray]]:
    h, w = occ.shape
    sx, sy = start_xy
    gx, gy = goal_xy
    start = make_one_hot_xy(sx, sy, w, h)
    goal = make_one_hot_xy(gx, gy, w, h)
    occ_t = torch.from_numpy(occ[None, None]).to(device)
    start_t = torch.from_numpy(start[None, None]).to(device)
    goal_t = torch.from_numpy(goal[None, None]).to(device)
    extra_input_t = None
    if int(getattr(model, "extra_input_channels", 0)) > 0:
        clearance_input = build_clearance_input_map(
            occ_map=occ,
            clip_distance=float(getattr(model, "clearance_input_clip_distance", 0.0)),
        )[None, None].astype(np.float32)
        extra_input_t = torch.from_numpy(clearance_input).to(device)
    yaw_zero = torch.tensor([0.0], dtype=torch.float32, device=device)

    _maybe_cuda_sync(device)
    with torch.no_grad():
        out = model(
            occ_t,
            start_t,
            goal_t,
            start_yaw=yaw_zero,
            goal_yaw=yaw_zero,
            extra_input_maps=extra_input_t,
        )
    _maybe_cuda_sync(device)

    cost = out.cost_map[0].detach().cpu().numpy().astype(np.float32)
    if str(getattr(model, "output_mode", "cost_map")) == "residual_heuristic":
        residual = cost[0] if cost.shape[0] == 1 else np.min(cost, axis=0).astype(np.float32)
        residual = decode_residual_prediction_np(
            residual,
            transform=str(getattr(model, "residual_target_transform", "none")),
        )
        scale = None
        if out.scale_map is not None:
            scale = out.scale_map[0].detach().cpu().numpy().astype(np.float32)
            scale = scale[0] if scale.shape[0] == 1 else np.min(scale, axis=0).astype(np.float32)
        confidence = None
        if out.confidence_map is not None:
            confidence = out.confidence_map[0].detach().cpu().numpy().astype(np.float32)
            confidence = confidence[0] if confidence.shape[0] == 1 else np.min(confidence, axis=0).astype(np.float32)
        return {
            "guidance_cost": None,
            "heuristic_residual_map": apply_residual_scale_np(residual, scale).astype(np.float32),
            "residual_confidence_map": confidence.astype(np.float32) if confidence is not None else None,
        }

    cost_2d = cost[0] if cost.shape[0] == 1 else np.min(cost, axis=0).astype(np.float32)
    return {
        "guidance_cost": clip_cost_map_with_obstacles(cost_2d, occ, obstacle_cost=1.0).astype(np.float32),
        "heuristic_residual_map": None,
        "residual_confidence_map": None,
    }


def _run_astar(occ: np.ndarray, start_xy: XY, goal_xy: XY) -> Astar8ConnStats:
    return astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        allow_corner_cut=False,
    )


def _run_v3(
    model: torch.nn.Module,
    occ: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    device: str,
    residual_weight: float,
) -> Astar8ConnStats:
    maps = _infer_v3_maps(model, occ, start_xy, goal_xy, device)
    return astar_8conn_stats(
        occ_map=occ,
        start_xy=start_xy,
        goal_xy=goal_xy,
        guidance_cost=maps["guidance_cost"],
        heuristic_residual_map=maps["heuristic_residual_map"],
        residual_confidence_map=maps["residual_confidence_map"],
        lambda_guidance=0.0,
        residual_weight=residual_weight,
        heuristic_mode="octile",
        heuristic_weight=1.0,
        allow_corner_cut=False,
    )


def _path_xy(path: Optional[Sequence[XY]]) -> Optional[np.ndarray]:
    if not path:
        return None
    return np.asarray(path, dtype=np.float32)


def _expanded_xy(expanded: Sequence[XY], max_points: int = 900) -> Optional[np.ndarray]:
    if not expanded:
        return None
    arr = np.asarray(expanded, dtype=np.float32)
    if len(arr) > max_points:
        step = int(math.ceil(len(arr) / max_points))
        arr = arr[::step]
    return arr


def _plot_case(ax: plt.Axes, case: CaseResult, show_scene: bool = True) -> None:
    ax.imshow(case.occ, cmap="gray_r", origin="upper", vmin=0.0, vmax=1.0)
    astar_exp = _expanded_xy(case.astar.expanded_xy)
    v3_exp = _expanded_xy(case.v3.expanded_xy)
    if astar_exp is not None:
        ax.scatter(astar_exp[:, 0], astar_exp[:, 1], s=3, c="#91c7e8", alpha=0.22, linewidths=0)
    if v3_exp is not None:
        ax.scatter(v3_exp[:, 0], v3_exp[:, 1], s=3, c="#f6a77a", alpha=0.28, linewidths=0)
    astar_path = _path_xy(case.astar.path)
    if astar_path is not None:
        ax.plot(astar_path[:, 0], astar_path[:, 1], color="#2166ac", linewidth=1.6, linestyle="--", label="A*")
    v3_path = _path_xy(case.v3.path)
    if v3_path is not None:
        ax.plot(v3_path[:, 0], v3_path[:, 1], color="#d73027", linewidth=2.2, label="V3")
    ax.scatter([case.start_xy[0]], [case.start_xy[1]], c="#1a9850", marker="o", s=42, edgecolors="white", linewidths=0.6)
    ax.scatter([case.goal_xy[0]], [case.goal_xy[1]], c="#111827", marker="*", s=70, edgecolors="white", linewidths=0.5)
    title_left = f"{case.group}" if not show_scene else f"{case.group} | {case.scene}"
    ax.set_title(
        f"{title_left}\nA* {case.astar.expanded_nodes} / V3 {case.v3.expanded_nodes}",
        fontsize=9.5,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#d1d5db")


def _save_figure(fig: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _select_by_difficulty(results: Sequence[CaseResult], per_group: int) -> List[CaseResult]:
    successful = [r for r in results if r.astar.success and r.v3.success]
    if len(successful) < per_group * 3:
        raise RuntimeError(f"only {len(successful)} successful street cases, need {per_group * 3}")
    ordered = sorted(successful, key=lambda r: r.astar.expanded_nodes)
    bins = {
        "Easy": ordered[: max(per_group, len(ordered) // 3)],
        "Medium": ordered[len(ordered) // 3 : max(2 * len(ordered) // 3, len(ordered) // 3 + per_group)],
        "Hard": ordered[2 * len(ordered) // 3 :],
    }
    selected: List[CaseResult] = []
    for group, items in bins.items():
        if len(items) <= per_group:
            picks = items
        else:
            positions = np.linspace(0, len(items) - 1, per_group + 2, dtype=int)[1:-1]
            picks = [items[int(pos)] for pos in positions]
        for item in picks[:per_group]:
            item.group = group
            selected.append(item)
    return selected


def build_street_cases(
    model: torch.nn.Module,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[CaseResult]:
    maps = _load_street_maps(args.dataset, args.split)
    results: List[CaseResult] = []
    attempts = 0
    while len(results) < args.street_candidates and attempts < args.street_candidates * 10:
        attempts += 1
        map_index = rng.randrange(len(maps))
        occ = _occ_from_design(maps[map_index])
        try:
            start_xy, goal_xy = _sample_start_goal(occ, rng, args.min_start_goal_dist)
        except RuntimeError:
            continue
        astar = _run_astar(occ, start_xy, goal_xy)
        if not astar.success:
            continue
        v3 = _run_v3(model, occ, start_xy, goal_xy, args.device, args.residual_weight)
        if not v3.success:
            continue
        results.append(
            CaseResult(
                group="",
                scene=f"street #{map_index}",
                map_index=map_index,
                start_xy=start_xy,
                goal_xy=goal_xy,
                occ=occ,
                astar=astar,
                v3=v3,
            )
        )

    selected = _select_by_difficulty(results, args.street_per_difficulty)
    rerun: List[CaseResult] = []
    for case in selected:
        astar = _run_astar(case.occ, case.start_xy, case.goal_xy)
        v3 = _run_v3(model, case.occ, case.start_xy, case.goal_xy, args.device, args.residual_weight)
        rerun.append(
            CaseResult(
                group=case.group,
                scene=case.scene,
                map_index=case.map_index,
                start_xy=case.start_xy,
                goal_xy=case.goal_xy,
                occ=case.occ,
                astar=astar,
                v3=v3,
            )
        )
    return rerun


def build_trap_cases(model: torch.nn.Module, args: argparse.Namespace, rng: random.Random) -> List[CaseResult]:
    groups = [("Easy", 4), ("Medium", 3), ("Hard", 2)]
    builders = {
        "dead_end": TRAP._build_dead_end_case,
        "bugtrap": TRAP._build_bugtrap_case,
        "offset_gate": TRAP._build_offset_gate_case,
        "comb": TRAP._build_comb_case,
    }
    cases: List[CaseResult] = []
    for trap_type in args.trap_types:
        for group, width in groups:
            if trap_type not in builders:
                raise ValueError(f"unknown trap type: {trap_type}")
            astar = None
            v3 = None
            occ = None
            start_xy = None
            goal_xy = None
            for _ in range(40):
                occ_i, start_i, goal_i = builders[trap_type](args.trap_grid_size, width, rng)
                astar_i = _run_astar(occ_i, start_i, goal_i)
                if not astar_i.success:
                    continue
                v3_i = _run_v3(model, occ_i, start_i, goal_i, args.device, args.residual_weight)
                if not v3_i.success:
                    continue
                occ, start_xy, goal_xy, astar, v3 = occ_i, start_i, goal_i, astar_i, v3_i
                break
            if occ is None or start_xy is None or goal_xy is None or astar is None or v3 is None:
                raise RuntimeError(f"failed to build successful trap case: {trap_type}/{group}")
            cases.append(
                CaseResult(
                    group=group,
                    scene=trap_type,
                    map_index=-1,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    occ=occ,
                    astar=astar,
                    v3=v3,
                )
            )
    return cases


def plot_street_gallery(cases: Sequence[CaseResult], out_path: Path, per_group: int) -> None:
    groups = ["Easy", "Medium", "Hard"]
    fig, axes = plt.subplots(len(groups), per_group, figsize=(3.25 * per_group, 9.2), facecolor="white")
    axes = np.asarray(axes).reshape(len(groups), per_group)
    for row_idx, group in enumerate(groups):
        items = [c for c in cases if c.group == group][:per_group]
        for col_idx, ax in enumerate(axes[row_idx]):
            ax.axis("off")
            if col_idx < len(items):
                _plot_case(ax, items[col_idx], show_scene=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Street Dataset: V3 Planning Across Difficulty Levels", fontsize=14, weight="bold", y=0.992)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.96))
    _save_figure(fig, out_path)


def plot_trap_gallery(cases: Sequence[CaseResult], out_path: Path) -> None:
    trap_types = list(dict.fromkeys(c.scene for c in cases))
    groups = ["Easy", "Medium", "Hard"]
    fig, axes = plt.subplots(len(trap_types), len(groups), figsize=(10.2, 3.15 * len(trap_types)), facecolor="white")
    axes = np.asarray(axes).reshape(len(trap_types), len(groups))
    for row_idx, trap_type in enumerate(trap_types):
        for col_idx, group in enumerate(groups):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            item = next((c for c in cases if c.scene == trap_type and c.group == group), None)
            if item is not None:
                _plot_case(ax, item, show_scene=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Synthetic Trap Scenes: V3 Planning Across Scene Types and Difficulty", fontsize=14, weight="bold", y=0.992)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    _save_figure(fig, out_path)


def write_case_metrics(path: Path, cases: Sequence[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "scene",
                "map_index",
                "start_x",
                "start_y",
                "goal_x",
                "goal_y",
                "astar_success",
                "v3_success",
                "astar_expanded",
                "v3_expanded",
                "astar_points",
                "v3_points",
            ]
        )
        for c in cases:
            writer.writerow(
                [
                    c.group,
                    c.scene,
                    c.map_index,
                    c.start_xy[0],
                    c.start_xy[1],
                    c.goal_xy[0],
                    c.goal_xy[1],
                    int(c.astar.success),
                    int(c.v3.success),
                    c.astar.expanded_nodes,
                    c.v3.expanded_nodes,
                    len(c.astar.path or []),
                    len(c.v3.path or []),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V3 PyTorch planning figures for paper cases.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_V3_CKPT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--street-candidates", type=int, default=72)
    parser.add_argument("--street-per-difficulty", type=int, default=3)
    parser.add_argument("--min-start-goal-dist", type=float, default=22.0)
    parser.add_argument("--residual-weight", type=float, default=1.25)
    parser.add_argument("--trap-grid-size", type=int, default=64)
    parser.add_argument(
        "--trap-types",
        nargs="+",
        default=["dead_end", "bugtrap", "offset_gate", "comb"],
        choices=["dead_end", "bugtrap", "offset_gate", "comb"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    model = load_guidance_encoder(args.ckpt, device=args.device)
    model.eval()

    street_cases = build_street_cases(model, args, rng)
    trap_cases = build_trap_cases(model, args, rng)

    plot_street_gallery(street_cases, args.output_dir / "street_v3_difficulty_gallery.png", args.street_per_difficulty)
    plot_trap_gallery(trap_cases, args.output_dir / "trap_v3_scene_difficulty_gallery.png")
    write_case_metrics(args.output_dir / "street_case_metrics.csv", street_cases)
    write_case_metrics(args.output_dir / "trap_case_metrics.csv", trap_cases)

    meta = {
        "ckpt": str(args.ckpt.resolve()),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "device": args.device,
        "seed": args.seed,
        "residual_weight": args.residual_weight,
        "street_cases": len(street_cases),
        "trap_cases": len(trap_cases),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved_output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
