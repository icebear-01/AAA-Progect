"""Visualize baseline vs guided Hybrid A* comparisons on dataset samples."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from eval_guided_hybrid_astar_demo import save_case_plot
from hybrid_astar_guided import GuidedHybridAstar
from neural_astar.api.guidance_infer import infer_cost_map, infer_cost_volume
from neural_astar.datasets import ParkingGuidanceDataset


@dataclass
class CaseLog:
    case_id: int
    idx: int
    baseline_success: bool
    baseline_expanded: int
    baseline_runtime_ms: float
    guided_success: bool
    guided_expanded: int
    guided_runtime_ms: float

    @property
    def delta_expanded(self) -> int:
        return int(self.guided_expanded - self.baseline_expanded)

    @property
    def delta_runtime_ms(self) -> float:
        return float(self.guided_runtime_ms - self.baseline_runtime_ms)


CASE_RE = re.compile(
    r"case=(?P<case>\d+)\s+idx=(?P<idx>\d+).*?"
    r"baseline\(success=(?P<bs>True|False), expanded=(?P<be>\d+), runtime_ms=(?P<br>[0-9.]+)\)\s+"
    r"guided\[(?P<lam>[0-9.]+)\]=\(success=(?P<gs>True|False), expanded=(?P<ge>\d+), runtime_ms=(?P<gr>[0-9.]+)\)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize dataset baseline vs guided Hybrid A* comparisons.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--results-log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--orientation-bins", type=int, default=8)
    p.add_argument("--yaw-bins", type=int, default=72)
    p.add_argument("--n-steer", type=int, default=5)
    p.add_argument("--motion-step", type=float, default=0.5)
    p.add_argument("--primitive-length", type=float, default=2.5)
    p.add_argument("--wheel-base", type=float, default=2.7)
    p.add_argument("--max-steer", type=float, default=0.60)
    p.add_argument("--lambda-guidance", type=float, default=0.4)
    p.add_argument("--guidance-integration-mode", type=str, default="heuristic_bonus")
    p.add_argument("--guidance-temperature", type=float, default=0.7)
    p.add_argument("--guidance-power", type=float, default=1.5)
    p.add_argument("--guidance-bonus-threshold", type=float, default=0.6)
    p.add_argument("--max-expansions", type=int, default=40000)
    p.add_argument("--smooth-iter", type=int, default=2)
    return p.parse_args()


def load_cases(log_path: Path) -> List[CaseLog]:
    text = log_path.read_text(encoding="utf-8")
    cases: List[CaseLog] = []
    for m in CASE_RE.finditer(text):
        cases.append(
            CaseLog(
                case_id=int(m.group("case")),
                idx=int(m.group("idx")),
                baseline_success=(m.group("bs") == "True"),
                baseline_expanded=int(m.group("be")),
                baseline_runtime_ms=float(m.group("br")),
                guided_success=(m.group("gs") == "True"),
                guided_expanded=int(m.group("ge")),
                guided_runtime_ms=float(m.group("gr")),
            )
        )
    if not cases:
        raise ValueError(f"No case lines parsed from {log_path}")
    return cases


def select_cases(cases: List[CaseLog]) -> Dict[str, CaseLog]:
    common = [c for c in cases if c.baseline_success and c.guided_success]
    if not common:
        raise ValueError("No common-success cases found in log")

    common_sorted = sorted(common, key=lambda c: (c.delta_expanded, c.delta_runtime_ms, c.idx))
    deltas = np.array([c.delta_expanded for c in common_sorted], dtype=np.float32)
    median_val = float(np.median(deltas))
    median_case = min(common_sorted, key=lambda c: (abs(c.delta_expanded - median_val), abs(c.delta_runtime_ms)))
    worst_case = max(common_sorted, key=lambda c: (c.delta_expanded, c.delta_runtime_ms, -c.idx))

    picks = {
        "best_improve": common_sorted[0],
        "median_case": median_case,
        "worst_regress": worst_case,
    }
    # Avoid duplicate picks by falling back to nearby neighbors.
    used = set()
    for name in list(picks.keys()):
        case = picks[name]
        if case.idx in used:
            for alt in common_sorted:
                if alt.idx not in used:
                    picks[name] = alt
                    case = alt
                    break
        used.add(case.idx)
    return picks


def build_planner(args: argparse.Namespace) -> GuidedHybridAstar:
    return GuidedHybridAstar(
        yaw_bins=args.yaw_bins,
        n_steer=args.n_steer,
        motion_step=args.motion_step,
        primitive_length=args.primitive_length,
        wheel_base=args.wheel_base,
        max_steer=args.max_steer,
        guidance_integration_mode=args.guidance_integration_mode,
        guidance_temperature=args.guidance_temperature,
        guidance_power=args.guidance_power,
        guidance_bonus_threshold=args.guidance_bonus_threshold,
    )


def save_summary_plot(cases: List[CaseLog], selected: Dict[str, CaseLog], out_path: Path) -> None:
    common = [c for c in cases if c.baseline_success and c.guided_success]
    ordered = sorted(common, key=lambda c: (c.delta_expanded, c.delta_runtime_ms, c.idx))
    x = np.arange(len(ordered), dtype=np.int32)
    delta_exp = np.array([c.delta_expanded for c in ordered], dtype=np.float32)
    delta_rt = np.array([c.delta_runtime_ms for c in ordered], dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, delta_exp, color="tab:blue", linewidth=1.8)
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("guided - baseline\nexpanded nodes")
    axes[0].set_title("Hybrid A* Comparison Summary")

    axes[1].plot(x, delta_rt, color="tab:orange", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel("guided - baseline\nruntime ms")
    axes[1].set_xlabel("Cases sorted by expansion delta")

    lookup = {c.idx: i for i, c in enumerate(ordered)}
    color_map = {
        "best_improve": "green",
        "median_case": "purple",
        "worst_regress": "red",
    }
    for label, case in selected.items():
        xi = lookup.get(case.idx)
        if xi is None:
            continue
        axes[0].scatter([xi], [case.delta_expanded], c=color_map[label], s=45, zorder=3)
        axes[1].scatter([xi], [case.delta_runtime_ms], c=color_map[label], s=45, zorder=3)
        axes[0].annotate(label, (xi, case.delta_expanded), textcoords="offset points", xytext=(5, 6), fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ds = ParkingGuidanceDataset(args.data_dir, orientation_bins=args.orientation_bins)
    cases = load_cases(args.results_log)
    selected = select_cases(cases)
    planner = build_planner(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_summary_plot(cases, selected, args.output_dir / "summary.png")

    for label, case in selected.items():
        sample = ds[case.idx]
        occ = sample["occ_map"].numpy()[0]
        start_pose_np = sample["start_pose"].numpy()
        goal_pose_np = sample["goal_pose"].numpy()
        start_pose = (float(start_pose_np[0]), float(start_pose_np[1]), float(start_pose_np[2]))
        goal_pose = (float(goal_pose_np[0]), float(goal_pose_np[1]), float(goal_pose_np[2]))
        start_xy = (int(round(start_pose[0])), int(round(start_pose[1])))
        goal_xy = (int(round(goal_pose[0])), int(round(goal_pose[1])))

        cost_volume = infer_cost_volume(
            ckpt_path=args.ckpt,
            occ_map_numpy=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            start_yaw=start_pose[2],
            goal_yaw=goal_pose[2],
            device=args.device,
        )
        cost_map_2d = infer_cost_map(
            ckpt_path=args.ckpt,
            occ_map_numpy=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            start_yaw=start_pose[2],
            goal_yaw=goal_pose[2],
            device=args.device,
        )

        res_base = planner.plan(
            occ_map=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost_map=cost_volume,
            lambda_guidance=0.0,
            max_expansions=args.max_expansions,
        )
        res_guided = planner.plan(
            occ_map=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost_map=cost_volume,
            lambda_guidance=args.lambda_guidance,
            max_expansions=args.max_expansions,
        )

        out_path = args.output_dir / f"{label}_idx{case.idx:04d}.png"
        save_case_plot(
            occ=occ,
            cost_map=cost_map_2d,
            start_xy=start_xy,
            goal_xy=goal_xy,
            res_base=res_base,
            res_guided=res_guided,
            lambda_guidance=args.lambda_guidance,
            out_path=out_path,
            smooth_iter=args.smooth_iter,
        )
        print(
            f"{label}: idx={case.idx} delta_expanded={case.delta_expanded} "
            f"delta_runtime_ms={case.delta_runtime_ms:.3f} out={out_path}"
        )


if __name__ == "__main__":
    main()
