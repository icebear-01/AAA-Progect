"""Evaluate baseline vs guided Hybrid A* on a directory of expert samples."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from hybrid_astar_guided import GuidedHybridAstar
from neural_astar.api.guidance_infer import infer_cost_volume
from neural_astar.datasets import ParkingGuidanceDataset


@dataclass
class Stat:
    success: int = 0
    expanded_nodes: float = 0.0
    runtime_ms: float = 0.0
    path_length: float = 0.0
    total: int = 0

    def update(self, ok: bool, expanded: int, runtime_ms: float, path_len: float) -> None:
        self.total += 1
        self.success += int(ok)
        self.expanded_nodes += float(expanded)
        self.runtime_ms += float(runtime_ms)
        self.path_length += float(path_len)

    def summary(self) -> Dict[str, float]:
        denom = max(1, self.total)
        return {
            "success_rate": self.success / denom,
            "expanded_nodes": self.expanded_nodes / denom,
            "runtime_ms": self.runtime_ms / denom,
            "path_length": self.path_length / denom,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate guided Hybrid A* on expert sample directory")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lambda-guidance", type=float, default=0.2)
    p.add_argument(
        "--lambda-sweep",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of lambda values to evaluate in one run.",
    )
    p.add_argument("--sweep-out", type=Path, default=None, help="Optional CSV output for sweep summary.")
    p.add_argument(
        "--guidance-source",
        type=str,
        default="auto",
        choices=["auto", "ckpt", "target_cost", "target_cost_orient", "zero"],
        help="Guidance source. auto prefers ckpt, then orientation target, then 2D target, then zero.",
    )
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--orientation-bins",
        type=int,
        default=1,
        help="Dataset-side orientation bins when using target_cost_orient or zero-cost volumes.",
    )
    p.add_argument(
        "--normalize-guidance-cost",
        dest="normalize_guidance_cost",
        action="store_true",
        help="Normalize guidance cost by free-space percentiles before planning.",
    )
    p.add_argument(
        "--no-normalize-guidance-cost",
        dest="normalize_guidance_cost",
        action="store_false",
        help="Disable guidance normalization before planning.",
    )
    p.add_argument("--guidance-norm-p-low", type=float, default=5.0)
    p.add_argument("--guidance-norm-p-high", type=float, default=95.0)
    p.add_argument("--guidance-clip-low", type=float, default=0.05)
    p.add_argument("--guidance-clip-high", type=float, default=0.95)
    p.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Sigmoid-logit temperature for planner-side guidance shaping. <1 sharpens.",
    )
    p.add_argument(
        "--guidance-power",
        type=float,
        default=1.0,
        help="Planner-side power transform on normalized guidance cost. >1 emphasizes low-cost corridors.",
    )
    p.add_argument(
        "--guidance-bonus-threshold",
        type=float,
        default=0.5,
        help="Only used in heuristic_bonus mode. Costs below this threshold receive queue-priority bonus.",
    )

    p.add_argument("--yaw-bins", type=int, default=72)
    p.add_argument("--n-steer", type=int, default=5)
    p.add_argument("--motion-step", type=float, default=0.5)
    p.add_argument("--primitive-length", type=float, default=2.5)
    p.add_argument("--wheel-base", type=float, default=2.7)
    p.add_argument("--max-steer", type=float, default=0.60)
    p.add_argument(
        "--guidance-integration-mode",
        type=str,
        default="g_cost",
        choices=["g_cost", "heuristic_bias", "heuristic_bonus"],
        help="How guidance affects planner search: accumulate into g-cost, add raw heuristic bias, or only reward low-cost regions.",
    )
    p.add_argument("--allow-reverse", dest="allow_reverse", action="store_true")
    p.add_argument("--no-allow-reverse", dest="allow_reverse", action="store_false")
    p.add_argument("--strict-goal-pose", action="store_true")
    p.add_argument("--use-rs-shot", action="store_true")
    p.add_argument("--goal-tolerance-yaw-deg", type=float, default=5.0)
    p.add_argument("--max-expansions", type=int, default=40000)
    p.set_defaults(normalize_guidance_cost=True)
    p.set_defaults(allow_reverse=True)
    return p.parse_args()


def resolve_guidance_source(args: argparse.Namespace, sample: dict) -> str:
    if args.guidance_source != "auto":
        return args.guidance_source
    if args.ckpt is not None:
        return "ckpt"
    if args.orientation_bins > 1 and "target_cost_orient" in sample:
        return "target_cost_orient"
    if "target_cost" in sample:
        return "target_cost"
    return "zero"


def load_guidance(
    args: argparse.Namespace,
    sample: dict,
    source: str,
    occ: np.ndarray,
    start_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
) -> np.ndarray:
    start_xy = (int(round(start_pose[0])), int(round(start_pose[1])))
    goal_xy = (int(round(goal_pose[0])), int(round(goal_pose[1])))

    if source == "ckpt":
        if args.ckpt is None:
            raise ValueError("guidance-source=ckpt requires --ckpt")
        volume = infer_cost_volume(
            ckpt_path=args.ckpt,
            occ_map_numpy=occ,
            start_xy=start_xy,
            goal_xy=goal_xy,
            start_yaw=start_pose[2],
            goal_yaw=goal_pose[2],
            device=args.device,
        )
        return volume[0] if volume.shape[0] == 1 else volume
    if source == "target_cost_orient":
        if "target_cost_orient" not in sample:
            raise KeyError("Dataset sample does not contain target_cost_orient")
        return sample["target_cost_orient"].numpy()
    if source == "target_cost":
        return sample["target_cost"].numpy()[0]
    if args.orientation_bins > 1:
        return np.zeros((args.orientation_bins,) + occ.shape, dtype=np.float32)
    return np.zeros_like(occ, dtype=np.float32)


def write_sweep_csv(out_path: Path, baseline: Stat, guided_stats: Dict[float, Stat]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_summary = baseline.summary()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_guidance",
                "success_rate",
                "expanded_nodes",
                "runtime_ms",
                "path_length",
                "baseline_success_rate",
                "baseline_expanded_nodes",
                "baseline_runtime_ms",
                "baseline_path_length",
            ],
        )
        writer.writeheader()
        for lam in sorted(guided_stats.keys()):
            guided_summary = guided_stats[lam].summary()
            writer.writerow(
                {
                    "lambda_guidance": lam,
                    **guided_summary,
                    "baseline_success_rate": baseline_summary["success_rate"],
                    "baseline_expanded_nodes": baseline_summary["expanded_nodes"],
                    "baseline_runtime_ms": baseline_summary["runtime_ms"],
                    "baseline_path_length": baseline_summary["path_length"],
                }
            )


def choose_best_lambda(guided_stats: Dict[float, Stat]) -> tuple[float, Dict[str, float]]:
    ranked = []
    for lam, stat in guided_stats.items():
        summary = stat.summary()
        ranked.append(
            (
                -summary["success_rate"],
                summary["expanded_nodes"],
                summary["runtime_ms"],
                summary["path_length"],
                lam,
                summary,
            )
        )
    ranked.sort()
    _, _, _, _, best_lambda, best_summary = ranked[0]
    return float(best_lambda), best_summary


def main() -> None:
    args = parse_args()
    ds = ParkingGuidanceDataset(args.data_dir, orientation_bins=args.orientation_bins)
    planner = GuidedHybridAstar(
        yaw_bins=args.yaw_bins,
        n_steer=args.n_steer,
        motion_step=args.motion_step,
        primitive_length=args.primitive_length,
        wheel_base=args.wheel_base,
        max_steer=args.max_steer,
        guidance_integration_mode=args.guidance_integration_mode,
        normalize_guidance_cost=args.normalize_guidance_cost,
        guidance_norm_p_low=args.guidance_norm_p_low,
        guidance_norm_p_high=args.guidance_norm_p_high,
        guidance_clip_low=args.guidance_clip_low,
        guidance_clip_high=args.guidance_clip_high,
        guidance_temperature=args.guidance_temperature,
        guidance_power=args.guidance_power,
        guidance_bonus_threshold=args.guidance_bonus_threshold,
        allow_reverse=args.allow_reverse,
        strict_goal_pose=args.strict_goal_pose,
        use_rs_shot=args.use_rs_shot,
        goal_tolerance_yaw_deg=args.goal_tolerance_yaw_deg,
    )

    lambda_values = (
        [float(x) for x in args.lambda_sweep]
        if args.lambda_sweep is not None and len(args.lambda_sweep) > 0
        else [float(args.lambda_guidance)]
    )

    indices: List[int] = list(range(len(ds)))
    if args.max_samples > 0:
        indices = indices[: int(args.max_samples)]

    baseline = Stat()
    guided_stats = {float(lam): Stat() for lam in lambda_values}

    print(f"dataset_size={len(ds)} eval_samples={len(indices)} lambda_values={lambda_values}")
    for case_id, idx in enumerate(indices):
        sample = ds[idx]
        occ = sample["occ_map"].numpy()[0]
        start_pose_np = sample["start_pose"].numpy()
        goal_pose_np = sample["goal_pose"].numpy()
        start_pose = (float(start_pose_np[0]), float(start_pose_np[1]), float(start_pose_np[2]))
        goal_pose = (float(goal_pose_np[0]), float(goal_pose_np[1]), float(goal_pose_np[2]))

        source = resolve_guidance_source(args, sample)
        guidance_cost = load_guidance(
            args=args,
            sample=sample,
            source=source,
            occ=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
        )

        base = planner.plan(
            occ_map=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost_map=guidance_cost,
            lambda_guidance=0.0,
            max_expansions=args.max_expansions,
        )
        baseline.update(base.success, base.expanded_nodes, base.runtime_ms, base.path_length)

        case_parts = [
            f"case={case_id:03d}",
            f"idx={idx:05d}",
            f"source={source}",
            f"baseline(success={base.success}, expanded={base.expanded_nodes}, runtime_ms={base.runtime_ms:.3f})",
        ]
        for lam in lambda_values:
            gd = planner.plan(
                occ_map=occ,
                start_pose=start_pose,
                goal_pose=goal_pose,
                cost_map=guidance_cost,
                lambda_guidance=float(lam),
                max_expansions=args.max_expansions,
            )
            guided_stats[float(lam)].update(
                gd.success,
                gd.expanded_nodes,
                gd.runtime_ms,
                gd.path_length,
            )
            case_parts.append(
                f"guided[{lam:.3f}]=(success={gd.success}, expanded={gd.expanded_nodes}, runtime_ms={gd.runtime_ms:.3f})"
            )
        print(" ".join(case_parts))

    baseline_summary = baseline.summary()
    print("\n=== Summary ===")
    print(
        "baseline: "
        f"success_rate={baseline_summary['success_rate']:.3f}, "
        f"expanded_nodes={baseline_summary['expanded_nodes']:.1f}, "
        f"runtime_ms={baseline_summary['runtime_ms']:.3f}, "
        f"path_length={baseline_summary['path_length']:.3f}"
    )
    for lam in lambda_values:
        guided_summary = guided_stats[float(lam)].summary()
        print(
            f"guided(lambda={lam}): "
            f"success_rate={guided_summary['success_rate']:.3f}, "
            f"expanded_nodes={guided_summary['expanded_nodes']:.1f}, "
            f"runtime_ms={guided_summary['runtime_ms']:.3f}, "
            f"path_length={guided_summary['path_length']:.3f}"
        )

    best_lambda, best_summary = choose_best_lambda(guided_stats)
    print(
        "best_lambda: "
        f"value={best_lambda}, "
        f"success_rate={best_summary['success_rate']:.3f}, "
        f"expanded_nodes={best_summary['expanded_nodes']:.1f}, "
        f"runtime_ms={best_summary['runtime_ms']:.3f}, "
        f"path_length={best_summary['path_length']:.3f}"
    )

    if args.sweep_out is not None:
        write_sweep_csv(args.sweep_out, baseline=baseline, guided_stats=guided_stats)
        print(f"sweep_csv={args.sweep_out}")


if __name__ == "__main__":
    main()
