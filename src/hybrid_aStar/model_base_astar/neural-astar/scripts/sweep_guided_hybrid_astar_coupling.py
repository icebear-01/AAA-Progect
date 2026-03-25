"""Sweep planner-side guidance coupling hyperparameters on an expert dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from hybrid_astar_guided import GuidedHybridAstar
from neural_astar.datasets import ParkingGuidanceDataset

from eval_guided_hybrid_astar_dataset import Stat, load_guidance, resolve_guidance_source


@dataclass
class PlannerConfig:
    integration_mode: str
    lambda_guidance: float
    guidance_temperature: float
    guidance_power: float
    guidance_bonus_threshold: float


@dataclass
class SweepRow:
    integration_mode: str
    lambda_guidance: float
    guidance_temperature: float
    guidance_power: float
    guidance_bonus_threshold: float
    success_rate: float
    expanded_nodes: float
    runtime_ms: float
    path_length: float
    baseline_success_rate: float
    baseline_expanded_nodes: float
    baseline_runtime_ms: float
    baseline_path_length: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep guided Hybrid A* planner coupling settings.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0, help="0 means use all samples")
    p.add_argument(
        "--guidance-source",
        type=str,
        default="auto",
        choices=["auto", "ckpt", "target_cost", "target_cost_orient", "zero"],
    )
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--orientation-bins", type=int, default=1)
    p.add_argument(
        "--integration-modes",
        type=str,
        nargs="+",
        default=["heuristic_bonus"],
        choices=["g_cost", "heuristic_bias", "heuristic_bonus"],
    )
    p.add_argument("--lambda-values", type=float, nargs="+", required=True)
    p.add_argument("--temperature-values", type=float, nargs="+", required=True)
    p.add_argument("--power-values", type=float, nargs="+", required=True)
    p.add_argument("--bonus-threshold-values", type=float, nargs="+", default=[0.6])
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=10)

    p.add_argument("--yaw-bins", type=int, default=72)
    p.add_argument("--n-steer", type=int, default=5)
    p.add_argument("--motion-step", type=float, default=0.5)
    p.add_argument("--primitive-length", type=float, default=2.5)
    p.add_argument("--wheel-base", type=float, default=2.7)
    p.add_argument("--max-steer", type=float, default=0.60)
    p.add_argument("--allow-reverse", dest="allow_reverse", action="store_true")
    p.add_argument("--no-allow-reverse", dest="allow_reverse", action="store_false")
    p.add_argument("--strict-goal-pose", action="store_true")
    p.add_argument("--use-rs-shot", action="store_true")
    p.add_argument("--goal-tolerance-yaw-deg", type=float, default=5.0)
    p.add_argument("--max-expansions", type=int, default=40000)

    p.add_argument(
        "--normalize-guidance-cost",
        dest="normalize_guidance_cost",
        action="store_true",
    )
    p.add_argument(
        "--no-normalize-guidance-cost",
        dest="normalize_guidance_cost",
        action="store_false",
    )
    p.add_argument("--guidance-norm-p-low", type=float, default=5.0)
    p.add_argument("--guidance-norm-p-high", type=float, default=95.0)
    p.add_argument("--guidance-clip-low", type=float, default=0.05)
    p.add_argument("--guidance-clip-high", type=float, default=0.95)
    p.set_defaults(normalize_guidance_cost=True)
    p.set_defaults(allow_reverse=True)
    return p.parse_args()


def build_planner(args: argparse.Namespace, cfg: PlannerConfig) -> GuidedHybridAstar:
    return GuidedHybridAstar(
        yaw_bins=args.yaw_bins,
        n_steer=args.n_steer,
        motion_step=args.motion_step,
        primitive_length=args.primitive_length,
        wheel_base=args.wheel_base,
        max_steer=args.max_steer,
        guidance_integration_mode=cfg.integration_mode,
        normalize_guidance_cost=args.normalize_guidance_cost,
        guidance_norm_p_low=args.guidance_norm_p_low,
        guidance_norm_p_high=args.guidance_norm_p_high,
        guidance_clip_low=args.guidance_clip_low,
        guidance_clip_high=args.guidance_clip_high,
        guidance_temperature=cfg.guidance_temperature,
        guidance_power=cfg.guidance_power,
        guidance_bonus_threshold=cfg.guidance_bonus_threshold,
        allow_reverse=args.allow_reverse,
        strict_goal_pose=args.strict_goal_pose,
        use_rs_shot=args.use_rs_shot,
        goal_tolerance_yaw_deg=args.goal_tolerance_yaw_deg,
    )


def enumerate_configs(args: argparse.Namespace) -> List[PlannerConfig]:
    configs: List[PlannerConfig] = []
    thresholds = [float(x) for x in args.bonus_threshold_values]
    for mode in args.integration_modes:
        use_thresholds = thresholds if mode == "heuristic_bonus" else [0.6]
        for lam in args.lambda_values:
            for temp in args.temperature_values:
                for power in args.power_values:
                    for threshold in use_thresholds:
                        configs.append(
                            PlannerConfig(
                                integration_mode=str(mode),
                                lambda_guidance=float(lam),
                                guidance_temperature=float(temp),
                                guidance_power=float(power),
                                guidance_bonus_threshold=float(threshold),
                            )
                        )
    return configs


def rank_key(row: SweepRow) -> Tuple[float, float, float, float]:
    return (-row.success_rate, row.expanded_nodes, row.runtime_ms, row.path_length)


def write_rows(out_csv: Path, rows: Sequence[SweepRow]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(rows[0]).keys()) if rows else list(SweepRow.__dataclass_fields__.keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    ds = ParkingGuidanceDataset(args.data_dir, orientation_bins=args.orientation_bins)
    indices = list(range(len(ds)))
    if args.max_samples > 0:
        indices = indices[: int(args.max_samples)]

    sample_cache: List[Tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float], np.ndarray]] = []
    for idx in indices:
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
        sample_cache.append((occ, start_pose, goal_pose, guidance_cost))

    baseline_planner = build_planner(
        args,
        PlannerConfig(
            integration_mode="g_cost",
            lambda_guidance=0.0,
            guidance_temperature=1.0,
            guidance_power=1.0,
            guidance_bonus_threshold=0.6,
        ),
    )
    baseline = Stat()
    for occ, start_pose, goal_pose, guidance_cost in sample_cache:
        res = baseline_planner.plan(
            occ_map=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost_map=guidance_cost,
            lambda_guidance=0.0,
            max_expansions=args.max_expansions,
        )
        baseline.update(res.success, res.expanded_nodes, res.runtime_ms, res.path_length)
    baseline_summary = baseline.summary()
    print(
        "baseline "
        f"success_rate={baseline_summary['success_rate']:.3f} "
        f"expanded_nodes={baseline_summary['expanded_nodes']:.1f} "
        f"runtime_ms={baseline_summary['runtime_ms']:.3f} "
        f"path_length={baseline_summary['path_length']:.3f}"
    )

    rows: List[SweepRow] = []
    configs = enumerate_configs(args)
    print(f"configs={len(configs)} eval_samples={len(sample_cache)}")
    for cfg_id, cfg in enumerate(configs, start=1):
        planner = build_planner(args, cfg)
        stat = Stat()
        for occ, start_pose, goal_pose, guidance_cost in sample_cache:
            res = planner.plan(
                occ_map=occ,
                start_pose=start_pose,
                goal_pose=goal_pose,
                cost_map=guidance_cost,
                lambda_guidance=cfg.lambda_guidance,
                max_expansions=args.max_expansions,
            )
            stat.update(res.success, res.expanded_nodes, res.runtime_ms, res.path_length)
        summary = stat.summary()
        row = SweepRow(
            integration_mode=cfg.integration_mode,
            lambda_guidance=cfg.lambda_guidance,
            guidance_temperature=cfg.guidance_temperature,
            guidance_power=cfg.guidance_power,
            guidance_bonus_threshold=cfg.guidance_bonus_threshold,
            success_rate=summary["success_rate"],
            expanded_nodes=summary["expanded_nodes"],
            runtime_ms=summary["runtime_ms"],
            path_length=summary["path_length"],
            baseline_success_rate=baseline_summary["success_rate"],
            baseline_expanded_nodes=baseline_summary["expanded_nodes"],
            baseline_runtime_ms=baseline_summary["runtime_ms"],
            baseline_path_length=baseline_summary["path_length"],
        )
        rows.append(row)
        print(
            f"[{cfg_id:03d}/{len(configs):03d}] "
            f"mode={cfg.integration_mode} lambda={cfg.lambda_guidance:.3f} "
            f"temp={cfg.guidance_temperature:.3f} power={cfg.guidance_power:.3f} "
            f"thr={cfg.guidance_bonus_threshold:.3f} "
            f"success={row.success_rate:.3f} expanded={row.expanded_nodes:.1f} "
            f"runtime={row.runtime_ms:.3f}"
        )

    rows.sort(key=rank_key)
    write_rows(args.out_csv, rows)
    print(f"wrote_csv={args.out_csv}")
    print("top_configs:")
    for row in rows[: max(1, args.top_k)]:
        print(
            f"mode={row.integration_mode} lambda={row.lambda_guidance:.3f} "
            f"temp={row.guidance_temperature:.3f} power={row.guidance_power:.3f} "
            f"thr={row.guidance_bonus_threshold:.3f} "
            f"success={row.success_rate:.3f} expanded={row.expanded_nodes:.1f} "
            f"runtime={row.runtime_ms:.3f} path={row.path_length:.3f}"
        )


if __name__ == "__main__":
    main()
