"""Screen cases where V3 is both efficient and farther from obstacles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import PlanningNPZGuidanceDataset
from neural_astar.utils.residual_confidence import resolve_residual_confidence_map
from plot_case_compare_cn import _infer_residual_map, _onehot_xy, _run_astar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--min-exp-reduction", type=float, default=50.0)
    parser.add_argument("--max-length-delta", type=float, default=2.0)
    parser.add_argument("--min-mean-clearance-gain", type=float, default=0.5)
    return parser.parse_args()


def _clearance_stats(dist_map: np.ndarray, path: list[tuple[int, int]] | None) -> tuple[float, float]:
    if not path:
        return 0.0, 0.0
    vals = np.asarray([dist_map[y, x] for x, y in path], dtype=np.float32)
    return float(vals.min()), float(vals.mean())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_guidance_encoder(args.ckpt, device=device)
    clip = float(getattr(model, "clearance_input_clip_distance", 0.0))
    dataset = PlanningNPZGuidanceDataset(
        npz_path=args.data_npz,
        split=args.split,
        orientation_bins=1,
        clearance_input_clip_distance=clip,
    )
    eval_count = min(len(dataset), int(args.max_samples)) if args.max_samples > 0 else len(dataset)

    cached = []
    for idx in range(eval_count):
        sample = dataset[idx]
        occ = sample["occ_map"].numpy()[0].astype(np.float32)
        start_xy = _onehot_xy(sample["start_map"])
        goal_xy = _onehot_xy(sample["goal_map"])
        dist_map = distance_transform_edt(occ < 0.5).astype(np.float32)
        pred_residual, learned_confidence_map = _infer_residual_map(model, sample, device=device)
        conf = resolve_residual_confidence_map(
            mode="learned_spike",
            occ_map=occ,
            residual_map=pred_residual,
            learned_confidence_map=learned_confidence_map,
            kernel_size=3,
            strength=0.75,
            min_confidence=0.1,
        )
        astar = _run_astar(
            occ,
            start_xy,
            goal_xy,
            heuristic_mode="euclidean",
            diagonal_cost=float(np.sqrt(2.0)),
            allow_corner_cut=True,
        )
        cached.append((idx, occ, start_xy, goal_xy, dist_map, pred_residual, conf, astar))
        if (idx + 1) % 50 == 0 or idx + 1 == eval_count:
            print(f"cached {idx + 1}/{eval_count}", flush=True)

    configs = [
        ("cw1_sd2_p1", 1.0, 2.0, 1.0),
        ("cw2_sd2_p1", 2.0, 2.0, 1.0),
        ("cw3_sd2_p1", 3.0, 2.0, 1.0),
        ("cw15_sd3_p2", 1.5, 3.0, 2.0),
        ("cw2_sd3_p2", 2.0, 3.0, 2.0),
        ("cw3_sd3_p2", 3.0, 3.0, 2.0),
    ]

    rows = []
    for cfg_name, cw, sd, power in configs:
        for idx, occ, start_xy, goal_xy, dist_map, pred_residual, conf, astar in cached:
            v3 = _run_astar(
                occ,
                start_xy,
                goal_xy,
                heuristic_mode="octile",
                heuristic_residual_map=pred_residual,
                residual_confidence_map=conf,
                residual_weight=float(args.residual_weight),
                clearance_weight=cw,
                clearance_safe_distance=sd,
                clearance_power=power,
                clearance_integration_mode="g_cost",
                diagonal_cost=float(np.sqrt(2.0)),
                allow_corner_cut=True,
            )
            if not astar.stats.success or not v3.stats.success:
                continue
            a_min, a_mean = _clearance_stats(dist_map, astar.stats.path)
            v_min, v_mean = _clearance_stats(dist_map, v3.stats.path)
            exp_gain = int(astar.stats.expanded_nodes - v3.stats.expanded_nodes)
            exp_red = exp_gain / astar.stats.expanded_nodes * 100.0 if astar.stats.expanded_nodes else 0.0
            len_delta = float(v3.path_length - astar.path_length)
            min_gain = float(v_min - a_min)
            mean_gain = float(v_mean - a_mean)
            score = (
                exp_gain
                + 8.0 * exp_red
                + 90.0 * max(0.0, min_gain)
                + 45.0 * max(0.0, mean_gain)
                - 80.0 * max(0.0, len_delta)
            )
            rows.append(
                {
                    "idx": idx,
                    "config": cfg_name,
                    "clearance_weight": cw,
                    "clearance_safe_distance": sd,
                    "clearance_power": power,
                    "start_x": start_xy[0],
                    "start_y": start_xy[1],
                    "goal_x": goal_xy[0],
                    "goal_y": goal_xy[1],
                    "astar_expanded": astar.stats.expanded_nodes,
                    "v3_expanded": v3.stats.expanded_nodes,
                    "exp_gain": exp_gain,
                    "exp_red_pct": exp_red,
                    "astar_path_length": astar.path_length,
                    "v3_path_length": v3.path_length,
                    "len_delta": len_delta,
                    "astar_min_clearance": a_min,
                    "v3_min_clearance": v_min,
                    "min_clearance_gain": min_gain,
                    "astar_mean_clearance": a_mean,
                    "v3_mean_clearance": v_mean,
                    "mean_clearance_gain": mean_gain,
                    "score": score,
                }
            )
        print(f"finished {cfg_name}", flush=True)

    all_path = args.output_dir / "clearance_screen_test400_all_configs.csv"
    with all_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    strict = [
        row
        for row in rows
        if row["exp_red_pct"] >= float(args.min_exp_reduction)
        and row["len_delta"] <= float(args.max_length_delta)
        and row["mean_clearance_gain"] >= float(args.min_mean_clearance_gain)
        and row["v3_min_clearance"] >= max(1.4142135, row["astar_min_clearance"])
    ]
    strict.sort(key=lambda row: row["score"], reverse=True)
    strict_path = args.output_dir / "clearance_screen_test400_strict.csv"
    with strict_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(strict)

    print(f"all {all_path}")
    print(f"strict {strict_path} count={len(strict)}")
    for row in strict[:20]:
        print(
            {
                key: row[key]
                for key in [
                    "idx",
                    "config",
                    "astar_expanded",
                    "v3_expanded",
                    "exp_red_pct",
                    "len_delta",
                    "astar_min_clearance",
                    "v3_min_clearance",
                    "astar_mean_clearance",
                    "v3_mean_clearance",
                    "mean_clearance_gain",
                    "score",
                ]
            }
        )


if __name__ == "__main__":
    main()
