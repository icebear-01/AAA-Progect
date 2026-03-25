"""Generate extra goal/policy/dist tuples for high-value training maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from skimage.measure import label

ROOT = Path(__file__).resolve().parents[1]
PLANNING_DATASETS_DIR = ROOT / "planning-datasets"
if str(PLANNING_DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNING_DATASETS_DIR))

from planning_datasets_utils import dijkstra_dist, extract_policy, get_mechanism  # noqa: E402

from neural_astar.utils.guidance_targets import build_residual_heuristic_maps
from neural_astar.utils.guidance_training import load_hard_case_indices


def _goal_candidates(
    map_design: np.ndarray,
    *,
    edge_size: int,
) -> List[Tuple[int, int]]:
    free_mask = np.asarray(map_design, dtype=np.float32) > 0.5
    limage = label(free_mask.astype(np.uint8), background=0, connectivity=1)
    num_pixels = np.bincount(limage.flatten())
    num_pixels[0] = 0
    if np.max(num_pixels) > 0:
        candidate_mask = limage == int(np.argmax(num_pixels))
    else:
        candidate_mask = free_mask
    if int(edge_size) > 0:
        edge_mask = np.zeros_like(candidate_mask, dtype=bool)
        edge_mask[:edge_size, :] = True
        edge_mask[-edge_size:, :] = True
        edge_mask[:, :edge_size] = True
        edge_mask[:, -edge_size:] = True
        if np.any(candidate_mask & edge_mask):
            candidate_mask = candidate_mask & edge_mask
    ys, xs = np.where(candidate_mask)
    return [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]


def _goal_score(map_design: np.ndarray, goal_rc: Tuple[int, int]) -> float:
    occ_map = (1.0 - np.asarray(map_design, dtype=np.float32)).astype(np.float32)
    goal_y, goal_x = int(goal_rc[0]), int(goal_rc[1])
    _, _, residual = build_residual_heuristic_maps(
        occ_map=occ_map,
        goal_xy=(goal_x, goal_y),
    )
    free_vals = residual[np.asarray(map_design, dtype=np.float32) > 0.5]
    if free_vals.size == 0:
        return 0.0
    return float(np.mean(free_vals) + np.percentile(free_vals, 95.0))


def _one_hot_goal(goal_rc: Tuple[int, int], shape_hw: Tuple[int, int]) -> np.ndarray:
    goal_map = np.zeros((1, int(shape_hw[0]), int(shape_hw[1])), dtype=np.float32)
    goal_map[0, int(goal_rc[0]), int(goal_rc[1])] = 1.0
    return goal_map


def _existing_goal_rc(goal_map: np.ndarray) -> Tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(goal_map[0])), goal_map.shape[-2:])
    return int(y), int(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate extra high-value goal replay NPZ.")
    p.add_argument("--train-npz", type=Path, required=True)
    p.add_argument("--priority-csv", type=Path, required=True)
    p.add_argument("--priority-column", type=str, default="priority_score")
    p.add_argument("--top-fraction", type=float, default=0.15)
    p.add_argument("--min-priority", type=float, default=0.0)
    p.add_argument("--max-maps", type=int, default=256)
    p.add_argument("--goals-per-map", type=int, default=2)
    p.add_argument("--candidate-goals", type=int, default=12)
    p.add_argument("--edge-ratio", type=float, default=0.25)
    p.add_argument("--mechanism", type=str, default="moore")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-npz", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.goals_per_map <= 0:
        raise ValueError(f"--goals-per-map must be positive, got {args.goals_per_map}")
    if args.candidate_goals <= 0:
        raise ValueError(f"--candidate-goals must be positive, got {args.candidate_goals}")
    if args.edge_ratio < 0.0 or args.edge_ratio > 0.5:
        raise ValueError(f"--edge-ratio must be in [0, 0.5], got {args.edge_ratio}")

    mechanism = get_mechanism(args.mechanism)
    if getattr(mechanism, "num_orient", 1) != 1:
        raise ValueError("This replay generator currently supports orientation-free grid datasets only.")
    rng = np.random.default_rng(args.seed)

    selected_indices = load_hard_case_indices(
        args.priority_csv,
        delta_column=args.priority_column,
        top_fraction=args.top_fraction,
        min_delta=args.min_priority,
        max_count=args.max_maps,
    )
    if not selected_indices:
        raise RuntimeError("No source maps selected from priority CSV.")

    with np.load(args.train_npz) as data:
        map_designs = np.asarray(data["arr_0"], dtype=np.float32)
        base_goal_maps = np.asarray(data["arr_1"], dtype=np.float32)

    extra_map_designs: List[np.ndarray] = []
    extra_goal_maps: List[np.ndarray] = []
    extra_opt_policies: List[np.ndarray] = []
    extra_opt_dists: List[np.ndarray] = []
    extra_source_indices: List[int] = []
    extra_goal_scores: List[float] = []

    height, width = map_designs.shape[1:]
    edge_size = int(round(float(args.edge_ratio) * min(height, width)))

    for source_idx in selected_indices:
        map_design = map_designs[int(source_idx)]
        existing_goal = _existing_goal_rc(base_goal_maps[int(source_idx)])
        candidates = [
            rc for rc in _goal_candidates(map_design, edge_size=edge_size)
            if rc != existing_goal
        ]
        if not candidates:
            continue
        seen = {existing_goal}
        for _ in range(int(args.goals_per_map)):
            remaining = [rc for rc in candidates if rc not in seen]
            if not remaining:
                break
            sample_count = min(len(remaining), int(args.candidate_goals))
            sampled_idx = rng.choice(len(remaining), size=sample_count, replace=False)
            sampled_goals = [remaining[int(i)] for i in np.atleast_1d(sampled_idx).tolist()]
            best_goal = max(sampled_goals, key=lambda rc: _goal_score(map_design, rc))
            goal_y, goal_x = best_goal
            goal_loc = (0, int(goal_y), int(goal_x))
            opt_value = dijkstra_dist(map_design, mechanism, goal_loc).astype(np.float32)
            opt_policy = extract_policy(map_design, mechanism, opt_value).astype(np.float32)

            extra_map_designs.append(map_design.astype(np.float32))
            extra_goal_maps.append(_one_hot_goal(best_goal, (height, width)))
            extra_opt_policies.append(opt_policy[None, ...][0])
            extra_opt_dists.append(opt_value.astype(np.float32))
            extra_source_indices.append(int(source_idx))
            extra_goal_scores.append(_goal_score(map_design, best_goal))
            seen.add(best_goal)

    if not extra_map_designs:
        raise RuntimeError("Failed to generate any replay goals.")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        map_designs=np.stack(extra_map_designs, axis=0).astype(np.float32),
        goal_maps=np.stack(extra_goal_maps, axis=0).astype(np.float32),
        opt_policies=np.stack(extra_opt_policies, axis=0).astype(np.float32),
        opt_dists=np.stack(extra_opt_dists, axis=0).astype(np.float32),
        source_indices=np.asarray(extra_source_indices, dtype=np.int64),
        goal_scores=np.asarray(extra_goal_scores, dtype=np.float32),
    )
    print(
        "saved_goal_replay_npz={} replay_samples={} source_maps={} mean_goal_score={:.4f}".format(
            args.out_npz,
            len(extra_map_designs),
            len(set(extra_source_indices)),
            float(np.mean(extra_goal_scores)),
        )
    )


if __name__ == "__main__":
    main()
