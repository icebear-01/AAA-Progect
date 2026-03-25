"""Generate extra hard-map replay samples by recombining priority map tiles."""

from __future__ import annotations

import argparse
import csv
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


def _load_priority_rows(
    csv_path: Path,
    *,
    priority_column: str,
    top_fraction: float,
    min_priority: float,
    max_maps: int,
) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "idx" not in reader.fieldnames:
            raise ValueError(f"{csv_path} missing required column: idx")
        if priority_column not in reader.fieldnames:
            raise ValueError(f"{csv_path} missing required column: {priority_column}")
        for row in reader:
            score = float(row[priority_column])
            if score < float(min_priority):
                continue
            rows.append((int(row["idx"]), score))
    if not rows:
        raise RuntimeError("No priority maps selected.")
    rows.sort(key=lambda item: item[1], reverse=True)
    if int(max_maps) > 0:
        keep = min(len(rows), int(max_maps))
    else:
        frac = min(max(float(top_fraction), 0.0), 1.0)
        keep = len(rows) if frac <= 0.0 else max(1, int(round(len(rows) * frac)))
    return rows[:keep]


def _goal_candidates(
    map_design: np.ndarray,
    *,
    edge_size: int,
) -> List[Tuple[int, int]]:
    free_mask = np.asarray(map_design, dtype=np.float32) > 0.5
    limage = label(free_mask.astype(np.uint8), background=0, connectivity=1)
    num_pixels = np.bincount(limage.flatten())
    num_pixels[0] = 0
    candidate_mask = (limage == int(np.argmax(num_pixels))) if np.max(num_pixels) > 0 else free_mask
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


def _assemble_hard_map(
    map_designs: np.ndarray,
    source_pool: Sequence[int],
    source_probs: np.ndarray,
    *,
    tile_splits: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int]]:
    if int(tile_splits) <= 0:
        raise ValueError(f"tile_splits must be positive, got {tile_splits}")
    height, width = map_designs.shape[-2:]
    if height % int(tile_splits) != 0 or width % int(tile_splits) != 0:
        raise ValueError(
            f"Map size {(height, width)} must be divisible by tile_splits={tile_splits}"
        )
    block_h = height // int(tile_splits)
    block_w = width // int(tile_splits)
    out = np.zeros((height, width), dtype=np.float32)
    chosen_sources: List[int] = []
    for by in range(int(tile_splits)):
        for bx in range(int(tile_splits)):
            src_idx = int(rng.choice(source_pool, p=source_probs))
            chosen_sources.append(src_idx)
            y0 = by * block_h
            y1 = y0 + block_h
            x0 = bx * block_w
            x1 = x0 + block_w
            out[y0:y1, x0:x1] = map_designs[src_idx, y0:y1, x0:x1]
    return out, chosen_sources


def _one_hot_goal(goal_rc: Tuple[int, int], shape_hw: Tuple[int, int]) -> np.ndarray:
    goal_map = np.zeros((1, int(shape_hw[0]), int(shape_hw[1])), dtype=np.float32)
    goal_map[0, int(goal_rc[0]), int(goal_rc[1])] = 1.0
    return goal_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate hard-map replay NPZ from priority tiles.")
    p.add_argument("--train-npz", type=Path, required=True)
    p.add_argument("--priority-csv", type=Path, required=True)
    p.add_argument("--priority-column", type=str, default="priority_score")
    p.add_argument("--top-fraction", type=float, default=0.15)
    p.add_argument("--min-priority", type=float, default=0.0)
    p.add_argument("--max-maps", type=int, default=128)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--candidate-goals", type=int, default=12)
    p.add_argument("--edge-ratio", type=float, default=0.25)
    p.add_argument("--tile-splits", type=int, default=2)
    p.add_argument("--mechanism", type=str, default="moore")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-npz", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError(f"--num-samples must be positive, got {args.num_samples}")
    if args.candidate_goals <= 0:
        raise ValueError(f"--candidate-goals must be positive, got {args.candidate_goals}")
    if args.edge_ratio < 0.0 or args.edge_ratio > 0.5:
        raise ValueError(f"--edge-ratio must be in [0, 0.5], got {args.edge_ratio}")

    mechanism = get_mechanism(args.mechanism)
    if getattr(mechanism, "num_orient", 1) != 1:
        raise ValueError("This hard-map replay generator currently supports orientation-free grid datasets only.")
    rng = np.random.default_rng(args.seed)

    priority_rows = _load_priority_rows(
        args.priority_csv,
        priority_column=args.priority_column,
        top_fraction=args.top_fraction,
        min_priority=args.min_priority,
        max_maps=args.max_maps,
    )
    source_pool = [int(idx) for idx, _ in priority_rows]
    source_score_map = {int(idx): float(score) for idx, score in priority_rows}
    source_weights = np.asarray([float(score) for _, score in priority_rows], dtype=np.float64)
    if np.sum(source_weights) <= 0.0:
        source_probs = np.full(len(source_pool), 1.0 / float(len(source_pool)), dtype=np.float64)
    else:
        source_probs = source_weights / np.sum(source_weights)

    with np.load(args.train_npz) as data:
        map_designs = np.asarray(data["arr_0"], dtype=np.float32)

    height, width = map_designs.shape[1:]
    edge_size = int(round(float(args.edge_ratio) * min(height, width)))

    replay_map_designs: List[np.ndarray] = []
    replay_goal_maps: List[np.ndarray] = []
    replay_opt_policies: List[np.ndarray] = []
    replay_opt_dists: List[np.ndarray] = []
    replay_source_indices: List[int] = []
    replay_goal_scores: List[float] = []
    replay_component_sources: List[np.ndarray] = []

    for _ in range(int(args.num_samples)):
        map_design, component_sources = _assemble_hard_map(
            map_designs,
            source_pool,
            source_probs,
            tile_splits=int(args.tile_splits),
            rng=rng,
        )
        candidates = _goal_candidates(map_design, edge_size=edge_size)
        if not candidates:
            continue
        sample_count = min(len(candidates), int(args.candidate_goals))
        sampled_idx = rng.choice(len(candidates), size=sample_count, replace=False)
        sampled_goals = [candidates[int(i)] for i in np.atleast_1d(sampled_idx).tolist()]
        best_goal = max(sampled_goals, key=lambda rc: _goal_score(map_design, rc))
        goal_loc = (0, int(best_goal[0]), int(best_goal[1]))
        opt_value = dijkstra_dist(map_design, mechanism, goal_loc).astype(np.float32)
        opt_policy = extract_policy(map_design, mechanism, opt_value).astype(np.float32)

        component_priority = [float(source_score_map[int(src)]) for src in component_sources]
        dominant_source = int(component_sources[int(np.argmax(component_priority))])

        replay_map_designs.append(map_design.astype(np.float32))
        replay_goal_maps.append(_one_hot_goal(best_goal, (height, width)))
        replay_opt_policies.append(opt_policy.astype(np.float32))
        replay_opt_dists.append(opt_value.astype(np.float32))
        replay_source_indices.append(dominant_source)
        replay_goal_scores.append(_goal_score(map_design, best_goal))
        replay_component_sources.append(np.asarray(component_sources, dtype=np.int64))

    if not replay_map_designs:
        raise RuntimeError("Failed to generate any hard-map replay samples.")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        map_designs=np.stack(replay_map_designs, axis=0).astype(np.float32),
        goal_maps=np.stack(replay_goal_maps, axis=0).astype(np.float32),
        opt_policies=np.stack(replay_opt_policies, axis=0).astype(np.float32),
        opt_dists=np.stack(replay_opt_dists, axis=0).astype(np.float32),
        source_indices=np.asarray(replay_source_indices, dtype=np.int64),
        goal_scores=np.asarray(replay_goal_scores, dtype=np.float32),
        component_source_indices=np.stack(replay_component_sources, axis=0).astype(np.int64),
    )
    print(
        "saved_hardmap_replay_npz={} replay_samples={} source_maps={} mean_goal_score={:.4f}".format(
            args.out_npz,
            len(replay_map_designs),
            len(set(replay_source_indices)),
            float(np.mean(replay_goal_scores)),
        )
    )


if __name__ == "__main__":
    main()
