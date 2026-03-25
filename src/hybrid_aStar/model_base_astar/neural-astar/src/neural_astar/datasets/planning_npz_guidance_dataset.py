"""Guidance dataset adapter for planning-datasets .npz format.

This adapter reads original planning-datasets arrays (arr_0..arr_11) and
converts each sample to the Route-A guidance training schema:
- occ_map [1,H,W], 1=obstacle, 0=free
- start_map [1,H,W], one-hot
- goal_map [1,H,W], one-hot
- opt_traj [1,H,W], expert path mask
- target_cost [1,H,W], dense distance-field supervision target
- astar_expanded_map [1,H,W], normalized projection of improved-heuristic 2D A* expansions
- exact_heuristic_map [1,H,W], exact 8-connected distance-to-goal map
- octile_heuristic_map [1,H,W], anchor heuristic map used by improved A*
- residual_heuristic_map [1,H,W], non-negative residual h*(x)-h_octile(x)
- expanded_trace_map [1,H,W], zeros for legacy 2D data without Hybrid A* traces
- opt_traj_orient [K,H,W], optional orientation-aware corridor
- target_cost_orient [K,H,W], optional orientation-aware dense target
- start_pose [3], yaw defaults to zero for legacy 2D data
- goal_pose [3], yaw defaults to zero for legacy 2D data

Optionally, train-time goal replay samples can be appended from a separate NPZ
containing additional `(map_design, goal_map, opt_policy, opt_dist)` tuples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_astar_guided.grid_astar import astar_8conn_stats
from neural_astar.utils.coords import rc_to_xy
from neural_astar.utils.guidance_targets import (
    build_clearance_input_map,
    build_clearance_penalty_map,
    build_expanded_xy_map,
    build_residual_heuristic_maps,
    build_target_cost_map,
)


class PlanningNPZGuidanceDataset(Dataset):
    """Load planning-datasets split and sample random starts on-the-fly."""

    def __init__(
        self,
        npz_path: str | Path,
        split: str = "train",
        seed: int = 1234,
        orientation_bins: int = 1,
        goal_replay_npz: str | Path | None = None,
        pct1: float = 0.55,
        pct2: float = 0.70,
        pct3: float = 0.85,
        clearance_safe_distance: float = 0.0,
        clearance_power: float = 2.0,
        clearance_target_weight: float = 0.0,
        clearance_penalize_corridor: bool = False,
        clearance_input_clip_distance: float = 0.0,
    ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"npz not found: {self.npz_path}")
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"split must be train/valid/test, got {split}")
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.pcts = np.array([pct1, pct2, pct3, 1.0], dtype=np.float32)
        self.orientation_bins = int(max(1, orientation_bins))
        self._heuristic_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.clearance_safe_distance = float(clearance_safe_distance)
        self.clearance_power = float(clearance_power)
        self.clearance_target_weight = float(clearance_target_weight)
        self.clearance_penalize_corridor = bool(clearance_penalize_corridor)
        self.clearance_input_clip_distance = float(clearance_input_clip_distance)

        idx_base = {"train": 0, "valid": 4, "test": 8}[split]
        with np.load(self.npz_path) as data:
            self.map_designs = np.asarray(data[f"arr_{idx_base}"], dtype=np.float32)
            self.goal_maps = np.asarray(data[f"arr_{idx_base+1}"], dtype=np.float32)
            self.opt_policies = np.asarray(data[f"arr_{idx_base+2}"], dtype=np.float32)
            self.opt_dists = np.asarray(data[f"arr_{idx_base+3}"], dtype=np.float32)
        self.source_indices = np.arange(int(self.map_designs.shape[0]), dtype=np.int64)

        if self.goal_maps.ndim != 4 or self.goal_maps.shape[1] != 1:
            raise ValueError(f"goal_maps must be [N,1,H,W], got {self.goal_maps.shape}")
        if self.opt_policies.ndim != 5:
            raise ValueError(f"opt_policies must be [N,A,O,H,W], got {self.opt_policies.shape}")
        if self.opt_dists.ndim != 4 or self.opt_dists.shape[1] != 1:
            raise ValueError(f"opt_dists must be [N,1,H,W], got {self.opt_dists.shape}")
        if goal_replay_npz is not None:
            self._append_goal_replay_npz(goal_replay_npz)

    def __len__(self) -> int:
        return int(self.map_designs.shape[0])

    def _append_goal_replay_npz(self, replay_npz: str | Path) -> None:
        replay_path = Path(replay_npz)
        if not replay_path.exists():
            raise FileNotFoundError(f"goal replay npz not found: {replay_path}")
        with np.load(replay_path) as data:
            required = ["map_designs", "goal_maps", "opt_policies", "opt_dists", "source_indices"]
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"{replay_path} missing required arrays: {missing}")
            replay_map_designs = np.asarray(data["map_designs"], dtype=np.float32)
            replay_goal_maps = np.asarray(data["goal_maps"], dtype=np.float32)
            replay_opt_policies = np.asarray(data["opt_policies"], dtype=np.float32)
            replay_opt_dists = np.asarray(data["opt_dists"], dtype=np.float32)
            replay_source_indices = np.asarray(data["source_indices"], dtype=np.int64)

        if replay_map_designs.ndim != 3:
            raise ValueError(
                f"replay map_designs must be [N,H,W], got {tuple(replay_map_designs.shape)}"
            )
        if replay_goal_maps.ndim != 4 or replay_goal_maps.shape[1] != 1:
            raise ValueError(
                f"replay goal_maps must be [N,1,H,W], got {tuple(replay_goal_maps.shape)}"
            )
        if replay_opt_policies.ndim != 5:
            raise ValueError(
                "replay opt_policies must be [N,A,O,H,W], got "
                f"{tuple(replay_opt_policies.shape)}"
            )
        if replay_opt_dists.ndim != 4 or replay_opt_dists.shape[1] != 1:
            raise ValueError(
                f"replay opt_dists must be [N,1,H,W], got {tuple(replay_opt_dists.shape)}"
            )
        replay_count = int(replay_map_designs.shape[0])
        if replay_goal_maps.shape[0] != replay_count or replay_opt_policies.shape[0] != replay_count:
            raise ValueError(
                "Replay arrays must agree on leading dimension: "
                f"maps={replay_map_designs.shape[0]} goals={replay_goal_maps.shape[0]} "
                f"policies={replay_opt_policies.shape[0]}"
            )
        if replay_opt_dists.shape[0] != replay_count or replay_source_indices.shape[0] != replay_count:
            raise ValueError(
                "Replay arrays must agree on leading dimension: "
                f"dists={replay_opt_dists.shape[0]} source_indices={replay_source_indices.shape[0]} "
                f"expected={replay_count}"
            )
        base_shape = tuple(self.map_designs.shape[1:])
        if tuple(replay_map_designs.shape[1:]) != base_shape:
            raise ValueError(
                f"Replay map shapes must match base dataset: {tuple(replay_map_designs.shape[1:])} vs {base_shape}"
            )

        self.map_designs = np.concatenate([self.map_designs, replay_map_designs], axis=0)
        self.goal_maps = np.concatenate([self.goal_maps, replay_goal_maps], axis=0)
        self.opt_policies = np.concatenate([self.opt_policies, replay_opt_policies], axis=0)
        self.opt_dists = np.concatenate([self.opt_dists, replay_opt_dists], axis=0)
        self.source_indices = np.concatenate([self.source_indices, replay_source_indices], axis=0)

    def build_sampling_weights(
        self,
        emphasized_base_indices: List[int],
        emphasized_weight: float,
    ) -> torch.Tensor:
        weights = torch.ones(len(self), dtype=torch.double)
        boosted = set(int(idx) for idx in emphasized_base_indices)
        boost = max(float(emphasized_weight), 1.0)
        for sample_idx, source_idx in enumerate(self.source_indices.tolist()):
            if int(source_idx) in boosted:
                weights[sample_idx] = boost
        return weights

    def _get_random_start_map(self, opt_dist: np.ndarray) -> np.ndarray:
        """Sample one start point from percentile-binned distance field."""
        od_vct = opt_dist.flatten()
        od_vals = od_vct[od_vct > od_vct.min()]
        if od_vals.size == 0:
            # fallback: random non-min location
            candidate = np.where(od_vct > od_vct.min())[0]
            if candidate.size == 0:
                candidate = np.arange(od_vct.size)
            start_idx = int(self.rng.choice(candidate))
        else:
            od_th = np.percentile(od_vals, 100.0 * (1.0 - self.pcts))
            r = int(self.rng.integers(0, len(od_th) - 1))
            start_candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
            candidate = np.where(start_candidate)[0]
            if candidate.size == 0:
                candidate = np.where(od_vct > od_vct.min())[0]
            start_idx = int(self.rng.choice(candidate))

        start_map = np.zeros_like(opt_dist, dtype=np.float32)
        start_map.ravel()[start_idx] = 1.0
        return start_map

    @staticmethod
    def _next_loc(current_loc: Tuple[int, int, int], one_hot_action: np.ndarray) -> Tuple[int, int, int]:
        # (orient, y, x) deltas.
        action_to_move = [
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
            (0, +1, 0),
            (0, -1, +1),
            (0, -1, -1),
            (0, +1, +1),
            (0, +1, -1),
        ]
        move = action_to_move[int(np.argmax(one_hot_action))]
        return (
            int(current_loc[0] + move[0]),
            int(current_loc[1] + move[1]),
            int(current_loc[2] + move[2]),
        )

    def _get_opt_traj(
        self,
        start_map: np.ndarray,
        goal_map: np.ndarray,
        opt_policy: np.ndarray,
    ) -> np.ndarray:
        """Follow optimal policy from sampled start to goal."""
        opt_traj = np.zeros_like(start_map, dtype=np.float32)
        policy = np.transpose(opt_policy, (1, 2, 3, 0))  # [O,H,W,A]

        current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())

        steps = 0
        max_steps = int(start_map.shape[-1] * start_map.shape[-2] * 4)
        while goal_loc != current_loc:
            opt_traj[current_loc] = 1.0
            next_loc = self._next_loc(current_loc, policy[current_loc])
            if (
                next_loc[0] < 0
                or next_loc[0] >= opt_traj.shape[0]
                or next_loc[1] < 0
                or next_loc[1] >= opt_traj.shape[1]
                or next_loc[2] < 0
                or next_loc[2] >= opt_traj.shape[2]
            ):
                raise RuntimeError("Next location from optimal policy is out of bounds")
            if opt_traj[next_loc] > 0.5:
                raise RuntimeError("Loop detected while following optimal policy")
            current_loc = next_loc
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Exceeded max_steps while following optimal policy")
        return opt_traj

    @staticmethod
    def _pose_from_one_hot(one_hot_1hw: np.ndarray) -> np.ndarray:
        arr = np.asarray(one_hot_1hw[0], dtype=np.float32)
        y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
        px, py = rc_to_xy(y, x, width=arr.shape[1], height=arr.shape[0])
        return np.array([float(px), float(py), 0.0], dtype=np.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        map_design = self.map_designs[index]  # 1=free, 0=obstacle
        goal_map = self.goal_maps[index]      # [1,H,W]
        opt_policy = self.opt_policies[index] # [A,O,H,W]
        opt_dist = self.opt_dists[index]      # [1,H,W]

        start_map = None
        opt_traj = None
        last_err: Exception | None = None
        for _ in range(16):
            try:
                start_map = self._get_random_start_map(opt_dist)
                opt_traj = self._get_opt_traj(start_map, goal_map, opt_policy)
                break
            except RuntimeError as err:
                last_err = err
                continue
        if start_map is None or opt_traj is None:
            raise RuntimeError(f"Failed to sample valid start/trajectory for index={index}: {last_err}")

        occ_map = (1.0 - map_design)[None, ...].astype(np.float32)  # [1,H,W], 1=obstacle
        start_pose = self._pose_from_one_hot(start_map)
        goal_pose = self._pose_from_one_hot(goal_map)
        target_cost = build_target_cost_map(
            occ_map[0],
            opt_traj[0],
            clearance_weight=self.clearance_target_weight,
            clearance_safe_distance=self.clearance_safe_distance,
            clearance_power=self.clearance_power,
            clearance_penalize_path=self.clearance_penalize_corridor,
        )[None, ...].astype(np.float32)
        clearance_penalty_map = build_clearance_penalty_map(
            occ_map=occ_map[0],
            safe_distance=self.clearance_safe_distance,
            power=self.clearance_power,
        )[None, ...].astype(np.float32)
        clearance_input_map = build_clearance_input_map(
            occ_map=occ_map[0],
            clip_distance=self.clearance_input_clip_distance,
        )[None, ...].astype(np.float32)
        cached = self._heuristic_cache.get(int(index))
        if cached is None:
            cached = build_residual_heuristic_maps(
                occ_map=occ_map[0],
                goal_xy=(int(goal_pose[0]), int(goal_pose[1])),
            )
            self._heuristic_cache[int(index)] = cached
        exact_heuristic_map, octile_heuristic_map, residual_heuristic_map = cached
        astar_stats = astar_8conn_stats(
            occ_map=occ_map[0],
            start_xy=(int(start_pose[0]), int(start_pose[1])),
            goal_xy=(int(goal_pose[0]), int(goal_pose[1])),
            heuristic_mode="octile",
            heuristic_weight=1.0,
            allow_corner_cut=True,
        )
        astar_expanded_map = build_expanded_xy_map(
            occ_map=occ_map[0],
            expanded_xy=np.asarray(astar_stats.expanded_xy, dtype=np.float32),
        )[None, ...].astype(np.float32)
        expanded_trace_map = np.zeros_like(target_cost, dtype=np.float32)

        sample = {
            "occ_map": torch.from_numpy(occ_map),
            "start_map": torch.from_numpy(start_map.astype(np.float32)),
            "goal_map": torch.from_numpy(goal_map.astype(np.float32)),
            "opt_traj": torch.from_numpy(opt_traj.astype(np.float32)),
            "target_cost": torch.from_numpy(target_cost),
            "clearance_penalty_map": torch.from_numpy(clearance_penalty_map),
            "clearance_input_map": torch.from_numpy(clearance_input_map),
            "astar_expanded_map": torch.from_numpy(astar_expanded_map),
            "exact_heuristic_map": torch.from_numpy(exact_heuristic_map[None, ...].astype(np.float32)),
            "octile_heuristic_map": torch.from_numpy(octile_heuristic_map[None, ...].astype(np.float32)),
            "residual_heuristic_map": torch.from_numpy(residual_heuristic_map[None, ...].astype(np.float32)),
            "expanded_trace_map": torch.from_numpy(expanded_trace_map),
            "start_pose": torch.from_numpy(start_pose),
            "goal_pose": torch.from_numpy(goal_pose),
        }
        if self.orientation_bins > 1:
            sample["opt_traj_orient"] = torch.from_numpy(
                np.repeat(opt_traj.astype(np.float32), self.orientation_bins, axis=0)
            )
            sample["target_cost_orient"] = torch.from_numpy(
                np.repeat(target_cost.astype(np.float32), self.orientation_bins, axis=0)
            )
        return sample
