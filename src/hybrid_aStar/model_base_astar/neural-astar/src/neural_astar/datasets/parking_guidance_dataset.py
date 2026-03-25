"""Dataset for parking guidance cost-map learning.

Each sample returns base tensors with shape [1, H, W]:
- occ_map: 1=obstacle, 0=free
- start_map: one-hot start location
- goal_map: one-hot goal location
- opt_traj: expert corridor/path mask
- target_cost: dense distance-field supervision target
- astar_expanded_map: normalized projection of improved-heuristic 2D A* expanded cells
- exact_heuristic_map: exact 8-connected distance-to-goal map
- octile_heuristic_map: anchor heuristic map used by improved A*
- residual_heuristic_map: non-negative residual h*(x)-h_octile(x)
- expanded_trace_map: optional normalized projection of Hybrid A* expanded rollout cells
- opt_traj_orient: optional [K, H, W] orientation-aware corridor
- target_cost_orient: optional [K, H, W] orientation-aware dense target
- start_pose: [3], (x, y, yaw_rad)
- goal_pose: [3], (x, y, yaw_rad)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_astar_guided.grid_astar import astar_8conn_stats
from neural_astar.utils.coords import rc_to_xy
from neural_astar.utils.guidance_targets import (
    build_clearance_input_map,
    build_clearance_penalty_map,
    build_expanded_xy_map,
    build_expanded_trace_map,
    build_residual_heuristic_maps,
    build_orientation_target_maps,
    build_target_cost_map,
)


class ParkingGuidanceDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        orientation_bins: int = 1,
        clearance_safe_distance: float = 0.0,
        clearance_power: float = 2.0,
        clearance_target_weight: float = 0.0,
        clearance_penalize_corridor: bool = False,
        clearance_input_clip_distance: float = 0.0,
    ):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        self.files: List[Path] = sorted(self.root_dir.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in: {self.root_dir}")
        self.orientation_bins = int(max(1, orientation_bins))
        self.clearance_safe_distance = float(clearance_safe_distance)
        self.clearance_power = float(clearance_power)
        self.clearance_target_weight = float(clearance_target_weight)
        self.clearance_penalize_corridor = bool(clearance_penalize_corridor)
        self.clearance_input_clip_distance = float(clearance_input_clip_distance)

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _to_chw(arr: np.ndarray, key: str) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2:
            return a[None, ...]
        if a.ndim == 3:
            return a
        raise ValueError(f"{key} must be [H,W] or [C,H,W], got {a.shape}")

    @staticmethod
    def _pose_from_data(
        data: np.lib.npyio.NpzFile,
        key: str,
        one_hot_map: np.ndarray,
    ) -> np.ndarray:
        pose_key = f"{key}_pose"
        if pose_key in data.files:
            pose = np.asarray(data[pose_key], dtype=np.float32).reshape(-1)
            if pose.size < 3:
                raise ValueError(f"{pose_key} must have at least 3 values, got {pose.shape}")
            return pose[:3]

        arr = np.asarray(one_hot_map[0], dtype=np.float32)
        y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
        px, py = rc_to_xy(y, x, width=arr.shape[1], height=arr.shape[0])
        return np.array([float(px), float(py), 0.0], dtype=np.float32)

    @staticmethod
    def _one_hot_xy(one_hot_map: np.ndarray) -> tuple[int, int]:
        arr = np.asarray(one_hot_map[0], dtype=np.float32)
        y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return int(x), int(y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[idx]
        with np.load(file_path) as data:
            missing = {"occ_map", "start_map", "goal_map", "opt_traj"} - set(data.files)
            if missing:
                raise KeyError(f"Missing keys {missing} in {file_path}")

            occ_map = self._to_chw(data["occ_map"], "occ_map")
            start_map = self._to_chw(data["start_map"], "start_map")
            goal_map = self._to_chw(data["goal_map"], "goal_map")
            opt_traj = self._to_chw(data["opt_traj"], "opt_traj")
            start_pose = self._pose_from_data(data, "start", start_map)
            goal_pose = self._pose_from_data(data, "goal", goal_map)
            if "target_cost" in data.files and self.clearance_target_weight <= 0.0:
                target_cost = self._to_chw(data["target_cost"], "target_cost")
            else:
                target_cost = build_target_cost_map(
                    occ_map[0],
                    opt_traj[0],
                    clearance_weight=self.clearance_target_weight,
                    clearance_safe_distance=self.clearance_safe_distance,
                    clearance_power=self.clearance_power,
                    clearance_penalize_path=self.clearance_penalize_corridor,
                )[None, ...]
            clearance_penalty_map = build_clearance_penalty_map(
                occ_map=occ_map[0],
                safe_distance=self.clearance_safe_distance,
                power=self.clearance_power,
            )[None, ...].astype(np.float32)
            clearance_input_map = build_clearance_input_map(
                occ_map=occ_map[0],
                clip_distance=self.clearance_input_clip_distance,
            )[None, ...].astype(np.float32)
            if "astar_expanded_map" in data.files:
                astar_expanded_map = self._to_chw(data["astar_expanded_map"], "astar_expanded_map")
            elif "astar_expanded_xy" in data.files:
                astar_expanded_map = build_expanded_xy_map(
                    occ_map=occ_map[0],
                    expanded_xy=np.asarray(data["astar_expanded_xy"], dtype=np.float32),
                )[None, ...]
            else:
                start_xy = self._one_hot_xy(start_map)
                goal_xy = self._one_hot_xy(goal_map)
                astar_stats = astar_8conn_stats(
                    occ_map=occ_map[0],
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    heuristic_mode="octile",
                    heuristic_weight=1.0,
                    allow_corner_cut=True,
                )
                astar_expanded_map = build_expanded_xy_map(
                    occ_map=occ_map[0],
                    expanded_xy=np.asarray(astar_stats.expanded_xy, dtype=np.float32),
                )[None, ...]
            if {"exact_heuristic_map", "octile_heuristic_map", "residual_heuristic_map"} <= set(data.files):
                exact_heuristic_map = self._to_chw(data["exact_heuristic_map"], "exact_heuristic_map")
                octile_heuristic_map = self._to_chw(data["octile_heuristic_map"], "octile_heuristic_map")
                residual_heuristic_map = self._to_chw(
                    data["residual_heuristic_map"],
                    "residual_heuristic_map",
                )
            else:
                exact_heuristic_map, octile_heuristic_map, residual_heuristic_map = build_residual_heuristic_maps(
                    occ_map=occ_map[0],
                    goal_xy=(int(goal_pose[0]), int(goal_pose[1])),
                )
                exact_heuristic_map = exact_heuristic_map[None, ...]
                octile_heuristic_map = octile_heuristic_map[None, ...]
                residual_heuristic_map = residual_heuristic_map[None, ...]
            if "expanded_trace_map" in data.files:
                expanded_trace_map = self._to_chw(data["expanded_trace_map"], "expanded_trace_map")
            elif "expanded_trace_xy" in data.files:
                expanded_trace_map = build_expanded_trace_map(
                    occ_map=occ_map[0],
                    expanded_trace_xy=np.asarray(data["expanded_trace_xy"], dtype=np.float32),
                )[None, ...]
            else:
                expanded_trace_map = np.zeros_like(opt_traj, dtype=np.float32)

            opt_traj_orient = None
            target_cost_orient = None
            if self.orientation_bins > 1:
                saved_traj_orient = None
                saved_cost_orient = None
                if "opt_traj_orient" in data.files:
                    saved_traj_orient = self._to_chw(data["opt_traj_orient"], "opt_traj_orient")
                if "target_cost_orient" in data.files:
                    saved_cost_orient = self._to_chw(data["target_cost_orient"], "target_cost_orient")

                if (
                    saved_traj_orient is not None
                    and saved_cost_orient is not None
                    and saved_traj_orient.shape[0] == self.orientation_bins
                    and saved_cost_orient.shape[0] == self.orientation_bins
                ):
                    opt_traj_orient = saved_traj_orient
                    target_cost_orient = saved_cost_orient
                elif "path_poses" in data.files:
                    opt_traj_orient, target_cost_orient = build_orientation_target_maps(
                        occ_map=occ_map[0],
                        opt_traj=opt_traj[0],
                        path_poses=np.asarray(data["path_poses"], dtype=np.float32),
                        yaw_bins=self.orientation_bins,
                        clearance_weight=self.clearance_target_weight,
                        clearance_safe_distance=self.clearance_safe_distance,
                        clearance_power=self.clearance_power,
                        clearance_penalize_path=self.clearance_penalize_corridor,
                    )
                else:
                    opt_traj_orient = np.repeat(opt_traj, self.orientation_bins, axis=0)
                    target_cost_orient = np.repeat(target_cost, self.orientation_bins, axis=0)

        if occ_map.shape != start_map.shape or occ_map.shape != goal_map.shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: "
                f"occ={occ_map.shape}, start={start_map.shape}, goal={goal_map.shape}"
            )
        if occ_map.shape != opt_traj.shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: occ={occ_map.shape}, opt_traj={opt_traj.shape}"
            )
        if occ_map.shape != target_cost.shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: occ={occ_map.shape}, target_cost={target_cost.shape}"
            )
        if occ_map.shape != astar_expanded_map.shape:
            raise ValueError(
                "Shape mismatch in "
                f"{file_path}: occ={occ_map.shape}, astar_expanded_map={astar_expanded_map.shape}"
            )
        if occ_map.shape != exact_heuristic_map.shape:
            raise ValueError(
                "Shape mismatch in "
                f"{file_path}: occ={occ_map.shape}, exact_heuristic_map={exact_heuristic_map.shape}"
            )
        if occ_map.shape != octile_heuristic_map.shape:
            raise ValueError(
                "Shape mismatch in "
                f"{file_path}: occ={occ_map.shape}, octile_heuristic_map={octile_heuristic_map.shape}"
            )
        if occ_map.shape != residual_heuristic_map.shape:
            raise ValueError(
                "Shape mismatch in "
                f"{file_path}: occ={occ_map.shape}, residual_heuristic_map={residual_heuristic_map.shape}"
            )
        if occ_map.shape != expanded_trace_map.shape:
            raise ValueError(
                "Shape mismatch in "
                f"{file_path}: occ={occ_map.shape}, expanded_trace_map={expanded_trace_map.shape}"
            )

        sample = {
            "occ_map": torch.from_numpy(occ_map),
            "start_map": torch.from_numpy(start_map),
            "goal_map": torch.from_numpy(goal_map),
            "opt_traj": torch.from_numpy(opt_traj),
            "target_cost": torch.from_numpy(target_cost.astype(np.float32)),
            "clearance_penalty_map": torch.from_numpy(clearance_penalty_map.astype(np.float32)),
            "clearance_input_map": torch.from_numpy(clearance_input_map.astype(np.float32)),
            "astar_expanded_map": torch.from_numpy(astar_expanded_map.astype(np.float32)),
            "exact_heuristic_map": torch.from_numpy(exact_heuristic_map.astype(np.float32)),
            "octile_heuristic_map": torch.from_numpy(octile_heuristic_map.astype(np.float32)),
            "residual_heuristic_map": torch.from_numpy(residual_heuristic_map.astype(np.float32)),
            "expanded_trace_map": torch.from_numpy(expanded_trace_map.astype(np.float32)),
            "start_pose": torch.from_numpy(start_pose),
            "goal_pose": torch.from_numpy(goal_pose),
        }
        if opt_traj_orient is not None and target_cost_orient is not None:
            sample["opt_traj_orient"] = torch.from_numpy(opt_traj_orient.astype(np.float32))
            sample["target_cost_orient"] = torch.from_numpy(target_cost_orient.astype(np.float32))
        return sample
