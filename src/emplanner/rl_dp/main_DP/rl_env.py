"""
Simple reinforcement learning environment built on the s-l grid with obstacles.

The environment follows a Gym-like API with `reset()` and `step(action)`. The
agent moves along discrete `s` columns, selecting one lateral index per step
to form a path from `s = 0` to the final column. Obstacles are enforced, and a
composite reward encourages collision-free and smooth trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sl_grid import GridSpec, build_grid, default_training_grid_spec
from sl_obstacles import Obstacle, generate_random_obstacles


_SCENARIO_DATASET_CACHE: Dict[str, Dict[str, object]] = {}


@dataclass
class StepResult:
    observation: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, object]


@dataclass(frozen=True)
class DPCandidateResult:
    feasible: bool
    total_cost: float
    avg_cost: float
    path_indices: Tuple[int, ...]


class SLPathEnv:
    """
    Reinforcement learning environment over an s-l sampling grid.

    State representation:
        - `s_index`: current longitudinal column index (0-based)
        - `path_l_indices`: indices of chosen `l` positions up to current step
        - `occupancy`: boolean mask (s_samples x l_samples) marking collision cells
          (vehicle footprint if dimensions are provided, otherwise obstacle cells)
        - `start_l`: continuous starting l coordinate (used as extra input feature)
    Action:
        - Choose the next lateral index (integer in [0, l_samples - 1]) for the
          subsequent column. Optionally, a lateral move limit can be enforced.
    Reward structure:
        - Collision penalty if the vehicle footprint at interpolated points intersects an obstacle or goes out of bounds.
        - Smoothness cost proportional to lateral change (first-order term).
        - Large penalty if no valid action exists at the current state.
        - Terminal reward when successfully reaching the final column.
    Episode terminates on collision or when the agent reaches the last column.
    """

    def __init__(
        self,
        grid_spec: GridSpec,
        *,
        min_obstacles: int = 0,
        max_obstacles: int = 10,
        obstacle_length_range: Sequence[float] = (0.6, 1.8),
        obstacle_width_range: Sequence[float] = (0.4, 1.4),
        avoid_obstacle_overlap: bool = False,
        obstacle_overlap_clearance: float = 0.0,
        obstacle_sampling_attempts_per_obstacle: int = 24,
        smoothness_penalty: float = 2.0,
        base_smoothness_penalty: float = 1.0,
        move_limit_penalty_multiplier: float = 5.0,
        lateral_reference: float = 0.0,  ##参考线l
        reference_penalty: float = 5,
        max_slope: float = 1.0,
        slope_penalty: float = 10.0,
        max_slope_penalty: float = 20.0,
        collision_penalty: float = -100.0,
        terminal_reward: float = 60.0,
        non_goal_penalty: float = 0.0,
        no_valid_action_penalty: Optional[float] = None,
        lateral_move_limit: Optional[int] = None,
        collision_inflation: Optional[float] = None,
        coarse_collision_inflation: float = 0.2, ##粗略碰撞膨胀
        fine_collision_inflation: float = 0.2, ##精细碰撞膨胀
        interpolation_points: int = 3,
        vehicle_length: float = 0.0,
        vehicle_width: float = 0.0,
        start_clear_fraction: float = 0.2,   ##起始区域清除比例
        scenario_pool_size: int = 1,
        scenario_top_k: int = 1,
        scenario_min_obstacles: int = 0,
        scenario_max_avg_cost: Optional[float] = None,
        scenario_max_attempts: Optional[int] = None,
        scenario_dataset_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if grid_spec.s_samples < 2:
            raise ValueError("grid_spec must have at least 2 columns")
        self.grid_spec = grid_spec
        if min_obstacles < 0:
            raise ValueError("min_obstacles must be non-negative")
        if max_obstacles < min_obstacles:
            raise ValueError("max_obstacles must be >= min_obstacles")
        self.min_obstacles = int(min_obstacles)
        self.max_obstacles = max_obstacles
        self.obstacle_length_range = obstacle_length_range
        self.obstacle_width_range = obstacle_width_range
        self.avoid_obstacle_overlap = bool(avoid_obstacle_overlap)
        self.obstacle_overlap_clearance = float(obstacle_overlap_clearance)
        self.obstacle_sampling_attempts_per_obstacle = max(
            1, int(obstacle_sampling_attempts_per_obstacle)
        )
        self.smoothness_penalty = smoothness_penalty
        self.base_smoothness_penalty = base_smoothness_penalty
        self.move_limit_penalty_multiplier = move_limit_penalty_multiplier
        self.lateral_reference = lateral_reference
        self.reference_penalty = reference_penalty
        self.max_slope = max_slope
        self.slope_penalty = slope_penalty
        self.max_slope_penalty = max_slope_penalty
        self.collision_penalty = collision_penalty
        self.terminal_reward = terminal_reward
        self.non_goal_penalty = non_goal_penalty
        if no_valid_action_penalty is None:
            no_valid_action_penalty = collision_penalty
        self.no_valid_action_penalty = float(no_valid_action_penalty)
        self.lateral_move_limit = lateral_move_limit
        self.max_start_index = grid_spec.s_samples - 1
        if collision_inflation is not None:
            coarse_collision_inflation = collision_inflation
            fine_collision_inflation = collision_inflation
        if fine_collision_inflation < 0.0:
            raise ValueError("fine_collision_inflation must be non-negative")
        if coarse_collision_inflation < 0.0:
            raise ValueError("coarse_collision_inflation must be non-negative")
        self.coarse_collision_inflation = float(coarse_collision_inflation)
        self.fine_collision_inflation = float(fine_collision_inflation)
        interpolation_points = int(interpolation_points)
        if interpolation_points < 0:
            raise ValueError("interpolation_points must be non-negative")
        self.interpolation_points = interpolation_points
        vehicle_length = float(vehicle_length)
        vehicle_width = float(vehicle_width)
        if vehicle_length < 0.0 or vehicle_width < 0.0:
            raise ValueError("vehicle_length/vehicle_width must be non-negative")
        if (vehicle_length > 0.0) != (vehicle_width > 0.0):
            raise ValueError("vehicle_length/vehicle_width must both be zero or positive")
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        if not 0.0 <= start_clear_fraction <= 1.0:
            raise ValueError("start_clear_fraction must be in [0, 1]")
        self.start_clear_fraction = float(start_clear_fraction)
        self.scenario_pool_size = max(1, int(scenario_pool_size))
        self.scenario_top_k = max(1, int(scenario_top_k))
        self.scenario_min_obstacles = max(0, int(scenario_min_obstacles))
        if scenario_max_avg_cost is not None:
            scenario_max_avg_cost = float(scenario_max_avg_cost)
        self.scenario_max_avg_cost = scenario_max_avg_cost
        if scenario_max_attempts is not None:
            scenario_max_attempts = max(1, int(scenario_max_attempts))
        self.scenario_max_attempts = scenario_max_attempts
        self.scenario_dataset_path = str(scenario_dataset_path) if scenario_dataset_path else None
        self._scenario_dataset = self._load_scenario_dataset(self.scenario_dataset_path)
        self._rng = np.random.default_rng(seed)

        self._s_grid: np.ndarray
        self._l_grid: np.ndarray
        self._occupancy: np.ndarray
        self._obstacles: Sequence[Obstacle]
        self._path_indices: Sequence[int]
        self._s_index: int
        self._start_l: float
        self._last_action_mask: Optional[np.ndarray] = None
        self._last_scenario_dp_result: Optional[DPCandidateResult] = None

        self.reset()

    def _inflated_corners(self, obstacle: Obstacle) -> np.ndarray:
        """Return obstacle corners after applying collision inflation."""
        center = np.asarray(obstacle.center, dtype=float)
        scale = 1.0 + self.coarse_collision_inflation
        half_length = max(0.5 * obstacle.length * scale, 1e-6)
        half_width = max(0.5 * obstacle.width * scale, 1e-6)
        local = np.array(
            [
                [half_length, half_width],
                [half_length, -half_width],
                [-half_length, -half_width],
                [-half_length, half_width],
            ],
            dtype=float,
        )
        c, s = np.cos(obstacle.yaw), np.sin(obstacle.yaw)
        rotation = np.array([[c, -s], [s, c]], dtype=float)
        return local @ rotation.T + center

    @property
    def occupancy(self) -> np.ndarray:
        """Return the current occupancy grid (True = collision for the footprint)."""
        return self._occupancy.copy()

    @property
    def s_index(self) -> int:
        """Return the current longitudinal column index."""
        return self._s_index

    @property
    def path_indices(self) -> Sequence[int]:
        """Return the sequence of selected lateral indices so far."""
        return tuple(self._path_indices)

    @property
    def obstacles(self) -> Sequence[Obstacle]:
        """Return the current obstacle list."""
        return tuple(self._obstacles)

    @property
    def start_l(self) -> float:
        """Return the continuous starting lateral coordinate."""
        return float(self._start_l)

    @property
    def last_scenario_dp_result(self) -> Optional[DPCandidateResult]:
        """Return the most recent DP evaluation result for the sampled scenario."""
        return self._last_scenario_dp_result

    def _load_scenario_dataset(self, dataset_path: Optional[str]) -> Optional[Dict[str, object]]:
        if not dataset_path:
            return None
        dataset_file = Path(dataset_path).expanduser().resolve()
        cache_key = str(dataset_file)
        payload = _SCENARIO_DATASET_CACHE.get(cache_key)
        if payload is None:
            if not dataset_file.exists():
                raise FileNotFoundError(f"Scenario dataset not found: {dataset_file}")
            payload = json.loads(dataset_file.read_text(encoding="utf-8"))
            scenarios = payload.get("scenarios")
            if not isinstance(scenarios, list) or not scenarios:
                raise ValueError(f"Scenario dataset is empty or malformed: {dataset_file}")
            _SCENARIO_DATASET_CACHE[cache_key] = payload
        self._validate_scenario_dataset(payload, dataset_file)
        return payload

    def _validate_scenario_dataset(self, payload: Dict[str, object], dataset_file: Path) -> None:
        grid = payload.get("grid")
        if not isinstance(grid, dict):
            return
        expected = self.grid_spec
        s_range = tuple(grid.get("s_range", expected.s_range))
        l_range = tuple(grid.get("l_range", expected.l_range))
        s_samples = int(grid.get("s_samples", expected.s_samples))
        l_samples = int(grid.get("l_samples", expected.l_samples))
        if (
            tuple(map(float, s_range)) != tuple(map(float, expected.s_range))
            or tuple(map(float, l_range)) != tuple(map(float, expected.l_range))
            or s_samples != expected.s_samples
            or l_samples != expected.l_samples
        ):
            raise ValueError(
                "Scenario dataset grid does not match environment grid: "
                f"{dataset_file}"
            )

    def reset(self, start_l: float | None = None) -> Dict[str, np.ndarray]:
        """Reset the environment and return the initial observation."""
        spec = self.grid_spec
        self._s_grid, self._l_grid = build_grid(spec)

        (
            self._obstacles,
            self._occupancy,
            start_l_value,
            initial_l,
            self._last_scenario_dp_result,
        ) = self._sample_screened_scenario(start_l=start_l)
        self._path_indices = []
        l_coords = self._l_grid[0, :]
        if start_l is not None:
            start_l_value = float(np.clip(start_l, spec.l_range[0], spec.l_range[1]))
            initial_l = int(np.argmin(np.abs(l_coords - start_l_value)))
        self._start_l = start_l_value
        self._path_indices.append(int(initial_l))
        if self.grid_spec.s_samples > 1:
            self._s_index = 1
        else:
            self._s_index = 0
        return self._build_observation()

    def _sample_start_l_value(self, start_l: float | None) -> float:
        l_min, l_max = self.grid_spec.l_range
        if start_l is None:
            return float(self._rng.uniform(l_min, l_max))
        return float(np.clip(start_l, l_min, l_max))

    def _obstacle_from_record(self, entry: Dict[str, object]) -> Obstacle:
        center = entry.get("center", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError("Scenario obstacle center must contain exactly two values.")
        return Obstacle(
            center=(float(center[0]), float(center[1])),
            length=float(entry["length"]),
            width=float(entry["width"]),
            yaw=float(entry.get("yaw", 0.0)),
        )

    def _sample_precomputed_scenario(
        self,
        *,
        start_l: float | None = None,
    ) -> Tuple[List[Obstacle], np.ndarray, float, int, Optional[DPCandidateResult]]:
        if self._scenario_dataset is None:
            raise RuntimeError("Scenario dataset is not loaded.")
        scenarios = self._scenario_dataset["scenarios"]
        if not isinstance(scenarios, list) or not scenarios:
            raise RuntimeError("Scenario dataset contains no scenarios.")

        record = scenarios[int(self._rng.integers(0, len(scenarios)))]
        if not isinstance(record, dict):
            raise ValueError("Scenario record must be a JSON object.")
        raw_obstacles = record.get("obstacles", [])
        if not isinstance(raw_obstacles, list):
            raise ValueError("Scenario record obstacles must be a list.")
        obstacles = [self._obstacle_from_record(entry) for entry in raw_obstacles]
        occupancy = self._build_occupancy(obstacles)
        l_coords = self._l_grid[0, :]

        if start_l is None:
            start_l_value = float(record.get("start_l", self._sample_start_l_value(None)))
        else:
            start_l_value = self._sample_start_l_value(start_l)
        initial_l = int(np.argmin(np.abs(l_coords - start_l_value)))

        dp_result: Optional[DPCandidateResult] = None
        path_indices = record.get("path_indices")
        if (
            start_l is None
            and isinstance(path_indices, list)
            and path_indices
        ):
            dp_result = DPCandidateResult(
                feasible=True,
                total_cost=float(record.get("dp_total_cost", 0.0)),
                avg_cost=float(record.get("dp_avg_cost", 0.0)),
                path_indices=tuple(int(index) for index in path_indices),
            )
        return obstacles, occupancy, start_l_value, initial_l, dp_result

    def _generate_candidate_obstacles(self) -> List[Obstacle]:
        obstacles = generate_random_obstacles(
            self.grid_spec.s_range,
            self.grid_spec.l_range,
            min_count=self.min_obstacles,
            max_count=self.max_obstacles,
            length_range=self.obstacle_length_range,
            width_range=self.obstacle_width_range,
            avoid_overlap=self.avoid_obstacle_overlap,
            overlap_clearance=self.obstacle_overlap_clearance,
            max_sampling_attempts_per_obstacle=self.obstacle_sampling_attempts_per_obstacle,
            rng=self._rng,
        )
        if self.start_clear_fraction <= 0.0 or not obstacles:
            return obstacles
        s_min, s_max = self.grid_spec.s_range
        clear_limit = s_min + (s_max - s_min) * self.start_clear_fraction
        filtered: List[Obstacle] = []
        for obstacle in obstacles:
            corners = self._inflated_corners(obstacle)
            if corners[:, 0].min() < clear_limit:
                continue
            filtered.append(obstacle)
        return filtered

    def _is_screening_enabled(self) -> bool:
        return (
            self.scenario_pool_size > 1
            or self.scenario_min_obstacles > 0
            or self.scenario_max_avg_cost is not None
        )

    def _sample_screened_scenario(
        self,
        *,
        start_l: float | None = None,
    ) -> Tuple[List[Obstacle], np.ndarray, float, int, Optional[DPCandidateResult]]:
        if self._scenario_dataset is not None:
            return self._sample_precomputed_scenario(start_l=start_l)
        start_l_value = self._sample_start_l_value(start_l)
        l_coords = self._l_grid[0, :]
        initial_l = int(np.argmin(np.abs(l_coords - start_l_value)))

        if not self._is_screening_enabled():
            obstacles = self._generate_candidate_obstacles()
            occupancy = self._build_occupancy(obstacles)
            self._obstacles = obstacles
            return obstacles, occupancy, start_l_value, initial_l, None

        target_size = max(1, self.scenario_pool_size)
        max_attempts = self.scenario_max_attempts or max(target_size * 4, target_size)
        accepted: List[Tuple[List[Obstacle], np.ndarray, DPCandidateResult]] = []
        feasible_with_min_obstacles: List[Tuple[List[Obstacle], np.ndarray, DPCandidateResult]] = []
        feasible_any: List[Tuple[List[Obstacle], np.ndarray, DPCandidateResult]] = []
        last_candidate: Optional[Tuple[List[Obstacle], np.ndarray, DPCandidateResult]] = None

        for _ in range(max_attempts):
            obstacles = self._generate_candidate_obstacles()
            occupancy = self._build_occupancy(obstacles)
            self._obstacles = obstacles
            dp_result = self._evaluate_scenario_with_dp(initial_l)
            last_candidate = (obstacles, occupancy, dp_result)

            if not dp_result.feasible:
                continue
            feasible_any.append((obstacles, occupancy, dp_result))

            if len(obstacles) < self.scenario_min_obstacles:
                continue
            feasible_with_min_obstacles.append((obstacles, occupancy, dp_result))

            if (
                self.scenario_max_avg_cost is not None
                and dp_result.avg_cost > self.scenario_max_avg_cost
            ):
                continue
            accepted.append((obstacles, occupancy, dp_result))
            if len(accepted) >= target_size:
                break

        selected_pool = accepted or feasible_with_min_obstacles or feasible_any
        if selected_pool:
            selected_pool.sort(key=lambda item: (item[2].avg_cost, item[2].total_cost))
            top_k = min(len(selected_pool), self.scenario_top_k)
            pick_idx = int(self._rng.integers(0, top_k)) if top_k > 1 else 0
            obstacles, occupancy, dp_result = selected_pool[pick_idx]
            self._obstacles = obstacles
            return obstacles, occupancy, start_l_value, initial_l, dp_result

        if last_candidate is not None:
            obstacles, occupancy, dp_result = last_candidate
            self._obstacles = obstacles
            return obstacles, occupancy, start_l_value, initial_l, dp_result

        obstacles = self._generate_candidate_obstacles()
        occupancy = self._build_occupancy(obstacles)
        self._obstacles = obstacles
        dp_result = self._evaluate_scenario_with_dp(initial_l)
        return obstacles, occupancy, start_l_value, initial_l, dp_result

    def _dp_transition_cost(
        self,
        prev_s: float,
        prev_l: float,
        curr_s: float,
        curr_l: float,
    ) -> float:
        cost = abs(curr_l - prev_l) * self.base_smoothness_penalty
        delta_s = abs(curr_s - prev_s)
        if delta_s > 0.0:
            slope = abs((curr_l - prev_l) / delta_s)
            slope_excess = max(0.0, slope - self.max_slope)
            cost += slope_excess * self.max_slope_penalty
        cost += abs(curr_l - self.lateral_reference) * self.reference_penalty
        return float(cost)

    def _evaluate_scenario_with_dp(self, start_l_index: int) -> DPCandidateResult:
        s_count = self.grid_spec.s_samples
        l_count = self.grid_spec.l_samples
        if s_count <= 1:
            return DPCandidateResult(
                feasible=True,
                total_cost=0.0,
                avg_cost=0.0,
                path_indices=(int(start_l_index),),
            )

        costs = np.full((s_count, l_count), np.inf, dtype=np.float64)
        parents = np.full((s_count, l_count), -1, dtype=np.int32)
        costs[0, start_l_index] = 0.0
        s_coords = self._s_grid[:, 0]
        l_coords = self._l_grid[0, :]

        for s_idx in range(1, s_count):
            prev_s = float(s_coords[s_idx - 1])
            curr_s = float(s_coords[s_idx])
            for curr_l_idx in range(l_count):
                candidate_l_indices = range(l_count)
                if self.lateral_move_limit is not None:
                    low = max(0, curr_l_idx - int(self.lateral_move_limit))
                    high = min(l_count, curr_l_idx + int(self.lateral_move_limit) + 1)
                    candidate_l_indices = range(low, high)

                best_cost = math.inf
                best_parent = -1
                curr_l = float(l_coords[curr_l_idx])
                for prev_l_idx in candidate_l_indices:
                    prev_cost = float(costs[s_idx - 1, prev_l_idx])
                    if not math.isfinite(prev_cost):
                        continue
                    prev_l = float(l_coords[prev_l_idx])
                    if self._interpolated_hits_any_obstacle(prev_s, prev_l, curr_s, curr_l):
                        continue
                    total_cost = prev_cost + self._dp_transition_cost(prev_s, prev_l, curr_s, curr_l)
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_parent = prev_l_idx

                if best_parent >= 0:
                    costs[s_idx, curr_l_idx] = best_cost
                    parents[s_idx, curr_l_idx] = best_parent

        last_row = costs[s_count - 1]
        best_last_l = int(np.argmin(last_row))
        best_total_cost = float(last_row[best_last_l])
        if not math.isfinite(best_total_cost):
            return DPCandidateResult(
                feasible=False,
                total_cost=float("inf"),
                avg_cost=float("inf"),
                path_indices=(int(start_l_index),),
            )

        path = [best_last_l]
        curr_l_idx = best_last_l
        for s_idx in range(s_count - 1, 0, -1):
            curr_l_idx = int(parents[s_idx, curr_l_idx])
            if curr_l_idx < 0:
                return DPCandidateResult(
                    feasible=False,
                    total_cost=float("inf"),
                    avg_cost=float("inf"),
                    path_indices=(int(start_l_index),),
                )
            path.append(curr_l_idx)
        path.reverse()
        avg_cost = best_total_cost / max(1, s_count - 1)
        return DPCandidateResult(
            feasible=True,
            total_cost=best_total_cost,
            avg_cost=avg_cost,
            path_indices=tuple(int(idx) for idx in path),
        )

    def step(self, action: int) -> StepResult:
        """
        Advance the environment by choosing a lateral index for the next column.

        Args:
            action: lateral index for column `self._s_index`.

        Returns:
            StepResult containing new observation, reward, termination flag, info.
        """
        if not 0 <= action < self.grid_spec.l_samples:
            reward = self.collision_penalty
            done = True
            info = {"reason": "out_of_bounds"}
            if self.non_goal_penalty > 0.0:
                reward -= self.non_goal_penalty
                info["non_goal_penalty"] = self.non_goal_penalty
            observation = self._build_observation()
            return StepResult(observation, reward, done, info)

        action_mask = self._last_action_mask
        if action_mask is None:
            action_mask = self._compute_action_mask()
        if not np.any(action_mask):
            reward = self.no_valid_action_penalty
            done = True
            info = {
                "reason": "no_valid_action",
                "no_valid_action_penalty": self.no_valid_action_penalty,
            }
            if self.non_goal_penalty > 0.0:
                reward -= self.non_goal_penalty
                info["non_goal_penalty"] = self.non_goal_penalty
            self._path_indices.append(action)
            observation = self._build_observation()
            return StepResult(observation, reward, done, info)

        reward = 0.0
        done = False
        info: Dict[str, object] = {}

        # Enforce lateral move limit for smoother paths.
        if self.path_indices and self.lateral_move_limit is not None:
            prev = self.path_indices[-1]
            delta = abs(action - prev)
            if delta > self.lateral_move_limit:
                excess = delta - self.lateral_move_limit
                move_penalty = (
                    excess
                    * self.smoothness_penalty
                    * self.move_limit_penalty_multiplier
                )
                reward -= move_penalty
                info["move_limit_cost"] = move_penalty

        current_column = min(self._s_index, self.grid_spec.s_samples - 1)
        current_s = float(self._s_grid[current_column, 0])
        current_l = float(self._l_grid[current_column, action])

        prev_s = 0.0
        prev_l = 0.0
        if self._path_indices:
            prev_column = max(self._s_index - 1, 0)
            prev_column = min(prev_column, self.grid_spec.s_samples - 1)
            prev_l = float(self._l_grid[prev_column, self._path_indices[-1]])
            prev_s = float(self._s_grid[prev_column, 0])

        # Collision check.
        if self._path_indices and self._interpolated_hits_any_obstacle(
            prev_s, prev_l, current_s, current_l
        ):
            reward = self.collision_penalty
            done = True
            info["reason"] = "collision"
            if self.non_goal_penalty > 0.0:
                reward -= self.non_goal_penalty
                info["non_goal_penalty"] = self.non_goal_penalty
            self._path_indices.append(action)
            observation = self._build_observation()
            return StepResult(observation, reward, done, info)

        # Smoothness penalties.
        if self._path_indices:
            delta_s = abs(current_s - prev_s)

            # 基础一阶平滑性：任何横向变化都扣分。
            smoothness_cost = abs(current_l - prev_l) * self.base_smoothness_penalty
            reward -= smoothness_cost
            info["smoothness_cost"] = smoothness_cost

            # 斜率超阈惩罚。
            if delta_s > 0.0:
                slope = abs((current_l - prev_l) / delta_s)
                slope_excess = max(0.0, slope - self.max_slope)
                slope_cost = -slope_excess * self.max_slope_penalty
                reward += slope_cost
                info["slope_cost"] = slope_cost

        # Penalize deviation from the reference line.
        ref_cost = abs(current_l - self.lateral_reference) * self.reference_penalty
        reward -= ref_cost
        info["reference_cost"] = ref_cost

        self._path_indices.append(action)
        self._s_index += 1

        if self._s_index >= self.grid_spec.s_samples:
            done = True
            reward += self.terminal_reward
            info["reason"] = "goal_reached"

        observation = self._build_observation()
        return StepResult(observation, reward, done, info)

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Construct the observation dictionary for the current state."""
        action_mask = self._compute_action_mask()
        self._last_action_mask = action_mask.copy()
        return {
            "s_index": np.array(self._s_index, dtype=np.int32),
            "path_indices": np.array(self._path_indices, dtype=np.int32),
            "occupancy": self._occupancy.copy(),
            "s_coords": self._s_grid[:, 0],
            "l_coords": self._l_grid[0, :],
            "start_l": np.array(self._start_l, dtype=np.float32),
            "action_mask": action_mask,
            "obstacle_corners": self._obstacle_corners_array(),
        }

    def _build_occupancy(self, obstacles: Sequence[Obstacle]) -> np.ndarray:
        """Rasterize obstacles onto the discrete grid (with optional vehicle footprint)."""
        spec = self.grid_spec
        occupancy = np.zeros((spec.s_samples, spec.l_samples), dtype=bool)
        s_coords = self._s_grid[:, 0]
        l_coords = self._l_grid[0, :]

        coarse_scale = 1.0 + self.coarse_collision_inflation
        fine_scale = 1.0 + self.fine_collision_inflation
        eps = 1e-6
        use_vehicle_footprint = self.vehicle_length > 0.0 and self.vehicle_width > 0.0
        vehicle_radius = 0.0
        if use_vehicle_footprint:
            veh_half_len = max(0.5 * self.vehicle_length, eps)
            veh_half_wid = max(0.5 * self.vehicle_width, eps)
            vehicle_radius = math.hypot(veh_half_len, veh_half_wid)
        vehicle_heading = 0.0  # assume vehicle aligned with s-axis for occupancy

        for obstacle in obstacles:
            center = np.asarray(obstacle.center, dtype=float)
            coarse_half_length = max(0.5 * obstacle.length * coarse_scale, eps)
            coarse_half_width = max(0.5 * obstacle.width * coarse_scale, eps)
            fine_half_length = max(0.5 * obstacle.length * fine_scale, eps)
            fine_half_width = max(0.5 * obstacle.width * fine_scale, eps)

            c, s = np.cos(obstacle.yaw), np.sin(obstacle.yaw)
            rotation = np.array([[c, -s], [s, c]], dtype=float)

            coarse_corners = np.array(
                [
                    [coarse_half_length, coarse_half_width],
                    [coarse_half_length, -coarse_half_width],
                    [-coarse_half_length, -coarse_half_width],
                    [-coarse_half_length, coarse_half_width],
                ],
                dtype=float,
            )
            coarse_corners = coarse_corners @ rotation.T + center
            s_min, l_min = coarse_corners.min(axis=0)
            s_max, l_max = coarse_corners.max(axis=0)
            if use_vehicle_footprint:
                s_min -= vehicle_radius
                s_max += vehicle_radius
                l_min -= vehicle_radius
                l_max += vehicle_radius

            s_min = max(s_min, spec.s_range[0])
            s_max = min(s_max, spec.s_range[1])
            l_min = max(l_min, spec.l_range[0])
            l_max = min(l_max, spec.l_range[1])

            s_mask = (s_coords >= s_min) & (s_coords <= s_max)
            l_mask = (l_coords >= l_min) & (l_coords <= l_max)
            s_idx = np.where(s_mask)[0]
            l_idx = np.where(l_mask)[0]
            if s_idx.size == 0 or l_idx.size == 0:
                continue

            if use_vehicle_footprint:
                for s_index in s_idx:
                    s_value = float(s_coords[s_index])
                    for l_index in l_idx:
                        if occupancy[s_index, l_index]:
                            continue
                        l_value = float(l_coords[l_index])
                        if self._footprint_hits_obstacle(
                            s_value, l_value, vehicle_heading, obstacle
                        ):
                            occupancy[s_index, l_index] = True
                continue

            pts_s, pts_l = np.meshgrid(s_coords[s_idx], l_coords[l_idx], indexing="ij")
            points = np.stack([pts_s, pts_l], axis=-1).reshape(-1, 2)
            local_points = (points - center) @ rotation

            inside = (np.abs(local_points[:, 0]) <= fine_half_length + eps) & (
                np.abs(local_points[:, 1]) <= fine_half_width + eps
            )
            if not inside.any():
                continue
            inside = inside.reshape(len(s_idx), len(l_idx))
            occupancy[np.ix_(s_idx, l_idx)] |= inside

        return occupancy

    def _obstacle_corners_array(self) -> np.ndarray:
        """Return current obstacle corners as an array shaped (N, 4, 2)."""
        if not self._obstacles:
            return np.zeros((0, 4, 2), dtype=np.float32)
        corners = [np.asarray(obstacle.corners(), dtype=np.float32) for obstacle in self._obstacles]
        return np.stack(corners, axis=0)

    @staticmethod
    def _obb_intersects_obb(
        c0_s: float,
        c0_l: float,
        half_len0: float,
        half_wid0: float,
        yaw0: float,
        c1_s: float,
        c1_l: float,
        half_len1: float,
        half_wid1: float,
        yaw1: float,
    ) -> bool:
        cos0 = math.cos(yaw0)
        sin0 = math.sin(yaw0)
        cos1 = math.cos(yaw1)
        sin1 = math.sin(yaw1)

        u0x, u0y = cos0, sin0
        v0x, v0y = -sin0, cos0
        u1x, u1y = cos1, sin1
        v1x, v1y = -sin1, cos1

        t_x = c1_s - c0_s
        t_y = c1_l - c0_l
        eps = 1e-9

        def axis_separates(ax: float, ay: float) -> bool:
            ra = half_len0 * abs(u0x * ax + u0y * ay) + half_wid0 * abs(v0x * ax + v0y * ay)
            rb = half_len1 * abs(u1x * ax + u1y * ay) + half_wid1 * abs(v1x * ax + v1y * ay)
            dist = abs(t_x * ax + t_y * ay)
            return dist > (ra + rb + eps)

        if axis_separates(u0x, u0y):
            return False
        if axis_separates(v0x, v0y):
            return False
        if axis_separates(u1x, u1y):
            return False
        if axis_separates(v1x, v1y):
            return False
        return True

    def _point_hits_obstacle(
        self,
        s_value: float,
        l_value: float,
        obstacle: Obstacle,
    ) -> bool:
        center_s, center_l = obstacle.center
        cos_yaw, sin_yaw = math.cos(obstacle.yaw), math.sin(obstacle.yaw)
        ds = s_value - center_s
        dl = l_value - center_l

        local_s = cos_yaw * ds + sin_yaw * dl
        local_l = -sin_yaw * ds + cos_yaw * dl

        scale = 1.0 + self.fine_collision_inflation
        half_len = max(0.5 * obstacle.length * scale, 1e-6)
        half_wid = max(0.5 * obstacle.width * scale, 1e-6)

        return abs(local_s) <= half_len and abs(local_l) <= half_wid

    def _footprint_hits_obstacle(
        self,
        s_value: float,
        l_value: float,
        heading: float,
        obstacle: Obstacle,
    ) -> bool:
        if self.vehicle_length <= 0.0 or self.vehicle_width <= 0.0:
            return self._point_hits_obstacle(s_value, l_value, obstacle)

        scale = 1.0 + self.fine_collision_inflation
        obs_half_len = max(0.5 * obstacle.length * scale, 1e-6)
        obs_half_wid = max(0.5 * obstacle.width * scale, 1e-6)
        veh_half_len = max(0.5 * self.vehicle_length, 1e-6)
        veh_half_wid = max(0.5 * self.vehicle_width, 1e-6)

        center_s, center_l = obstacle.center
        return self._obb_intersects_obb(
            s_value,
            l_value,
            veh_half_len,
            veh_half_wid,
            heading,
            center_s,
            center_l,
            obs_half_len,
            obs_half_wid,
            obstacle.yaw,
        )

    def _footprint_hits_any_obstacle(
        self,
        s_value: float,
        l_value: float,
        heading: float,
    ) -> bool:
        if not self._obstacles:
            return False
        for obstacle in self._obstacles:
            if self._footprint_hits_obstacle(s_value, l_value, heading, obstacle):
                return True
        return False

    def _interpolated_hits_any_obstacle(
        self,
        start_s: float,
        start_l: float,
        end_s: float,
        end_l: float,
    ) -> bool:
        if not self._obstacles:
            return False
        delta_s = end_s - start_s
        delta_l = end_l - start_l
        heading = math.atan2(delta_l, delta_s) if delta_s or delta_l else 0.0
        if self._footprint_hits_any_obstacle(end_s, end_l, heading):
            return True
        count = self.interpolation_points
        if count <= 0:
            return False
        t_values = [(idx + 1) / (count + 1) for idx in range(count)]
        for t in t_values:
            s_mid = start_s + (end_s - start_s) * t
            l_mid = start_l + (end_l - start_l) * t
            if self._footprint_hits_any_obstacle(s_mid, l_mid, heading):
                return True
        return False

    def _sample_start_index(self) -> int:
        """Sample a valid starting column index."""
        lower = 0
        upper = min(self.max_start_index, self.grid_spec.s_samples - 1)
        if lower > upper:
            lower, upper = 0, self.grid_spec.s_samples - 1
        return int(self._rng.integers(lower, upper + 1))

    def _compute_action_mask(self) -> np.ndarray:
        """
        Build a boolean mask of valid actions for the next step.
        - 若设置了 lateral_move_limit，则屏蔽超过最大横向跳变的动作
        - 插值点判碰：若车辆在插值点的车身与障碍矩形相交，则屏蔽对应动作
        """
        l_count = self.grid_spec.l_samples
        mask = np.ones(l_count, dtype=bool)

        # 避免越界：到达终点或超出时直接返回全 False（上层会兜底）。
        if self._s_index >= self.grid_spec.s_samples:
            return np.zeros(l_count, dtype=bool)

        # 屏蔽过大的横向跳变。
        if self.lateral_move_limit is not None and self._path_indices:
            prev_idx = int(self._path_indices[-1])
            max_delta = int(self.lateral_move_limit)
            idxs = np.arange(l_count, dtype=int)
            valid_move = np.abs(idxs - prev_idx) <= max_delta
            mask &= valid_move

        # 插值点碰撞检查：若车身在任一插值点与障碍相交，则屏蔽该动作。
        if self._path_indices:
            prev_idx = int(self._path_indices[-1])
            prev_col = max(self._s_index - 1, 0)
            prev_col = min(prev_col, self.grid_spec.s_samples - 1)
            prev_s = float(self._s_grid[prev_col, 0])
            prev_l = float(self._l_grid[prev_col, prev_idx])
            curr_col = min(self._s_index, self.grid_spec.s_samples - 1)
            curr_s = float(self._s_grid[curr_col, 0])

            candidate_idxs = np.where(mask)[0]
            if candidate_idxs.size > 0:
                l_coords = self._l_grid[0, :]
                for cand_idx in candidate_idxs:
                    curr_l = float(l_coords[cand_idx])
                    if self._interpolated_hits_any_obstacle(prev_s, prev_l, curr_s, curr_l):
                        mask[cand_idx] = False

        return mask


def run_random_episode(env: SLPathEnv) -> StepResult:
    """Roll out a single episode with random actions."""
    observation = env.reset()
    done = False
    total_reward = 0.0
    info: Dict[str, object] = {}

    while not done:
        action = env._rng.integers(0, env.grid_spec.l_samples)  # random policy
        result = env.step(int(action))
        observation = result.observation
        total_reward += result.reward
        done = result.done
        info = result.info

    info = dict(info)
    info["total_reward"] = total_reward
    info["path_indices"] = observation["path_indices"]
    return StepResult(observation, total_reward, True, info)


if __name__ == "__main__":
    spec = default_training_grid_spec()
    env = SLPathEnv(spec, seed=42, max_obstacles=5, lateral_move_limit=3)
    result = run_random_episode(env)
    print("Episode finished:", result.info)
