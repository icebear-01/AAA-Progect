"""Guided Hybrid A* with kinematic bicycle rollout.

Coordinate convention:
- World/state coordinate is (x, y, yaw).
- NumPy indexing is map[y, x].

This implementation is intended as a practical full-featured baseline:
- Kinematic bicycle motion primitives
- Forward / reverse expansion
- Steering, steering-change, reverse, and gear-switch penalties
- Guidance integration: g += lambda_guidance * cost_map[y, x]
- Vehicle footprint collision checking (disc approximation)
- Optional obstacle-aware holonomic heuristic (2D Dijkstra)
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from neural_astar.utils.coords import validate_xy
from .rs_shot import find_rs_shot


Pose = Tuple[float, float, float]  # (x, y, yaw_rad)
NodeKey = Tuple[int, int, int, int]  # (x_idx, y_idx, yaw_idx, dir_idx)


@dataclass
class GuidedHybridAstarResult:
    success: bool
    path: List[Pose]
    expanded_nodes: int
    runtime_ms: float
    path_length: float
    expanded_trace_xy: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class NodeRecord:
    pose: Pose
    g: float
    parent: Optional[NodeKey]
    direction: int  # +1 forward, -1 reverse
    steer: float
    traj_from_parent: List[Pose]


class GuidedHybridAstar:
    def __init__(
        self,
        xy_resolution: float = 1.0,
        yaw_bins: int = 72,
        wheel_base: float = 2.7,
        max_steer: float = 0.60,
        n_steer: int = 5,
        motion_step: float = 0.5,
        primitive_length: float = 2.5,
        allow_reverse: bool = True,
        reverse_penalty: float = 2.0,
        steer_penalty: float = 0.1,
        steer_change_penalty: float = 0.2,
        direction_change_penalty: float = 1.0,
        lambda_guidance: float = 0.0,
        guidance_integration_mode: str = "g_cost",
        normalize_guidance_cost: bool = True,
        guidance_norm_p_low: float = 5.0,
        guidance_norm_p_high: float = 95.0,
        guidance_clip_low: float = 0.05,
        guidance_clip_high: float = 0.95,
        guidance_temperature: float = 1.0,
        guidance_power: float = 1.0,
        guidance_bonus_threshold: float = 0.5,
        vehicle_radius: float = 0.9,
        use_holonomic_heuristic: bool = True,
        goal_tolerance_xy: float = 1.5,
        goal_tolerance_yaw_deg: float = 5.0,
        use_rs_shot: bool = False,
        rs_shot_trigger_dist: float = 5.0,
        rs_sample_ds: float = 0.25,
        rs_endpoint_tol: float = 1e-3,
        rs_max_iter: int = 28,
        strict_goal_pose: bool = False,
        # Backward-compatible aliases used by earlier demo code.
        theta_bins: Optional[int] = None,
        step_size: Optional[float] = None,
    ):
        if theta_bins is not None:
            yaw_bins = int(theta_bins)
        if step_size is not None:
            primitive_length = float(step_size)

        self.xy_resolution = float(xy_resolution)
        self.yaw_bins = int(yaw_bins)
        self.wheel_base = float(wheel_base)
        self.max_steer = float(max_steer)
        self.n_steer = int(max(3, n_steer))
        self.motion_step = float(motion_step)
        self.primitive_length = float(primitive_length)
        self.allow_reverse = bool(allow_reverse)

        self.reverse_penalty = float(reverse_penalty)
        self.steer_penalty = float(steer_penalty)
        self.steer_change_penalty = float(steer_change_penalty)
        self.direction_change_penalty = float(direction_change_penalty)
        self.lambda_guidance = float(lambda_guidance)
        self.guidance_integration_mode = str(guidance_integration_mode)
        self.normalize_guidance_cost = bool(normalize_guidance_cost)
        self.guidance_norm_p_low = float(guidance_norm_p_low)
        self.guidance_norm_p_high = float(guidance_norm_p_high)
        self.guidance_clip_low = float(guidance_clip_low)
        self.guidance_clip_high = float(guidance_clip_high)
        self.guidance_temperature = float(guidance_temperature)
        self.guidance_power = float(guidance_power)
        self.guidance_bonus_threshold = float(guidance_bonus_threshold)
        if self.guidance_norm_p_low > self.guidance_norm_p_high:
            raise ValueError(
                f"guidance_norm_p_low > guidance_norm_p_high: "
                f"{self.guidance_norm_p_low} > {self.guidance_norm_p_high}"
            )
        if self.guidance_clip_low > self.guidance_clip_high:
            raise ValueError(
                f"guidance_clip_low > guidance_clip_high: "
                f"{self.guidance_clip_low} > {self.guidance_clip_high}"
            )
        if self.guidance_temperature <= 0.0:
            raise ValueError(
                f"guidance_temperature must be > 0, got {self.guidance_temperature}"
            )
        if self.guidance_power <= 0.0:
            raise ValueError(f"guidance_power must be > 0, got {self.guidance_power}")
        if self.guidance_bonus_threshold <= 0.0 or self.guidance_bonus_threshold > 1.0:
            raise ValueError(
                "guidance_bonus_threshold must be in (0, 1], got "
                f"{self.guidance_bonus_threshold}"
            )
        if self.guidance_integration_mode not in {"g_cost", "heuristic_bias", "heuristic_bonus"}:
            raise ValueError(
                f"Unknown guidance_integration_mode: {self.guidance_integration_mode}"
            )

        self.vehicle_radius = float(vehicle_radius)
        self.use_holonomic_heuristic = bool(use_holonomic_heuristic)
        self.goal_tolerance_xy = float(goal_tolerance_xy)
        self.goal_tolerance_yaw = math.radians(float(goal_tolerance_yaw_deg))
        self.use_rs_shot = bool(use_rs_shot)
        self.rs_shot_trigger_dist = float(rs_shot_trigger_dist)
        self.rs_sample_ds = float(rs_sample_ds)
        self.rs_endpoint_tol = float(rs_endpoint_tol)
        self.rs_max_iter = int(rs_max_iter)
        self.strict_goal_pose = bool(strict_goal_pose)

        self._steer_candidates = np.linspace(-self.max_steer, self.max_steer, self.n_steer)
        self._direction_candidates = [1, -1] if self.allow_reverse else [1]

    @staticmethod
    def _normalize_angle(theta: float) -> float:
        return math.atan2(math.sin(theta), math.cos(theta))

    def _yaw_to_bin(self, yaw: float) -> int:
        yaw_n = self._normalize_angle(yaw)
        t = (yaw_n + math.pi) / (2.0 * math.pi)
        return self._yaw_to_bin_count(yaw_n, self.yaw_bins)

    @staticmethod
    def _yaw_to_bin_count(yaw: float, yaw_bins: int) -> int:
        yaw_n = math.atan2(math.sin(yaw), math.cos(yaw))
        t = (yaw_n + math.pi) / (2.0 * math.pi)
        return int(round(t * int(yaw_bins))) % int(yaw_bins)

    def _pose_to_key(self, pose: Pose, direction: int) -> NodeKey:
        x, y, yaw = pose
        x_idx = int(round(x / self.xy_resolution))
        y_idx = int(round(y / self.xy_resolution))
        yaw_idx = self._yaw_to_bin(yaw)
        dir_idx = 0 if direction >= 0 else 1
        return x_idx, y_idx, yaw_idx, dir_idx

    def _pose_to_grid_xy(self, pose: Pose) -> Tuple[int, int]:
        x, y, _ = pose
        return int(round(x)), int(round(y))

    def _build_collision_offsets(self) -> List[Tuple[int, int]]:
        r = int(math.ceil(self.vehicle_radius))
        offsets: List[Tuple[int, int]] = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= self.vehicle_radius * self.vehicle_radius:
                    offsets.append((dx, dy))
        return offsets

    @staticmethod
    def _compute_holonomic_heuristic(occ: np.ndarray, goal_xy: Tuple[int, int]) -> np.ndarray:
        """2D obstacle-aware lower bound via Dijkstra (8-neighbor)."""
        h, w = occ.shape
        gx, gy = goal_xy

        inf = float("inf")
        dist = np.full((h, w), inf, dtype=np.float32)
        if occ[gy, gx] > 0.5:
            return dist

        pq: List[Tuple[float, int, int]] = []
        dist[gy, gx] = 0.0
        heapq.heappush(pq, (0.0, gx, gy))

        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        while pq:
            d, x, y = heapq.heappop(pq)
            if d > dist[y, x]:
                continue

            for dx, dy, c in neighbors:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if occ[ny, nx] > 0.5:
                    continue
                nd = d + c
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))

        return dist

    def _in_collision(
        self,
        pose: Pose,
        occ: np.ndarray,
        collision_offsets: List[Tuple[int, int]],
    ) -> bool:
        h, w = occ.shape
        cx, cy = self._pose_to_grid_xy(pose)

        for dx, dy in collision_offsets:
            x = cx + dx
            y = cy + dy
            if x < 0 or x >= w or y < 0 or y >= h:
                return True
            if occ[y, x] > 0.5:
                return True
        return False

    def _simulate_primitive(
        self,
        pose: Pose,
        steer: float,
        direction: int,
        occ: np.ndarray,
        cost_map: np.ndarray,
        collision_offsets: List[Tuple[int, int]],
    ) -> Optional[Tuple[Pose, List[Pose], float, float]]:
        """Rollout one bicycle-model primitive.

        Returns:
            (end_pose, trajectory_poses, traveled_length, mean_guidance_cost)
        """
        x, y, yaw = pose
        n_steps = max(2, int(round(self.primitive_length / self.motion_step)))

        traj: List[Pose] = [pose]
        guidance_acc = 0.0
        traveled = 0.0

        for _ in range(n_steps):
            x += direction * self.motion_step * math.cos(yaw)
            y += direction * self.motion_step * math.sin(yaw)
            yaw += direction * self.motion_step / self.wheel_base * math.tan(steer)
            yaw = self._normalize_angle(yaw)

            state = (x, y, yaw)
            if self._in_collision(state, occ, collision_offsets):
                return None

            xi, yi = self._pose_to_grid_xy(state)
            guidance_acc += self._lookup_guidance_cost(cost_map, state)
            traveled += self.motion_step
            traj.append(state)

        end_pose = traj[-1]
        mean_guidance = guidance_acc / max(1, n_steps)
        return end_pose, traj, traveled, mean_guidance

    def _heuristic(
        self,
        pose: Pose,
        goal_pose: Pose,
        holonomic_dist: Optional[np.ndarray],
    ) -> float:
        x, y, _ = pose
        gx, gy, _ = goal_pose
        h_euclid = math.hypot(gx - x, gy - y)

        if holonomic_dist is None:
            return h_euclid

        xi, yi = int(round(x)), int(round(y))
        hh = float(holonomic_dist[yi, xi])
        if not math.isfinite(hh):
            return h_euclid + 1e3
        return max(h_euclid, hh)

    def _is_goal(self, pose: Pose, goal_pose: Pose) -> bool:
        x, y, yaw = pose
        gx, gy, gyaw = goal_pose
        if self.strict_goal_pose:
            return (
                abs(gx - x) <= 1e-6
                and abs(gy - y) <= 1e-6
                and abs(self._normalize_angle(yaw - gyaw)) <= 1e-6
            )
        if math.hypot(gx - x, gy - y) > self.goal_tolerance_xy:
            return False
        dyaw = abs(self._normalize_angle(yaw - gyaw))
        return dyaw <= self.goal_tolerance_yaw

    def _try_rs_shot(
        self,
        current_pose: Pose,
        goal_pose: Pose,
        occ_map: np.ndarray,
        collision_offsets: List[Tuple[int, int]],
    ) -> Optional[List[Pose]]:
        max_curvature = math.tan(self.max_steer) / max(self.wheel_base, 1e-6)
        candidate = find_rs_shot(
            start=current_pose,
            goal=goal_pose,
            max_curvature=max_curvature,
            allow_reverse=self.allow_reverse,
            collision_checker=lambda p: self._in_collision(p, occ_map, collision_offsets),
            sample_ds=self.rs_sample_ds,
            endpoint_tol=self.rs_endpoint_tol,
            max_iter=self.rs_max_iter,
        )
        if candidate is None:
            return None

        return candidate.poses

    @staticmethod
    def _path_length(path: List[Pose]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for (x0, y0, _), (x1, y1, _) in zip(path[:-1], path[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    @staticmethod
    def _reconstruct_path(records: Dict[NodeKey, NodeRecord], goal_key: NodeKey) -> List[Pose]:
        segments: List[List[Pose]] = []
        key = goal_key

        while True:
            rec = records[key]
            if rec.parent is None:
                start_pose = rec.pose
                break
            segments.append(rec.traj_from_parent)
            key = rec.parent

        path: List[Pose] = [start_pose]
        for seg in reversed(segments):
            path.extend(seg[1:])
        return path

    @staticmethod
    def _collect_expanded_trace_xy(
        records: Dict[NodeKey, NodeRecord],
        closed: set[NodeKey],
    ) -> List[Tuple[float, float]]:
        """Collect rollout points from expanded nodes for visualization."""
        trace: List[Tuple[float, float]] = []
        for key in closed:
            rec = records.get(key)
            if rec is None:
                continue
            for px, py, _ in rec.traj_from_parent:
                trace.append((float(px), float(py)))
        return trace

    def _prepare_guidance_channel(self, cost_channel: np.ndarray, occ_map: np.ndarray) -> np.ndarray:
        cost = np.asarray(cost_channel, dtype=np.float32).copy()
        occ = np.asarray(occ_map, dtype=np.float32)
        free = occ < 0.5

        if self.normalize_guidance_cost and np.any(free):
            v = cost[free]
            lo = float(np.percentile(v, self.guidance_norm_p_low))
            hi = float(np.percentile(v, self.guidance_norm_p_high))
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo + 1e-6:
                cost = (cost - lo) / (hi - lo)

        cost = np.clip(cost, self.guidance_clip_low, self.guidance_clip_high)
        if not math.isclose(self.guidance_temperature, 1.0):
            eps = 1e-4
            cost = np.clip(cost, eps, 1.0 - eps)
            logits = np.log(cost) - np.log1p(-cost)
            cost = 1.0 / (1.0 + np.exp(-(logits / self.guidance_temperature)))
        if not math.isclose(self.guidance_power, 1.0):
            cost = np.power(np.clip(cost, 0.0, 1.0), self.guidance_power)
        cost = np.where(occ > 0.5, 1.0, cost)
        return cost.astype(np.float32)

    def _prepare_guidance_cost(self, cost_map: np.ndarray, occ_map: np.ndarray) -> np.ndarray:
        """Normalize + clip guidance cost to avoid local extreme traps."""
        cost = np.asarray(cost_map, dtype=np.float32)
        occ = np.asarray(occ_map, dtype=np.float32)
        if cost.ndim == 2:
            if cost.shape != occ.shape:
                raise ValueError(f"occ_map and cost_map shape mismatch: {occ.shape} vs {cost.shape}")
            return self._prepare_guidance_channel(cost, occ)
        if cost.ndim == 3:
            if cost.shape[1:] != occ.shape:
                raise ValueError(
                    f"occ_map and cost_map shape mismatch: {occ.shape} vs {cost.shape[1:]}"
                )
            prepared = np.stack(
                [self._prepare_guidance_channel(cost[idx], occ) for idx in range(cost.shape[0])],
                axis=0,
            )
        return prepared.astype(np.float32)

    def _guidance_priority_bias(self, mean_guidance: float) -> float:
        if self.guidance_integration_mode == "heuristic_bias":
            return float(mean_guidance)
        if self.guidance_integration_mode == "heuristic_bonus":
            threshold = self.guidance_bonus_threshold
            scaled_bonus = (float(mean_guidance) - threshold) / max(threshold, 1e-6)
            return min(0.0, scaled_bonus)
        return 0.0
        raise ValueError(f"cost_map must be [H,W] or [K,H,W], got {cost.shape}")

    def _lookup_guidance_cost(self, cost_map: np.ndarray, pose: Pose) -> float:
        xi, yi = self._pose_to_grid_xy(pose)
        if cost_map.ndim == 2:
            return float(cost_map[yi, xi])
        yaw_idx = self._yaw_to_bin_count(pose[2], int(cost_map.shape[0]))
        return float(cost_map[yaw_idx, yi, xi])

    def plan(
        self,
        occ_map: np.ndarray,
        start_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        cost_map: np.ndarray,
        lambda_guidance: Optional[float] = None,
        max_expansions: int = 80000,
    ) -> GuidedHybridAstarResult:
        """Run guided Hybrid A* planning.

        g-cost update during expansion:
            g_new = g_old + motion_cost + lambda_guidance * guidance_cost
        where guidance_cost is averaged along the rolled-out primitive.

        Heading handling:
        - Start heading is strict: planning starts exactly from given (x, y, yaw).
        - If `strict_goal_pose=False`, goal heading/position use tolerances.
        - If `strict_goal_pose=True`, success requires exact terminal pose, and
          RS-shot analytic expansion is used to connect to goal.
        """
        t0 = time.perf_counter()

        occ = np.asarray(occ_map, dtype=np.float32)
        raw_cost = np.asarray(cost_map, dtype=np.float32)
        cost = self._prepare_guidance_cost(raw_cost, occ)

        h, w = occ.shape

        sx, sy, syaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        gx, gy, gyaw = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])

        validate_xy(int(round(sx)), int(round(sy)), w, h)
        validate_xy(int(round(gx)), int(round(gy)), w, h)

        start = (sx, sy, self._normalize_angle(syaw))
        goal = (gx, gy, self._normalize_angle(gyaw))

        lam = self.lambda_guidance if lambda_guidance is None else float(lambda_guidance)

        collision_offsets = self._build_collision_offsets()
        if self._in_collision(start, occ, collision_offsets) or self._in_collision(
            goal, occ, collision_offsets
        ):
            return GuidedHybridAstarResult(
                success=False,
                path=[],
                expanded_nodes=0,
                runtime_ms=(time.perf_counter() - t0) * 1e3,
                path_length=0.0,
                expanded_trace_xy=[],
            )

        holonomic_dist: Optional[np.ndarray] = None
        if self.use_holonomic_heuristic:
            holonomic_dist = self._compute_holonomic_heuristic(
                occ, (int(round(goal[0])), int(round(goal[1])))
            )

        start_key = self._pose_to_key(start, direction=1)
        records: Dict[NodeKey, NodeRecord] = {
            start_key: NodeRecord(
                pose=start,
                g=0.0,
                parent=None,
                direction=1,
                steer=0.0,
                traj_from_parent=[start],
            )
        }

        open_heap: List[Tuple[float, int, NodeKey]] = []
        counter = itertools.count()
        f0 = self._heuristic(start, goal, holonomic_dist)
        heapq.heappush(open_heap, (f0, next(counter), start_key))

        closed: set[NodeKey] = set()
        expanded_nodes = 0

        while open_heap and expanded_nodes < max_expansions:
            _, _, current_key = heapq.heappop(open_heap)
            if current_key in closed:
                continue
            closed.add(current_key)

            current = records[current_key]
            expanded_nodes += 1

            dist_to_goal = math.hypot(goal[0] - current.pose[0], goal[1] - current.pose[1])
            can_try_rs = self.use_rs_shot and (dist_to_goal <= self.rs_shot_trigger_dist)
            if can_try_rs:
                rs_try = self._try_rs_shot(
                    current_pose=current.pose,
                    goal_pose=goal,
                    occ_map=occ,
                    collision_offsets=collision_offsets,
                )
                if rs_try is not None:
                    shot_path = rs_try
                    prefix = self._reconstruct_path(records, current_key)
                    full_path = prefix + shot_path[1:]
                    return GuidedHybridAstarResult(
                        success=True,
                        path=full_path,
                        expanded_nodes=expanded_nodes,
                        runtime_ms=(time.perf_counter() - t0) * 1e3,
                        path_length=self._path_length(full_path),
                        expanded_trace_xy=self._collect_expanded_trace_xy(records, closed),
                    )

            if self._is_goal(current.pose, goal):
                path = self._reconstruct_path(records, current_key)
                return GuidedHybridAstarResult(
                    success=True,
                    path=path,
                    expanded_nodes=expanded_nodes,
                    runtime_ms=(time.perf_counter() - t0) * 1e3,
                    path_length=self._path_length(path),
                    expanded_trace_xy=self._collect_expanded_trace_xy(records, closed),
                )

            for direction in self._direction_candidates:
                for steer in self._steer_candidates:
                    rollout = self._simulate_primitive(
                        pose=current.pose,
                        steer=float(steer),
                        direction=int(direction),
                        occ=occ,
                        cost_map=cost,
                        collision_offsets=collision_offsets,
                    )
                    if rollout is None:
                        continue

                    next_pose, traj, traveled, mean_guidance = rollout
                    next_key = self._pose_to_key(next_pose, direction=int(direction))

                    add_cost = traveled
                    if self.guidance_integration_mode == "g_cost":
                        add_cost += lam * mean_guidance
                    add_cost += self.steer_penalty * abs(float(steer))
                    add_cost += self.steer_change_penalty * abs(float(steer) - current.steer)
                    if direction < 0:
                        add_cost += self.reverse_penalty * traveled
                    if current.parent is not None and direction != current.direction:
                        add_cost += self.direction_change_penalty

                    g_new = current.g + add_cost

                    if next_key not in records or g_new < records[next_key].g:
                        records[next_key] = NodeRecord(
                            pose=next_pose,
                            g=g_new,
                            parent=current_key,
                            direction=int(direction),
                            steer=float(steer),
                            traj_from_parent=traj,
                        )
                        h_new = self._heuristic(next_pose, goal, holonomic_dist)
                        f_new = g_new + h_new
                        if self.guidance_integration_mode != "g_cost":
                            f_new += lam * self._guidance_priority_bias(mean_guidance)
                        heapq.heappush(open_heap, (f_new, next(counter), next_key))

        return GuidedHybridAstarResult(
            success=False,
            path=[],
            expanded_nodes=expanded_nodes,
            runtime_ms=(time.perf_counter() - t0) * 1e3,
            path_length=0.0,
            expanded_trace_xy=self._collect_expanded_trace_xy(records, closed),
        )
