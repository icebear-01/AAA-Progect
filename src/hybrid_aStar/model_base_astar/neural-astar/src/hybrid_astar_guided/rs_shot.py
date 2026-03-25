"""Reeds-Shepp-style analytic shot connector for Hybrid A*.

This module provides a practical, dependency-free "RS-shot" implementation:
- Piecewise-constant curvature segments (L/R/S) with forward/reverse support.
- Nonlinear solve for segment lengths to match target pose.
- Collision-checked dense rollout for direct connection to goal.

It is intended for analytic expansion in Hybrid A*, not for globally optimal
Reeds-Shepp distance computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


Pose = Tuple[float, float, float]  # (x, y, yaw_rad)


@dataclass
class RSShotPath:
    poses: List[Pose]
    length: float
    directions: List[int]
    curvatures: List[float]
    segment_lengths: List[float]
    direction_switches: int


def normalize_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _propagate_pose(pose: Pose, curvature: float, direction: int, length: float) -> Pose:
    x, y, yaw = pose
    if length <= 0.0:
        return x, y, yaw

    if abs(curvature) < 1e-10:
        x_new = x + direction * length * math.cos(yaw)
        y_new = y + direction * length * math.sin(yaw)
        return x_new, y_new, yaw

    delta = direction * curvature * length
    yaw_new = normalize_angle(yaw + delta)
    x_new = x + (math.sin(yaw + delta) - math.sin(yaw)) / curvature
    y_new = y - (math.cos(yaw + delta) - math.cos(yaw)) / curvature
    return x_new, y_new, yaw_new


def _rollout_segments(
    start: Pose,
    directions: Sequence[int],
    curvatures: Sequence[float],
    segment_lengths: Sequence[float],
    sample_ds: float,
) -> List[Pose]:
    poses: List[Pose] = [start]
    state = start

    for direction, curvature, seg_len in zip(directions, curvatures, segment_lengths):
        remain = float(max(0.0, seg_len))
        if remain <= 0.0:
            continue

        while remain > sample_ds:
            state = _propagate_pose(state, curvature, direction, sample_ds)
            poses.append(state)
            remain -= sample_ds

        if remain > 1e-9:
            state = _propagate_pose(state, curvature, direction, remain)
            poses.append(state)

    return poses


def _endpoint_residual(
    start: Pose,
    goal: Pose,
    directions: Sequence[int],
    curvatures: Sequence[float],
    lengths: np.ndarray,
) -> np.ndarray:
    state = start
    for direction, curvature, seg_len in zip(directions, curvatures, lengths):
        state = _propagate_pose(state, curvature, int(direction), float(seg_len))
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    dyaw = normalize_angle(state[2] - goal[2])
    return np.array([dx, dy, dyaw], dtype=np.float64)


def _numeric_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    y0 = fn(x)
    jac = np.zeros((y0.shape[0], x.shape[0]), dtype=np.float64)
    for i in range(x.shape[0]):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] = max(0.0, xm[i] - eps)
        yp = fn(xp)
        ym = fn(xm)
        denom = xp[i] - xm[i]
        if denom < 1e-10:
            jac[:, i] = 0.0
        else:
            jac[:, i] = (yp - ym) / denom
    return jac


def _solve_lengths_levenberg(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    init_lengths: np.ndarray,
    max_iter: int,
    endpoint_tol: float,
) -> Tuple[np.ndarray, float]:
    x = np.maximum(init_lengths.astype(np.float64), 0.0)
    best_x = x.copy()
    best_norm = float(np.linalg.norm(residual_fn(x)))
    mu = 1e-3

    for _ in range(max_iter):
        r = residual_fn(x)
        r_norm = float(np.linalg.norm(r))
        if r_norm < best_norm:
            best_norm = r_norm
            best_x = x.copy()
        if r_norm <= endpoint_tol:
            return x, r_norm

        jac = _numeric_jacobian(residual_fn, x)
        a_mat = jac.T @ jac + mu * np.eye(x.shape[0], dtype=np.float64)
        g_vec = jac.T @ r
        try:
            delta = np.linalg.solve(a_mat, -g_vec)
        except np.linalg.LinAlgError:
            break

        improved = False
        step = 1.0
        for _ in range(8):
            cand = np.maximum(0.0, x + step * delta)
            cand_norm = float(np.linalg.norm(residual_fn(cand)))
            if cand_norm < r_norm:
                x = cand
                improved = True
                break
            step *= 0.5

        if improved:
            mu = max(1e-6, mu * 0.5)
        else:
            mu = min(1e6, mu * 10.0)

    return best_x, best_norm


def _build_seed_lengths(dxy: float, dyaw: float, max_curvature: float) -> List[np.ndarray]:
    turn = abs(dyaw) / max(max_curvature, 1e-6)
    base = max(1.0, dxy)
    seeds = [
        np.array([0.30 * base, 0.40 * base + turn, 0.30 * base], dtype=np.float64),
        np.array([0.50 * base, 0.10 * base + turn, 0.50 * base], dtype=np.float64),
        np.array([1.00, base + turn, 1.00], dtype=np.float64),
        np.array([0.10, base + turn, 0.10], dtype=np.float64),
        np.array([base, 0.10, 0.10], dtype=np.float64),
        np.array([0.10, base, 0.10], dtype=np.float64),
    ]
    return seeds


def _controls_from_types(control_types: Sequence[str], max_curvature: float) -> List[float]:
    curvatures: List[float] = []
    for c in control_types:
        if c == "L":
            curvatures.append(max_curvature)
        elif c == "R":
            curvatures.append(-max_curvature)
        else:
            curvatures.append(0.0)
    return curvatures


def _count_direction_switches(directions: Sequence[int]) -> int:
    switches = 0
    for i in range(1, len(directions)):
        if directions[i] != directions[i - 1]:
            switches += 1
    return switches


def find_rs_shot(
    start: Pose,
    goal: Pose,
    max_curvature: float,
    allow_reverse: bool,
    collision_checker: Callable[[Pose], bool],
    sample_ds: float = 0.25,
    endpoint_tol: float = 1e-3,
    max_iter: int = 28,
) -> Optional[RSShotPath]:
    """Find a collision-free RS-style analytic connector.

    Returns:
        RSShotPath if a direct connector is found, otherwise None.
    """
    if max_curvature <= 1e-8:
        return None

    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    dxy = float(math.hypot(dx, dy))
    dyaw = float(normalize_angle(goal[2] - start[2]))
    seeds = _build_seed_lengths(dxy=dxy, dyaw=dyaw, max_curvature=max_curvature)

    control_patterns = [
        ("L", "S", "L"),
        ("R", "S", "R"),
        ("L", "S", "R"),
        ("R", "S", "L"),
        ("L", "R", "L"),
        ("R", "L", "R"),
    ]

    if allow_reverse:
        direction_patterns = [
            (1, 1, 1),
            (-1, -1, -1),
            (1, 1, -1),
            (1, -1, -1),
            (-1, 1, 1),
            (-1, -1, 1),
            (1, -1, 1),
            (-1, 1, -1),
        ]
    else:
        direction_patterns = [(1, 1, 1)]

    best_path: Optional[RSShotPath] = None
    best_score = float("inf")

    for control_types in control_patterns:
        curvatures = _controls_from_types(control_types, max_curvature=max_curvature)
        for directions in direction_patterns:

            def residual_fn(lengths: np.ndarray) -> np.ndarray:
                return _endpoint_residual(
                    start=start,
                    goal=goal,
                    directions=directions,
                    curvatures=curvatures,
                    lengths=lengths,
                )

            for seed in seeds:
                solved_lengths, residual_norm = _solve_lengths_levenberg(
                    residual_fn=residual_fn,
                    init_lengths=seed,
                    max_iter=max_iter,
                    endpoint_tol=endpoint_tol,
                )
                if residual_norm > endpoint_tol:
                    continue

                seg_lengths = [float(max(0.0, v)) for v in solved_lengths.tolist()]
                total_length = float(sum(seg_lengths))
                if total_length <= 1e-6:
                    continue

                poses = _rollout_segments(
                    start=start,
                    directions=directions,
                    curvatures=curvatures,
                    segment_lengths=seg_lengths,
                    sample_ds=sample_ds,
                )
                if len(poses) < 2:
                    continue

                poses[-1] = goal
                colliding = any(collision_checker(p) for p in poses[1:])
                if colliding:
                    continue

                switches = _count_direction_switches(directions)
                # Prefer shorter paths; tiny tie-break on fewer gear switches.
                score = total_length + 1e-3 * float(switches)
                if score < best_score:
                    best_score = score
                    best_path = RSShotPath(
                        poses=poses,
                        length=total_length,
                        directions=[int(v) for v in directions],
                        curvatures=[float(v) for v in curvatures],
                        segment_lengths=seg_lengths,
                        direction_switches=switches,
                    )

    return best_path
