"""
Utilities for generating random rectangular obstacles in the s-l plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """Represent a rectangular obstacle with arbitrary orientation."""

    center: Tuple[float, float]
    length: float
    width: float
    yaw: float

    def corners(self) -> np.ndarray:
        """Return the four s-l corner points in counter-clockwise order."""
        half_l = self.length / 2.0
        half_w = self.width / 2.0
        # Rectangle corners in local s-l coordinates before rotation.
        local = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=float,
        )
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s], [s, c]])
        rotated = local @ rotation.T
        return rotated + np.asarray(self.center)


def _radius_from_extent(length: float, width: float) -> float:
    """Return the circumradius of the rectangle."""
    return float(np.hypot(length, width) * 0.5)


def _trim_range(value_range: Sequence[float], margin: float) -> Tuple[float, float]:
    """Shrink a range by `margin` on both sides to keep shapes inside."""
    start, end = float(value_range[0]), float(value_range[1])
    if end < start:
        raise ValueError("range start must be <= range end")
    new_start = start + margin
    new_end = end - margin
    if new_start > new_end:
        # Fallback: collapse to the midpoint so we can still place obstacles.
        midpoint = (start + end) / 2.0
        return midpoint, midpoint
    return new_start, new_end


def _obstacles_roughly_overlap(
    lhs: Obstacle,
    rhs: Obstacle,
    *,
    extra_clearance: float = 0.0,
) -> bool:
    """Fast coarse overlap test using circumcircle distance."""
    lhs_center = np.asarray(lhs.center, dtype=float)
    rhs_center = np.asarray(rhs.center, dtype=float)
    lhs_radius = _radius_from_extent(lhs.length, lhs.width)
    rhs_radius = _radius_from_extent(rhs.length, rhs.width)
    min_center_distance = lhs_radius + rhs_radius + float(extra_clearance)
    center_distance = float(np.linalg.norm(lhs_center - rhs_center))
    return center_distance < min_center_distance


def generate_random_obstacles(
    s_range: Sequence[float],
    l_range: Sequence[float],
    *,
    min_count: int = 0,
    max_count: int = 10,
    length_range: Sequence[float] = (0.2, 1.2),
    width_range: Sequence[float] = (0.2, 1.0),
    avoid_overlap: bool = False,
    overlap_clearance: float = 0.0,
    max_sampling_attempts_per_obstacle: int = 24,
    rng: Optional[np.random.Generator] = None,
) -> List[Obstacle]:
    """
    Create a random number of oriented rectangular obstacles.

    The actual number of returned obstacles is sampled uniformly from
    `min_count..max_count`. Obstacles are fully contained within the provided
    s/l ranges.
    """

    if min_count < 0:
        raise ValueError("min_count must be non-negative")
    if max_count < 0:
        raise ValueError("max_count must be non-negative")
    if min_count > max_count:
        raise ValueError("min_count must not exceed max_count")
    if max_sampling_attempts_per_obstacle < 1:
        raise ValueError("max_sampling_attempts_per_obstacle must be >= 1")
    rng = rng or np.random.default_rng()
    count = int(rng.integers(min_count, max_count + 1))
    if count == 0:
        return []

    length_low, length_high = map(float, length_range)
    width_low, width_high = map(float, width_range)
    if length_low <= 0.0 or width_low <= 0.0:
        raise ValueError("length and width ranges must be positive")

    obstacles: List[Obstacle] = []
    for _ in range(count):
        sampled_obstacle: Optional[Obstacle] = None
        for _attempt in range(int(max_sampling_attempts_per_obstacle)):
            length = float(rng.uniform(length_low, length_high))
            width = float(rng.uniform(width_low, width_high))
            yaw = float(rng.uniform(0.0, np.pi))
            margin = _radius_from_extent(length, width)
            s_min, s_max = _trim_range(s_range, margin)
            l_min, l_max = _trim_range(l_range, margin)
            center_s = float(rng.uniform(s_min, s_max)) if s_min != s_max else s_min
            center_l = float(rng.uniform(l_min, l_max)) if l_min != l_max else l_min
            candidate = Obstacle(
                center=(center_s, center_l),
                length=length,
                width=width,
                yaw=yaw,
            )
            if avoid_overlap and any(
                _obstacles_roughly_overlap(
                    candidate,
                    existing,
                    extra_clearance=overlap_clearance,
                )
                for existing in obstacles
            ):
                continue
            sampled_obstacle = candidate
            break
        if sampled_obstacle is None:
            if obstacles and avoid_overlap:
                continue
            sampled_obstacle = candidate
        obstacles.append(sampled_obstacle)
    return obstacles


def obstacles_as_polygons(obstacles: Iterable[Obstacle]) -> List[np.ndarray]:
    """Convert obstacles into arrays of shape (4, 2) representing corners."""
    return [obstacle.corners() for obstacle in obstacles]
