"""Coordinate helpers.

Coordinate convention used by Route A additions:
- World/state coordinate is (x, y).
- NumPy / torch map indexing is [y, x].
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


XY = Tuple[int, int]
RC = Tuple[int, int]


def validate_xy(x: int, y: int, width: int, height: int) -> None:
    """Validate that world coordinate (x, y) is in map bounds."""
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"(x, y)=({x}, {y}) out of bounds for width={width}, height={height}"
        )


def xy_to_rc(x: int, y: int, width: int, height: int) -> RC:
    """Convert world (x, y) to array index (row, col) = [y, x]."""
    validate_xy(x, y, width, height)
    return y, x


def rc_to_xy(r: int, c: int, width: int, height: int) -> XY:
    """Convert array index (row, col) to world (x, y)."""
    if not (0 <= c < width and 0 <= r < height):
        raise ValueError(
            f"(row, col)=({r}, {c}) out of bounds for width={width}, height={height}"
        )
    return c, r


def make_one_hot_xy(x: int, y: int, width: int, height: int) -> np.ndarray:
    """Create one-hot map [H, W] with 1 at world (x, y)."""
    r, c = xy_to_rc(x, y, width, height)
    m = np.zeros((height, width), dtype=np.float32)
    m[r, c] = 1.0
    return m


def clip_cost_map_with_obstacles(
    cost_map: np.ndarray, occ_map: np.ndarray, obstacle_cost: float = 1.0
) -> np.ndarray:
    """Apply obstacle mask to cost map.

    occ_map semantics: 1=obstacle, 0=free.
    cost_map semantics: lower is better.
    """
    if cost_map.shape != occ_map.shape:
        raise ValueError(
            f"Shape mismatch: cost_map={cost_map.shape}, occ_map={occ_map.shape}"
        )
    out = np.asarray(cost_map, dtype=np.float32).copy()
    out[np.asarray(occ_map) > 0.5] = float(obstacle_cost)
    return out
