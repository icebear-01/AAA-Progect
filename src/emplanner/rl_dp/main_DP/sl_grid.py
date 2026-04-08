"""
Core utilities for working with the discrete s-l sampling grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

DEFAULT_S_RANGE: Tuple[float, float] = (0.0, 8.0)
DEFAULT_S_SAMPLES = 9
DEFAULT_L_SPACING = 0.35

# Keep the lateral range close to +/-4 m while ensuring 0.0 is a sampled row
# and the grid spacing stays exactly 0.35 m.
_DEFAULT_L_HALF_STEPS = int(round(4.0 / DEFAULT_L_SPACING))
DEFAULT_L_RANGE: Tuple[float, float] = (
    -DEFAULT_L_SPACING * _DEFAULT_L_HALF_STEPS,
    DEFAULT_L_SPACING * _DEFAULT_L_HALF_STEPS,
)
DEFAULT_L_SAMPLES = _DEFAULT_L_HALF_STEPS * 2 + 1


@dataclass(frozen=True)
class GridSpec:
    """Hold the grid configuration."""

    s_range: Tuple[float, float]
    l_range: Tuple[float, float]
    s_samples: int = 9
    l_samples: int = 11

    def __post_init__(self) -> None:
        s_min, s_max = self.s_range
        l_min, l_max = self.l_range
        if s_min > s_max:
            raise ValueError("s_range minimum must not exceed maximum")
        if l_min > l_max:
            raise ValueError("l_range minimum must not exceed maximum")
        if self.s_samples < 2:
            raise ValueError("s_samples must be at least 2 to include the origin")
        if self.l_samples < 2:
            raise ValueError("l_samples must be at least 2 to include the origin")
        if not (s_min <= 0.0 <= s_max):
            raise ValueError("s_range must include 0.0 so the origin is sampled")
        if not (l_min <= 0.0 <= l_max):
            raise ValueError("l_range must include 0.0 so the origin is sampled")


def default_training_grid_spec() -> GridSpec:
    """Return the shared default grid used by training and demos."""
    return GridSpec(
        s_range=DEFAULT_S_RANGE,
        l_range=DEFAULT_L_RANGE,
        s_samples=DEFAULT_S_SAMPLES,
        l_samples=DEFAULT_L_SAMPLES,
    )


def build_grid(spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return coordinate vectors (s_grid, l_grid) for the sampling grid."""
    s_values = np.linspace(*spec.s_range, spec.s_samples)
    l_values = np.linspace(*spec.l_range, spec.l_samples)
    return np.meshgrid(s_values, l_values, indexing="ij")
