"""Sanity run for Neural A* on a random grid.

Creates a random occupancy map (1=obstacle, 0=free), samples random free start/goal,
and runs one forward pass through Neural A* search.
"""

from __future__ import annotations

import random
import time
from typing import Tuple

import numpy as np
import torch

from neural_astar.planner import NeuralAstar
from neural_astar.utils.coords import make_one_hot_xy


def sample_free_xy(occ: np.ndarray, rng: random.Random) -> Tuple[int, int]:
    h, w = occ.shape
    while True:
        x = rng.randrange(w)
        y = rng.randrange(h)
        if occ[y, x] < 0.5:
            return x, y


def main() -> None:
    seed = 1234
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    h = 64
    w = 64
    obstacle_prob = 0.22

    occ_map = (np.random.rand(h, w) < obstacle_prob).astype(np.float32)
    start_xy = sample_free_xy(occ_map, rng)
    goal_xy = sample_free_xy(occ_map, rng)
    while goal_xy == start_xy:
        goal_xy = sample_free_xy(occ_map, rng)

    # Existing Neural A* expects map_design: 1=free, 0=obstacle.
    map_design = 1.0 - occ_map
    start_map = make_one_hot_xy(start_xy[0], start_xy[1], w, h)
    goal_map = make_one_hot_xy(goal_xy[0], goal_xy[1], w, h)

    map_t = torch.from_numpy(map_design[None, None])
    start_t = torch.from_numpy(start_map[None, None])
    goal_t = torch.from_numpy(goal_map[None, None])

    planner = NeuralAstar()
    planner.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = planner(map_t, start_t, goal_t)
    runtime_ms = (time.perf_counter() - t0) * 1e3

    path_length = float(out.paths[0, 0].sum().item())
    expanded_nodes = float(out.histories[0, 0].sum().item())
    found_path = bool(path_length > 0)

    print(f"start_xy={start_xy}, goal_xy={goal_xy}")
    print(f"found_path={found_path}")
    print(f"path_length={path_length:.0f}")
    print(f"expanded_nodes={expanded_nodes:.0f}")
    print(f"runtime_ms={runtime_ms:.3f}")


if __name__ == "__main__":
    main()
