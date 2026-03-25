"""RL scaffold environment.

Pipeline:
(occ_map, start, goal) -> infer cost_map -> guided Hybrid A* -> reward

Reward shape (example):
    -path_length - 0.1 * expanded_nodes - 10 * collision - 5 * failure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from hybrid_astar_guided import GuidedHybridAstar
from neural_astar.api.guidance_infer import infer_cost_map


@dataclass
class RouteARLEnvConfig:
    ckpt_path: str
    lambda_guidance: float = 2.0
    device: str = "cpu"


class RouteARLEnv:
    def __init__(self, cfg: RouteARLEnvConfig):
        self.cfg = cfg
        self.planner = GuidedHybridAstar(lambda_guidance=cfg.lambda_guidance)

    def rollout_once(
        self,
        occ_map: np.ndarray,
        start_xy: Tuple[int, int],
        goal_xy: Tuple[int, int],
        start_theta: float = 0.0,
        goal_theta: float = 0.0,
    ) -> Dict[str, float]:
        cost_map = infer_cost_map(
            ckpt_path=self.cfg.ckpt_path,
            occ_map_numpy=occ_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            device=self.cfg.device,
        )
        result = self.planner.plan(
            occ_map=occ_map,
            start_pose=(start_xy[0], start_xy[1], start_theta),
            goal_pose=(goal_xy[0], goal_xy[1], goal_theta),
            cost_map=cost_map,
            lambda_guidance=self.cfg.lambda_guidance,
        )

        collision = 0.0
        failure = 0.0 if result.success else 1.0
        reward = -result.path_length - 0.1 * float(result.expanded_nodes) - 10.0 * collision - 5.0 * failure

        return {
            "reward": float(reward),
            "success": float(result.success),
            "path_length": float(result.path_length),
            "expanded_nodes": float(result.expanded_nodes),
        }
