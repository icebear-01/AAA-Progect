from __future__ import annotations

from typing import Dict

import numpy as np

from sl_grid import GridSpec


def encode_observation(
    observation: Dict[str, np.ndarray],
    spec: GridSpec,
    include_action_mask: bool = True,
    max_obstacles: int = 10,
) -> np.ndarray:
    """
    将环境观测字典编码成扁平特征向量。
    布局：
        - 1 个栅格平面：占用（0/1）
        - 障碍角点：最多 `max_obstacles` 个，每个 4 个角点 (s, l) 归一化并展平
        - （可选）动作掩码向量，直接拼到尾部，不再作为栅格平面
        - 归一化的 s / l 标量特征 + 起始 l 连续值（归一化）
    """
    occupancy = observation["occupancy"].astype(np.float32)
    occupancy = np.clip(occupancy, 0.0, 1.0)
    occupancy_flat = occupancy.reshape(-1)

    s_min, s_max = spec.s_range
    l_min, l_max = spec.l_range
    s_span = max(s_max - s_min, 1e-6)
    l_span = max(l_max - l_min, 1e-6)

    s_index = int(np.clip(observation.get("s_index", 0), 0, spec.s_samples - 1))
    s_coords = observation.get("s_coords")
    if s_coords is None or np.size(s_coords) != spec.s_samples:
        s_coords = np.linspace(s_min, s_max, spec.s_samples, dtype=np.float32)
    s_coord = float(s_coords[s_index])

    l_coords = observation.get("l_coords")
    if l_coords is None or np.size(l_coords) != spec.l_samples:
        l_coords = np.linspace(l_min, l_max, spec.l_samples, dtype=np.float32)
    path_indices = np.asarray(observation.get("path_indices", []))
    if path_indices.size > 0:
        last_idx = int(np.clip(path_indices[-1], 0, spec.l_samples - 1))
        l_coord = float(l_coords[last_idx])
    else:
        l_coord = float(np.mean(spec.l_range))

    obstacle_corners = observation.get("obstacle_corners")
    if obstacle_corners is None:
        obstacle_corners = np.zeros((0, 4, 2), dtype=np.float32)
    obstacle_corners = np.asarray(obstacle_corners, dtype=np.float32)
    max_obstacles = max(0, int(max_obstacles))
    if obstacle_corners.ndim != 3 or obstacle_corners.shape[1:] != (4, 2):
        obstacle_corners = np.zeros((0, 4, 2), dtype=np.float32)
    if obstacle_corners.size > 0:
        centers = obstacle_corners.mean(axis=1)
        current = np.array([s_coord, l_coord], dtype=np.float32)
        dists = np.sum((centers - current) ** 2, axis=1)
        order = np.argsort(dists)
        obstacle_corners = obstacle_corners[order]
    if obstacle_corners.shape[0] > max_obstacles:
        obstacle_corners = obstacle_corners[:max_obstacles]

    padded = np.zeros((max_obstacles, 4, 2), dtype=np.float32)
    if obstacle_corners.size > 0:
        padded[: obstacle_corners.shape[0]] = obstacle_corners
    obstacle_corners = padded

    obstacle_corners[..., 0] = np.clip(
        (obstacle_corners[..., 0] - s_min) / s_span, 0.0, 1.0
    )
    obstacle_corners[..., 1] = np.clip(
        (obstacle_corners[..., 1] - l_min) / l_span, 0.0, 1.0
    )
    obstacle_flat = obstacle_corners.reshape(-1)

    action_mask_flat = np.zeros(0, dtype=np.float32)
    if include_action_mask:
        raw_mask = observation.get("action_mask")
        if raw_mask is None:
            raw_mask = np.ones(spec.l_samples, dtype=bool)
        action_mask = np.asarray(raw_mask, dtype=np.float32)
        if action_mask.shape[0] != spec.l_samples:
            action_mask = np.ones(spec.l_samples, dtype=np.float32)
        action_mask_flat = action_mask.reshape(-1)

    s_norm = (
        float(s_index) / max(spec.s_samples - 1, 1) if spec.s_samples > 1 else 0.0
    )
    l_norm = (l_coord - l_min) / l_span
    start_l = observation.get("start_l")
    if start_l is None:
        start_l_value = l_coord
    else:
        start_l_value = float(np.asarray(start_l).reshape(-1)[0])
    start_l_norm = np.clip((start_l_value - l_min) / l_span, 0.0, 1.0)

    extras = np.array([s_norm, l_norm, start_l_norm], dtype=np.float32)
    return np.concatenate(
        [occupancy_flat, obstacle_flat, action_mask_flat, extras]
    )
