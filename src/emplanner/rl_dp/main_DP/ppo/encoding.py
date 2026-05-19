from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from sl_grid import GridSpec


def _current_l_coord(
    observation: Dict[str, np.ndarray],
    spec: GridSpec,
    l_coords: np.ndarray,
) -> float:
    current_l_index = observation.get("current_l_index")
    if current_l_index is not None:
        try:
            idx = int(np.asarray(current_l_index).reshape(-1)[0])
            idx = int(np.clip(idx, 0, spec.l_samples - 1))
            return float(l_coords[idx])
        except Exception:
            pass

    path_indices = np.asarray(observation.get("path_indices", []))
    if path_indices.size > 0:
        last_idx = int(np.clip(path_indices[-1], 0, spec.l_samples - 1))
        return float(l_coords[last_idx])
    return float(np.mean(spec.l_range))


def _fill_obstacle_features(
    out: np.ndarray,
    observation: Dict[str, np.ndarray],
    current_s: float,
    current_l: float,
    s_min: float,
    s_span: float,
    l_min: float,
    l_span: float,
    max_obstacles: int,
) -> None:
    obstacle_corners_norm = observation.get("obstacle_corners_norm")
    obstacle_centers = observation.get("obstacle_centers")
    if obstacle_corners_norm is not None and obstacle_centers is not None:
        corners_norm = np.asarray(obstacle_corners_norm, dtype=np.float32)
        centers = np.asarray(obstacle_centers, dtype=np.float32)
        if (
            corners_norm.ndim == 3
            and corners_norm.shape[1:] == (4, 2)
            and centers.shape == (corners_norm.shape[0], 2)
        ):
            if centers.shape[0] > 0:
                current = np.array([current_s, current_l], dtype=np.float32)
                dists = np.sum((centers - current) ** 2, axis=1)
                order = np.argsort(dists)
                corners_norm = corners_norm[order]
            take = min(max_obstacles, corners_norm.shape[0])
            if take > 0:
                out_view = out.reshape(max_obstacles, 4, 2)
                out_view[:take] = corners_norm[:take]
            return

    obstacle_corners = observation.get("obstacle_corners")
    if obstacle_corners is None:
        obstacle_corners = np.zeros((0, 4, 2), dtype=np.float32)
    obstacle_corners = np.asarray(obstacle_corners, dtype=np.float32)
    if obstacle_corners.ndim != 3 or obstacle_corners.shape[1:] != (4, 2):
        return
    if obstacle_corners.size > 0:
        centers = obstacle_corners.mean(axis=1)
        current = np.array([current_s, current_l], dtype=np.float32)
        dists = np.sum((centers - current) ** 2, axis=1)
        order = np.argsort(dists)
        obstacle_corners = obstacle_corners[order]
    take = min(max_obstacles, obstacle_corners.shape[0])
    if take <= 0:
        return
    selected = obstacle_corners[:take].copy()
    selected[..., 0] = np.clip((selected[..., 0] - s_min) / s_span, 0.0, 1.0)
    selected[..., 1] = np.clip((selected[..., 1] - l_min) / l_span, 0.0, 1.0)
    out.reshape(max_obstacles, 4, 2)[:take] = selected


def encode_observation_into(
    observation: Dict[str, np.ndarray],
    spec: GridSpec,
    out: np.ndarray,
    *,
    include_action_mask: bool = True,
    max_obstacles: int = 10,
) -> np.ndarray:
    max_obstacles = max(0, int(max_obstacles))
    occupancy_size = spec.s_samples * spec.l_samples
    obstacle_size = max_obstacles * 4 * 2
    action_mask_size = spec.l_samples if include_action_mask else 0
    expected_size = occupancy_size + obstacle_size + action_mask_size + 3
    if out.shape[0] != expected_size:
        raise ValueError(f"output size mismatch: got {out.shape[0]}, expected {expected_size}")
    out.fill(0.0)

    occupancy_flat = observation.get("occupancy_flat")
    if occupancy_flat is not None:
        occupancy_flat = np.asarray(occupancy_flat, dtype=np.float32).reshape(-1)
        if occupancy_flat.shape[0] == occupancy_size:
            out[:occupancy_size] = occupancy_flat
        else:
            occupancy_flat = None
    if occupancy_flat is None:
        occupancy = np.asarray(observation["occupancy"], dtype=np.float32)
        np.clip(occupancy.reshape(-1), 0.0, 1.0, out=out[:occupancy_size])

    s_min, s_max = spec.s_range
    l_min, l_max = spec.l_range
    s_span = max(s_max - s_min, 1e-6)
    l_span = max(l_max - l_min, 1e-6)

    s_index = int(np.clip(observation.get("s_index", 0), 0, spec.s_samples - 1))
    s_coords = observation.get("s_coords")
    if s_coords is None or np.size(s_coords) != spec.s_samples:
        s_coords = np.linspace(s_min, s_max, spec.s_samples, dtype=np.float32)
    else:
        s_coords = np.asarray(s_coords, dtype=np.float32)
    s_coord = float(s_coords[s_index])

    l_coords = observation.get("l_coords")
    if l_coords is None or np.size(l_coords) != spec.l_samples:
        l_coords = np.linspace(l_min, l_max, spec.l_samples, dtype=np.float32)
    else:
        l_coords = np.asarray(l_coords, dtype=np.float32)
    l_coord = _current_l_coord(observation, spec, l_coords)

    obstacle_start = occupancy_size
    obstacle_end = obstacle_start + obstacle_size
    _fill_obstacle_features(
        out[obstacle_start:obstacle_end],
        observation,
        s_coord,
        l_coord,
        s_min,
        s_span,
        l_min,
        l_span,
        max_obstacles,
    )

    cursor = obstacle_end
    if include_action_mask:
        raw_mask = observation.get("action_mask")
        if raw_mask is None:
            out[cursor : cursor + spec.l_samples] = 1.0
        else:
            action_mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1)
            if action_mask.shape[0] != spec.l_samples:
                out[cursor : cursor + spec.l_samples] = 1.0
            else:
                out[cursor : cursor + spec.l_samples] = action_mask
        cursor += spec.l_samples

    s_norm = float(s_index) / max(spec.s_samples - 1, 1) if spec.s_samples > 1 else 0.0
    l_norm = (l_coord - l_min) / l_span
    start_l = observation.get("start_l")
    if start_l is None:
        start_l_value = l_coord
    else:
        start_l_value = float(np.asarray(start_l).reshape(-1)[0])
    start_l_norm = np.clip((start_l_value - l_min) / l_span, 0.0, 1.0)
    out[cursor : cursor + 3] = (s_norm, l_norm, start_l_norm)
    return out


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
    max_obstacles = max(0, int(max_obstacles))
    feature_dim = (
        spec.s_samples * spec.l_samples
        + max_obstacles * 4 * 2
        + (spec.l_samples if include_action_mask else 0)
        + 3
    )
    encoded = np.empty(feature_dim, dtype=np.float32)
    return encode_observation_into(
        observation,
        spec,
        encoded,
        include_action_mask=include_action_mask,
        max_obstacles=max_obstacles,
    )


def encode_observation_batch(
    observations: Sequence[Dict[str, np.ndarray]],
    spec: GridSpec,
    include_action_mask: bool = True,
    max_obstacles: int = 10,
) -> np.ndarray:
    max_obstacles = max(0, int(max_obstacles))
    feature_dim = (
        spec.s_samples * spec.l_samples
        + max_obstacles * 4 * 2
        + (spec.l_samples if include_action_mask else 0)
        + 3
    )
    batch = np.empty((len(observations), feature_dim), dtype=np.float32)
    for idx, observation in enumerate(observations):
        encode_observation_into(
            observation,
            spec,
            batch[idx],
            include_action_mask=include_action_mask,
            max_obstacles=max_obstacles,
        )
    return batch
