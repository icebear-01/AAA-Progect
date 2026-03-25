"""Utilities for building dense guidance supervision targets."""

from __future__ import annotations

import heapq
import math

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
except Exception:
    _distance_transform_edt = None


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    path_mask = np.asarray(mask, dtype=bool)
    if not np.any(path_mask):
        return np.full(path_mask.shape, np.inf, dtype=np.float32)

    if _distance_transform_edt is not None:
        return _distance_transform_edt(~path_mask).astype(np.float32)

    ys, xs = np.where(path_mask)
    yy, xx = np.mgrid[0 : path_mask.shape[0], 0 : path_mask.shape[1]]
    dist2 = np.full(path_mask.shape, np.inf, dtype=np.float32)
    for y, x in zip(ys, xs):
        d2 = (yy - y).astype(np.float32) ** 2 + (xx - x).astype(np.float32) ** 2
        dist2 = np.minimum(dist2, d2)
    return np.sqrt(dist2).astype(np.float32)


def build_target_cost_map(
    occ_map: np.ndarray,
    opt_traj: np.ndarray,
    normalize_quantile: float = 95.0,
    clearance_weight: float = 0.0,
    clearance_safe_distance: float = 0.0,
    clearance_power: float = 2.0,
    clearance_penalize_path: bool = False,
) -> np.ndarray:
    """Create a dense target cost map from occupancy and expert corridor.

    Semantics:
    - 0 on expert path
    - increases smoothly with Euclidean distance from expert path
    - 1 on obstacle cells
    """
    occ = np.asarray(occ_map, dtype=np.float32)
    traj = np.asarray(opt_traj, dtype=np.float32)

    if occ.shape != traj.shape:
        raise ValueError(f"Shape mismatch: occ={occ.shape}, opt_traj={traj.shape}")

    free_mask = occ < 0.5
    path_mask = traj > 0.5

    if not np.any(path_mask):
        target = np.ones_like(occ, dtype=np.float32)
        target[~free_mask] = 1.0
        return target.astype(np.float32)

    dist = _distance_to_mask(path_mask)
    target = np.zeros_like(dist, dtype=np.float32)

    if np.any(free_mask):
        ref = dist[free_mask]
        scale = float(np.percentile(ref, normalize_quantile))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = float(np.max(ref)) if np.max(ref) > 1e-6 else 1.0
        target = np.clip(dist / scale, 0.0, 1.0).astype(np.float32)

    target[path_mask] = 0.0
    target[~free_mask] = 1.0
    if float(clearance_weight) > 0.0 and float(clearance_safe_distance) > 0.0:
        clearance_penalty = build_clearance_penalty_map(
            occ_map=occ,
            safe_distance=float(clearance_safe_distance),
            power=float(clearance_power),
        )
        if not bool(clearance_penalize_path):
            clearance_penalty[path_mask] = 0.0
        target = np.clip(
            target + float(clearance_weight) * clearance_penalty,
            0.0,
            1.0,
        ).astype(np.float32)
        if not bool(clearance_penalize_path):
            target[path_mask] = 0.0
        target[~free_mask] = 1.0
    return target.astype(np.float32)


def build_clearance_penalty_map(
    occ_map: np.ndarray,
    safe_distance: float,
    power: float = 2.0,
) -> np.ndarray:
    """Build a normalized penalty that is large near obstacles and zero far away."""
    occ = np.asarray(occ_map, dtype=np.float32)
    obstacle_mask = occ > 0.5
    free_mask = ~obstacle_mask
    if float(safe_distance) <= 0.0:
        return np.zeros_like(occ, dtype=np.float32)

    clearance = _distance_to_mask(obstacle_mask)
    penalty = np.zeros_like(clearance, dtype=np.float32)
    scale = max(float(safe_distance), 1e-6)
    penalty[free_mask] = np.clip(
        (scale - clearance[free_mask]) / scale,
        0.0,
        1.0,
    ).astype(np.float32)
    if float(power) != 1.0:
        penalty[free_mask] = np.power(
            penalty[free_mask],
            max(float(power), 1e-6),
        ).astype(np.float32)
    penalty[obstacle_mask] = 1.0
    return penalty.astype(np.float32)


def build_clearance_input_map(
    occ_map: np.ndarray,
    clip_distance: float,
) -> np.ndarray:
    """Build a clipped obstacle-distance input channel in [0, 1]."""
    occ = np.asarray(occ_map, dtype=np.float32)
    obstacle_mask = occ > 0.5
    free_mask = ~obstacle_mask
    if float(clip_distance) <= 0.0:
        return np.zeros_like(occ, dtype=np.float32)

    clearance = _distance_to_mask(obstacle_mask)
    scale = max(float(clip_distance), 1e-6)
    clearance_input = np.zeros_like(clearance, dtype=np.float32)
    clearance_input[free_mask] = np.clip(
        clearance[free_mask] / scale,
        0.0,
        1.0,
    ).astype(np.float32)
    clearance_input[obstacle_mask] = 0.0
    return clearance_input.astype(np.float32)


def build_octile_heuristic_map(
    shape_hw: tuple[int, int],
    goal_xy: tuple[int, int],
    diagonal_cost: float = math.sqrt(2.0),
) -> np.ndarray:
    """Build an octile-distance heuristic map for an 8-connected grid."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    yy, xx = np.mgrid[0:h, 0:w]
    dx = np.abs(xx.astype(np.float32) - float(gx))
    dy = np.abs(yy.astype(np.float32) - float(gy))
    dmin = np.minimum(dx, dy)
    dmax = np.maximum(dx, dy)
    return (dmax + (float(diagonal_cost) - 1.0) * dmin).astype(np.float32)


def build_exact_grid_heuristic_map(
    occ_map: np.ndarray,
    goal_xy: tuple[int, int],
    diagonal_cost: float = math.sqrt(2.0),
    allow_corner_cut: bool = True,
) -> np.ndarray:
    """Run reverse 8-connected Dijkstra to obtain exact distance-to-goal map.

    Diagonal transitions are always rejected when both side-adjacent cells are
    occupied. When ``allow_corner_cut`` is ``False``, any occupied side-adjacent
    cell also rejects the diagonal transition.
    """
    if diagonal_cost <= 0.0:
        raise ValueError(f"diagonal_cost must be positive, got {diagonal_cost}")

    occ = np.asarray(occ_map, dtype=np.float32)
    h, w = occ.shape
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    if gx < 0 or gx >= w or gy < 0 or gy >= h:
        raise ValueError(f"goal_xy out of bounds: {goal_xy} for shape {(h, w)}")

    dist = np.full((h, w), np.inf, dtype=np.float32)
    if occ[gy, gx] > 0.5:
        return dist

    dist[gy, gx] = 0.0
    heap = [(0.0, gx, gy)]
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]
    while heap:
        cur_d, x, y = heapq.heappop(heap)
        if cur_d > float(dist[y, x]) + 1e-6:
            continue
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if occ[ny, nx] > 0.5:
                continue
            is_diag = (dx != 0) and (dy != 0)
            if is_diag:
                side_block_x = occ[y, nx] > 0.5
                side_block_y = occ[ny, x] > 0.5
                if side_block_x and side_block_y:
                    continue
                if (not allow_corner_cut) and (side_block_x or side_block_y):
                    continue
            step = float(diagonal_cost) if is_diag else 1.0
            nd = float(cur_d) + step
            if nd + 1e-6 < float(dist[ny, nx]):
                dist[ny, nx] = nd
                heapq.heappush(heap, (nd, nx, ny))
    return dist.astype(np.float32)


def exact_grid_heuristic_from_opt_dist(
    occ_map: np.ndarray,
    opt_dist: np.ndarray,
) -> np.ndarray:
    """Convert planning-datasets optimal distance tensor to exact 2D distance-to-goal."""
    occ = np.asarray(occ_map, dtype=np.float32)
    raw = np.asarray(opt_dist, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError(f"opt_dist must be [H,W] or [1,H,W], got {raw.shape}")
    if raw.shape != occ.shape:
        raise ValueError(f"Shape mismatch: occ={occ.shape}, opt_dist={raw.shape}")

    sentinel = float(raw.min())
    free_mask = occ < 0.5
    exact = np.full_like(raw, np.inf, dtype=np.float32)
    valid = free_mask & (raw > sentinel)
    exact[valid] = np.maximum(-raw[valid], 0.0)
    goal_mask = free_mask & (np.abs(raw) <= 1e-6)
    exact[goal_mask] = 0.0
    return exact.astype(np.float32)


def build_residual_heuristic_maps(
    occ_map: np.ndarray,
    goal_xy: tuple[int, int],
    exact_heuristic_map: np.ndarray | None = None,
    diagonal_cost: float = math.sqrt(2.0),
    allow_corner_cut: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact, anchor-octile, and residual heuristic maps."""
    occ = np.asarray(occ_map, dtype=np.float32)
    free_mask = occ < 0.5
    octile = build_octile_heuristic_map(
        shape_hw=occ.shape,
        goal_xy=goal_xy,
        diagonal_cost=diagonal_cost,
    )
    if exact_heuristic_map is None:
        exact = build_exact_grid_heuristic_map(
            occ_map=occ,
            goal_xy=goal_xy,
            diagonal_cost=diagonal_cost,
            allow_corner_cut=allow_corner_cut,
        )
    else:
        exact = np.asarray(exact_heuristic_map, dtype=np.float32)
        if exact.shape != occ.shape:
            raise ValueError(f"Shape mismatch: occ={occ.shape}, exact={exact.shape}")

    residual = np.zeros_like(occ, dtype=np.float32)
    valid = free_mask & np.isfinite(exact)
    residual[valid] = np.maximum(exact[valid] - octile[valid], 0.0).astype(np.float32)
    exact_out = np.zeros_like(occ, dtype=np.float32)
    exact_out[valid] = exact[valid]
    octile[~free_mask] = 0.0
    residual[~free_mask] = 0.0
    return exact_out.astype(np.float32), octile.astype(np.float32), residual.astype(np.float32)


def build_expanded_trace_map(
    occ_map: np.ndarray,
    expanded_trace_xy: np.ndarray,
    normalize_quantile: float = 99.0,
) -> np.ndarray:
    """Project Hybrid A* expanded rollout points into a normalized XY heatmap.

    Semantics:
    - 0 means the planner never expanded through that free cell
    - larger values mean that expanded nodes repeatedly rolled through that cell
    - obstacle cells are forced to 0 because they are not valid learning targets
    """
    return build_expanded_xy_map(
        occ_map=occ_map,
        expanded_xy=expanded_trace_xy,
        normalize_quantile=normalize_quantile,
    )


def build_expanded_xy_map(
    occ_map: np.ndarray,
    expanded_xy: np.ndarray,
    normalize_quantile: float = 99.0,
) -> np.ndarray:
    """Project generic expanded XY cells into a normalized free-space heatmap."""
    occ = np.asarray(occ_map, dtype=np.float32)
    free_mask = occ < 0.5
    heat = np.zeros_like(occ, dtype=np.float32)

    trace = np.asarray(expanded_xy, dtype=np.float32)
    if trace.size == 0:
        return heat
    if trace.ndim != 2 or trace.shape[1] < 2:
        raise ValueError(f"expanded_xy must be [N,2+] or empty, got {trace.shape}")

    h, w = occ.shape
    xs = np.rint(trace[:, 0]).astype(np.int32)
    ys = np.rint(trace[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(valid):
        return heat

    xs = xs[valid]
    ys = ys[valid]
    for x, y in zip(xs, ys):
        if free_mask[y, x]:
            heat[y, x] += 1.0

    active = heat > 0.0
    if np.any(active):
        ref = heat[active]
        scale = float(np.percentile(ref, normalize_quantile))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = float(np.max(ref)) if float(np.max(ref)) > 1e-6 else 1.0
        heat = np.clip(heat / scale, 0.0, 1.0).astype(np.float32)

    heat[~free_mask] = 0.0
    return heat.astype(np.float32)


def yaw_to_bin(yaw: float, yaw_bins: int) -> int:
    if yaw_bins <= 0:
        raise ValueError(f"yaw_bins must be positive, got {yaw_bins}")
    yaw_n = math.atan2(math.sin(float(yaw)), math.cos(float(yaw)))
    t = (yaw_n + math.pi) / (2.0 * math.pi)
    return int(round(t * int(yaw_bins))) % int(yaw_bins)


def _path_yaws_from_poses(path_poses: np.ndarray) -> np.ndarray:
    path = np.asarray(path_poses, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        raise ValueError(f"path_poses must be [N,2+] with N>0, got {path.shape}")
    if path.shape[1] >= 3:
        return path[:, 2].astype(np.float32)

    yaws = np.zeros((path.shape[0],), dtype=np.float32)
    last_yaw = 0.0
    for i in range(path.shape[0]):
        if path.shape[0] == 1:
            dx = 1.0
            dy = 0.0
        elif i + 1 < path.shape[0]:
            dx = float(path[i + 1, 0] - path[i, 0])
            dy = float(path[i + 1, 1] - path[i, 1])
        else:
            dx = float(path[i, 0] - path[i - 1, 0])
            dy = float(path[i, 1] - path[i - 1, 1])

        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
            yaws[i] = last_yaw
            continue
        last_yaw = math.atan2(dy, dx)
        yaws[i] = last_yaw
    return yaws.astype(np.float32)


def build_orientation_corridor(
    opt_traj: np.ndarray,
    path_poses: np.ndarray,
    yaw_bins: int,
) -> np.ndarray:
    """Assign each 2D corridor cell to the nearest expert-path yaw bin."""
    if yaw_bins <= 1:
        return np.asarray(opt_traj, dtype=np.float32)[None, ...]

    traj = np.asarray(opt_traj, dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"opt_traj must be [H,W], got {traj.shape}")

    path = np.asarray(path_poses, dtype=np.float32)
    if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
        raise ValueError(f"path_poses must be [N,2+] with N>0, got {path.shape}")

    h, w = traj.shape
    path_x = np.rint(path[:, 0]).astype(np.int32)
    path_y = np.rint(path[:, 1]).astype(np.int32)
    valid = (path_x >= 0) & (path_x < w) & (path_y >= 0) & (path_y < h)
    if not np.any(valid):
        out = np.zeros((int(yaw_bins), h, w), dtype=np.float32)
        return out

    path_x = path_x[valid]
    path_y = path_y[valid]
    path_yaws = _path_yaws_from_poses(path)[valid]
    path_bins = np.asarray([yaw_to_bin(float(yaw), yaw_bins) for yaw in path_yaws], dtype=np.int32)

    corridor_cells = np.argwhere(traj > 0.5)
    out = np.zeros((int(yaw_bins), h, w), dtype=np.float32)
    if corridor_cells.size == 0:
        return out

    path_xy = np.stack([path_x, path_y], axis=1).astype(np.float32)
    for y, x in corridor_cells:
        d2 = np.sum((path_xy - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
        path_idx = int(np.argmin(d2))
        out[int(path_bins[path_idx]), int(y), int(x)] = 1.0
    return out.astype(np.float32)


def build_orientation_target_maps(
    occ_map: np.ndarray,
    opt_traj: np.ndarray,
    path_poses: np.ndarray,
    yaw_bins: int,
    normalize_quantile: float = 95.0,
    clearance_weight: float = 0.0,
    clearance_safe_distance: float = 0.0,
    clearance_power: float = 2.0,
    clearance_penalize_path: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create orientation-aware corridor and dense target volumes."""
    occ = np.asarray(occ_map, dtype=np.float32)
    traj = np.asarray(opt_traj, dtype=np.float32)
    if occ.shape != traj.shape:
        raise ValueError(f"Shape mismatch: occ={occ.shape}, opt_traj={traj.shape}")

    base_target = build_target_cost_map(
        occ_map=occ,
        opt_traj=traj,
        normalize_quantile=normalize_quantile,
        clearance_weight=clearance_weight,
        clearance_safe_distance=clearance_safe_distance,
        clearance_power=clearance_power,
        clearance_penalize_path=clearance_penalize_path,
    )
    orient_traj = build_orientation_corridor(
        opt_traj=traj,
        path_poses=path_poses,
        yaw_bins=int(yaw_bins),
    )

    if int(yaw_bins) <= 1:
        return orient_traj.astype(np.float32), base_target[None, ...].astype(np.float32)

    target_volume = np.ones((int(yaw_bins),) + occ.shape, dtype=np.float32)
    for yaw_idx in range(int(yaw_bins)):
        if np.any(orient_traj[yaw_idx] > 0.5):
            orient_target = build_target_cost_map(
                occ_map=occ,
                opt_traj=orient_traj[yaw_idx],
                normalize_quantile=normalize_quantile,
                clearance_weight=clearance_weight,
                clearance_safe_distance=clearance_safe_distance,
                clearance_power=clearance_power,
                clearance_penalize_path=clearance_penalize_path,
            )
            target_volume[yaw_idx] = np.maximum(base_target, orient_target).astype(np.float32)
        else:
            target_volume[yaw_idx] = np.ones_like(base_target, dtype=np.float32)
            target_volume[yaw_idx][occ >= 0.5] = 1.0

    return orient_traj.astype(np.float32), target_volume.astype(np.float32)
