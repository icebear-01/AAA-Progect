"""Simple 8-connected grid A* utilities.

Coordinate convention:
- World/state: (x, y)
- NumPy map indexing: map[y, x]
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from neural_astar.utils.coords import validate_xy
from neural_astar.utils.guidance_targets import build_clearance_penalty_map


XY = Tuple[int, int]


def _heuristic(
    x: int,
    y: int,
    gx: int,
    gy: int,
    diagonal_cost: float,
    mode: str,
) -> float:
    dx = abs(int(gx) - int(x))
    dy = abs(int(gy) - int(y))
    if mode == "euclidean":
        return math.hypot(dx, dy)
    if mode == "manhattan":
        return float(dx + dy)
    if mode == "chebyshev":
        return float(max(dx, dy))
    if mode == "octile":
        dmin = float(min(dx, dy))
        dmax = float(max(dx, dy))
        return dmax + (float(diagonal_cost) - 1.0) * dmin
    raise ValueError(f"Unknown heuristic_mode: {mode}")


def _guidance_priority_bias(
    guidance_val: float,
    guidance_integration_mode: str,
    guidance_bonus_threshold: float,
) -> float:
    if guidance_integration_mode == "heuristic_bias":
        return float(guidance_val)
    if guidance_integration_mode == "heuristic_bonus":
        threshold = float(guidance_bonus_threshold)
        scaled_bonus = (float(guidance_val) - threshold) / max(threshold, 1e-6)
        return min(0.0, scaled_bonus)
    if guidance_integration_mode == "g_cost":
        return 0.0
    raise ValueError(f"Unknown guidance_integration_mode: {guidance_integration_mode}")


def _clearance_priority_bias(
    clearance_val: float,
    clearance_integration_mode: str,
) -> float:
    if clearance_integration_mode == "g_cost":
        return 0.0
    if clearance_integration_mode == "heuristic_bias":
        return float(clearance_val)
    if clearance_integration_mode == "priority_tie_break":
        return float(clearance_val)
    raise ValueError(f"Unknown clearance_integration_mode: {clearance_integration_mode}")


@dataclass
class Astar8ConnStats:
    path: Optional[List[XY]]
    expanded_nodes: int
    success: bool
    expanded_xy: List[XY] = field(default_factory=list)


def _neighbors_8(x: int, y: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for dx, dy in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            out.append((nx, ny, dx, dy))
    return out


def reconstruct_path(came_from: Dict[XY, Optional[XY]], current: XY) -> List[XY]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _astar_8conn_impl(
    occ_map: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    guidance_cost: Optional[np.ndarray] = None,
    heuristic_residual_map: Optional[np.ndarray] = None,
    residual_confidence_map: Optional[np.ndarray] = None,
    lambda_guidance: float = 0.0,
    residual_weight: float = 1.0,
    diagonal_cost: float = math.sqrt(2.0),
    allow_corner_cut: bool = True,
    heuristic_mode: str = "euclidean",
    heuristic_weight: float = 1.0,
    guidance_integration_mode: str = "g_cost",
    guidance_bonus_threshold: float = 0.5,
    clearance_weight: float = 0.0,
    clearance_safe_distance: float = 0.0,
    clearance_power: float = 2.0,
    clearance_integration_mode: str = "g_cost",
) -> Astar8ConnStats:
    """Run 8-connected A* on occupancy map.

    Args:
        occ_map: [H, W], 1=obstacle, 0=free
        start_xy: world (x, y)
        goal_xy: world (x, y)
        guidance_cost: optional [H, W], lower is better
        heuristic_residual_map: optional [H, W], non-negative residual heuristic added
            on top of the analytic anchor heuristic.
        residual_confidence_map: optional [H, W] in [0,1], multiplies the learned
            residual before it is injected into the heuristic.
        lambda_guidance: scale of guidance term in g-cost
        residual_weight: scale applied to ``heuristic_residual_map``.
        diagonal_cost: move cost for diagonal steps.
            Set to 1.0 for unit-step shortest path.
        allow_corner_cut: whether diagonal can pass when exactly one side-adjacent
            cell is obstacle. When both side-adjacent cells are occupied, the
            diagonal move is always rejected.
        heuristic_mode: heuristic used for h(x). ``octile`` is a strong admissible
            default for 8-connected planning.
        heuristic_weight: multiplier on heuristic term. ``1.0`` keeps standard A*.
        guidance_integration_mode: how guidance affects search. ``g_cost`` adds to
            accumulated path cost, ``heuristic_bias`` adds raw guidance to priority,
            and ``heuristic_bonus`` only rewards low-cost cells.
        guidance_bonus_threshold: threshold used by ``heuristic_bonus``.
        clearance_weight: scale of obstacle-clearance bias/penalty.
        clearance_safe_distance: clearance radius in grid cells. Free cells closer
            than this distance to obstacles receive extra penalty.
        clearance_power: shaping exponent used by the clearance penalty map.
        clearance_integration_mode: how clearance affects search. ``g_cost`` adds
            penalty to accumulated path cost, ``heuristic_bias`` adds a weak
            obstacle-avoidance bias to node priority, and ``priority_tie_break``
            uses the clearance penalty only as a secondary heap key.
    """
    if diagonal_cost <= 0.0:
        raise ValueError(f"diagonal_cost must be positive, got {diagonal_cost}")
    if heuristic_weight < 0.0:
        raise ValueError(f"heuristic_weight must be non-negative, got {heuristic_weight}")
    if residual_weight < 0.0:
        raise ValueError(f"residual_weight must be non-negative, got {residual_weight}")
    if clearance_weight < 0.0:
        raise ValueError(f"clearance_weight must be non-negative, got {clearance_weight}")
    if clearance_safe_distance < 0.0:
        raise ValueError(
            f"clearance_safe_distance must be non-negative, got {clearance_safe_distance}"
        )
    if clearance_integration_mode not in {"g_cost", "heuristic_bias", "priority_tie_break"}:
        raise ValueError(f"Unknown clearance_integration_mode: {clearance_integration_mode}")
    if guidance_bonus_threshold <= 0.0 or guidance_bonus_threshold > 1.0:
        raise ValueError(
            "guidance_bonus_threshold must be in (0, 1], got "
            f"{guidance_bonus_threshold}"
        )
    if guidance_integration_mode not in {"g_cost", "heuristic_bias", "heuristic_bonus"}:
        raise ValueError(f"Unknown guidance_integration_mode: {guidance_integration_mode}")
    if heuristic_mode not in {"euclidean", "manhattan", "chebyshev", "octile"}:
        raise ValueError(f"Unknown heuristic_mode: {heuristic_mode}")

    occ = np.asarray(occ_map, dtype=np.float32)
    h, w = occ.shape
    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    validate_xy(sx, sy, w, h)
    validate_xy(gx, gy, w, h)
    if occ[sy, sx] > 0.5 or occ[gy, gx] > 0.5:
        return Astar8ConnStats(path=None, expanded_nodes=0, success=False, expanded_xy=[])

    if guidance_cost is None:
        guidance = np.zeros_like(occ, dtype=np.float32)
    else:
        guidance = np.asarray(guidance_cost, dtype=np.float32)
        if guidance.shape != occ.shape:
            raise ValueError(
                f"guidance_cost shape mismatch: {guidance.shape} vs occ {occ.shape}"
            )
    if heuristic_residual_map is None:
        residual_map = np.zeros_like(occ, dtype=np.float32)
    else:
        residual_map = np.asarray(heuristic_residual_map, dtype=np.float32)
        if residual_map.shape != occ.shape:
            raise ValueError(
                f"heuristic_residual_map shape mismatch: {residual_map.shape} vs occ {occ.shape}"
            )
        residual_map = np.maximum(residual_map, 0.0).astype(np.float32)
    if residual_confidence_map is None:
        residual_conf = np.ones_like(occ, dtype=np.float32)
    else:
        residual_conf = np.asarray(residual_confidence_map, dtype=np.float32)
        if residual_conf.shape != occ.shape:
            raise ValueError(
                f"residual_confidence_map shape mismatch: {residual_conf.shape} vs occ {occ.shape}"
            )
        residual_conf = np.clip(residual_conf, 0.0, 1.0).astype(np.float32)
    residual_conf[occ > 0.5] = 0.0
    residual_map = np.where(np.isfinite(residual_map), residual_map, 0.0).astype(np.float32)
    effective_residual = (residual_map * residual_conf).astype(np.float32)
    if float(clearance_weight) > 0.0 and float(clearance_safe_distance) > 0.0:
        clearance_penalty = build_clearance_penalty_map(
            occ_map=occ,
            safe_distance=float(clearance_safe_distance),
            power=float(clearance_power),
        ).astype(np.float32)
        clearance_penalty[occ > 0.5] = 0.0
    else:
        clearance_penalty = np.zeros_like(occ, dtype=np.float32)

    start = (sx, sy)
    goal = (gx, gy)
    start_h = float(heuristic_weight) * _heuristic(
        sx,
        sy,
        gx,
        gy,
        diagonal_cost=diagonal_cost,
        mode=heuristic_mode,
    ) + float(residual_weight) * float(effective_residual[sy, sx])

    open_heap: List[Tuple[float, float, int, XY]] = []
    push_id = 0
    start_clearance_bias = (
        float(clearance_weight) * float(clearance_penalty[sy, sx])
        if clearance_integration_mode == "priority_tie_break"
        else 0.0
    )
    heapq.heappush(
        open_heap,
        (start_h, start_clearance_bias, push_id, start),
    )

    came_from: Dict[XY, Optional[XY]] = {start: None}
    g_score: Dict[XY, float] = {start: 0.0}
    closed: set[XY] = set()
    expanded_nodes = 0

    while open_heap:
        _, _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded_nodes += 1
        if current == goal:
            return Astar8ConnStats(
                path=reconstruct_path(came_from, current),
                expanded_nodes=expanded_nodes,
                success=True,
                expanded_xy=list(closed),
            )

        cx, cy = current
        for nx, ny, dx, dy in _neighbors_8(cx, cy, w, h):
            if occ[ny, nx] > 0.5:
                continue
            is_diagonal = (dx != 0) and (dy != 0)
            if is_diagonal:
                side_block_x = occ[cy, nx] > 0.5
                side_block_y = occ[ny, cx] > 0.5
                if side_block_x and side_block_y:
                    continue
                if (not allow_corner_cut) and (side_block_x or side_block_y):
                    continue
            move_cost = float(diagonal_cost) if is_diagonal else 1.0
            guidance_val = float(guidance[ny, nx])

            tentative_g = g_score[current] + move_cost
            if guidance_integration_mode == "g_cost":
                tentative_g += float(lambda_guidance) * guidance_val
            if float(clearance_weight) > 0.0 and clearance_integration_mode == "g_cost":
                tentative_g += float(clearance_weight) * float(clearance_penalty[ny, nx])
            neighbor = (nx, ny)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + float(heuristic_weight) * _heuristic(
                    nx,
                    ny,
                    gx,
                    gy,
                    diagonal_cost=diagonal_cost,
                    mode=heuristic_mode,
                )
                f += float(residual_weight) * float(effective_residual[ny, nx])
                if guidance_integration_mode != "g_cost":
                    f += float(lambda_guidance) * _guidance_priority_bias(
                        guidance_val=guidance_val,
                        guidance_integration_mode=guidance_integration_mode,
                        guidance_bonus_threshold=guidance_bonus_threshold,
                    )
                clearance_bias = 0.0
                if float(clearance_weight) > 0.0 and clearance_integration_mode != "g_cost":
                    clearance_bias = float(clearance_weight) * _clearance_priority_bias(
                        clearance_val=float(clearance_penalty[ny, nx]),
                        clearance_integration_mode=clearance_integration_mode,
                    )
                    if clearance_integration_mode == "heuristic_bias":
                        f += clearance_bias
                        clearance_bias = 0.0
                push_id += 1
                heapq.heappush(open_heap, (f, clearance_bias, push_id, neighbor))

    return Astar8ConnStats(
        path=None,
        expanded_nodes=expanded_nodes,
        success=False,
        expanded_xy=list(closed),
    )


def astar_8conn(
    occ_map: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    guidance_cost: Optional[np.ndarray] = None,
    heuristic_residual_map: Optional[np.ndarray] = None,
    residual_confidence_map: Optional[np.ndarray] = None,
    lambda_guidance: float = 0.0,
    residual_weight: float = 1.0,
    diagonal_cost: float = math.sqrt(2.0),
    allow_corner_cut: bool = True,
    heuristic_mode: str = "euclidean",
    heuristic_weight: float = 1.0,
    guidance_integration_mode: str = "g_cost",
    guidance_bonus_threshold: float = 0.5,
    clearance_weight: float = 0.0,
    clearance_safe_distance: float = 0.0,
    clearance_power: float = 2.0,
    clearance_integration_mode: str = "g_cost",
) -> Optional[List[XY]]:
    return _astar_8conn_impl(
        occ_map=occ_map,
        start_xy=start_xy,
        goal_xy=goal_xy,
        guidance_cost=guidance_cost,
        heuristic_residual_map=heuristic_residual_map,
        residual_confidence_map=residual_confidence_map,
        lambda_guidance=lambda_guidance,
        residual_weight=residual_weight,
        diagonal_cost=diagonal_cost,
        allow_corner_cut=allow_corner_cut,
        heuristic_mode=heuristic_mode,
        heuristic_weight=heuristic_weight,
        guidance_integration_mode=guidance_integration_mode,
        guidance_bonus_threshold=guidance_bonus_threshold,
        clearance_weight=clearance_weight,
        clearance_safe_distance=clearance_safe_distance,
        clearance_power=clearance_power,
        clearance_integration_mode=clearance_integration_mode,
    ).path


def astar_8conn_stats(
    occ_map: np.ndarray,
    start_xy: XY,
    goal_xy: XY,
    guidance_cost: Optional[np.ndarray] = None,
    heuristic_residual_map: Optional[np.ndarray] = None,
    residual_confidence_map: Optional[np.ndarray] = None,
    lambda_guidance: float = 0.0,
    residual_weight: float = 1.0,
    diagonal_cost: float = math.sqrt(2.0),
    allow_corner_cut: bool = True,
    heuristic_mode: str = "euclidean",
    heuristic_weight: float = 1.0,
    guidance_integration_mode: str = "g_cost",
    guidance_bonus_threshold: float = 0.5,
    clearance_weight: float = 0.0,
    clearance_safe_distance: float = 0.0,
    clearance_power: float = 2.0,
    clearance_integration_mode: str = "g_cost",
) -> Astar8ConnStats:
    return _astar_8conn_impl(
        occ_map=occ_map,
        start_xy=start_xy,
        goal_xy=goal_xy,
        guidance_cost=guidance_cost,
        heuristic_residual_map=heuristic_residual_map,
        residual_confidence_map=residual_confidence_map,
        lambda_guidance=lambda_guidance,
        residual_weight=residual_weight,
        diagonal_cost=diagonal_cost,
        allow_corner_cut=allow_corner_cut,
        heuristic_mode=heuristic_mode,
        heuristic_weight=heuristic_weight,
        guidance_integration_mode=guidance_integration_mode,
        guidance_bonus_threshold=guidance_bonus_threshold,
        clearance_weight=clearance_weight,
        clearance_safe_distance=clearance_safe_distance,
        clearance_power=clearance_power,
        clearance_integration_mode=clearance_integration_mode,
    )


def path_length_8conn(path_xy: Sequence[XY], diagonal_cost: float = math.sqrt(2.0)) -> float:
    """Compute 8-connected polyline length with configurable diagonal step cost."""
    if len(path_xy) < 2:
        return 0.0
    if diagonal_cost <= 0.0:
        raise ValueError(f"diagonal_cost must be positive, got {diagonal_cost}")

    total = 0.0
    for i in range(1, len(path_xy)):
        x0, y0 = int(path_xy[i - 1][0]), int(path_xy[i - 1][1])
        x1, y1 = int(path_xy[i][0]), int(path_xy[i][1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx > 1 or dy > 1:
            raise ValueError(f"Non-8conn step in path at index {i-1}->{i}: {(x0, y0)} -> {(x1, y1)}")
        if dx == 0 and dy == 0:
            continue
        total += float(diagonal_cost) if (dx == 1 and dy == 1) else 1.0
    return total
