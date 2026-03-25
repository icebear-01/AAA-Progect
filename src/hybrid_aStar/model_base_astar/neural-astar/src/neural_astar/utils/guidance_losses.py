"""Training losses for guidance cost-map learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def expansion_proxy_loss(
    cost_map: torch.Tensor,
    target_traj: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float = 0.05,
    softness: float = 0.05,
    budget_multiplier: float = 1.5,
) -> torch.Tensor:
    """Penalize off-corridor low-cost spillover as a proxy for search effort.

    The loss treats corridor cells as the small set that should remain attractive
    to search. It penalizes:
    1) off-corridor cells whose predicted cost is too close to the corridor cost
    2) total free-space mass that falls below the corridor-derived threshold
    """
    if cost_map.shape != target_traj.shape:
        raise ValueError(
            f"cost_map and target_traj shape mismatch: {tuple(cost_map.shape)} vs {tuple(target_traj.shape)}"
        )
    if free_mask.ndim != 4 or free_mask.shape[1] != 1:
        raise ValueError(f"free_mask must be [B,1,H,W], got {tuple(free_mask.shape)}")

    temp = max(float(softness), 1e-4)
    losses = []
    for bi in range(cost_map.shape[0]):
        free_mask_2d = free_mask[bi, 0] > 0.5
        free_count = free_mask_2d.float().sum().clamp_min(1.0)
        if not bool(free_mask_2d.any()):
            continue

        for ci in range(cost_map.shape[1]):
            pos_mask = (target_traj[bi, ci] > 0.5) & free_mask_2d
            if not bool(pos_mask.any()):
                continue

            neg_mask = free_mask_2d & (~pos_mask)
            if not bool(neg_mask.any()):
                continue

            pos_vals = cost_map[bi, ci][pos_mask]
            neg_vals = cost_map[bi, ci][neg_mask]
            free_vals = cost_map[bi, ci][free_mask_2d]

            threshold = pos_vals.mean() + float(margin)
            neg_low_mass = torch.sigmoid((threshold - neg_vals) / temp).mean()
            total_low_mass = torch.sigmoid((threshold - free_vals) / temp).mean()

            target_frac = pos_mask.float().sum() / free_count
            allowed_frac = torch.clamp(target_frac * float(budget_multiplier), 0.0, 1.0)
            mass_penalty = F.relu(total_low_mass - allowed_frac)
            losses.append(neg_low_mass + mass_penalty)

    if not losses:
        return torch.zeros((), device=cost_map.device, dtype=cost_map.dtype)
    return torch.stack(losses).mean()


def hybrid_expansion_loss(
    cost_map: torch.Tensor,
    target_traj: torch.Tensor,
    expanded_trace_map: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Penalize low predicted cost on cells heavily visited by Hybrid A* expansions."""
    return _visited_expansion_loss(
        cost_map=cost_map,
        target_traj=target_traj,
        visited_map=expanded_trace_map,
        free_mask=free_mask,
        margin=margin,
        map_name="expanded_trace_map",
    )


def astar_expansion_loss(
    cost_map: torch.Tensor,
    target_traj: torch.Tensor,
    astar_expanded_map: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Penalize low predicted cost on cells heavily visited by 2D A* expansions."""
    return _visited_expansion_loss(
        cost_map=cost_map,
        target_traj=target_traj,
        visited_map=astar_expanded_map,
        free_mask=free_mask,
        margin=margin,
        map_name="astar_expanded_map",
    )


def _visited_expansion_loss(
    cost_map: torch.Tensor,
    target_traj: torch.Tensor,
    visited_map: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float,
    map_name: str,
) -> torch.Tensor:
    """Shared visitation-weighted loss for search expansion supervision."""
    if cost_map.shape != target_traj.shape:
        raise ValueError(
            f"cost_map and target_traj shape mismatch: {tuple(cost_map.shape)} vs {tuple(target_traj.shape)}"
        )
    if visited_map.ndim != 4 or visited_map.shape[1] != 1:
        raise ValueError(
            f"{map_name} must be [B,1,H,W], got {tuple(visited_map.shape)}"
        )
    if free_mask.ndim != 4 or free_mask.shape[1] != 1:
        raise ValueError(f"free_mask must be [B,1,H,W], got {tuple(free_mask.shape)}")

    losses = []
    for bi in range(cost_map.shape[0]):
        free_mask_2d = free_mask[bi, 0] > 0.5
        if not bool(free_mask_2d.any()):
            continue

        corridor_union = target_traj[bi].amax(dim=0) > 0.5
        trace_weights = visited_map[bi, 0]
        hard_mask = (trace_weights > 0.0) & free_mask_2d & (~corridor_union)
        if not bool(hard_mask.any()):
            continue

        weights = trace_weights[hard_mask]
        for ci in range(cost_map.shape[1]):
            pos_mask = (target_traj[bi, ci] > 0.5) & free_mask_2d
            if not bool(pos_mask.any()):
                continue

            pos_vals = cost_map[bi, ci][pos_mask]
            hard_vals = cost_map[bi, ci][hard_mask]
            threshold = pos_vals.mean() + float(margin)
            penalty = F.relu(threshold - hard_vals)
            losses.append((penalty * weights).sum() / weights.sum().clamp_min(1e-6))

    if not losses:
        return torch.zeros((), device=cost_map.device, dtype=cost_map.dtype)
    return torch.stack(losses).mean()
