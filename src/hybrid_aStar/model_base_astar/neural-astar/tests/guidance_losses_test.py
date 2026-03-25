from __future__ import annotations

import torch

from neural_astar.utils.guidance_losses import (
    astar_expansion_loss,
    expansion_proxy_loss,
    hybrid_expansion_loss,
)


def test_expansion_proxy_loss_penalizes_broad_low_cost_spillover():
    free_mask = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[0, 0, 1:3, 1:3] = 1.0

    focused = torch.full((1, 1, 4, 4), 0.8, dtype=torch.float32)
    focused[0, 0, 1:3, 1:3] = 0.1

    spilled = torch.full((1, 1, 4, 4), 0.3, dtype=torch.float32)
    spilled[0, 0, 1:3, 1:3] = 0.1

    focused_loss = expansion_proxy_loss(
        focused,
        target,
        free_mask,
        margin=0.05,
        softness=0.05,
        budget_multiplier=1.5,
    )
    spilled_loss = expansion_proxy_loss(
        spilled,
        target,
        free_mask,
        margin=0.05,
        softness=0.05,
        budget_multiplier=1.5,
    )

    assert spilled_loss.item() > focused_loss.item()


def test_expansion_proxy_loss_returns_zero_without_corridor():
    free_mask = torch.ones((1, 1, 3, 3), dtype=torch.float32)
    target = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
    cost = torch.full((1, 1, 3, 3), 0.5, dtype=torch.float32)
    loss = expansion_proxy_loss(cost, target, free_mask)
    assert loss.item() == 0.0


def test_hybrid_expansion_loss_penalizes_low_cost_on_expanded_offpath_cells():
    free_mask = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[0, 0, 1:3, 1:3] = 1.0
    expanded_trace = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    expanded_trace[0, 0, 0, 0] = 1.0
    expanded_trace[0, 0, 0, 1] = 0.5

    focused = torch.full((1, 1, 4, 4), 0.8, dtype=torch.float32)
    focused[0, 0, 1:3, 1:3] = 0.1

    leaked = focused.clone()
    leaked[0, 0, 0, 0] = 0.05
    leaked[0, 0, 0, 1] = 0.10

    focused_loss = hybrid_expansion_loss(
        cost_map=focused,
        target_traj=target,
        expanded_trace_map=expanded_trace,
        free_mask=free_mask,
        margin=0.05,
    )
    leaked_loss = hybrid_expansion_loss(
        cost_map=leaked,
        target_traj=target,
        expanded_trace_map=expanded_trace,
        free_mask=free_mask,
        margin=0.05,
    )

    assert leaked_loss.item() > focused_loss.item()


def test_astar_expansion_loss_penalizes_low_cost_on_expanded_offpath_cells():
    free_mask = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[0, 0, 1:3, 1:3] = 1.0
    expanded_map = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    expanded_map[0, 0, 0, 2] = 1.0
    expanded_map[0, 0, 0, 3] = 0.5

    focused = torch.full((1, 1, 4, 4), 0.8, dtype=torch.float32)
    focused[0, 0, 1:3, 1:3] = 0.1

    leaked = focused.clone()
    leaked[0, 0, 0, 2] = 0.05
    leaked[0, 0, 0, 3] = 0.12

    focused_loss = astar_expansion_loss(
        cost_map=focused,
        target_traj=target,
        astar_expanded_map=expanded_map,
        free_mask=free_mask,
        margin=0.05,
    )
    leaked_loss = astar_expansion_loss(
        cost_map=leaked,
        target_traj=target,
        astar_expanded_map=expanded_map,
        free_mask=free_mask,
        margin=0.05,
    )

    assert leaked_loss.item() > focused_loss.item()
