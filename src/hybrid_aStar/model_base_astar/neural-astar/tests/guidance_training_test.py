from __future__ import annotations

import csv

import torch

from neural_astar.utils.guidance_training import (
    build_case_sampling_weights,
    build_residual_regression_weights,
    compute_linear_warmup_scale,
    load_hard_case_indices,
    masked_smooth_l1_mean,
    resolve_best_checkpoint_score,
)


def test_build_residual_regression_weights_emphasizes_near_path_and_expanded_cells():
    free_mask = torch.ones((1, 1, 3, 4), dtype=torch.float32)
    free_mask[0, 0, 0, 0] = 0.0

    target_cost_dense = torch.ones((1, 1, 3, 4), dtype=torch.float32)
    target_cost_dense[0, 0, 1, 1] = 0.0
    target_cost_dense[0, 0, 1, 2] = 0.25

    astar_expanded_map = torch.zeros((1, 1, 3, 4), dtype=torch.float32)
    astar_expanded_map[0, 0, 2, 3] = 1.0

    weights = build_residual_regression_weights(
        free_mask=free_mask,
        target_cost_dense=target_cost_dense,
        astar_expanded_map=astar_expanded_map,
        near_path_weight=1.0,
        astar_expanded_weight=2.0,
    )

    assert weights.shape == target_cost_dense.shape
    assert weights[0, 0, 0, 0].item() == 0.0
    assert weights[0, 0, 1, 1].item() > weights[0, 0, 0, 1].item()
    assert weights[0, 0, 1, 2].item() > weights[0, 0, 0, 1].item()
    assert weights[0, 0, 2, 3].item() > weights[0, 0, 0, 1].item()


def test_resolve_best_checkpoint_score_auto_falls_back_to_val_loss():
    metric, score = resolve_best_checkpoint_score(
        "auto",
        val_loss=0.25,
        val_grid_astar_sr=None,
        val_grid_astar_expanded=None,
    )

    assert metric == "val_loss"
    assert score == (0.25,)


def test_resolve_best_checkpoint_score_grid_metric_prioritizes_success_then_expanded():
    metric_a, score_a = resolve_best_checkpoint_score(
        "grid_astar_expanded",
        val_loss=0.5,
        val_grid_astar_sr=1.0,
        val_grid_astar_expanded=550.0,
    )
    metric_b, score_b = resolve_best_checkpoint_score(
        "grid_astar_expanded",
        val_loss=0.2,
        val_grid_astar_sr=0.95,
        val_grid_astar_expanded=400.0,
    )

    assert metric_a == "grid_astar_expanded"
    assert metric_b == "grid_astar_expanded"
    assert score_a < score_b


def test_compute_linear_warmup_scale_behaves_as_expected():
    assert compute_linear_warmup_scale(epoch_index=0, warmup_epochs=5) == 0.0
    assert compute_linear_warmup_scale(epoch_index=1, warmup_epochs=5) == 0.2
    assert compute_linear_warmup_scale(epoch_index=3, warmup_epochs=5) == 0.6
    assert compute_linear_warmup_scale(epoch_index=5, warmup_epochs=5) == 1.0
    assert compute_linear_warmup_scale(epoch_index=9, warmup_epochs=5) == 1.0
    assert compute_linear_warmup_scale(epoch_index=1, warmup_epochs=0) == 1.0


def test_load_hard_case_indices_filters_and_sorts_rows(tmp_path):
    csv_path = tmp_path / "case_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "learned_minus_improved"])
        writer.writerow([3, -5.0])
        writer.writerow([7, 2.0])
        writer.writerow([1, 8.0])
        writer.writerow([9, 4.0])

    indices = load_hard_case_indices(
        csv_path,
        top_fraction=0.5,
        min_delta=1.0,
    )

    assert indices == [1, 9]


def test_build_case_sampling_weights_emphasizes_requested_indices():
    weights = build_case_sampling_weights(
        5,
        emphasized_indices=[1, 3, 10, -1],
        emphasized_weight=4.0,
    )

    assert weights.tolist() == [1.0, 4.0, 1.0, 4.0, 1.0]


def test_masked_smooth_l1_mean_supports_weighting():
    pred = torch.tensor([[[[0.0, 2.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    weights = torch.tensor([[[[0.0, 2.0]]]], dtype=torch.float32)

    unweighted = masked_smooth_l1_mean(pred, target)
    weighted = masked_smooth_l1_mean(pred, target, weights)

    assert float(unweighted) > 0.0
    assert torch.isclose(weighted, torch.tensor(1.5))
