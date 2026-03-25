from __future__ import annotations

import numpy as np

from neural_astar.utils.confidence_rl import (
    compute_confidence_rl_reward,
    decode_residual_prediction,
    select_topk_confidence_cells,
)


def test_decode_residual_prediction_handles_log1p():
    pred = np.log1p(np.array([[0.0, 1.5], [2.0, 3.0]], dtype=np.float32))
    decoded = decode_residual_prediction(pred, "log1p")

    assert np.allclose(decoded, np.array([[0.0, 1.5], [2.0, 3.0]], dtype=np.float32))


def test_select_topk_confidence_cells_respects_free_mask_and_topk():
    residual = np.array(
        [[0.1, 3.0, 1.0], [2.0, 5.0, 4.0]],
        dtype=np.float32,
    )
    occ = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    mask = select_topk_confidence_cells(residual, occ, topk=2, min_residual=0.5)

    expected = np.array(
        [[False, False, False], [False, True, True]],
        dtype=bool,
    )
    assert np.array_equal(mask, expected)


def test_compute_confidence_rl_reward_penalizes_path_growth_and_failure():
    reward = compute_confidence_rl_reward(
        baseline_expanded=600,
        sampled_expanded=500,
        baseline_path_length=65.0,
        sampled_path_length=66.5,
        sampled_success=False,
        reward_scale=100.0,
        path_length_penalty=20.0,
        failure_penalty=400.0,
    )

    # 100 search gain - 30 path penalty - 400 failure penalty = -330
    assert reward == -3.3
