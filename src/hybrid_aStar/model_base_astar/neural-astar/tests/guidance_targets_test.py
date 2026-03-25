from __future__ import annotations

import numpy as np

from neural_astar.utils.guidance_targets import (
    build_exact_grid_heuristic_map,
    build_octile_heuristic_map,
    build_residual_heuristic_maps,
    build_expanded_trace_map,
    build_orientation_target_maps,
    build_target_cost_map,
    exact_grid_heuristic_from_opt_dist,
    yaw_to_bin,
)


def test_build_target_cost_map_smoke():
    occ = np.zeros((9, 9), dtype=np.float32)
    occ[0, 0] = 1.0
    traj = np.zeros((9, 9), dtype=np.float32)
    traj[4, 2:7] = 1.0

    target = build_target_cost_map(occ, traj)

    assert target.shape == occ.shape
    assert np.isfinite(target).all()
    assert (target >= 0.0).all()
    assert (target <= 1.0).all()
    assert target[4, 4] == 0.0
    assert target[0, 0] == 1.0
    assert target[4, 1] >= target[4, 2]
    assert target[1, 4] > 0.0


def test_build_target_cost_map_empty_path_defaults_to_high_cost():
    occ = np.zeros((5, 5), dtype=np.float32)
    target = build_target_cost_map(occ, np.zeros_like(occ))

    assert target.shape == (5, 5)
    assert np.all(target == 1.0)


def test_build_orientation_target_maps_smoke():
    occ = np.zeros((8, 8), dtype=np.float32)
    opt = np.zeros((8, 8), dtype=np.float32)
    opt[3, 1:6] = 1.0
    path = np.array(
        [[1.0, 3.0, 0.0], [3.0, 3.0, 0.0], [5.0, 3.0, 0.0]],
        dtype=np.float32,
    )

    orient_traj, orient_target = build_orientation_target_maps(
        occ_map=occ,
        opt_traj=opt,
        path_poses=path,
        yaw_bins=8,
    )

    assert orient_traj.shape == (8, 8, 8)
    assert orient_target.shape == (8, 8, 8)
    assert np.isclose(orient_traj.sum(), opt.sum())
    heading_bin = yaw_to_bin(0.0, 8)
    assert orient_traj[heading_bin, 3, 3] == 1.0
    assert orient_target[heading_bin, 3, 3] == 0.0


def test_build_expanded_trace_map_projects_and_normalizes_counts():
    occ = np.zeros((6, 6), dtype=np.float32)
    occ[0, 0] = 1.0
    trace = np.array(
        [
            [1.1, 2.0],
            [1.0, 2.2],
            [3.0, 4.0],
            [3.0, 4.0],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )

    heat = build_expanded_trace_map(occ, trace)

    assert heat.shape == occ.shape
    assert np.isfinite(heat).all()
    assert 0.0 <= float(heat.min()) <= float(heat.max()) <= 1.0
    assert heat[0, 0] == 0.0
    assert heat[2, 1] > 0.0
    assert heat[4, 3] >= heat[2, 1]


def test_build_residual_heuristic_maps_are_non_negative_and_consistent():
    occ = np.zeros((7, 7), dtype=np.float32)
    occ[1:6, 3] = 1.0
    occ[3, 3] = 0.0
    goal = (6, 6)

    exact, octile, residual = build_residual_heuristic_maps(occ, goal_xy=goal)

    assert exact.shape == occ.shape
    assert octile.shape == occ.shape
    assert residual.shape == occ.shape
    assert np.all(residual >= 0.0)
    assert np.isclose(octile[goal[1], goal[0]], 0.0)
    assert np.isclose(exact[goal[1], goal[0]], 0.0)
    free_mask = occ < 0.5
    assert np.allclose(exact[free_mask], octile[free_mask] + residual[free_mask], atol=1e-5)
    assert float(residual.max()) > 0.0


def test_exact_grid_heuristic_from_opt_dist_matches_manual_encoding():
    occ = np.zeros((3, 3), dtype=np.float32)
    opt_dist = np.array(
        [[-2.0, -1.0, 0.0], [-2.4142137, -1.4142135, -1.0], [-2.8284271, -2.4142137, -2.0]],
        dtype=np.float32,
    )

    exact = exact_grid_heuristic_from_opt_dist(occ, opt_dist)

    assert exact.shape == occ.shape
    assert np.isclose(exact[0, 2], 0.0)
    assert np.isclose(exact[0, 1], 1.0)
    assert np.isclose(exact[1, 1], 1.4142135, atol=1e-5)


def test_exact_and_octile_helpers_agree_in_empty_map():
    occ = np.zeros((6, 6), dtype=np.float32)
    goal = (5, 5)

    exact = build_exact_grid_heuristic_map(occ, goal_xy=goal)
    octile = build_octile_heuristic_map(occ.shape, goal_xy=goal)

    assert np.allclose(exact, octile, atol=1e-5)
