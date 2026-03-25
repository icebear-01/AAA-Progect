from __future__ import annotations

import math
import numpy as np

from hybrid_astar_guided import GuidedHybridAstar
from hybrid_astar_guided.grid_astar import astar_8conn, astar_8conn_stats, path_length_8conn
from neural_astar.utils.guidance_targets import build_exact_grid_heuristic_map, build_octile_heuristic_map


def test_astar_8conn_guidance_affects_path():
    occ = np.zeros((7, 7), dtype=np.float32)
    start = (0, 3)
    goal = (6, 3)

    # Penalize the straight row, forcing a detour when lambda > 0.
    guidance = np.zeros_like(occ)
    guidance[3, 1:6] = 10.0

    path_no_guidance = astar_8conn(occ, start, goal, guidance_cost=guidance, lambda_guidance=0.0)
    path_guided = astar_8conn(occ, start, goal, guidance_cost=guidance, lambda_guidance=2.0)

    assert path_no_guidance is not None
    assert path_guided is not None

    mid_no = sum(1 for x, y in path_no_guidance if y == 3 and 1 <= x <= 5)
    mid_guided = sum(1 for x, y in path_guided if y == 3 and 1 <= x <= 5)
    assert mid_guided < mid_no


def test_astar_8conn_diagonal_cost_and_corner_cut_options():
    occ = np.zeros((4, 4), dtype=np.float32)
    # Block side-adjacent cells around start so only corner-cut diagonal can pass.
    occ[0, 1] = 1.0
    occ[1, 0] = 1.0
    start = (0, 0)
    goal = (3, 3)

    path_no_cut = astar_8conn(occ, start, goal, allow_corner_cut=False)
    path_allow_cut = astar_8conn(occ, start, goal, allow_corner_cut=True)
    assert path_no_cut is None
    assert path_allow_cut is not None

    path_unit = astar_8conn(occ, start, goal, allow_corner_cut=True, diagonal_cost=1.0)
    assert path_unit is not None
    assert path_length_8conn(path_unit, diagonal_cost=1.0) == float(len(path_unit) - 1)


def test_astar_8conn_stats_reports_expanded_nodes():
    occ = np.zeros((7, 7), dtype=np.float32)
    start = (0, 3)
    goal = (6, 3)
    stats = astar_8conn_stats(occ, start, goal)
    assert stats.success
    assert stats.path is not None
    assert stats.expanded_nodes >= len(stats.path)


def test_astar_8conn_octile_heuristic_is_no_worse_than_euclidean():
    occ = np.zeros((48, 48), dtype=np.float32)
    occ[10:38, 22] = 1.0
    occ[24, 22] = 0.0
    start = (2, 2)
    goal = (44, 44)

    euclid = astar_8conn_stats(
        occ,
        start,
        goal,
        heuristic_mode="euclidean",
    )
    octile = astar_8conn_stats(
        occ,
        start,
        goal,
        heuristic_mode="octile",
    )

    assert euclid.success
    assert octile.success
    assert octile.expanded_nodes <= euclid.expanded_nodes


def test_astar_8conn_heuristic_bonus_can_reward_low_cost_corridor():
    occ = np.zeros((12, 12), dtype=np.float32)
    start = (1, 6)
    goal = (10, 6)
    guidance = np.full_like(occ, 0.9, dtype=np.float32)
    guidance[6, 1:11] = 0.1

    baseline = astar_8conn_stats(
        occ,
        start,
        goal,
        guidance_cost=guidance,
        lambda_guidance=0.0,
        heuristic_mode="octile",
    )
    guided = astar_8conn_stats(
        occ,
        start,
        goal,
        guidance_cost=guidance,
        lambda_guidance=0.5,
        heuristic_mode="octile",
        guidance_integration_mode="heuristic_bonus",
        guidance_bonus_threshold=0.6,
    )

    assert baseline.success
    assert guided.success
    assert guided.expanded_nodes <= baseline.expanded_nodes


def test_astar_8conn_residual_heuristic_can_reduce_expansions():
    occ = np.zeros((48, 48), dtype=np.float32)
    occ[6:42, 20] = 1.0
    occ[18, 20] = 0.0
    occ[30, 20] = 0.0
    start = (2, 24)
    goal = (44, 24)

    exact = build_exact_grid_heuristic_map(occ, goal_xy=goal)
    octile = build_octile_heuristic_map(occ.shape, goal_xy=goal)
    residual = np.maximum(exact - octile, 0.0).astype(np.float32)

    baseline = astar_8conn_stats(
        occ,
        start,
        goal,
        heuristic_mode="octile",
    )
    guided = astar_8conn_stats(
        occ,
        start,
        goal,
        heuristic_mode="octile",
        heuristic_residual_map=residual,
        residual_weight=1.0,
    )

    assert baseline.success
    assert guided.success
    assert guided.expanded_nodes <= baseline.expanded_nodes


def test_astar_8conn_zero_residual_confidence_matches_baseline():
    occ = np.zeros((40, 40), dtype=np.float32)
    occ[8:32, 18] = 1.0
    occ[20, 18] = 0.0
    start = (2, 20)
    goal = (36, 20)

    residual = np.full_like(occ, 3.0, dtype=np.float32)
    zero_conf = np.zeros_like(occ, dtype=np.float32)

    baseline = astar_8conn_stats(
        occ,
        start,
        goal,
        heuristic_mode="octile",
    )
    gated = astar_8conn_stats(
        occ,
        start,
        goal,
        heuristic_mode="octile",
        heuristic_residual_map=residual,
        residual_confidence_map=zero_conf,
        residual_weight=1.0,
    )

    assert baseline.success
    assert gated.success
    assert gated.expanded_nodes == baseline.expanded_nodes


def test_guided_hybrid_astar_smoke():
    occ = np.zeros((32, 32), dtype=np.float32)
    cost = np.zeros_like(occ)

    planner = GuidedHybridAstar(lambda_guidance=1.0)
    res = planner.plan(
        occ_map=occ,
        start_pose=(2, 2, 0.0),
        goal_pose=(28, 28, 0.0),
        cost_map=cost,
        lambda_guidance=1.0,
    )

    assert res.success
    assert res.expanded_nodes > 0
    assert res.path_length > 0.0
    assert len(res.path) > 1


def test_guided_hybrid_astar_strict_goal_with_rs_shot():
    occ = np.zeros((48, 48), dtype=np.float32)
    cost = np.zeros_like(occ)

    goal = (40.0, 36.0, math.radians(70.0))
    planner = GuidedHybridAstar(
        allow_reverse=True,
        strict_goal_pose=True,
        use_rs_shot=True,
        rs_shot_trigger_dist=20.0,
        goal_tolerance_xy=0.0,
        goal_tolerance_yaw_deg=0.0,
    )
    res = planner.plan(
        occ_map=occ,
        start_pose=(4.0, 5.0, 0.0),
        goal_pose=goal,
        cost_map=cost,
        lambda_guidance=0.0,
    )

    assert res.success
    assert len(res.path) > 1
    ex, ey, eyaw = res.path[-1]
    assert abs(ex - goal[0]) < 1e-2
    assert abs(ey - goal[1]) < 1e-2
    assert abs(math.atan2(math.sin(eyaw - goal[2]), math.cos(eyaw - goal[2]))) < 1e-2


def test_guided_hybrid_astar_guidance_on_g_smoke():
    occ = np.zeros((32, 32), dtype=np.float32)
    # Lower guidance values near goal.
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    dist = np.sqrt((xx - 26) ** 2 + (yy - 26) ** 2).astype(np.float32)
    cost = dist / max(float(dist.max()), 1.0)

    planner = GuidedHybridAstar()
    res = planner.plan(
        occ_map=occ,
        start_pose=(2.0, 2.0, 0.0),
        goal_pose=(26.0, 26.0, 0.0),
        cost_map=cost,
        lambda_guidance=1.0,
        max_expansions=20000,
    )

    assert res.success
    assert res.expanded_nodes > 0


def test_guided_hybrid_astar_guidance_normalize_clip_smoke():
    occ = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    cost = np.sqrt((xx - 28) ** 2 + (yy - 28) ** 2).astype(np.float32)
    cost /= max(float(cost.max()), 1.0)
    # Add outliers to validate normalization/clipping path.
    cost[0, 0] = 100.0
    cost[1, 1] = 50.0

    planner = GuidedHybridAstar(
        normalize_guidance_cost=True,
        guidance_norm_p_low=5.0,
        guidance_norm_p_high=95.0,
        guidance_clip_low=0.05,
        guidance_clip_high=0.95,
    )
    res = planner.plan(
        occ_map=occ,
        start_pose=(2.0, 2.0, 0.0),
        goal_pose=(28.0, 28.0, 0.0),
        cost_map=cost,
        lambda_guidance=1.0,
        max_expansions=20000,
    )

    assert res.success
    assert res.expanded_nodes > 0


def test_guided_hybrid_astar_accepts_orientation_cost_volume():
    occ = np.zeros((24, 24), dtype=np.float32)
    cost = np.zeros((8, 24, 24), dtype=np.float32)
    cost[:] = 0.5
    cost[0, :, :] = 0.1

    planner = GuidedHybridAstar(lambda_guidance=1.0)
    res = planner.plan(
        occ_map=occ,
        start_pose=(2.0, 2.0, 0.0),
        goal_pose=(20.0, 20.0, 0.0),
        cost_map=cost,
        lambda_guidance=1.0,
        max_expansions=20000,
    )

    assert res.success
    assert res.expanded_nodes > 0


def test_guided_hybrid_astar_guidance_temperature_power_transform():
    occ = np.zeros((2, 3), dtype=np.float32)
    cost = np.array(
        [
            [0.25, 0.50, 0.75],
            [0.10, 0.90, 0.30],
        ],
        dtype=np.float32,
    )
    occ[1, 1] = 1.0

    planner = GuidedHybridAstar(
        normalize_guidance_cost=False,
        guidance_clip_low=0.0,
        guidance_clip_high=1.0,
        guidance_temperature=0.5,
        guidance_power=2.0,
    )
    prepared = planner._prepare_guidance_cost(cost, occ)

    assert prepared.shape == cost.shape
    assert prepared[0, 0] < cost[0, 0]
    assert prepared[0, 2] > cost[0, 2]
    assert prepared[1, 1] == 1.0


def test_guided_hybrid_astar_invalid_guidance_transform_params():
    try:
        GuidedHybridAstar(guidance_temperature=0.0)
        raise AssertionError("Expected ValueError for non-positive temperature")
    except ValueError:
        pass

    try:
        GuidedHybridAstar(guidance_integration_mode="unknown")
        raise AssertionError("Expected ValueError for unknown guidance mode")
    except ValueError:
        pass

    try:
        GuidedHybridAstar(guidance_bonus_threshold=0.0)
        raise AssertionError("Expected ValueError for non-positive bonus threshold")
    except ValueError:
        pass


def test_guided_hybrid_astar_heuristic_bias_mode_smoke():
    occ = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    cost = np.sqrt((xx - 26) ** 2 + (yy - 26) ** 2).astype(np.float32)
    cost /= max(float(cost.max()), 1.0)
    planner = GuidedHybridAstar(guidance_integration_mode="heuristic_bias")
    res = planner.plan(
        occ_map=occ,
        start_pose=(2.0, 2.0, 0.0),
        goal_pose=(26.0, 26.0, 0.0),
        cost_map=cost,
        lambda_guidance=1.0,
        max_expansions=20000,
    )
    assert res.success
    assert res.expanded_nodes > 0


def test_guided_hybrid_astar_heuristic_bonus_mode_smoke():
    occ = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    cost = np.sqrt((xx - 26) ** 2 + (yy - 26) ** 2).astype(np.float32)
    cost /= max(float(cost.max()), 1.0)
    planner = GuidedHybridAstar(
        guidance_integration_mode="heuristic_bonus",
        guidance_bonus_threshold=0.6,
    )
    res = planner.plan(
        occ_map=occ,
        start_pose=(2.0, 2.0, 0.0),
        goal_pose=(26.0, 26.0, 0.0),
        cost_map=cost,
        lambda_guidance=1.0,
        max_expansions=20000,
    )
    assert res.success
    assert res.expanded_nodes > 0


def test_guided_hybrid_astar_heuristic_bonus_bias_is_non_positive():
    planner = GuidedHybridAstar(
        guidance_integration_mode="heuristic_bonus",
        guidance_bonus_threshold=0.5,
    )
    assert planner._guidance_priority_bias(0.2) < 0.0
    assert planner._guidance_priority_bias(0.5) == 0.0
    assert planner._guidance_priority_bias(0.8) == 0.0

    try:
        GuidedHybridAstar(guidance_power=0.0)
        raise AssertionError("Expected ValueError for non-positive power")
    except ValueError:
        pass
