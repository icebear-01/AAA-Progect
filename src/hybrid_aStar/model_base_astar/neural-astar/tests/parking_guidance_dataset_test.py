from __future__ import annotations

import numpy as np

from neural_astar.datasets import ParkingGuidanceDataset


def test_parking_guidance_dataset_loads_npz(tmp_path):
    h, w = 16, 16
    occ = np.zeros((h, w), dtype=np.float32)
    start = np.zeros((1, h, w), dtype=np.float32)
    goal = np.zeros((1, h, w), dtype=np.float32)
    opt = np.zeros((1, h, w), dtype=np.float32)

    start[0, 2, 3] = 1.0
    goal[0, 10, 11] = 1.0
    opt[0, 2:11, 3] = 1.0

    np.savez_compressed(
        tmp_path / "sample_00000.npz",
        occ_map=occ,
        start_map=start,
        goal_map=goal,
        opt_traj=opt,
    )

    ds = ParkingGuidanceDataset(tmp_path)
    assert len(ds) == 1

    sample = ds[0]
    assert set(sample.keys()) == {
        "occ_map",
        "start_map",
        "goal_map",
        "opt_traj",
        "target_cost",
        "astar_expanded_map",
        "exact_heuristic_map",
        "octile_heuristic_map",
        "residual_heuristic_map",
        "expanded_trace_map",
        "start_pose",
        "goal_pose",
    }
    assert tuple(sample["occ_map"].shape) == (1, h, w)
    assert tuple(sample["start_map"].shape) == (1, h, w)
    assert tuple(sample["goal_map"].shape) == (1, h, w)
    assert tuple(sample["opt_traj"].shape) == (1, h, w)
    assert tuple(sample["target_cost"].shape) == (1, h, w)
    assert tuple(sample["astar_expanded_map"].shape) == (1, h, w)
    assert tuple(sample["exact_heuristic_map"].shape) == (1, h, w)
    assert tuple(sample["octile_heuristic_map"].shape) == (1, h, w)
    assert tuple(sample["residual_heuristic_map"].shape) == (1, h, w)
    assert tuple(sample["expanded_trace_map"].shape) == (1, h, w)
    assert tuple(sample["start_pose"].shape) == (3,)
    assert tuple(sample["goal_pose"].shape) == (3,)
    assert tuple(sample["start_pose"].numpy().tolist()) == (3.0, 2.0, 0.0)
    assert tuple(sample["goal_pose"].numpy().tolist()) == (11.0, 10.0, 0.0)
    assert sample["target_cost"].numpy()[0, 2, 3] == 0.0


def test_parking_guidance_dataset_builds_orientation_targets_from_path(tmp_path):
    h, w = 12, 12
    occ = np.zeros((h, w), dtype=np.float32)
    start = np.zeros((1, h, w), dtype=np.float32)
    goal = np.zeros((1, h, w), dtype=np.float32)
    opt = np.zeros((1, h, w), dtype=np.float32)

    start[0, 2, 2] = 1.0
    goal[0, 2, 8] = 1.0
    opt[0, 2, 2:9] = 1.0
    path_poses = np.array(
        [[2.0, 2.0, 0.0], [4.0, 2.0, 0.0], [6.0, 2.0, 0.0], [8.0, 2.0, 0.0]],
        dtype=np.float32,
    )

    np.savez_compressed(
        tmp_path / "sample_00000.npz",
        occ_map=occ,
        start_map=start,
        goal_map=goal,
        opt_traj=opt,
        path_poses=path_poses,
    )

    ds = ParkingGuidanceDataset(tmp_path, orientation_bins=8)
    sample = ds[0]

    assert tuple(sample["opt_traj_orient"].shape) == (8, h, w)
    assert tuple(sample["target_cost_orient"].shape) == (8, h, w)
    assert tuple(sample["astar_expanded_map"].shape) == (1, h, w)
    assert tuple(sample["residual_heuristic_map"].shape) == (1, h, w)
    assert tuple(sample["expanded_trace_map"].shape) == (1, h, w)
    assert float(sample["opt_traj_orient"].sum().item()) == float(opt.sum())
    assert sample["target_cost_orient"].numpy().min() == 0.0


def test_parking_guidance_dataset_builds_expanded_trace_map_from_xy(tmp_path):
    h, w = 10, 10
    occ = np.zeros((h, w), dtype=np.float32)
    start = np.zeros((1, h, w), dtype=np.float32)
    goal = np.zeros((1, h, w), dtype=np.float32)
    opt = np.zeros((1, h, w), dtype=np.float32)
    start[0, 1, 1] = 1.0
    goal[0, 8, 8] = 1.0
    opt[0, 1:9, 1] = 1.0
    expanded_trace_xy = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0]], dtype=np.float32)

    np.savez_compressed(
        tmp_path / "sample_00000.npz",
        occ_map=occ,
        start_map=start,
        goal_map=goal,
        opt_traj=opt,
        expanded_trace_xy=expanded_trace_xy,
    )

    ds = ParkingGuidanceDataset(tmp_path)
    sample = ds[0]

    assert tuple(sample["astar_expanded_map"].shape) == (1, h, w)
    assert float(sample["astar_expanded_map"].sum().item()) > 0.0
    assert tuple(sample["residual_heuristic_map"].shape) == (1, h, w)
    assert float(sample["residual_heuristic_map"].min().item()) >= 0.0
    assert tuple(sample["expanded_trace_map"].shape) == (1, h, w)
    assert float(sample["expanded_trace_map"].sum().item()) > 0.0
