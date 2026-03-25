from __future__ import annotations

import numpy as np

from neural_astar.datasets import PlanningNPZGuidanceDataset


def _write_base_npz(path, map_designs, goal_maps, opt_policies, opt_dists):
    np.savez_compressed(
        path,
        map_designs,
        goal_maps,
        opt_policies,
        opt_dists,
        map_designs[:1],
        goal_maps[:1],
        opt_policies[:1],
        opt_dists[:1],
        map_designs[:1],
        goal_maps[:1],
        opt_policies[:1],
        opt_dists[:1],
    )


def test_planning_npz_guidance_dataset_appends_goal_replay_npz(tmp_path):
    map_designs = np.ones((2, 4, 4), dtype=np.float32)
    goal_maps = np.zeros((2, 1, 4, 4), dtype=np.float32)
    goal_maps[0, 0, 0, 0] = 1.0
    goal_maps[1, 0, 3, 3] = 1.0
    opt_policies = np.zeros((2, 8, 1, 4, 4), dtype=np.float32)
    opt_dists = np.zeros((2, 1, 4, 4), dtype=np.float32)
    base_npz = tmp_path / "base.npz"
    _write_base_npz(base_npz, map_designs, goal_maps, opt_policies, opt_dists)

    replay_goal_maps = np.zeros((2, 1, 4, 4), dtype=np.float32)
    replay_goal_maps[0, 0, 1, 2] = 1.0
    replay_goal_maps[1, 0, 2, 1] = 1.0
    replay_npz = tmp_path / "goal_replay.npz"
    np.savez_compressed(
        replay_npz,
        map_designs=np.stack([map_designs[0], map_designs[1]], axis=0),
        goal_maps=replay_goal_maps,
        opt_policies=np.stack([opt_policies[0], opt_policies[1]], axis=0),
        opt_dists=np.stack([opt_dists[0], opt_dists[1]], axis=0),
        source_indices=np.array([0, 1], dtype=np.int64),
    )

    ds = PlanningNPZGuidanceDataset(base_npz, split="train", goal_replay_npz=replay_npz)

    assert len(ds) == 4
    assert ds.source_indices.tolist() == [0, 1, 0, 1]
    assert np.allclose(ds.goal_maps[2:], replay_goal_maps)

    weights = ds.build_sampling_weights(emphasized_base_indices=[1], emphasized_weight=5.0)
    assert weights.tolist() == [1.0, 5.0, 1.0, 5.0]
