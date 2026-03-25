from __future__ import annotations

import torch
from torch.utils.data import Dataset

from neural_astar.datasets import ReplayAugmentedDataset


class _DummyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {"value": torch.tensor([int(index)], dtype=torch.int64)}


def test_replay_augmented_dataset_repeats_selected_indices():
    ds = ReplayAugmentedDataset(_DummyDataset(), emphasized_base_indices=[1, 3, 9], repeat_factor=2)

    assert len(ds) == 8
    assert ds.base_index_for_sample(0) == 0
    assert ds.base_index_for_sample(3) == 3
    assert ds.base_index_for_sample(4) == 1
    assert ds.base_index_for_sample(5) == 3
    assert ds.base_index_for_sample(6) == 1
    assert ds.base_index_for_sample(7) == 3
    assert int(ds[6]["value"].item()) == 1


def test_replay_augmented_dataset_build_sampling_weights_uses_base_indices():
    ds = ReplayAugmentedDataset(_DummyDataset(), emphasized_base_indices=[2], repeat_factor=2)
    weights = ds.build_sampling_weights(emphasized_base_indices=[2], emphasized_weight=5.0)

    assert weights.tolist() == [1.0, 1.0, 5.0, 1.0, 5.0, 5.0]
