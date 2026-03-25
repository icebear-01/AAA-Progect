from __future__ import annotations

import torch
from torch.utils.data import Dataset

from neural_astar.datasets import SpatialAugmentedDataset


class _WeightedDummyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        occ = torch.zeros((1, 4, 4), dtype=torch.float32)
        occ[0, int(index), int(index)] = 1.0
        return {"occ_map": occ}

    def build_sampling_weights(self, emphasized_base_indices, emphasized_weight):
        weights = torch.ones(2, dtype=torch.double)
        if 1 in set(int(idx) for idx in emphasized_base_indices):
            weights[1] = max(float(emphasized_weight), 1.0)
        return weights


def test_spatial_augment_dataset_delegates_sampling_weights_to_base_dataset():
    ds = SpatialAugmentedDataset(_WeightedDummyDataset(), mode="flip")

    weights = ds.build_sampling_weights(emphasized_base_indices=[1], emphasized_weight=5.0)

    assert weights.tolist() == [1.0, 1.0, 5.0, 5.0]
