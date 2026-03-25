"""Dataset wrapper that repeats selected base indices for extra replay."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ReplayAugmentedDataset(Dataset):
    """Append repeated copies of emphasized base samples.

    Repeated samples are fetched from the original base dataset on every access.
    For datasets that resample starts/goals on-the-fly, this effectively yields
    more unique training examples concentrated on high-value maps.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        emphasized_base_indices: List[int],
        repeat_factor: int,
    ):
        if int(repeat_factor) < 0:
            raise ValueError(f"repeat_factor must be non-negative, got {repeat_factor}")
        self.base_dataset = base_dataset
        self.repeat_factor = int(repeat_factor)
        base_len = len(self.base_dataset)
        self.emphasized_base_indices = [
            int(idx) for idx in emphasized_base_indices if 0 <= int(idx) < base_len
        ]
        self.extra_indices = self.emphasized_base_indices * self.repeat_factor

    def __len__(self) -> int:
        return len(self.base_dataset) + len(self.extra_indices)

    def base_index_for_sample(self, index: int) -> int:
        base_len = len(self.base_dataset)
        if int(index) < base_len:
            return int(index)
        extra_idx = int(index) - base_len
        return int(self.extra_indices[extra_idx])

    def build_sampling_weights(
        self,
        emphasized_base_indices: List[int],
        emphasized_weight: float,
    ) -> torch.Tensor:
        weights = torch.ones(len(self), dtype=torch.double)
        boosted = set(int(idx) for idx in emphasized_base_indices)
        boost = max(float(emphasized_weight), 1.0)
        for sample_idx in range(len(self)):
            if self.base_index_for_sample(sample_idx) in boosted:
                weights[sample_idx] = boost
        return weights

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.base_dataset[self.base_index_for_sample(index)]
