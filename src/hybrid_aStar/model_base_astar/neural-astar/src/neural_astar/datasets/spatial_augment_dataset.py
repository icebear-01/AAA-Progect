"""Spatial augmentation wrapper for guidance datasets."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from neural_astar.utils.guidance_targets import yaw_to_bin


_ORIENTATION_KEYS = {"opt_traj_orient", "target_cost_orient"}


def _normalize_yaw(yaw: float) -> float:
    return float(math.atan2(math.sin(float(yaw)), math.cos(float(yaw))))


def _transform_yaw(yaw: float, transform_name: str) -> float:
    if transform_name == "identity":
        return _normalize_yaw(yaw)
    if transform_name == "flip_lr":
        return _normalize_yaw(math.pi - float(yaw))
    if transform_name == "rot90":
        return _normalize_yaw(float(yaw) - (math.pi / 2.0))
    if transform_name == "rot180":
        return _normalize_yaw(float(yaw) + math.pi)
    if transform_name == "rot270":
        return _normalize_yaw(float(yaw) + (math.pi / 2.0))
    raise ValueError(f"Unknown transform: {transform_name}")


def _apply_spatial_transform(tensor: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "identity":
        return tensor.clone()
    if transform_name == "flip_lr":
        return torch.flip(tensor, dims=(-1,))
    if transform_name == "rot90":
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    if transform_name == "rot180":
        return torch.rot90(tensor, k=2, dims=(-2, -1))
    if transform_name == "rot270":
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    raise ValueError(f"Unknown transform: {transform_name}")


def _orientation_channel_permutation(channels: int, transform_name: str) -> List[int]:
    if int(channels) <= 1:
        return list(range(int(channels)))
    centers = [
        -math.pi + (2.0 * math.pi) * ((float(idx) + 0.5) / float(channels))
        for idx in range(int(channels))
    ]
    return [yaw_to_bin(_transform_yaw(yaw, transform_name), int(channels)) for yaw in centers]


class SpatialAugmentedDataset(Dataset):
    """Repeat each base sample under a fixed set of spatial augmentations."""

    def __init__(self, base_dataset: Dataset, mode: str = "rot4"):
        self.base_dataset = base_dataset
        self.mode = str(mode)
        probe = self.base_dataset[0]
        occ_map = probe["occ_map"]
        if occ_map.ndim != 3:
            raise ValueError(f"Expected occ_map [C,H,W], got {tuple(occ_map.shape)}")
        self.spatial_shape = (int(occ_map.shape[-2]), int(occ_map.shape[-1]))
        self.transforms = self._resolve_transforms(self.mode, *self.spatial_shape)

    @staticmethod
    def _resolve_transforms(mode: str, height: int, width: int) -> List[str]:
        if mode == "none":
            return ["identity"]
        if mode == "flip":
            return ["identity", "flip_lr"]
        if mode == "rot4":
            if int(height) != int(width):
                raise ValueError(
                    "--train-augment-mode=rot4 requires square maps, got "
                    f"{height}x{width}"
                )
            return ["identity", "rot90", "rot180", "rot270"]
        raise ValueError(f"Unknown train augment mode: {mode}")

    def __len__(self) -> int:
        return len(self.base_dataset) * len(self.transforms)

    def base_index_for_sample(self, index: int) -> int:
        return int(index) // len(self.transforms)

    def build_sampling_weights(
        self,
        emphasized_base_indices: List[int],
        emphasized_weight: float,
    ) -> torch.Tensor:
        if hasattr(self.base_dataset, "build_sampling_weights"):
            base_weights = self.base_dataset.build_sampling_weights(
                emphasized_base_indices=emphasized_base_indices,
                emphasized_weight=emphasized_weight,
            )
            if base_weights.ndim != 1 or int(base_weights.shape[0]) != len(self.base_dataset):
                raise ValueError(
                    "base_dataset.build_sampling_weights must return [len(base_dataset)] weights, got "
                    f"{tuple(base_weights.shape)}"
                )
            return base_weights.repeat_interleave(len(self.transforms))
        weights = torch.ones(len(self), dtype=torch.double)
        boosted = set(int(idx) for idx in emphasized_base_indices)
        boost = max(float(emphasized_weight), 1.0)
        for sample_idx in range(len(self)):
            if self.base_index_for_sample(sample_idx) in boosted:
                weights[sample_idx] = boost
        return weights

    @staticmethod
    def _onehot_xy(one_hot_map: torch.Tensor) -> tuple[int, int]:
        arr = one_hot_map[0].detach().cpu().numpy()
        y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
        return int(x), int(y)

    def _transform_orientation_volume(
        self,
        tensor: torch.Tensor,
        transform_name: str,
    ) -> torch.Tensor:
        spatial = _apply_spatial_transform(tensor, transform_name)
        if spatial.ndim != 3 or spatial.shape[0] <= 1:
            return spatial
        perm = _orientation_channel_permutation(int(spatial.shape[0]), transform_name)
        out = torch.zeros_like(spatial)
        for old_idx, new_idx in enumerate(perm):
            out[new_idx] = spatial[old_idx]
        return out

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        base_idx = self.base_index_for_sample(index)
        transform_name = self.transforms[int(index) % len(self.transforms)]
        sample = self.base_dataset[base_idx]
        out: Dict[str, torch.Tensor] = {}

        for key, value in sample.items():
            if not torch.is_tensor(value):
                out[key] = value
                continue
            if value.ndim >= 2 and tuple(value.shape[-2:]) == self.spatial_shape:
                if key in _ORIENTATION_KEYS:
                    out[key] = self._transform_orientation_volume(value, transform_name)
                else:
                    out[key] = _apply_spatial_transform(value, transform_name)
            else:
                out[key] = value.clone()

        for pose_key, map_key in [("start_pose", "start_map"), ("goal_pose", "goal_map")]:
            pose = out.get(pose_key)
            onehot = out.get(map_key)
            if pose is None or onehot is None:
                continue
            x, y = self._onehot_xy(onehot)
            pose = pose.clone()
            pose[0] = float(x)
            pose[1] = float(y)
            if pose.numel() >= 3:
                pose[2] = float(_transform_yaw(float(pose[2].item()), transform_name))
            out[pose_key] = pose

        return out
