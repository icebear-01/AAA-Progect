"""Helpers for decoding and scaling residual-heuristic predictions."""

from __future__ import annotations

import numpy as np
import torch


def decode_residual_prediction_np(
    residual_pred: np.ndarray,
    transform: str,
) -> np.ndarray:
    pred = np.asarray(residual_pred, dtype=np.float32)
    if transform == "none":
        return pred.astype(np.float32)
    if transform == "log1p":
        return np.expm1(pred).astype(np.float32)
    raise ValueError(f"Unknown residual_target_transform: {transform}")


def decode_residual_prediction_torch(
    residual_pred: torch.Tensor,
    transform: str,
) -> torch.Tensor:
    if transform == "none":
        return residual_pred
    if transform == "log1p":
        return torch.expm1(residual_pred)
    raise ValueError(f"Unknown residual_target_transform: {transform}")


def apply_residual_scale_np(
    residual_pred: np.ndarray,
    scale_map: np.ndarray | None,
) -> np.ndarray:
    residual = np.asarray(residual_pred, dtype=np.float32)
    if scale_map is None:
        return residual.astype(np.float32)
    return (residual * np.asarray(scale_map, dtype=np.float32)).astype(np.float32)


def apply_residual_scale_torch(
    residual_pred: torch.Tensor,
    scale_map: torch.Tensor | None,
) -> torch.Tensor:
    if scale_map is None:
        return residual_pred
    return residual_pred * scale_map
