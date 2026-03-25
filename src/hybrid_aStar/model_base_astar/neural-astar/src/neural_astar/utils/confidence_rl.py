"""Helpers for reinforcement-style confidence-head fine-tuning."""

from __future__ import annotations

import numpy as np


def decode_residual_prediction(
    residual_pred: np.ndarray,
    transform: str,
) -> np.ndarray:
    pred = np.asarray(residual_pred, dtype=np.float32)
    if transform == "none":
        return pred.astype(np.float32)
    if transform == "log1p":
        return np.expm1(pred).astype(np.float32)
    raise ValueError(f"Unknown residual_target_transform: {transform}")


def select_topk_confidence_cells(
    residual_map: np.ndarray,
    occ_map: np.ndarray,
    *,
    topk: int,
    min_residual: float = 0.0,
) -> np.ndarray:
    residual = np.asarray(residual_map, dtype=np.float32)
    occ = np.asarray(occ_map, dtype=np.float32)
    if residual.shape != occ.shape:
        raise ValueError(f"Shape mismatch: residual={residual.shape}, occ={occ.shape}")

    free_mask = occ < 0.5
    score = residual.copy()
    score[~free_mask] = -np.inf
    score[score < float(min_residual)] = -np.inf

    if int(topk) <= 0:
        return np.isfinite(score)

    flat = score.reshape(-1)
    finite = np.flatnonzero(np.isfinite(flat))
    if finite.size <= int(topk):
        return np.isfinite(score)

    top_idx = finite[np.argpartition(flat[finite], -int(topk))[-int(topk) :]]
    mask = np.zeros_like(flat, dtype=bool)
    mask[top_idx] = True
    return mask.reshape(score.shape)


def compute_confidence_rl_reward(
    *,
    baseline_expanded: int,
    sampled_expanded: int,
    baseline_path_length: float,
    sampled_path_length: float,
    sampled_success: bool,
    reward_scale: float,
    path_length_penalty: float,
    failure_penalty: float,
) -> float:
    reward = float(baseline_expanded) - float(sampled_expanded)
    reward -= float(path_length_penalty) * max(
        float(sampled_path_length) - float(baseline_path_length),
        0.0,
    )
    if not bool(sampled_success):
        reward -= float(failure_penalty)
    return reward / max(float(reward_scale), 1e-6)
