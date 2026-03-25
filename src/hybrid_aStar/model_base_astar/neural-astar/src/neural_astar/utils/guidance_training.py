"""Training helpers for guidance and residual-heuristic models."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import torch


def compute_linear_warmup_scale(epoch_index: int, warmup_epochs: int) -> float:
    """Linearly ramp residual-region emphasis from 0 to 1 across warmup epochs."""
    cur_epoch = int(epoch_index)
    ramp_epochs = int(warmup_epochs)
    if ramp_epochs <= 0:
        return 1.0
    if cur_epoch <= 0:
        return 0.0
    return min(float(cur_epoch) / float(ramp_epochs), 1.0)


def _expand_single_channel_map(map_1hw: torch.Tensor, channels: int, map_name: str) -> torch.Tensor:
    if map_1hw.ndim != 4 or map_1hw.shape[1] != 1:
        raise ValueError(f"{map_name} must be [B,1,H,W], got {tuple(map_1hw.shape)}")
    return map_1hw.expand(-1, int(channels), -1, -1)


def build_residual_regression_weights(
    free_mask: torch.Tensor,
    target_cost_dense: torch.Tensor,
    astar_expanded_map: torch.Tensor,
    *,
    near_path_weight: float = 0.0,
    astar_expanded_weight: float = 0.0,
) -> torch.Tensor:
    """Build per-pixel weights for residual-heuristic regression.

    The weighting emphasizes cells that matter more to A* queue ordering:
    - cells close to the expert path/corridor
    - cells frequently expanded by baseline improved A*
    """
    if target_cost_dense.ndim != 4:
        raise ValueError(
            f"target_cost_dense must be [B,C,H,W], got {tuple(target_cost_dense.shape)}"
        )

    channels = int(target_cost_dense.shape[1])
    free_mask_ch = _expand_single_channel_map(free_mask, channels, "free_mask")
    astar_expanded_ch = _expand_single_channel_map(
        astar_expanded_map,
        channels,
        "astar_expanded_map",
    )
    if target_cost_dense.shape != free_mask_ch.shape:
        raise ValueError(
            "target_cost_dense and free_mask shapes must match after channel expansion: "
            f"{tuple(target_cost_dense.shape)} vs {tuple(free_mask_ch.shape)}"
        )

    weights = free_mask_ch.clone()
    if float(near_path_weight) > 0.0:
        near_path = torch.clamp(1.0 - target_cost_dense, min=0.0, max=1.0)
        weights = weights + float(near_path_weight) * near_path * free_mask_ch
    if float(astar_expanded_weight) > 0.0:
        weights = weights + float(astar_expanded_weight) * astar_expanded_ch * free_mask_ch
    return weights.clamp_min(0.0)


def resolve_best_checkpoint_score(
    best_checkpoint_metric: str,
    *,
    val_loss: float,
    val_grid_astar_sr: Optional[float] = None,
    val_grid_astar_expanded: Optional[float] = None,
) -> Tuple[str, Tuple[float, ...]]:
    """Return the resolved metric name and a lexicographically comparable score."""
    metric = str(best_checkpoint_metric)
    if metric == "auto":
        metric = (
            "grid_astar_expanded"
            if (val_grid_astar_sr is not None and val_grid_astar_expanded is not None)
            else "val_loss"
        )

    if metric == "val_loss":
        return metric, (float(val_loss),)

    if metric == "grid_astar_expanded":
        if val_grid_astar_sr is None or val_grid_astar_expanded is None:
            raise ValueError(
                "grid_astar_expanded checkpoint selection requires both "
                "val_grid_astar_sr and val_grid_astar_expanded"
            )
        return metric, (
            -float(val_grid_astar_sr),
            float(val_grid_astar_expanded),
            float(val_loss),
        )

    raise ValueError(f"Unknown best_checkpoint_metric: {best_checkpoint_metric}")


def load_hard_case_indices(
    case_metrics_csv: str | Path,
    *,
    delta_column: str = "learned_minus_improved",
    top_fraction: float = 0.15,
    min_delta: float = 0.0,
    max_count: int = 0,
) -> List[int]:
    """Load hardest regression cases from a case-metrics CSV.

    The CSV is expected to contain dataset-local indices and a delta column where
    larger values mean the learned method regressed relative to the improved A* baseline.
    """
    csv_path = Path(case_metrics_csv)
    rows: List[Tuple[int, float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "idx" not in reader.fieldnames:
            raise ValueError(f"{csv_path} missing required column: idx")
        if delta_column not in reader.fieldnames:
            raise ValueError(f"{csv_path} missing required column: {delta_column}")
        for row in reader:
            delta = float(row[delta_column])
            if delta < float(min_delta):
                continue
            rows.append((int(row["idx"]), delta))

    if not rows:
        return []

    rows.sort(key=lambda item: item[1], reverse=True)
    if int(max_count) > 0:
        keep = min(len(rows), int(max_count))
    else:
        frac = min(max(float(top_fraction), 0.0), 1.0)
        keep = len(rows) if frac <= 0.0 else max(1, int(round(frac * len(rows))))
    return [idx for idx, _ in rows[:keep]]


def build_case_sampling_weights(
    dataset_size: int,
    *,
    emphasized_indices: List[int],
    emphasized_weight: float,
) -> torch.Tensor:
    """Build per-sample weights for weighted replay of hard cases."""
    if int(dataset_size) <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    weights = torch.ones(int(dataset_size), dtype=torch.double)
    boost = max(float(emphasized_weight), 1.0)
    for idx in emphasized_indices:
        if 0 <= int(idx) < int(dataset_size):
            weights[int(idx)] = boost
    return weights


def masked_smooth_l1_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean Smooth L1, optionally weighted per element."""
    loss_map = torch.nn.functional.smooth_l1_loss(
        pred,
        target,
        reduction="none",
    )
    if weights is None:
        return loss_map.mean()
    if loss_map.shape != weights.shape:
        raise ValueError(
            "pred/target and weights must have the same shape for masked_smooth_l1_mean: "
            f"{tuple(loss_map.shape)} vs {tuple(weights.shape)}"
        )
    return (loss_map * weights).sum() / weights.sum().clamp_min(1.0)
