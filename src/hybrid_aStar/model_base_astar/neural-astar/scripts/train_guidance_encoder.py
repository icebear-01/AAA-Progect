"""Train guidance encoder for cost-map or residual-heuristic supervision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from hybrid_astar_guided.grid_astar import astar_8conn_stats
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import (
    ParkingGuidanceDataset,
    PlanningNPZGuidanceDataset,
    ReplayAugmentedDataset,
    SpatialAugmentedDataset,
)
from neural_astar.models import GuidanceEncoder
from neural_astar.utils.coords import rc_to_xy
from neural_astar.utils.guidance_losses import (
    astar_expansion_loss,
    expansion_proxy_loss,
    hybrid_expansion_loss,
)
from neural_astar.utils.guidance_training import (
    build_case_sampling_weights,
    build_residual_regression_weights,
    compute_linear_warmup_scale,
    load_hard_case_indices,
    masked_smooth_l1_mean,
    resolve_best_checkpoint_score,
)
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    apply_residual_scale_torch,
    decode_residual_prediction_np,
    decode_residual_prediction_torch,
)


def _split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    return train_ds, val_ds


def _tv_loss(cost_map: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(cost_map[:, :, 1:, :] - cost_map[:, :, :-1, :]).mean()
    dx = torch.abs(cost_map[:, :, :, 1:] - cost_map[:, :, :, :-1]).mean()
    return dx + dy


def _onehot_xy(one_hot_1hw: torch.Tensor) -> Tuple[int, int]:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return rc_to_xy(y, x, width=arr.shape[1], height=arr.shape[0])


def _extract_pose_yaws(batch: Dict[str, torch.Tensor], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    start_pose = batch.get("start_pose")
    goal_pose = batch.get("goal_pose")
    batch_size = batch["occ_map"].shape[0]

    if start_pose is None:
        start_yaw = torch.zeros(batch_size, device=device, dtype=batch["occ_map"].dtype)
    else:
        start_yaw = start_pose.to(device=device, dtype=batch["occ_map"].dtype)[:, 2]

    if goal_pose is None:
        goal_yaw = torch.zeros(batch_size, device=device, dtype=batch["occ_map"].dtype)
    else:
        goal_yaw = goal_pose.to(device=device, dtype=batch["occ_map"].dtype)[:, 2]

    return start_yaw, goal_yaw


def _extract_extra_input_maps(
    batch: Dict[str, torch.Tensor],
    model: GuidanceEncoder,
    device: str,
) -> torch.Tensor | None:
    if int(getattr(model, "extra_input_channels", 0)) <= 0:
        return None
    clearance_input = batch.get("clearance_input_map")
    if clearance_input is None:
        return None
    return clearance_input.to(device)


def _adapt_init_state_input_channels(
    model: GuidanceEncoder,
    init_state: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], list[str]]:
    adapted_state = dict(init_state)
    model_state = model.state_dict()
    adapted_keys: list[str] = []
    for key, model_tensor in model_state.items():
        ckpt_tensor = adapted_state.get(key)
        if ckpt_tensor is None or not isinstance(ckpt_tensor, torch.Tensor):
            continue
        if ckpt_tensor.ndim != 4 or model_tensor.ndim != 4:
            continue
        if ckpt_tensor.shape[0] != model_tensor.shape[0]:
            continue
        if ckpt_tensor.shape[2:] != model_tensor.shape[2:]:
            continue
        if ckpt_tensor.shape[1] >= model_tensor.shape[1]:
            continue
        padded = torch.zeros(
            model_tensor.shape,
            dtype=ckpt_tensor.dtype,
            device=ckpt_tensor.device,
        )
        padded[:, : ckpt_tensor.shape[1], :, :] = ckpt_tensor
        adapted_state[key] = padded
        adapted_keys.append(key)
    return adapted_state, adapted_keys


def _expand_like_channels(mask_1hw: torch.Tensor, channels: int) -> torch.Tensor:
    if mask_1hw.ndim != 4 or mask_1hw.shape[1] != 1:
        raise ValueError(f"mask must be [B,1,H,W], got {tuple(mask_1hw.shape)}")
    return mask_1hw.expand(-1, int(channels), -1, -1)


def _extract_supervision_tensors(
    batch: Dict[str, torch.Tensor],
    orientation_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_traj = batch["opt_traj"]
    target_cost = batch["target_cost"]
    if int(orientation_bins) <= 1:
        return target_traj, target_cost

    target_traj_orient = batch.get("opt_traj_orient")
    target_cost_orient = batch.get("target_cost_orient")
    if target_traj_orient is not None and target_cost_orient is not None:
        return target_traj_orient, target_cost_orient

    return (
        target_traj.repeat(1, int(orientation_bins), 1, 1),
        target_cost.repeat(1, int(orientation_bins), 1, 1),
    )


def _extract_residual_targets(
    batch: Dict[str, torch.Tensor],
    orientation_bins: int,
) -> torch.Tensor:
    residual = batch.get("residual_heuristic_map")
    if residual is None:
        raise KeyError("residual_heuristic_map is required for supervision-target=residual_heuristic")
    if int(orientation_bins) <= 1:
        return residual
    return residual.repeat(1, int(orientation_bins), 1, 1)


def _extract_clearance_penalty(
    batch: Dict[str, torch.Tensor],
    channels: int,
    device: str,
) -> torch.Tensor | None:
    clearance_penalty = batch.get("clearance_penalty_map")
    if clearance_penalty is None:
        return None
    clearance_penalty = clearance_penalty.to(device=device, dtype=batch["occ_map"].dtype)
    return _expand_like_channels(clearance_penalty, channels=channels)


def _transform_residual_target(
    residual_target: torch.Tensor,
    transform: str,
) -> torch.Tensor:
    if transform == "none":
        return residual_target
    if transform == "log1p":
        return torch.log1p(residual_target)
    raise ValueError(f"Unknown residual_target_transform: {transform}")


def _effective_residual_prediction_torch(
    residual_pred: torch.Tensor,
    residual_scale_map: torch.Tensor | None,
    transform: str,
) -> torch.Tensor:
    return apply_residual_scale_torch(
        decode_residual_prediction_torch(residual_pred, transform),
        residual_scale_map,
    )


def _residual_scale_target(
    pred_residual_raw: torch.Tensor,
    target_residual_raw: torch.Tensor,
    weights: torch.Tensor,
    scale_max: float,
) -> torch.Tensor:
    numer = (weights * pred_residual_raw.detach() * target_residual_raw).sum(dim=(-2, -1), keepdim=True)
    denom = (weights * pred_residual_raw.detach().pow(2)).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (numer / denom).clamp(0.0, float(scale_max))


def _confidence_target_from_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    denom = max(float(scale), 1e-6)
    return torch.exp(-torch.abs(pred.detach() - target) / denom).clamp(0.0, 1.0)


def _confidence_target_from_spike_teacher(
    pred: torch.Tensor,
    occ: torch.Tensor,
    kernel_size: int,
    strength: float,
    min_confidence: float,
) -> torch.Tensor:
    if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    residual = pred.detach().clamp_min(0.0)
    pad = int(kernel_size) // 2
    smooth = F.avg_pool2d(
        F.pad(residual, (pad, pad, pad, pad), mode="replicate"),
        kernel_size=int(kernel_size),
        stride=1,
    )
    smooth = smooth.clamp_min(1e-4)
    spike_ratio = (residual - smooth).clamp_min(0.0) / smooth
    conf = 1.0 / (1.0 + float(strength) * spike_ratio)
    conf = conf.clamp(float(min_confidence), 1.0)
    occ_mask = occ.expand(-1, conf.shape[1], -1, -1)
    return torch.where(occ_mask > 0.5, torch.zeros_like(conf), conf)


def _configure_trainable_parameters(
    model: GuidanceEncoder,
    confidence_only_finetune: bool,
) -> Tuple[int, int]:
    total_params = 0
    trainable_params = 0
    if not confidence_only_finetune:
        for param in model.parameters():
            total_params += int(param.numel())
            trainable_params += int(param.numel())
        return total_params, trainable_params

    for name, param in model.named_parameters():
        total_params += int(param.numel())
        trainable = name.startswith("conf_head.")
        param.requires_grad_(trainable)
        if trainable:
            trainable_params += int(param.numel())
    return total_params, trainable_params


def _ranking_loss(
    cost_map: torch.Tensor,
    target_traj: torch.Tensor,
    target_cost_dense: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float,
    neg_min_cost: float,
    hard_fraction: float,
) -> torch.Tensor:
    losses = []
    frac = float(min(max(hard_fraction, 0.0), 1.0))
    for bi in range(cost_map.shape[0]):
        free_mask_b = _expand_like_channels(free_mask[bi : bi + 1], channels=cost_map.shape[1])[0]
        pos_mask = (target_traj[bi] > 0.5) & (free_mask_b > 0.5)
        neg_mask = (
            (free_mask_b > 0.5)
            & (~pos_mask)
            & (target_cost_dense[bi] >= float(neg_min_cost))
        )
        pos_vals = cost_map[bi][pos_mask]
        neg_vals = cost_map[bi][neg_mask]
        if pos_vals.numel() == 0 or neg_vals.numel() == 0:
            continue

        if frac > 0.0:
            pos_k = max(1, int(round(frac * float(pos_vals.numel()))))
            neg_k = max(1, int(round(frac * float(neg_vals.numel()))))
            pos_vals = torch.topk(pos_vals, k=min(pos_k, pos_vals.numel()), largest=True).values
            neg_vals = torch.topk(neg_vals, k=min(neg_k, neg_vals.numel()), largest=False).values

        losses.append(F.relu(float(margin) + pos_vals.mean() - neg_vals.mean()))

    if not losses:
        return torch.zeros((), device=cost_map.device, dtype=cost_map.dtype)
    return torch.stack(losses).mean()


def _residual_frontier_ranking_loss(
    pred_residual_raw: torch.Tensor,
    target_residual_raw: torch.Tensor,
    astar_expanded_map: torch.Tensor,
    free_mask: torch.Tensor,
    margin: float,
    hard_fraction: float,
    min_gap: float,
) -> torch.Tensor:
    losses = []
    frac = float(min(max(hard_fraction, 0.0), 0.5))
    free_mask_ch = _expand_like_channels(free_mask, channels=pred_residual_raw.shape[1])
    frontier = _expand_like_channels(astar_expanded_map, channels=pred_residual_raw.shape[1]) > 0.5
    valid = frontier & (free_mask_ch > 0.5)

    for bi in range(pred_residual_raw.shape[0]):
        pred_vals = pred_residual_raw[bi][valid[bi]]
        target_vals = target_residual_raw[bi][valid[bi]]
        if pred_vals.numel() < 2 or target_vals.numel() < 2:
            continue
        order = torch.argsort(target_vals)
        k = max(1, int(round(float(pred_vals.numel()) * frac))) if frac > 0.0 else max(1, pred_vals.numel() // 4)
        k = min(k, pred_vals.numel() // 2 if pred_vals.numel() >= 2 else 1)
        if k <= 0:
            continue
        neg_idx = order[:k]
        pos_idx = order[-k:]
        target_gap = float(target_vals[pos_idx].mean() - target_vals[neg_idx].mean())
        if target_gap < float(min_gap):
            continue
        pos_pred = pred_vals[pos_idx]
        neg_pred = pred_vals[neg_idx]
        losses.append(F.relu(float(margin) + neg_pred.mean() - pos_pred.mean()))

    if not losses:
        return torch.zeros((), device=pred_residual_raw.device, dtype=pred_residual_raw.dtype)
    return torch.stack(losses).mean()


def _grid_astar_metrics(
    model: GuidanceEncoder,
    loader: DataLoader,
    device: str,
    lambda_guidance: float,
    max_samples: int,
    heuristic_mode: str,
    heuristic_weight: float,
    guidance_integration_mode: str,
    guidance_bonus_threshold: float,
    residual_weight: float,
) -> Tuple[float, float]:
    model.eval()
    successes = 0
    total = 0
    total_expanded = 0.0
    with torch.no_grad():
        for batch in loader:
            occ = batch["occ_map"].to(device)
            start = batch["start_map"].to(device)
            goal = batch["goal_map"].to(device)
            start_yaw, goal_yaw = _extract_pose_yaws(batch, device=device)
            extra_input_maps = _extract_extra_input_maps(batch, model, device)
            out = model(
                occ,
                start,
                goal,
                start_yaw=start_yaw,
                goal_yaw=goal_yaw,
                extra_input_maps=extra_input_maps,
            )
            cost = out.cost_map.detach().cpu().numpy()
            confidence = (
                None
                if out.confidence_map is None
                else out.confidence_map.detach().cpu().numpy()
            )
            scale = (
                None
                if out.scale_map is None
                else out.scale_map.detach().cpu().numpy()
            )

            for i in range(occ.shape[0]):
                occ_i = occ[i, 0].detach().cpu().numpy()
                start_xy = _onehot_xy(start[i])
                goal_xy = _onehot_xy(goal[i])
                residual_transform = str(getattr(model, "residual_target_transform", "none"))
                result = astar_8conn_stats(
                    occ_map=occ_i,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    heuristic_mode=heuristic_mode,
                    heuristic_weight=heuristic_weight,
                    guidance_cost=(
                        None
                        if model.output_mode == "residual_heuristic"
                        else (
                            cost[i, 0]
                            if cost.shape[1] == 1
                            else np.min(cost[i], axis=0).astype(np.float32)
                        )
                    ),
                    heuristic_residual_map=(
                        apply_residual_scale_np(
                            decode_residual_prediction_np(cost[i, 0], residual_transform),
                            (None if scale is None else scale[i, 0]),
                        )
                        if (model.output_mode == "residual_heuristic" and cost.shape[1] == 1)
                        else None
                    ),
                    residual_confidence_map=(
                        confidence[i, 0].astype(np.float32)
                        if (
                            model.output_mode == "residual_heuristic"
                            and confidence is not None
                            and confidence.shape[1] == 1
                        )
                        else None
                    ),
                    lambda_guidance=(0.0 if model.output_mode == "residual_heuristic" else lambda_guidance),
                    residual_weight=(float(residual_weight) if model.output_mode == "residual_heuristic" else 1.0),
                    guidance_integration_mode=guidance_integration_mode,
                    guidance_bonus_threshold=guidance_bonus_threshold,
                )
                successes += int(result.success)
                total_expanded += float(result.expanded_nodes)
                total += 1
                if total >= max_samples:
                    denom = max(total, 1)
                    return successes / denom, total_expanded / denom
    denom = max(total, 1)
    return successes / denom, total_expanded / denom


def _run_epoch(
    model: GuidanceEncoder,
    teacher_model: GuidanceEncoder | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    device: str,
    supervision_target: str,
    ranking_loss_weight: float,
    ranking_margin: float,
    ranking_neg_min_cost: float,
    ranking_hard_fraction: float,
    expansion_loss_weight: float,
    expansion_margin: float,
    expansion_softness: float,
    expansion_budget_multiplier: float,
    astar_expansion_loss_weight: float,
    astar_expansion_margin: float,
    hybrid_expansion_loss_weight: float,
    hybrid_expansion_margin: float,
    residual_target_transform: str,
    confidence_loss_weight: float,
    confidence_target_scale: float,
    confidence_target_mode: str,
    confidence_teacher_kernel: int,
    confidence_teacher_strength: float,
    confidence_teacher_min: float,
    residual_scale_loss_weight: float,
    residual_scale_max: float,
    residual_frontier_ranking_weight: float,
    residual_frontier_ranking_margin: float,
    residual_frontier_hard_fraction: float,
    residual_frontier_min_gap: float,
    residual_near_path_weight: float,
    residual_astar_expanded_weight: float,
    residual_weight_warmup_epochs: int,
    distill_residual_weight: float,
    distill_confidence_weight: float,
    distill_scale_weight: float,
    ignore_obstacles_in_bce: bool,
    corridor_weight: float,
    obstacle_penalty_weight: float,
    clearance_target_weight: float,
    clearance_residual_weight: float,
    clearance_penalize_corridor: bool,
    tv_weight: float,
    train: bool,
) -> Dict[str, float]:
    model.train(mode=train)
    bce_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0
    total_rank = 0.0
    total_expand = 0.0
    total_astar_expand = 0.0
    total_hybrid_expand = 0.0
    total_distill = 0.0
    total_obs = 0.0
    total_tv = 0.0
    n_batches = 0
    orientation_bins = int(model.orientation_bins)
    residual_weight_scale = compute_linear_warmup_scale(
        epoch_index=epoch_index,
        warmup_epochs=residual_weight_warmup_epochs,
    )

    for batch in loader:
        occ = batch["occ_map"].to(device)
        start = batch["start_map"].to(device)
        goal = batch["goal_map"].to(device)
        target_traj_raw, target_cost_raw = _extract_supervision_tensors(
            batch=batch,
            orientation_bins=orientation_bins,
        )
        target_traj = target_traj_raw.to(device)
        target_cost_dense = target_cost_raw.to(device)
        clearance_penalty = _extract_clearance_penalty(
            batch=batch,
            channels=target_traj.shape[1],
            device=device,
        )
        if clearance_penalty is not None and (not bool(clearance_penalize_corridor)):
            clearance_penalty = clearance_penalty * (1.0 - target_traj)
        target_cost_supervision = target_cost_dense
        if clearance_penalty is not None and float(clearance_target_weight) > 0.0:
            target_cost_supervision = torch.clamp(
                target_cost_dense + float(clearance_target_weight) * clearance_penalty,
                0.0,
                1.0,
            )
        residual_target = None
        residual_target_raw = None
        if supervision_target == "residual_heuristic":
            residual_target_raw = _extract_residual_targets(batch, orientation_bins=orientation_bins).to(device)
            if clearance_penalty is not None and float(clearance_residual_weight) > 0.0:
                residual_target_raw = residual_target_raw + float(clearance_residual_weight) * clearance_penalty
            residual_target = _transform_residual_target(
                residual_target_raw,
                transform=residual_target_transform,
            )
        astar_expanded_map = batch["astar_expanded_map"].to(device)
        expanded_trace_map = batch["expanded_trace_map"].to(device)
        start_yaw, goal_yaw = _extract_pose_yaws(batch, device=device)
        extra_input_maps = _extract_extra_input_maps(batch, model, device)
        traj_w = 1.0 + max(0.0, float(corridor_weight) - 1.0) * target_traj
        free_mask = (1.0 - occ).clamp(0.0, 1.0)
        free_mask_ch = _expand_like_channels(free_mask, channels=target_traj.shape[1])

        if train:
            optimizer.zero_grad()

        out = model(
            occ,
            start,
            goal,
            start_yaw=start_yaw,
            goal_yaw=goal_yaw,
            extra_input_maps=extra_input_maps,
        )
        teacher_out = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_out = teacher_model(
                    occ,
                    start,
                    goal,
                    start_yaw=start_yaw,
                    goal_yaw=goal_yaw,
                    extra_input_maps=extra_input_maps,
                )

        if supervision_target == "corridor_bce":
            target_cost = (1.0 - target_traj).clamp(0.0, 1.0)
            bce_map = bce_fn(out.logits_cost, target_cost)
            if ignore_obstacles_in_bce:
                w = traj_w * free_mask_ch
                primary = (bce_map * w).sum() / w.sum().clamp_min(1.0)
            else:
                w = traj_w
                primary = (bce_map * w).sum() / w.sum().clamp_min(1.0)
            bce = primary
            reg = torch.zeros((), device=device, dtype=primary.dtype)
        elif supervision_target in {"distance_field", "astar_guidance"}:
            reg_map = torch.nn.functional.smooth_l1_loss(
                out.cost_map,
                target_cost_supervision,
                reduction="none",
            )
            w = traj_w * free_mask_ch
            primary = (reg_map * w).sum() / w.sum().clamp_min(1.0)
            reg = primary
            bce = torch.zeros((), device=device, dtype=primary.dtype)
        else:
            if residual_target is None:
                raise RuntimeError("residual_target was not prepared")
            if residual_target_raw is None:
                raise RuntimeError("residual_target_raw was not prepared")
            reg_map = torch.nn.functional.smooth_l1_loss(
                out.cost_map,
                residual_target,
                reduction="none",
            )
            w = build_residual_regression_weights(
                free_mask=free_mask,
                target_cost_dense=target_cost_supervision,
                astar_expanded_map=astar_expanded_map,
                near_path_weight=float(residual_near_path_weight) * residual_weight_scale,
                astar_expanded_weight=float(residual_astar_expanded_weight) * residual_weight_scale,
            )
            primary = (reg_map * w).sum() / w.sum().clamp_min(1.0)
            pred_residual_raw = _effective_residual_prediction_torch(
                residual_pred=out.cost_map,
                residual_scale_map=out.scale_map,
                transform=residual_target_transform,
            )
            if out.scale_map is not None and float(residual_scale_loss_weight) > 0.0:
                scale_target = _residual_scale_target(
                    pred_residual_raw=decode_residual_prediction_torch(
                        out.cost_map,
                        residual_target_transform,
                    ),
                    target_residual_raw=residual_target_raw,
                    weights=w,
                    scale_max=float(residual_scale_max),
                )
                scale_term = torch.nn.functional.smooth_l1_loss(
                    out.scale_map,
                    scale_target,
                    reduction="mean",
                )
                primary = primary + float(residual_scale_loss_weight) * scale_term
            if out.confidence_map is not None and float(confidence_loss_weight) > 0.0:
                if confidence_target_mode == "error":
                    confidence_target = _confidence_target_from_error(
                        pred=out.cost_map,
                        target=residual_target,
                        scale=float(confidence_target_scale),
                    )
                elif confidence_target_mode == "spike_teacher":
                    confidence_target = _confidence_target_from_spike_teacher(
                        pred=out.cost_map,
                        occ=occ,
                        kernel_size=int(confidence_teacher_kernel),
                        strength=float(confidence_teacher_strength),
                        min_confidence=float(confidence_teacher_min),
                    )
                else:
                    raise ValueError(
                        f"Unknown confidence_target_mode: {confidence_target_mode}"
                    )
                confidence_loss_map = torch.nn.functional.smooth_l1_loss(
                    out.confidence_map,
                    confidence_target,
                    reduction="none",
                )
                confidence_term = (confidence_loss_map * w).sum() / w.sum().clamp_min(1.0)
                primary = primary + float(confidence_loss_weight) * confidence_term
            reg = primary
            bce = torch.zeros((), device=device, dtype=primary.dtype)

        distill = torch.zeros((), device=device, dtype=primary.dtype)
        if supervision_target == "residual_heuristic":
            obs_pen = (out.cost_map * occ).mean()
            if teacher_out is not None:
                if float(distill_residual_weight) > 0.0:
                    distill = distill + float(distill_residual_weight) * masked_smooth_l1_mean(
                        out.cost_map,
                        teacher_out.cost_map.detach(),
                        w,
                    )
                if (
                    float(distill_confidence_weight) > 0.0
                    and out.confidence_map is not None
                    and teacher_out.confidence_map is not None
                ):
                    distill = distill + float(distill_confidence_weight) * masked_smooth_l1_mean(
                        out.confidence_map,
                        teacher_out.confidence_map.detach(),
                        w,
                    )
                if (
                    float(distill_scale_weight) > 0.0
                    and out.scale_map is not None
                    and teacher_out.scale_map is not None
                ):
                    distill = distill + float(distill_scale_weight) * masked_smooth_l1_mean(
                        out.scale_map,
                        teacher_out.scale_map.detach(),
                    )
        else:
            obs_pen = ((1.0 - out.cost_map) * occ).mean()
        tv = _tv_loss(out.cost_map)
        if supervision_target == "residual_heuristic":
            rank = _residual_frontier_ranking_loss(
                pred_residual_raw=pred_residual_raw,
                target_residual_raw=residual_target_raw,
                astar_expanded_map=astar_expanded_map,
                free_mask=free_mask,
                margin=residual_frontier_ranking_margin,
                hard_fraction=residual_frontier_hard_fraction,
                min_gap=residual_frontier_min_gap,
            )
            expand = torch.zeros((), device=device, dtype=primary.dtype)
            astar_expand = torch.zeros((), device=device, dtype=primary.dtype)
            hybrid_expand = torch.zeros((), device=device, dtype=primary.dtype)
        else:
            rank = _ranking_loss(
                cost_map=out.cost_map,
                target_traj=target_traj,
                target_cost_dense=target_cost_supervision,
                free_mask=free_mask,
                margin=ranking_margin,
                neg_min_cost=ranking_neg_min_cost,
                hard_fraction=ranking_hard_fraction,
            )
            expand = expansion_proxy_loss(
                cost_map=out.cost_map,
                target_traj=target_traj,
                free_mask=free_mask,
                margin=expansion_margin,
                softness=expansion_softness,
                budget_multiplier=expansion_budget_multiplier,
            )
            astar_expand = astar_expansion_loss(
                cost_map=out.cost_map,
                target_traj=target_traj,
                astar_expanded_map=astar_expanded_map,
                free_mask=free_mask,
                margin=astar_expansion_margin,
            )
            hybrid_expand = hybrid_expansion_loss(
                cost_map=out.cost_map,
                target_traj=target_traj,
                expanded_trace_map=expanded_trace_map,
                free_mask=free_mask,
                margin=hybrid_expansion_margin,
            )
        loss = (
            primary
            + (
                float(residual_frontier_ranking_weight) * rank
                if supervision_target == "residual_heuristic"
                else float(ranking_loss_weight) * rank
            )
            + float(expansion_loss_weight) * expand
            + float(astar_expansion_loss_weight) * astar_expand
            + float(hybrid_expansion_loss_weight) * hybrid_expand
            + distill
            + obstacle_penalty_weight * obs_pen
            + tv_weight * tv
        )

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce.item())
        total_reg += float(reg.item())
        total_rank += float(rank.item())
        total_expand += float(expand.item())
        total_astar_expand += float(astar_expand.item())
        total_hybrid_expand += float(hybrid_expand.item())
        total_distill += float(distill.item())
        total_obs += float(obs_pen.item())
        total_tv += float(tv.item())
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "reg": total_reg / denom,
        "rank": total_rank / denom,
        "expand": total_expand / denom,
        "astar_expand": total_astar_expand / denom,
        "hybrid_expand": total_hybrid_expand / denom,
        "distill": total_distill / denom,
        "obs_pen": total_obs / denom,
        "tv": total_tv / denom,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train guidance encoder")
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--val-dir", type=Path, default=None)
    p.add_argument(
        "--train-npz",
        type=Path,
        default=None,
        help="planning-datasets .npz path (arr_0..arr_11). If set, train/val are loaded from npz splits.",
    )
    p.add_argument(
        "--val-npz",
        type=Path,
        default=None,
        help="Optional separate planning-datasets .npz for validation.",
    )
    p.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Split used from --train-npz.",
    )
    p.add_argument(
        "--val-split",
        type=str,
        default="valid",
        choices=["train", "valid", "test"],
        help="Split used from --val-npz/--train-npz for validation.",
    )
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "adamw"],
        help="Optimizer used for training.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay used by the optimizer.",
    )
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="Learning-rate schedule used across epochs.",
    )
    p.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm the learning rate during the first N epochs.",
    )
    p.add_argument(
        "--lr-warmup-start-factor",
        type=float,
        default=0.1,
        help="Initial LR factor for linear warmup; 0.1 means start at 10% of --lr.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--train-augment-mode",
        type=str,
        default="none",
        choices=["none", "flip", "rot4"],
        help="Spatial augmentation applied to the training split only.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument(
        "--orientation-bins",
        type=int,
        default=1,
        help="Number of guidance yaw bins. 1 keeps the original single-channel cost map.",
    )
    p.add_argument(
        "--arch",
        type=str,
        default="unet",
        choices=["unet", "unet_transformer", "unet_transformer_v2", "unet_transformer_v3", "legacy_cnn"],
        help="Guidance encoder architecture.",
    )
    p.add_argument(
        "--transformer-depth",
        type=int,
        default=2,
        help="Number of transformer blocks when using a transformer-based arch.",
    )
    p.add_argument(
        "--transformer-heads",
        type=int,
        default=8,
        help="Number of attention heads when using a transformer-based arch.",
    )
    p.add_argument(
        "--transformer-mlp-ratio",
        type=float,
        default=4.0,
        help="MLP expansion ratio inside transformer blocks.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("model/guidance_encoder"))
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize model weights before training.",
    )
    p.add_argument(
        "--resume-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint used to resume training exactly, including model, optimizer, "
            "scheduler, and starting epoch."
        ),
    )
    p.add_argument(
        "--distill-teacher-ckpt",
        type=Path,
        default=None,
        help="Optional teacher checkpoint used for output-space distillation.",
    )
    p.add_argument(
        "--distill-residual-weight",
        type=float,
        default=0.0,
        help="Weight of teacher residual-map distillation loss.",
    )
    p.add_argument(
        "--distill-confidence-weight",
        type=float,
        default=0.0,
        help="Weight of teacher confidence-map distillation loss.",
    )
    p.add_argument(
        "--distill-scale-weight",
        type=float,
        default=0.0,
        help="Weight of teacher residual-scale distillation loss.",
    )
    p.add_argument(
        "--predict-confidence",
        action="store_true",
        help="Add a learned variance/confidence head for residual-heuristic supervision.",
    )
    p.add_argument(
        "--confidence-loss-weight",
        type=float,
        default=0.25,
        help="Auxiliary weight for supervising the learned confidence head.",
    )
    p.add_argument(
        "--confidence-target-scale",
        type=float,
        default=0.25,
        help="Residual error scale used to build confidence targets exp(-|e|/scale).",
    )
    p.add_argument(
        "--predict-residual-scale",
        action="store_true",
        help="Predict a learned global residual scale alpha(map,start,goal).",
    )
    p.add_argument(
        "--residual-scale-max",
        type=float,
        default=2.0,
        help="Upper bound for the learned residual scale head.",
    )
    p.add_argument(
        "--residual-scale-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for supervising the learned residual scale head.",
    )
    p.add_argument(
        "--confidence-target-mode",
        type=str,
        default="error",
        choices=["error", "spike_teacher"],
        help="Supervision target used for the confidence head.",
    )
    p.add_argument(
        "--confidence-head-kernel",
        type=int,
        default=1,
        help="Odd kernel size for the confidence head. 1 keeps the legacy 1x1 head.",
    )
    p.add_argument(
        "--confidence-teacher-kernel",
        type=int,
        default=5,
        help="Odd kernel size used when --confidence-target-mode=spike_teacher.",
    )
    p.add_argument(
        "--confidence-teacher-strength",
        type=float,
        default=0.5,
        help="Spike suppression strength used for spike-teacher confidence labels.",
    )
    p.add_argument(
        "--confidence-teacher-min",
        type=float,
        default=0.25,
        help="Minimum confidence used for spike-teacher confidence labels.",
    )
    p.add_argument(
        "--confidence-only-finetune",
        action="store_true",
        help="Freeze the pretrained residual network and train only conf_head parameters.",
    )
    p.add_argument(
        "--disable-pose-yaw-cond",
        action="store_true",
        help="Disable start/goal yaw conditioning channels.",
    )

    p.add_argument("--ignore-obstacles-in-bce", action="store_true")
    p.add_argument(
        "--supervision-target",
        type=str,
        default="corridor_bce",
        choices=["corridor_bce", "distance_field", "astar_guidance", "residual_heuristic"],
        help="Primary supervision target for guidance learning.",
    )
    p.add_argument(
        "--corridor-weight",
        type=float,
        default=4.0,
        help="Pixel weight for expert corridor in the primary loss.",
    )
    p.add_argument("--obstacle-penalty-weight", type=float, default=0.0)
    p.add_argument(
        "--clearance-safe-distance",
        type=float,
        default=0.0,
        help="Obstacle clearance radius in grid cells used to build a soft proximity penalty map.",
    )
    p.add_argument(
        "--clearance-power",
        type=float,
        default=2.0,
        help="Power used by the obstacle-clearance penalty profile.",
    )
    p.add_argument(
        "--clearance-target-weight",
        type=float,
        default=0.0,
        help="Weight used to add obstacle-clearance penalty into dense target-cost supervision.",
    )
    p.add_argument(
        "--clearance-residual-weight",
        type=float,
        default=0.0,
        help="Weight used to add obstacle-clearance penalty into residual-heuristic targets.",
    )
    p.add_argument(
        "--clearance-penalize-corridor",
        action="store_true",
        help="Also penalize expert-corridor cells that lie close to obstacles.",
    )
    p.add_argument(
        "--clearance-input-clip-distance",
        type=float,
        default=0.0,
        help="If >0, append a clipped obstacle-distance input channel to the model.",
    )
    p.add_argument("--tv-weight", type=float, default=0.0)
    p.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=0.0,
        help="Weight of pairwise ranking loss between expert corridor and off-corridor cells.",
    )
    p.add_argument(
        "--ranking-margin",
        type=float,
        default=0.05,
        help="Margin used in ranking loss: expert_cost + margin <= negative_cost.",
    )
    p.add_argument(
        "--ranking-neg-min-cost",
        type=float,
        default=0.25,
        help="Minimum dense target cost used to define ranking negatives.",
    )
    p.add_argument(
        "--ranking-hard-fraction",
        type=float,
        default=0.1,
        help="Fraction of hardest positives/negatives used in ranking loss; 0 means all.",
    )
    p.add_argument(
        "--expansion-loss-weight",
        type=float,
        default=0.0,
        help="Weight of search-effort proxy loss that penalizes broad low-cost spillover.",
    )
    p.add_argument(
        "--expansion-margin",
        type=float,
        default=0.05,
        help="Margin above corridor cost used to define attractive cells in the proxy loss.",
    )
    p.add_argument(
        "--expansion-softness",
        type=float,
        default=0.05,
        help="Sigmoid softness used in the expansion proxy loss.",
    )
    p.add_argument(
        "--expansion-budget-multiplier",
        type=float,
        default=1.5,
        help="Allowed low-cost free-space mass relative to expert corridor mass.",
    )
    p.add_argument(
        "--astar-expansion-loss-weight",
        type=float,
        default=0.0,
        help="Weight of 2D A* expansion-map auxiliary loss.",
    )
    p.add_argument(
        "--astar-expansion-margin",
        type=float,
        default=0.05,
        help="Margin above corridor cost enforced on high-visitation 2D A* cells.",
    )
    p.add_argument(
        "--hybrid-expansion-loss-weight",
        type=float,
        default=0.0,
        help="Weight of Hybrid A* expansion-trace auxiliary loss.",
    )
    p.add_argument(
        "--hybrid-expansion-margin",
        type=float,
        default=0.05,
        help="Margin above corridor cost enforced on high-visitation expanded cells.",
    )

    p.add_argument("--compute-grid-astar-metric", action="store_true")
    p.add_argument("--astar-lambda", type=float, default=1.0)
    p.add_argument("--astar-max-samples", type=int, default=64)
    p.add_argument(
        "--astar-heuristic-mode",
        type=str,
        default="octile",
        choices=["euclidean", "manhattan", "chebyshev", "octile"],
    )
    p.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    p.add_argument("--astar-residual-weight", type=float, default=1.0)
    p.add_argument(
        "--residual-target-transform",
        type=str,
        default="log1p",
        choices=["none", "log1p"],
        help="Transform applied to residual heuristic targets during training.",
    )
    p.add_argument(
        "--residual-near-path-weight",
        type=float,
        default=0.25,
        help="Extra residual-regression weight on cells near the expert path.",
    )
    p.add_argument(
        "--residual-astar-expanded-weight",
        type=float,
        default=0.75,
        help="Extra residual-regression weight on cells frequently expanded by improved A*.",
    )
    p.add_argument(
        "--residual-weight-warmup-epochs",
        type=int,
        default=5,
        help="Linearly ramp residual-region emphasis during the first N epochs.",
    )
    p.add_argument(
        "--residual-frontier-ranking-weight",
        type=float,
        default=0.0,
        help="Weight of residual frontier ranking loss on baseline-expanded cells.",
    )
    p.add_argument(
        "--residual-frontier-ranking-margin",
        type=float,
        default=0.05,
        help="Margin used in residual frontier ranking loss.",
    )
    p.add_argument(
        "--residual-frontier-hard-fraction",
        type=float,
        default=0.15,
        help="Fraction of lowest/highest target-residual frontier cells used in ranking.",
    )
    p.add_argument(
        "--residual-frontier-min-gap",
        type=float,
        default=0.1,
        help="Minimum target residual gap required to form a frontier ranking pair.",
    )
    p.add_argument(
        "--best-checkpoint-metric",
        type=str,
        default="auto",
        choices=["auto", "val_loss", "grid_astar_expanded"],
        help="Metric used to choose best.pt. auto prefers grid-A* expanded nodes when available.",
    )
    p.add_argument(
        "--astar-guidance-integration-mode",
        type=str,
        default="heuristic_bonus",
        choices=["g_cost", "heuristic_bias", "heuristic_bonus"],
    )
    p.add_argument("--astar-guidance-bonus-threshold", type=float, default=0.6)
    p.add_argument(
        "--hard-case-csv",
        type=Path,
        default=None,
        help="Optional case_metrics.csv used to oversample hard regression cases.",
    )
    p.add_argument(
        "--hard-case-delta-column",
        type=str,
        default="learned_minus_improved",
        help="CSV column used to score hard cases when --hard-case-csv is set.",
    )
    p.add_argument(
        "--hard-case-top-fraction",
        type=float,
        default=0.15,
        help="Fraction of highest-regression cases replayed when --hard-case-csv is set.",
    )
    p.add_argument(
        "--hard-case-min-delta",
        type=float,
        default=0.0,
        help="Minimum learned-minus-improved delta required to mark a case as hard.",
    )
    p.add_argument(
        "--hard-case-max-count",
        type=int,
        default=0,
        help="Optional cap on the number of replayed hard cases. 0 keeps the top fraction.",
    )
    p.add_argument(
        "--hard-case-boost-weight",
        type=float,
        default=4.0,
        help="Sampling weight assigned to replayed hard cases.",
    )
    p.add_argument(
        "--hard-case-repeat-factor",
        type=int,
        default=0,
        help="Append each selected hard case this many extra times before augmentation.",
    )
    p.add_argument(
        "--train-goal-replay-npz",
        type=Path,
        default=None,
        help="Optional NPZ of extra train-time goal/policy/dist tuples appended to the train split.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume_ckpt is not None and args.init_ckpt is not None:
        raise ValueError("--resume-ckpt and --init-ckpt are mutually exclusive")

    if args.best_checkpoint_metric == "grid_astar_expanded" and (not args.compute_grid_astar_metric):
        raise ValueError(
            "--best-checkpoint-metric=grid_astar_expanded requires --compute-grid-astar-metric"
        )
    if args.residual_near_path_weight < 0.0:
        raise ValueError(
            f"--residual-near-path-weight must be non-negative, got {args.residual_near_path_weight}"
        )
    if args.residual_astar_expanded_weight < 0.0:
        raise ValueError(
            "--residual-astar-expanded-weight must be non-negative, got "
            f"{args.residual_astar_expanded_weight}"
        )
    if args.residual_weight_warmup_epochs < 0:
        raise ValueError(
            "--residual-weight-warmup-epochs must be non-negative, got "
            f"{args.residual_weight_warmup_epochs}"
        )
    if args.min_lr < 0.0:
        raise ValueError(f"--min-lr must be non-negative, got {args.min_lr}")
    if args.weight_decay < 0.0:
        raise ValueError(f"--weight-decay must be non-negative, got {args.weight_decay}")
    if args.lr_warmup_epochs < 0:
        raise ValueError(
            f"--lr-warmup-epochs must be non-negative, got {args.lr_warmup_epochs}"
        )
    if args.lr_warmup_start_factor <= 0.0 or args.lr_warmup_start_factor > 1.0:
        raise ValueError(
            "--lr-warmup-start-factor must be in (0, 1], got "
            f"{args.lr_warmup_start_factor}"
        )
    if args.hard_case_top_fraction < 0.0 or args.hard_case_top_fraction > 1.0:
        raise ValueError(
            "--hard-case-top-fraction must be in [0, 1], got "
            f"{args.hard_case_top_fraction}"
        )
    if args.hard_case_max_count < 0:
        raise ValueError(
            f"--hard-case-max-count must be non-negative, got {args.hard_case_max_count}"
        )
    if args.hard_case_repeat_factor < 0:
        raise ValueError(
            f"--hard-case-repeat-factor must be non-negative, got {args.hard_case_repeat_factor}"
        )
    if args.hard_case_boost_weight < 1.0:
        raise ValueError(
            "--hard-case-boost-weight must be >= 1.0, got "
            f"{args.hard_case_boost_weight}"
        )
    if args.predict_confidence and args.supervision_target != "residual_heuristic":
        raise ValueError("--predict-confidence currently requires --supervision-target residual_heuristic")
    if args.confidence_loss_weight < 0.0:
        raise ValueError(f"--confidence-loss-weight must be non-negative, got {args.confidence_loss_weight}")
    if args.confidence_target_scale <= 0.0:
        raise ValueError(
            f"--confidence-target-scale must be positive, got {args.confidence_target_scale}"
        )
    if args.residual_scale_max <= 0.0:
        raise ValueError(f"--residual-scale-max must be positive, got {args.residual_scale_max}")
    if args.residual_scale_loss_weight < 0.0:
        raise ValueError(
            f"--residual-scale-loss-weight must be non-negative, got {args.residual_scale_loss_weight}"
        )
    if args.confidence_head_kernel <= 0 or args.confidence_head_kernel % 2 == 0:
        raise ValueError(
            f"--confidence-head-kernel must be a positive odd integer, got {args.confidence_head_kernel}"
        )
    if args.confidence_teacher_kernel <= 0 or args.confidence_teacher_kernel % 2 == 0:
        raise ValueError(
            "--confidence-teacher-kernel must be a positive odd integer, got "
            f"{args.confidence_teacher_kernel}"
        )
    if args.confidence_teacher_strength < 0.0:
        raise ValueError(
            f"--confidence-teacher-strength must be non-negative, got {args.confidence_teacher_strength}"
        )
    if args.confidence_teacher_min < 0.0 or args.confidence_teacher_min > 1.0:
        raise ValueError(
            f"--confidence-teacher-min must be in [0, 1], got {args.confidence_teacher_min}"
        )
    if args.confidence_only_finetune and not args.predict_confidence:
        raise ValueError("--confidence-only-finetune requires --predict-confidence")
    if args.confidence_only_finetune and args.init_ckpt is None:
        raise ValueError("--confidence-only-finetune requires --init-ckpt")
    if args.predict_residual_scale and args.supervision_target != "residual_heuristic":
        raise ValueError("--predict-residual-scale requires --supervision-target residual_heuristic")
    if args.residual_frontier_ranking_weight < 0.0:
        raise ValueError(
            "--residual-frontier-ranking-weight must be non-negative, got "
            f"{args.residual_frontier_ranking_weight}"
        )
    if args.residual_frontier_hard_fraction < 0.0 or args.residual_frontier_hard_fraction > 0.5:
        raise ValueError(
            "--residual-frontier-hard-fraction must be in [0, 0.5], got "
            f"{args.residual_frontier_hard_fraction}"
        )
    if args.residual_frontier_min_gap < 0.0:
        raise ValueError(
            f"--residual-frontier-min-gap must be non-negative, got {args.residual_frontier_min_gap}"
        )
    if args.distill_residual_weight < 0.0:
        raise ValueError(
            f"--distill-residual-weight must be non-negative, got {args.distill_residual_weight}"
        )
    if args.distill_confidence_weight < 0.0:
        raise ValueError(
            "--distill-confidence-weight must be non-negative, got "
            f"{args.distill_confidence_weight}"
        )
    if args.distill_scale_weight < 0.0:
        raise ValueError(
            f"--distill-scale-weight must be non-negative, got {args.distill_scale_weight}"
        )
    if (
        args.distill_residual_weight > 0.0
        or args.distill_confidence_weight > 0.0
        or args.distill_scale_weight > 0.0
    ) and args.distill_teacher_ckpt is None:
        raise ValueError(
            "Teacher distillation weights require --distill-teacher-ckpt"
        )
    if (
        args.distill_teacher_ckpt is not None
        and args.supervision_target != "residual_heuristic"
    ):
        raise ValueError(
            "--distill-teacher-ckpt currently requires --supervision-target residual_heuristic"
        )
    if args.arch not in {"unet_transformer", "unet_transformer_v2", "unet_transformer_v3"} and (
        args.transformer_depth != 2
        or args.transformer_heads != 8
        or abs(float(args.transformer_mlp_ratio) - 4.0) > 1e-8
    ):
        print(
            "warning: transformer-specific args are set but --arch is not transformer-based; "
            "they will be ignored."
        )
    if args.arch in {"unet_transformer", "unet_transformer_v2", "unet_transformer_v3"} and args.transformer_depth <= 0:
        raise ValueError(
            f"--transformer-depth must be positive for transformer-based arch, got {args.transformer_depth}"
        )
    if args.arch in {"unet_transformer", "unet_transformer_v2", "unet_transformer_v3"} and args.transformer_heads <= 0:
        raise ValueError(
            f"--transformer-heads must be positive for transformer-based arch, got {args.transformer_heads}"
        )
    if args.arch in {"unet_transformer", "unet_transformer_v2", "unet_transformer_v3"} and args.transformer_mlp_ratio <= 0.0:
        raise ValueError(
            "--transformer-mlp-ratio must be positive for transformer-based arch, got "
            f"{args.transformer_mlp_ratio}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_npz is not None:
        if args.train_dir is not None:
            raise ValueError("Use either --train-dir or --train-npz, not both.")
        train_dataset = PlanningNPZGuidanceDataset(
            npz_path=args.train_npz,
            split=args.train_split,
            seed=args.seed,
            orientation_bins=args.orientation_bins,
            goal_replay_npz=args.train_goal_replay_npz,
            clearance_safe_distance=args.clearance_safe_distance,
            clearance_power=args.clearance_power,
            clearance_target_weight=args.clearance_target_weight,
            clearance_penalize_corridor=args.clearance_penalize_corridor,
            clearance_input_clip_distance=args.clearance_input_clip_distance,
        )
        val_dataset = PlanningNPZGuidanceDataset(
            npz_path=(args.val_npz if args.val_npz is not None else args.train_npz),
            split=args.val_split,
            seed=args.seed + 1,
            orientation_bins=args.orientation_bins,
            clearance_safe_distance=args.clearance_safe_distance,
            clearance_power=args.clearance_power,
            clearance_target_weight=args.clearance_target_weight,
            clearance_penalize_corridor=args.clearance_penalize_corridor,
            clearance_input_clip_distance=args.clearance_input_clip_distance,
        )
    else:
        if args.train_dir is None:
            raise ValueError("Either --train-dir or --train-npz must be provided.")
        train_dataset_full = ParkingGuidanceDataset(
            args.train_dir,
            orientation_bins=args.orientation_bins,
            clearance_safe_distance=args.clearance_safe_distance,
            clearance_power=args.clearance_power,
            clearance_target_weight=args.clearance_target_weight,
            clearance_penalize_corridor=args.clearance_penalize_corridor,
            clearance_input_clip_distance=args.clearance_input_clip_distance,
        )
        if args.val_dir is not None:
            train_dataset = train_dataset_full
            val_dataset = ParkingGuidanceDataset(
                args.val_dir,
                orientation_bins=args.orientation_bins,
                clearance_safe_distance=args.clearance_safe_distance,
                clearance_power=args.clearance_power,
                clearance_target_weight=args.clearance_target_weight,
                clearance_penalize_corridor=args.clearance_penalize_corridor,
                clearance_input_clip_distance=args.clearance_input_clip_distance,
            )
        else:
            train_dataset, val_dataset = _split_dataset(
                train_dataset_full, val_ratio=args.val_ratio, seed=args.seed
            )

    hard_case_indices = []
    if args.hard_case_csv is not None:
        hard_case_indices = load_hard_case_indices(
            args.hard_case_csv,
            delta_column=args.hard_case_delta_column,
            top_fraction=args.hard_case_top_fraction,
            min_delta=args.hard_case_min_delta,
            max_count=args.hard_case_max_count,
        )
    base_train_size = len(train_dataset)
    if hard_case_indices and int(args.hard_case_repeat_factor) > 0:
        train_dataset = ReplayAugmentedDataset(
            train_dataset,
            emphasized_base_indices=hard_case_indices,
            repeat_factor=int(args.hard_case_repeat_factor),
        )
    if args.train_augment_mode != "none":
        train_dataset = SpatialAugmentedDataset(
            train_dataset,
            mode=args.train_augment_mode,
        )

    train_sampler = None
    if hard_case_indices:
        if hasattr(train_dataset, "build_sampling_weights"):
            train_weights = train_dataset.build_sampling_weights(
                emphasized_base_indices=hard_case_indices,
                emphasized_weight=args.hard_case_boost_weight,
            )
        else:
            train_weights = build_case_sampling_weights(
                len(train_dataset),
                emphasized_indices=hard_case_indices,
                emphasized_weight=args.hard_case_boost_weight,
            )
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    use_pose_yaw_cond = not bool(args.disable_pose_yaw_cond)
    astar_expansion_loss_weight = float(args.astar_expansion_loss_weight)
    if args.supervision_target == "astar_guidance" and astar_expansion_loss_weight <= 0.0:
        astar_expansion_loss_weight = 1.0
    output_mode = "residual_heuristic" if args.supervision_target == "residual_heuristic" else "cost_map"
    model = GuidanceEncoder(
        base_channels=args.base_channels,
        arch=args.arch,
        use_pose_yaw_cond=use_pose_yaw_cond,
        orientation_bins=args.orientation_bins,
        output_mode=output_mode,
        residual_target_transform=(
            str(args.residual_target_transform)
            if output_mode == "residual_heuristic"
            else "none"
        ),
        predict_confidence=bool(args.predict_confidence),
        confidence_head_kernel=int(args.confidence_head_kernel),
        predict_residual_scale=bool(args.predict_residual_scale),
        residual_scale_max=float(args.residual_scale_max),
        transformer_depth=int(args.transformer_depth),
        transformer_heads=int(args.transformer_heads),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
        extra_input_channels=(1 if float(args.clearance_input_clip_distance) > 0.0 else 0),
        clearance_input_clip_distance=float(args.clearance_input_clip_distance),
    ).to(device)
    teacher_model = None
    if args.distill_teacher_ckpt is not None:
        teacher_model = load_guidance_encoder(args.distill_teacher_ckpt, device=device)
        if getattr(teacher_model, "output_mode", "cost_map") != "residual_heuristic":
            raise ValueError("Teacher checkpoint must be a residual-heuristic model.")
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
    if args.init_ckpt is not None:
        init_payload = torch.load(args.init_ckpt, map_location=device)
        init_state = init_payload.get("model_state_dict", init_payload)
        init_state, adapted_input_keys = _adapt_init_state_input_channels(model, init_state)
        allow_partial_init = (
            bool(args.predict_confidence)
            or bool(args.predict_residual_scale)
            or float(args.clearance_input_clip_distance) > 0.0
            or bool(adapted_input_keys)
        )
        load_res = model.load_state_dict(init_state, strict=not allow_partial_init)
        print(f"initialized_from={args.init_ckpt}")
        if adapted_input_keys:
            print(f"adapted_input_channel_keys={adapted_input_keys}")
        if allow_partial_init:
            missing = getattr(load_res, "missing_keys", [])
            unexpected = getattr(load_res, "unexpected_keys", [])
            if missing or unexpected:
                print(f"partial_init missing_keys={list(missing)} unexpected_keys={list(unexpected)}")
    total_params, trainable_params = _configure_trainable_parameters(
        model=model,
        confidence_only_finetune=bool(args.confidence_only_finetune),
    )
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_parameters,
        lr=args.lr,
        weight_decay=float(args.weight_decay),
    )
    scheduler = None
    warmup_epochs = int(args.lr_warmup_epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(args.lr_warmup_start_factor),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        if args.lr_scheduler == "cosine" and int(args.epochs) > warmup_epochs:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(args.epochs) - warmup_epochs),
                eta_min=float(args.min_lr),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = warmup
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=float(args.min_lr),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.output_dir / "best.pt"
    last_ckpt = args.output_dir / "last.pt"

    best_score = None
    best_metric_name = "val_loss"
    best_metric_snapshot = {
        "val_loss": float("inf"),
        "val_grid_astar_sr": None,
        "val_grid_astar_expanded": None,
    }
    start_epoch = 1

    if best_ckpt.exists():
        best_payload = torch.load(best_ckpt, map_location=device)
        best_metric_name = str(best_payload.get("resolved_best_checkpoint_metric", "val_loss"))
        _, best_score = resolve_best_checkpoint_score(
            best_metric_name,
            val_loss=best_payload.get("val_loss"),
            val_grid_astar_sr=best_payload.get("val_grid_astar_sr"),
            val_grid_astar_expanded=best_payload.get("val_grid_astar_expanded"),
        )
        best_metric_snapshot = {
            "val_loss": float(best_payload.get("val_loss", float("inf"))),
            "val_grid_astar_sr": best_payload.get("val_grid_astar_sr"),
            "val_grid_astar_expanded": best_payload.get("val_grid_astar_expanded"),
        }

    if args.resume_ckpt is not None:
        resume_payload = torch.load(args.resume_ckpt, map_location=device)
        resume_state = resume_payload.get("model_state_dict", resume_payload)
        resume_state, adapted_input_keys = _adapt_init_state_input_channels(model, resume_state)
        load_res = model.load_state_dict(resume_state, strict=(not bool(adapted_input_keys)))
        if adapted_input_keys:
            print(f"resumed_with_adapted_input_channel_keys={adapted_input_keys}")
        if adapted_input_keys:
            missing = getattr(load_res, "missing_keys", [])
            unexpected = getattr(load_res, "unexpected_keys", [])
            if missing or unexpected:
                print(f"partial_resume missing_keys={list(missing)} unexpected_keys={list(unexpected)}")
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = resume_payload.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        if best_score is None:
            resume_metric_name = str(
                resume_payload.get("resolved_best_checkpoint_metric", args.best_checkpoint_metric)
            )
            best_metric_name = resume_metric_name
            _, best_score = resolve_best_checkpoint_score(
                resume_metric_name,
                val_loss=resume_payload.get("val_loss"),
                val_grid_astar_sr=resume_payload.get("val_grid_astar_sr"),
                val_grid_astar_expanded=resume_payload.get("val_grid_astar_expanded"),
            )
            best_metric_snapshot = {
                "val_loss": float(resume_payload.get("val_loss", float("inf"))),
                "val_grid_astar_sr": resume_payload.get("val_grid_astar_sr"),
                "val_grid_astar_expanded": resume_payload.get("val_grid_astar_expanded"),
            }
        print(f"resumed_from={args.resume_ckpt}")
        print(f"resume_start_epoch={start_epoch}")

    print(
        f"train_size={len(train_dataset)} val_size={len(val_dataset)} "
        f"(base_train_size={base_train_size}, augment={args.train_augment_mode})"
    )
    print(
        f"trainable_params={trainable_params} total_params={total_params} "
        f"(confidence_only_finetune={bool(args.confidence_only_finetune)})"
    )
    if hard_case_indices:
        print(
            "hard_case_replay="
            f"{len(hard_case_indices)} cases "
            f"(boost_weight={args.hard_case_boost_weight:.2f}, "
            f"top_fraction={args.hard_case_top_fraction:.3f}, "
            f"min_delta={args.hard_case_min_delta:.3f})"
        )
    if teacher_model is not None:
        print(
            "distill_teacher="
            f"{args.distill_teacher_ckpt} "
            f"(residual={args.distill_residual_weight:.3f}, "
            f"confidence={args.distill_confidence_weight:.3f}, "
            f"scale={args.distill_scale_weight:.3f})"
        )

    if start_epoch > args.epochs:
        print(
            f"resume_start_epoch={start_epoch} exceeds --epochs={args.epochs}; "
            "nothing to do."
        )
        print(f"best_ckpt={best_ckpt}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            teacher_model=teacher_model,
            loader=train_loader,
            optimizer=optimizer,
            epoch_index=epoch,
            device=device,
            supervision_target=args.supervision_target,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_neg_min_cost=args.ranking_neg_min_cost,
            ranking_hard_fraction=args.ranking_hard_fraction,
            expansion_loss_weight=args.expansion_loss_weight,
            expansion_margin=args.expansion_margin,
            expansion_softness=args.expansion_softness,
            expansion_budget_multiplier=args.expansion_budget_multiplier,
            astar_expansion_loss_weight=astar_expansion_loss_weight,
            astar_expansion_margin=args.astar_expansion_margin,
            hybrid_expansion_loss_weight=args.hybrid_expansion_loss_weight,
            hybrid_expansion_margin=args.hybrid_expansion_margin,
            residual_target_transform=args.residual_target_transform,
            confidence_loss_weight=args.confidence_loss_weight,
            confidence_target_scale=args.confidence_target_scale,
            confidence_target_mode=args.confidence_target_mode,
            confidence_teacher_kernel=args.confidence_teacher_kernel,
            confidence_teacher_strength=args.confidence_teacher_strength,
            confidence_teacher_min=args.confidence_teacher_min,
            residual_scale_loss_weight=args.residual_scale_loss_weight,
            residual_scale_max=args.residual_scale_max,
            residual_frontier_ranking_weight=args.residual_frontier_ranking_weight,
            residual_frontier_ranking_margin=args.residual_frontier_ranking_margin,
            residual_frontier_hard_fraction=args.residual_frontier_hard_fraction,
            residual_frontier_min_gap=args.residual_frontier_min_gap,
            residual_near_path_weight=args.residual_near_path_weight,
            residual_astar_expanded_weight=args.residual_astar_expanded_weight,
            residual_weight_warmup_epochs=args.residual_weight_warmup_epochs,
            distill_residual_weight=args.distill_residual_weight,
            distill_confidence_weight=args.distill_confidence_weight,
            distill_scale_weight=args.distill_scale_weight,
            ignore_obstacles_in_bce=args.ignore_obstacles_in_bce,
            corridor_weight=args.corridor_weight,
            obstacle_penalty_weight=args.obstacle_penalty_weight,
            clearance_target_weight=args.clearance_target_weight,
            clearance_residual_weight=args.clearance_residual_weight,
            clearance_penalize_corridor=args.clearance_penalize_corridor,
            tv_weight=args.tv_weight,
            train=True,
        )
        val_metrics = _run_epoch(
            model=model,
            teacher_model=teacher_model,
            loader=val_loader,
            optimizer=optimizer,
            epoch_index=epoch,
            device=device,
            supervision_target=args.supervision_target,
            ranking_loss_weight=args.ranking_loss_weight,
            ranking_margin=args.ranking_margin,
            ranking_neg_min_cost=args.ranking_neg_min_cost,
            ranking_hard_fraction=args.ranking_hard_fraction,
            expansion_loss_weight=args.expansion_loss_weight,
            expansion_margin=args.expansion_margin,
            expansion_softness=args.expansion_softness,
            expansion_budget_multiplier=args.expansion_budget_multiplier,
            astar_expansion_loss_weight=astar_expansion_loss_weight,
            astar_expansion_margin=args.astar_expansion_margin,
            hybrid_expansion_loss_weight=args.hybrid_expansion_loss_weight,
            hybrid_expansion_margin=args.hybrid_expansion_margin,
            residual_target_transform=args.residual_target_transform,
            confidence_loss_weight=args.confidence_loss_weight,
            confidence_target_scale=args.confidence_target_scale,
            confidence_target_mode=args.confidence_target_mode,
            confidence_teacher_kernel=args.confidence_teacher_kernel,
            confidence_teacher_strength=args.confidence_teacher_strength,
            confidence_teacher_min=args.confidence_teacher_min,
            residual_scale_loss_weight=args.residual_scale_loss_weight,
            residual_scale_max=args.residual_scale_max,
            residual_frontier_ranking_weight=args.residual_frontier_ranking_weight,
            residual_frontier_ranking_margin=args.residual_frontier_ranking_margin,
            residual_frontier_hard_fraction=args.residual_frontier_hard_fraction,
            residual_frontier_min_gap=args.residual_frontier_min_gap,
            residual_near_path_weight=args.residual_near_path_weight,
            residual_astar_expanded_weight=args.residual_astar_expanded_weight,
            residual_weight_warmup_epochs=args.residual_weight_warmup_epochs,
            distill_residual_weight=args.distill_residual_weight,
            distill_confidence_weight=args.distill_confidence_weight,
            distill_scale_weight=args.distill_scale_weight,
            ignore_obstacles_in_bce=args.ignore_obstacles_in_bce,
            corridor_weight=args.corridor_weight,
            obstacle_penalty_weight=args.obstacle_penalty_weight,
            clearance_target_weight=args.clearance_target_weight,
            clearance_residual_weight=args.clearance_residual_weight,
            clearance_penalize_corridor=args.clearance_penalize_corridor,
            tv_weight=args.tv_weight,
            train=False,
        )

        astar_sr = None
        astar_expanded = None
        if args.compute_grid_astar_metric:
            astar_sr, astar_expanded = _grid_astar_metrics(
                model=model,
                loader=val_loader,
                device=device,
                lambda_guidance=args.astar_lambda,
                max_samples=args.astar_max_samples,
                heuristic_mode=args.astar_heuristic_mode,
                heuristic_weight=args.astar_heuristic_weight,
                guidance_integration_mode=args.astar_guidance_integration_mode,
                guidance_bonus_threshold=args.astar_guidance_bonus_threshold,
                residual_weight=args.astar_residual_weight,
            )

        msg = (
            f"epoch={epoch:03d} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"train_bce={train_metrics['bce']:.5f} "
            f"val_bce={val_metrics['bce']:.5f} "
            f"train_reg={train_metrics['reg']:.5f} "
            f"val_reg={val_metrics['reg']:.5f} "
            f"train_rank={train_metrics['rank']:.5f} "
            f"val_rank={val_metrics['rank']:.5f} "
            f"train_expand={train_metrics['expand']:.5f} "
            f"val_expand={val_metrics['expand']:.5f} "
            f"train_astar_expand={train_metrics['astar_expand']:.5f} "
            f"val_astar_expand={val_metrics['astar_expand']:.5f} "
            f"train_hybrid_expand={train_metrics['hybrid_expand']:.5f} "
            f"val_hybrid_expand={val_metrics['hybrid_expand']:.5f} "
            f"train_distill={train_metrics['distill']:.5f} "
            f"val_distill={val_metrics['distill']:.5f}"
        )
        if astar_sr is not None:
            msg += f" val_grid_astar_sr={astar_sr:.4f}"
        if astar_expanded is not None:
            msg += f" val_grid_astar_expanded={astar_expanded:.1f}"
        print(msg)

        resolved_metric_name, candidate_score = resolve_best_checkpoint_score(
            args.best_checkpoint_metric,
            val_loss=val_metrics["loss"],
            val_grid_astar_sr=astar_sr,
            val_grid_astar_expanded=astar_expanded,
        )
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (None if scheduler is None else scheduler.state_dict()),
            "val_loss": val_metrics["loss"],
            "val_grid_astar_sr": astar_sr,
            "val_grid_astar_expanded": astar_expanded,
            "resolved_best_checkpoint_metric": resolved_metric_name,
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": args.base_channels,
                "obstacle_cost": 1.0,
                "arch": args.arch,
                "use_pose_yaw_cond": use_pose_yaw_cond,
                "orientation_bins": int(args.orientation_bins),
                "cost_semantics": (
                    "residual_to_octile_heuristic"
                    if args.supervision_target == "residual_heuristic"
                    else (
                        "dense_distance_to_opt_traj"
                        if args.supervision_target in {"distance_field", "astar_guidance"}
                        else "low_on_opt_traj"
                    )
                ),
                "supervision_target": args.supervision_target,
                "output_mode": output_mode,
                "predict_confidence": bool(args.predict_confidence),
                "confidence_loss_weight": float(args.confidence_loss_weight),
                "confidence_target_scale": float(args.confidence_target_scale),
                "confidence_target_mode": str(args.confidence_target_mode),
                "confidence_head_kernel": int(args.confidence_head_kernel),
                "confidence_teacher_kernel": int(args.confidence_teacher_kernel),
                "confidence_teacher_strength": float(args.confidence_teacher_strength),
                "confidence_teacher_min": float(args.confidence_teacher_min),
                "confidence_only_finetune": bool(args.confidence_only_finetune),
                "predict_residual_scale": bool(args.predict_residual_scale),
                "residual_scale_max": float(args.residual_scale_max),
                "transformer_depth": int(args.transformer_depth),
                "transformer_heads": int(args.transformer_heads),
                "transformer_mlp_ratio": float(args.transformer_mlp_ratio),
                "extra_input_channels": int(getattr(model, "extra_input_channels", 0)),
                "clearance_input_clip_distance": float(args.clearance_input_clip_distance),
                "optimizer": str(args.optimizer),
                "weight_decay": float(args.weight_decay),
                "lr_warmup_epochs": int(args.lr_warmup_epochs),
                "lr_warmup_start_factor": float(args.lr_warmup_start_factor),
                "distill_teacher_ckpt": (
                    None if args.distill_teacher_ckpt is None else str(args.distill_teacher_ckpt)
                ),
                "distill_residual_weight": float(args.distill_residual_weight),
                "distill_confidence_weight": float(args.distill_confidence_weight),
                "distill_scale_weight": float(args.distill_scale_weight),
                "residual_scale_loss_weight": float(args.residual_scale_loss_weight),
                "residual_target_transform": (
                    str(args.residual_target_transform)
                    if output_mode == "residual_heuristic"
                    else "none"
                ),
                "residual_near_path_weight": float(args.residual_near_path_weight),
                "residual_astar_expanded_weight": float(args.residual_astar_expanded_weight),
                "residual_weight_warmup_epochs": int(args.residual_weight_warmup_epochs),
                "residual_frontier_ranking_weight": float(args.residual_frontier_ranking_weight),
                "residual_frontier_ranking_margin": float(args.residual_frontier_ranking_margin),
                "residual_frontier_hard_fraction": float(args.residual_frontier_hard_fraction),
                "residual_frontier_min_gap": float(args.residual_frontier_min_gap),
                "corridor_weight": float(args.corridor_weight),
                "ranking_loss_weight": float(args.ranking_loss_weight),
                "ranking_margin": float(args.ranking_margin),
                "ranking_neg_min_cost": float(args.ranking_neg_min_cost),
                "ranking_hard_fraction": float(args.ranking_hard_fraction),
                "expansion_loss_weight": float(args.expansion_loss_weight),
                "expansion_margin": float(args.expansion_margin),
                "expansion_softness": float(args.expansion_softness),
                "expansion_budget_multiplier": float(args.expansion_budget_multiplier),
                "astar_expansion_loss_weight": float(astar_expansion_loss_weight),
                "astar_expansion_margin": float(args.astar_expansion_margin),
                "hybrid_expansion_loss_weight": float(args.hybrid_expansion_loss_weight),
                "hybrid_expansion_margin": float(args.hybrid_expansion_margin),
                "astar_heuristic_mode": str(args.astar_heuristic_mode),
                "astar_heuristic_weight": float(args.astar_heuristic_weight),
                "astar_guidance_integration_mode": str(args.astar_guidance_integration_mode),
                "astar_guidance_bonus_threshold": float(args.astar_guidance_bonus_threshold),
                "astar_residual_weight": float(args.astar_residual_weight),
            },
            "train_args": vars(args),
        }
        torch.save(payload, last_ckpt)

        if best_score is None or candidate_score < best_score:
            best_score = candidate_score
            best_metric_name = resolved_metric_name
            best_metric_snapshot = {
                "val_loss": float(val_metrics["loss"]),
                "val_grid_astar_sr": (None if astar_sr is None else float(astar_sr)),
                "val_grid_astar_expanded": (
                    None if astar_expanded is None else float(astar_expanded)
                ),
            }
            torch.save(payload, best_ckpt)
            if resolved_metric_name == "grid_astar_expanded":
                print(
                    "new best checkpoint: "
                    f"{best_ckpt} (metric=grid_astar_expanded, "
                    f"val_grid_astar_sr={astar_sr:.4f}, "
                    f"val_grid_astar_expanded={astar_expanded:.1f}, "
                    f"val_loss={val_metrics['loss']:.5f})"
                )
            else:
                print(f"new best checkpoint: {best_ckpt} (val_loss={val_metrics['loss']:.5f})")

        if scheduler is not None:
            scheduler.step()

    if best_metric_name == "grid_astar_expanded":
        print(
            "finished. "
            f"best_metric=grid_astar_expanded "
            f"best_val_grid_astar_sr={best_metric_snapshot['val_grid_astar_sr']:.4f} "
            f"best_val_grid_astar_expanded={best_metric_snapshot['val_grid_astar_expanded']:.1f} "
            f"best_val_loss={best_metric_snapshot['val_loss']:.5f}"
        )
    else:
        print(f"finished. best_val_loss={best_metric_snapshot['val_loss']:.5f}")
    print(f"best_ckpt={best_ckpt}")


if __name__ == "__main__":
    main()
