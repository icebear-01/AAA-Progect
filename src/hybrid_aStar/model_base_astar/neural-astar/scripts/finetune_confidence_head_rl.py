"""Fine-tune residual confidence heads with a simple REINFORCE objective."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from hybrid_astar_guided.grid_astar import astar_8conn_stats, path_length_8conn
from neural_astar.api.guidance_infer import load_guidance_encoder
from neural_astar.datasets import ParkingGuidanceDataset, PlanningNPZGuidanceDataset, SpatialAugmentedDataset
from neural_astar.utils.confidence_rl import (
    compute_confidence_rl_reward,
    select_topk_confidence_cells,
)
from neural_astar.utils.guidance_training import (
    build_case_sampling_weights,
    load_hard_case_indices,
)
from neural_astar.utils.residual_prediction import (
    apply_residual_scale_np,
    decode_residual_prediction_np,
)
from neural_astar.utils.residual_confidence import build_residual_confidence_map


def _split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    return train_ds, val_ds


def _onehot_xy(one_hot_1hw: torch.Tensor) -> Tuple[int, int]:
    arr = one_hot_1hw[0].detach().cpu().numpy()
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return int(x), int(y)


def _extract_pose_yaws(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    occ = batch["occ_map"]
    start_pose = batch.get("start_pose")
    goal_pose = batch.get("goal_pose")
    if start_pose is None:
        start_yaw = None
    else:
        start_yaw = start_pose.to(device=device, dtype=occ.dtype)[:, 2]
    if goal_pose is None:
        goal_yaw = None
    else:
        goal_yaw = goal_pose.to(device=device, dtype=occ.dtype)[:, 2]
    return start_yaw, goal_yaw


def _freeze_non_confidence_params(model: torch.nn.Module) -> int:
    trainable = 0
    for name, param in model.named_parameters():
        enabled = name.startswith("conf_head.")
        param.requires_grad_(enabled)
        if enabled:
            trainable += int(param.numel())
    return trainable


def _build_deterministic_confidence(
    *,
    occ_map: np.ndarray,
    residual_map: np.ndarray,
    learned_confidence: np.ndarray,
    combine_mode: str,
    spike_kernel: int,
    spike_strength: float,
    spike_min: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if combine_mode == "learned":
        spike = np.ones_like(residual_map, dtype=np.float32)
    elif combine_mode == "learned_spike":
        spike = build_residual_confidence_map(
            residual_map=residual_map,
            occ_map=occ_map,
            mode="spike_suppression",
            kernel_size=int(spike_kernel),
            strength=float(spike_strength),
            min_confidence=float(spike_min),
        )
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")
    deterministic = np.clip(learned_confidence, 0.0, 1.0).astype(np.float32) * spike
    deterministic[occ_map > 0.5] = 0.0
    return deterministic.astype(np.float32), spike.astype(np.float32)


def _run_planner(
    *,
    occ_map: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    residual_map: np.ndarray,
    confidence_map: np.ndarray,
    residual_weight: float,
    heuristic_mode: str,
    diagonal_cost: float,
    allow_corner_cut: bool,
) -> Tuple[bool, int, float]:
    result = astar_8conn_stats(
        occ_map=occ_map,
        start_xy=start_xy,
        goal_xy=goal_xy,
        heuristic_mode=heuristic_mode,
        heuristic_residual_map=residual_map,
        residual_confidence_map=confidence_map,
        residual_weight=float(residual_weight),
        diagonal_cost=float(diagonal_cost),
        allow_corner_cut=bool(allow_corner_cut),
    )
    path_length = (
        0.0
        if result.path is None
        else path_length_8conn(result.path, diagonal_cost=float(diagonal_cost))
    )
    return bool(result.success), int(result.expanded_nodes), float(path_length)


def _evaluate_model(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    device: str,
    max_samples: int,
    residual_weight: float,
    heuristic_mode: str,
    diagonal_cost: float,
    allow_corner_cut: bool,
    combine_mode: str,
    spike_kernel: int,
    spike_strength: float,
    spike_min: float,
) -> Dict[str, float]:
    model.eval()
    total = 0
    success = 0
    expanded_sum = 0.0
    path_sum = 0.0
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            occ = sample["occ_map"].unsqueeze(0).to(device)
            start = sample["start_map"].unsqueeze(0).to(device)
            goal = sample["goal_map"].unsqueeze(0).to(device)
            start_yaw, goal_yaw = _extract_pose_yaws(
                {
                    "occ_map": occ,
                    "start_pose": sample.get("start_pose", None).unsqueeze(0)
                    if sample.get("start_pose") is not None
                    else None,
                    "goal_pose": sample.get("goal_pose", None).unsqueeze(0)
                    if sample.get("goal_pose") is not None
                    else None,
                },
                device=device,
            )
            out = model(occ, start, goal, start_yaw=start_yaw, goal_yaw=goal_yaw)
            if out.confidence_map is None:
                raise RuntimeError("RL fine-tuning requires a model with confidence_map output")

            occ_np = occ[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_np = out.cost_map[0, 0].detach().cpu().numpy().astype(np.float32)
            residual_np = decode_residual_prediction_np(
                pred_np,
                str(getattr(model, "residual_target_transform", "none")),
            )
            if out.scale_map is not None:
                residual_np = apply_residual_scale_np(
                    residual_np,
                    out.scale_map[0, 0].detach().cpu().numpy().astype(np.float32),
                )
            learned_conf = out.confidence_map[0, 0].detach().cpu().numpy().astype(np.float32)
            deterministic_conf, _ = _build_deterministic_confidence(
                occ_map=occ_np,
                residual_map=residual_np,
                learned_confidence=learned_conf,
                combine_mode=combine_mode,
                spike_kernel=spike_kernel,
                spike_strength=spike_strength,
                spike_min=spike_min,
            )
            ok, expanded, path_length = _run_planner(
                occ_map=occ_np,
                start_xy=_onehot_xy(start[0]),
                goal_xy=_onehot_xy(goal[0]),
                residual_map=residual_np,
                confidence_map=deterministic_conf,
                residual_weight=residual_weight,
                heuristic_mode=heuristic_mode,
                diagonal_cost=diagonal_cost,
                allow_corner_cut=allow_corner_cut,
            )
            total += 1
            success += int(ok)
            expanded_sum += float(expanded)
            path_sum += float(path_length)
            if 0 < int(max_samples) <= total:
                break
    denom = max(total, 1)
    return {
        "success_rate": success / denom,
        "expanded_nodes": expanded_sum / denom,
        "path_length": path_sum / denom,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune confidence head with REINFORCE.")
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--train-npz", type=Path, default=None)
    data_group.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--val-npz", type=Path, default=None)
    p.add_argument("--train-split", type=str, default="train", choices=["train", "valid", "test"])
    p.add_argument("--val-split", type=str, default="valid", choices=["train", "valid", "test"])
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--train-augment-mode", type=str, default="none", choices=["none", "flip", "rot4"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--val-max-samples", type=int, default=64)
    p.add_argument("--residual-weight", type=float, default=1.25)
    p.add_argument("--heuristic-mode", type=str, default="octile", choices=["euclidean", "manhattan", "chebyshev", "octile"])
    p.add_argument("--diagonal-cost", type=float, default=float(np.sqrt(2.0)))
    p.add_argument("--allow-corner-cut", dest="allow_corner_cut", action="store_true")
    p.add_argument("--no-allow-corner-cut", dest="allow_corner_cut", action="store_false")
    p.add_argument("--confidence-combine-mode", type=str, default="learned_spike", choices=["learned", "learned_spike"])
    p.add_argument("--spike-kernel", type=int, default=5)
    p.add_argument("--spike-strength", type=float, default=0.5)
    p.add_argument("--spike-min", type=float, default=0.25)
    p.add_argument("--rl-topk", type=int, default=128)
    p.add_argument("--rl-min-residual", type=float, default=0.1)
    p.add_argument("--rl-min-prob", type=float, default=0.05)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--path-length-penalty", type=float, default=100.0)
    p.add_argument("--failure-penalty", type=float, default=4096.0)
    p.add_argument("--entropy-weight", type=float, default=1e-3)
    p.add_argument("--hard-case-csv", type=Path, default=None)
    p.add_argument("--hard-case-delta-column", type=str, default="learned_minus_improved")
    p.add_argument("--hard-case-only", action="store_true")
    p.add_argument("--hard-case-top-fraction", type=float, default=0.15)
    p.add_argument("--hard-case-min-delta", type=float, default=64.0)
    p.add_argument("--hard-case-max-count", type=int, default=256)
    p.add_argument("--hard-case-boost-weight", type=float, default=4.0)
    p.set_defaults(allow_corner_cut=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_npz is not None:
        train_dataset = PlanningNPZGuidanceDataset(
            npz_path=args.train_npz,
            split=args.train_split,
            seed=args.seed,
            orientation_bins=1,
        )
        val_dataset = PlanningNPZGuidanceDataset(
            npz_path=(args.val_npz if args.val_npz is not None else args.train_npz),
            split=args.val_split,
            seed=args.seed + 1,
            orientation_bins=1,
        )
    else:
        full_dataset = ParkingGuidanceDataset(args.train_dir, orientation_bins=1)
        train_dataset, val_dataset = _split_dataset(full_dataset, args.val_ratio, args.seed)

    hard_case_indices = []
    if args.hard_case_csv is not None:
        hard_case_indices = load_hard_case_indices(
            args.hard_case_csv,
            delta_column=args.hard_case_delta_column,
            top_fraction=args.hard_case_top_fraction,
            min_delta=args.hard_case_min_delta,
            max_count=args.hard_case_max_count,
        )
    if args.hard_case_only:
        if not hard_case_indices:
            raise ValueError("--hard-case-only requires non-empty hard-case selection")
        valid_indices = [int(idx) for idx in hard_case_indices if 0 <= int(idx) < len(train_dataset)]
        if not valid_indices:
            raise ValueError("No valid hard-case indices remain after dataset bounds check")
        train_dataset = Subset(train_dataset, valid_indices)
        hard_case_indices = list(range(len(valid_indices)))
    if args.max_train_samples > 0:
        train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), int(args.max_train_samples)))))
    base_train_size = len(train_dataset)
    if args.train_augment_mode != "none":
        train_dataset = SpatialAugmentedDataset(train_dataset, mode=args.train_augment_mode)

    train_sampler = None
    if hard_case_indices:
        if hasattr(train_dataset, "build_sampling_weights"):
            weights = train_dataset.build_sampling_weights(
                emphasized_base_indices=hard_case_indices,
                emphasized_weight=args.hard_case_boost_weight,
            )
        else:
            weights = build_case_sampling_weights(
                len(train_dataset),
                emphasized_indices=hard_case_indices,
                emphasized_weight=args.hard_case_boost_weight,
            )
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    model = load_guidance_encoder(args.ckpt, device=device)
    if getattr(model, "output_mode", None) != "residual_heuristic":
        raise ValueError("RL fine-tuning currently requires a residual-heuristic checkpoint")
    if getattr(model, "predict_confidence", False) is not True or getattr(model, "conf_head", None) is None:
        raise ValueError("RL fine-tuning requires a checkpoint with predict_confidence=True")

    trainable_params = _freeze_non_confidence_params(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.output_dir / "best.pt"
    last_ckpt = args.output_dir / "last.pt"
    best_val_expanded = float("inf")

    print(
        f"train_size={len(train_dataset)} val_size={len(val_dataset)} "
        f"(base_train_size={base_train_size}, augment={args.train_augment_mode})"
    )
    print(
        f"trainable_params={trainable_params} residual_weight={args.residual_weight:.3f} "
        f"combine_mode={args.confidence_combine_mode}"
    )
    if hard_case_indices:
        print(
            "hard_case_replay="
            f"{len(hard_case_indices)} cases "
            f"(boost_weight={args.hard_case_boost_weight:.2f}, "
            f"top_fraction={args.hard_case_top_fraction:.3f}, "
            f"min_delta={args.hard_case_min_delta:.3f})"
        )

    prob_floor = float(args.rl_min_prob)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_entropy = 0.0
        total_baseline_expanded = 0.0
        total_sampled_expanded = 0.0
        updates = 0

        for batch in train_loader:
            occ = batch["occ_map"].to(device)
            start = batch["start_map"].to(device)
            goal = batch["goal_map"].to(device)
            start_yaw, goal_yaw = _extract_pose_yaws(batch, device=device)
            out = model(occ, start, goal, start_yaw=start_yaw, goal_yaw=goal_yaw)
            if out.confidence_map is None:
                raise RuntimeError("Model did not return confidence_map")

            occ_np = occ[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_np = out.cost_map[0, 0].detach().cpu().numpy().astype(np.float32)
            residual_np = decode_residual_prediction_np(
                pred_np,
                str(getattr(model, "residual_target_transform", "none")),
            )
            if out.scale_map is not None:
                residual_np = apply_residual_scale_np(
                    residual_np,
                    out.scale_map[0, 0].detach().cpu().numpy().astype(np.float32),
                )
            learned_conf = out.confidence_map[0, 0].clamp(prob_floor, 1.0 - prob_floor)
            deterministic_conf_np, spike_np = _build_deterministic_confidence(
                occ_map=occ_np,
                residual_map=residual_np,
                learned_confidence=learned_conf.detach().cpu().numpy().astype(np.float32),
                combine_mode=args.confidence_combine_mode,
                spike_kernel=args.spike_kernel,
                spike_strength=args.spike_strength,
                spike_min=args.spike_min,
            )

            start_xy = _onehot_xy(start[0])
            goal_xy = _onehot_xy(goal[0])
            base_success, base_expanded, base_path = _run_planner(
                occ_map=occ_np,
                start_xy=start_xy,
                goal_xy=goal_xy,
                residual_map=residual_np,
                confidence_map=deterministic_conf_np,
                residual_weight=args.residual_weight,
                heuristic_mode=args.heuristic_mode,
                diagonal_cost=args.diagonal_cost,
                allow_corner_cut=args.allow_corner_cut,
            )
            active_mask = select_topk_confidence_cells(
                residual_map=residual_np,
                occ_map=occ_np,
                topk=args.rl_topk,
                min_residual=args.rl_min_residual,
            )
            if not np.any(active_mask):
                continue

            active_flat = np.flatnonzero(active_mask.reshape(-1))
            probs = learned_conf.reshape(-1)[torch.from_numpy(active_flat).to(device=device, dtype=torch.long)]
            dist = Bernoulli(probs=probs)
            sampled = dist.sample()
            sampled_conf_np = deterministic_conf_np.copy().reshape(-1)
            if args.confidence_combine_mode == "learned_spike":
                sampled_conf_np[active_flat] = spike_np.reshape(-1)[active_flat] * sampled.detach().cpu().numpy().astype(np.float32)
            else:
                sampled_conf_np[active_flat] = sampled.detach().cpu().numpy().astype(np.float32)
            sampled_conf_np = sampled_conf_np.reshape(deterministic_conf_np.shape).astype(np.float32)

            sampled_success, sampled_expanded, sampled_path = _run_planner(
                occ_map=occ_np,
                start_xy=start_xy,
                goal_xy=goal_xy,
                residual_map=residual_np,
                confidence_map=sampled_conf_np,
                residual_weight=args.residual_weight,
                heuristic_mode=args.heuristic_mode,
                diagonal_cost=args.diagonal_cost,
                allow_corner_cut=args.allow_corner_cut,
            )
            reward = compute_confidence_rl_reward(
                baseline_expanded=base_expanded,
                sampled_expanded=sampled_expanded,
                baseline_path_length=base_path,
                sampled_path_length=sampled_path,
                sampled_success=sampled_success and base_success,
                reward_scale=args.reward_scale,
                path_length_penalty=args.path_length_penalty,
                failure_penalty=args.failure_penalty,
            )

            log_prob = dist.log_prob(sampled).mean()
            entropy = dist.entropy().mean()
            loss = -float(reward) * log_prob - float(args.entropy_weight) * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_reward += float(reward)
            total_entropy += float(entropy.item())
            total_baseline_expanded += float(base_expanded)
            total_sampled_expanded += float(sampled_expanded)
            updates += 1

        val_metrics = _evaluate_model(
            model=model,
            dataset=val_dataset,
            device=device,
            max_samples=args.val_max_samples,
            residual_weight=args.residual_weight,
            heuristic_mode=args.heuristic_mode,
            diagonal_cost=args.diagonal_cost,
            allow_corner_cut=args.allow_corner_cut,
            combine_mode=args.confidence_combine_mode,
            spike_kernel=args.spike_kernel,
            spike_strength=args.spike_strength,
            spike_min=args.spike_min,
        )
        denom = max(updates, 1)
        print(
            f"epoch={epoch:03d} "
            f"updates={updates} "
            f"train_loss={total_loss/denom:.5f} "
            f"train_reward={total_reward/denom:.5f} "
            f"train_entropy={total_entropy/denom:.5f} "
            f"baseline_expanded={total_baseline_expanded/denom:.1f} "
            f"sampled_expanded={total_sampled_expanded/denom:.1f} "
            f"val_sr={val_metrics['success_rate']:.4f} "
            f"val_expanded={val_metrics['expanded_nodes']:.1f} "
            f"val_path_length={val_metrics['path_length']:.4f}"
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "rl_cfg": vars(args),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": model.base_channels,
                "obstacle_cost": model.obstacle_cost,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "predict_confidence": model.predict_confidence,
                "confidence_head_kernel": model.confidence_head_kernel,
                "residual_target_transform": model.residual_target_transform,
            },
        }
        torch.save(payload, last_ckpt)
        if float(val_metrics["expanded_nodes"]) < best_val_expanded:
            best_val_expanded = float(val_metrics["expanded_nodes"])
            torch.save(payload, best_ckpt)
            print(f"new best checkpoint: {best_ckpt} (val_expanded={best_val_expanded:.1f})")

    print(f"finished. best_val_expanded={best_val_expanded:.1f}")
    print(f"best_ckpt={best_ckpt}")


if __name__ == "__main__":
    main()
