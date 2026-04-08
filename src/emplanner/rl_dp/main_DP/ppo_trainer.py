"""
PPO training CLI for SLPathEnv with TensorBoard logging and checkpoint support.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import time

from ppo import (
    ActorCritic as _ActorCritic,
    PPOConfig,
    PPOTrainer,
    encode_observation as _encode_observation,
)
from rl_env import SLPathEnv
from sl_grid import GridSpec, default_training_grid_spec

# 兼容旧接口：允许其他模块仍然从 ppo_trainer 导入模型及编码函数
ActorCritic = _ActorCritic
encode_observation = _encode_observation

__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "ActorCritic",
    "encode_observation",
    "train",
]


def _apply_training_hparam_overrides(config_data: Dict[str, object], args: argparse.Namespace) -> None:
    """根据命令行参数覆盖 PPOConfig 中的训练超参。"""
    default_num_envs = PPOConfig.__dataclass_fields__["num_envs"].default
    num_envs = int(config_data.get("num_envs", default_num_envs))
    if args.num_envs is not None:
        num_envs = max(1, int(args.num_envs))
        config_data["num_envs"] = num_envs
    else:
        config_data["num_envs"] = num_envs

    if args.buffer_size is not None:
        total_steps = max(1, int(args.buffer_size))
        rollout_steps = max(1, total_steps // max(1, num_envs))
        config_data["rollout_steps"] = rollout_steps

    if args.num_epoch is not None:
        config_data["num_epochs"] = max(1, int(args.num_epoch))

    if args.batch_size is not None:
        config_data["mini_batch_size"] = max(1, int(args.batch_size))

    if args.gamma is not None:
        config_data["gamma"] = float(args.gamma)

    if args.lambd is not None:
        config_data["gae_lambda"] = float(args.lambd)

    if args.epsilon is not None:
        config_data["clip_ratio"] = float(args.epsilon)

    if args.lr is not None:
        config_data["lr"] = float(args.lr)

    if args.max_grad_norm is not None:
        config_data["max_grad_norm"] = float(args.max_grad_norm)


def train(
    env: SLPathEnv,
    config: PPOConfig,
    *,
    total_updates: int,
    log_interval: int = 10,
    start_update: int = 1,
    resume_state: Dict[str, object] | None = None,
) -> None:
    if start_update > total_updates:
        print(
            f"Requested start_update={start_update} exceeds total_updates={total_updates}; nothing to do."
        )
        return

    trainer = PPOTrainer(env, config)
    if resume_state:
        trainer.actor_critic.load_state_dict(resume_state["model_state"])
        trainer.optimizer.load_state_dict(resume_state["optimizer_state"])
        for state in trainer.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(trainer.device)
        trainer.global_step = int(resume_state.get("global_step", trainer.global_step))
        trainer.load_value_normalizer_state(resume_state.get("value_normalizer_state"))
        print(f"Resumed optimizer and policy state (global_step={trainer.global_step}).")

    last_completed_update = start_update - 1
    current_update = start_update - 1
    try:
        for update in range(start_update, total_updates + 1):
            current_update = update
            update_start_time = time.perf_counter()

            rollout_start_time = time.perf_counter()
            episode_returns = trainer.collect_rollout()
            rollout_duration = time.perf_counter() - rollout_start_time
            trainer.log({"timing/rollout_seconds": rollout_duration}, update)
            if trainer.rollout_stats:
                trainer.log(trainer.rollout_stats, update)

            update_phase_start = time.perf_counter()
            metrics = trainer.update()
            update_duration = time.perf_counter() - update_phase_start
            trainer.log({"timing/update_seconds": update_duration}, update)

            avg_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
            log_start_time = time.perf_counter()
            if episode_returns:
                trainer.log({"train/episode_return": avg_return}, update)

            trainer.log(
                {
                    "train/policy_loss": metrics.get("policy_loss", 0.0),
                    "train/value_loss": metrics.get("value_loss", 0.0),
                    "train/entropy": metrics.get("entropy", 0.0),
                    "train/kl_divergence": metrics.get("kl_divergence", 0.0),
                },
                update,
            )
            log_duration = time.perf_counter() - log_start_time

            checkpoint_duration = 0.0
            checkpoint_interval = config.checkpoint_interval or 0
            if checkpoint_interval > 0 and update % checkpoint_interval == 0:
                checkpoint_start_time = time.perf_counter()
                checkpoint_path = trainer.save_checkpoint(config.checkpoint_path, update=update)
                checkpoint_duration = time.perf_counter() - checkpoint_start_time
                trainer.log({"timing/checkpoint_seconds": checkpoint_duration}, update)
                print(
                    f"Checkpoint saved at update {update} to {checkpoint_path} "
                    f"({checkpoint_duration:.2f}s)"
                )

            total_duration = time.perf_counter() - update_start_time
            trainer.log({"timing/update_total_seconds": total_duration}, update)

            if update % log_interval == 0:
                stats = trainer.rollout_stats
                print(
                    f"Update {update:04d} | "
                    f"avg_return={avg_return:.3f} | "
                    f"policy_loss={metrics.get('policy_loss', 0.0):.4f} | "
                    f"value_loss={metrics.get('value_loss', 0.0):.4f} | "
                    f"kl={metrics.get('kl_divergence', 0.0):.4f} | "
                    f"time rollout={rollout_duration:.2f}s update={update_duration:.2f}s "
                    f"log={log_duration:.2f}s checkpoint={checkpoint_duration:.2f}s "
                    f"total={total_duration:.2f}s"
                )
                if stats:
                    print(
                        "  Rollout breakdown | "
                        f"encode={stats.get('timing/rollout_encode_seconds', 0.0):.2f}s "
                        f"(numpy={stats.get('timing/rollout_encode_numpy_seconds', 0.0):.2f}s, "
                        f"to_tensor={stats.get('timing/rollout_encode_tensor_seconds', 0.0):.2f}s) | "
                        f"policy={stats.get('timing/rollout_policy_seconds', 0.0):.2f}s "
                        f"(forward={stats.get('timing/rollout_policy_forward_seconds', 0.0):.2f}s, "
                        f"sample={stats.get('timing/rollout_policy_sample_seconds', 0.0):.2f}s) | "
                        f"env={stats.get('timing/rollout_env_seconds', 0.0):.2f}s | "
                        f"buffer={stats.get('timing/rollout_buffer_seconds', 0.0):.2f}s | "
                        f"reset={stats.get('timing/rollout_reset_seconds', 0.0):.2f}s"
                    )

            last_completed_update = update
    except KeyboardInterrupt:
        saved_update = max(last_completed_update, current_update)
        print(f"Training interrupted. Saving checkpoint at update {saved_update}.")
        checkpoint_path = trainer.save_checkpoint(config.checkpoint_path, update=saved_update)
        print(f"Checkpoint saved to {checkpoint_path}")
        return
    except Exception:
        saved_update = max(last_completed_update, current_update)
        print(f"Exception during training. Saving checkpoint at update {saved_update}.")
        checkpoint_path = trainer.save_checkpoint(config.checkpoint_path, update=saved_update)
        print(f"Checkpoint saved to {checkpoint_path}")
        raise
    else:
        checkpoint_path = trainer.save_checkpoint(config.checkpoint_path, update=last_completed_update)
        print(f"Checkpoint saved to {checkpoint_path}")
    finally:
        trainer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the SLPathEnv.")
    parser.add_argument("--updates", type=int, default=80000, help="number of PPO updates")
    parser.add_argument("--log-interval", type=int, default=10, help="logging frequency")
    parser.add_argument("--buffer-size", type=int, default=None, help="每轮采样总步数（跨所有 env）")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="并行环境数量（默认 4）",
    )
    parser.add_argument("--num-epoch", type=int, default=None, help="每次 update 的 epoch 数")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO 小批次大小")
    parser.add_argument("--gamma", type=float, default=None, help="折扣因子")
    parser.add_argument("--lambd", type=float, default=None, help="GAE λ 系数")
    parser.add_argument("--epsilon", type=float, default=None, help="PPO 剪辑系数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument(
        "--obstacle-count-min",
        type=int,
        default=None,
        help="每个场景随机障碍物数量下限",
    )
    parser.add_argument(
        "--obstacle-count-max",
        type=int,
        default=None,
        help="每个场景随机障碍物数量上限",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="熵系数（默认沿用配置或 0.01）",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=None,
        help="KL 正则系数（默认沿用配置或 0.0）",
    )
    parser.add_argument("--max-grad-norm", type=float, default=None, help="梯度裁剪阈值")
    parser.add_argument("--max-steps", type=int, default=None, help="总训练步数上限")
    parser.add_argument(
        "--scenario-pool-size",
        type=int,
        default=None,
        help="每次 reset 随机生成并用传统 DP 评估的候选场景数",
    )
    parser.add_argument(
        "--scenario-top-k",
        type=int,
        default=None,
        help="从 DP 评分最好的前 K 个候选场景中随机采样一个",
    )
    parser.add_argument(
        "--scenario-min-obstacles",
        type=int,
        default=None,
        help="筛选场景时要求的最少障碍物数量",
    )
    parser.add_argument(
        "--scenario-max-avg-cost",
        type=float,
        default=None,
        help="传统 DP 平均每步代价上限，超出则视为场景质量不够高",
    )
    parser.add_argument(
        "--scenario-max-attempts",
        type=int,
        default=None,
        help="为凑齐高质量候选场景允许的最大采样次数",
    )
    parser.add_argument(
        "--scenario-dataset",
        type=str,
        default=None,
        help="离线筛好的场景库 JSON；指定后训练直接从场景库采样",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: runs/ppo)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to save the trained policy checkpoint (overrides --checkpoint-dir)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="main/checkpoints",
        help="base directory for timestamped checkpoints when --checkpoint is not provided",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to an existing PPO checkpoint to resume training from",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="save a checkpoint every N updates (disabled by default)",
    )
    parser.add_argument("--device", type=str, default=None, help="override training device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    resume_state: Dict[str, object] | None = None
    start_update = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu")
        spec = GridSpec(**payload["grid_spec"])
        env_kwargs = payload.get("environment", {})
        if args.scenario_pool_size is not None:
            env_kwargs["scenario_pool_size"] = max(1, int(args.scenario_pool_size))
        if args.scenario_top_k is not None:
            env_kwargs["scenario_top_k"] = max(1, int(args.scenario_top_k))
        if args.obstacle_count_min is not None:
            env_kwargs["min_obstacles"] = max(0, int(args.obstacle_count_min))
        if args.obstacle_count_max is not None:
            env_kwargs["max_obstacles"] = max(0, int(args.obstacle_count_max))
        if args.scenario_min_obstacles is not None:
            env_kwargs["scenario_min_obstacles"] = max(0, int(args.scenario_min_obstacles))
        if args.scenario_max_avg_cost is not None:
            env_kwargs["scenario_max_avg_cost"] = float(args.scenario_max_avg_cost)
        if args.scenario_max_attempts is not None:
            env_kwargs["scenario_max_attempts"] = max(1, int(args.scenario_max_attempts))
        if args.scenario_dataset is not None:
            env_kwargs["scenario_dataset_path"] = args.scenario_dataset
        env = SLPathEnv(spec, **env_kwargs)

        config_payload = dict(payload["config"])
        config_payload.setdefault("kl_coef", PPOConfig.__dataclass_fields__["kl_coef"].default)
        config_payload.setdefault("num_envs", PPOConfig.__dataclass_fields__["num_envs"].default)
        base_config = PPOConfig(**config_payload)
        config_data = asdict(base_config)

        saved_device = config_data.get("device", default_device)
        device_name = args.device or saved_device or default_device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        config_data["device"] = device_name

        if args.log_dir and args.log_dir != config_data.get("log_dir"):
            config_data["log_dir"] = args.log_dir
        if args.checkpoint_interval is not None:
            config_data["checkpoint_interval"] = args.checkpoint_interval
        if args.beta is not None:
            config_data["entropy_coef"] = args.beta
        if args.kl_coef is not None:
            config_data["kl_coef"] = args.kl_coef
        default_num_envs = PPOConfig.__dataclass_fields__["num_envs"].default
        if args.num_envs is not None:
            config_data["num_envs"] = max(1, int(args.num_envs))
        else:
            config_data.setdefault("num_envs", default_num_envs)

        if args.checkpoint:
            config_data["checkpoint_path"] = args.checkpoint
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_data["checkpoint_path"] = str(
                Path(args.checkpoint_dir) / f"{Path(resume_path).stem}_{timestamp}.pt"
            )

        _apply_training_hparam_overrides(config_data, args)
        config = PPOConfig(**config_data)

        saved_update = int(payload.get("update") or 0)
        start_update = saved_update + 1
        resume_state = {
            "model_state": payload["model_state"],
            "optimizer_state": payload.get("optimizer_state", {}),
            "global_step": payload.get("global_step", 0),
            "value_normalizer_state": payload.get("value_normalizer_state"),
        }
        print(f"Resuming from checkpoint {resume_path} (completed update {saved_update}).")
    else:
        device_name = args.device or default_device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        spec = default_training_grid_spec()
        env = SLPathEnv(
            spec,
            min_obstacles=max(0, int(args.obstacle_count_min))
            if args.obstacle_count_min is not None
            else 0,
            max_obstacles=max(0, int(args.obstacle_count_max))
            if args.obstacle_count_max is not None
            else 6,
            lateral_move_limit=3,
            scenario_pool_size=max(1, int(args.scenario_pool_size))
            if args.scenario_pool_size is not None
            else 1,
            scenario_top_k=max(1, int(args.scenario_top_k))
            if args.scenario_top_k is not None
            else 1,
            scenario_min_obstacles=max(0, int(args.scenario_min_obstacles))
            if args.scenario_min_obstacles is not None
            else 0,
            scenario_max_avg_cost=float(args.scenario_max_avg_cost)
            if args.scenario_max_avg_cost is not None
            else None,
            scenario_max_attempts=max(1, int(args.scenario_max_attempts))
            if args.scenario_max_attempts is not None
            else None,
            scenario_dataset_path=args.scenario_dataset,
            seed=42,
        )
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = str(Path(args.checkpoint_dir) / f"ppo_policy_{run_timestamp}.pt")
        log_dir = args.log_dir or "runs/ppo"
        default_num_envs = PPOConfig.__dataclass_fields__["num_envs"].default
        config_kwargs: Dict[str, object] = {
            "device": device_name,
            "log_dir": log_dir,
            "checkpoint_path": checkpoint_path,
            "num_envs": (
                max(1, int(args.num_envs)) if args.num_envs is not None else default_num_envs
            ),
        }
        if args.checkpoint_interval is not None:
            config_kwargs["checkpoint_interval"] = args.checkpoint_interval
        if args.beta is not None:
            config_kwargs["entropy_coef"] = args.beta
        if args.kl_coef is not None:
            config_kwargs["kl_coef"] = args.kl_coef

        _apply_training_hparam_overrides(config_kwargs, args)
        config = PPOConfig(**config_kwargs)

    total_updates = int(args.updates)
    if args.max_steps is not None and args.max_steps > 0:
        steps_per_update = config.rollout_steps * config.num_envs
        if steps_per_update > 0:
            max_updates_from_steps = max(1, int(args.max_steps // steps_per_update))
            if max_updates_from_steps < total_updates:
                total_updates = max_updates_from_steps
                print(
                    f"Limiting total updates to {total_updates} based on max_steps={args.max_steps}."
                )

    if start_update > total_updates:
        print(
            f"Checkpoint already at update {start_update - 1}; target updates={total_updates}. Nothing to do."
        )
        return

    print(f"Using entropy_coef={config.entropy_coef}, kl_coef={config.kl_coef}, num_envs={config.num_envs}")

    train(
        env,
        config,
        total_updates=total_updates,
        log_interval=args.log_interval,
        start_update=start_update,
        resume_state=resume_state,
    )


if __name__ == "__main__":
    main()
