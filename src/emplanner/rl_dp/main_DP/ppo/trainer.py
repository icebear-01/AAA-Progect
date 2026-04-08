from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from rl_env import SLPathEnv

from .buffer import RolloutBuffer, RunningNormalizer
from .config import PPOConfig
from .encoding import encode_observation
from .model import ActorCritic


class PPOTrainer:
    """负责 PPO rollout、更新、日志与 checkpoint。"""

    def __init__(self, env: SLPathEnv, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.primary_env = env
        self._env_spec = env.grid_spec
        self._base_env_kwargs = self._extract_env_kwargs(env)

        self.envs: List[SLPathEnv] = [env]
        for idx in range(1, max(1, int(config.num_envs))):
            clone_kwargs = dict(self._base_env_kwargs)
            base_seed = clone_kwargs.pop("seed", None)
            clone_seed = None if base_seed is None else int(base_seed) + idx
            clone_kwargs["seed"] = clone_seed
            clone_env = SLPathEnv(self._env_spec, **clone_kwargs)
            self.envs.append(clone_env)

        sample_obs = self.envs[0].reset()
        feature_dim = encode_observation(sample_obs, self._env_spec).shape[0]
        action_dim = self._env_spec.l_samples

        occupancy_shape = (self._env_spec.s_samples, self._env_spec.l_samples)
        grid_spatial_size = occupancy_shape[0] * occupancy_shape[1]
        extra_dim = feature_dim % grid_spatial_size
        grid_channels = (feature_dim - extra_dim) // grid_spatial_size
        if grid_channels < 1:
            raise ValueError("Encoded observation must contain at least one grid plane.")
        if extra_dim < 0:
            raise ValueError("Encoded observation dimensionality is inconsistent with grid size.")

        self.actor_critic = ActorCritic(
            occupancy_shape=occupancy_shape,
            grid_channels=grid_channels,
            extra_dim=extra_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.lr)
        self.buffer = RolloutBuffer()
        self.value_normalizer = RunningNormalizer() if config.normalize_value_targets else None
        self._last_value_buffer = torch.zeros(
            len(self.envs), dtype=torch.float32, device=self.device
        )

        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.global_step = 0
        self.rollout_stats: Dict[str, float] = {}

    def _extract_env_kwargs(self, env: SLPathEnv) -> Dict[str, object]:
        return {
            "min_obstacles": env.min_obstacles,
            "max_obstacles": env.max_obstacles,
            "obstacle_length_range": env.obstacle_length_range,
            "obstacle_width_range": env.obstacle_width_range,
            "avoid_obstacle_overlap": env.avoid_obstacle_overlap,
            "obstacle_overlap_clearance": env.obstacle_overlap_clearance,
            "obstacle_sampling_attempts_per_obstacle": env.obstacle_sampling_attempts_per_obstacle,
            "smoothness_penalty": env.smoothness_penalty,
            "base_smoothness_penalty": env.base_smoothness_penalty,
            "move_limit_penalty_multiplier": env.move_limit_penalty_multiplier,
            "lateral_reference": env.lateral_reference,
            "reference_penalty": env.reference_penalty,
            "max_slope": env.max_slope,
            "slope_penalty": env.slope_penalty,
            "max_slope_penalty": env.max_slope_penalty,
            "collision_penalty": env.collision_penalty,
            "terminal_reward": env.terminal_reward,
            "base_smoothness_penalty": env.base_smoothness_penalty,
            "non_goal_penalty": env.non_goal_penalty,
            "no_valid_action_penalty": env.no_valid_action_penalty,
            "lateral_move_limit": env.lateral_move_limit,
            "coarse_collision_inflation": env.coarse_collision_inflation,
            "fine_collision_inflation": env.fine_collision_inflation,
            "interpolation_points": env.interpolation_points,
            "vehicle_length": env.vehicle_length,
            "vehicle_width": env.vehicle_width,
            "start_clear_fraction": env.start_clear_fraction,
            "scenario_pool_size": env.scenario_pool_size,
            "scenario_top_k": env.scenario_top_k,
            "scenario_min_obstacles": env.scenario_min_obstacles,
            "scenario_max_avg_cost": env.scenario_max_avg_cost,
            "scenario_max_attempts": env.scenario_max_attempts,
            "scenario_dataset_path": env.scenario_dataset_path,
            "seed": None,
        }

    def collect_rollout(self) -> List[float]:
        """采样固定步数的环境交互数据。"""
        self.buffer.clear()
        encode_numpy_time = 0.0
        encode_tensor_time = 0.0
        policy_forward_time = 0.0
        policy_sample_time = 0.0
        env_time = 0.0
        buffer_time = 0.0
        reset_time = 0.0
        resets = 0

        num_envs = len(self.envs)
        observations: List[Dict[str, np.ndarray]] = []
        episode_returns_env = [0.0 for _ in range(num_envs)]
        all_episode_returns: List[float] = []

        self.actor_critic.eval()

        reset_start = time.perf_counter()
        for env in self.envs:
            observations.append(env.reset())
        reset_time += time.perf_counter() - reset_start
        resets += num_envs

        for _ in range(self.config.rollout_steps):
            encode_np_start = time.perf_counter()
            state_np_batch = np.stack(
                [encode_observation(obs, self._env_spec) for obs in observations],
                axis=0,
            )
            mask_np_batch = np.stack(
                [obs["action_mask"].astype(bool) for obs in observations],
                axis=0,
            )
            encode_numpy_time += time.perf_counter() - encode_np_start

            tensor_start = time.perf_counter()
            state_batch = torch.from_numpy(state_np_batch).to(self.device)
            mask_batch = torch.from_numpy(mask_np_batch).to(self.device)
            encode_tensor_time += time.perf_counter() - tensor_start

            policy_forward_start = time.perf_counter()
            with torch.no_grad():
                logits_batch, value_batch = self.actor_critic(state_batch)
            policy_forward_time += time.perf_counter() - policy_forward_start

            # 应用动作掩码，避免采样明显无效的动作。
            mask_any = mask_batch.any(dim=1, keepdim=True)
            masked_logits_batch = logits_batch.masked_fill(~mask_batch, -1e9)
            masked_logits_batch = torch.where(mask_any, masked_logits_batch, logits_batch)

            policy_sample_start = time.perf_counter()
            with torch.no_grad():
                log_probs_batch = torch.log_softmax(masked_logits_batch, dim=-1)
                probs_batch = torch.exp(log_probs_batch)
                action_batch = torch.multinomial(probs_batch, num_samples=1).squeeze(-1)
                log_prob_batch = log_probs_batch.gather(-1, action_batch.unsqueeze(-1)).squeeze(-1)
            policy_sample_time += time.perf_counter() - policy_sample_start

            for idx, env in enumerate(self.envs):
                env_start = time.perf_counter()
                result = env.step(int(action_batch[idx].item()))
                env_time += time.perf_counter() - env_start

                reward = torch.tensor(result.reward, dtype=torch.float32, device=self.device)
                done = torch.tensor(result.done, dtype=torch.float32, device=self.device)

                buffer_start = time.perf_counter()
                self.buffer.add(
                    state_batch[idx].detach(),
                    action_batch[idx].detach(),
                    log_prob_batch[idx].detach(),
                    reward.detach(),
                    done.detach(),
                    value_batch[idx].detach(),
                    masked_logits_batch[idx].detach(),
                    mask_batch[idx].detach(),
                )
                buffer_time += time.perf_counter() - buffer_start

                episode_returns_env[idx] += float(result.reward)
                observations[idx] = result.observation

                if result.done:
                    all_episode_returns.append(episode_returns_env[idx])
                    episode_returns_env[idx] = 0.0
                    reset_start = time.perf_counter()
                    observations[idx] = env.reset()
                    reset_time += time.perf_counter() - reset_start
                    resets += 1

            self.global_step += num_envs

        encode_np_start = time.perf_counter()
        bootstrap_np_batch = np.stack(
            [encode_observation(obs, self._env_spec) for obs in observations],
            axis=0,
        )
        encode_numpy_time += time.perf_counter() - encode_np_start

        tensor_start = time.perf_counter()
        bootstrap_states = torch.from_numpy(bootstrap_np_batch).to(self.device)
        encode_tensor_time += time.perf_counter() - tensor_start

        with torch.no_grad():
            _, bootstrap_values = self.actor_critic(bootstrap_states)
        self._last_value_buffer = bootstrap_values.detach()

        self.actor_critic.train()

        encode_time = encode_numpy_time + encode_tensor_time
        policy_time = policy_forward_time + policy_sample_time
        steps = float(self.config.rollout_steps * num_envs)

        self.rollout_stats = {
            "timing/rollout_encode_seconds": encode_time,
            "timing/rollout_encode_numpy_seconds": encode_numpy_time,
            "timing/rollout_encode_tensor_seconds": encode_tensor_time,
            "timing/rollout_policy_seconds": policy_time,
            "timing/rollout_policy_forward_seconds": policy_forward_time,
            "timing/rollout_policy_sample_seconds": policy_sample_time,
            "timing/rollout_env_seconds": env_time,
            "timing/rollout_buffer_seconds": buffer_time,
            "timing/rollout_reset_seconds": reset_time,
            "stats/rollout_steps": steps,
            "stats/rollout_resets": float(resets),
            "stats/rollout_episodes": float(len(all_episode_returns)),
        }

        return all_episode_returns

    def _compute_advantages(
        self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        num_envs = len(self.envs)
        rollout_steps = self.config.rollout_steps

        rewards = rewards.reshape(rollout_steps, num_envs)
        dones = dones.reshape(rollout_steps, num_envs)
        values = values.reshape(rollout_steps, num_envs)

        advantages = torch.zeros_like(values, device=self.device)
        last_advantage = torch.zeros(num_envs, device=self.device)
        next_values = self._last_value_buffer.to(self.device)

        for step in reversed(range(rollout_steps)):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_values * mask - values[step]
            last_advantage = (
                delta + self.config.gamma * self.config.gae_lambda * mask * last_advantage
            )
            advantages[step] = last_advantage
            next_values = values[step]

        returns = advantages + values
        return {
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }

    def update(self) -> Dict[str, float]:
        (
            states,
            actions,
            old_log_probs,
            rewards,
            dones,
            values,
            old_logits,
            action_masks,
        ) = self.buffer.stack()

        targets = self._compute_advantages(rewards, dones, values)
        advantages = targets["advantages"]
        returns = targets["returns"]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if self.value_normalizer is not None:
            self.value_normalizer.update(returns)

        losses: Dict[str, float] = {}
        dataset_size = states.size(0)

        for _ in range(self.config.num_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.config.mini_batch_size):
                batch_idx = indices[start : start + self.config.mini_batch_size]
                batch_states = states[batch_idx].to(self.device)
                batch_actions = actions[batch_idx].to(self.device)
                batch_old_log_probs = old_log_probs[batch_idx].to(self.device)
                batch_returns = returns[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)
                batch_old_logits = old_logits[batch_idx].to(self.device)
                batch_action_masks = action_masks[batch_idx].to(self.device)

                logits, values_pred = self.actor_critic(batch_states)

                # 训练阶段同样施加动作掩码，保持分布一致。
                mask_any = batch_action_masks.any(dim=1, keepdim=True)
                masked_logits = logits.masked_fill(~batch_action_masks, -1e9)
                masked_logits = torch.where(mask_any, masked_logits, logits)

                masked_old_logits = batch_old_logits  # 已在 rollout 阶段遮罩

                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                old_probs = torch.softmax(masked_old_logits, dim=-1)
                new_probs = torch.softmax(masked_logits, dim=-1)
                eps = 1e-8
                kl = torch.sum(
                    old_probs
                    * (
                        torch.log(old_probs.clamp_min(eps))
                        - torch.log(new_probs.clamp_min(eps))
                    ),
                    dim=-1,
                ).mean()

                if self.value_normalizer is not None:
                    norm_returns = self.value_normalizer.normalize(batch_returns)
                    norm_values = self.value_normalizer.normalize(values_pred)
                    value_loss = nn.functional.mse_loss(norm_values, norm_returns)
                else:
                    value_loss = nn.functional.mse_loss(values_pred, batch_returns)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.kl_coef * kl
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                losses["policy_loss"] = float(policy_loss.item())
                losses["value_loss"] = float(value_loss.item())
                losses["entropy"] = float(entropy.item())
                losses["kl_divergence"] = float(kl.item())

        return losses

    def log(self, scalars: Dict[str, float], step: int) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)

    def save_checkpoint(self, path: str, *, update: int | None = None) -> Path:
        path_obj = Path(path)
        if update is not None:
            suffix = path_obj.suffix
            path_obj = path_obj.with_name(f"{path_obj.stem}_update_{int(update):04d}{suffix}")
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.actor_critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grid_spec": asdict(self._env_spec),
            "config": asdict(self.config),
            "global_step": self.global_step,
            "update": update,
            "value_normalizer_state": (
                self.value_normalizer.state_dict() if self.value_normalizer is not None else None
            ),
            "environment": {
                "min_obstacles": self.primary_env.min_obstacles,
                "max_obstacles": self.primary_env.max_obstacles,
                "obstacle_length_range": self.primary_env.obstacle_length_range,
                "obstacle_width_range": self.primary_env.obstacle_width_range,
                "smoothness_penalty": self.primary_env.smoothness_penalty,
                "base_smoothness_penalty": self.primary_env.base_smoothness_penalty,
                "move_limit_penalty_multiplier": self.primary_env.move_limit_penalty_multiplier,
                "lateral_reference": self.primary_env.lateral_reference,
                "reference_penalty": self.primary_env.reference_penalty,
                "max_slope": self.primary_env.max_slope,
                "slope_penalty": self.primary_env.slope_penalty,
                "max_slope_penalty": self.primary_env.max_slope_penalty,
                "collision_penalty": self.primary_env.collision_penalty,
                "terminal_reward": self.primary_env.terminal_reward,
                "base_smoothness_penalty": self.primary_env.base_smoothness_penalty,
                "non_goal_penalty": self.primary_env.non_goal_penalty,
                "no_valid_action_penalty": self.primary_env.no_valid_action_penalty,
                "lateral_move_limit": self.primary_env.lateral_move_limit,
                "coarse_collision_inflation": self.primary_env.coarse_collision_inflation,
                "fine_collision_inflation": self.primary_env.fine_collision_inflation,
                "interpolation_points": self.primary_env.interpolation_points,
                "vehicle_length": self.primary_env.vehicle_length,
                "vehicle_width": self.primary_env.vehicle_width,
                "start_clear_fraction": self.primary_env.start_clear_fraction,
                "scenario_pool_size": self.primary_env.scenario_pool_size,
                "scenario_top_k": self.primary_env.scenario_top_k,
                "scenario_min_obstacles": self.primary_env.scenario_min_obstacles,
                "scenario_max_avg_cost": self.primary_env.scenario_max_avg_cost,
                "scenario_max_attempts": self.primary_env.scenario_max_attempts,
                "scenario_dataset_path": self.primary_env.scenario_dataset_path,
            },
        }
        torch.save(payload, path_obj)
        return path_obj

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def load_value_normalizer_state(self, state: dict | None) -> None:
        if self.value_normalizer is not None and state:
            self.value_normalizer.load_state_dict(state)
