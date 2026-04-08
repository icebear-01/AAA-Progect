from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 256
    num_epochs: int = 4
    mini_batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 1e-4
    value_coef: float = 0.75
    entropy_coef: float = 0.1
    kl_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    device: str = "cpu"
    num_envs: int = 4
    log_dir: str = "runs/ppo"
    checkpoint_path: str = "checkpoints/ppo_policy.pt"
    checkpoint_interval: int | None = None
    normalize_value_targets: bool = True