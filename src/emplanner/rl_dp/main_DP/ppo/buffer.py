from __future__ import annotations

import math
from typing import List, Tuple

import torch


class RolloutBuffer:
    """缓存 PPO 采样得到的序列。"""

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []
        self.action_masks: List[torch.Tensor] = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.logits.append(logits)
        self.action_masks.append(action_mask)

    def stack(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.stack(self.rewards),
            torch.stack(self.dones),
            torch.stack(self.values),
            torch.stack(self.logits),
            torch.stack(self.action_masks),
        )

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logits.clear()
        self.action_masks.clear()


class RunningNormalizer:
    """跟踪标量目标的滑动均值/方差，用于稳定训练。"""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.eps = float(epsilon)
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().view(-1).to("cpu")
        if flat.numel() == 0:
            return
        batch_count = float(flat.numel())
        batch_mean = float(flat.mean().item())
        batch_var = float(flat.var(unbiased=False).item())

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        adjusted_total = max(total_count, 1e-6)

        new_mean = self.mean + delta * batch_count / adjusted_total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / adjusted_total

        self.mean = new_mean
        self.var = m2 / adjusted_total
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        std = math.sqrt(self.var + self.eps)
        mean_tensor = values.new_full((), self.mean)
        std_tensor = values.new_full((), std)
        return (values - mean_tensor) / std_tensor

    def state_dict(self) -> dict:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "count": float(self.count),
            "eps": float(self.eps),
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.mean = float(state.get("mean", self.mean))
        self.var = max(float(state.get("var", self.var)), 0.0)
        self.count = max(float(state.get("count", self.count)), 1e-8)
        self.eps = float(state.get("eps", self.eps))
