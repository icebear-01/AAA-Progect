from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """共享骨干网络，输出策略 logits 与状态价值。"""

    def __init__(
        self,
        occupancy_shape: Tuple[int, int],
        grid_channels: int,
        extra_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.occupancy_shape = occupancy_shape
        self.grid_channels = grid_channels
        self.extra_dim = extra_dim
        self.grid_feature_size = occupancy_shape[0] * occupancy_shape[1] * grid_channels

        # 卷积干路提取空间特征
        self.conv_trunk = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        fc_input_dim = 64 + max(extra_dim, 0)
        self.trunk = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        batch_size = x.size(0)
        grid = x[..., : self.grid_feature_size].view(
            batch_size, self.grid_channels, *self.occupancy_shape
        )
        conv_features = self.conv_trunk(grid).view(batch_size, -1)

        if self.extra_dim > 0:
            extra_features = x[..., self.grid_feature_size :]
            features = torch.cat([conv_features, extra_features], dim=-1)
        else:
            features = conv_features

        features = self.trunk(features)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)

        if squeeze_batch:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value
