"""Guidance encoder for Route A.

Supports two architectures:
- ``unet``: default yaw-aware encoder-decoder used for new training.
- ``unet_transformer``: U-Net with lightweight transformer blocks in the bottleneck.
- ``unet_transformer_v2``: U-Net with multi-scale window attention and goal-conditioned cross-attention.
- ``unet_transformer_v3``: ``v2`` plus decoder-side gated skip fusion.
- ``legacy_cnn``: compatibility path for older checkpoints.

Input convention:
- Base channels are ``[occ_map, start_map, goal_map]``.
- If yaw conditioning is enabled, four extra channels are appended:
  ``start_sin, start_cos, goal_sin, goal_cos``.
- Optional extra input maps can be appended after the base channels.
- All tensors are ``[B, 1, H, W]`` except yaw inputs, which are scalar per batch.

Output convention:
- ``cost_map`` is a guidance tensor ``[B, K, H, W]`` where lower is better.
- Obstacle cells are masked to high cost through ``occ_map``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidanceEncoderOutput(NamedTuple):
    logits_cost: torch.Tensor
    cost_map: torch.Tensor
    logits_confidence: Optional[torch.Tensor] = None
    confidence_map: Optional[torch.Tensor] = None
    variance_map: Optional[torch.Tensor] = None
    logits_scale: Optional[torch.Tensor] = None
    scale_map: Optional[torch.Tensor] = None


def _group_norm_channels(num_channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_norm_channels(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SkipGate(nn.Module):
    def __init__(self, decoder_channels: int, skip_channels: int):
        super().__init__()
        groups = _group_norm_channels(skip_channels)
        self.decoder_proj = nn.Conv2d(decoder_channels, skip_channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, skip_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate_logits = self.gate(torch.cat([self.decoder_proj(x_up), skip], dim=1))
        gate = 1.0 + 0.5 * torch.tanh(gate_logits)
        return skip * gate


class BottleneckTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if int(channels) % int(num_heads) != 0:
            raise ValueError(
                f"channels must be divisible by num_heads, got channels={channels}, num_heads={num_heads}"
            )
        if float(mlp_ratio) <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        hidden = int(round(float(channels) * float(mlp_ratio)))
        self.pos_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=int(num_heads),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x + self.pos_conv(x)
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


def _window_partition(
    x: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    ws = int(max(1, window_size))
    pad_h = (ws - (h % ws)) % ws
    pad_w = (ws - (w % ws)) % ws
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, c, hp // ws, ws, wp // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws * ws, c)
    return x, (h, w, hp, wp)


def _window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    meta: tuple[int, int, int, int],
    batch_size: int,
    channels: int,
) -> torch.Tensor:
    h, w, hp, wp = meta
    ws = int(max(1, window_size))
    x = windows.view(batch_size, hp // ws, wp // ws, ws, ws, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, channels, hp, wp)
    return x[:, :, :h, :w]


def _build_relative_position_index(window_size: int) -> torch.Tensor:
    ws = int(window_size)
    coords_h = torch.arange(ws)
    coords_w = torch.arange(ws)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]
    coords_flat = coords.reshape(2, ws * ws)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [L, L, 2]
    relative_coords[:, :, 0] += ws - 1
    relative_coords[:, :, 1] += ws - 1
    relative_coords[:, :, 0] *= (2 * ws - 1)
    return relative_coords.sum(-1).to(torch.long)


class WindowTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if int(channels) % int(num_heads) != 0:
            raise ValueError(
                f"channels must be divisible by num_heads, got channels={channels}, num_heads={num_heads}"
            )
        if int(window_size) <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if int(shift_size) < 0:
            raise ValueError(f"shift_size must be non-negative, got {shift_size}")
        if int(shift_size) >= int(window_size):
            raise ValueError(
                f"shift_size must be smaller than window_size, got shift_size={shift_size}, "
                f"window_size={window_size}"
            )
        if float(mlp_ratio) <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        hidden = int(round(float(channels) * float(mlp_ratio)))
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(channels) // int(num_heads)
        self.scale = self.head_dim ** -0.5
        self.pos_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads)
        )
        self.register_buffer(
            "relative_position_index",
            _build_relative_position_index(self.window_size),
            persistent=False,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _build_attention_mask(
        self,
        meta: tuple[int, int, int, int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.shift_size <= 0:
            return None

        _, _, hp, wp = meta
        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros((1, 1, hp, wp), device=device, dtype=torch.float32)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, :, h_slice, w_slice] = float(cnt)
                cnt += 1

        mask_windows, _ = _window_partition(img_mask, ws)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        x = x + self.pos_conv(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        windows, meta = _window_partition(x, self.window_size)
        attn_in = self.norm1(windows)
        num_windows, num_tokens, channels = attn_in.shape
        qkv = self.qkv(attn_in).reshape(
            num_windows,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(num_tokens, num_tokens, self.num_heads)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)
        attn_mask = self._build_attention_mask(meta, attn.device)
        if attn_mask is not None:
            windows_per_batch = (meta[2] // self.window_size) * (meta[3] // self.window_size)
            attn = attn.view(b, windows_per_batch, self.num_heads, num_tokens, num_tokens)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)
        attn = torch.softmax(attn, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(num_windows, num_tokens, channels)
        windows = windows + self.proj(attn_out)
        windows = windows + self.mlp(self.norm2(windows))
        out = _window_unpartition(windows, self.window_size, meta, batch_size=b, channels=c)
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return out


class GoalCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if int(channels) % int(num_heads) != 0:
            raise ValueError(
                f"channels must be divisible by num_heads, got channels={channels}, num_heads={num_heads}"
            )
        hidden = int(round(float(channels) * float(mlp_ratio)))
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=int(num_heads),
            batch_first=True,
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.film = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )

    @staticmethod
    def _context_from_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled_mask = F.adaptive_max_pool2d(mask, output_size=x.shape[-2:])
        pooled_mask = pooled_mask.expand(-1, x.shape[1], -1, -1)
        denom = pooled_mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return (x * pooled_mask).sum(dim=(2, 3)) / denom

    @staticmethod
    def _dense_context_tokens(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        kernel = 3 if min(h, w) <= 8 else 5
        pooled_mask = F.adaptive_max_pool2d(mask, output_size=(h, w))
        if kernel > 1:
            pooled_mask = F.avg_pool2d(
                pooled_mask,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
        pooled_mask = pooled_mask / pooled_mask.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled_mask = pooled_mask.clamp(0.0, 1.0)

        weighted = x * pooled_mask.expand(-1, x.shape[1], -1, -1)
        dense_tokens = weighted.flatten(2).transpose(1, 2)
        global_ctx = GoalCrossAttentionBlock._context_from_mask(x, mask)
        dense_tokens = dense_tokens + pooled_mask.flatten(2).transpose(1, 2) * global_ctx.unsqueeze(1)
        return dense_tokens, global_ctx

    def forward(
        self,
        x: torch.Tensor,
        start_map: torch.Tensor,
        goal_map: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        start_tokens, start_ctx = self._dense_context_tokens(x, start_map)
        goal_tokens, goal_ctx = self._dense_context_tokens(x, goal_map)
        context = torch.cat(
            [
                start_tokens,
                goal_tokens,
                start_ctx.unsqueeze(1),
                goal_ctx.unsqueeze(1),
            ],
            dim=1,
        )

        attn_out, _ = self.attn(
            self.query_norm(tokens),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.token_mlp(self.query_norm(tokens))

        film = self.film(torch.cat([start_ctx, goal_ctx], dim=1))
        scale, bias = torch.chunk(film, 2, dim=1)
        scale = torch.tanh(scale).view(b, c, 1, 1)
        bias = bias.view(b, c, 1, 1)

        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x * (1.0 + scale) + bias


class MultiScaleTransformerStage(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        depth: int = 2,
        window_size: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if int(depth) <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "window": WindowTransformerBlock(
                            channels=channels,
                            num_heads=num_heads,
                            window_size=window_size,
                            shift_size=(0 if (idx % 2 == 0 or window_size <= 1) else max(1, window_size // 2)),
                            mlp_ratio=mlp_ratio,
                        ),
                        "goal": GoalCrossAttentionBlock(
                            channels=channels,
                            num_heads=num_heads,
                            mlp_ratio=max(1.0, mlp_ratio / 2.0),
                        ),
                    }
                )
                for idx in range(int(depth))
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        start_map: torch.Tensor,
        goal_map: torch.Tensor,
    ) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block["window"](out)
            out = block["goal"](out, start_map, goal_map)
        return out


class GuidanceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        base_channels: int = 32,
        obstacle_cost: float = 1.0,
        arch: str = "unet",
        use_pose_yaw_cond: bool = True,
        orientation_bins: int = 1,
        output_mode: str = "cost_map",
        residual_target_transform: str = "none",
        predict_confidence: bool = False,
        confidence_head_kernel: int = 1,
        predict_residual_scale: bool = False,
        residual_scale_max: float = 2.0,
        transformer_depth: int = 2,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        extra_input_channels: int | None = None,
        clearance_input_clip_distance: float = 0.0,
    ):
        super().__init__()
        if arch not in {"unet", "unet_transformer", "unet_transformer_v2", "unet_transformer_v3", "legacy_cnn"}:
            raise ValueError(f"Unsupported arch: {arch}")
        if output_mode not in {"cost_map", "residual_heuristic"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}")
        if residual_target_transform not in {"none", "log1p"}:
            raise ValueError(
                f"Unsupported residual_target_transform: {residual_target_transform}"
            )
        if bool(predict_residual_scale) and output_mode != "residual_heuristic":
            raise ValueError("predict_residual_scale requires output_mode=residual_heuristic")
        if int(confidence_head_kernel) <= 0 or int(confidence_head_kernel) % 2 == 0:
            raise ValueError(
                f"confidence_head_kernel must be a positive odd integer, got {confidence_head_kernel}"
            )
        if float(residual_scale_max) <= 0.0:
            raise ValueError(f"residual_scale_max must be positive, got {residual_scale_max}")

        self.arch = str(arch)
        self.use_pose_yaw_cond = bool(use_pose_yaw_cond)
        self.obstacle_cost = float(obstacle_cost)
        self.orientation_bins = int(max(1, orientation_bins))
        self.output_mode = str(output_mode)
        self.residual_target_transform = str(residual_target_transform)
        self.predict_confidence = bool(predict_confidence)
        self.confidence_head_kernel = int(confidence_head_kernel)
        self.predict_residual_scale = bool(predict_residual_scale)
        self.residual_scale_max = float(residual_scale_max)
        self.transformer_depth = int(transformer_depth)
        self.transformer_heads = int(transformer_heads)
        self.transformer_mlp_ratio = float(transformer_mlp_ratio)
        self.clearance_input_clip_distance = float(clearance_input_clip_distance)
        base_expected_channels = 3 + (4 if self.use_pose_yaw_cond else 0)
        if extra_input_channels is None:
            inferred_extra_channels = (
                0
                if in_channels is None
                else max(0, int(in_channels) - base_expected_channels)
            )
            self.extra_input_channels = int(inferred_extra_channels)
        else:
            self.extra_input_channels = int(max(0, int(extra_input_channels)))
        expected_channels = base_expected_channels + self.extra_input_channels
        self.in_channels = int(expected_channels if in_channels is None else in_channels)
        if self.in_channels != expected_channels:
            raise ValueError(
                "GuidanceEncoder input channel mismatch: "
                f"in_channels={self.in_channels}, expected={expected_channels} "
                f"(base={base_expected_channels}, extra={self.extra_input_channels})"
            )
        self.base_channels = int(base_channels)

        if self.arch == "legacy_cnn":
            if self.predict_confidence or self.predict_residual_scale:
                raise ValueError(
                    "predict_confidence and predict_residual_scale are only supported for arch=unet"
                )
            self.net = self._build_legacy_cnn(
                self.in_channels,
                self.base_channels,
                self.orientation_bins,
            )
        else:
            c1 = self.base_channels
            c2 = self.base_channels * 2
            c3 = self.base_channels * 4
            c4 = self.base_channels * 8

            self.enc1 = ConvBlock(self.in_channels, c1)
            self.enc2 = DownBlock(c1, c2)
            self.enc3 = DownBlock(c2, c3)
            self.bottleneck = DownBlock(c3, c4)
            if self.arch == "unet_transformer":
                if self.transformer_depth <= 0:
                    raise ValueError(
                        f"transformer_depth must be positive for arch=unet_transformer, got {self.transformer_depth}"
                    )
                self.transformer_blocks = nn.Sequential(
                    *[
                        BottleneckTransformerBlock(
                            channels=c4,
                            num_heads=self.transformer_heads,
                            mlp_ratio=self.transformer_mlp_ratio,
                        )
                        for _ in range(self.transformer_depth)
                    ]
                )
                self.mid_transformer_blocks = None
            elif self.arch in {"unet_transformer_v2", "unet_transformer_v3"}:
                if self.transformer_depth <= 0:
                    raise ValueError(
                        f"transformer_depth must be positive for arch={self.arch}, got "
                        f"{self.transformer_depth}"
                    )
                self.mid_transformer_blocks = MultiScaleTransformerStage(
                    channels=c3,
                    num_heads=self.transformer_heads,
                    depth=max(1, self.transformer_depth),
                    window_size=4,
                    mlp_ratio=self.transformer_mlp_ratio,
                )
                self.transformer_blocks = MultiScaleTransformerStage(
                    channels=c4,
                    num_heads=self.transformer_heads,
                    depth=max(1, self.transformer_depth),
                    window_size=4,
                    mlp_ratio=self.transformer_mlp_ratio,
                )
            else:
                self.transformer_blocks = None
                self.mid_transformer_blocks = None
            self.dec3 = UpBlock(c4, c3, c3)
            self.dec2 = UpBlock(c3, c2, c2)
            self.dec1 = UpBlock(c2, c1, c1)
            if self.arch == "unet_transformer_v3":
                self.skip_gate3 = SkipGate(c4, c3)
                self.skip_gate2 = SkipGate(c3, c2)
                self.skip_gate1 = SkipGate(c2, c1)
            else:
                self.skip_gate3 = None
                self.skip_gate2 = None
                self.skip_gate1 = None
            self.head = nn.Conv2d(c1, self.orientation_bins, kernel_size=1)
            if self.predict_confidence:
                if self.confidence_head_kernel == 1:
                    self.conf_head = nn.Conv2d(c1, self.orientation_bins, kernel_size=1)
                else:
                    groups = _group_norm_channels(c1)
                    self.conf_head = nn.Sequential(
                        nn.Conv2d(
                            c1,
                            c1,
                            kernel_size=self.confidence_head_kernel,
                            padding=self.confidence_head_kernel // 2,
                            bias=False,
                        ),
                        nn.GroupNorm(groups, c1),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(c1, self.orientation_bins, kernel_size=1),
                    )
            else:
                self.conf_head = None
            if self.predict_residual_scale:
                self.scale_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(c1, self.orientation_bins),
                )
            else:
                self.scale_head = None

    @staticmethod
    def _build_legacy_cnn(
        in_channels: int,
        base_channels: int,
        orientation_bins: int,
    ) -> nn.Sequential:
        half = max(4, base_channels // 2)
        return nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, half, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, int(max(1, orientation_bins)), kernel_size=1),
        )

    @staticmethod
    def _normalize_batch_yaw(
        yaw: Optional[Union[torch.Tensor, float]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if yaw is None:
            return torch.zeros(batch_size, device=device, dtype=dtype)

        if isinstance(yaw, (float, int)):
            return torch.full((batch_size,), float(yaw), device=device, dtype=dtype)

        yaw_t = yaw.to(device=device, dtype=dtype)
        if yaw_t.ndim == 0:
            return yaw_t.expand(batch_size)
        if yaw_t.ndim == 2 and yaw_t.shape[1] == 1:
            yaw_t = yaw_t[:, 0]
        if yaw_t.ndim != 1 or yaw_t.shape[0] != batch_size:
            raise ValueError(f"yaw must broadcast to [B], got shape={tuple(yaw_t.shape)}")
        return yaw_t

    def _build_input(
        self,
        occ_map: torch.Tensor,
        start_map: torch.Tensor,
        goal_map: torch.Tensor,
        start_yaw: Optional[Union[torch.Tensor, float]],
        goal_yaw: Optional[Union[torch.Tensor, float]],
        extra_input_maps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = [occ_map, start_map, goal_map]
        batch_size = occ_map.shape[0]
        dtype = occ_map.dtype
        device = occ_map.device
        if self.use_pose_yaw_cond:
            syaw = self._normalize_batch_yaw(start_yaw, batch_size, device, dtype)
            gyaw = self._normalize_batch_yaw(goal_yaw, batch_size, device, dtype)

            syaw_sin = torch.sin(syaw).view(batch_size, 1, 1, 1) * start_map
            syaw_cos = torch.cos(syaw).view(batch_size, 1, 1, 1) * start_map
            gyaw_sin = torch.sin(gyaw).view(batch_size, 1, 1, 1) * goal_map
            gyaw_cos = torch.cos(gyaw).view(batch_size, 1, 1, 1) * goal_map
            x.extend([syaw_sin, syaw_cos, gyaw_sin, gyaw_cos])
        if self.extra_input_channels > 0:
            if extra_input_maps is None:
                extra_input_maps = torch.zeros(
                    (batch_size, self.extra_input_channels, occ_map.shape[-2], occ_map.shape[-1]),
                    dtype=dtype,
                    device=device,
                )
            else:
                extra_input_maps = extra_input_maps.to(device=device, dtype=dtype)
                if extra_input_maps.ndim != 4:
                    raise ValueError(
                        f"extra_input_maps must be [B,C,H,W], got {tuple(extra_input_maps.shape)}"
                    )
                if extra_input_maps.shape[0] != batch_size or extra_input_maps.shape[-2:] != occ_map.shape[-2:]:
                    raise ValueError(
                        "extra_input_maps shape mismatch: "
                        f"got {tuple(extra_input_maps.shape)}, expected batch={batch_size}, "
                        f"hw={tuple(occ_map.shape[-2:])}"
                    )
                if extra_input_maps.shape[1] != self.extra_input_channels:
                    raise ValueError(
                        f"GuidanceEncoder expected extra_input_channels={self.extra_input_channels}, "
                        f"got {extra_input_maps.shape[1]}"
                    )
            x.append(extra_input_maps)
        return torch.cat(x, dim=1)

    def forward(
        self,
        occ_map: torch.Tensor,
        start_map: torch.Tensor,
        goal_map: torch.Tensor,
        start_yaw: Optional[Union[torch.Tensor, float]] = None,
        goal_yaw: Optional[Union[torch.Tensor, float]] = None,
        extra_input_maps: Optional[torch.Tensor] = None,
    ) -> GuidanceEncoderOutput:
        if occ_map.ndim != 4 or start_map.ndim != 4 or goal_map.ndim != 4:
            raise ValueError("Inputs must be [B,1,H,W]")
        if occ_map.shape != start_map.shape or occ_map.shape != goal_map.shape:
            raise ValueError(
                f"Shape mismatch: occ={occ_map.shape}, start={start_map.shape}, goal={goal_map.shape}"
            )

        x = self._build_input(
            occ_map,
            start_map,
            goal_map,
            start_yaw,
            goal_yaw,
            extra_input_maps,
        )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"GuidanceEncoder expected in_channels={self.in_channels}, got input channels={x.shape[1]}"
            )

        if self.arch == "legacy_cnn":
            logits_cost = self.net(x)
            logits_confidence = None
            logits_scale = None
        else:
            s1 = self.enc1(x)
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            if self.mid_transformer_blocks is not None:
                s3 = self.mid_transformer_blocks(s3, start_map, goal_map)
            z = self.bottleneck(s3)
            if isinstance(self.transformer_blocks, MultiScaleTransformerStage):
                z = self.transformer_blocks(z, start_map, goal_map)
            elif self.transformer_blocks is not None:
                z = self.transformer_blocks(z)
            skip3 = self.skip_gate3(z, s3) if self.skip_gate3 is not None else s3
            z = self.dec3(z, skip3)
            skip2 = self.skip_gate2(z, s2) if self.skip_gate2 is not None else s2
            z = self.dec2(z, skip2)
            skip1 = self.skip_gate1(z, s1) if self.skip_gate1 is not None else s1
            z = self.dec1(z, skip1)
            logits_cost = self.head(z)
            logits_confidence = (
                self.conf_head(z)
                if self.conf_head is not None
                else None
            )
            logits_scale = (
                self.scale_head(z).view(z.shape[0], self.orientation_bins, 1, 1)
                if self.scale_head is not None
                else None
            )

        if self.output_mode == "residual_heuristic":
            cost_map = F.softplus(logits_cost)
            occ_mask = occ_map.expand(-1, cost_map.shape[1], -1, -1)
            cost_map = torch.where(
                occ_mask > 0.5,
                torch.zeros_like(cost_map),
                cost_map,
            )
            if logits_confidence is None:
                variance_map = None
                confidence_map = None
            else:
                variance_map = F.softplus(logits_confidence) + 1e-4
                confidence_map = 1.0 / (1.0 + variance_map)
                confidence_map = torch.where(
                    occ_mask > 0.5,
                    torch.zeros_like(confidence_map),
                    confidence_map,
                )
            if logits_scale is None:
                scale_map = None
            else:
                scale_map = self.residual_scale_max * torch.sigmoid(logits_scale)
        else:
            cost_map = torch.sigmoid(logits_cost)
            occ_mask = occ_map.expand(-1, cost_map.shape[1], -1, -1)
            cost_map = torch.where(
                occ_mask > 0.5,
                torch.full_like(cost_map, self.obstacle_cost),
                cost_map,
            )
            variance_map = None
            confidence_map = None
            logits_scale = None
            scale_map = None
        return GuidanceEncoderOutput(
            logits_cost=logits_cost,
            cost_map=cost_map,
            logits_confidence=logits_confidence,
            confidence_map=confidence_map,
            variance_map=variance_map,
            logits_scale=logits_scale,
            scale_map=scale_map,
        )
