from __future__ import annotations

import torch

from neural_astar.models import GuidanceEncoder


def test_guidance_encoder_unet_yaw_smoke():
    model = GuidanceEncoder(base_channels=8, arch="unet", use_pose_yaw_cond=True)
    occ = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 2, 3] = 1.0
    goal[:, :, 25, 20] = 1.0
    start_yaw = torch.tensor([0.0, 1.25], dtype=torch.float32)
    goal_yaw = torch.tensor([0.5, -0.75], dtype=torch.float32)

    out = model(occ, start, goal, start_yaw=start_yaw, goal_yaw=goal_yaw)

    assert out.logits_cost.shape == (2, 1, 32, 32)
    assert out.cost_map.shape == (2, 1, 32, 32)
    assert torch.isfinite(out.cost_map).all()
    assert model.in_channels == 7
    assert model.orientation_bins == 1


def test_guidance_encoder_orientation_volume_smoke():
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet",
        use_pose_yaw_cond=True,
        orientation_bins=8,
    )
    occ = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 3, 4] = 1.0
    goal[:, :, 20, 18] = 1.0

    out = model(
        occ,
        start,
        goal,
        start_yaw=torch.tensor([0.25], dtype=torch.float32),
        goal_yaw=torch.tensor([-0.5], dtype=torch.float32),
    )

    assert out.logits_cost.shape == (1, 8, 24, 24)
    assert out.cost_map.shape == (1, 8, 24, 24)
    assert torch.isfinite(out.cost_map).all()


def test_guidance_encoder_legacy_cnn_smoke():
    model = GuidanceEncoder(base_channels=8, arch="legacy_cnn", use_pose_yaw_cond=False)
    occ = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 1, 1] = 1.0
    goal[:, :, 14, 14] = 1.0

    out = model(occ, start, goal)

    assert out.logits_cost.shape == (1, 1, 16, 16)
    assert out.cost_map.shape == (1, 1, 16, 16)


def test_guidance_encoder_residual_output_mode_masks_obstacles_to_zero():
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
    )
    occ = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    occ[:, :, 7, 9] = 1.0
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 1, 1] = 1.0
    goal[:, :, 14, 14] = 1.0

    out = model(occ, start, goal)

    assert out.logits_cost.shape == (1, 1, 16, 16)
    assert out.cost_map.shape == (1, 1, 16, 16)
    assert torch.isfinite(out.cost_map).all()
    assert float(out.cost_map.min()) >= 0.0
    assert out.cost_map[0, 0, 7, 9] == 0.0


def test_guidance_encoder_unet_transformer_smoke():
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet_transformer",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        predict_confidence=True,
        predict_residual_scale=True,
        transformer_depth=1,
        transformer_heads=4,
    )
    occ = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 2, 3] = 1.0
    goal[:, :, 25, 20] = 1.0

    out = model(occ, start, goal)

    assert out.logits_cost.shape == (1, 1, 32, 32)
    assert out.cost_map.shape == (1, 1, 32, 32)
    assert out.confidence_map is not None
    assert out.scale_map is not None
    assert torch.isfinite(out.cost_map).all()


def test_guidance_encoder_unet_transformer_v2_smoke():
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet_transformer_v2",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        predict_confidence=True,
        predict_residual_scale=True,
        transformer_depth=2,
        transformer_heads=4,
    )
    occ = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 2, 3] = 1.0
    goal[:, :, 25, 20] = 1.0

    out = model(occ, start, goal)

    assert out.logits_cost.shape == (1, 1, 32, 32)
    assert out.cost_map.shape == (1, 1, 32, 32)
    assert out.confidence_map is not None
    assert out.scale_map is not None
    assert torch.isfinite(out.cost_map).all()
    window_block = model.transformer_blocks.blocks[0]["window"]
    assert window_block.relative_position_bias_table.shape[1] == 4
    assert window_block.shift_size == 0
    assert model.transformer_blocks.blocks[1]["window"].shift_size == 2


def test_guidance_encoder_unet_transformer_v3_smoke():
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet_transformer_v3",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        predict_confidence=True,
        predict_residual_scale=True,
        transformer_depth=2,
        transformer_heads=4,
    )
    occ = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    start = torch.zeros_like(occ)
    goal = torch.zeros_like(occ)
    start[:, :, 2, 3] = 1.0
    goal[:, :, 25, 20] = 1.0

    out = model(occ, start, goal)

    assert out.logits_cost.shape == (1, 1, 32, 32)
    assert out.cost_map.shape == (1, 1, 32, 32)
    assert out.confidence_map is not None
    assert out.scale_map is not None
    assert torch.isfinite(out.cost_map).all()
    assert model.skip_gate3 is not None
    assert model.skip_gate2 is not None
    assert model.skip_gate1 is not None
