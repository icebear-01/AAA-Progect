from __future__ import annotations

import numpy as np
import torch

from neural_astar.api.guidance_infer import infer_cost_map, infer_cost_volume
from neural_astar.models import GuidanceEncoder


def test_infer_cost_map_smoke(tmp_path):
    model = GuidanceEncoder(base_channels=8)
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
            },
        },
        ckpt_path,
    )

    h, w = 16, 16
    occ = np.zeros((h, w), dtype=np.float32)
    occ[8, 8] = 1.0

    cost = infer_cost_map(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(14, 14),
        start_yaw=0.25,
        goal_yaw=-0.5,
        device="cpu",
    )

    assert cost.shape == (h, w)
    assert np.isfinite(cost).all()
    assert (cost >= 0.0).all() and (cost <= 1.0).all()
    assert cost[8, 8] == 1.0


def test_infer_cost_volume_smoke(tmp_path):
    model = GuidanceEncoder(base_channels=8, orientation_bins=6)
    ckpt_path = tmp_path / "model_volume.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((12, 12), dtype=np.float32)
    occ[5, 6] = 1.0

    cost = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(10, 10),
        start_yaw=0.0,
        goal_yaw=1.0,
        device="cpu",
    )

    assert cost.shape == (6, 12, 12)
    assert np.isfinite(cost).all()
    assert np.all(cost[:, 5, 6] == 1.0)


def test_infer_cost_volume_decodes_log_residual_predictions(tmp_path):
    model = GuidanceEncoder(
        base_channels=8,
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        residual_target_transform="log1p",
    )
    ckpt_path = tmp_path / "model_residual.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "residual_target_transform": model.residual_target_transform,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((10, 10), dtype=np.float32)
    occ[4, 4] = 1.0

    pred = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(8, 8),
        device="cpu",
    )

    assert pred.shape == (1, 10, 10)
    assert np.isfinite(pred).all()
    assert np.all(pred >= 0.0)
    assert pred[0, 4, 4] == 1.0


def test_infer_cost_volume_supports_residual_model_with_confidence_head(tmp_path):
    model = GuidanceEncoder(
        base_channels=8,
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        residual_target_transform="log1p",
        predict_confidence=True,
        confidence_head_kernel=3,
    )
    ckpt_path = tmp_path / "model_residual_conf.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "residual_target_transform": model.residual_target_transform,
                "predict_confidence": model.predict_confidence,
                "confidence_head_kernel": model.confidence_head_kernel,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((10, 10), dtype=np.float32)
    pred = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(8, 8),
        device="cpu",
    )

    assert pred.shape == (1, 10, 10)
    assert np.isfinite(pred).all()


def test_infer_cost_volume_supports_residual_model_with_scale_head(tmp_path):
    model = GuidanceEncoder(
        base_channels=8,
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        residual_target_transform="log1p",
        predict_residual_scale=True,
        residual_scale_max=2.0,
    )
    ckpt_path = tmp_path / "model_residual_scale.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "residual_target_transform": model.residual_target_transform,
                "predict_residual_scale": model.predict_residual_scale,
                "residual_scale_max": model.residual_scale_max,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((10, 10), dtype=np.float32)
    pred = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(8, 8),
        device="cpu",
    )

    assert pred.shape == (1, 10, 10)
    assert np.isfinite(pred).all()
    assert np.all(pred >= 0.0)


def test_infer_cost_volume_supports_unet_transformer_checkpoint(tmp_path):
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet_transformer",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        residual_target_transform="log1p",
        predict_confidence=True,
        confidence_head_kernel=3,
        predict_residual_scale=True,
        residual_scale_max=2.0,
        transformer_depth=1,
        transformer_heads=4,
    )
    ckpt_path = tmp_path / "model_unet_transformer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "residual_target_transform": model.residual_target_transform,
                "predict_confidence": model.predict_confidence,
                "confidence_head_kernel": model.confidence_head_kernel,
                "predict_residual_scale": model.predict_residual_scale,
                "residual_scale_max": model.residual_scale_max,
                "transformer_depth": model.transformer_depth,
                "transformer_heads": model.transformer_heads,
                "transformer_mlp_ratio": model.transformer_mlp_ratio,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((10, 10), dtype=np.float32)
    pred = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(8, 8),
        device="cpu",
    )

    assert pred.shape == (1, 10, 10)
    assert np.isfinite(pred).all()


def test_infer_cost_volume_supports_unet_transformer_v2_checkpoint(tmp_path):
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet_transformer_v2",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        residual_target_transform="log1p",
        predict_confidence=True,
        confidence_head_kernel=3,
        predict_residual_scale=True,
        residual_scale_max=2.0,
        transformer_depth=1,
        transformer_heads=4,
    )
    ckpt_path = tmp_path / "model_unet_transformer_v2.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "residual_target_transform": model.residual_target_transform,
                "predict_confidence": model.predict_confidence,
                "confidence_head_kernel": model.confidence_head_kernel,
                "predict_residual_scale": model.predict_residual_scale,
                "residual_scale_max": model.residual_scale_max,
                "transformer_depth": model.transformer_depth,
                "transformer_heads": model.transformer_heads,
                "transformer_mlp_ratio": model.transformer_mlp_ratio,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((10, 10), dtype=np.float32)
    pred = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(8, 8),
        device="cpu",
    )

    assert pred.shape == (1, 10, 10)
    assert np.isfinite(pred).all()


def test_infer_cost_volume_supports_unet_transformer_v3_checkpoint(tmp_path):
    model = GuidanceEncoder(
        base_channels=8,
        arch="unet_transformer_v3",
        use_pose_yaw_cond=False,
        output_mode="residual_heuristic",
        residual_target_transform="log1p",
        predict_confidence=True,
        confidence_head_kernel=3,
        predict_residual_scale=True,
        residual_scale_max=2.0,
        transformer_depth=2,
        transformer_heads=4,
    )
    ckpt_path = tmp_path / "model_unet_transformer_v3.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": {
                "in_channels": model.in_channels,
                "base_channels": 8,
                "obstacle_cost": 1.0,
                "arch": model.arch,
                "use_pose_yaw_cond": model.use_pose_yaw_cond,
                "orientation_bins": model.orientation_bins,
                "output_mode": model.output_mode,
                "residual_target_transform": model.residual_target_transform,
                "predict_confidence": model.predict_confidence,
                "confidence_head_kernel": model.confidence_head_kernel,
                "predict_residual_scale": model.predict_residual_scale,
                "residual_scale_max": model.residual_scale_max,
                "transformer_depth": model.transformer_depth,
                "transformer_heads": model.transformer_heads,
                "transformer_mlp_ratio": model.transformer_mlp_ratio,
            },
        },
        ckpt_path,
    )

    occ = np.zeros((10, 10), dtype=np.float32)
    pred = infer_cost_volume(
        ckpt_path=ckpt_path,
        occ_map_numpy=occ,
        start_xy=(1, 1),
        goal_xy=(8, 8),
        device="cpu",
    )

    assert pred.shape == (1, 10, 10)
    assert np.isfinite(pred).all()
