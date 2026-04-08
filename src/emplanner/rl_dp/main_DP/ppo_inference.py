#!/usr/bin/env python3
"""
Run inference with a trained PPO policy, report step rewards, and plot results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import Polygon
from torch.distributions import Categorical

from ppo import ActorCritic, PPOConfig, encode_observation
from rl_env import SLPathEnv
from sl_grid import GridSpec, build_grid


PAPER_COLORS = {
    "dp_path": "#1f5aa6",
    "rl_path": "#d04a02",
    "reference": "#4c566a",
    "grid": "#8a939c",
    "obstacle_fill": "#9aa0a6",
    "obstacle_edge": "#5f6368",
}


def _pick_font_family(candidates: Sequence[str]) -> Optional[str]:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def _apply_paper_style() -> None:
    serif_family = _pick_font_family(["AR PL UMing CN", "SimSun", "Songti SC", "STSong"])
    sans_family = _pick_font_family(["Noto Sans CJK SC"])
    font_family = [
        name
        for name in [serif_family, sans_family, "DejaVu Serif", "DejaVu Sans"]
        if name
    ]
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "axes.unicode_minus": False,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _build_text_font_properties():
    font_family = _pick_font_family(
        [
            "AR PL UMing CN",
            "SimSun",
            "Songti SC",
            "STSong",
            "Noto Sans CJK SC",
        ]
    )
    if not font_family:
        return None
    return font_manager.FontProperties(family=font_family)


def _load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[ActorCritic, SLPathEnv, PPOConfig]:
    payload = torch.load(checkpoint_path, map_location=device)

    state_dict = payload.get("model_state", {})
    conv_weight = state_dict.get("conv_trunk.0.weight")
    conv_in_channels = conv_weight.shape[1] if conv_weight is not None else None

    config = PPOConfig(**payload["config"])
    spec = GridSpec(**payload["grid_spec"])

    env_kwargs: Dict[str, object] = payload.get("environment", {})
    env = SLPathEnv(spec, **env_kwargs)

    sample_obs = env.reset()
    feature_dim = encode_observation(sample_obs, spec).shape[0]
    action_dim = spec.l_samples

    occupancy_shape = (spec.s_samples, spec.l_samples)
    grid_spatial_size = occupancy_shape[0] * occupancy_shape[1]
    extra_dim = feature_dim % grid_spatial_size
    grid_channels_expected = (feature_dim - extra_dim) // grid_spatial_size
    if grid_channels_expected < 1:
        raise ValueError(
            "Encoded observation must contain at least one grid plane."
        )
    if extra_dim < 0:
        raise ValueError(
            "Encoded observation dimensionality is inconsistent with grid size."
        )
    if conv_in_channels is not None and conv_in_channels != grid_channels_expected:
        raise ValueError(
            "Checkpoint input channels do not match current encoder output "
            f"(checkpoint expects {conv_in_channels}, encoder produces {grid_channels_expected}). "
            "Retrain or re-export the policy with the updated encoder."
        )
    grid_channels = (
        conv_in_channels if conv_in_channels is not None else grid_channels_expected
    )
    if grid_channels < 1:
        raise ValueError(
            "Encoded observation must contain at least one grid plane."
        )

    policy = ActorCritic(
        occupancy_shape=occupancy_shape,
        grid_channels=grid_channels,
        extra_dim=extra_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    policy.load_state_dict(payload["model_state"])
    policy.eval()
    setattr(policy, "include_action_mask_input", True)
    setattr(policy, "mask_logits_with_action_mask", True)

    if device.type != config.device:
        config = PPOConfig(**{**asdict(config), "device": device.type})

    return policy, env, config


def _choose_action(
    policy: ActorCritic,
    observation: dict,
    spec: GridSpec,
    device: torch.device,
    *,
    sample: bool,
    include_action_mask: Optional[bool] = None,
    mask_logits: Optional[bool] = None,
) -> Tuple[int, torch.Tensor, float]:
    include_mask = (
        include_action_mask
        if include_action_mask is not None
        else bool(getattr(policy, "include_action_mask_input", True))
    )
    apply_mask_logits = (
        mask_logits
        if mask_logits is not None
        else bool(getattr(policy, "mask_logits_with_action_mask", True))
    )
    encoded = encode_observation(
        observation, spec, include_action_mask=include_mask
    )
    state = torch.as_tensor(encoded, device=device)
    action_mask_np = observation.get("action_mask") if apply_mask_logits else None
    with torch.no_grad():
        forward_start = perf_counter()
        logits, _ = policy(state)
        forward_time = perf_counter() - forward_start
        masked_logits = logits
        if action_mask_np is not None:
            mask_tensor = torch.as_tensor(action_mask_np, device=device, dtype=torch.bool)
            has_valid = bool(mask_tensor.any().item())
            candidate_logits = logits.masked_fill(~mask_tensor, -1e9)
            if has_valid:
                masked_logits = candidate_logits
        if sample:
            dist = Categorical(logits=masked_logits)
            action = int(dist.sample().item())
            return action, masked_logits, forward_time
        action = int(torch.argmax(masked_logits).item())
        return action, masked_logits, forward_time


def _reward_components(
    result_reward: float,
    result_info: Dict[str, object],
    env: SLPathEnv,
) -> Dict[str, float]:
    components: Dict[str, float] = {}
    components["net_reward"] = float(result_reward)

    components["reference_cost"] = -float(result_info.get("reference_cost", 0.0))
    components["move_limit_cost"] = -float(result_info.get("move_limit_cost", 0.0))
    components["smoothness_cost"] = -float(result_info.get("smoothness_cost", 0.0))
    components["slope_cost"] = float(result_info.get("slope_cost", 0.0))
    components["non_goal_penalty"] = -float(result_info.get("non_goal_penalty", 0.0))

    reason = result_info.get("reason")
    components["terminal_reward"] = (
        float(env.terminal_reward) if reason == "goal_reached" else 0.0
    )
    collision_penalty = float(env.collision_penalty)
    if reason == "collision":
        components["collision_penalty"] = collision_penalty
    else:
        components["collision_penalty"] = 0.0

    if reason == "out_of_bounds":
        components["out_of_bounds_penalty"] = collision_penalty
    else:
        components["out_of_bounds_penalty"] = 0.0

    no_valid_action_penalty = float(getattr(env, "no_valid_action_penalty", 0.0))
    if reason == "no_valid_action":
        components["no_valid_action_penalty"] = no_valid_action_penalty
    else:
        components["no_valid_action_penalty"] = 0.0

    return components


def run_inference(
    policy: ActorCritic,
    env: SLPathEnv,
    *,
    episodes: int,
    sample: bool,
    device: torch.device,
    start_l: Optional[float] = None,
) -> Tuple[
    List[float],
    List[List[float]],
    List[List[Dict[str, float]]],
    List[Dict[str, object]],
    List[List[Dict[str, object]]],
    float,
]:
    episode_returns: List[float] = []
    per_episode_step_rewards: List[List[float]] = []
    per_episode_components: List[List[Dict[str, float]]] = []
    episode_paths: List[Dict[str, object]] = []
    per_episode_step_details: List[List[Dict[str, object]]] = []
    model_forward_time = 0.0

    for episode in range(1, episodes + 1):
        observation = env.reset(start_l=start_l)
        done = False
        total_reward = 0.0
        step_rewards: List[float] = []
        step_components: List[Dict[str, float]] = []
        episode_info: Optional[Dict[str, object]] = None
        step_details: List[Dict[str, object]] = []
        initial_indices = np.asarray(observation["path_indices"], dtype=np.int32)
        if initial_indices.size == 0:
            start_idx = 0
        else:
            start_idx = int(initial_indices[0])
        start_s = float(observation["s_coords"][0])
        if "start_l" in observation:
            start_l = float(np.asarray(observation["start_l"]).reshape(-1)[0])
        else:
            start_l = float(observation["l_coords"][start_idx])
        start_summary = {
            "s": start_s,
            "l": start_l,
            "index": start_idx,
        }
        base_components = {
            "net_reward": 0.0,
            "reference_cost": 0.0,
            "move_limit_cost": 0.0,
            "smoothness_cost": 0.0,
            "slope_cost": 0.0,
            "non_goal_penalty": 0.0,
            "terminal_reward": 0.0,
            "collision_penalty": 0.0,
            "out_of_bounds_penalty": 0.0,
            "no_valid_action_penalty": 0.0,
        }
        step_details.append(
            {
                "step": 0,
                "s": start_s,
                "s_index": 0,
                "s_grid_index": 0,
                "l": start_l,
                "index": start_idx,
                "delta_idx": 0,
                "delta_l": 0.0,
                "reward": 0.0,
                "components": dict(base_components),
                "info": {"reason": "start"},
            }
        )
        dp_result = env.last_scenario_dp_result
        dp_path_s: Optional[np.ndarray] = None
        dp_path_l: Optional[np.ndarray] = None
        if dp_result is not None and dp_result.feasible and dp_result.path_indices:
            full_s_grid, full_l_grid = build_grid(env.grid_spec)
            full_s_coords = full_s_grid[:, 0]
            full_l_coords = full_l_grid[0, :]
            if len(dp_result.path_indices) == len(full_s_coords):
                dp_path_s = np.asarray(full_s_coords, dtype=np.float32)
                dp_path_l = np.asarray(
                    [full_l_coords[int(idx)] for idx in dp_result.path_indices],
                    dtype=np.float32,
                )

        while not done:
            action, _, forward_time = _choose_action(
                policy,
                observation,
                env.grid_spec,
                device,
                sample=sample,
                include_action_mask=bool(
                    getattr(policy, "include_action_mask_input", True)
                ),
                mask_logits=bool(getattr(policy, "mask_logits_with_action_mask", True)),
            )
            model_forward_time += forward_time
            result = env.step(action)
            reward = float(result.reward)
            components = _reward_components(reward, result.info, env)

            step_rewards.append(reward)
            step_components.append(components)
            total_reward += reward
            episode_info = dict(result.info)

            observation = result.observation
            done = result.done
            path_indices_full = observation["path_indices"]
            current_step = len(path_indices_full) - 1
            s_coords = observation["s_coords"]
            s_index_obs = int(np.asarray(observation["s_index"], dtype=np.int32))
            grid_index = min(current_step, len(s_coords) - 1)
            s_value = float(s_coords[grid_index])
            l_index = int(path_indices_full[current_step])
            l_value = float(observation["l_coords"][l_index])
            if current_step > 0:
                prev_index = int(path_indices_full[current_step - 1])
                prev_l_value = float(observation["l_coords"][prev_index])
                delta_idx = l_index - prev_index
                delta_l = l_value - prev_l_value
            else:
                delta_idx = 0
                delta_l = 0.0
            step_details.append(
                {
                    "step": current_step,
                    "s": s_value,
                    "s_index": s_index_obs,
                    "s_grid_index": grid_index,
                    "l": l_value,
                    "index": l_index,
                    "delta_idx": delta_idx,
                    "delta_l": delta_l,
                    "reward": reward,
                    "components": components,
                    "info": dict(result.info),
                }
            )

        episode_returns.append(total_reward)
        per_episode_step_rewards.append(step_rewards)
        per_episode_components.append(step_components)

        path_s = np.asarray([detail["s"] for detail in step_details], dtype=np.float32)
        path_l = np.asarray([detail["l"] for detail in step_details], dtype=np.float32)
        obstacles = tuple(env.obstacles)
        occupancy_grid = np.asarray(observation["occupancy"], dtype=bool)
        episode_paths.append(
            {
                "path_s": path_s,
                "path_l": path_l,
                "obstacles": obstacles,
                "occupancy": occupancy_grid,
                "info": episode_info or {},
                "start": start_summary,
                "dp_path_s": dp_path_s,
                "dp_path_l": dp_path_l,
            }
        )
        per_episode_step_details.append(step_details)

        print(f"Episode {episode:02d} return: {total_reward:.3f}")

    return (
        episode_returns,
        per_episode_step_rewards,
        per_episode_components,
        episode_paths,
        per_episode_step_details,
        model_forward_time,
    )


def plot_episode_path(
    spec: GridSpec,
    path_s: np.ndarray,
    path_l: np.ndarray,
    obstacles: Sequence[object],
    occupancy: np.ndarray,
    episode_return: float,
    *,
    dp_path_s: Optional[np.ndarray] = None,
    dp_path_l: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    show: bool = True,
    paper_style: bool = False,
) -> Optional[Path]:
    if path_s.size == 0 or path_l.size == 0:
        raise ValueError("Empty path provided for plotting.")

    if paper_style:
        _apply_paper_style()
    text_font = _build_text_font_properties() if paper_style else None

    s_grid, l_grid = build_grid(spec)
    occupancy_mask = np.asarray(occupancy, dtype=bool)
    if occupancy_mask.shape != s_grid.shape:
        raise ValueError("Occupancy grid shape does not match sampling grid.")

    fig_width = 7.2 if paper_style else 8.0
    fig_height = 3.8 if paper_style else 5.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    grid_marker_size = 24 if paper_style else 28
    grid_marker_linewidth = 1.1 if paper_style else 1.3
    ax.scatter(
        s_grid.reshape(-1),
        l_grid.reshape(-1),
        s=grid_marker_size,
        facecolors="none",
        edgecolors=PAPER_COLORS["grid"],
        linewidths=grid_marker_linewidth,
        alpha=0.95,
        label="采样栅格点",
        zorder=1,
    )
    ax.plot(
        [spec.s_range[0], spec.s_range[1]],
        [0.0, 0.0],
        color=PAPER_COLORS["reference"],
        linestyle="--",
        linewidth=1.3 if paper_style else 1.5,
        label="参考线",
        zorder=1.5,
    )

    for obstacle in obstacles:
        polygon = Polygon(
            obstacle.corners(),
            closed=True,
            facecolor=PAPER_COLORS["obstacle_fill"],
            edgecolor=PAPER_COLORS["obstacle_edge"],
            linewidth=0.9 if paper_style else 1.1,
            alpha=0.35,
            zorder=2,
        )
        ax.add_patch(polygon)

    if (
        dp_path_s is not None
        and dp_path_l is not None
        and dp_path_s.size > 0
        and dp_path_l.size > 0
    ):
        ax.plot(
            dp_path_s,
            dp_path_l,
            color=PAPER_COLORS["dp_path"],
            linewidth=2.1,
            zorder=3,
            label="决策路径",
        )
    ax.plot(
        path_s,
        path_l,
        color=PAPER_COLORS["rl_path"],
        linewidth=2.1,
        zorder=4,
        label="RL策略路径",
    )

    ax.set_xlabel("s [m]", fontproperties=text_font)
    ax.set_ylabel("l [m]", fontproperties=text_font)
    ax.set_xlim(spec.s_range[0], spec.s_range[1])
    ax.set_ylim(spec.l_range[0], spec.l_range[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_kwargs = {"frameon": True, "framealpha": 0.92}
    if paper_style:
        legend_kwargs.update(
            {
                "loc": "lower center",
                "bbox_to_anchor": (0.5, 1.02),
                "ncol": 2,
                "columnspacing": 1.2,
                "handlelength": 2.0,
                "borderaxespad": 0.0,
            }
        )
    else:
        legend_kwargs["loc"] = "upper right"
    if text_font is not None:
        legend_kwargs["prop"] = text_font
    ax.legend(**legend_kwargs)
    ax.text(
        0.98,
        0.03,
        f"Return = {episode_return:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9 if paper_style else 10,
        color=PAPER_COLORS["reference"],
    )
    fig.tight_layout()
    saved_path: Optional[Path] = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220 if paper_style else 160)
        saved_path = output_path
    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained PPO policy and plot rewards."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained policy checkpoint. "
        "Default: latest *.pt inside --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("main/checkpoints"),
        help="Directory to search for checkpoints when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from the policy distribution instead of taking argmax actions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (cpu/cuda). Default: cpu; pass --device cuda to force GPU.",
    )
    parser.add_argument(
        "--start-l",
        type=float,
        default=None,
        help="Optional continuous start l value (overrides random start index).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG path for the inference visualization.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive matplotlib window.",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Use the same paper-style color palette as plot_compare.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = (args.device or "cpu").strip().lower()
    if device_name == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device_name = "cpu"
    elif device_name != "cpu":
        print(f"Unrecognized device '{args.device}', defaulting to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint_path: Optional[Path] = args.checkpoint
    if checkpoint_path is None:
        search_dirs = []
        if args.checkpoint_dir is not None:
            search_dirs.append(args.checkpoint_dir)
        default_dir = Path("checkpoints")
        alt_dir = Path("main") / "checkpoints"
        for candidate_dir in (default_dir, alt_dir):
            if candidate_dir not in search_dirs:
                search_dirs.append(candidate_dir)

        checkpoint_candidates: List[Path] = []
        for directory in search_dirs:
            directory = Path(directory)
            if directory.exists():
                checkpoint_candidates.extend(directory.glob("*.pt"))

        if not checkpoint_candidates:
            raise FileNotFoundError(
                "No checkpoint found. Specify --checkpoint or ensure *.pt files exist "
                f"in {', '.join(str(Path(d)) for d in search_dirs)}"
            )
        checkpoint_candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        checkpoint_path = checkpoint_candidates[0]

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    policy, env, config = _load_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"Config device: {config.device} | Inference device: {device.type}")

    start_time = perf_counter()
    (
        returns,
        step_rewards,
        components,
        paths,
        step_details,
        model_forward_time,
    ) = run_inference(
        policy,
        env,
        episodes=1,
        sample=args.sample,
        device=device,
        start_l=args.start_l,
    )
    inference_duration = perf_counter() - start_time
    print("Episode returns:", ", ".join(f"{r:.3f}" for r in returns))
    print(f"Inference time: {inference_duration * 1000:.2f} ms")
    print(f"Pure model forward time: {model_forward_time * 1000:.2f} ms")

    last_path = paths[-1]
    last_steps = step_details[-1]
    last_episode_idx = len(returns)
    start_info = last_path.get("start", {})
    if start_info:
        print(
            f"\nPlotted episode (Episode {last_episode_idx:02d}) start: "
            f"s={start_info['s']:.2f}, l={start_info['l']:.2f} (idx={start_info['index']})"
        )
    else:
        print(f"\nPlotted episode (Episode {last_episode_idx:02d}) details:")

    for detail in last_steps:
        step_id = detail["step"]
        s_value = detail["s"]
        l_value = detail["l"]
        l_index = detail["index"]
        delta_idx = detail.get("delta_idx", 0)
        delta_l = detail.get("delta_l", 0.0)
        reward = detail["reward"]
        components_dict = detail["components"]
        reason = detail.get("info", {}).get("reason")
        if reason == "start":
            s_grid_index = detail.get("s_grid_index")
            s_index_raw = detail.get("s_index")
            print(
                f"  Step {step_id:02d} | s={s_value:.2f} (s_idx={s_grid_index}, raw={s_index_raw}) "
                f"| l={l_value:.2f} (idx={l_index}) | start | Δidx={delta_idx:+d} | Δl={delta_l:+.3f}"
            )
            continue
        component_labels = [
            ("net_reward", "net"),
            ("reference_cost", "ref"),
            ("move_limit_cost", "move"),
            ("smoothness_cost", "smooth"),
            ("slope_cost", "slope"),
            ("non_goal_penalty", "non_goal"),
            ("terminal_reward", "terminal"),
            ("collision_penalty", "collision"),
            ("out_of_bounds_penalty", "out_of_bounds"),
            ("no_valid_action_penalty", "no_valid_action"),
        ]
        components_str = " | ".join(
            f"{label}={components_dict.get(key, 0.0):+,.3f}"
            for key, label in component_labels
        )
        reason_str = f" | reason={reason}" if reason else ""
        s_grid_index = detail.get("s_grid_index")
        s_index_raw = detail.get("s_index")
        print(
            f"  Step {step_id:02d} | s={s_value:.2f} (s_idx={s_grid_index}, raw={s_index_raw}) | l={l_value:.2f} "
            f"(idx={l_index}) | Δidx={delta_idx:+d} | Δl={delta_l:+.3f} | "
            f"reward={reward: .3f}{reason_str} | {components_str}"
        )

    if args.output is None:
        output_path = checkpoint_path.parent / "inference_plots" / (
            f"{checkpoint_path.stem}_inference.png"
        )
    else:
        output_path = args.output
    saved_path = plot_episode_path(
        env.grid_spec,
        last_path["path_s"],
        last_path["path_l"],
        last_path["obstacles"],
        last_path["occupancy"],
        returns[-1],
        dp_path_s=last_path.get("dp_path_s"),
        dp_path_l=last_path.get("dp_path_l"),
        output_path=output_path,
        show=not args.no_show,
        paper_style=args.paper_style,
    )
    if saved_path is not None:
        print(f"Saved visualization to {saved_path}")


if __name__ == "__main__":
    main()
