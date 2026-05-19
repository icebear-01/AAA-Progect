#!/usr/bin/env python3
"""
Benchmark RL-ONNX full-episode planning time against classic DP on the same obstacle sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

from benchmark_emplanner_classic_dp import convert_obstacles, run_classic_dp
from ppo import encode_observation
from rl_env import SLPathEnv
from sl_grid import DEFAULT_L_RANGE, DEFAULT_S_RANGE, GridSpec, build_grid
from sl_obstacles import generate_random_obstacles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RL-ONNX planning time against classic DP."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--obstacle-min", type=int, default=10)
    parser.add_argument("--obstacle-max", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sample-s-num", type=float, default=5.0)
    parser.add_argument("--car-width", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def _make_eval_env(payload: Dict[str, object]) -> SLPathEnv:
    spec = GridSpec(**payload["grid_spec"])
    env_kwargs = dict(payload.get("environment", {}))
    env_kwargs["scenario_dataset_path"] = None
    env_kwargs["scenario_pool_size"] = 1
    env_kwargs["scenario_top_k"] = 1
    env_kwargs["scenario_min_obstacles"] = 0
    env_kwargs["scenario_max_avg_cost"] = None
    env_kwargs["scenario_max_attempts"] = None
    env_kwargs["min_obstacles"] = 0
    env_kwargs["max_obstacles"] = 0
    return SLPathEnv(spec, **env_kwargs)


def _set_manual_scenario(
    env: SLPathEnv,
    *,
    obstacles,
    start_l: float = 0.0,
) -> Dict[str, np.ndarray]:
    env._s_grid, env._l_grid = build_grid(env.grid_spec)
    env._obstacles = list(obstacles)
    env._occupancy = env._build_occupancy(obstacles)
    l_coords = env._l_grid[0, :]
    initial_l = int(np.argmin(np.abs(l_coords - start_l)))
    env._start_l = float(start_l)
    env._path_indices = [initial_l]
    env._s_index = 1 if env.grid_spec.s_samples > 1 else 0
    env._last_action_mask = None
    env._last_scenario_dp_result = None
    return env._build_observation()


def _benchmark_rl_episode(
    *,
    env: SLPathEnv,
    session: ort.InferenceSession,
    observation: Dict[str, np.ndarray],
) -> Dict[str, float | str | int]:
    done = False
    total_reward = 0.0
    step_count = 0
    forward_ms_total = 0.0
    episode_start = perf_counter()

    while not done:
        encoded = encode_observation(
            observation,
            env.grid_spec,
            include_action_mask=True,
        ).astype(np.float32)[None, :]
        t0 = perf_counter()
        logits, _ = session.run(None, {"state": encoded})
        forward_ms_total += (perf_counter() - t0) * 1000.0

        logits = np.asarray(logits[0], dtype=np.float32)
        mask = np.asarray(observation.get("action_mask"), dtype=bool)
        if mask.shape[0] != logits.shape[0]:
            mask = np.ones_like(logits, dtype=bool)
        masked_logits = np.where(mask, logits, -1e9)
        action = int(np.argmax(masked_logits))

        result = env.step(action)
        observation = result.observation
        done = bool(result.done)
        total_reward += float(result.reward)
        step_count += 1
        if done:
            terminal_reason = str(result.info.get("reason", "unknown"))

    episode_total_ms = (perf_counter() - episode_start) * 1000.0
    return {
        "episode_ms": episode_total_ms,
        "forward_ms": forward_ms_total,
        "return": float(total_reward),
        "steps": int(step_count),
        "reason": terminal_reason,
    }


def _apply_chinese_style() -> FontProperties:
    font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    font_prop = FontProperties(fname=font_path)
    plt.rcParams.update(
        {
            "font.family": font_prop.get_name(),
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#fcfcfc",
            "axes.edgecolor": "#4a4f55",
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return font_prop


def _plot_comparison(
    *,
    obstacle_counts: List[int],
    rl_e2e_means: List[float],
    rl_forward_means: List[float],
    dp_means: List[float],
    output_path: Path,
) -> None:
    font_prop = _apply_chinese_style()
    cmap = LinearSegmentedColormap.from_list(
        "paper_runtime",
        ["#163b66", "#2a6f97", "#5aa6a6", "#d9c27d", "#d97b29", "#b33f2f"],
        N=256,
    )

    rl_matrix = np.column_stack(
        [np.asarray(rl_e2e_means, dtype=np.float64), np.asarray(rl_forward_means, dtype=np.float64)]
    )
    dp_matrix = np.column_stack([np.asarray(dp_means, dtype=np.float64)])

    fig, (ax_rl, ax_dp) = plt.subplots(
        1,
        2,
        figsize=(9.8, 8.8),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )

    im_rl = ax_rl.imshow(rl_matrix, aspect="auto", origin="lower", cmap=cmap)
    ax_rl.set_title("RL-ONNX 规划耗时", fontproperties=font_prop, pad=8)
    ax_rl.set_xticks([0, 1])
    ax_rl.set_xticklabels(["端到端", "前向累计"], fontproperties=font_prop)
    ax_rl.set_yticks(np.arange(0, len(obstacle_counts), 2))
    ax_rl.set_yticklabels([str(v) for v in obstacle_counts[::2]], fontproperties=font_prop)
    ax_rl.set_ylabel("障碍物数量", fontproperties=font_prop)
    for spine in ax_rl.spines.values():
        spine.set_color("#4a4f55")
        spine.set_linewidth(0.9)

    im_dp = ax_dp.imshow(dp_matrix, aspect="auto", origin="lower", cmap=cmap)
    ax_dp.set_title("传统DP 规划耗时", fontproperties=font_prop, pad=8)
    ax_dp.set_xticks([0])
    ax_dp.set_xticklabels(["经典DP"], fontproperties=font_prop)
    ax_dp.set_yticks(np.arange(0, len(obstacle_counts), 2))
    ax_dp.set_yticklabels([str(v) for v in obstacle_counts[::2]], fontproperties=font_prop)
    for spine in ax_dp.spines.values():
        spine.set_color("#4a4f55")
        spine.set_linewidth(0.9)

    cbar_rl = fig.colorbar(im_rl, ax=ax_rl, fraction=0.046, pad=0.03)
    cbar_rl.set_label("时间（ms）", fontproperties=font_prop)
    cbar_dp = fig.colorbar(im_dp, ax=ax_dp, fraction=0.08, pad=0.06)
    cbar_dp.set_label("时间（ms）", fontproperties=font_prop)

    fig.suptitle("10-30 个障碍物场景下 RL-ONNX 与传统DP耗时对比", fontproperties=font_prop, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    spec = GridSpec(**payload["grid_spec"])
    env = _make_eval_env(payload)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    sample_s = float(spec.s_range[1] - spec.s_range[0]) / max(spec.s_samples - 1, 1)
    sample_l = float(spec.l_range[1] - spec.l_range[0]) / max(spec.l_samples - 1, 1)

    obstacle_counts = list(range(int(args.obstacle_min), int(args.obstacle_max) + 1))
    rng = np.random.default_rng(int(args.seed))
    records: List[Dict[str, object]] = []

    for obstacle_count in obstacle_counts:
        rl_e2e_values: List[float] = []
        rl_forward_values: List[float] = []
        dp_values: List[float] = []
        success_count = 0

        for _ in range(int(args.trials)):
            obstacles = generate_random_obstacles(
                DEFAULT_S_RANGE,
                DEFAULT_L_RANGE,
                min_count=obstacle_count,
                max_count=obstacle_count,
                length_range=(0.6, 1.8),
                width_range=(0.4, 1.4),
                rng=rng,
            )

            observation = _set_manual_scenario(env, obstacles=obstacles, start_l=0.0)
            rl_result = _benchmark_rl_episode(
                env=env,
                session=session,
                observation=observation,
            )
            rl_e2e_values.append(float(rl_result["episode_ms"]))
            rl_forward_values.append(float(rl_result["forward_ms"]))
            if rl_result["reason"] == "goal_reached":
                success_count += 1

            classic_obstacles = convert_obstacles(obstacles)
            dp_start = perf_counter()
            _, feasible = run_classic_dp(
                col_node_num=spec.s_samples,
                row_node_num=spec.l_samples,
                sample_s=sample_s,
                sample_l=sample_l,
                sample_s_num=float(args.sample_s_num),
                obstacles=classic_obstacles,
                car_width=float(args.car_width),
            )
            dp_values.append((perf_counter() - dp_start) * 1000.0)

        record = {
            "obstacle_count": int(obstacle_count),
            "trials": int(args.trials),
            "rl_end_to_end": _stats(rl_e2e_values),
            "rl_forward": _stats(rl_forward_values),
            "classic_dp": _stats(dp_values),
            "rl_success_rate": float(success_count / max(1, int(args.trials))),
        }
        records.append(record)
        print(
            f"obs={obstacle_count} | "
            f"rl_e2e={record['rl_end_to_end']['mean_ms']:.3f} ms | "
            f"rl_forward={record['rl_forward']['mean_ms']:.3f} ms | "
            f"dp={record['classic_dp']['mean_ms']:.3f} ms | "
            f"rl_success={record['rl_success_rate']:.2f}"
        )

    json_path = output_dir / "compare_onnx_vs_classic_dp.json"
    fig_path = output_dir / "compare_onnx_vs_classic_dp_heatmap.png"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "checkpoint": str(checkpoint_path),
                    "onnx": str(onnx_path),
                    "obstacle_min": int(args.obstacle_min),
                    "obstacle_max": int(args.obstacle_max),
                    "trials": int(args.trials),
                    "sample_s_num": float(args.sample_s_num),
                    "car_width": float(args.car_width),
                    "grid_spec": {
                        "s_range": list(spec.s_range),
                        "l_range": list(spec.l_range),
                        "s_samples": int(spec.s_samples),
                        "l_samples": int(spec.l_samples),
                    },
                },
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_comparison(
        obstacle_counts=obstacle_counts,
        rl_e2e_means=[record["rl_end_to_end"]["mean_ms"] for record in records],
        rl_forward_means=[record["rl_forward"]["mean_ms"] for record in records],
        dp_means=[record["classic_dp"]["mean_ms"] for record in records],
        output_path=fig_path,
    )

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved heatmap to {fig_path}")


if __name__ == "__main__":
    main()
