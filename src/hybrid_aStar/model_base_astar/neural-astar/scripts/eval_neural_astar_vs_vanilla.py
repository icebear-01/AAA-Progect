"""Evaluate Neural A* against Vanilla A* on original shortest-path datasets.

Metrics follow the original training validation style:
- p_opt: fraction of samples where Neural A* path length equals Vanilla A*
- p_exp: average relative explored-node reduction vs Vanilla A*
- h_mean: harmonic mean of p_opt and p_exp

Coordinate/data conventions follow the upstream project.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import load_from_ptl_checkpoint, set_global_seeds


@dataclass
class Meter:
    n: int = 0
    p_opt_sum: float = 0.0
    p_exp_sum: float = 0.0
    exp_va_sum: float = 0.0
    exp_na_sum: float = 0.0
    path_va_sum: float = 0.0
    path_na_sum: float = 0.0
    success_va_sum: float = 0.0
    success_na_sum: float = 0.0
    t_va_ms_sum: float = 0.0
    t_na_ms_sum: float = 0.0

    def update(
        self,
        path_va: np.ndarray,
        path_na: np.ndarray,
        exp_va: np.ndarray,
        exp_na: np.ndarray,
        t_va_ms_batch: float,
        t_na_ms_batch: float,
    ) -> None:
        bsz = int(path_va.shape[0])
        self.n += bsz

        p_opt = float((path_va == path_na).mean())
        p_exp = float(np.maximum((exp_va - exp_na) / np.maximum(exp_va, 1e-10), 0.0).mean())
        self.p_opt_sum += p_opt * bsz
        self.p_exp_sum += p_exp * bsz

        self.exp_va_sum += float(exp_va.sum())
        self.exp_na_sum += float(exp_na.sum())
        self.path_va_sum += float(path_va.sum())
        self.path_na_sum += float(path_na.sum())
        self.success_va_sum += float((path_va > 0).sum())
        self.success_na_sum += float((path_na > 0).sum())

        # batch runtime normalized to per-sample runtime
        self.t_va_ms_sum += float(t_va_ms_batch)
        self.t_na_ms_sum += float(t_na_ms_batch)

    def summary(self) -> dict:
        n = max(1, self.n)
        p_opt = self.p_opt_sum / n
        p_exp = self.p_exp_sum / n
        h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))
        return {
            "n": self.n,
            "p_opt": p_opt,
            "p_exp": p_exp,
            "h_mean": h_mean,
            "success_va": self.success_va_sum / n,
            "success_na": self.success_na_sum / n,
            "exp_va": self.exp_va_sum / n,
            "exp_na": self.exp_na_sum / n,
            "path_va": self.path_va_sum / n,
            "path_na": self.path_na_sum / n,
            "runtime_va_ms": self.t_va_ms_sum / n,
            "runtime_na_ms": self.t_na_ms_sum / n,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Neural A* vs Vanilla A*")
    p.add_argument(
        "--dataset",
        type=str,
        default="planning-datasets/data/mpd/mazes_032_moore_c8",
        help="Dataset path prefix without .npz suffix",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--num-starts", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0, help="0 means full split")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing lightning checkpoint(s). "
        "Default: model/<dataset_name>",
    )
    p.add_argument("--encoder-input", type=str, default="m+")
    p.add_argument("--encoder-arch", type=str, default="CNN")
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--g-ratio", type=float, default=0.5)
    p.add_argument(
        "--backend",
        type=str,
        default="pq",
        choices=["pq", "differentiable"],
        help="A* backend for both planners. pq is closer to classical A* inference.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    use_diff = args.backend == "differentiable"

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU")
        device = "cpu"

    dataname = os.path.basename(args.dataset)
    ckpt_dir = args.checkpoint_dir or f"model/{dataname}"

    na = NeuralAstar(
        g_ratio=args.g_ratio,
        Tmax=1.0,
        encoder_input=args.encoder_input,
        encoder_arch=args.encoder_arch,
        encoder_depth=args.encoder_depth,
        learn_obstacles=False,
        use_differentiable_astar=use_diff,
    )
    na.load_state_dict(load_from_ptl_checkpoint(ckpt_dir))
    na.eval().to(device)

    va = VanillaAstar(
        g_ratio=args.g_ratio,
        use_differentiable_astar=use_diff,
    )
    va.eval().to(device)

    loader = create_dataloader(
        args.dataset + ".npz",
        args.split,
        args.batch_size,
        num_starts=args.num_starts,
        shuffle=False,
    )

    meter = Meter()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

            map_designs, start_maps, goal_maps, _ = batch
            map_designs = map_designs.to(device)
            start_maps = start_maps.to(device)
            goal_maps = goal_maps.to(device)

            t0 = time.perf_counter()
            out_na = na(map_designs, start_maps, goal_maps)
            t1 = time.perf_counter()
            out_va = va(map_designs, start_maps, goal_maps)
            t2 = time.perf_counter()

            path_na = out_na.paths.sum((1, 2, 3)).detach().cpu().numpy()
            path_va = out_va.paths.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = out_na.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_va = out_va.histories.sum((1, 2, 3)).detach().cpu().numpy()

            meter.update(
                path_va=path_va,
                path_na=path_na,
                exp_va=exp_va,
                exp_na=exp_na,
                t_va_ms_batch=(t2 - t1) * 1e3,
                t_na_ms_batch=(t1 - t0) * 1e3,
            )

    s = meter.summary()
    print("=== Neural A* vs Vanilla A* ===")
    print(f"dataset={args.dataset}.npz split={args.split} backend={args.backend} n={s['n']}")
    print(
        "metrics: "
        f"p_opt={s['p_opt']:.4f}, "
        f"p_exp={s['p_exp']:.4f}, "
        f"h_mean={s['h_mean']:.4f}"
    )
    print(
        "success_rate: "
        f"vanilla={s['success_va']:.4f}, neural={s['success_na']:.4f}"
    )
    print(
        "expanded_nodes(mean): "
        f"vanilla={s['exp_va']:.2f}, neural={s['exp_na']:.2f}, "
        f"delta={s['exp_na'] - s['exp_va']:.2f}"
    )
    print(
        "path_length(mean): "
        f"vanilla={s['path_va']:.2f}, neural={s['path_na']:.2f}, "
        f"delta={s['path_na'] - s['path_va']:.2f}"
    )
    print(
        "runtime_per_sample_ms: "
        f"vanilla={s['runtime_va_ms']:.3f}, neural={s['runtime_na_ms']:.3f}"
    )

    better_exp = s["exp_na"] < s["exp_va"]
    better_path = math.isclose(s["path_na"], s["path_va"], rel_tol=1e-6, abs_tol=1e-6) or (
        s["path_na"] < s["path_va"]
    )
    if better_exp and better_path:
        print("result: neural_astar_better_or_equal_than_vanilla")
    else:
        print("result: neural_astar_not_better_than_vanilla_on_this_setting")


if __name__ == "__main__":
    main()

