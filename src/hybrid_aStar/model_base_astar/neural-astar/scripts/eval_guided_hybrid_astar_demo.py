"""Evaluate baseline vs guided minimal Hybrid A* demo."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hybrid_astar_guided import GuidedHybridAstar
from hybrid_astar_guided.grid_astar import astar_8conn
from neural_astar.api.guidance_infer import infer_cost_map
from neural_astar.utils.coords import clip_cost_map_with_obstacles


@dataclass
class Stat:
    success: int = 0
    expanded_nodes: float = 0.0
    runtime_ms: float = 0.0
    path_length: float = 0.0
    total: int = 0

    def update(self, ok: bool, expanded: int, runtime_ms: float, path_len: float) -> None:
        self.total += 1
        self.success += int(ok)
        self.expanded_nodes += float(expanded)
        self.runtime_ms += float(runtime_ms)
        self.path_length += float(path_len)

    def summary(self) -> str:
        n = max(1, self.total)
        return (
            f"success_rate={self.success / n:.3f}, "
            f"expanded_nodes={self.expanded_nodes / n:.1f}, "
            f"runtime_ms={self.runtime_ms / n:.3f}, "
            f"path_length={self.path_length / n:.3f}"
        )


@dataclass
class MapHints:
    start_xy: Optional[Tuple[int, int]] = None
    goal_xy: Optional[Tuple[int, int]] = None
    start_yaw: Optional[float] = None
    goal_yaw: Optional[float] = None


@dataclass
class MapLoadInfo:
    occ_key: str
    occ_semantics: str  # "obstacle1" or "passable1"


@dataclass
class CaseDetail:
    case_id: int
    map_index: Optional[int]
    start_xy: Tuple[int, int]
    goal_xy: Tuple[int, int]
    baseline_success: bool
    baseline_expanded: int
    baseline_runtime_ms: float
    baseline_path_length: float
    guided_success: bool
    guided_expanded: int
    guided_runtime_ms: float
    guided_path_length: float


def summarize_common_path_length(case_details: List[CaseDetail]) -> Tuple[int, float, float]:
    common = [c for c in case_details if c.baseline_success and c.guided_success]
    if not common:
        return 0, 0.0, 0.0
    n = len(common)
    b = sum(c.baseline_path_length for c in common) / float(n)
    g = sum(c.guided_path_length for c in common) / float(n)
    return n, float(b), float(g)


def _pick_occ_key(keys: set[str], all_files: List[str]) -> str:
    for k in ("occ_map", "occupancy_map", "occupancy", "map", "arr_0"):
        if k in keys:
            return k
    if not all_files:
        raise ValueError("No arrays found in npz")
    return all_files[0]


def _map_count_from_occ_array(arr: np.ndarray) -> int:
    x = np.asarray(arr)
    if x.ndim <= 2:
        return 1
    if x.ndim in (3, 4):
        return int(x.shape[0])
    return 1


def random_free_xy(occ: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    h, w = occ.shape
    for _ in range(10000):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if occ[y, x] < 0.5:
            return x, y
    raise RuntimeError("Failed to sample free cell")


def make_problem(
    size: int,
    obstacle_prob: float,
    rng: np.random.Generator,
    min_start_goal_dist: float = 0.0,
):
    for _ in range(200):
        occ = (rng.random((size, size)) < obstacle_prob).astype(np.float32)
        start_xy = random_free_xy(occ, rng)
        goal_xy = random_free_xy(occ, rng)
        if start_xy == goal_xy:
            continue
        if math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]) < float(
            min_start_goal_dist
        ):
            continue
        if astar_8conn(occ, start_xy, goal_xy) is None:
            continue
        s_pose = (start_xy[0], start_xy[1], float(rng.uniform(0.0, 2.0 * math.pi)))
        g_pose = (goal_xy[0], goal_xy[1], float(rng.uniform(0.0, 2.0 * math.pi)))
        return occ, start_xy, goal_xy, s_pose, g_pose
    raise RuntimeError("Failed to create solvable random problem")


def _select_2d_from_np_array(arr: np.ndarray, map_index: int, key: str) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim == 3:
        if x.shape[0] == 1:
            return x[0].astype(np.float32)
        if map_index < 0 or map_index >= x.shape[0]:
            raise ValueError(f"map_index out of range for {key}: {map_index} not in [0, {x.shape[0]})")
        return x[map_index].astype(np.float32)
    if x.ndim == 4:
        if x.shape[1] == 1:
            if map_index < 0 or map_index >= x.shape[0]:
                raise ValueError(
                    f"map_index out of range for {key}: {map_index} not in [0, {x.shape[0]})"
                )
            return x[map_index, 0].astype(np.float32)
        if x.shape[-1] == 1:
            if map_index < 0 or map_index >= x.shape[0]:
                raise ValueError(
                    f"map_index out of range for {key}: {map_index} not in [0, {x.shape[0]})"
                )
            return x[map_index, :, :, 0].astype(np.float32)
    raise ValueError(f"Cannot convert key={key} with shape={x.shape} to 2D map")


def _argmax_xy(one_hot_2d: np.ndarray) -> Tuple[int, int]:
    idx = int(np.argmax(one_hot_2d))
    y, x = np.unravel_index(idx, one_hot_2d.shape)
    return int(x), int(y)


def _resolve_occ_semantics(
    occ_key: str, occ_raw: np.ndarray, occ_semantics: str
) -> Tuple[np.ndarray, str]:
    """Convert raw map array to occupancy semantics: 1=obstacle, 0=free."""
    raw = np.asarray(occ_raw, dtype=np.float32)
    mode = occ_semantics
    if mode == "auto":
        if occ_key == "arr_0":
            # planning-datasets arr_0 uses 1=free(passable), 0=obstacle
            mode = "passable1"
        elif occ_key in {"occ_map", "occupancy_map", "occupancy"}:
            mode = "obstacle1"
        else:
            mode = "obstacle1"

    if mode == "obstacle1":
        occ = raw
    elif mode == "passable1":
        occ = 1.0 - raw
    else:
        raise ValueError(f"Unknown occ semantics: {mode}")
    return occ.astype(np.float32), mode


def load_occ_and_hints_from_npz(
    npz_path: Path,
    map_index: int,
    occ_semantics: str,
) -> Tuple[np.ndarray, MapHints, MapLoadInfo]:
    data = np.load(npz_path)
    keys = set(data.files)

    occ_key = _pick_occ_key(keys, list(data.files))

    occ_raw = _select_2d_from_np_array(data[occ_key], map_index=map_index, key=occ_key)
    occ, resolved_mode = _resolve_occ_semantics(
        occ_key=occ_key, occ_raw=occ_raw, occ_semantics=occ_semantics
    )
    hints = MapHints()

    if "start_pose" in keys:
        sp = np.asarray(data["start_pose"]).reshape(-1)
        if sp.size >= 3:
            hints.start_xy = (int(round(float(sp[0]))), int(round(float(sp[1]))))
            hints.start_yaw = float(sp[2])
    elif "start_xy" in keys:
        sxy = np.asarray(data["start_xy"]).reshape(-1)
        if sxy.size >= 2:
            hints.start_xy = (int(round(float(sxy[0]))), int(round(float(sxy[1]))))
    elif "start_map" in keys:
        sm = _select_2d_from_np_array(data["start_map"], map_index=map_index, key="start_map")
        hints.start_xy = _argmax_xy(sm)

    if "goal_pose" in keys:
        gp = np.asarray(data["goal_pose"]).reshape(-1)
        if gp.size >= 3:
            hints.goal_xy = (int(round(float(gp[0]))), int(round(float(gp[1]))))
            hints.goal_yaw = float(gp[2])
    elif "goal_xy" in keys:
        gxy = np.asarray(data["goal_xy"]).reshape(-1)
        if gxy.size >= 2:
            hints.goal_xy = (int(round(float(gxy[0]))), int(round(float(gxy[1]))))
    elif "goal_map" in keys:
        gm = _select_2d_from_np_array(data["goal_map"], map_index=map_index, key="goal_map")
        hints.goal_xy = _argmax_xy(gm)
    elif "arr_1" in keys:
        # Original planning-datasets format stores one-hot goal maps in arr_1.
        gm = _select_2d_from_np_array(data["arr_1"], map_index=map_index, key="arr_1")
        hints.goal_xy = _argmax_xy(gm)

    return occ.astype(np.float32), hints, MapLoadInfo(occ_key=occ_key, occ_semantics=resolved_mode)


def make_problem_on_given_map(
    occ: np.ndarray,
    rng: np.random.Generator,
    start_xy: Optional[Tuple[int, int]],
    goal_xy: Optional[Tuple[int, int]],
    start_yaw: Optional[float],
    goal_yaw: Optional[float],
    min_start_goal_dist: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], Tuple[float, float, float], Tuple[float, float, float]]:
    h, w = occ.shape

    fixed_start = start_xy is not None
    fixed_goal = goal_xy is not None
    for _ in range(2000):
        s_xy = start_xy if start_xy is not None else random_free_xy(occ, rng)
        g_xy = goal_xy if goal_xy is not None else random_free_xy(occ, rng)
        if s_xy == g_xy:
            if fixed_start and fixed_goal:
                raise RuntimeError("start and goal are identical on provided map")
            continue
        if math.hypot(g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]) < float(min_start_goal_dist):
            if fixed_start and fixed_goal:
                raise RuntimeError(
                    "provided start/goal distance is smaller than --min-start-goal-dist"
                )
            continue

        sx, sy = s_xy
        gx, gy = g_xy
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            raise ValueError(f"start out of bounds: {(sx, sy)} for map {(h, w)}")
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            raise ValueError(f"goal out of bounds: {(gx, gy)} for map {(h, w)}")
        if occ[sy, sx] > 0.5:
            raise ValueError(f"start on obstacle cell: {(sx, sy)}")
        if occ[gy, gx] > 0.5:
            raise ValueError(f"goal on obstacle cell: {(gx, gy)}")

        if astar_8conn(occ, s_xy, g_xy) is None:
            if fixed_start and fixed_goal:
                raise RuntimeError("No 2D feasible path between provided start/goal on map")
            continue

        s_yaw = float(start_yaw if start_yaw is not None else rng.uniform(0.0, 2.0 * math.pi))
        g_yaw = float(goal_yaw if goal_yaw is not None else rng.uniform(0.0, 2.0 * math.pi))
        s_pose = (float(s_xy[0]), float(s_xy[1]), s_yaw)
        g_pose = (float(g_xy[0]), float(g_xy[1]), g_yaw)
        return occ, s_xy, g_xy, s_pose, g_pose

    raise RuntimeError("Failed to sample a solvable start/goal pair on provided map")


def handcrafted_cost_to_goal(occ: np.ndarray, goal_xy: Tuple[int, int]) -> np.ndarray:
    h, w = occ.shape
    gx, gy = goal_xy
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2).astype(np.float32)
    dist = dist / max(float(dist.max()), 1.0)
    return clip_cost_map_with_obstacles(dist, occ, obstacle_cost=1.0)


def chaikin_smooth_path(path: List[Tuple[float, float, float]], iterations: int) -> np.ndarray:
    """Smooth polyline for visualization only."""
    if len(path) <= 2 or iterations <= 0:
        return np.array([[p[0], p[1]] for p in path], dtype=np.float32)

    pts = np.array([[p[0], p[1]] for p in path], dtype=np.float32)
    for _ in range(iterations):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            q1 = 0.75 * p + 0.25 * q
            q2 = 0.25 * p + 0.75 * q
            new_pts.append(q1)
            new_pts.append(q2)
        new_pts.append(pts[-1])
        pts = np.array(new_pts, dtype=np.float32)
    return pts


def save_case_plot(
    occ: np.ndarray,
    cost_map: np.ndarray,
    start_xy: Tuple[int, int],
    goal_xy: Tuple[int, int],
    res_base,
    res_guided,
    lambda_guidance: float,
    out_path: Path,
    smooth_iter: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _plot_exploration(ax, trace_xy, color: str) -> None:
        if not trace_xy:
            return
        pts = np.asarray(trace_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return
        # Downsample for readability/performance on large searches.
        max_pts = 50000
        if pts.shape[0] > max_pts:
            idx = np.linspace(0, pts.shape[0] - 1, max_pts, dtype=np.int64)
            pts = pts[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=color,
            s=4.0,
            alpha=0.10,
            marker=".",
            linewidths=0.0,
            zorder=2,
        )

    axes[0].imshow(1.0 - occ, cmap="gray")
    im = axes[0].imshow(cost_map, cmap="viridis", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[0].scatter([start_xy[0]], [start_xy[1]], c="lime", s=50, marker="o", label="start")
    axes[0].scatter([goal_xy[0]], [goal_xy[1]], c="red", s=50, marker="x", label="goal")
    axes[0].set_title("Occupancy + Guidance Cost")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_axis_off()
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(1.0 - occ, cmap="gray")
    _plot_exploration(axes[1], getattr(res_base, "expanded_trace_xy", None), color="deepskyblue")
    if res_base.path:
        bp = chaikin_smooth_path(res_base.path, iterations=smooth_iter)
        axes[1].plot(
            bp[:, 0],
            bp[:, 1],
            color="dodgerblue",
            linewidth=2.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
    axes[1].scatter([start_xy[0]], [start_xy[1]], c="lime", s=45, marker="o")
    axes[1].scatter([goal_xy[0]], [goal_xy[1]], c="red", s=45, marker="x")
    axes[1].set_title(
        f"Baseline\\n"
        f"success={res_base.success}, expanded={res_base.expanded_nodes}, "
        f"runtime={res_base.runtime_ms:.2f}ms"
    )
    axes[1].set_axis_off()

    axes[2].imshow(1.0 - occ, cmap="gray")
    _plot_exploration(axes[2], getattr(res_guided, "expanded_trace_xy", None), color="goldenrod")
    if res_guided.path:
        gp = chaikin_smooth_path(res_guided.path, iterations=smooth_iter)
        axes[2].plot(
            gp[:, 0],
            gp[:, 1],
            color="orange",
            linewidth=2.2,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
        )
    axes[2].scatter([start_xy[0]], [start_xy[1]], c="lime", s=45, marker="o")
    axes[2].scatter([goal_xy[0]], [goal_xy[1]], c="red", s=45, marker="x")
    axes[2].set_title(
        f"Guided (lambda={lambda_guidance})\\n"
        f"success={res_guided.success}, expanded={res_guided.expanded_nodes}, "
        f"runtime={res_guided.runtime_ms:.2f}ms"
    )
    axes[2].set_axis_off()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _case_plot_path(
    base_path: Path,
    case_id: int,
    map_index: Optional[int],
    plot_dir: Optional[Path] = None,
) -> Path:
    if plot_dir is not None:
        out_dir = plot_dir
    else:
        # Default folder behavior for --plot-all-cases:
        # --plot-out outputs/foo.png -> outputs/foo/
        # --plot-out outputs/foo     -> outputs/foo/
        if base_path.suffix:
            out_dir = base_path.parent / base_path.stem
        else:
            out_dir = base_path
    map_tag = f"_map{map_index}" if map_index is not None else ""
    name = f"case{case_id:03d}{map_tag}.png"
    return out_dir / name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs guided Hybrid A* demo")
    p.add_argument("--num-problems", type=int, default=20)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--obstacle-prob", type=float, default=0.22)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, use nondeterministic randomness.",
    )
    p.add_argument("--lambda-guidance", type=float, default=2.0)
    p.add_argument("--occ-npz", type=Path, default=None, help="Optional map npz to replace random maps")
    p.add_argument("--map-index", type=int, default=0, help="Index used when arrays in npz are batched")
    p.add_argument("--random-map-index", action="store_true", help="Sample map index randomly each problem.")
    p.add_argument(
        "--occ-semantics",
        type=str,
        default="auto",
        choices=["auto", "obstacle1", "passable1"],
        help="Semantics of occ array in npz: obstacle1 means 1=obstacle, passable1 means 1=free.",
    )
    p.add_argument(
        "--random-start-goal",
        action="store_true",
        help="Ignore start/goal hints in npz and sample random free start+goal.",
    )
    p.add_argument(
        "--min-start-goal-dist",
        type=float,
        default=0.0,
        help="Minimum Euclidean distance between start and goal in grid cells.",
    )
    p.add_argument("--start", nargs=2, type=int, default=None, metavar=("X", "Y"))
    p.add_argument("--goal", nargs=2, type=int, default=None, metavar=("X", "Y"))
    p.add_argument("--start-yaw-deg", type=float, default=None)
    p.add_argument("--goal-yaw-deg", type=float, default=None)

    p.add_argument("--ckpt", type=str, default=None, help="Optional guidance encoder checkpoint")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--invert-guidance-cost",
        action="store_true",
        help="Invert inferred guidance cost as (1-cost). Useful for legacy checkpoints trained with opposite semantics.",
    )
    p.add_argument("--yaw-bins", type=int, default=72)
    p.add_argument("--n-steer", type=int, default=5)
    p.add_argument("--motion-step", type=float, default=0.5)
    p.add_argument("--primitive-length", type=float, default=2.5)
    p.add_argument("--wheel-base", type=float, default=2.7)
    p.add_argument("--max-steer", type=float, default=0.60)
    p.add_argument(
        "--guidance-integration-mode",
        type=str,
        default="g_cost",
        choices=["g_cost", "heuristic_bias", "heuristic_bonus"],
        help="How guidance affects planner search: accumulate into g-cost, add raw heuristic bias, or only reward low-cost regions.",
    )
    p.add_argument(
        "--normalize-guidance-cost",
        dest="normalize_guidance_cost",
        action="store_true",
        help="Normalize guidance cost by free-space percentiles (default: enabled).",
    )
    p.add_argument(
        "--no-normalize-guidance-cost",
        dest="normalize_guidance_cost",
        action="store_false",
        help="Disable guidance cost normalization.",
    )
    p.add_argument("--guidance-norm-p-low", type=float, default=5.0)
    p.add_argument("--guidance-norm-p-high", type=float, default=95.0)
    p.add_argument("--guidance-clip-low", type=float, default=0.05)
    p.add_argument("--guidance-clip-high", type=float, default=0.95)
    p.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Sigmoid-logit temperature for planner-side guidance shaping. <1 sharpens.",
    )
    p.add_argument(
        "--guidance-power",
        type=float,
        default=1.0,
        help="Planner-side power transform on normalized guidance cost. >1 emphasizes low-cost corridors.",
    )
    p.add_argument(
        "--guidance-bonus-threshold",
        type=float,
        default=0.5,
        help="Only used in heuristic_bonus mode. Costs below this threshold receive queue-priority bonus.",
    )
    p.add_argument(
        "--allow-reverse",
        dest="allow_reverse",
        action="store_true",
        help="Enable reverse expansion (default: enabled).",
    )
    p.add_argument(
        "--no-allow-reverse",
        dest="allow_reverse",
        action="store_false",
        help="Disable reverse expansion.",
    )
    p.set_defaults(allow_reverse=True, normalize_guidance_cost=True)
    p.add_argument("--strict-goal-pose", action="store_true")
    p.add_argument(
        "--disable-rs-shot",
        action="store_true",
        help="Disable RS-shot (only relevant when --strict-goal-pose is enabled).",
    )
    p.add_argument("--rs-shot-trigger-dist", type=float, default=5.0)
    p.add_argument("--rs-sample-ds", type=float, default=0.25)
    p.add_argument("--rs-endpoint-tol", type=float, default=1e-3)
    p.add_argument("--rs-max-iter", type=int, default=28)
    p.add_argument("--max-expansions", type=int, default=80000)
    p.add_argument("--goal-tolerance-yaw-deg", type=float, default=5.0)
    p.add_argument("--plot-out", type=Path, default=None, help="Optional output PNG path")
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Optional output directory for --plot-all-cases. If omitted, auto-derived from --plot-out.",
    )
    p.add_argument(
        "--plot-all-cases",
        action="store_true",
        help="If set, save one plot per case into a folder (default derived from --plot-out, or --plot-dir).",
    )
    p.add_argument(
        "--plot-smooth-iter",
        type=int,
        default=2,
        help="Chaikin smoothing iterations for display only",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.seed is None:
        print("random_seed: nondeterministic")
    else:
        print(f"random_seed: {args.seed}")

    if args.plot_all_cases and args.plot_out is None and args.plot_dir is None:
        raise ValueError("--plot-all-cases requires --plot-out or --plot-dir")

    baseline = Stat()
    guided = Stat()
    case_details: List[CaseDetail] = []

    use_rs_shot = args.strict_goal_pose and (not args.disable_rs_shot)
    planner = GuidedHybridAstar(
        yaw_bins=args.yaw_bins,
        n_steer=args.n_steer,
        motion_step=args.motion_step,
        primitive_length=args.primitive_length,
        wheel_base=args.wheel_base,
        max_steer=args.max_steer,
        guidance_integration_mode=args.guidance_integration_mode,
        normalize_guidance_cost=args.normalize_guidance_cost,
        guidance_norm_p_low=args.guidance_norm_p_low,
        guidance_norm_p_high=args.guidance_norm_p_high,
        guidance_clip_low=args.guidance_clip_low,
        guidance_clip_high=args.guidance_clip_high,
        guidance_temperature=args.guidance_temperature,
        guidance_power=args.guidance_power,
        guidance_bonus_threshold=args.guidance_bonus_threshold,
        allow_reverse=args.allow_reverse,
        strict_goal_pose=args.strict_goal_pose,
        use_rs_shot=use_rs_shot,
        rs_shot_trigger_dist=args.rs_shot_trigger_dist,
        rs_sample_ds=args.rs_sample_ds,
        rs_endpoint_tol=args.rs_endpoint_tol,
        rs_max_iter=args.rs_max_iter,
        goal_tolerance_yaw_deg=args.goal_tolerance_yaw_deg,
        steer_penalty=0.2,
    )

    fixed_occ: Optional[np.ndarray] = None
    hints = MapHints()
    load_info: Optional[MapLoadInfo] = None
    npz_map_count = 1
    if args.occ_npz is not None:
        with np.load(args.occ_npz) as d:
            occ_key = _pick_occ_key(set(d.files), list(d.files))
            npz_map_count = _map_count_from_occ_array(np.asarray(d[occ_key]))
        if args.random_map_index:
            print(
                f"loaded map npz meta: {args.occ_npz} "
                f"(map_count={npz_map_count}, random_map_index=True)"
            )
        else:
            fixed_occ, hints, load_info = load_occ_and_hints_from_npz(
                args.occ_npz, map_index=args.map_index, occ_semantics=args.occ_semantics
            )
            print(
                f"loaded map npz: {args.occ_npz} shape={fixed_occ.shape} "
                f"(occ_key={load_info.occ_key}, semantics={load_info.occ_semantics})"
            )

    cli_start = (int(args.start[0]), int(args.start[1])) if args.start is not None else None
    cli_goal = (int(args.goal[0]), int(args.goal[1])) if args.goal is not None else None
    if args.random_start_goal and (cli_start is not None or cli_goal is not None):
        raise ValueError("--random-start-goal cannot be used with --start/--goal")
    start_yaw = (
        math.radians(float(args.start_yaw_deg))
        if args.start_yaw_deg is not None
        else (hints.start_yaw if hints.start_yaw is not None else None)
    )
    goal_yaw = (
        math.radians(float(args.goal_yaw_deg))
        if args.goal_yaw_deg is not None
        else (hints.goal_yaw if hints.goal_yaw is not None else None)
    )

    n_problems = int(args.num_problems)
    fixed_pair_by_hint = (
        fixed_occ is not None
        and (not args.random_map_index)
        and (not args.random_start_goal)
        and cli_start is None
        and cli_goal is None
        and hints.start_xy is not None
        and hints.goal_xy is not None
    )
    fixed_pair_by_cli = (
        fixed_occ is not None
        and (not args.random_map_index)
        and cli_start is not None
        and cli_goal is not None
    )
    if (fixed_pair_by_hint or fixed_pair_by_cli) and n_problems != 1:
        print("info: fixed map start/goal detected, forcing --num-problems=1")
        n_problems = 1

    plotted = False
    case_plot_base = args.plot_out if args.plot_out is not None else (args.plot_dir or Path("outputs/cases"))
    if args.plot_all_cases:
        preview_case_path = _case_plot_path(
            case_plot_base, case_id=0, map_index=None, plot_dir=args.plot_dir
        )
        print(f"plot_all_cases_dir: {preview_case_path.parent}")
    for i in range(n_problems):
        case_map_index: Optional[int] = None
        if args.occ_npz is None:
            occ, start_xy, goal_xy, start_pose, goal_pose = make_problem(
                size=args.size,
                obstacle_prob=args.obstacle_prob,
                rng=rng,
                min_start_goal_dist=args.min_start_goal_dist,
            )
        else:
            occ_i = fixed_occ
            hints_i = hints
            if args.random_map_index:
                sampled_index = int(rng.integers(0, max(1, npz_map_count)))
                case_map_index = sampled_index
                occ_i, hints_i, load_info_i = load_occ_and_hints_from_npz(
                    args.occ_npz, map_index=sampled_index, occ_semantics=args.occ_semantics
                )
                print(
                    f"problem={i:03d} map_index={sampled_index} "
                    f"(occ_key={load_info_i.occ_key}, semantics={load_info_i.occ_semantics})"
                )
            else:
                case_map_index = int(args.map_index)

            start_hint = None if args.random_start_goal else hints_i.start_xy
            goal_hint = None if args.random_start_goal else hints_i.goal_xy
            start_yaw_hint = (
                math.radians(float(args.start_yaw_deg))
                if args.start_yaw_deg is not None
                else (
                    None
                    if args.random_start_goal
                    else (hints_i.start_yaw if hints_i.start_yaw is not None else None)
                )
            )
            goal_yaw_hint = (
                math.radians(float(args.goal_yaw_deg))
                if args.goal_yaw_deg is not None
                else (
                    None
                    if args.random_start_goal
                    else (hints_i.goal_yaw if hints_i.goal_yaw is not None else None)
                )
            )

            occ, start_xy, goal_xy, start_pose, goal_pose = make_problem_on_given_map(
                occ=occ_i,
                rng=rng,
                start_xy=cli_start if cli_start is not None else start_hint,
                goal_xy=cli_goal if cli_goal is not None else goal_hint,
                start_yaw=start_yaw_hint,
                goal_yaw=goal_yaw_hint,
                min_start_goal_dist=args.min_start_goal_dist,
            )

        if args.ckpt is not None:
            cost_map = infer_cost_map(
                ckpt_path=args.ckpt,
                occ_map_numpy=occ,
                start_xy=start_xy,
                goal_xy=goal_xy,
                start_yaw=float(start_pose[2]),
                goal_yaw=float(goal_pose[2]),
                device=args.device,
            )
            if args.invert_guidance_cost:
                cost_map = clip_cost_map_with_obstacles(1.0 - cost_map, occ, obstacle_cost=1.0)
        else:
            cost_map = handcrafted_cost_to_goal(occ, goal_xy)

        res_base = planner.plan(
            occ_map=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost_map=cost_map,
            lambda_guidance=0.0,
            max_expansions=args.max_expansions,
        )
        baseline.update(
            res_base.success,
            res_base.expanded_nodes,
            res_base.runtime_ms,
            res_base.path_length,
        )

        res_guided = planner.plan(
            occ_map=occ,
            start_pose=start_pose,
            goal_pose=goal_pose,
            cost_map=cost_map,
            lambda_guidance=args.lambda_guidance,
            max_expansions=args.max_expansions,
        )
        guided.update(
            res_guided.success,
            res_guided.expanded_nodes,
            res_guided.runtime_ms,
            res_guided.path_length,
        )
        case_details.append(
            CaseDetail(
                case_id=i,
                map_index=case_map_index,
                start_xy=start_xy,
                goal_xy=goal_xy,
                baseline_success=bool(res_base.success),
                baseline_expanded=int(res_base.expanded_nodes),
                baseline_runtime_ms=float(res_base.runtime_ms),
                baseline_path_length=float(res_base.path_length),
                guided_success=bool(res_guided.success),
                guided_expanded=int(res_guided.expanded_nodes),
                guided_runtime_ms=float(res_guided.runtime_ms),
                guided_path_length=float(res_guided.path_length),
            )
        )

        print(
            f"problem={i:03d} "
            f"baseline(success={res_base.success}, expanded={res_base.expanded_nodes}, runtime_ms={res_base.runtime_ms:.3f}) "
            f"guided(success={res_guided.success}, expanded={res_guided.expanded_nodes}, runtime_ms={res_guided.runtime_ms:.3f})"
        )

        if args.plot_out is not None or (args.plot_all_cases and args.plot_dir is not None):
            if args.plot_all_cases:
                out_i = _case_plot_path(
                    case_plot_base,
                    case_id=i,
                    map_index=case_map_index,
                    plot_dir=args.plot_dir,
                )
                save_case_plot(
                    occ=occ,
                    cost_map=cost_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    res_base=res_base,
                    res_guided=res_guided,
                    lambda_guidance=args.lambda_guidance,
                    out_path=out_i,
                    smooth_iter=args.plot_smooth_iter,
                )
                print(f"saved plot: {out_i}")
            elif (not plotted) and (res_base.success or res_guided.success):
                save_case_plot(
                    occ=occ,
                    cost_map=cost_map,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    res_base=res_base,
                    res_guided=res_guided,
                    lambda_guidance=args.lambda_guidance,
                    out_path=args.plot_out,
                    smooth_iter=args.plot_smooth_iter,
                )
                print(f"saved plot: {args.plot_out}")
                plotted = True

    print("\n=== Summary ===")
    print(f"baseline: success_rate={baseline.success / max(1, baseline.total):.3f}, "
          f"expanded_nodes={baseline.expanded_nodes / max(1, baseline.total):.1f}, "
          f"runtime_ms={baseline.runtime_ms / max(1, baseline.total):.3f}")
    print(f"guided(lambda={args.lambda_guidance}): success_rate={guided.success / max(1, guided.total):.3f}, "
          f"expanded_nodes={guided.expanded_nodes / max(1, guided.total):.1f}, "
          f"runtime_ms={guided.runtime_ms / max(1, guided.total):.3f}")
    common_n, common_b_len, common_g_len = summarize_common_path_length(case_details)
    print(
        "path_length(common_success_only): "
        f"count={common_n}, baseline={common_b_len:.3f}, guided={common_g_len:.3f}"
    )
    print("\n=== Per-Case Details ===")
    for row in case_details:
        map_text = "na" if row.map_index is None else str(row.map_index)
        print(
            f"case={row.case_id:03d} map_index={map_text} start={row.start_xy} goal={row.goal_xy} "
            f"baseline(success={row.baseline_success}, expanded={row.baseline_expanded}, "
            f"runtime_ms={row.baseline_runtime_ms:.3f}, path_length={row.baseline_path_length:.3f}) "
            f"guided(success={row.guided_success}, expanded={row.guided_expanded}, "
            f"runtime_ms={row.guided_runtime_ms:.3f}, path_length={row.guided_path_length:.3f})"
        )


if __name__ == "__main__":
    main()
