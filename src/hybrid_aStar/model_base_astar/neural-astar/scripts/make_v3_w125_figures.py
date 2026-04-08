from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "A*": "#56B4E9",
    "Improved A*": "#009E73",
    "V3": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figure set for the original V3 w=1.25 checkpoint.")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--case-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_metric_triptych(output_path: Path, summary_map: dict[str, dict[str, float]]) -> None:
    methods = ["A*", "Improved A*", "V3"]
    metrics = [
        ("expanded_nodes", "Expanded Nodes"),
        ("runtime_ms", "Runtime (ms)"),
        ("path_length", "Path Length"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), dpi=180)
    for ax, (metric_key, title) in zip(axes, metrics):
        vals = [summary_map[m][metric_key] for m in methods]
        colors = [PALETTE[m] for m in methods]
        ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=15)
        for x, v in enumerate(vals):
            ax.text(x, v + max(vals) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Original V3 (w=1.25) on Original Grid Test400", fontsize=15, weight="bold")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_hist(output_path: Path, values: np.ndarray, title: str, xlabel: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    ax.hist(values, bins=24, color=color, edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="#334155", linestyle="--", linewidth=1.1)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Case Count", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_scatter(output_path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=180)
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    ax.scatter(x, y, s=18, alpha=0.7, color=PALETTE["V3"], edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#334155", linewidth=1.1)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(linestyle="--", alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_wtl(output_path: Path, diffs: np.ndarray) -> None:
    wins = int(np.sum(diffs < 0.0))
    ties = int(np.sum(diffs == 0.0))
    losses = int(np.sum(diffs > 0.0))
    labels = ["Win", "Tie", "Loss"]
    vals = [wins, ties, losses]
    colors = ["#009E73", "#94a3b8", "#D55E00"]
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=180)
    ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
    ax.set_title("V3 vs A*: Win / Tie / Loss", fontsize=14, weight="bold")
    ax.set_ylabel("Case Count", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals) * 0.02, f"{v}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_top_regressions(output_path: Path, top_rows: list[tuple[int, float]]) -> None:
    labels = [f"idx {idx}" for idx, _ in top_rows]
    vals = [v for _, v in top_rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=180)
    ax.bar(labels, vals, color=PALETTE["V3"], edgecolor="white", linewidth=0.8, width=0.65)
    ax.set_title("Top Regression Cases of V3 vs A*", fontsize=14, weight="bold")
    ax.set_ylabel("Extra Expanded Nodes over A*", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals) * 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = _load_rows(args.summary_csv)
    case_rows = _load_rows(args.case_csv)
    summary = {row["label"]: row for row in summary_rows}

    summary_map = {
        "A*": {
            "expanded_nodes": float(summary["traditional_astar"]["expanded_nodes"]),
            "runtime_ms": float(summary["traditional_astar"]["runtime_ms"]),
            "path_length": float(summary["traditional_astar"]["path_length"]),
        },
        "Improved A*": {
            "expanded_nodes": float(summary["improved_astar"]["expanded_nodes"]),
            "runtime_ms": float(summary["improved_astar"]["runtime_ms"]),
            "path_length": float(summary["improved_astar"]["path_length"]),
        },
        "V3": {
            "expanded_nodes": float(summary["replay32_v3"]["expanded_nodes"]),
            "runtime_ms": float(summary["replay32_v3"]["runtime_ms"]),
            "path_length": float(summary["replay32_v3"]["path_length"]),
        },
    }

    astar_exp = np.array([float(r["traditional_astar_expanded"]) for r in case_rows], dtype=np.float32)
    astar_run = np.array([float(r["traditional_astar_runtime_ms"]) for r in case_rows], dtype=np.float32)
    astar_len = np.array([float(r["traditional_astar_path_length"]) for r in case_rows], dtype=np.float32)
    v3_exp = np.array([float(r["replay32_v3_expanded"]) for r in case_rows], dtype=np.float32)
    v3_run = np.array([float(r["replay32_v3_runtime_ms"]) for r in case_rows], dtype=np.float32)
    v3_len = np.array([float(r["replay32_v3_path_length"]) for r in case_rows], dtype=np.float32)

    exp_gain = astar_exp - v3_exp
    runtime_gain = astar_run - v3_run
    path_delta = v3_len - astar_len
    regressions = v3_exp - astar_exp

    top_rows = sorted(
        [
            (int(r["idx"]), float(r["replay32_v3_expanded"]) - float(r["traditional_astar_expanded"]))
            for r in case_rows
            if float(r["replay32_v3_expanded"]) - float(r["traditional_astar_expanded"]) > 0.0
        ],
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    _save_metric_triptych(args.output_dir / "01_summary_triptych.png", summary_map)
    _save_scatter(
        args.output_dir / "02_astar_vs_v3_scatter.png",
        astar_exp,
        v3_exp,
        "Case-wise Expanded Nodes: A* vs V3",
        "A* Expanded Nodes",
        "V3 Expanded Nodes",
    )
    _save_hist(
        args.output_dir / "03_expanded_gain_hist.png",
        exp_gain,
        "Expanded-Node Gain of V3 over A*",
        "A* Expanded Nodes - V3 Expanded Nodes",
        PALETTE["V3"],
    )
    _save_hist(
        args.output_dir / "04_runtime_gain_hist.png",
        runtime_gain,
        "Runtime Gain of V3 over A*",
        "A* Runtime - V3 Runtime (ms)",
        "#009E73",
    )
    _save_hist(
        args.output_dir / "05_path_delta_hist.png",
        path_delta,
        "Path-Length Delta of V3 vs A*",
        "V3 Path Length - A* Path Length",
        "#CC79A7",
    )
    _save_wtl(args.output_dir / "06_win_tie_loss.png", regressions)
    if top_rows:
        _save_top_regressions(args.output_dir / "07_top_regression_cases.png", top_rows)

    with (args.output_dir / "summary_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["expanded_nodes_v3", summary_map["V3"]["expanded_nodes"]])
        w.writerow(["runtime_ms_v3", summary_map["V3"]["runtime_ms"]])
        w.writerow(["path_length_v3", summary_map["V3"]["path_length"]])
        w.writerow(["reg_count", int(np.sum(regressions > 0.0))])
        w.writerow(["big10_count", int(np.sum(regressions >= 10.0))])
        w.writerow(["big20_count", int(np.sum(regressions >= 20.0))])
        w.writerow(["max_reg", float(np.max(regressions))])
        w.writerow(["reg_sum", float(np.sum(regressions[regressions > 0.0]))])


if __name__ == "__main__":
    main()
