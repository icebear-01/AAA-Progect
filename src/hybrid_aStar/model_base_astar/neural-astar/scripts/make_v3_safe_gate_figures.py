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
    "V3-Safe": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate comparison figures for V3 safe-gating.")
    p.add_argument("--current-summary", type=Path, required=True)
    p.add_argument("--current-cases", type=Path, required=True)
    p.add_argument("--safe-summary", type=Path, required=True)
    p.add_argument("--safe-cases", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_metric_triptych(output_path: Path, summary_map: dict[str, dict[str, float]]) -> None:
    methods = ["A*", "Improved A*", "V3", "V3-Safe"]
    metrics = [
        ("expanded_nodes", "Expanded Nodes"),
        ("runtime_ms", "Runtime (ms)"),
        ("path_length", "Path Length"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=180)
    for ax, (metric_key, title) in zip(axes, metrics):
        vals = [summary_map[m][metric_key] for m in methods]
        colors = [PALETTE[m] for m in methods]
        ax.bar(methods, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.65)
        ax.set_title(title, fontsize=13, weight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=18)
        for x, v in enumerate(vals):
            ax.text(x, v + max(vals) * 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("V3 vs Safe-Gated V3 on Original Grid Test400", fontsize=15, weight="bold")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_astar_scatter(output_path: Path, astar: np.ndarray, current: np.ndarray, safe: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), dpi=180, sharex=True, sharey=True)
    lo = min(float(astar.min()), float(current.min()), float(safe.min()))
    hi = max(float(astar.max()), float(current.max()), float(safe.max()))
    series = [("V3", current), ("V3-Safe", safe)]
    for ax, (label, values) in zip(axes, series):
        ax.scatter(astar, values, s=16, alpha=0.7, color=PALETTE[label], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#334155", linewidth=1.1)
        ax.set_title(label, fontsize=13, weight="bold")
        ax.set_xlabel("A* Expanded Nodes", fontsize=11)
        ax.grid(linestyle="--", alpha=0.2)
    axes[0].set_ylabel("Method Expanded Nodes", fontsize=11)
    fig.suptitle("Case-wise Comparison Against A*", fontsize=15, weight="bold")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_regression_summary(output_path: Path, regression_stats: dict[str, dict[str, float]]) -> None:
    labels = ["Regression Count", "Count (>=10)", "Max Regression", "Regression Sum"]
    keys = ["reg_count", "big10_count", "max_reg", "reg_sum"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=180)
    current_vals = [regression_stats["V3"][k] for k in keys]
    safe_vals = [regression_stats["V3-Safe"][k] for k in keys]
    ax.bar(x - width / 2, current_vals, width=width, color=PALETTE["V3"], label="V3")
    ax.bar(x + width / 2, safe_vals, width=width, color=PALETTE["V3-Safe"], label="V3-Safe")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Regression Summary vs A*", fontsize=15, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for xpos, vals in [(x - width / 2, current_vals), (x + width / 2, safe_vals)]:
        for xx, vv in zip(xpos, vals):
            ax.text(xx, vv + max(current_vals + safe_vals) * 0.015, f"{vv:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_top_regressions(output_path: Path, top_rows: list[tuple[int, float, float]]) -> None:
    labels = [str(idx) for idx, _, _ in top_rows]
    current_vals = [curr for _, curr, _ in top_rows]
    safe_vals = [safe for _, _, safe in top_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    ax.bar(x - width / 2, current_vals, width=width, color=PALETTE["V3"], label="V3")
    ax.bar(x + width / 2, safe_vals, width=width, color=PALETTE["V3-Safe"], label="V3-Safe")
    ax.set_xticks(x)
    ax.set_xticklabels([f"idx {lab}" for lab in labels])
    ax.set_title("Top Regression Cases vs A*", fontsize=15, weight="bold")
    ax.set_ylabel("Extra Expanded Nodes over A*", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_summary_rows = _load_rows(args.current_summary)
    safe_summary_rows = _load_rows(args.safe_summary)
    current_cases = _load_rows(args.current_cases)
    safe_cases = _load_rows(args.safe_cases)

    current_summary = {row["label"]: row for row in current_summary_rows}
    safe_summary = {row["label"]: row for row in safe_summary_rows}

    summary_map = {
        "A*": {
            "expanded_nodes": float(current_summary["traditional_astar"]["expanded_nodes"]),
            "runtime_ms": float(current_summary["traditional_astar"]["runtime_ms"]),
            "path_length": float(current_summary["traditional_astar"]["path_length"]),
        },
        "Improved A*": {
            "expanded_nodes": float(current_summary["improved_astar"]["expanded_nodes"]),
            "runtime_ms": float(current_summary["improved_astar"]["runtime_ms"]),
            "path_length": float(current_summary["improved_astar"]["path_length"]),
        },
        "V3": {
            "expanded_nodes": float(current_summary["replay32_v3"]["expanded_nodes"]),
            "runtime_ms": float(current_summary["replay32_v3"]["runtime_ms"]),
            "path_length": float(current_summary["replay32_v3"]["path_length"]),
        },
        "V3-Safe": {
            "expanded_nodes": float(safe_summary["safe_v3"]["expanded_nodes"]),
            "runtime_ms": float(safe_summary["safe_v3"]["runtime_ms"]),
            "path_length": float(safe_summary["safe_v3"]["path_length"]),
        },
    }

    current_by_idx = {int(r["idx"]): r for r in current_cases}
    safe_by_idx = {int(r["idx"]): r for r in safe_cases}
    common_idx = sorted(set(current_by_idx) & set(safe_by_idx))

    astar = np.array([float(current_by_idx[i]["traditional_astar_expanded"]) for i in common_idx], dtype=np.float32)
    current = np.array([float(current_by_idx[i]["replay32_v3_expanded"]) for i in common_idx], dtype=np.float32)
    safe = np.array([float(safe_by_idx[i]["safe_v3_expanded"]) for i in common_idx], dtype=np.float32)

    current_diff = current - astar
    safe_diff = safe - astar

    regression_stats = {
        "V3": {
            "reg_count": float(np.sum(current_diff > 0.0)),
            "big10_count": float(np.sum(current_diff >= 10.0)),
            "max_reg": float(np.max(current_diff)),
            "reg_sum": float(np.sum(current_diff[current_diff > 0.0])),
        },
        "V3-Safe": {
            "reg_count": float(np.sum(safe_diff > 0.0)),
            "big10_count": float(np.sum(safe_diff >= 10.0)),
            "max_reg": float(np.max(safe_diff)),
            "reg_sum": float(np.sum(safe_diff[safe_diff > 0.0])),
        },
    }

    top_idx = np.argsort(current_diff)[-5:][::-1]
    top_rows = [(common_idx[int(k)], float(current_diff[int(k)]), float(safe_diff[int(k)])) for k in top_idx]

    _save_metric_triptych(args.output_dir / "01_safe_gate_summary_triptych.png", summary_map)
    _save_astar_scatter(args.output_dir / "02_astar_vs_v3_safe_scatter.png", astar, current, safe)
    _save_regression_summary(args.output_dir / "03_regression_summary.png", regression_stats)
    _save_top_regressions(args.output_dir / "04_top_regression_cases.png", top_rows)

    with (args.output_dir / "summary_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "expanded_nodes", "runtime_ms", "path_length", "reg_count", "big10_count", "max_reg", "reg_sum"])
        for label in ["V3", "V3-Safe"]:
            w.writerow(
                [
                    label,
                    summary_map[label]["expanded_nodes"],
                    summary_map[label]["runtime_ms"],
                    summary_map[label]["path_length"],
                    regression_stats[label]["reg_count"],
                    regression_stats[label]["big10_count"],
                    regression_stats[label]["max_reg"],
                    regression_stats[label]["reg_sum"],
                ]
            )


if __name__ == "__main__":
    main()
