"""Mine high-value replay cases from residual A* comparison metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from neural_astar.utils.case_mining import build_case_priority_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine replay cases from case_metrics.csv")
    p.add_argument("--case-metrics-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--top-fraction", type=float, default=0.15)
    p.add_argument("--max-count", type=int, default=256)
    p.add_argument("--regression-weight", type=float, default=1.0)
    p.add_argument("--oracle-gap-weight", type=float, default=1.0)
    p.add_argument("--min-regression", type=float, default=32.0)
    p.add_argument("--min-oracle-gap", type=float, default=64.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.case_metrics_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    prioritized = build_case_priority_rows(
        rows,
        regression_weight=args.regression_weight,
        oracle_gap_weight=args.oracle_gap_weight,
        min_regression=args.min_regression,
        min_oracle_gap=args.min_oracle_gap,
    )
    if not prioritized:
        raise SystemExit("No rows matched the requested thresholds.")

    keep = len(prioritized) if args.top_fraction <= 0.0 else max(1, int(round(float(args.top_fraction) * len(prioritized))))
    if int(args.max_count) > 0:
        keep = min(keep, int(args.max_count))
    selected = prioritized[:keep]

    fieldnames = [
        "idx",
        "learned_minus_improved",
        "oracle_minus_improved",
        "oracle_gap",
        "regression_score",
        "oracle_gap_score",
        "priority_score",
        "selected_by",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    print(
        f"selected={len(selected)} "
        f"from_total={len(rows)} "
        f"top_priority={float(selected[0]['priority_score']):.4f} "
        f"saved_csv={args.out_csv}"
    )


if __name__ == "__main__":
    main()
