"""Case mining helpers for replay and curriculum selection."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def positive_rank_scores(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    scores = np.zeros_like(arr, dtype=np.float32)
    positive = arr > 0.0
    if not np.any(positive):
        return scores
    pos_idx = np.flatnonzero(positive)
    pos_vals = arr[pos_idx]
    order = np.argsort(pos_vals, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    if len(order) == 1:
        ranks[order] = 1.0
    else:
        ranks[order] = np.linspace(1.0 / len(order), 1.0, num=len(order), dtype=np.float32)
    scores[pos_idx] = ranks
    return scores


def build_case_priority_rows(
    rows: Iterable[dict],
    *,
    regression_weight: float = 1.0,
    oracle_gap_weight: float = 1.0,
    min_regression: float = 0.0,
    min_oracle_gap: float = 0.0,
) -> List[dict]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        return []

    learned_minus_improved = np.array(
        [max(float(row.get("learned_minus_improved", 0.0)), 0.0) for row in materialized],
        dtype=np.float32,
    )
    oracle_gap = np.array(
        [max(-float(row.get("oracle_minus_improved", 0.0)), 0.0) for row in materialized],
        dtype=np.float32,
    )

    reg_scores = positive_rank_scores(learned_minus_improved)
    oracle_scores = positive_rank_scores(oracle_gap)

    out = []
    for idx, row in enumerate(materialized):
        reg = float(learned_minus_improved[idx])
        gap = float(oracle_gap[idx])
        if reg < float(min_regression) and gap < float(min_oracle_gap):
            continue
        selected_by = (
            "both"
            if reg >= float(min_regression) and gap >= float(min_oracle_gap)
            else ("regression" if reg >= float(min_regression) else "oracle_gap")
        )
        priority = float(regression_weight) * float(reg_scores[idx]) + float(oracle_gap_weight) * float(oracle_scores[idx])
        merged = dict(row)
        merged["oracle_gap"] = gap
        merged["regression_score"] = float(reg_scores[idx])
        merged["oracle_gap_score"] = float(oracle_scores[idx])
        merged["priority_score"] = float(priority)
        merged["selected_by"] = selected_by
        out.append(merged)

    out.sort(key=lambda row: float(row["priority_score"]), reverse=True)
    return out
