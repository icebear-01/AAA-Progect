from __future__ import annotations

from neural_astar.utils.case_mining import (
    build_case_priority_rows,
    positive_rank_scores,
)


def test_positive_rank_scores_only_scores_positive_entries():
    scores = positive_rank_scores([0.0, 5.0, -2.0, 10.0])

    assert scores.tolist() == [0.0, 0.5, 0.0, 1.0]


def test_build_case_priority_rows_combines_regression_and_oracle_gap():
    rows = [
        {"idx": "0", "learned_minus_improved": "80", "oracle_minus_improved": "-20"},
        {"idx": "1", "learned_minus_improved": "0", "oracle_minus_improved": "-200"},
        {"idx": "2", "learned_minus_improved": "120", "oracle_minus_improved": "-250"},
        {"idx": "3", "learned_minus_improved": "-10", "oracle_minus_improved": "0"},
    ]

    prioritized = build_case_priority_rows(
        rows,
        regression_weight=1.0,
        oracle_gap_weight=1.0,
        min_regression=32.0,
        min_oracle_gap=64.0,
    )

    assert [row["idx"] for row in prioritized] == ["2", "0", "1"]
    assert prioritized[0]["selected_by"] == "both"
    assert prioritized[1]["selected_by"] == "regression"
    assert prioritized[2]["selected_by"] == "oracle_gap"
    assert float(prioritized[0]["priority_score"]) > float(prioritized[1]["priority_score"])
