import pandas as pd

from scripts.analysis.lstm_attack_ranking_diagnostics import compute_daily_topk_overlap


def test_compute_daily_topk_overlap_returns_fraction_per_day() -> None:
    scores = pd.DataFrame(
        {
            "datetime": ["2025-01-02"] * 4,
            "instrument": ["A", "B", "C", "D"],
            "baseline_score": [4.0, 3.0, 2.0, 1.0],
            "attacked_score": [4.0, 1.0, 3.0, 2.0],
        }
    )

    out = compute_daily_topk_overlap(scores, topk=2)

    assert out.loc[0, "topk_overlap"] == 0.5
