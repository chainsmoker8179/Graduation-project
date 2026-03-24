from pathlib import Path

import pandas as pd

from scripts.build_partial_attack_union_asset import build_union_requested_keys
from scripts.run_partial_attack_backtest_multiseed import (
    build_multiseed_pgd_vs_fgsm_summary,
    build_multiseed_summary_stats,
)


def _make_reference_scores() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-02", "BBB"),
            ("2025-01-02", "CCC"),
            ("2025-01-03", "AAA"),
            ("2025-01-03", "BBB"),
            ("2025-01-03", "CCC"),
        ],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=index)


def test_build_union_requested_keys_merges_multiple_seed_masks() -> None:
    reference_scores = _make_reference_scores()

    keys = build_union_requested_keys(reference_scores, attack_ratio=0.5, seeds=[0, 1])

    assert keys == sorted(keys)
    assert len(keys) >= 4
    assert all(len(key) == 2 for key in keys)
    assert all(key[0].endswith("00:00:00") for key in keys)


def test_build_multiseed_summary_contains_expected_delta_rows() -> None:
    seed_metrics = pd.DataFrame(
        [
            {
                "seed": 0,
                "attackable_count": 10,
                "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost": -0.20,
                "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost": -0.30,
            },
            {
                "seed": 1,
                "attackable_count": 12,
                "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost": -0.10,
                "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost": -0.40,
            },
        ]
    )

    summary = build_multiseed_summary_stats(seed_metrics)

    assert "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost" in summary["metric"].tolist()
    row = summary.set_index("metric").loc["partial_pgd_minus_partial_clean__annualized_excess_return_with_cost"]
    assert abs(row["mean"] + 0.35) < 1e-12
    assert row["negative_ratio"] == 1.0


def test_build_multiseed_pgd_vs_fgsm_summary_compares_negative_strength() -> None:
    seed_metrics = pd.DataFrame(
        [
            {
                "seed": 0,
                "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost": -0.20,
                "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost": -0.30,
                "partial_fgsm_minus_partial_clean__rank_ic_mean": -0.010,
                "partial_pgd_minus_partial_clean__rank_ic_mean": -0.012,
            },
            {
                "seed": 1,
                "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost": -0.25,
                "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost": -0.35,
                "partial_fgsm_minus_partial_clean__rank_ic_mean": -0.011,
                "partial_pgd_minus_partial_clean__rank_ic_mean": -0.013,
            },
        ]
    )

    summary = build_multiseed_pgd_vs_fgsm_summary(
        seed_metrics,
        metrics=["annualized_excess_return_with_cost", "rank_ic_mean"],
    )

    assert summary["annualized_excess_return_with_cost"]["pgd_more_negative_ratio"] == 1.0
    assert summary["rank_ic_mean"]["pgd_mean"] < summary["rank_ic_mean"]["fgsm_mean"]
