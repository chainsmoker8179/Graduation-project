import pandas as pd

from scripts.analysis.lstm_attack_significance import summarize_paired_significance


def test_summarize_paired_significance_reports_effect_and_pvalue() -> None:
    df = pd.DataFrame(
        {
            "seed": [0, 0, 0, 1, 1, 1],
            "datetime": ["2025-01-02", "2025-01-03", "2025-01-06"] * 2,
            "partial_clean_excess_return_with_cost": [0.02, 0.01, 0.03, 0.01, 0.00, 0.02],
            "partial_fgsm_excess_return_with_cost": [0.00, -0.01, 0.01, -0.01, -0.02, 0.00],
        }
    )

    out = summarize_paired_significance(
        df,
        baseline_col="partial_clean_excess_return_with_cost",
        attacked_col="partial_fgsm_excess_return_with_cost",
        metric_name="excess_return_with_cost",
    )

    assert out["metric"] == "excess_return_with_cost"
    assert out["comparison"] == "partial_fgsm_vs_partial_clean"
    assert out["delta_mean"] < 0
    assert out["delta_median"] < 0
    assert 0.0 <= out["p_value"] <= 1.0


def test_summarize_paired_significance_drops_nan_before_wilcoxon() -> None:
    df = pd.DataFrame(
        {
            "partial_clean_rank_ic": [0.10, 0.11, 0.12],
            "partial_fgsm_rank_ic": [0.08, float("nan"), 0.07],
        }
    )

    out = summarize_paired_significance(
        df,
        baseline_col="partial_clean_rank_ic",
        attacked_col="partial_fgsm_rank_ic",
        metric_name="rank_ic",
    )

    assert out["sample_size"] == 2
    assert 0.0 <= out["p_value"] <= 1.0
