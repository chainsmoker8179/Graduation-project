from scripts.build_lstm_single_model_evidence_report import render_report


def test_render_report_contains_core_sections() -> None:
    text = render_report(
        significance_rows=[{"metric": "rank_ic", "p_value": 0.001, "comparison": "partial_fgsm_vs_partial_clean", "delta_mean": -0.01}],
        ranking_rows=[{"metric": "topk_overlap", "attacker": "FGSM", "value_mean": 0.8}],
        main_rows=[{"metric": "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost", "mean": -0.2}],
        ratio_rows=[{"ratio_label": "5%", "metric": "partial_fgsm_minus_partial_clean__rank_ic_mean", "mean": -0.01}],
        bootstrap_payload={"partial_fgsm_vs_partial_clean": {"annualized_excess_return_with_cost": {"ci95_lower": -0.3, "ci95_upper": -0.1}}},
    )

    assert "统计显著性" in text
    assert "排序机制" in text
    assert "主文放置建议" in text
