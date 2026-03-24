from scripts.build_physical_stat_significance_report import render_report


def test_render_report_includes_daily_significance_and_bootstrap_interval() -> None:
    text = render_report(
        summary_rows=[
            {"metric": "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost", "mean": -0.0001},
            {"metric": "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost", "mean": -0.0920},
            {"metric": "partial_fgsm_minus_partial_clean__rank_ic_mean", "mean": -0.0113},
            {"metric": "partial_pgd_minus_partial_clean__rank_ic_mean", "mean": -0.0111},
        ],
        significance_rows=[
            {
                "comparison": "partial_fgsm_vs_partial_clean",
                "metric": "excess_return_with_cost",
                "delta_mean": -0.0001,
                "p_value": 0.12,
            },
            {
                "comparison": "partial_pgd_vs_partial_clean",
                "metric": "excess_return_with_cost",
                "delta_mean": -0.0004,
                "p_value": 0.01,
            },
            {
                "comparison": "partial_fgsm_vs_partial_clean",
                "metric": "rank_ic",
                "delta_mean": -0.0113,
                "p_value": 1e-8,
            },
            {
                "comparison": "partial_pgd_vs_partial_clean",
                "metric": "rank_ic",
                "delta_mean": -0.0111,
                "p_value": 1e-6,
            },
        ],
        bootstrap_payload={
            "partial_fgsm_vs_partial_clean": {
                "annualized_excess_return_with_cost": {
                    "ci95_lower": -0.06,
                    "ci95_upper": 0.07,
                }
            },
            "partial_pgd_vs_partial_clean": {
                "annualized_excess_return_with_cost": {
                    "ci95_lower": -0.15,
                    "ci95_upper": -0.03,
                }
            }
        },
    )

    assert "显著性检验报告" in text
    assert "partial_fgsm_vs_partial_clean" in text
    assert "p=1.20e-01" in text
    assert "[-0.1500, -0.0300]" in text
    assert "FGSM 在排序层面的扰动显著" in text
    assert "收益打击不具有统计显著性" in text
    assert "PGD 在收益与排序两个层面都具有统计显著性" in text
