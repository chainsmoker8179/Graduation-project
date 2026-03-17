from pathlib import Path

from scripts.plotting.loaders import load_ranking_mechanism_data, load_significance_summary_data


def test_load_significance_summary_data_reads_generated_csv(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    target = report_dir / "lstm_single_model_evidence"
    target.mkdir(parents=True)
    (target / "significance_daily_metrics.csv").write_text(
        "metric,comparison,delta_mean,p_value\nexcess_return,partial_fgsm_vs_partial_clean,-0.1,0.01\n",
        encoding="utf-8",
    )

    df = load_significance_summary_data(report_dir)

    assert list(df["metric"]) == ["excess_return"]
    assert list(df["source"]) == ["daily"]


def test_load_ranking_mechanism_data_stacks_overlap_and_correlation(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    target = report_dir / "lstm_single_model_evidence"
    target.mkdir(parents=True)
    (target / "ranking_overlap_daily.csv").write_text(
        "datetime,topk,topk_overlap,seed,comparison\n2025-01-02,50,0.8,0,partial_fgsm_vs_partial_clean\n",
        encoding="utf-8",
    )
    (target / "ranking_correlation_daily.csv").write_text(
        "datetime,spearman,seed,comparison\n2025-01-02,0.7,0,partial_fgsm_vs_partial_clean\n",
        encoding="utf-8",
    )
    (target / "rank_shift_summary.csv").write_text(
        "datetime,rank_shift_mean,rank_shift_abs_mean,rank_shift_abs_max,seed,comparison\n2025-01-02,0.0,3.2,15.0,0,partial_fgsm_vs_partial_clean\n",
        encoding="utf-8",
    )

    df = load_ranking_mechanism_data(report_dir)

    assert set(df["metric"]) == {"topk_overlap", "spearman", "rank_shift_abs_mean"}
    assert set(df["attacker"]) == {"FGSM"}
