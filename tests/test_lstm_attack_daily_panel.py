from pathlib import Path

from scripts.analysis.lstm_attack_daily_panel import load_multiseed_daily_panel


def test_load_multiseed_daily_panel_stacks_seed_daily_files(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    seed0 = report_root / "partial_attack_backtest_multiseed_ratio5_union" / "seed_0"
    seed1 = report_root / "partial_attack_backtest_multiseed_ratio5_union" / "seed_1"
    seed0.mkdir(parents=True)
    seed1.mkdir(parents=True)
    payload = (
        "datetime,partial_clean_excess_return_with_cost,partial_fgsm_excess_return_with_cost,"
        "partial_pgd_excess_return_with_cost,partial_clean_rank_ic,partial_fgsm_rank_ic,"
        "partial_pgd_rank_ic\n"
        "2025-01-02,0.01,-0.02,-0.03,0.10,0.06,0.04\n"
    )
    (seed0 / "daily_comparison.csv").write_text(payload, encoding="utf-8")
    (seed1 / "daily_comparison.csv").write_text(payload, encoding="utf-8")

    df = load_multiseed_daily_panel(report_root)

    assert list(df["seed"]) == [0, 1]
    assert "fgsm_minus_partial_clean_excess_return_with_cost" in df.columns
    assert "pgd_minus_partial_clean_rank_ic" in df.columns
