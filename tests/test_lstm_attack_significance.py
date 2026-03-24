from pathlib import Path

import pandas as pd

from scripts.analysis.lstm_attack_daily_panel import load_multiseed_daily_panel
from scripts.analysis.lstm_attack_significance import summarize_paired_significance
from scripts.run_lstm_attack_significance import main as run_significance_main


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


def test_load_multiseed_daily_panel_accepts_custom_seed_root(tmp_path: Path) -> None:
    seed_root = tmp_path / "reports" / "physical_stat_case"
    seed_dir = seed_root / "seed_0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "daily_comparison.csv").write_text(
        "\n".join(
            [
                "datetime,partial_clean_excess_return_with_cost,partial_fgsm_excess_return_with_cost,partial_clean_rank_ic,partial_fgsm_rank_ic",
                "2025-01-02,0.010,0.005,0.100,0.090",
                "2025-01-03,0.020,0.000,0.110,0.095",
            ]
        ),
        encoding="utf-8",
    )

    panel = load_multiseed_daily_panel(tmp_path / "reports", seed_root_path=seed_root)

    assert list(panel["seed"]) == [0, 0]
    assert "fgsm_minus_partial_clean_excess_return_with_cost" in panel.columns
    assert panel["fgsm_minus_partial_clean_excess_return_with_cost"].tolist() == [-0.005, -0.02]


def test_run_significance_main_writes_outputs_for_custom_seed_root(tmp_path: Path, monkeypatch) -> None:
    seed_root = tmp_path / "reports" / "physical_stat_case"
    seed_dir = seed_root / "seed_0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "daily_comparison.csv").write_text(
        "\n".join(
            [
                "datetime,partial_clean_excess_return_with_cost,partial_fgsm_excess_return_with_cost,partial_pgd_excess_return_with_cost,partial_clean_rank_ic,partial_fgsm_rank_ic,partial_pgd_rank_ic",
                "2025-01-02,0.010,0.005,0.000,0.100,0.090,0.080",
                "2025-01-03,0.020,0.000,-0.010,0.110,0.095,0.085",
                "2025-01-06,0.015,0.004,-0.005,0.120,0.100,0.090",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_lstm_attack_significance.py",
            "--report-root",
            str(tmp_path / "reports"),
            "--seed-root-path",
            str(seed_root),
            "--out-dir",
            str(out_dir),
            "--bootstrap-reps",
            "20",
            "--block-size",
            "2",
            "--random-seed",
            "0",
        ],
    )

    run_significance_main()

    assert (out_dir / "significance_daily_metrics.csv").exists()
    assert (out_dir / "significance_block_bootstrap.json").exists()
