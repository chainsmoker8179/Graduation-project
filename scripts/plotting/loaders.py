from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.style import figure_data_dir


def _repo_root_from_report_dir(report_dir: Path) -> Path:
    return report_dir.resolve().parent


def _write_plot_data(df: pd.DataFrame, repo_root: Path, filename: str) -> None:
    out_path = figure_data_dir(repo_root) / filename
    df.to_csv(out_path, index=False)


def load_clean_alignment_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    versions = [
        ("expanded_v3", report_dir / "lstm_whitebox_attack_expanded_v3" / "attack_summary.json"),
        ("expanded_v6", report_dir / "lstm_whitebox_attack_expanded_v6" / "attack_summary.json"),
    ]
    rows: list[dict[str, float | str]] = []
    for version, path in versions:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "version": version,
                "spearman_to_reference": payload["clean_gate"]["spearman_to_reference"],
                "clean_loss": payload["clean_loss"],
                "fgsm_loss": payload["fgsm_loss"],
                "pgd_loss": payload["pgd_loss"],
            }
        )
    df = pd.DataFrame(rows)
    _write_plot_data(df, repo_root, "fig01_clean_alignment.csv")
    return df


def load_ratio_sensitivity_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    summary_path = report_dir / "partial_attack_ratio_sweep_multiseed" / "ratio_sweep_summary_stats.csv"
    summary_df = pd.read_csv(summary_path)
    attackers = {
        "FGSM": "partial_fgsm_minus_partial_clean__",
        "PGD": "partial_pgd_minus_partial_clean__",
    }
    metrics = [
        "annualized_excess_return_with_cost",
        "max_drawdown_with_cost",
        "rank_ic_mean",
        "information_ratio_with_cost",
    ]
    rows: list[dict[str, float | int | str]] = []
    for attacker, prefix in attackers.items():
        for metric in metrics:
            metric_key = f"{prefix}{metric}"
            subset = summary_df[summary_df["metric"] == metric_key].sort_values("ratio_pct")
            for _, row in subset.iterrows():
                rows.append(
                    {
                        "ratio_pct": int(row["ratio_pct"]),
                        "ratio_label": row["ratio_label"],
                        "attacker": attacker,
                        "metric": metric,
                        "delta_mean": float(row["mean"]),
                        "delta_std": float(row["std"]),
                        "degradation_mean": abs(float(row["mean"])),
                        "degradation_std": float(row["std"]),
                    }
                )
    df = pd.DataFrame(rows)
    _write_plot_data(df, repo_root, "fig06_ratio_sensitivity.csv")
    return df


def load_sample_shift_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    sample_path = report_dir / "lstm_whitebox_attack_expanded_v6" / "sample_metrics.csv"
    sample_df = pd.read_csv(sample_path)
    rows = []
    for attacker, column in [
        ("FGSM", "fgsm_abs_shift"),
        ("PGD", "pgd_abs_shift"),
    ]:
        for value in sample_df[column].tolist():
            rows.append({"attacker": attacker, "abs_pred_shift": float(value)})
    df = pd.DataFrame(rows)
    _write_plot_data(df, repo_root, "fig02_sample_shift.csv")
    return df


def load_cumulative_return_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    daily_path = report_dir / "partial_attack_backtest_multiseed_ratio5_union" / "seed_0" / "daily_comparison.csv"
    daily_df = pd.read_csv(daily_path)
    _write_plot_data(daily_df, repo_root, "fig04_cumulative_return_ratio5_seed0.csv")
    return daily_df


def load_multiseed_stability_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    summary_path = report_dir / "partial_attack_backtest_multiseed_ratio5_union" / "multiseed_summary_stats.csv"
    summary_df = pd.read_csv(summary_path)
    attackers = {
        "FGSM": "partial_fgsm_minus_partial_clean__",
        "PGD": "partial_pgd_minus_partial_clean__",
    }
    metrics = [
        "annualized_excess_return_with_cost",
        "max_drawdown_with_cost",
        "rank_ic_mean",
        "information_ratio_with_cost",
    ]
    rows: list[dict[str, float | str]] = []
    for attacker, prefix in attackers.items():
        for metric in metrics:
            metric_key = f"{prefix}{metric}"
            subset = summary_df[summary_df["metric"] == metric_key]
            if subset.empty:
                continue
            row = subset.iloc[0]
            rows.append(
                {
                    "attacker": attacker,
                    "metric": metric,
                    "delta_mean": float(row["mean"]),
                    "delta_std": float(row["std"]),
                    "degradation_mean": abs(float(row["mean"])),
                    "degradation_std": float(row["std"]),
                }
            )
    df = pd.DataFrame(rows)
    _write_plot_data(df, repo_root, "fig05_multiseed_ratio5.csv")
    return df


def _comparison_to_attacker(comparison: str) -> str:
    if "fgsm" in comparison.lower():
        return "FGSM"
    if "pgd" in comparison.lower():
        return "PGD"
    return comparison


def load_significance_summary_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    target_dir = report_dir / "lstm_single_model_evidence"
    rows: list[dict[str, object]] = []

    daily_path = target_dir / "significance_daily_metrics.csv"
    if daily_path.exists():
        daily_df = pd.read_csv(daily_path)
        for _, row in daily_df.iterrows():
            rows.append(
                {
                    "source": "daily",
                    "comparison": row["comparison"],
                    "attacker": _comparison_to_attacker(str(row["comparison"])),
                    "metric": row["metric"],
                    "delta_mean": float(row["delta_mean"]),
                    "p_value": float(row["p_value"]) if "p_value" in row else None,
                }
            )

    bootstrap_path = target_dir / "significance_block_bootstrap.json"
    if bootstrap_path.exists():
        payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        for comparison, metric_map in payload.items():
            for metric, summary in metric_map.items():
                rows.append(
                    {
                        "source": "bootstrap",
                        "comparison": comparison,
                        "attacker": _comparison_to_attacker(comparison),
                        "metric": metric,
                        "delta_mean": float(summary["bootstrap_mean"]),
                        "ci95_lower": float(summary["ci95_lower"]),
                        "ci95_upper": float(summary["ci95_upper"]),
                    }
                )

    df = pd.DataFrame(rows)
    _write_plot_data(df, repo_root, "fig07_significance_summary.csv")
    return df


def load_ranking_mechanism_data(report_dir: Path) -> pd.DataFrame:
    report_dir = Path(report_dir)
    repo_root = _repo_root_from_report_dir(report_dir)
    target_dir = report_dir / "lstm_single_model_evidence"
    rows: list[dict[str, object]] = []

    overlap_path = target_dir / "ranking_overlap_daily.csv"
    if overlap_path.exists():
        overlap_df = pd.read_csv(overlap_path)
        for _, row in overlap_df.iterrows():
            rows.append(
                {
                    "metric": "topk_overlap",
                    "value": float(row["topk_overlap"]),
                    "comparison": row["comparison"],
                    "attacker": _comparison_to_attacker(str(row["comparison"])),
                    "seed": int(row["seed"]),
                }
            )

    correlation_path = target_dir / "ranking_correlation_daily.csv"
    if correlation_path.exists():
        correlation_df = pd.read_csv(correlation_path)
        for _, row in correlation_df.iterrows():
            rows.append(
                {
                    "metric": "spearman",
                    "value": float(row["spearman"]),
                    "comparison": row["comparison"],
                    "attacker": _comparison_to_attacker(str(row["comparison"])),
                    "seed": int(row["seed"]),
                }
            )

    shift_path = target_dir / "rank_shift_summary.csv"
    if shift_path.exists():
        shift_df = pd.read_csv(shift_path)
        for _, row in shift_df.iterrows():
            rows.append(
                {
                    "metric": "rank_shift_abs_mean",
                    "value": float(row["rank_shift_abs_mean"]),
                    "comparison": row["comparison"],
                    "attacker": _comparison_to_attacker(str(row["comparison"])),
                    "seed": int(row["seed"]),
                }
            )

    df = pd.DataFrame(rows)
    _write_plot_data(df, repo_root, "fig08_ranking_mechanism.csv")
    return df
