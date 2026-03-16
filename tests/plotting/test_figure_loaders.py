from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_loaders_module():
    try:
        return importlib.import_module("scripts.plotting.loaders")
    except ModuleNotFoundError as exc:
        pytest.fail(f"scripts.plotting.loaders missing: {exc}")


def test_load_clean_alignment_data_extracts_v3_v6_metrics() -> None:
    loaders = _import_loaders_module()

    df = loaders.load_clean_alignment_data(
        report_dir=REPO_ROOT / "reports",
    )

    assert list(df["version"]) == ["expanded_v3", "expanded_v6"]
    assert set(df.columns) >= {
        "version",
        "spearman_to_reference",
        "clean_loss",
        "fgsm_loss",
        "pgd_loss",
    }
    assert df.loc[df["version"] == "expanded_v3", "spearman_to_reference"].item() == pytest.approx(0.10035, rel=1e-5)
    assert df.loc[df["version"] == "expanded_v6", "spearman_to_reference"].item() == pytest.approx(
        0.7513031653697968,
        rel=1e-9,
    )


def test_load_ratio_sensitivity_data_returns_three_ratios_and_two_attackers() -> None:
    loaders = _import_loaders_module()

    df = loaders.load_ratio_sensitivity_data(
        report_dir=REPO_ROOT / "reports",
    )

    assert sorted(df["ratio_pct"].unique().tolist()) == [1, 5, 10]
    assert sorted(df["attacker"].unique().tolist()) == ["FGSM", "PGD"]
    assert set(df["metric"].unique().tolist()) == {
        "annualized_excess_return_with_cost",
        "information_ratio_with_cost",
        "max_drawdown_with_cost",
        "rank_ic_mean",
    }

    target = df[
        (df["ratio_pct"] == 10)
        & (df["attacker"] == "PGD")
        & (df["metric"] == "annualized_excess_return_with_cost")
    ]
    assert len(target) == 1
    assert target["degradation_mean"].item() == pytest.approx(0.41284875170077784, rel=1e-9)
