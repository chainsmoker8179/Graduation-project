import torch
import pandas as pd
from types import SimpleNamespace

from partial_attack_backtest import (
    build_comparison_table,
    build_daily_comparison_table,
    build_daily_attack_mask,
    build_partial_score_tables,
    build_score_tables,
    run_score_backtests,
    select_attack_subset,
    summarize_backtest_outputs,
)
from legacy_lstm_attack_core import CleanGateMetrics
from scripts.run_partial_attack_backtest import _build_attack_fn


def _make_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-02", "BBB"),
            ("2025-01-02", "CCC"),
            ("2025-01-03", "AAA"),
            ("2025-01-03", "BBB"),
        ],
        names=["datetime", "instrument"],
    )


def _make_score_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"score": values}, index=_make_index())


def test_build_daily_attack_mask_is_reproducible() -> None:
    index = _make_index()

    mask_a = build_daily_attack_mask(index, ratio=0.05, seed=0)
    mask_b = build_daily_attack_mask(index, ratio=0.05, seed=0)

    assert mask_a.equals(mask_b)
    assert int(mask_a.loc[pd.IndexSlice["2025-01-02", :]].sum()) == 1
    assert int(mask_a.loc[pd.IndexSlice["2025-01-03", :]].sum()) == 1


def test_build_score_tables_only_replaces_attackable_keys() -> None:
    reference_scores = _make_score_frame([0.1, 0.2, 0.3, 0.4, 0.5])
    clean_scores = pd.DataFrame({"score": [1.1, 1.5]}, index=reference_scores.index[[1, 4]])
    fgsm_scores = pd.DataFrame({"score": [2.1, 2.5]}, index=reference_scores.index[[1, 4]])
    pgd_scores = pd.DataFrame({"score": [3.1, 3.5]}, index=reference_scores.index[[1, 4]])

    tables = build_score_tables(
        reference_scores=reference_scores,
        clean_scores=clean_scores,
        fgsm_scores=fgsm_scores,
        pgd_scores=pgd_scores,
        attackable_keys=clean_scores.index,
    )

    unchanged_index = reference_scores.index.difference(clean_scores.index)
    assert tables["reference_clean"].equals(reference_scores)
    assert tables["partial_clean"].loc[clean_scores.index, "score"].tolist() == [1.1, 1.5]
    assert tables["partial_fgsm"].loc[fgsm_scores.index, "score"].tolist() == [2.1, 2.5]
    assert tables["partial_pgd"].loc[pgd_scores.index, "score"].tolist() == [3.1, 3.5]
    assert tables["partial_clean"].loc[unchanged_index].equals(reference_scores.loc[unchanged_index])
    assert tables["partial_fgsm"].loc[unchanged_index].equals(reference_scores.loc[unchanged_index])
    assert tables["partial_pgd"].loc[unchanged_index].equals(reference_scores.loc[unchanged_index])


def test_select_attack_subset_returns_requested_keys_only() -> None:
    sample_asset = {
        "keys": [
            ("2025-01-02 00:00:00", "AAA"),
            ("2025-01-02 00:00:00", "BBB"),
            ("2025-01-03 00:00:00", "CCC"),
        ],
        "ohlcv": torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5),
        "label": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        "score": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }

    subset = select_attack_subset(
        sample_asset,
        requested_keys=[
            ("2025-01-03 00:00:00", "CCC"),
            ("2025-01-02 00:00:00", "AAA"),
        ],
    )

    assert subset["keys"] == [
        ("2025-01-03 00:00:00", "CCC"),
        ("2025-01-02 00:00:00", "AAA"),
    ]
    assert torch.allclose(subset["label"], torch.tensor([0.3, 0.1], dtype=torch.float32))
    assert torch.allclose(subset["score"], torch.tensor([3.0, 1.0], dtype=torch.float32))


def test_run_score_backtests_contains_all_score_groups() -> None:
    score_tables = {
        "reference_clean": _make_score_frame([0.1, 0.2, 0.3, 0.4, 0.5]),
        "partial_clean": _make_score_frame([0.2, 0.3, 0.4, 0.5, 0.6]),
        "partial_fgsm": _make_score_frame([0.3, 0.4, 0.5, 0.6, 0.7]),
        "partial_pgd": _make_score_frame([0.4, 0.5, 0.6, 0.7, 0.8]),
    }

    def fake_backtest_fn(*, executor, strategy, start_time, end_time, benchmark, account, exchange_kwargs):
        signal = strategy["kwargs"]["signal"]
        report_normal = pd.DataFrame(
            {
                "return": [float(signal.iloc[:, 0].mean())],
                "bench": [0.01],
                "cost": [0.001],
            },
            index=pd.Index([pd.Timestamp("2025-01-02")], name="datetime"),
        )
        indicator_df = pd.DataFrame({"value": {"ffr": 1.0, "pa": 0.0, "pos": 0.0}})
        return {"1day": (report_normal, pd.DataFrame())}, {"1day": (indicator_df, {"source": "fake"})}

    results = run_score_backtests(
        score_tables=score_tables,
        backtest_config={
            "executor": {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor", "kwargs": {}},
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"topk": 50, "n_drop": 5},
            },
            "backtest": {
                "start_time": "2025-01-01",
                "end_time": "2025-01-31",
                "benchmark": "SH000300",
                "account": 100000000,
                "exchange_kwargs": {"deal_price": "close"},
            },
        },
        backtest_fn=fake_backtest_fn,
    )

    assert sorted(results.keys()) == [
        "partial_clean",
        "partial_fgsm",
        "partial_pgd",
        "reference_clean",
    ]
    assert set(results["reference_clean"].keys()) == {"indicator_dict", "portfolio_metric_dict", "scores"}


def test_comparison_table_contains_primary_deltas() -> None:
    summary = {
        "reference_clean": {
            "metrics": {
                "annualized_excess_return_with_cost": 0.12,
                "max_drawdown_with_cost": -0.05,
                "rank_ic_mean": 0.08,
            }
        },
        "partial_clean": {
            "metrics": {
                "annualized_excess_return_with_cost": 0.11,
                "max_drawdown_with_cost": -0.06,
                "rank_ic_mean": 0.07,
            }
        },
        "partial_fgsm": {
            "metrics": {
                "annualized_excess_return_with_cost": 0.09,
                "max_drawdown_with_cost": -0.08,
                "rank_ic_mean": 0.05,
            }
        },
        "partial_pgd": {
            "metrics": {
                "annualized_excess_return_with_cost": 0.07,
                "max_drawdown_with_cost": -0.10,
                "rank_ic_mean": 0.03,
            }
        },
    }

    comparison = build_comparison_table(summary)

    assert "partial_clean_minus_reference_clean" in comparison.index
    assert "partial_fgsm_minus_partial_clean" in comparison.index
    assert "partial_pgd_minus_partial_clean" in comparison.index
    assert abs(comparison.loc["partial_fgsm_minus_partial_clean", "annualized_excess_return_with_cost"] + 0.02) < 1e-12
    assert abs(comparison.loc["partial_pgd_minus_partial_clean", "rank_ic_mean"] + 0.04) < 1e-12


def test_summarize_backtest_outputs_produces_metrics_for_each_group() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("2025-01-02", "AAA"),
            ("2025-01-02", "BBB"),
            ("2025-01-03", "AAA"),
            ("2025-01-03", "BBB"),
        ],
        names=["datetime", "instrument"],
    )
    label_df = pd.DataFrame({"label": [0.05, 0.01, 0.03, 0.02]}, index=index)
    score_tables = {
        "reference_clean": pd.DataFrame({"score": [0.10, 0.20, 0.30, 0.40]}, index=index),
        "partial_clean": pd.DataFrame({"score": [0.11, 0.21, 0.31, 0.41]}, index=index),
        "partial_fgsm": pd.DataFrame({"score": [0.09, 0.19, 0.29, 0.39]}, index=index),
        "partial_pgd": pd.DataFrame({"score": [0.08, 0.18, 0.28, 0.38]}, index=index),
    }

    def _report(value: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "return": [value, value / 2],
                "bench": [0.01, 0.01],
                "cost": [0.001, 0.001],
            },
            index=pd.Index([pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")], name="datetime"),
        )

    indicator_df = pd.DataFrame({"value": {"ffr": 1.0, "pa": 0.0, "pos": 0.0}})
    results = {
        name: {
            "scores": scores,
            "portfolio_metric_dict": {"1day": (_report(0.03 + i * 0.01), pd.DataFrame())},
            "indicator_dict": {"1day": (indicator_df, {"source": "fake"})},
        }
        for i, (name, scores) in enumerate(score_tables.items())
    }

    summary = summarize_backtest_outputs(results=results, label_df=label_df)

    assert sorted(summary.keys()) == [
        "partial_clean",
        "partial_fgsm",
        "partial_pgd",
        "reference_clean",
    ]
    assert "annualized_excess_return_with_cost" in summary["partial_clean"]["metrics"]
    assert "rank_ic_mean" in summary["partial_clean"]["metrics"]
    assert "daily_excess_return_with_cost" in summary["partial_clean"]
    assert "daily_rank_ic" in summary["partial_clean"]


def test_build_daily_comparison_table_contains_primary_delta_columns() -> None:
    daily_index = pd.Index([pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")], name="datetime")
    summary = {
        "reference_clean": {
            "daily_excess_return_with_cost": pd.Series([0.01, 0.02], index=daily_index),
            "daily_rank_ic": pd.Series([0.10, 0.20], index=daily_index),
        },
        "partial_clean": {
            "daily_excess_return_with_cost": pd.Series([0.009, 0.018], index=daily_index),
            "daily_rank_ic": pd.Series([0.09, 0.19], index=daily_index),
        },
        "partial_fgsm": {
            "daily_excess_return_with_cost": pd.Series([0.004, 0.010], index=daily_index),
            "daily_rank_ic": pd.Series([0.05, 0.10], index=daily_index),
        },
        "partial_pgd": {
            "daily_excess_return_with_cost": pd.Series([0.002, 0.006], index=daily_index),
            "daily_rank_ic": pd.Series([0.03, 0.08], index=daily_index),
        },
    }

    daily_comparison = build_daily_comparison_table(summary)

    assert "partial_clean_excess_return_with_cost" in daily_comparison.columns
    assert "fgsm_minus_partial_clean_excess_return_with_cost" in daily_comparison.columns
    assert "pgd_minus_partial_clean_rank_ic" in daily_comparison.columns
    assert abs(daily_comparison.loc[pd.Timestamp("2025-01-02"), "fgsm_minus_partial_clean_excess_return_with_cost"] + 0.005) < 1e-12


def test_build_partial_score_tables_reports_selected_and_attackable_counts() -> None:
    reference_scores = _make_score_frame([0.1, 0.2, 0.3, 0.4, 0.5])
    attack_mask = pd.Series(False, index=reference_scores.index)
    attack_mask.iloc[[1, 4]] = True
    sample_asset = {
        "keys": [
            ("2025-01-02", "BBB"),
            ("2025-01-03", "BBB"),
        ],
        "ohlcv": torch.ones(2, 2, 5),
        "label": torch.tensor([0.2, 0.5]),
        "score": torch.tensor([0.2, 0.5]),
    }

    def fake_attack_fn(selected_subset):
        clean_index = pd.MultiIndex.from_tuples(selected_subset["keys"], names=["datetime", "instrument"])
        return {
            "clean_scores": pd.DataFrame({"score": [1.2, 1.5]}, index=clean_index),
            "fgsm_scores": pd.DataFrame({"score": [2.2, 2.5]}, index=clean_index),
            "pgd_scores": pd.DataFrame({"score": [3.2, 3.5]}, index=clean_index),
            "failed_keys": [],
        }

    score_tables, summary = build_partial_score_tables(
        reference_scores=reference_scores,
        sample_asset=sample_asset,
        attack_mask=attack_mask,
        attack_fn=fake_attack_fn,
    )

    assert summary["selected_count"] == 2
    assert summary["attackable_count"] == 2
    assert summary["selected_ratio"] > 0
    assert score_tables["partial_clean"].loc[("2025-01-02", "BBB"), "score"] == 1.2
    assert score_tables["partial_fgsm"].loc[("2025-01-03", "BBB"), "score"] == 2.5


def test_attack_fn_passes_constraint_arguments(monkeypatch) -> None:
    calls = []

    class FakeModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :, 0].mean(dim=1)

    def fake_run_clean_gate(**kwargs):
        return CleanGateMetrics(
            clean_loss=0.1,
            clean_grad_mean_abs=1e-5,
            clean_grad_max_abs=1e-3,
            clean_grad_finite_rate=1.0,
            feature_finite_rate=1.0,
            clean_pred_mean=0.0,
            clean_pred_std=1.0,
            reference_score_mean=0.0,
            reference_score_std=1.0,
            spearman_to_reference=0.9,
            feature_mae_to_reference=None,
            feature_rmse_to_reference=None,
            feature_max_abs_to_reference=None,
        )

    def fake_validate_clean_gate(metrics, thresholds) -> None:
        return None

    def fake_fgsm_maximize_mse(**kwargs):
        calls.append(
            (
                "fgsm",
                kwargs["constraint_mode"],
                kwargs["tau_ret"],
                kwargs["lambda_ret"],
                kwargs["lambda_candle"],
                kwargs["lambda_vol"],
            )
        )
        return kwargs["x"]

    def fake_pgd_maximize_mse(**kwargs):
        calls.append(
            (
                "pgd",
                kwargs["constraint_mode"],
                kwargs["tau_ret"],
                kwargs["lambda_ret"],
                kwargs["lambda_candle"],
                kwargs["lambda_vol"],
            )
        )
        return kwargs["x"]

    monkeypatch.setattr("scripts.run_partial_attack_backtest.run_clean_gate", fake_run_clean_gate)
    monkeypatch.setattr("scripts.run_partial_attack_backtest.validate_clean_gate", fake_validate_clean_gate)
    monkeypatch.setattr("scripts.run_partial_attack_backtest.fgsm_maximize_mse", fake_fgsm_maximize_mse)
    monkeypatch.setattr("scripts.run_partial_attack_backtest.pgd_maximize_mse", fake_pgd_maximize_mse)

    args = SimpleNamespace(
        attack_batch_size=16,
        min_clean_grad_mean_abs=1e-6,
        min_spearman_to_reference=0.09,
        max_feature_mae_to_reference=0.05,
        max_feature_rmse_to_reference=0.12,
        max_feature_max_abs_to_reference=0.7,
        price_epsilon=0.01,
        volume_epsilon=0.02,
        price_floor=1e-6,
        volume_floor=1.0,
        pgd_steps=5,
        pgd_step_size=0.25,
        constraint_mode="physical_stat",
        tau_ret=0.005,
        tau_body=0.005,
        tau_range=0.01,
        tau_vol=0.05,
        lambda_ret=0.8,
        lambda_candle=0.4,
        lambda_vol=0.3,
    )
    attack_fn = _build_attack_fn(model=FakeModel(), feature_asset=None, device=torch.device("cpu"), args=args)

    selected_subset = {
        "keys": [("2025-01-02 00:00:00", "AAA")],
        "ohlcv": torch.ones(1, 2, 5),
        "label": torch.tensor([0.1], dtype=torch.float32),
        "score": torch.tensor([0.2], dtype=torch.float32),
    }

    result = attack_fn(selected_subset)

    assert list(result["clean_scores"].index) == [(pd.Timestamp("2025-01-02 00:00:00"), "AAA")]
    assert calls == [
        ("fgsm", "physical_stat", 0.005, 0.8, 0.4, 0.3),
        ("pgd", "physical_stat", 0.005, 0.8, 0.4, 0.3),
    ]
