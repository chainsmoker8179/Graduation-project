from __future__ import annotations

import copy
import math
import random
from typing import Iterable

import pandas as pd

from scripts.export_lstm_attack_assets import select_matched_rows_by_keys


def _coerce_multi_index(index_or_frame: pd.MultiIndex | pd.DataFrame | pd.Series) -> pd.MultiIndex:
    if isinstance(index_or_frame, pd.MultiIndex):
        return index_or_frame
    return index_or_frame.index


def build_daily_attack_mask(
    index_or_frame: pd.MultiIndex | pd.DataFrame | pd.Series,
    *,
    ratio: float,
    seed: int,
) -> pd.Series:
    index = _coerce_multi_index(index_or_frame)
    if index.nlevels != 2:
        raise ValueError("expected MultiIndex(['datetime', 'instrument'])")
    rng = random.Random(seed)
    mask = pd.Series(False, index=index)
    datetimes = index.get_level_values("datetime")
    for dt in pd.Index(datetimes.unique()):
        day_index = index[datetimes == dt]
        num_selected = max(1, math.ceil(len(day_index) * ratio))
        picked = rng.sample(list(day_index), k=num_selected)
        mask.loc[picked] = True
    return mask


def merge_partial_scores(
    *,
    reference_scores: pd.DataFrame,
    replacement_scores: pd.DataFrame,
    attackable_keys: Iterable[tuple[str, str]] | pd.MultiIndex,
) -> pd.DataFrame:
    merged = reference_scores.copy()
    attackable_index = pd.MultiIndex.from_tuples(list(attackable_keys), names=reference_scores.index.names)
    merged.loc[attackable_index, :] = replacement_scores.loc[attackable_index, :]
    return merged


def build_score_tables(
    *,
    reference_scores: pd.DataFrame,
    clean_scores: pd.DataFrame,
    fgsm_scores: pd.DataFrame,
    pgd_scores: pd.DataFrame,
    attackable_keys: Iterable[tuple[str, str]] | pd.MultiIndex,
) -> dict[str, pd.DataFrame]:
    return {
        "reference_clean": reference_scores.copy(),
        "partial_clean": merge_partial_scores(
            reference_scores=reference_scores,
            replacement_scores=clean_scores,
            attackable_keys=attackable_keys,
        ),
        "partial_fgsm": merge_partial_scores(
            reference_scores=reference_scores,
            replacement_scores=fgsm_scores,
            attackable_keys=attackable_keys,
        ),
        "partial_pgd": merge_partial_scores(
            reference_scores=reference_scores,
            replacement_scores=pgd_scores,
            attackable_keys=attackable_keys,
        ),
    }


def select_attack_subset(
    sample_asset: dict[str, object],
    *,
    requested_keys: list[tuple[str, str]] | pd.MultiIndex,
) -> dict[str, object]:
    return select_matched_rows_by_keys(sample_asset, requested_keys)


def build_partial_score_tables(
    *,
    reference_scores: pd.DataFrame,
    sample_asset: dict[str, object],
    attack_mask: pd.Series,
    attack_fn,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    selected_keys = [tuple(key) for key, is_selected in attack_mask.items() if bool(is_selected)]
    selected_subset = select_attack_subset(sample_asset, requested_keys=selected_keys)
    attack_result = attack_fn(selected_subset)

    clean_scores = attack_result["clean_scores"]
    fgsm_scores = attack_result["fgsm_scores"]
    pgd_scores = attack_result["pgd_scores"]
    attackable_keys = clean_scores.index
    score_tables = build_score_tables(
        reference_scores=reference_scores,
        clean_scores=clean_scores,
        fgsm_scores=fgsm_scores,
        pgd_scores=pgd_scores,
        attackable_keys=attackable_keys,
    )
    summary = {
        "selected_count": len(selected_keys),
        "selected_ratio": float(attack_mask.mean()),
        "selected_available_count": len(selected_subset["keys"]),
        "selected_missing_count": len(selected_keys) - len(selected_subset["keys"]),
        "attackable_count": len(attackable_keys),
        "attackable_ratio": float(len(attackable_keys) / len(reference_scores)) if len(reference_scores) > 0 else 0.0,
        "failed_keys": attack_result.get("failed_keys", []),
        "failure_reasons": attack_result.get("failure_reasons", {}),
    }
    return score_tables, summary


def run_score_backtests(
    *,
    score_tables: dict[str, pd.DataFrame],
    backtest_config: dict[str, object],
    backtest_fn=None,
) -> dict[str, dict[str, object]]:
    if backtest_fn is None:
        from qlib.backtest import backtest as backtest_fn

    results: dict[str, dict[str, object]] = {}
    for group_name, scores in score_tables.items():
        group_config = copy.deepcopy(backtest_config)
        strategy_config = group_config["strategy"]
        strategy_kwargs = dict(strategy_config.get("kwargs", {}))
        strategy_kwargs["signal"] = scores
        strategy_config["kwargs"] = strategy_kwargs

        portfolio_metric_dict, indicator_dict = backtest_fn(
            executor=group_config["executor"],
            strategy=strategy_config,
            **group_config["backtest"],
        )
        results[group_name] = {
            "scores": scores,
            "portfolio_metric_dict": portfolio_metric_dict,
            "indicator_dict": indicator_dict,
        }
    return results


def build_comparison_table(summary: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for group_name, group_summary in summary.items():
        rows[group_name] = dict(group_summary["metrics"])

    def _metric_delta(left: str, right: str) -> dict[str, float]:
        left_metrics = summary[left]["metrics"]
        right_metrics = summary[right]["metrics"]
        shared_keys = left_metrics.keys() & right_metrics.keys()
        return {key: float(left_metrics[key] - right_metrics[key]) for key in shared_keys}

    rows["partial_clean_minus_reference_clean"] = _metric_delta("partial_clean", "reference_clean")
    rows["partial_fgsm_minus_partial_clean"] = _metric_delta("partial_fgsm", "partial_clean")
    rows["partial_pgd_minus_partial_clean"] = _metric_delta("partial_pgd", "partial_clean")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)


def _safe_mean_std_ratio(series: pd.Series) -> float | None:
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return None
    return float(series.mean() / std)


def summarize_backtest_outputs(
    *,
    results: dict[str, dict[str, object]],
    label_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    from qlib.contrib.evaluate import indicator_analysis, risk_analysis
    from qlib.contrib.eva.alpha import calc_ic

    summary: dict[str, dict[str, object]] = {}
    label_col = label_df.columns[0]
    for group_name, group_result in results.items():
        freq = next(iter(group_result["portfolio_metric_dict"]))
        report_normal, positions_normal = group_result["portfolio_metric_dict"][freq]
        indicators_normal = group_result["indicator_dict"][freq][0]
        indicator_df = indicators_normal if "value" in getattr(indicators_normal, "columns", []) else indicator_analysis(indicators_normal)

        score_df = group_result["scores"]
        score_col = score_df.columns[0]
        joined = score_df[[score_col]].join(label_df[[label_col]], how="inner")
        ic, ric = calc_ic(joined[score_col], joined[label_col])

        daily_excess_without_cost = report_normal["return"] - report_normal["bench"]
        daily_excess_with_cost = report_normal["return"] - report_normal["bench"] - report_normal["cost"]
        excess_without_cost = risk_analysis(daily_excess_without_cost, freq=freq)
        excess_with_cost = risk_analysis(daily_excess_with_cost, freq=freq)

        indicator_metrics = indicator_df["value"].to_dict() if "value" in indicator_df.columns else {}
        metrics = {
            "annualized_excess_return_without_cost": float(excess_without_cost.loc["annualized_return", "risk"]),
            "annualized_excess_return_with_cost": float(excess_with_cost.loc["annualized_return", "risk"]),
            "information_ratio_without_cost": float(excess_without_cost.loc["information_ratio", "risk"]),
            "information_ratio_with_cost": float(excess_with_cost.loc["information_ratio", "risk"]),
            "max_drawdown_without_cost": float(excess_without_cost.loc["max_drawdown", "risk"]),
            "max_drawdown_with_cost": float(excess_with_cost.loc["max_drawdown", "risk"]),
            "ic_mean": float(ic.mean()),
            "icir": _safe_mean_std_ratio(ic),
            "rank_ic_mean": float(ric.mean()),
            "rank_icir": _safe_mean_std_ratio(ric),
        }
        if "pa" in indicator_metrics:
            metrics["pa"] = float(indicator_metrics["pa"])
        if "ffr" in indicator_metrics:
            metrics["ffr"] = float(indicator_metrics["ffr"])
        if "pos" in indicator_metrics:
            metrics["pos"] = float(indicator_metrics["pos"])

        summary[group_name] = {
            "metrics": metrics,
            "report_normal": report_normal,
            "positions_normal": positions_normal,
            "indicator_analysis": indicator_df,
            "daily_excess_return_without_cost": daily_excess_without_cost,
            "daily_excess_return_with_cost": daily_excess_with_cost,
            "daily_ic": ic,
            "daily_rank_ic": ric,
        }
    return summary


def build_daily_comparison_table(summary: dict[str, dict[str, object]]) -> pd.DataFrame:
    daily = pd.DataFrame()
    for group_name, group_summary in summary.items():
        daily[f"{group_name}_excess_return_with_cost"] = group_summary["daily_excess_return_with_cost"]
        daily[f"{group_name}_rank_ic"] = group_summary["daily_rank_ic"]

    daily["partial_clean_minus_reference_clean_excess_return_with_cost"] = (
        daily["partial_clean_excess_return_with_cost"] - daily["reference_clean_excess_return_with_cost"]
    )
    daily["fgsm_minus_partial_clean_excess_return_with_cost"] = (
        daily["partial_fgsm_excess_return_with_cost"] - daily["partial_clean_excess_return_with_cost"]
    )
    daily["pgd_minus_partial_clean_excess_return_with_cost"] = (
        daily["partial_pgd_excess_return_with_cost"] - daily["partial_clean_excess_return_with_cost"]
    )
    daily["partial_clean_minus_reference_clean_rank_ic"] = daily["partial_clean_rank_ic"] - daily["reference_clean_rank_ic"]
    daily["fgsm_minus_partial_clean_rank_ic"] = daily["partial_fgsm_rank_ic"] - daily["partial_clean_rank_ic"]
    daily["pgd_minus_partial_clean_rank_ic"] = daily["partial_pgd_rank_ic"] - daily["partial_clean_rank_ic"]
    return daily.sort_index()


__all__ = [
    "build_comparison_table",
    "build_daily_comparison_table",
    "build_daily_attack_mask",
    "build_partial_score_tables",
    "build_score_tables",
    "merge_partial_scores",
    "run_score_backtests",
    "select_attack_subset",
    "summarize_backtest_outputs",
]
