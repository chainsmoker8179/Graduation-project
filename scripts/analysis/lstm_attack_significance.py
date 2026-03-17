from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TRADING_DAYS_PER_YEAR = 252


def summarize_paired_significance(
    df: pd.DataFrame,
    baseline_col: str,
    attacked_col: str,
    metric_name: str,
) -> dict[str, Any]:
    deltas = (df[attacked_col] - df[baseline_col]).dropna()
    stat = wilcoxon(deltas)
    attacked_prefix = attacked_col.split("_", 2)[1]
    return {
        "metric": metric_name,
        "comparison": f"partial_{attacked_prefix}_vs_partial_clean",
        "delta_mean": float(deltas.mean()),
        "delta_median": float(deltas.median()),
        "p_value": float(stat.pvalue),
        "sample_size": int(deltas.shape[0]),
    }


def build_significance_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        summarize_paired_significance(
            panel,
            baseline_col="partial_clean_excess_return_with_cost",
            attacked_col="partial_fgsm_excess_return_with_cost",
            metric_name="excess_return_with_cost",
        ),
        summarize_paired_significance(
            panel,
            baseline_col="partial_clean_excess_return_with_cost",
            attacked_col="partial_pgd_excess_return_with_cost",
            metric_name="excess_return_with_cost",
        ),
        summarize_paired_significance(
            panel,
            baseline_col="partial_clean_rank_ic",
            attacked_col="partial_fgsm_rank_ic",
            metric_name="rank_ic",
        ),
        summarize_paired_significance(
            panel,
            baseline_col="partial_clean_rank_ic",
            attacked_col="partial_pgd_rank_ic",
            metric_name="rank_ic",
        ),
    ]
    return pd.DataFrame(rows)


def _annualized_return(series: pd.Series) -> float:
    return float(series.mean() * TRADING_DAYS_PER_YEAR)


def _information_ratio(series: pd.Series) -> float | None:
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return None
    return float((series.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(series: pd.Series) -> float:
    wealth = (1.0 + series).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def _sample_block_indices(length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    starts = rng.integers(0, length, size=int(np.ceil(length / block_size)) + 1)
    chunks = []
    total = 0
    for start in starts:
        stop = min(start + block_size, length)
        block = np.arange(start, stop)
        chunks.append(block)
        total += len(block)
        if total >= length:
            break
    return np.concatenate(chunks)[:length]


def bootstrap_path_metric_deltas(
    panel: pd.DataFrame,
    *,
    attacked_prefix: str,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    random_seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(random_seed)
    attacked_col = f"partial_{attacked_prefix}_excess_return_with_cost"
    baseline_col = "partial_clean_excess_return_with_cost"
    grouped = {
        int(seed): frame.sort_values("datetime").reset_index(drop=True)
        for seed, frame in panel.groupby("seed", sort=True)
    }

    boot_rows: list[dict[str, float]] = []
    for _ in range(n_bootstrap):
        metric_deltas = {
            "annualized_excess_return_with_cost": [],
            "information_ratio_with_cost": [],
            "max_drawdown_with_cost": [],
        }
        for frame in grouped.values():
            idx = _sample_block_indices(len(frame), block_size=block_size, rng=rng)
            baseline = frame.iloc[idx][baseline_col].reset_index(drop=True)
            attacked = frame.iloc[idx][attacked_col].reset_index(drop=True)

            metric_deltas["annualized_excess_return_with_cost"].append(_annualized_return(attacked) - _annualized_return(baseline))
            metric_deltas["max_drawdown_with_cost"].append(_max_drawdown(attacked) - _max_drawdown(baseline))

            baseline_ir = _information_ratio(baseline)
            attacked_ir = _information_ratio(attacked)
            metric_deltas["information_ratio_with_cost"].append(
                float(attacked_ir - baseline_ir) if attacked_ir is not None and baseline_ir is not None else np.nan
            )

        boot_rows.append({metric: float(np.nanmean(values)) for metric, values in metric_deltas.items()})

    boot_df = pd.DataFrame(boot_rows)
    output: dict[str, dict[str, float | int]] = {}
    for metric in boot_df.columns:
        series = boot_df[metric].dropna()
        output[metric] = {
            "bootstrap_mean": float(series.mean()),
            "ci95_lower": float(series.quantile(0.025)),
            "ci95_upper": float(series.quantile(0.975)),
            "n_bootstrap": int(series.shape[0]),
            "block_size": int(block_size),
        }
    return output
