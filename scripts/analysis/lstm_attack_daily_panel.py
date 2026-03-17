from __future__ import annotations

from pathlib import Path

import pandas as pd


_DELTA_SPECS = (
    (
        "fgsm_minus_partial_clean_excess_return_with_cost",
        "partial_fgsm_excess_return_with_cost",
        "partial_clean_excess_return_with_cost",
    ),
    (
        "pgd_minus_partial_clean_excess_return_with_cost",
        "partial_pgd_excess_return_with_cost",
        "partial_clean_excess_return_with_cost",
    ),
    (
        "fgsm_minus_partial_clean_rank_ic",
        "partial_fgsm_rank_ic",
        "partial_clean_rank_ic",
    ),
    (
        "pgd_minus_partial_clean_rank_ic",
        "partial_pgd_rank_ic",
        "partial_clean_rank_ic",
    ),
)


def _with_required_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for output_col, attacked_col, baseline_col in _DELTA_SPECS:
        if output_col not in df.columns and attacked_col in df.columns and baseline_col in df.columns:
            df[output_col] = df[attacked_col] - df[baseline_col]
    return df


def load_multiseed_daily_panel(report_root: str | Path) -> pd.DataFrame:
    report_root = Path(report_root)
    seed_root = report_root / "partial_attack_backtest_multiseed_ratio5_union"
    frames: list[pd.DataFrame] = []
    for seed_dir in sorted(seed_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        df = pd.read_csv(seed_dir / "daily_comparison.csv")
        df["seed"] = seed
        frames.append(_with_required_delta_columns(df))
    if not frames:
        raise FileNotFoundError(f"no daily comparison files found under {seed_root}")
    return pd.concat(frames, ignore_index=True)
