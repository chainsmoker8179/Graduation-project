from __future__ import annotations

import pandas as pd


def compute_daily_topk_overlap(scores: pd.DataFrame, topk: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dt, part in scores.groupby("datetime", sort=True):
        effective_topk = min(topk, len(part))
        if effective_topk == 0:
            continue
        baseline_top = set(part.nlargest(effective_topk, "baseline_score")["instrument"])
        attacked_top = set(part.nlargest(effective_topk, "attacked_score")["instrument"])
        rows.append(
            {
                "datetime": dt,
                "topk": effective_topk,
                "topk_overlap": len(baseline_top & attacked_top) / float(effective_topk),
            }
        )
    return pd.DataFrame(rows)


def compute_daily_score_spearman(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dt, part in scores.groupby("datetime", sort=True):
        rows.append(
            {
                "datetime": dt,
                "spearman": part["baseline_score"].corr(part["attacked_score"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def summarize_rank_shift(scores: pd.DataFrame) -> pd.DataFrame:
    ranked = scores.copy()
    ranked["baseline_rank"] = ranked.groupby("datetime")["baseline_score"].rank(method="average", ascending=False)
    ranked["attacked_rank"] = ranked.groupby("datetime")["attacked_score"].rank(method="average", ascending=False)
    ranked["rank_shift"] = ranked["attacked_rank"] - ranked["baseline_rank"]
    return (
        ranked.groupby("datetime", sort=True)["rank_shift"]
        .agg(rank_shift_mean="mean", rank_shift_abs_mean=lambda s: s.abs().mean(), rank_shift_abs_max=lambda s: s.abs().max())
        .reset_index()
    )
