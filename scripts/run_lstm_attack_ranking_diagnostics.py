from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.lstm_attack_ranking_diagnostics import (
    compute_daily_score_spearman,
    compute_daily_topk_overlap,
    summarize_rank_shift,
)
from scripts.analysis.report_artifact_locator import resolve_report_artifact


def _load_score_table(repo_root: Path, report_subdir: str, seed: int, score_name: str) -> pd.DataFrame:
    filename = f"{score_name}_scores.pkl" if not score_name.endswith("_scores") else f"{score_name}.pkl"
    relative_path = Path("reports") / report_subdir / f"seed_{seed}" / filename
    path = resolve_report_artifact(repo_root, relative_path)
    return pd.read_pickle(path).rename(columns={"score": score_name})


def _build_score_frame(repo_root: Path, report_subdir: str, seed: int, attacked_name: str) -> pd.DataFrame:
    baseline = _load_score_table(repo_root, report_subdir, seed, "partial_clean").rename(columns={"partial_clean": "baseline_score"})
    attacked = _load_score_table(repo_root, report_subdir, seed, attacked_name).rename(columns={attacked_name: "attacked_score"})
    merged = baseline.join(attacked, how="inner")
    merged = merged.reset_index()
    merged["seed"] = seed
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ranking diagnostics for LSTM white-box attack outputs.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report-subdir", default="partial_attack_backtest_multiseed_ratio5_union")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/lstm_single_model_evidence"))
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_rows: list[pd.DataFrame] = []
    correlation_rows: list[pd.DataFrame] = []
    shift_rows: list[pd.DataFrame] = []
    for seed in range(5):
        for attacked_name in ["partial_fgsm", "partial_pgd"]:
            frame = _build_score_frame(repo_root, args.report_subdir, seed, attacked_name)
            comparison = f"{attacked_name}_vs_partial_clean"

            overlap = compute_daily_topk_overlap(frame, topk=args.topk)
            overlap["seed"] = seed
            overlap["comparison"] = comparison
            overlap_rows.append(overlap)

            correlation = compute_daily_score_spearman(frame)
            correlation["seed"] = seed
            correlation["comparison"] = comparison
            correlation_rows.append(correlation)

            shift = summarize_rank_shift(frame)
            shift["seed"] = seed
            shift["comparison"] = comparison
            shift_rows.append(shift)

    overlap_df = pd.concat(overlap_rows, ignore_index=True)
    correlation_df = pd.concat(correlation_rows, ignore_index=True)
    shift_df = pd.concat(shift_rows, ignore_index=True)

    overlap_path = out_dir / "ranking_overlap_daily.csv"
    correlation_path = out_dir / "ranking_correlation_daily.csv"
    shift_path = out_dir / "rank_shift_summary.csv"
    overlap_df.to_csv(overlap_path, index=False)
    correlation_df.to_csv(correlation_path, index=False)
    shift_df.to_csv(shift_path, index=False)

    print(f"overlap_csv={overlap_path}")
    print(f"correlation_csv={correlation_path}")
    print(f"rank_shift_csv={shift_path}")


if __name__ == "__main__":
    main()
