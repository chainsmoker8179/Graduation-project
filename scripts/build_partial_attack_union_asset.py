#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_feature_bridge import LEGACY_LSTM_FEATURES
from partial_attack_backtest import build_daily_attack_mask
from scripts.export_lstm_attack_assets import (
    DEFAULT_LABEL_EXPR,
    _build_legacy_feature_handler,
    _build_raw_test_split,
    _find_robust_zscore_processor,
    _normalize_datetime_index,
    build_matched_reference,
    export_matched_raw_windows,
    extract_normalization_stats,
    filter_matched_reference_by_keys,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a shared raw-only asset for multiseed partial-attack backtests.")
    parser.add_argument("--pred-pkl", type=Path, default=Path("origin_model_pred/LSTM/prediction/pred.pkl"))
    parser.add_argument("--label-pkl", type=Path, default=Path("origin_model_pred/LSTM/prediction/label.pkl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provider-uri", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--market", type=str, default="all")
    parser.add_argument("--start-time", type=str, default="2019-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--fit-start-time", type=str, default="2019-01-01")
    parser.add_argument("--fit-end-time", type=str, default="2023-12-31")
    parser.add_argument("--test-start-time", type=str, default="2025-01-01")
    parser.add_argument("--test-end-time", type=str, default="2025-10-31")
    parser.add_argument("--label-expr", type=str, default=DEFAULT_LABEL_EXPR)
    parser.add_argument("--raw-window-len", type=int, default=80)
    parser.add_argument("--attack-ratio", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-keys", type=int, default=None)
    return parser.parse_args(argv)


def build_union_requested_keys(
    reference_scores: pd.DataFrame,
    *,
    attack_ratio: float,
    seeds: list[int],
    start_time: str | None = None,
    end_time: str | None = None,
    max_keys: int | None = None,
) -> list[tuple[str, str]]:
    reference_scores = _normalize_datetime_index(reference_scores)
    if start_time is not None or end_time is not None:
        dt_index = reference_scores.index.get_level_values("datetime")
        mask = pd.Series(True, index=reference_scores.index)
        if start_time is not None:
            mask &= dt_index >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= dt_index <= pd.Timestamp(end_time)
        reference_scores = reference_scores.loc[mask.to_numpy()]

    key_set: set[tuple[str, str]] = set()
    for seed in seeds:
        attack_mask = build_daily_attack_mask(reference_scores, ratio=attack_ratio, seed=seed)
        for key, selected in attack_mask.items():
            if selected:
                key_set.add((str(pd.Timestamp(key[0])), str(key[1])))

    keys = sorted(key_set)
    if max_keys is not None:
        keys = keys[:max_keys]
    return keys


def write_requested_keys_csv(keys: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "instrument"])
        writer.writeheader()
        for dt, inst in keys:
            writer.writerow({"datetime": dt, "instrument": inst})


def build_rawonly_asset(args: argparse.Namespace, requested_keys: list[tuple[str, str]]) -> dict[str, Any]:
    pred_df = pd.read_pickle(args.pred_pkl)
    label_df = pd.read_pickle(args.label_pkl)
    matched_reference = build_matched_reference(
        pred_df=pred_df,
        label_df=label_df,
        date_from=args.test_start_time,
        date_to=args.test_end_time,
        max_samples=None,
        seed=0,
    )
    matched_reference = filter_matched_reference_by_keys(matched_reference, requested_keys)
    if matched_reference.empty:
        raise ValueError("matched reference is empty after filtering requested multiseed keys")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_requested_keys_csv(requested_keys, args.out_dir / "requested_keys.csv")
    matched_reference.reset_index().to_csv(args.out_dir / "matched_reference.csv", index=False)

    handler = _build_legacy_feature_handler(args)
    processor = _find_robust_zscore_processor(handler)
    normalization_stats = extract_normalization_stats(processor, LEGACY_LSTM_FEATURES)
    (args.out_dir / "normalization_stats.json").write_text(
        json.dumps(normalization_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_test_split = _build_raw_test_split(args)
    sample_asset = export_matched_raw_windows(matched_reference, raw_test_split)
    torch.save(sample_asset, args.out_dir / "matched_ohlcv_windows.pt")

    summary = {
        "requested_key_rows": len(requested_keys),
        "matched_reference_rows": int(len(matched_reference)),
        "exported_sample_rows": int(sample_asset["ohlcv"].shape[0]),
        "raw_window_len": int(sample_asset["ohlcv"].shape[1]),
        "raw_feature_dim": int(sample_asset["ohlcv"].shape[2]),
        "missing_raw_keys": int(len(sample_asset.get("missing_keys", []))),
        "seeds": list(args.seeds),
        "attack_ratio": float(args.attack_ratio),
    }
    (args.out_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    reference_scores = pd.read_pickle(args.pred_pkl)
    requested_keys = build_union_requested_keys(
        reference_scores,
        attack_ratio=args.attack_ratio,
        seeds=args.seeds,
        start_time=args.test_start_time,
        end_time=args.test_end_time,
        max_keys=args.max_keys,
    )
    summary = build_rawonly_asset(args, requested_keys)
    print(f"requested_key_rows={summary['requested_key_rows']}")
    print(f"matched_reference_rows={summary['matched_reference_rows']}")
    print(f"exported_sample_rows={summary['exported_sample_rows']}")
    print(f"export_summary_json={args.out_dir / 'export_summary.json'}")


if __name__ == "__main__":
    main()
