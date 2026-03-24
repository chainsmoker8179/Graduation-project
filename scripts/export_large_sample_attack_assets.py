#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_feature_bridge import LegacyLSTMFeatureBridge
from legacy_lstm_preprocess import FillnaLayer, RobustZScoreNormLayer
from scripts.export_lstm_attack_assets import (
    DEFAULT_LABEL_EXPR,
    _build_raw_test_split,
    build_matched_reference,
    export_matched_raw_windows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast export large-sample attack assets via raw-window reconstruction and torch feature rebuild."
    )
    parser.add_argument("--pred-pkl", type=Path, required=True)
    parser.add_argument("--label-pkl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--normalization-stats", type=Path, required=True)
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
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_normalization_stats(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_feature_windows_from_raw(
    sample_asset: dict[str, Any],
    *,
    normalization_stats: dict[str, Any],
) -> dict[str, Any]:
    center = torch.tensor(normalization_stats["center"], dtype=torch.float32)
    scale = torch.tensor(normalization_stats["scale"], dtype=torch.float32)
    scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)

    bridge = LegacyLSTMFeatureBridge(input_window_len=int(sample_asset["ohlcv"].shape[1]))
    norm = RobustZScoreNormLayer(
        center=center,
        scale=scale,
        clip_outlier=bool(normalization_stats.get("clip_outlier", False)),
    )
    fillna = FillnaLayer()

    with torch.no_grad():
        features = fillna(norm(bridge(sample_asset["ohlcv"].float()))).cpu()

    return {
        "keys": list(sample_asset["keys"]),
        "features": features,
        "missing_keys": list(sample_asset.get("missing_keys", [])),
        "feature_source": "torch_bridge_from_raw",
    }


def build_export_summary(
    *,
    matched_reference_rows: int,
    sample_asset: dict[str, Any],
    feature_asset: dict[str, Any],
    normalization_stats_source: str | Path,
) -> dict[str, Any]:
    return {
        "matched_reference_rows": int(matched_reference_rows),
        "exported_sample_rows": int(sample_asset["ohlcv"].shape[0]),
        "raw_window_len": int(sample_asset["ohlcv"].shape[1]),
        "raw_feature_dim": int(sample_asset["ohlcv"].shape[2]),
        "missing_raw_keys": int(len(sample_asset.get("missing_keys", []))),
        "exported_feature_rows": int(feature_asset["features"].shape[0]),
        "feature_window_len": int(feature_asset["features"].shape[1]),
        "feature_dim": int(feature_asset["features"].shape[2]),
        "feature_source": str(feature_asset["feature_source"]),
        "normalization_stats_source": str(normalization_stats_source),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_pickle(args.pred_pkl)
    label_df = pd.read_pickle(args.label_pkl)
    matched_reference = build_matched_reference(
        pred_df=pred_df,
        label_df=label_df,
        date_from=args.test_start_time,
        date_to=args.test_end_time,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if matched_reference.empty:
        raise ValueError("matched reference is empty")
    matched_reference.reset_index().to_csv(args.out_dir / "matched_reference.csv", index=False)

    sample_asset = export_matched_raw_windows(matched_reference, _build_raw_test_split(args))
    torch.save(sample_asset, args.out_dir / "matched_ohlcv_windows.pt")

    normalization_stats = load_normalization_stats(args.normalization_stats)
    (args.out_dir / "normalization_stats.json").write_text(
        json.dumps(normalization_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    feature_asset = build_feature_windows_from_raw(sample_asset, normalization_stats=normalization_stats)
    torch.save(feature_asset, args.out_dir / "matched_feature_windows.pt")

    summary = build_export_summary(
        matched_reference_rows=len(matched_reference),
        sample_asset=sample_asset,
        feature_asset=feature_asset,
        normalization_stats_source=args.normalization_stats,
    )
    (args.out_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
