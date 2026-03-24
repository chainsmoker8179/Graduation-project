#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_feature_bridge import LEGACY_LSTM_FEATURES


DEFAULT_LABEL_EXPR = "Ref($close, -2) / Ref($close, -1) - 1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export assets for raw OHLCV LSTM white-box attacks.")
    parser.add_argument("--pred-pkl", type=Path, default=Path("prediction/pred.pkl"))
    parser.add_argument("--label-pkl", type=Path, default=Path("prediction/label.pkl"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/lstm_attack"))
    parser.add_argument("--provider-uri", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--market", type=str, default="all")
    parser.add_argument("--start-time", type=str, default="2019-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--fit-start-time", type=str, default="2019-01-01")
    parser.add_argument("--fit-end-time", type=str, default="2023-12-31")
    parser.add_argument("--test-start-time", type=str, default="2025-01-01")
    parser.add_argument("--test-end-time", type=str, default="2025-12-31")
    parser.add_argument("--label-expr", type=str, default=DEFAULT_LABEL_EXPR)
    parser.add_argument("--raw-window-len", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--requested-keys-csv", type=Path, default=None)
    return parser.parse_args()


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("expected MultiIndex(['datetime', 'instrument'])")
    names = list(df.index.names)
    df = df.copy()
    if names != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])
    index_frame = df.index.to_frame(index=False)
    index_frame["datetime"] = pd.to_datetime(index_frame["datetime"])
    df.index = pd.MultiIndex.from_frame(index_frame)
    return df.sort_index()


def _canonicalize_key(key: tuple[Any, Any]) -> tuple[str, str]:
    return str(pd.Timestamp(key[0])), str(key[1])


def select_matched_rows_by_keys(
    sample_asset: dict[str, Any],
    requested_keys: list[tuple[str, str]] | pd.MultiIndex,
) -> dict[str, Any]:
    key_lookup = {_canonicalize_key(tuple(key)): idx for idx, key in enumerate(sample_asset["keys"])}
    selected_indices = [
        key_lookup[_canonicalize_key(tuple(key))] for key in requested_keys if _canonicalize_key(tuple(key)) in key_lookup
    ]
    selected_keys = [tuple(sample_asset["keys"][idx]) for idx in selected_indices]

    subset = {
        "keys": selected_keys,
        "ohlcv": sample_asset["ohlcv"][selected_indices],
        "label": sample_asset["label"][selected_indices],
        "score": sample_asset["score"][selected_indices],
    }
    if "missing_keys" in sample_asset:
        subset["missing_keys"] = sample_asset["missing_keys"]
    return subset


def filter_matched_reference_by_keys(
    matched_reference: pd.DataFrame,
    requested_keys: list[tuple[str, str]] | pd.MultiIndex,
) -> pd.DataFrame:
    matched_reference = _normalize_datetime_index(matched_reference)
    requested_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp(key[0]), str(key[1]))
            for key in requested_keys
        ],
        names=["datetime", "instrument"],
    )
    requested_frame = pd.DataFrame(index=requested_index)
    filtered = requested_frame.join(matched_reference, how="left").dropna(how="any")
    return filtered


def _load_requested_keys_csv(path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    required_columns = {"datetime", "instrument"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"requested keys csv must contain columns {sorted(required_columns)}")
    return [(str(pd.Timestamp(row["datetime"])), str(row["instrument"])) for _, row in df.iterrows()]


def build_matched_reference(
    pred_df: pd.DataFrame,
    label_df: pd.DataFrame,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    pred_df = _normalize_datetime_index(pred_df)
    label_df = _normalize_datetime_index(label_df)

    pred_col = pred_df.columns[0]
    label_col = label_df.columns[0]

    merged = pred_df[[pred_col]].rename(columns={pred_col: "score"}).join(
        label_df[[label_col]].rename(columns={label_col: "label"}),
        how="inner",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
    if date_from is not None or date_to is not None:
        dt_index = merged.index.get_level_values("datetime")
        mask = pd.Series(True, index=merged.index)
        if date_from is not None:
            mask &= dt_index >= pd.Timestamp(date_from)
        if date_to is not None:
            mask &= dt_index <= pd.Timestamp(date_to)
        merged = merged.loc[mask.to_numpy()]
    if max_samples is not None and len(merged) > max_samples:
        rng = random.Random(seed)
        selected = sorted(rng.sample(list(merged.index), k=max_samples))
        merged = merged.loc[selected]
    return merged


def _flatten_feature_name(col: Any) -> str:
    if isinstance(col, tuple):
        return str(col[-1])
    return str(col)


def extract_normalization_stats(processor: Any, expected_features: list[str]) -> dict[str, Any]:
    feature_to_idx = {_flatten_feature_name(col): idx for idx, col in enumerate(processor.cols)}
    missing = [name for name in expected_features if name not in feature_to_idx]
    if missing:
        raise ValueError(f"missing normalization stats for features: {missing}")

    order = [feature_to_idx[name] for name in expected_features]
    center = np.asarray(processor.mean_train, dtype=np.float64)[order].tolist()
    scale = np.asarray(processor.std_train, dtype=np.float64)[order].tolist()
    return {
        "feature_names": list(expected_features),
        "center": center,
        "scale": scale,
        "clip_outlier": bool(processor.clip_outlier),
    }


def _load_feature_expressions(csv_path: Path, feature_names: list[str]) -> list[str]:
    expr_by_name: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            expr_by_name[row["name"]] = row["expression"]
    missing = [name for name in feature_names if name not in expr_by_name]
    if missing:
        raise ValueError(f"missing feature expressions for: {missing}")
    return [expr_by_name[name] for name in feature_names]


def build_alpha158_handler_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "start_time": args.start_time,
        "end_time": args.end_time,
        "fit_start_time": args.fit_start_time,
        "fit_end_time": args.fit_end_time,
        "instruments": args.market,
        "infer_processors": [
            {
                "class": "FilterCol",
                "kwargs": {
                    "fields_group": "feature",
                    "col_list": list(LEGACY_LSTM_FEATURES),
                },
            },
            {
                "class": "RobustZScoreNorm",
                "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": args.fit_start_time,
                    "fit_end_time": args.fit_end_time,
                },
            },
            {
                "class": "Fillna",
                "kwargs": {
                    "fields_group": "feature",
                },
            },
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": [args.label_expr],
    }


def _build_legacy_feature_handler(args: argparse.Namespace) -> Any:
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158

    provider_uri = os.path.expanduser(args.provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    return Alpha158(**build_alpha158_handler_kwargs(args))


def _find_robust_zscore_processor(handler: Any) -> Any:
    for processor in handler.infer_processors:
        if type(processor).__name__ == "RobustZScoreNorm":
            return processor
    raise ValueError("RobustZScoreNorm processor not found in handler.infer_processors")


def _build_raw_test_split(args: argparse.Namespace) -> Any:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data.dataset import TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    provider_uri = os.path.expanduser(args.provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    raw_start_time = max(
        pd.Timestamp(args.start_time),
        pd.Timestamp(args.test_start_time) - pd.Timedelta(days=max(args.raw_window_len * 3, 365)),
    ).strftime("%Y-%m-%d")

    loader = QlibDataLoader(
        config={
            "feature": ["$open", "$high", "$low", "$close", "$volume"],
            "label": [args.label_expr],
        }
    )
    handler = DataHandlerLP(
        instruments=args.market,
        start_time=raw_start_time,
        end_time=args.end_time,
        data_loader=loader,
    )
    dataset = TSDatasetH(
        handler=handler,
        segments={"test": (args.test_start_time, args.test_end_time)},
        step_len=args.raw_window_len,
    )
    return dataset.prepare("test")


def _build_feature_test_split(args: argparse.Namespace) -> Any:
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import TSDatasetH

    provider_uri = os.path.expanduser(args.provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    handler = Alpha158(**build_alpha158_handler_kwargs(args))
    dataset = TSDatasetH(
        handler=handler,
        segments={"test": (args.test_start_time, args.test_end_time)},
        step_len=20,
    )
    return prepare_feature_test_split(dataset)


def prepare_feature_test_split(dataset: Any) -> Any:
    from qlib.data.dataset.handler import DataHandlerLP

    split = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    split.config(fillna_type="ffill+bfill")
    return split


def export_matched_raw_windows(matched_reference: pd.DataFrame, raw_test_split: Any) -> dict[str, Any]:
    matched_reference = _normalize_datetime_index(matched_reference)
    index_lookup = {
        (pd.Timestamp(idx[0]), idx[1]): row_idx for row_idx, idx in enumerate(raw_test_split.get_index())
    }

    keys: list[tuple[str, str]] = []
    ohlcv_rows: list[torch.Tensor] = []
    labels: list[float] = []
    scores: list[float] = []
    missing_keys: list[tuple[str, str]] = []

    for idx, row in matched_reference.iterrows():
        lookup_key = (pd.Timestamp(idx[0]), idx[1])
        if lookup_key not in index_lookup:
            missing_keys.append((str(idx[0]), str(idx[1])))
            continue
        sample = raw_test_split[index_lookup[lookup_key]]
        ohlcv = torch.tensor(sample[:, :5], dtype=torch.float32)
        if not torch.isfinite(ohlcv).all():
            missing_keys.append((str(idx[0]), str(idx[1])))
            continue
        keys.append((str(idx[0]), str(idx[1])))
        ohlcv_rows.append(ohlcv)
        labels.append(float(row["label"]))
        scores.append(float(row["score"]))

    if not ohlcv_rows:
        raise ValueError("no matched raw OHLCV windows could be reconstructed from the test split")

    ohlcv = torch.stack(ohlcv_rows, dim=0)

    return {
        "keys": keys,
        "ohlcv": ohlcv,
        "label": torch.tensor(labels, dtype=torch.float32),
        "score": torch.tensor(scores, dtype=torch.float32),
        "missing_keys": missing_keys,
    }


def export_matched_feature_windows(
    matched_reference: pd.DataFrame,
    feature_test_split: Any,
    *,
    feature_dim: int,
) -> dict[str, Any]:
    matched_reference = _normalize_datetime_index(matched_reference)
    index_lookup = {
        (pd.Timestamp(idx[0]), idx[1]): row_idx for row_idx, idx in enumerate(feature_test_split.get_index())
    }

    keys: list[tuple[str, str]] = []
    feature_rows: list[torch.Tensor] = []
    missing_keys: list[tuple[str, str]] = []

    for idx in matched_reference.index:
        lookup_key = (pd.Timestamp(idx[0]), idx[1])
        if lookup_key not in index_lookup:
            missing_keys.append((str(idx[0]), str(idx[1])))
            continue
        sample = feature_test_split[index_lookup[lookup_key]]
        features = torch.tensor(sample[:, :feature_dim], dtype=torch.float32)
        if not torch.isfinite(features).all():
            missing_keys.append((str(idx[0]), str(idx[1])))
            continue
        feature_rows.append(features)
        keys.append((str(idx[0]), str(idx[1])))

    if not feature_rows:
        raise ValueError("no matched feature windows could be reconstructed from the test split")

    features = torch.stack(feature_rows, dim=0)
    return {
        "keys": keys,
        "features": features,
        "missing_keys": missing_keys,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading pred/label pickles", flush=True)
    pred_df = pd.read_pickle(args.pred_pkl)
    label_df = pd.read_pickle(args.label_pkl)
    matched_reference = build_matched_reference(
        pred_df=pred_df,
        label_df=label_df,
        date_from=args.test_start_time,
        date_to=args.test_end_time,
        max_samples=None if args.requested_keys_csv is not None else args.max_samples,
        seed=args.seed,
    )
    if args.requested_keys_csv is not None:
        requested_keys = _load_requested_keys_csv(args.requested_keys_csv)
        matched_reference = filter_matched_reference_by_keys(matched_reference, requested_keys)
    if matched_reference.empty:
        raise ValueError("matched reference is empty after intersecting pred.pkl and label.pkl")

    matched_path = args.out_dir / "matched_reference.csv"
    matched_reference.reset_index().to_csv(matched_path, index=False)
    print(f"[2/4] matched reference rows: {len(matched_reference)}", flush=True)

    print("[3/4] fitting legacy normalization statistics in Qlib", flush=True)
    handler = _build_legacy_feature_handler(args)
    processor = _find_robust_zscore_processor(handler)
    normalization_stats = extract_normalization_stats(processor, LEGACY_LSTM_FEATURES)
    stats_path = args.out_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(normalization_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] exporting matched raw OHLCV windows", flush=True)
    raw_test_split = _build_raw_test_split(args)
    sample_asset = export_matched_raw_windows(matched_reference, raw_test_split)
    sample_path = args.out_dir / "matched_ohlcv_windows.pt"
    torch.save(sample_asset, sample_path)

    print("[4.5/4] exporting matched reference feature windows", flush=True)
    feature_test_split = _build_feature_test_split(args)
    feature_asset = export_matched_feature_windows(
        matched_reference,
        feature_test_split,
        feature_dim=len(LEGACY_LSTM_FEATURES),
    )
    feature_path = args.out_dir / "matched_feature_windows.pt"
    torch.save(feature_asset, feature_path)

    summary = {
        "matched_reference_rows": int(len(matched_reference)),
        "exported_sample_rows": int(sample_asset["ohlcv"].shape[0]),
        "raw_window_len": int(sample_asset["ohlcv"].shape[1]),
        "raw_feature_dim": int(sample_asset["ohlcv"].shape[2]),
        "missing_raw_keys": int(len(sample_asset["missing_keys"])),
        "exported_feature_rows": int(feature_asset["features"].shape[0]),
        "feature_window_len": int(feature_asset["features"].shape[1]),
        "feature_dim": int(feature_asset["features"].shape[2]),
        "missing_feature_keys": int(len(feature_asset["missing_keys"])),
    }
    summary_path = args.out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"matched_reference_rows={summary['matched_reference_rows']}")
    print(f"exported_sample_rows={summary['exported_sample_rows']}")
    print(f"raw_window_shape={tuple(sample_asset['ohlcv'].shape)}")
    print(f"matched_reference_csv={matched_path}")
    print(f"normalization_stats_json={stats_path}")
    print(f"matched_ohlcv_windows_pt={sample_path}")
    print(f"matched_feature_windows_pt={feature_path}")


if __name__ == "__main__":
    main()
