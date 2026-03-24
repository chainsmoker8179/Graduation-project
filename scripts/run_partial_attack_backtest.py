#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_attack_core import (
    CleanGateThresholds,
    LegacyRawLSTMPipeline,
    fgsm_maximize_mse,
    pgd_maximize_mse,
    run_clean_gate,
    validate_clean_gate,
)
from partial_attack_backtest import (
    build_comparison_table,
    build_daily_attack_mask,
    build_daily_comparison_table,
    build_partial_score_tables,
    run_score_backtests,
    summarize_backtest_outputs,
)
from scripts.export_lstm_attack_assets import select_matched_rows_by_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run partial-stock FGSM/PGD backtests on the legacy raw-OHLCV LSTM.")
    parser.add_argument("--pred-pkl", type=Path, default=Path("prediction/pred.pkl"))
    parser.add_argument("--label-pkl", type=Path, default=Path("prediction/label.pkl"))
    parser.add_argument("--asset-dir", type=Path, default=Path("artifacts/lstm_attack_expanded_v6"))
    parser.add_argument("--state-dict-path", type=Path, default=Path("model/lstm_state_dict.pt"))
    parser.add_argument("--config-path", type=Path, default=Path("model/lstm_state_dict_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/partial_attack_backtest"))
    parser.add_argument("--provider-uri", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack-ratio", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--benchmark", type=str, default="SH000300")
    parser.add_argument("--account", type=float, default=100000000)
    parser.add_argument("--start-time", type=str, default=None)
    parser.add_argument("--end-time", type=str, default=None)
    parser.add_argument("--price-epsilon", type=float, default=0.01)
    parser.add_argument("--volume-epsilon", type=float, default=0.02)
    parser.add_argument("--price-floor", type=float, default=1e-6)
    parser.add_argument("--volume-floor", type=float, default=1.0)
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--pgd-step-size", type=float, default=0.25)
    parser.add_argument("--attack-batch-size", type=int, default=256)
    parser.add_argument("--constraint-mode", type=str, default="physical_stat", choices=["none", "physical", "physical_stat"])
    parser.add_argument("--tau-ret", type=float, default=0.005)
    parser.add_argument("--tau-body", type=float, default=0.005)
    parser.add_argument("--tau-range", type=float, default=0.01)
    parser.add_argument("--tau-vol", type=float, default=0.05)
    parser.add_argument("--lambda-ret", type=float, default=0.8)
    parser.add_argument("--lambda-candle", type=float, default=0.4)
    parser.add_argument("--lambda-vol", type=float, default=0.3)
    parser.add_argument("--min-clean-grad-mean-abs", type=float, default=1e-6)
    parser.add_argument("--min-spearman-to-reference", type=float, default=0.09)
    parser.add_argument("--max-feature-mae-to-reference", type=float, default=0.05)
    parser.add_argument("--max-feature-rmse-to-reference", type=float, default=0.12)
    parser.add_argument("--max-feature-max-abs-to-reference", type=float, default=0.7)
    parser.add_argument("--deal-price", type=str, default="close")
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0015)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("expected MultiIndex(['datetime', 'instrument'])")
    df = df.copy()
    if list(df.index.names) != ["datetime", "instrument"]:
        df.index = df.index.set_names(["datetime", "instrument"])
    frame = df.index.to_frame(index=False)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str)
    df.index = pd.MultiIndex.from_frame(frame)
    return df.sort_index()


def _load_attack_assets(asset_dir: Path) -> dict[str, Any]:
    sample_asset = torch.load(asset_dir / "matched_ohlcv_windows.pt", map_location="cpu")
    feature_asset_path = asset_dir / "matched_feature_windows.pt"
    feature_asset = torch.load(feature_asset_path, map_location="cpu") if feature_asset_path.exists() else None
    normalization_stats = json.loads((asset_dir / "normalization_stats.json").read_text(encoding="utf-8"))
    matched_reference_path = asset_dir / "matched_reference.csv"
    matched_reference = pd.read_csv(matched_reference_path) if matched_reference_path.exists() else None
    return {
        "sample_asset": sample_asset,
        "feature_asset": feature_asset,
        "normalization_stats": normalization_stats,
        "matched_reference": matched_reference,
    }


def _subset_feature_asset(feature_asset: dict[str, Any] | None, requested_keys: list[tuple[str, str]]) -> torch.Tensor | None:
    if feature_asset is None:
        return None
    key_lookup = {(str(pd.Timestamp(key[0])), str(key[1])): idx for idx, key in enumerate(feature_asset["keys"])}
    feature_indices = [key_lookup[(str(pd.Timestamp(key[0])), str(key[1]))] for key in requested_keys]
    return feature_asset["features"][feature_indices]


def _score_frame(keys: list[tuple[str, str]], values: torch.Tensor) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(key[0]), str(key[1])) for key in keys],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({"score": values.detach().cpu().numpy()}, index=index).sort_index()


def _build_backtest_config(args: argparse.Namespace, reference_scores: pd.DataFrame) -> dict[str, Any]:
    from qlib.utils import get_date_by_shift

    dt_index = reference_scores.index.get_level_values("datetime")
    start_time = args.start_time or str(dt_index.min().date())
    end_time = args.end_time or str(get_date_by_shift(dt_index.max(), 1).date())
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "topk": args.topk,
                "n_drop": args.n_drop,
            },
        },
        "backtest": {
            "start_time": start_time,
            "end_time": end_time,
            "account": args.account,
            "benchmark": args.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": args.limit_threshold,
                "deal_price": args.deal_price,
                "open_cost": args.open_cost,
                "close_cost": args.close_cost,
                "min_cost": args.min_cost,
            },
        },
    }


def _init_qlib(provider_uri: str) -> None:
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=os.path.expanduser(provider_uri), region=REG_CN)


def _filter_frame_by_date(df: pd.DataFrame, start_time: str | None, end_time: str | None) -> pd.DataFrame:
    if start_time is None and end_time is None:
        return df
    dt_index = df.index.get_level_values("datetime")
    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        mask &= dt_index >= pd.Timestamp(start_time)
    if end_time is not None:
        mask &= dt_index <= pd.Timestamp(end_time)
    return df.loc[mask.to_numpy()]


def _build_attack_fn(
    *,
    model: LegacyRawLSTMPipeline,
    feature_asset: dict[str, Any] | None,
    device: torch.device,
    args: argparse.Namespace,
):
    thresholds = CleanGateThresholds(
        min_clean_grad_mean_abs=args.min_clean_grad_mean_abs,
        min_spearman_to_reference=args.min_spearman_to_reference,
        max_feature_mae_to_reference=args.max_feature_mae_to_reference,
        max_feature_rmse_to_reference=args.max_feature_rmse_to_reference,
        max_feature_max_abs_to_reference=args.max_feature_max_abs_to_reference,
    )

    def _attack(selected_subset: dict[str, Any]) -> dict[str, Any]:
        if len(selected_subset["keys"]) == 0:
            empty_index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
            empty_scores = pd.DataFrame({"score": []}, index=empty_index)
            return {
                "clean_scores": empty_scores,
                "fgsm_scores": empty_scores.copy(),
                "pgd_scores": empty_scores.copy(),
                "failed_keys": [],
                "failure_reasons": {},
                "clean_gate_metrics": None,
            }

        keys = list(selected_subset["keys"])
        x_all = selected_subset["ohlcv"]
        y_all = selected_subset["label"]
        reference_scores_all = selected_subset["score"]
        reference_features_all = _subset_feature_asset(feature_asset, keys)

        clean_frames: list[pd.DataFrame] = []
        fgsm_frames: list[pd.DataFrame] = []
        pgd_frames: list[pd.DataFrame] = []
        failed_keys: list[tuple[str, str]] = []
        failure_reasons: dict[str, int] = {}
        clean_gate_metrics: list[dict[str, Any]] = []

        for start in range(0, len(keys), args.attack_batch_size):
            end = min(start + args.attack_batch_size, len(keys))
            batch_keys = keys[start:end]
            x = x_all[start:end].to(device)
            y = y_all[start:end].to(device)
            reference_scores = reference_scores_all[start:end].to(device)
            reference_features = None
            if reference_features_all is not None:
                reference_features = reference_features_all[start:end].to(device)

            clean_gate = run_clean_gate(
                model=model,
                x=x,
                y=y,
                reference_scores=reference_scores,
                reference_features=reference_features,
            )
            clean_gate_metrics.append(asdict(clean_gate))
            try:
                validate_clean_gate(clean_gate, thresholds)
            except ValueError as exc:
                failed_keys.extend(batch_keys)
                failure_reasons[str(exc)] = failure_reasons.get(str(exc), 0) + len(batch_keys)
                continue

            fgsm_x = fgsm_maximize_mse(
                model=model,
                x=x,
                y=y,
                price_epsilon=args.price_epsilon,
                volume_epsilon=args.volume_epsilon,
                price_floor=args.price_floor,
                volume_floor=args.volume_floor,
                constraint_mode=args.constraint_mode,
                tau_ret=args.tau_ret,
                tau_body=args.tau_body,
                tau_range=args.tau_range,
                tau_vol=args.tau_vol,
                lambda_ret=args.lambda_ret,
                lambda_candle=args.lambda_candle,
                lambda_vol=args.lambda_vol,
            )
            pgd_x = pgd_maximize_mse(
                model=model,
                x=x,
                y=y,
                price_epsilon=args.price_epsilon,
                volume_epsilon=args.volume_epsilon,
                num_steps=args.pgd_steps,
                step_size=args.pgd_step_size,
                price_floor=args.price_floor,
                volume_floor=args.volume_floor,
                constraint_mode=args.constraint_mode,
                tau_ret=args.tau_ret,
                tau_body=args.tau_body,
                tau_range=args.tau_range,
                tau_vol=args.tau_vol,
                lambda_ret=args.lambda_ret,
                lambda_candle=args.lambda_candle,
                lambda_vol=args.lambda_vol,
            )
            with torch.no_grad():
                clean_pred = model(x)
                fgsm_pred = model(fgsm_x)
                pgd_pred = model(pgd_x)

            clean_frames.append(_score_frame(batch_keys, clean_pred))
            fgsm_frames.append(_score_frame(batch_keys, fgsm_pred))
            pgd_frames.append(_score_frame(batch_keys, pgd_pred))

        empty_index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        empty_scores = pd.DataFrame({"score": []}, index=empty_index)
        return {
            "clean_scores": pd.concat(clean_frames).sort_index() if clean_frames else empty_scores,
            "fgsm_scores": pd.concat(fgsm_frames).sort_index() if fgsm_frames else empty_scores.copy(),
            "pgd_scores": pd.concat(pgd_frames).sort_index() if pgd_frames else empty_scores.copy(),
            "failed_keys": failed_keys,
            "failure_reasons": failure_reasons,
            "clean_gate_metrics": clean_gate_metrics,
        }

    return _attack


def _write_report(
    out_dir: Path,
    generation_summary: dict[str, Any],
    group_metrics: dict[str, dict[str, Any]],
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# 部分股票白盒攻击回测实验报告",
        "",
        "## 攻击设置",
        "",
        f"- constraint_mode: {generation_summary['constraint_mode']}",
        f"- tau_ret: {generation_summary['tau_ret']}",
        f"- tau_body: {generation_summary['tau_body']}",
        f"- tau_range: {generation_summary['tau_range']}",
        f"- tau_vol: {generation_summary['tau_vol']}",
        f"- lambda_ret: {generation_summary['lambda_ret']}",
        f"- lambda_candle: {generation_summary['lambda_candle']}",
        f"- lambda_vol: {generation_summary['lambda_vol']}",
        f"- selected_count: {generation_summary['selected_count']}",
        f"- selected_available_count: {generation_summary['selected_available_count']}",
        f"- selected_missing_count: {generation_summary['selected_missing_count']}",
        f"- attackable_count: {generation_summary['attackable_count']}",
        f"- selected_ratio: {generation_summary['selected_ratio']:.6f}",
        f"- attackable_ratio: {generation_summary['attackable_ratio']:.6f}",
        "",
        "## 四组主指标",
        "",
    ]
    for group_name, metrics in group_metrics.items():
        lines.extend(
            [
                f"### {group_name}",
                "",
                f"- annualized_excess_return_with_cost: {metrics.get('annualized_excess_return_with_cost')}",
                f"- max_drawdown_with_cost: {metrics.get('max_drawdown_with_cost')}",
                f"- rank_ic_mean: {metrics.get('rank_ic_mean')}",
                "",
            ]
        )
    lines.extend(
        [
            "## 主比较差值",
            "",
            comparison.to_markdown(),
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    reference_scores = _normalize_frame(pd.read_pickle(args.pred_pkl))
    label_df = _normalize_frame(pd.read_pickle(args.label_pkl))

    assets = _load_attack_assets(args.asset_dir)
    sample_asset = assets["sample_asset"]
    feature_asset = assets["feature_asset"]
    normalization_stats = assets["normalization_stats"]
    matched_reference = assets["matched_reference"]

    if matched_reference is not None and not matched_reference.empty:
        asset_start_time = str(pd.to_datetime(matched_reference["datetime"]).min().date())
        asset_end_time = str(pd.to_datetime(matched_reference["datetime"]).max().date())
        reference_scores = _filter_frame_by_date(reference_scores, asset_start_time, asset_end_time)
        label_df = _filter_frame_by_date(label_df, asset_start_time, asset_end_time)

    model = LegacyRawLSTMPipeline(
        normalization_stats=normalization_stats,
        state_dict_path=args.state_dict_path,
        config_path=args.config_path,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    attack_mask = build_daily_attack_mask(reference_scores, ratio=args.attack_ratio, seed=args.seed)
    attack_fn = _build_attack_fn(model=model, feature_asset=feature_asset, device=device, args=args)
    score_tables, generation_summary = build_partial_score_tables(
        reference_scores=reference_scores,
        sample_asset=sample_asset,
        attack_mask=attack_mask,
        attack_fn=attack_fn,
    )
    generation_summary.update(
        {
            "constraint_mode": args.constraint_mode,
            "tau_ret": args.tau_ret,
            "tau_body": args.tau_body,
            "tau_range": args.tau_range,
            "tau_vol": args.tau_vol,
            "lambda_ret": args.lambda_ret,
            "lambda_candle": args.lambda_candle,
            "lambda_vol": args.lambda_vol,
        }
    )

    _init_qlib(args.provider_uri)
    backtest_config = _build_backtest_config(args, reference_scores)
    backtest_results = run_score_backtests(score_tables=score_tables, backtest_config=backtest_config)
    summary = summarize_backtest_outputs(results=backtest_results, label_df=label_df)
    comparison = build_comparison_table(summary)
    daily_comparison = build_daily_comparison_table(summary)

    attack_mask.to_frame("selected").to_csv(args.out_dir / "attack_mask.csv")
    for name, scores in score_tables.items():
        scores.to_pickle(args.out_dir / f"{name}_scores.pkl")
    comparison.to_csv(args.out_dir / "comparison.csv")
    daily_comparison.to_csv(args.out_dir / "daily_comparison.csv")

    json_summary = {
        "generation": generation_summary,
        "groups": {group_name: group_summary["metrics"] for group_name, group_summary in summary.items()},
        "comparison": comparison.to_dict(orient="index"),
        "backtest_config": backtest_config,
    }
    (args.out_dir / "backtest_summary.json").write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_report(
        args.out_dir,
        generation_summary=generation_summary,
        group_metrics={group_name: group_summary["metrics"] for group_name, group_summary in summary.items()},
        comparison=comparison,
    )

    print(f"attackable_count={generation_summary['attackable_count']}")
    print(f"summary_json={args.out_dir / 'backtest_summary.json'}")
    print(f"comparison_csv={args.out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
