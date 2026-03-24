#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_feature_bridge import LegacyLSTMFeatureBridge
from legacy_lstm_preprocess import FillnaLayer, RobustZScoreNormLayer
from whitebox_attack_core import (
    CleanGateThresholds,
    RawFeatureAttackPipeline,
    fgsm_maximize_mse,
    pgd_maximize_mse,
    run_clean_gate,
    usage_ratio,
    validate_clean_gate,
)
from whitebox_attack_models import load_model_adapter_from_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke white-box attack on Transformer/TCN/LSTM models.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--price-epsilon", type=float, default=0.01)
    parser.add_argument("--volume-epsilon", type=float, default=0.02)
    parser.add_argument("--price-floor", type=float, default=1e-6)
    parser.add_argument("--volume-floor", type=float, default=1.0)
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--pgd-step-size", type=float, default=0.25)
    parser.add_argument("--min-clean-grad-mean-abs", type=float, default=1e-6)
    parser.add_argument("--min-spearman-to-reference", type=float, default=0.09)
    parser.add_argument("--max-feature-mae-to-reference", type=float, default=0.05)
    parser.add_argument("--max-feature-rmse-to-reference", type=float, default=0.12)
    parser.add_argument("--max-feature-max-abs-to-reference", type=float, default=0.7)
    return parser.parse_args(argv)


def _choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _canonicalize_key(datetime_value: Any, instrument_value: Any) -> tuple[str, str]:
    return str(pd.Timestamp(datetime_value)), str(instrument_value)


def _to_tensor_1d(value: Any, indices: list[int] | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if indices is None:
        return tensor
    return tensor[indices]


def _tensor_row_is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def _load_attack_assets(asset_dir: Path, max_samples: int) -> dict[str, Any]:
    sample_asset = torch.load(asset_dir / "matched_ohlcv_windows.pt", map_location="cpu")
    feature_asset = torch.load(asset_dir / "matched_feature_windows.pt", map_location="cpu")
    normalization_stats = json.loads((asset_dir / "normalization_stats.json").read_text(encoding="utf-8"))

    feature_lookup = {
        _canonicalize_key(key[0], key[1]): idx
        for idx, key in enumerate(feature_asset["keys"])
    }
    reference_lookup: dict[tuple[str, str], Any] = {}
    reference_csv_path = asset_dir / "matched_reference.csv"
    if reference_csv_path.exists():
        reference_df = pd.read_csv(reference_csv_path)
        reference_lookup = {
            _canonicalize_key(row["datetime"], row["instrument"]): row
            for _, row in reference_df.iterrows()
        }

    sample_label = _to_tensor_1d(sample_asset.get("label")) if "label" in sample_asset else None
    sample_score = _to_tensor_1d(sample_asset.get("score")) if "score" in sample_asset else None

    keys: list[tuple[str, str]] = []
    sample_indices: list[int] = []
    feature_indices: list[int] = []
    labels: list[float] = []
    scores: list[float] = []
    for idx, raw_key in enumerate(sample_asset["keys"]):
        key = _canonicalize_key(raw_key[0], raw_key[1])
        if key not in feature_lookup:
            continue
        label_value: float
        score_value: float
        if sample_label is not None and sample_score is not None:
            label_value = float(sample_label[idx].item())
            score_value = float(sample_score[idx].item())
        elif key in reference_lookup:
            row = reference_lookup[key]
            label_value = float(row["label"])
            score_value = float(row["score"])
        else:
            continue

        if not _tensor_row_is_finite(sample_asset["ohlcv"][idx]):
            continue
        if not _tensor_row_is_finite(feature_asset["features"][feature_lookup[key]]):
            continue
        if not torch.isfinite(torch.tensor(label_value)):
            continue
        if not torch.isfinite(torch.tensor(score_value)):
            continue

        keys.append(key)
        sample_indices.append(idx)
        feature_indices.append(feature_lookup[key])
        labels.append(label_value)
        scores.append(score_value)
        if len(keys) >= max_samples:
            break

    if not keys:
        raise ValueError(f"no aligned keys found under {asset_dir}")

    return {
        "keys": keys,
        "ohlcv": sample_asset["ohlcv"][sample_indices].float(),
        "label": torch.tensor(labels, dtype=torch.float32),
        "score": torch.tensor(scores, dtype=torch.float32),
        "features": feature_asset["features"][feature_indices].float(),
        "normalization_stats": normalization_stats,
    }


def _build_pipeline(
    *,
    normalization_stats: dict[str, Any],
    raw_window_len: int,
    model: torch.nn.Module,
) -> RawFeatureAttackPipeline:
    center = torch.tensor(normalization_stats["center"], dtype=torch.float32)
    scale = torch.tensor(normalization_stats["scale"], dtype=torch.float32)
    scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
    return RawFeatureAttackPipeline(
        bridge=LegacyLSTMFeatureBridge(input_window_len=raw_window_len),
        norm=RobustZScoreNormLayer(
            center=center,
            scale=scale,
            clip_outlier=bool(normalization_stats.get("clip_outlier", False)),
        ),
        fillna=FillnaLayer(),
        model=model,
    )


def _sample_level_table(
    *,
    keys: list[tuple[str, str]],
    label: torch.Tensor,
    reference_score: torch.Tensor,
    clean_pred: torch.Tensor,
    fgsm_pred: torch.Tensor,
    pgd_pred: torch.Tensor,
) -> pd.DataFrame:
    rows = []
    for i, key in enumerate(keys):
        rows.append(
            {
                "datetime": key[0],
                "instrument": key[1],
                "label": float(label[i].item()),
                "reference_score": float(reference_score[i].item()),
                "clean_pred": float(clean_pred[i].item()),
                "fgsm_pred": float(fgsm_pred[i].item()),
                "pgd_pred": float(pgd_pred[i].item()),
                "clean_abs_error": float((clean_pred[i] - label[i]).abs().item()),
                "fgsm_abs_error": float((fgsm_pred[i] - label[i]).abs().item()),
                "pgd_abs_error": float((pgd_pred[i] - label[i]).abs().item()),
                "fgsm_abs_shift": float((fgsm_pred[i] - clean_pred[i]).abs().item()),
                "pgd_abs_shift": float((pgd_pred[i] - clean_pred[i]).abs().item()),
            }
        )
    return pd.DataFrame(rows)


def _render_report(summary: dict[str, Any]) -> str:
    clean_gate = summary["clean_gate"]
    return "\n".join(
        [
            f"# {summary['model_name']} 白盒攻击 Smoke 报告",
            "",
            "## 实验设置",
            f"- 样本数: `{summary['sample_count']}`",
            f"- 资产目录: `{summary['asset_dir']}`",
            f"- 配置文件: `{summary['config_path']}`",
            f"- 权重文件: `{summary['state_dict_path']}`",
            f"- price_epsilon: `{summary['price_epsilon']}`",
            f"- volume_epsilon: `{summary['volume_epsilon']}`",
            f"- pgd_steps: `{summary['pgd_steps']}`",
            f"- pgd_step_size: `{summary['pgd_step_size']}`",
            "",
            "## Clean Gate",
            f"- clean_loss: `{clean_gate['clean_loss']:.6f}`",
            f"- clean_grad_mean_abs: `{clean_gate['clean_grad_mean_abs']:.6e}`",
            f"- clean_grad_finite_rate: `{clean_gate['clean_grad_finite_rate']:.6f}`",
            f"- spearman_to_reference: `{clean_gate['spearman_to_reference']}`",
            f"- feature_mae_to_reference: `{clean_gate['feature_mae_to_reference']}`",
            f"- feature_rmse_to_reference: `{clean_gate['feature_rmse_to_reference']}`",
            f"- feature_max_abs_to_reference: `{clean_gate['feature_max_abs_to_reference']}`",
            "",
            "## 攻击结果",
            f"- FGSM MSE: `{summary['fgsm_loss']:.6f}`",
            f"- PGD MSE: `{summary['pgd_loss']:.6f}`",
            f"- FGSM 平均价格预算使用率: `{summary['fgsm_usage']['price_ratio_mean']:.6f}`",
            f"- PGD 平均价格预算使用率: `{summary['pgd_usage']['price_ratio_mean']:.6f}`",
            f"- FGSM 平均成交量预算使用率: `{summary['fgsm_usage']['volume_ratio_mean']:.6f}`",
            f"- PGD 平均成交量预算使用率: `{summary['pgd_usage']['volume_ratio_mean']:.6f}`",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = _choose_device(args.device)

    attack_assets = _load_attack_assets(args.asset_dir, args.max_samples)
    feature_model = load_model_adapter_from_paths(
        config_path=args.config_path,
        state_dict_path=args.state_dict_path,
        model_name=args.model_name,
        device=device,
    )
    pipeline = _build_pipeline(
        normalization_stats=attack_assets["normalization_stats"],
        raw_window_len=attack_assets["ohlcv"].shape[1],
        model=feature_model,
    ).to(device)
    pipeline.eval()
    for param in pipeline.parameters():
        param.requires_grad_(False)

    x = attack_assets["ohlcv"].to(device)
    y = attack_assets["label"].to(device)
    reference_score = attack_assets["score"].to(device)
    reference_features = attack_assets["features"].to(device)

    clean_gate_thresholds = CleanGateThresholds(
        min_clean_grad_mean_abs=args.min_clean_grad_mean_abs,
        min_spearman_to_reference=args.min_spearman_to_reference,
        max_feature_mae_to_reference=args.max_feature_mae_to_reference,
        max_feature_rmse_to_reference=args.max_feature_rmse_to_reference,
        max_feature_max_abs_to_reference=args.max_feature_max_abs_to_reference,
    )
    clean_gate = run_clean_gate(
        model=pipeline,
        x=x,
        y=y,
        reference_scores=reference_score,
        reference_features=reference_features,
    )
    validate_clean_gate(clean_gate, clean_gate_thresholds)

    with torch.no_grad():
        clean_pred = pipeline(x).detach().cpu()

    fgsm_x = fgsm_maximize_mse(
        model=pipeline,
        x=x,
        y=y,
        price_epsilon=args.price_epsilon,
        volume_epsilon=args.volume_epsilon,
        price_floor=args.price_floor,
        volume_floor=args.volume_floor,
    )
    pgd_x = pgd_maximize_mse(
        model=pipeline,
        x=x,
        y=y,
        price_epsilon=args.price_epsilon,
        volume_epsilon=args.volume_epsilon,
        num_steps=args.pgd_steps,
        step_size=args.pgd_step_size,
        price_floor=args.price_floor,
        volume_floor=args.volume_floor,
    )

    with torch.no_grad():
        fgsm_pred = pipeline(fgsm_x).detach().cpu()
        pgd_pred = pipeline(pgd_x).detach().cpu()

    fgsm_loss = float(F.mse_loss(fgsm_pred, y.detach().cpu()).item())
    pgd_loss = float(F.mse_loss(pgd_pred, y.detach().cpu()).item())
    sample_metrics = _sample_level_table(
        keys=attack_assets["keys"],
        label=y.detach().cpu(),
        reference_score=reference_score.detach().cpu(),
        clean_pred=clean_pred,
        fgsm_pred=fgsm_pred,
        pgd_pred=pgd_pred,
    )

    summary = {
        "model_name": args.model_name,
        "config_path": str(args.config_path),
        "state_dict_path": str(args.state_dict_path),
        "asset_dir": str(args.asset_dir),
        "sample_count": len(attack_assets["keys"]),
        "price_epsilon": args.price_epsilon,
        "volume_epsilon": args.volume_epsilon,
        "price_floor": args.price_floor,
        "volume_floor": args.volume_floor,
        "pgd_steps": args.pgd_steps,
        "pgd_step_size": args.pgd_step_size,
        "clean_gate": asdict(clean_gate),
        "clean_gate_thresholds": asdict(clean_gate_thresholds),
        "clean_loss": float(F.mse_loss(clean_pred, y.detach().cpu()).item()),
        "fgsm_loss": fgsm_loss,
        "pgd_loss": pgd_loss,
        "fgsm_usage": usage_ratio(
            fgsm_x.detach().cpu(),
            x.detach().cpu(),
            price_epsilon=args.price_epsilon,
            volume_epsilon=args.volume_epsilon,
            price_floor=args.price_floor,
            volume_floor=args.volume_floor,
        ),
        "pgd_usage": usage_ratio(
            pgd_x.detach().cpu(),
            x.detach().cpu(),
            price_epsilon=args.price_epsilon,
            volume_epsilon=args.volume_epsilon,
            price_floor=args.price_floor,
            volume_floor=args.volume_floor,
        ),
    }

    (args.out_dir / "attack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sample_metrics.to_csv(args.out_dir / "sample_metrics.csv", index=False)
    (args.out_dir / "attack_report.md").write_text(_render_report(summary), encoding="utf-8")

    print(f"attack_summary={args.out_dir / 'attack_summary.json'}")


if __name__ == "__main__":
    main()
