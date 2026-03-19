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

from legacy_lstm_attack_core import (
    CleanGateMetrics,
    CleanGateThresholds,
    LegacyRawLSTMPipeline,
    compute_attack_objective,
    compute_input_gradients,
    fgsm_maximize_mse,
    pgd_maximize_mse,
    run_clean_gate,
    usage_ratio,
    validate_clean_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke FGSM/PGD attacks on the legacy raw-OHLCV LSTM pipeline.")
    parser.add_argument("--asset-dir", type=Path, default=Path("artifacts/lstm_attack"))
    parser.add_argument("--state-dict-path", type=Path, default=Path("model/lstm_state_dict.pt"))
    parser.add_argument("--config-path", type=Path, default=Path("model/lstm_state_dict_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/lstm_whitebox_attack"))
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
    parser.add_argument("--constraint-mode", type=str, default="none", choices=["none", "physical", "physical_stat"])
    parser.add_argument("--tau-ret", type=float, default=0.005)
    parser.add_argument("--tau-body", type=float, default=0.005)
    parser.add_argument("--tau-range", type=float, default=0.01)
    parser.add_argument("--tau-vol", type=float, default=0.05)
    parser.add_argument("--lambda-ret", type=float, default=0.8)
    parser.add_argument("--lambda-candle", type=float, default=0.4)
    parser.add_argument("--lambda-vol", type=float, default=0.3)
    return parser.parse_args(argv)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_model_config_path(config_path: Path, state_dict_path: Path) -> Path:
    if config_path.exists():
        return config_path

    sibling_candidates = [
        state_dict_path.parent / "model_config.json",
        state_dict_path.with_name(f"{state_dict_path.stem}_config.json"),
    ]
    for candidate in sibling_candidates:
        if candidate.exists():
            return candidate

    matching_configs = sorted(state_dict_path.parent.glob("*config*.json"))
    if len(matching_configs) == 1:
        return matching_configs[0]

    raise FileNotFoundError(
        f"Could not find model config. Tried explicit path {config_path} and sibling candidates near {state_dict_path}."
    )


def _load_attack_assets(asset_dir: Path, max_samples: int) -> dict[str, Any]:
    sample_asset = torch.load(asset_dir / "matched_ohlcv_windows.pt", map_location="cpu")
    feature_asset_path = asset_dir / "matched_feature_windows.pt"
    feature_asset = torch.load(feature_asset_path, map_location="cpu") if feature_asset_path.exists() else None
    if feature_asset is not None:
        feature_lookup = {tuple(key): idx for idx, key in enumerate(feature_asset["keys"])}
        keep_indices = [i for i, key in enumerate(sample_asset["keys"]) if tuple(key) in feature_lookup]
        if len(keep_indices) != len(sample_asset["keys"]):
            sample_asset["keys"] = [sample_asset["keys"][i] for i in keep_indices]
            sample_asset["ohlcv"] = sample_asset["ohlcv"][keep_indices]
            sample_asset["label"] = sample_asset["label"][keep_indices]
            sample_asset["score"] = sample_asset["score"][keep_indices]
        aligned_feature_idx = [feature_lookup[tuple(key)] for key in sample_asset["keys"]]
        feature_asset = {
            "keys": list(sample_asset["keys"]),
            "features": feature_asset["features"][aligned_feature_idx],
            "missing_keys": feature_asset.get("missing_keys", []),
        }
    if max_samples < len(sample_asset["keys"]):
        keep = slice(0, max_samples)
        sample_asset["keys"] = sample_asset["keys"][keep]
        sample_asset["ohlcv"] = sample_asset["ohlcv"][keep]
        sample_asset["label"] = sample_asset["label"][keep]
        sample_asset["score"] = sample_asset["score"][keep]
        if feature_asset is not None:
            feature_asset["keys"] = feature_asset["keys"][keep]
            feature_asset["features"] = feature_asset["features"][keep]
    normalization_stats = json.loads((asset_dir / "normalization_stats.json").read_text(encoding="utf-8"))
    return {
        "sample_asset": sample_asset,
        "feature_asset": feature_asset,
        "normalization_stats": normalization_stats,
    }


def _sample_level_table(
    keys: list[tuple[str, str]],
    clean_pred: torch.Tensor,
    fgsm_pred: torch.Tensor,
    pgd_pred: torch.Tensor,
    label: torch.Tensor,
) -> pd.DataFrame:
    rows = []
    for i, key in enumerate(keys):
        rows.append(
            {
                "datetime": key[0],
                "instrument": key[1],
                "label": float(label[i].item()),
                "clean_pred": float(clean_pred[i].item()),
                "fgsm_pred": float(fgsm_pred[i].item()),
                "pgd_pred": float(pgd_pred[i].item()),
                "fgsm_abs_shift": float((fgsm_pred[i] - clean_pred[i]).abs().item()),
                "pgd_abs_shift": float((pgd_pred[i] - clean_pred[i]).abs().item()),
            }
        )
    return pd.DataFrame(rows)


def _mean_abs_or_zero(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.abs().mean().item())


def _stat_shift_summary(x_adv: torch.Tensor, x_clean: torch.Tensor, *, price_floor: float) -> dict[str, float]:
    adv_close = torch.clamp(x_adv[..., 3], min=0.0) + price_floor
    clean_close = torch.clamp(x_clean[..., 3], min=0.0) + price_floor
    adv_ret = torch.log(adv_close[..., 1:]) - torch.log(adv_close[..., :-1])
    clean_ret = torch.log(clean_close[..., 1:]) - torch.log(clean_close[..., :-1])

    adv_open = torch.clamp(x_adv[..., 0].abs(), min=price_floor)
    clean_open = torch.clamp(x_clean[..., 0].abs(), min=price_floor)
    adv_body = (x_adv[..., 3] - x_adv[..., 0]) / adv_open
    clean_body = (x_clean[..., 3] - x_clean[..., 0]) / clean_open
    adv_range = (x_adv[..., 1] - x_adv[..., 2]) / adv_open
    clean_range = (x_clean[..., 1] - x_clean[..., 2]) / clean_open

    adv_logvol = torch.log1p(torch.clamp(x_adv[..., 4], min=0.0))
    clean_logvol = torch.log1p(torch.clamp(x_clean[..., 4], min=0.0))
    adv_dlogvol = adv_logvol[..., 1:] - adv_logvol[..., :-1]
    clean_dlogvol = clean_logvol[..., 1:] - clean_logvol[..., :-1]

    return {
        "mean_abs_ret_shift": _mean_abs_or_zero(adv_ret - clean_ret),
        "mean_abs_body_shift": _mean_abs_or_zero(adv_body - clean_body),
        "mean_abs_range_shift": _mean_abs_or_zero(adv_range - clean_range),
        "mean_abs_dlogvol_shift": _mean_abs_or_zero(adv_dlogvol - clean_dlogvol),
    }


def _physical_constraints_satisfied(x: torch.Tensor, *, price_floor: float, volume_floor: float) -> bool:
    open_ = x[..., 0]
    high = x[..., 1]
    low = x[..., 2]
    close = x[..., 3]
    volume = x[..., 4]
    satisfied = (
        (open_ >= price_floor)
        & (high >= price_floor)
        & (low >= price_floor)
        & (close >= price_floor)
        & (volume >= volume_floor)
        & (high >= torch.maximum(open_, close))
        & (low <= torch.minimum(open_, close))
        & (low <= high)
    )
    return bool(satisfied.all().item())


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    config_path = resolve_model_config_path(args.config_path, args.state_dict_path)

    assets = _load_attack_assets(args.asset_dir, args.max_samples)
    sample_asset = assets["sample_asset"]
    feature_asset = assets["feature_asset"]
    normalization_stats = assets["normalization_stats"]

    model = LegacyRawLSTMPipeline(
        normalization_stats=normalization_stats,
        state_dict_path=args.state_dict_path,
        config_path=config_path,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    x = sample_asset["ohlcv"].to(device)
    y = sample_asset["label"].to(device)
    ref_score = sample_asset["score"].to(device)
    ref_features = feature_asset["features"].to(device) if feature_asset is not None else None
    clean_gate_thresholds = CleanGateThresholds(
        min_clean_grad_mean_abs=args.min_clean_grad_mean_abs,
        min_spearman_to_reference=args.min_spearman_to_reference,
        max_feature_mae_to_reference=args.max_feature_mae_to_reference,
        max_feature_rmse_to_reference=args.max_feature_rmse_to_reference,
        max_feature_max_abs_to_reference=args.max_feature_max_abs_to_reference,
    )

    clean_gate = run_clean_gate(
        model=model,
        x=x,
        y=y,
        reference_scores=ref_score,
        reference_features=ref_features,
    )
    validate_clean_gate(clean_gate, clean_gate_thresholds)

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
        clean_loss = F.mse_loss(clean_pred, y)
        fgsm_loss = F.mse_loss(fgsm_pred, y)
        pgd_loss = F.mse_loss(pgd_pred, y)
        clean_objective = compute_attack_objective(
            model=model,
            x_adv=x,
            y=y,
            x_clean=x,
            constraint_mode=args.constraint_mode,
            tau_ret=args.tau_ret,
            tau_body=args.tau_body,
            tau_range=args.tau_range,
            tau_vol=args.tau_vol,
            lambda_ret=args.lambda_ret,
            lambda_candle=args.lambda_candle,
            lambda_vol=args.lambda_vol,
            price_floor=args.price_floor,
        )
        fgsm_objective = compute_attack_objective(
            model=model,
            x_adv=fgsm_x,
            y=y,
            x_clean=x,
            constraint_mode=args.constraint_mode,
            tau_ret=args.tau_ret,
            tau_body=args.tau_body,
            tau_range=args.tau_range,
            tau_vol=args.tau_vol,
            lambda_ret=args.lambda_ret,
            lambda_candle=args.lambda_candle,
            lambda_vol=args.lambda_vol,
            price_floor=args.price_floor,
        )
        pgd_objective = compute_attack_objective(
            model=model,
            x_adv=pgd_x,
            y=y,
            x_clean=x,
            constraint_mode=args.constraint_mode,
            tau_ret=args.tau_ret,
            tau_body=args.tau_body,
            tau_range=args.tau_range,
            tau_vol=args.tau_vol,
            lambda_ret=args.lambda_ret,
            lambda_candle=args.lambda_candle,
            lambda_vol=args.lambda_vol,
            price_floor=args.price_floor,
        )

    fgsm_shift = _stat_shift_summary(fgsm_x, x, price_floor=args.price_floor)
    pgd_shift = _stat_shift_summary(pgd_x, x, price_floor=args.price_floor)
    fgsm_physical_ok = _physical_constraints_satisfied(
        fgsm_x,
        price_floor=args.price_floor,
        volume_floor=args.volume_floor,
    )
    pgd_physical_ok = _physical_constraints_satisfied(
        pgd_x,
        price_floor=args.price_floor,
        volume_floor=args.volume_floor,
    )

    sample_table = _sample_level_table(sample_asset["keys"], clean_pred, fgsm_pred, pgd_pred, y)
    sample_csv = args.out_dir / "sample_metrics.csv"
    sample_table.to_csv(sample_csv, index=False)

    summary = {
        "num_samples": int(x.shape[0]),
        "constraint_mode": args.constraint_mode,
        "clean_gate": asdict(clean_gate),
        "clean_gate_thresholds": asdict(clean_gate_thresholds),
        "clean_loss": float(clean_loss.item()),
        "fgsm_loss": float(fgsm_loss.item()),
        "pgd_loss": float(pgd_loss.item()),
        "objective_clean": float(clean_objective.objective.item()),
        "objective_fgsm": float(fgsm_objective.objective.item()),
        "objective_pgd": float(pgd_objective.objective.item()),
        "fgsm_adv_success": bool(fgsm_loss.item() > clean_loss.item()),
        "pgd_adv_success": bool(pgd_loss.item() > clean_loss.item()),
        "fgsm_mean_abs_pred_shift": float((fgsm_pred - clean_pred).abs().mean().item()),
        "pgd_mean_abs_pred_shift": float((pgd_pred - clean_pred).abs().mean().item()),
        "ret_penalty_fgsm": float(fgsm_objective.ret_penalty.item()),
        "ret_penalty_pgd": float(pgd_objective.ret_penalty.item()),
        "candle_penalty_fgsm": float(fgsm_objective.candle_penalty.item()),
        "candle_penalty_pgd": float(pgd_objective.candle_penalty.item()),
        "vol_penalty_fgsm": float(fgsm_objective.vol_penalty.item()),
        "vol_penalty_pgd": float(pgd_objective.vol_penalty.item()),
        "mean_abs_ret_shift_fgsm": fgsm_shift["mean_abs_ret_shift"],
        "mean_abs_ret_shift_pgd": pgd_shift["mean_abs_ret_shift"],
        "mean_abs_body_shift_fgsm": fgsm_shift["mean_abs_body_shift"],
        "mean_abs_body_shift_pgd": pgd_shift["mean_abs_body_shift"],
        "mean_abs_range_shift_fgsm": fgsm_shift["mean_abs_range_shift"],
        "mean_abs_range_shift_pgd": pgd_shift["mean_abs_range_shift"],
        "mean_abs_dlogvol_shift_fgsm": fgsm_shift["mean_abs_dlogvol_shift"],
        "mean_abs_dlogvol_shift_pgd": pgd_shift["mean_abs_dlogvol_shift"],
        "physical_constraints_satisfied_fgsm": fgsm_physical_ok,
        "physical_constraints_satisfied_pgd": pgd_physical_ok,
        "strict_attack_success_fgsm": bool(
            fgsm_loss.item() > clean_loss.item()
            and fgsm_physical_ok
            and fgsm_objective.objective.item() > clean_objective.objective.item()
        ),
        "strict_attack_success_pgd": bool(
            pgd_loss.item() > clean_loss.item()
            and pgd_physical_ok
            and pgd_objective.objective.item() > clean_objective.objective.item()
        ),
        "fgsm_usage": usage_ratio(
            fgsm_x,
            x,
            price_epsilon=args.price_epsilon,
            volume_epsilon=args.volume_epsilon,
            price_floor=args.price_floor,
            volume_floor=args.volume_floor,
        ),
        "pgd_usage": usage_ratio(
            pgd_x,
            x,
            price_epsilon=args.price_epsilon,
            volume_epsilon=args.volume_epsilon,
            price_floor=args.price_floor,
            volume_floor=args.volume_floor,
        ),
    }
    summary_path = args.out_dir / "attack_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_md = args.out_dir / "attack_report.md"
    report_md.write_text(
        "\n".join(
            [
                "# LSTM 白盒攻击 Smoke 结果",
                "",
                f"- 样本数：{summary['num_samples']}",
                f"- 约束模式：{summary['constraint_mode']}",
                f"- clean_loss：{summary['clean_loss']:.8f}",
                f"- fgsm_loss：{summary['fgsm_loss']:.8f}",
                f"- pgd_loss：{summary['pgd_loss']:.8f}",
                f"- objective_clean：{summary['objective_clean']:.8f}",
                f"- objective_fgsm：{summary['objective_fgsm']:.8f}",
                f"- objective_pgd：{summary['objective_pgd']:.8f}",
                f"- clean Spearman 对齐：{summary['clean_gate']['spearman_to_reference']}",
                f"- clean 特征 MAE：{summary['clean_gate']['feature_mae_to_reference']}",
                f"- clean 特征 RMSE：{summary['clean_gate']['feature_rmse_to_reference']}",
                f"- clean gate 阈值：`{summary['clean_gate_thresholds']}`",
                f"- FGSM 平均预测偏移：{summary['fgsm_mean_abs_pred_shift']:.8f}",
                f"- PGD 平均预测偏移：{summary['pgd_mean_abs_pred_shift']:.8f}",
                f"- FGSM 物理约束满足：{summary['physical_constraints_satisfied_fgsm']}",
                f"- PGD 物理约束满足：{summary['physical_constraints_satisfied_pgd']}",
                f"- FGSM strict success：{summary['strict_attack_success_fgsm']}",
                f"- PGD strict success：{summary['strict_attack_success_pgd']}",
                f"- FGSM penalty(ret/candle/vol)：{summary['ret_penalty_fgsm']:.8f} / {summary['candle_penalty_fgsm']:.8f} / {summary['vol_penalty_fgsm']:.8f}",
                f"- PGD penalty(ret/candle/vol)：{summary['ret_penalty_pgd']:.8f} / {summary['candle_penalty_pgd']:.8f} / {summary['vol_penalty_pgd']:.8f}",
                f"- 样本级明细：`{sample_csv}`",
                f"- 汇总 JSON：`{summary_path}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"clean_loss={summary['clean_loss']:.8f}")
    print(f"fgsm_loss={summary['fgsm_loss']:.8f}")
    print(f"pgd_loss={summary['pgd_loss']:.8f}")
    print(f"sample_metrics_csv={sample_csv}")
    print(f"attack_summary_json={summary_path}")


if __name__ == "__main__":
    main()
