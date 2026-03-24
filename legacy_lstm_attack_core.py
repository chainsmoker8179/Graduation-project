from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from legacy_lstm_feature_bridge import LegacyLSTMFeatureBridge
from legacy_lstm_predictor import load_legacy_lstm_from_files
from legacy_lstm_preprocess import FillnaLayer, RobustZScoreNormLayer
from whitebox_attack_core import RawFeatureAttackPipeline


@dataclass
class CleanGateMetrics:
    clean_loss: float
    clean_grad_mean_abs: float
    clean_grad_max_abs: float
    clean_grad_finite_rate: float
    feature_finite_rate: float
    clean_pred_mean: float
    clean_pred_std: float
    reference_score_mean: float
    reference_score_std: float
    spearman_to_reference: float | None
    feature_mae_to_reference: float | None
    feature_rmse_to_reference: float | None
    feature_max_abs_to_reference: float | None


@dataclass
class CleanGateThresholds:
    min_clean_grad_mean_abs: float = 1e-6
    min_spearman_to_reference: float | None = 0.09
    max_feature_mae_to_reference: float | None = 0.05
    max_feature_rmse_to_reference: float | None = 0.12
    max_feature_max_abs_to_reference: float | None = 0.7


@dataclass
class ConstrainedAttackObjective:
    mse_loss: torch.Tensor
    ret_penalty: torch.Tensor
    candle_penalty: torch.Tensor
    vol_penalty: torch.Tensor
    objective: torch.Tensor


class LegacyRawLSTMPipeline(RawFeatureAttackPipeline):
    def __init__(
        self,
        normalization_stats: dict[str, Any],
        *,
        state_dict_path: Path,
        config_path: Path,
    ) -> None:
        bridge = LegacyLSTMFeatureBridge()
        norm = RobustZScoreNormLayer(
            center=torch.tensor(normalization_stats["center"], dtype=torch.float32),
            scale=torch.tensor(normalization_stats["scale"], dtype=torch.float32),
            clip_outlier=bool(normalization_stats["clip_outlier"]),
        )
        fillna = FillnaLayer()
        predictor = load_legacy_lstm_from_files(config_path=config_path, state_dict_path=state_dict_path)
        super().__init__(bridge=bridge, norm=norm, fillna=fillna, model=predictor)
        self.predictor = predictor


def relative_budget(
    x_clean: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float = 1e-6,
    volume_floor: float = 1.0,
) -> torch.Tensor:
    price_base = torch.maximum(x_clean[..., :4].abs(), torch.full_like(x_clean[..., :4], price_floor))
    volume_base = torch.maximum(x_clean[..., 4:].abs(), torch.full_like(x_clean[..., 4:], volume_floor))
    price_budget = price_base * price_epsilon
    volume_budget = volume_base * volume_epsilon
    return torch.cat([price_budget, volume_budget], dim=-1)


def project_relative_box(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float = 1e-6,
    volume_floor: float = 1.0,
) -> torch.Tensor:
    budget = relative_budget(
        x_clean,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
    )
    lower = torch.clamp(x_clean - budget, min=0.0)
    upper = torch.clamp(x_clean + budget, min=0.0)
    return torch.maximum(torch.minimum(x_adv, upper), lower)


def project_financial_feasible_box(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float = 1e-6,
    volume_floor: float = 1.0,
) -> torch.Tensor:
    projected = project_relative_box(
        x_adv,
        x_clean,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
    )

    price = torch.clamp(projected[..., :4], min=price_floor)
    volume = torch.clamp(projected[..., 4:], min=volume_floor)

    open_ = price[..., 0]
    high = price[..., 1]
    low = price[..., 2]
    close = price[..., 3]

    high = torch.maximum(high, torch.maximum(open_, close))
    low = torch.minimum(low, torch.minimum(open_, close))
    low = torch.minimum(low, high)

    return torch.cat(
        [
            open_.unsqueeze(-1),
            high.unsqueeze(-1),
            low.unsqueeze(-1),
            close.unsqueeze(-1),
            volume,
        ],
        dim=-1,
    )


def compute_input_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_var = x.detach().clone().requires_grad_(True)
    pred = model(x_var)
    loss = F.mse_loss(pred, y)
    loss.backward()
    return loss.detach(), x_var.grad.detach().clone()


def _hinge_squared_tolerance_penalty(
    adv_metric: torch.Tensor,
    clean_metric: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    if adv_metric.numel() == 0:
        return adv_metric.new_tensor(0.0)
    excess = F.relu((adv_metric - clean_metric).abs() / tau - 1.0)
    return excess.square().mean()


def compute_return_penalty(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    tau_ret: float,
    price_floor: float = 1e-6,
) -> torch.Tensor:
    adv_close = torch.clamp(x_adv[..., 3], min=0.0) + price_floor
    clean_close = torch.clamp(x_clean[..., 3], min=0.0) + price_floor
    adv_ret = torch.log(adv_close[..., 1:]) - torch.log(adv_close[..., :-1])
    clean_ret = torch.log(clean_close[..., 1:]) - torch.log(clean_close[..., :-1])
    return _hinge_squared_tolerance_penalty(adv_ret, clean_ret, tau=tau_ret)


def compute_candle_penalty(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    tau_body: float,
    tau_range: float,
    price_floor: float = 1e-6,
) -> torch.Tensor:
    adv_open = torch.clamp(x_adv[..., 0].abs(), min=price_floor)
    clean_open = torch.clamp(x_clean[..., 0].abs(), min=price_floor)

    adv_body = (x_adv[..., 3] - x_adv[..., 0]) / adv_open
    clean_body = (x_clean[..., 3] - x_clean[..., 0]) / clean_open

    adv_range = (x_adv[..., 1] - x_adv[..., 2]) / adv_open
    clean_range = (x_clean[..., 1] - x_clean[..., 2]) / clean_open

    body_penalty = _hinge_squared_tolerance_penalty(adv_body, clean_body, tau=tau_body)
    range_penalty = _hinge_squared_tolerance_penalty(adv_range, clean_range, tau=tau_range)
    return 0.5 * (body_penalty + range_penalty)


def compute_volume_penalty(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    tau_vol: float,
) -> torch.Tensor:
    adv_volume = torch.clamp(x_adv[..., 4], min=0.0)
    clean_volume = torch.clamp(x_clean[..., 4], min=0.0)
    adv_log_volume = torch.log1p(adv_volume)
    clean_log_volume = torch.log1p(clean_volume)
    adv_delta = adv_log_volume[..., 1:] - adv_log_volume[..., :-1]
    clean_delta = clean_log_volume[..., 1:] - clean_log_volume[..., :-1]
    return _hinge_squared_tolerance_penalty(adv_delta, clean_delta, tau=tau_vol)


def compute_constrained_attack_objective(
    model: nn.Module,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    tau_ret: float,
    tau_body: float,
    tau_range: float,
    tau_vol: float,
    lambda_ret: float,
    lambda_candle: float,
    lambda_vol: float,
    price_floor: float = 1e-6,
) -> ConstrainedAttackObjective:
    pred = model(x_adv)
    mse_loss = F.mse_loss(pred, y)
    ret_penalty = compute_return_penalty(x_adv, x_clean, tau_ret=tau_ret, price_floor=price_floor)
    candle_penalty = compute_candle_penalty(
        x_adv,
        x_clean,
        tau_body=tau_body,
        tau_range=tau_range,
        price_floor=price_floor,
    )
    vol_penalty = compute_volume_penalty(x_adv, x_clean, tau_vol=tau_vol)
    objective = mse_loss - lambda_ret * ret_penalty - lambda_candle * candle_penalty - lambda_vol * vol_penalty
    return ConstrainedAttackObjective(
        mse_loss=mse_loss,
        ret_penalty=ret_penalty,
        candle_penalty=candle_penalty,
        vol_penalty=vol_penalty,
        objective=objective,
    )


def compute_attack_objective(
    model: nn.Module,
    x_adv: torch.Tensor,
    y: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    constraint_mode: str,
    tau_ret: float,
    tau_body: float,
    tau_range: float,
    tau_vol: float,
    lambda_ret: float,
    lambda_candle: float,
    lambda_vol: float,
    price_floor: float = 1e-6,
) -> ConstrainedAttackObjective:
    if constraint_mode in {"none", "physical"}:
        pred = model(x_adv)
        mse_loss = F.mse_loss(pred, y)
        zero = mse_loss.new_tensor(0.0)
        return ConstrainedAttackObjective(
            mse_loss=mse_loss,
            ret_penalty=zero,
            candle_penalty=zero,
            vol_penalty=zero,
            objective=mse_loss,
        )
    if constraint_mode == "physical_stat":
        return compute_constrained_attack_objective(
            model=model,
            x_adv=x_adv,
            y=y,
            x_clean=x_clean,
            tau_ret=tau_ret,
            tau_body=tau_body,
            tau_range=tau_range,
            tau_vol=tau_vol,
            lambda_ret=lambda_ret,
            lambda_candle=lambda_candle,
            lambda_vol=lambda_vol,
            price_floor=price_floor,
        )
    raise ValueError(f"Unsupported constraint_mode: {constraint_mode}")


def compute_attack_objective_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    x_clean: torch.Tensor,
    constraint_mode: str,
    tau_ret: float,
    tau_body: float,
    tau_range: float,
    tau_vol: float,
    lambda_ret: float,
    lambda_candle: float,
    lambda_vol: float,
    price_floor: float = 1e-6,
) -> tuple[ConstrainedAttackObjective, torch.Tensor]:
    x_var = x.detach().clone().requires_grad_(True)
    result = compute_attack_objective(
        model=model,
        x_adv=x_var,
        y=y,
        x_clean=x_clean,
        constraint_mode=constraint_mode,
        tau_ret=tau_ret,
        tau_body=tau_body,
        tau_range=tau_range,
        tau_vol=tau_vol,
        lambda_ret=lambda_ret,
        lambda_candle=lambda_candle,
        lambda_vol=lambda_vol,
        price_floor=price_floor,
    )
    result.objective.backward()
    return result, x_var.grad.detach().clone()


def project_with_constraint_mode(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float,
    volume_floor: float,
    constraint_mode: str,
) -> torch.Tensor:
    if constraint_mode == "none":
        return project_relative_box(
            x_adv,
            x_clean,
            price_epsilon=price_epsilon,
            volume_epsilon=volume_epsilon,
            price_floor=price_floor,
            volume_floor=volume_floor,
        )
    if constraint_mode in {"physical", "physical_stat"}:
        return project_financial_feasible_box(
            x_adv,
            x_clean,
            price_epsilon=price_epsilon,
            volume_epsilon=volume_epsilon,
            price_floor=price_floor,
            volume_floor=volume_floor,
        )
    raise ValueError(f"Unsupported constraint_mode: {constraint_mode}")


def fgsm_maximize_mse(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float = 1e-6,
    volume_floor: float = 1.0,
    constraint_mode: str = "none",
    tau_ret: float = 0.005,
    tau_body: float = 0.005,
    tau_range: float = 0.01,
    tau_vol: float = 0.05,
    lambda_ret: float = 0.8,
    lambda_candle: float = 0.4,
    lambda_vol: float = 0.3,
) -> torch.Tensor:
    if constraint_mode == "physical_stat":
        _, grad = compute_attack_objective_gradients(
            model=model,
            x=x,
            y=y,
            x_clean=x,
            constraint_mode=constraint_mode,
            tau_ret=tau_ret,
            tau_body=tau_body,
            tau_range=tau_range,
            tau_vol=tau_vol,
            lambda_ret=lambda_ret,
            lambda_candle=lambda_candle,
            lambda_vol=lambda_vol,
            price_floor=price_floor,
        )
    else:
        _, grad = compute_input_gradients(model=model, x=x, y=y)
    budget = relative_budget(
        x,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
    )
    adv = x + budget * grad.sign()
    return project_with_constraint_mode(
        adv,
        x,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
        constraint_mode=constraint_mode,
    ).detach()


def pgd_maximize_mse(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    num_steps: int,
    step_size: float,
    price_floor: float = 1e-6,
    volume_floor: float = 1.0,
    constraint_mode: str = "none",
    tau_ret: float = 0.005,
    tau_body: float = 0.005,
    tau_range: float = 0.01,
    tau_vol: float = 0.05,
    lambda_ret: float = 0.8,
    lambda_candle: float = 0.4,
    lambda_vol: float = 0.3,
) -> torch.Tensor:
    budget = relative_budget(
        x,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
    )
    step = budget * step_size
    adv = x.detach().clone()
    for _ in range(num_steps):
        if constraint_mode == "physical_stat":
            _, grad = compute_attack_objective_gradients(
                model=model,
                x=adv,
                y=y,
                x_clean=x,
                constraint_mode=constraint_mode,
                tau_ret=tau_ret,
                tau_body=tau_body,
                tau_range=tau_range,
                tau_vol=tau_vol,
                lambda_ret=lambda_ret,
                lambda_candle=lambda_candle,
                lambda_vol=lambda_vol,
                price_floor=price_floor,
            )
        else:
            _, grad = compute_input_gradients(model=model, x=adv, y=y)
        adv = adv + step * grad.sign()
        adv = project_with_constraint_mode(
            adv,
            x,
            price_epsilon=price_epsilon,
            volume_epsilon=volume_epsilon,
            price_floor=price_floor,
            volume_floor=volume_floor,
            constraint_mode=constraint_mode,
        ).detach()
    return adv


def spearman_correlation(pred: torch.Tensor, ref: torch.Tensor) -> float | None:
    pred_np = pred.detach().cpu().numpy()
    ref_np = ref.detach().cpu().numpy()
    if len(pred_np) < 2:
        return None
    pred_rank = pd.Series(pred_np).rank(method="average")
    ref_rank = pd.Series(ref_np).rank(method="average")
    corr = pred_rank.corr(ref_rank)
    if corr is None or math.isnan(corr):
        return None
    return float(corr)


def usage_ratio(
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float,
    volume_floor: float,
) -> dict[str, float]:
    budget = relative_budget(
        x_clean,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
    )
    ratio = (x_adv - x_clean).abs() / torch.clamp(budget, min=1e-12)
    return {
        "price_ratio_mean": float(ratio[..., :4].mean().item()),
        "price_ratio_max": float(ratio[..., :4].max().item()),
        "volume_ratio_mean": float(ratio[..., 4:].mean().item()),
        "volume_ratio_max": float(ratio[..., 4:].max().item()),
    }


def run_clean_gate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    reference_scores: torch.Tensor,
    reference_features: torch.Tensor | None = None,
) -> CleanGateMetrics:
    with torch.no_grad():
        features = model.forward_features(x)
        pred = model.predictor(features)

    loss, grad = compute_input_gradients(model=model, x=x, y=y)
    if reference_features is not None:
        feature_diff = features - reference_features
        feature_mae = float(feature_diff.abs().mean().item())
        feature_rmse = float(torch.sqrt(torch.mean(feature_diff.square())).item())
        feature_max_abs = float(feature_diff.abs().max().item())
    else:
        feature_mae = None
        feature_rmse = None
        feature_max_abs = None
    return CleanGateMetrics(
        clean_loss=float(loss.item()),
        clean_grad_mean_abs=float(grad.abs().mean().item()),
        clean_grad_max_abs=float(grad.abs().max().item()),
        clean_grad_finite_rate=float(torch.isfinite(grad).float().mean().item()),
        feature_finite_rate=float(torch.isfinite(features).float().mean().item()),
        clean_pred_mean=float(pred.mean().item()),
        clean_pred_std=float(pred.std(unbiased=False).item()),
        reference_score_mean=float(reference_scores.mean().item()),
        reference_score_std=float(reference_scores.std(unbiased=False).item()),
        spearman_to_reference=spearman_correlation(pred, reference_scores),
        feature_mae_to_reference=feature_mae,
        feature_rmse_to_reference=feature_rmse,
        feature_max_abs_to_reference=feature_max_abs,
    )


def validate_clean_gate(metrics: CleanGateMetrics, thresholds: CleanGateThresholds) -> None:
    failures: list[str] = []
    if metrics.feature_finite_rate < 1.0:
        failures.append(f"feature_finite_rate={metrics.feature_finite_rate}")
    if metrics.clean_grad_finite_rate < 1.0:
        failures.append(f"clean_grad_finite_rate={metrics.clean_grad_finite_rate}")
    if metrics.clean_grad_mean_abs < thresholds.min_clean_grad_mean_abs:
        failures.append(f"clean_grad_mean_abs={metrics.clean_grad_mean_abs} < {thresholds.min_clean_grad_mean_abs}")
    if thresholds.min_spearman_to_reference is not None:
        if metrics.spearman_to_reference is None or metrics.spearman_to_reference < thresholds.min_spearman_to_reference:
            failures.append(
                f"spearman_to_reference={metrics.spearman_to_reference} < {thresholds.min_spearman_to_reference}"
            )
    if thresholds.max_feature_mae_to_reference is not None and metrics.feature_mae_to_reference is not None:
        if metrics.feature_mae_to_reference > thresholds.max_feature_mae_to_reference:
            failures.append(
                f"feature_mae_to_reference={metrics.feature_mae_to_reference} > {thresholds.max_feature_mae_to_reference}"
            )
    if thresholds.max_feature_rmse_to_reference is not None and metrics.feature_rmse_to_reference is not None:
        if metrics.feature_rmse_to_reference > thresholds.max_feature_rmse_to_reference:
            failures.append(
                f"feature_rmse_to_reference={metrics.feature_rmse_to_reference} > {thresholds.max_feature_rmse_to_reference}"
            )
    if thresholds.max_feature_max_abs_to_reference is not None and metrics.feature_max_abs_to_reference is not None:
        if metrics.feature_max_abs_to_reference > thresholds.max_feature_max_abs_to_reference:
            failures.append(
                f"feature_max_abs_to_reference={metrics.feature_max_abs_to_reference} > {thresholds.max_feature_max_abs_to_reference}"
            )
    if failures:
        raise ValueError("clean gate failed: " + "; ".join(failures))


__all__ = [
    "CleanGateMetrics",
    "CleanGateThresholds",
    "LegacyRawLSTMPipeline",
    "RawFeatureAttackPipeline",
    "compute_input_gradients",
    "fgsm_maximize_mse",
    "pgd_maximize_mse",
    "project_relative_box",
    "relative_budget",
    "run_clean_gate",
    "spearman_correlation",
    "usage_ratio",
    "validate_clean_gate",
]
