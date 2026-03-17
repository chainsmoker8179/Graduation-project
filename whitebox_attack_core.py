from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RawFeatureAttackPipeline(nn.Module):
    def __init__(
        self,
        *,
        bridge: nn.Module,
        norm: nn.Module,
        fillna: nn.Module,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.norm = norm
        self.fillna = fillna
        self.model = model

    def forward_features(self, raw_ohlcv: torch.Tensor) -> torch.Tensor:
        features = self.bridge(raw_ohlcv)
        features = self.norm(features)
        features = self.fillna(features)
        return features

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def forward(self, raw_ohlcv: torch.Tensor) -> torch.Tensor:
        return self.forward_from_features(self.forward_features(raw_ohlcv))


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


def compute_input_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_var = x.detach().clone().requires_grad_(True)
    pred = model(x_var)
    loss = F.mse_loss(pred, y)
    loss.backward()
    return loss.detach(), x_var.grad.detach().clone()


def fgsm_maximize_mse(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    price_epsilon: float,
    volume_epsilon: float,
    price_floor: float = 1e-6,
    volume_floor: float = 1.0,
) -> torch.Tensor:
    _, grad = compute_input_gradients(model=model, x=x, y=y)
    budget = relative_budget(
        x,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
    )
    adv = x + budget * grad.sign()
    return project_relative_box(
        adv,
        x,
        price_epsilon=price_epsilon,
        volume_epsilon=volume_epsilon,
        price_floor=price_floor,
        volume_floor=volume_floor,
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
        _, grad = compute_input_gradients(model=model, x=adv, y=y)
        adv = adv + step * grad.sign()
        adv = project_relative_box(
            adv,
            x,
            price_epsilon=price_epsilon,
            volume_epsilon=volume_epsilon,
            price_floor=price_floor,
            volume_floor=volume_floor,
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
        if hasattr(model, "forward_from_features"):
            pred = model.forward_from_features(features)
        elif hasattr(model, "predictor"):
            pred = model.predictor(features)
        else:
            pred = model(features)

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
    if metrics.clean_grad_mean_abs <= thresholds.min_clean_grad_mean_abs:
        failures.append(
            f"clean_grad_mean_abs={metrics.clean_grad_mean_abs} <= {thresholds.min_clean_grad_mean_abs}"
        )
    if thresholds.min_spearman_to_reference is not None:
        if metrics.spearman_to_reference is None or metrics.spearman_to_reference < thresholds.min_spearman_to_reference:
            failures.append(
                "spearman_to_reference="
                f"{metrics.spearman_to_reference} < {thresholds.min_spearman_to_reference}"
            )
    if (
        thresholds.max_feature_mae_to_reference is not None
        and metrics.feature_mae_to_reference is not None
        and metrics.feature_mae_to_reference > thresholds.max_feature_mae_to_reference
    ):
        failures.append(
            "feature_mae_to_reference="
            f"{metrics.feature_mae_to_reference} > {thresholds.max_feature_mae_to_reference}"
        )
    if (
        thresholds.max_feature_rmse_to_reference is not None
        and metrics.feature_rmse_to_reference is not None
        and metrics.feature_rmse_to_reference > thresholds.max_feature_rmse_to_reference
    ):
        failures.append(
            "feature_rmse_to_reference="
            f"{metrics.feature_rmse_to_reference} > {thresholds.max_feature_rmse_to_reference}"
        )
    if (
        thresholds.max_feature_max_abs_to_reference is not None
        and metrics.feature_max_abs_to_reference is not None
        and metrics.feature_max_abs_to_reference > thresholds.max_feature_max_abs_to_reference
    ):
        failures.append(
            "feature_max_abs_to_reference="
            f"{metrics.feature_max_abs_to_reference} > {thresholds.max_feature_max_abs_to_reference}"
        )
    if failures:
        raise ValueError("clean gate failed: " + "; ".join(failures))
