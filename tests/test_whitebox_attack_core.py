import torch
import torch.nn as nn

from whitebox_attack_core import (
    CleanGateMetrics,
    CleanGateThresholds,
    RawFeatureAttackPipeline,
    fgsm_maximize_mse,
    project_relative_box,
    run_clean_gate,
    validate_clean_gate,
)


class DummyBridge(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2]


class DummyNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyFillna(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyAdapter(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, -1, :].sum(dim=-1)


def test_pipeline_backward_reaches_raw_ohlcv() -> None:
    pipe = RawFeatureAttackPipeline(
        bridge=DummyBridge(),
        norm=DummyNorm(),
        fillna=DummyFillna(),
        model=DummyAdapter(),
    )
    x = torch.randn(4, 20, 5, requires_grad=True)
    y = pipe(x).sum()
    y.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_project_relative_box_keeps_values_within_budget() -> None:
    x_clean = torch.tensor(
        [[[10.0, 12.0, 9.0, 11.0, 1000.0], [11.0, 13.0, 10.0, 12.0, 1100.0]]],
        dtype=torch.float32,
    )
    x_adv = x_clean * 10.0

    projected = project_relative_box(
        x_adv,
        x_clean,
        price_epsilon=0.01,
        volume_epsilon=0.02,
    )

    assert torch.all(projected >= 0)
    assert torch.all((projected[..., :4] - x_clean[..., :4]).abs() <= x_clean[..., :4].abs() * 0.01 + 1e-6)
    assert torch.all((projected[..., 4:] - x_clean[..., 4:]).abs() <= x_clean[..., 4:].abs() * 0.02 + 1e-6)


def test_fgsm_uses_shared_pipeline_and_preserves_shape() -> None:
    pipe = RawFeatureAttackPipeline(
        bridge=DummyBridge(),
        norm=DummyNorm(),
        fillna=DummyFillna(),
        model=DummyAdapter(),
    )
    x = torch.randn(4, 20, 5)
    y = torch.zeros(4)

    adv = fgsm_maximize_mse(
        model=pipe,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
    )

    assert adv.shape == x.shape


def test_clean_gate_reports_feature_alignment_metrics() -> None:
    pipe = RawFeatureAttackPipeline(
        bridge=DummyBridge(),
        norm=DummyNorm(),
        fillna=DummyFillna(),
        model=DummyAdapter(),
    )
    x = torch.randn(4, 20, 5)
    y = torch.zeros(4)
    reference_features = pipe.forward_features(x).detach()
    reference_scores = pipe(x).detach()

    metrics = run_clean_gate(
        model=pipe,
        x=x,
        y=y,
        reference_scores=reference_scores,
        reference_features=reference_features,
    )

    assert isinstance(metrics, CleanGateMetrics)
    assert metrics.feature_mae_to_reference == 0.0
    assert metrics.feature_rmse_to_reference == 0.0
    assert metrics.feature_max_abs_to_reference == 0.0


def test_validate_clean_gate_rejects_low_alignment_thresholds() -> None:
    metrics = CleanGateMetrics(
        clean_loss=0.1,
        clean_grad_mean_abs=1e-4,
        clean_grad_max_abs=1e-3,
        clean_grad_finite_rate=1.0,
        feature_finite_rate=1.0,
        clean_pred_mean=0.0,
        clean_pred_std=1.0,
        reference_score_mean=0.0,
        reference_score_std=1.0,
        spearman_to_reference=0.1,
        feature_mae_to_reference=0.2,
        feature_rmse_to_reference=0.3,
        feature_max_abs_to_reference=0.4,
    )
    thresholds = CleanGateThresholds(
        min_clean_grad_mean_abs=1e-6,
        min_spearman_to_reference=0.2,
        max_feature_mae_to_reference=0.1,
        max_feature_rmse_to_reference=0.2,
        max_feature_max_abs_to_reference=0.3,
    )

    try:
        validate_clean_gate(metrics, thresholds)
    except ValueError as exc:
        message = str(exc)
        assert "spearman" in message.lower()
        assert "feature_mae" in message.lower()
    else:
        raise AssertionError("validate_clean_gate should reject low alignment metrics")
