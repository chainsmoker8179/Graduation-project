import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_lstm_whitebox_attack import (
    CleanGateMetrics,
    CleanGateThresholds,
    compute_input_gradients,
    fgsm_maximize_mse,
    pgd_maximize_mse,
    run_clean_gate,
    validate_clean_gate,
)


class ToyPredictor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))


class ToyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = ToyPredictor()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.forward_features(x))


def _make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.2, 11.2, 9.1, 10.7, 1050.0],
            ],
            [
                [20.0, 21.0, 19.0, 20.5, 2000.0],
                [20.1, 21.2, 19.2, 20.6, 2100.0],
            ],
        ],
        dtype=torch.float32,
    )
    y = torch.zeros(x.size(0), dtype=torch.float32)
    return x, y


def test_clean_input_gradients_are_non_empty_and_finite():
    model = ToyPredictor()
    x, y = _make_batch()

    loss, grad = compute_input_gradients(model=model, x=x, y=y)

    assert loss.item() > 0
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().mean().item() > 0


def test_fgsm_increases_mse_within_relative_budget():
    model = ToyPredictor()
    x, y = _make_batch()

    clean_loss = F.mse_loss(model(x), y)
    adv = fgsm_maximize_mse(
        model=model,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
    )
    adv_loss = F.mse_loss(model(adv), y)

    assert adv_loss.item() > clean_loss.item()
    assert torch.all(adv >= 0)
    assert torch.all((adv[..., :4] - x[..., :4]).abs() <= x[..., :4].abs() * 0.01 + 1e-6)
    assert torch.all((adv[..., 4:] - x[..., 4:]).abs() <= x[..., 4:].abs() * 0.02 + 1e-6)


def test_pgd_increases_mse_within_relative_budget():
    model = ToyPredictor()
    x, y = _make_batch()

    clean_loss = F.mse_loss(model(x), y)
    adv = pgd_maximize_mse(
        model=model,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
        num_steps=4,
        step_size=0.5,
    )
    adv_loss = F.mse_loss(model(adv), y)

    assert adv_loss.item() > clean_loss.item()
    assert torch.all(adv >= 0)
    assert torch.all((adv[..., :4] - x[..., :4]).abs() <= x[..., :4].abs() * 0.01 + 1e-6)
    assert torch.all((adv[..., 4:] - x[..., 4:]).abs() <= x[..., 4:].abs() * 0.02 + 1e-6)


def test_clean_gate_reports_feature_alignment_metrics():
    model = ToyPipeline()
    x, y = _make_batch()
    reference_features = x[..., :2].clone()
    reference_scores = model(x).detach()

    metrics = run_clean_gate(
        model=model,
        x=x,
        y=y,
        reference_scores=reference_scores,
        reference_features=reference_features,
    )

    assert metrics.feature_mae_to_reference == 0.0
    assert metrics.feature_rmse_to_reference == 0.0
    assert metrics.feature_max_abs_to_reference == 0.0


def test_validate_clean_gate_rejects_low_alignment_thresholds():
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
