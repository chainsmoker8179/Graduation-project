import torch
import torch.nn as nn
import torch.nn.functional as F

from legacy_lstm_attack_core import compute_constrained_attack_objective
from scripts.run_lstm_whitebox_attack import (
    CleanGateMetrics,
    CleanGateThresholds,
    compute_input_gradients,
    fgsm_maximize_mse,
    parse_args,
    pgd_maximize_mse,
    resolve_model_config_path,
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


def test_physical_fgsm_preserves_kline_constraints():
    model = ToyPredictor()
    x, y = _make_batch()

    adv = fgsm_maximize_mse(
        model=model,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
        constraint_mode="physical",
    )

    assert torch.all(adv[..., 1] >= torch.maximum(adv[..., 0], adv[..., 3]))
    assert torch.all(adv[..., 2] <= torch.minimum(adv[..., 0], adv[..., 3]))
    assert torch.all(adv[..., 2] <= adv[..., 1])
    assert torch.all((adv[..., :4] - x[..., :4]).abs() <= x[..., :4].abs() * 0.01 + 1e-6)
    assert torch.all((adv[..., 4:] - x[..., 4:]).abs() <= x[..., 4:].abs() * 0.02 + 1e-6)


def test_physical_stat_attack_preserves_constraints_and_changes_loss():
    model = ToyPredictor()
    x, y = _make_batch()

    x_var = x.clone().requires_grad_(True)
    objective = compute_constrained_attack_objective(
        model=model,
        x_adv=x_var,
        y=y,
        x_clean=x,
        tau_ret=0.005,
        tau_body=0.005,
        tau_range=0.01,
        tau_vol=0.05,
        lambda_ret=1.0,
        lambda_candle=0.5,
        lambda_vol=0.3,
    )
    objective.objective.backward()
    assert x_var.grad is not None
    assert x_var.grad.abs().mean().item() > 0

    clean_loss = F.mse_loss(model(x), y)
    adv = fgsm_maximize_mse(
        model=model,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
        constraint_mode="physical_stat",
        tau_ret=0.005,
        tau_body=0.005,
        tau_range=0.01,
        tau_vol=0.05,
        lambda_ret=1.0,
        lambda_candle=0.5,
        lambda_vol=0.3,
    )
    adv_loss = F.mse_loss(model(adv), y)

    assert adv.shape == x.shape
    assert adv_loss.item() > clean_loss.item()
    assert torch.all(adv[..., 1] >= torch.maximum(adv[..., 0], adv[..., 3]))
    assert torch.all(adv[..., 2] <= torch.minimum(adv[..., 0], adv[..., 3]))
    assert torch.all(adv[..., 2] <= adv[..., 1])


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


def test_parse_args_accepts_constraint_mode_and_penalty_hparams():
    args = parse_args(
        [
            "--constraint-mode",
            "physical_stat",
            "--tau-ret",
            "0.006",
            "--tau-body",
            "0.007",
            "--tau-range",
            "0.02",
            "--tau-vol",
            "0.06",
            "--lambda-ret",
            "1.1",
            "--lambda-candle",
            "0.6",
            "--lambda-vol",
            "0.4",
        ]
    )

    assert args.constraint_mode == "physical_stat"
    assert args.tau_ret == 0.006
    assert args.tau_body == 0.007
    assert args.tau_range == 0.02
    assert args.tau_vol == 0.06
    assert args.lambda_ret == 1.1
    assert args.lambda_candle == 0.6
    assert args.lambda_vol == 0.4


def test_parse_args_uses_tuned_default_penalty_weights():
    args = parse_args([])

    assert args.lambda_ret == 0.8
    assert args.lambda_candle == 0.4
    assert args.lambda_vol == 0.3


def test_resolve_model_config_path_falls_back_to_state_dict_sibling(tmp_path):
    state_dict_path = tmp_path / "lstm_state_dict.pt"
    sibling_config_path = tmp_path / "model_config.json"
    state_dict_path.write_bytes(b"placeholder")
    sibling_config_path.write_text("{}", encoding="utf-8")

    resolved = resolve_model_config_path(
        config_path=tmp_path / "missing_config.json",
        state_dict_path=state_dict_path,
    )

    assert resolved == sibling_config_path
