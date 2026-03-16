from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_attack_core import (
    CleanGateThresholds as NewCleanGateThresholds,
    fgsm_maximize_mse as new_fgsm_maximize_mse,
    pgd_maximize_mse as new_pgd_maximize_mse,
    run_clean_gate as new_run_clean_gate,
    validate_clean_gate as new_validate_clean_gate,
)
from scripts.run_lstm_whitebox_attack import (
    CleanGateThresholds as OldCleanGateThresholds,
    fgsm_maximize_mse as old_fgsm_maximize_mse,
    pgd_maximize_mse as old_pgd_maximize_mse,
    run_clean_gate as old_run_clean_gate,
    validate_clean_gate as old_validate_clean_gate,
)


class ToyPredictor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))


class ToyPipeline(nn.Module):
    def __init__(self) -> None:
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


def test_fgsm_matches_existing_attack_core() -> None:
    model = ToyPredictor()
    x, y = _make_batch()

    old_adv = old_fgsm_maximize_mse(model=model, x=x, y=y, price_epsilon=0.01, volume_epsilon=0.02)
    new_adv = new_fgsm_maximize_mse(model=model, x=x, y=y, price_epsilon=0.01, volume_epsilon=0.02)

    assert torch.allclose(new_adv, old_adv)


def test_pgd_matches_existing_attack_core() -> None:
    model = ToyPredictor()
    x, y = _make_batch()

    old_adv = old_pgd_maximize_mse(
        model=model,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
        num_steps=4,
        step_size=0.5,
    )
    new_adv = new_pgd_maximize_mse(
        model=model,
        x=x,
        y=y,
        price_epsilon=0.01,
        volume_epsilon=0.02,
        num_steps=4,
        step_size=0.5,
    )

    assert torch.allclose(new_adv, old_adv)


def test_clean_gate_and_validation_match_existing_attack_core() -> None:
    model = ToyPipeline()
    x, y = _make_batch()
    reference_features = x[..., :2].clone()
    reference_scores = model(x).detach()

    old_metrics = old_run_clean_gate(
        model=model,
        x=x,
        y=y,
        reference_scores=reference_scores,
        reference_features=reference_features,
    )
    new_metrics = new_run_clean_gate(
        model=model,
        x=x,
        y=y,
        reference_scores=reference_scores,
        reference_features=reference_features,
    )
    assert new_metrics == old_metrics

    old_thresholds = OldCleanGateThresholds()
    new_thresholds = NewCleanGateThresholds()
    old_validate_clean_gate(old_metrics, old_thresholds)
    new_validate_clean_gate(new_metrics, new_thresholds)


def test_validate_clean_gate_skips_feature_thresholds_without_reference_features() -> None:
    metrics = new_run_clean_gate(
        model=ToyPipeline(),
        x=_make_batch()[0],
        y=_make_batch()[1],
        reference_scores=ToyPipeline()(_make_batch()[0]).detach(),
        reference_features=None,
    )

    thresholds = NewCleanGateThresholds(
        min_clean_grad_mean_abs=1e-6,
        min_spearman_to_reference=0.0,
        max_feature_mae_to_reference=0.05,
        max_feature_rmse_to_reference=0.12,
        max_feature_max_abs_to_reference=0.7,
    )

    new_validate_clean_gate(metrics, thresholds)
