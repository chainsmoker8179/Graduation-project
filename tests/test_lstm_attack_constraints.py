from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_attack_core import (
    compute_candle_penalty,
    compute_constrained_attack_objective,
    compute_return_penalty,
    compute_volume_penalty,
    project_financial_feasible_box,
)


class ToyPredictor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))


def test_project_financial_feasible_box_enforces_kline_constraints() -> None:
    x_clean = torch.tensor([[[10.0, 11.0, 9.0, 10.5, 1000.0]]], dtype=torch.float32)
    x_adv = torch.tensor([[[10.0, 9.0, 12.0, 10.5, -5.0]]], dtype=torch.float32)

    projected = project_financial_feasible_box(
        x_adv,
        x_clean,
        price_epsilon=0.05,
        volume_epsilon=0.1,
        price_floor=1e-6,
        volume_floor=0.0,
    )

    open_, high_, low_, close_, volume_ = projected[0, 0]
    assert high_ >= max(open_, close_)
    assert low_ <= min(open_, close_)
    assert low_ <= high_
    assert volume_ >= 0


def test_project_financial_feasible_box_preserves_relative_budget() -> None:
    x_clean = torch.tensor([[[10.0, 11.0, 9.0, 10.5, 1000.0]]], dtype=torch.float32)
    x_adv = torch.tensor([[[100.0, 100.0, 0.0, 100.0, 5000.0]]], dtype=torch.float32)

    projected = project_financial_feasible_box(
        x_adv,
        x_clean,
        price_epsilon=0.05,
        volume_epsilon=0.1,
        price_floor=1e-6,
        volume_floor=1.0,
    )

    assert torch.all((projected[..., :4] - x_clean[..., :4]).abs() <= x_clean[..., :4].abs() * 0.05 + 1e-6)
    assert torch.all(
        (projected[..., 4:] - x_clean[..., 4:]).abs()
        <= torch.maximum(x_clean[..., 4:].abs(), torch.ones_like(x_clean[..., 4:])) * 0.1 + 1e-6
    )


def test_return_penalty_is_zero_on_clean_input() -> None:
    x = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.1, 1000.0],
                [10.2, 11.2, 9.2, 10.3, 1100.0],
                [10.4, 11.4, 9.4, 10.5, 1200.0],
            ]
        ],
        dtype=torch.float32,
    )

    penalty = compute_return_penalty(x, x, tau_ret=0.005)

    assert torch.allclose(penalty, torch.tensor(0.0))


def test_return_penalty_stays_zero_within_tolerance() -> None:
    x_clean = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.1, 1000.0],
                [10.2, 11.2, 9.2, 10.3, 1100.0],
                [10.4, 11.4, 9.4, 10.5, 1200.0],
            ]
        ],
        dtype=torch.float32,
    )
    x_adv = x_clean.clone()
    x_adv[0, 2, 3] = 10.505

    penalty = compute_return_penalty(x_adv, x_clean, tau_ret=0.005)

    assert torch.allclose(penalty, torch.tensor(0.0))


def test_candle_penalty_is_zero_on_clean_input() -> None:
    x = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.2, 11.3, 9.3, 10.7, 1100.0],
            ]
        ],
        dtype=torch.float32,
    )

    penalty = compute_candle_penalty(x, x, tau_body=0.005, tau_range=0.01)

    assert torch.allclose(penalty, torch.tensor(0.0))


def test_candle_penalty_stays_zero_within_tolerance() -> None:
    x_clean = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.2, 11.3, 9.3, 10.7, 1100.0],
            ]
        ],
        dtype=torch.float32,
    )
    x_adv = x_clean.clone()
    x_adv[0, 0, 1] = 11.05
    x_adv[0, 0, 2] = 8.99
    x_adv[0, 0, 3] = 10.53

    penalty = compute_candle_penalty(x_adv, x_clean, tau_body=0.005, tau_range=0.01)

    assert torch.allclose(penalty, torch.tensor(0.0))


def test_volume_penalty_is_zero_on_clean_input() -> None:
    x = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.2, 11.3, 9.3, 10.7, 1100.0],
                [10.3, 11.4, 9.4, 10.8, 1200.0],
            ]
        ],
        dtype=torch.float32,
    )

    penalty = compute_volume_penalty(x, x, tau_vol=0.05)

    assert torch.allclose(penalty, torch.tensor(0.0))


def test_volume_penalty_stays_zero_within_tolerance() -> None:
    x_clean = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.2, 11.3, 9.3, 10.7, 1100.0],
                [10.3, 11.4, 9.4, 10.8, 1200.0],
            ]
        ],
        dtype=torch.float32,
    )
    x_adv = x_clean.clone()
    x_adv[0, 2, 4] = 1210.0

    penalty = compute_volume_penalty(x_adv, x_clean, tau_vol=0.05)

    assert torch.allclose(penalty, torch.tensor(0.0))


def test_constrained_objective_matches_mse_on_clean_input() -> None:
    model = ToyPredictor()
    x = torch.tensor(
        [
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.2, 11.3, 9.3, 10.7, 1100.0],
                [10.3, 11.4, 9.4, 10.8, 1200.0],
            ]
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0.0], dtype=torch.float32)

    result = compute_constrained_attack_objective(
        model=model,
        x_adv=x,
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

    assert torch.allclose(result.ret_penalty, torch.tensor(0.0))
    assert torch.allclose(result.candle_penalty, torch.tensor(0.0))
    assert torch.allclose(result.vol_penalty, torch.tensor(0.0))
    assert torch.allclose(result.objective, result.mse_loss)
