import torch

from alpha158_regression import rolling_corr, rolling_rsquare


def test_rolling_rsquare_returns_nan_for_constant_window() -> None:
    x = torch.full((1, 5), 1.35, dtype=torch.float32)

    y = rolling_rsquare(x, window=5, dim=1)

    assert y.shape == (1, 1)
    assert torch.isnan(y).all()


def test_rolling_corr_returns_nan_when_one_side_has_zero_variance() -> None:
    x = torch.full((1, 5), 1.35, dtype=torch.float32)
    y = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]], dtype=torch.float32)

    out = rolling_corr(x, y, window=5, dim=1)

    assert out.shape == (1, 1)
    assert torch.isnan(out).all()


def test_rolling_rsquare_keeps_gradients_finite_when_nan_output_is_masked() -> None:
    x = torch.full((1, 5), 1.35, dtype=torch.float32, requires_grad=True)

    out = rolling_rsquare(x, window=5, dim=1)
    masked = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    masked.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rolling_corr_keeps_gradients_finite_when_nan_output_is_masked() -> None:
    x = torch.full((1, 5), 1.35, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0]], dtype=torch.float32)

    out = rolling_corr(x, y, window=5, dim=1)
    masked = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    masked.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
