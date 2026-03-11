"""Rolling regression-based operators: Slope, Rsquare, Resi, Corr (module 6)."""

from __future__ import annotations

import torch

from alpha158_rolling import rolling_unfold


_EPS = 1e-12


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return num / (den + eps)


def rolling_slope(x: torch.Tensor, window: int, dim: int = 1) -> torch.Tensor:
    """Rolling slope vs time index (1..N)."""
    xu = rolling_unfold(x, window, dim=dim)  # (B, L-N+1, N)
    N = xu.size(-1)
    device = xu.device
    dtype = xu.dtype
    t = torch.arange(1, N + 1, device=device, dtype=dtype)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    x_mean = xu.mean(dim=-1, keepdim=True)
    cov = ((t - t_mean) * (xu - x_mean)).sum(dim=-1)
    slope = _safe_div(cov, t_var)
    return slope


def rolling_rsquare(x: torch.Tensor, window: int, dim: int = 1, eps: float = _EPS) -> torch.Tensor:
    """Rolling R^2 for regression of x on time index."""
    xu = rolling_unfold(x, window, dim=dim)
    N = xu.size(-1)
    device = xu.device
    dtype = xu.dtype
    t = torch.arange(1, N + 1, device=device, dtype=dtype)
    t_mean = t.mean()

    x_mean = xu.mean(dim=-1, keepdim=True)
    cov = ((t - t_mean) * (xu - x_mean)).sum(dim=-1)
    var_t = ((t - t_mean) ** 2).sum()
    var_x = ((xu - x_mean) ** 2).sum(dim=-1)

    r2 = (cov ** 2) / (var_t * (var_x + eps))
    return r2


def rolling_resi(x: torch.Tensor, window: int, dim: int = 1, eps: float = _EPS) -> torch.Tensor:
    """Rolling regression residual at last point (x_t - (a + b*t))."""
    xu = rolling_unfold(x, window, dim=dim)
    N = xu.size(-1)
    device = xu.device
    dtype = xu.dtype
    t = torch.arange(1, N + 1, device=device, dtype=dtype)
    t_mean = t.mean()
    var_t = ((t - t_mean) ** 2).sum()

    x_mean = xu.mean(dim=-1, keepdim=True)
    cov = ((t - t_mean) * (xu - x_mean)).sum(dim=-1)
    slope = _safe_div(cov, var_t)
    intercept = x_mean.squeeze(-1) - slope * t_mean

    x_last = xu[..., -1]
    y_hat_last = intercept + slope * t[-1]
    resi = x_last - y_hat_last
    return resi


def rolling_corr(x: torch.Tensor, y: torch.Tensor, window: int, dim: int = 1, eps: float = _EPS) -> torch.Tensor:
    """Rolling correlation of x and y."""
    xu = rolling_unfold(x, window, dim=dim)
    yu = rolling_unfold(y, window, dim=dim)
    x_mean = xu.mean(dim=-1, keepdim=True)
    y_mean = yu.mean(dim=-1, keepdim=True)

    cov = ((xu - x_mean) * (yu - y_mean)).sum(dim=-1)
    var_x = ((xu - x_mean) ** 2).sum(dim=-1)
    var_y = ((yu - y_mean) ** 2).sum(dim=-1)

    corr = cov / (torch.sqrt(var_x + eps) * torch.sqrt(var_y + eps))
    return corr


__all__ = [
    "rolling_slope",
    "rolling_rsquare",
    "rolling_resi",
    "rolling_corr",
]
