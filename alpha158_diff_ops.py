"""Differentiable operator approximations with BPDA helpers.

Module 1: element-wise max/min and comparison approximations.
"""

from __future__ import annotations

from typing import Union

import torch


DEFAULT_TAU_MAXMIN = 5.0
DEFAULT_TEMP_CMP = 0.2
_EPS = 1e-6


def _param_like(x: torch.Tensor, p: Union[float, torch.Tensor]) -> torch.Tensor:
    """Return p as a tensor on x's device/dtype, clamped for stability."""
    pt = torch.as_tensor(p, dtype=x.dtype, device=x.device)
    return torch.clamp(pt, min=_EPS)


def bpda(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Straight-through BPDA: hard forward, soft backward."""
    return (hard - soft).detach() + soft


def smooth_max_pair(a: torch.Tensor, b: torch.Tensor, tau: Union[float, torch.Tensor] = DEFAULT_TAU_MAXMIN) -> torch.Tensor:
    """Differentiable approximation of element-wise max(a,b) using log-sum-exp."""
    tau_t = _param_like(a, tau)
    a, b = torch.broadcast_tensors(a, b)
    stacked = torch.stack([a, b], dim=-1)
    return torch.logsumexp(stacked * tau_t, dim=-1) / tau_t


def smooth_min_pair(a: torch.Tensor, b: torch.Tensor, tau: Union[float, torch.Tensor] = DEFAULT_TAU_MAXMIN) -> torch.Tensor:
    """Differentiable approximation of element-wise min(a,b)."""
    return -smooth_max_pair(-a, -b, tau=tau)


def bpda_max_pair(a: torch.Tensor, b: torch.Tensor, tau: Union[float, torch.Tensor] = DEFAULT_TAU_MAXMIN) -> torch.Tensor:
    """BPDA version of element-wise max: hard forward, smooth backward."""
    hard = torch.maximum(a, b)
    soft = smooth_max_pair(a, b, tau=tau)
    return bpda(hard, soft)


def bpda_min_pair(a: torch.Tensor, b: torch.Tensor, tau: Union[float, torch.Tensor] = DEFAULT_TAU_MAXMIN) -> torch.Tensor:
    """BPDA version of element-wise min: hard forward, smooth backward."""
    hard = torch.minimum(a, b)
    soft = smooth_min_pair(a, b, tau=tau)
    return bpda(hard, soft)


def soft_greater(a: torch.Tensor, b: torch.Tensor, temperature: Union[float, torch.Tensor] = DEFAULT_TEMP_CMP) -> torch.Tensor:
    """Differentiable approximation of I(a > b) using sigmoid((a-b)/T)."""
    t = _param_like(a, temperature)
    return torch.sigmoid((a - b) / t)


def soft_less(a: torch.Tensor, b: torch.Tensor, temperature: Union[float, torch.Tensor] = DEFAULT_TEMP_CMP) -> torch.Tensor:
    """Differentiable approximation of I(a < b)."""
    t = _param_like(a, temperature)
    return torch.sigmoid((b - a) / t)


def bpda_greater(a: torch.Tensor, b: torch.Tensor, temperature: Union[float, torch.Tensor] = DEFAULT_TEMP_CMP) -> torch.Tensor:
    """BPDA version of I(a > b): hard forward, sigmoid backward."""
    hard = (a > b).to(dtype=a.dtype)
    soft = soft_greater(a, b, temperature=temperature)
    return bpda(hard, soft)


def bpda_less(a: torch.Tensor, b: torch.Tensor, temperature: Union[float, torch.Tensor] = DEFAULT_TEMP_CMP) -> torch.Tensor:
    """BPDA version of I(a < b): hard forward, sigmoid backward."""
    hard = (a < b).to(dtype=a.dtype)
    soft = soft_less(a, b, temperature=temperature)
    return bpda(hard, soft)


__all__ = [
    "DEFAULT_TAU_MAXMIN",
    "DEFAULT_TEMP_CMP",
    "bpda",
    "smooth_max_pair",
    "smooth_min_pair",
    "bpda_max_pair",
    "bpda_min_pair",
    "soft_greater",
    "soft_less",
    "bpda_greater",
    "bpda_less",
]
