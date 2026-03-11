"""Soft quantile and BPDA quantile utilities (module 4)."""

from __future__ import annotations

from typing import Union

import torch

from alpha158_softsort import soft_sort


DEFAULT_REG_STRENGTH = 0.3
DEFAULT_PICK_STRENGTH = 0.3
_EPS = 1e-6


def soft_quantile_window(
    values_2d: torch.Tensor,
    q: float,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    pick_strength: float = DEFAULT_PICK_STRENGTH,
    eps: float = _EPS,
) -> torch.Tensor:
    """Differentiable quantile for 2D values (B, N) using soft sort + soft pick.

    Args:
        values_2d: (B, N)
        q: quantile in [0,1]
        reg_strength: temperature for soft_sort (tau). smaller -> closer to hard sort.
        pick_strength: temperature for soft pick (lower -> closer to hard index)
    Returns:
        (B,) quantile values in original scale.
    """
    if values_2d.ndim != 2:
        raise ValueError(f"values_2d must be 2D, got {values_2d.shape}")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0,1]")

    x = values_2d
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)
    xz = (x - mu) / (std + eps)

    # soft sort (ascending)
    xs = soft_sort(xz, tau=reg_strength, direction="ASCENDING")  # (B, N)

    B, N = xs.shape
    pos = q * (N - 1)
    k = torch.arange(N, device=xs.device, dtype=xs.dtype)

    # Soft pick around position pos
    logits = -((k - pos) ** 2) / pick_strength
    w = torch.softmax(logits, dim=-1)  # (N,)
    qz = (xs * w.view(1, -1)).sum(dim=1)  # (B,)

    # back to original scale
    return qz * (std.squeeze(1) + eps) + mu.squeeze(1)


def bpda_quantile_window(
    values_2d: torch.Tensor,
    q: float,
    reg_strength: float = DEFAULT_REG_STRENGTH,
    pick_strength: float = DEFAULT_PICK_STRENGTH,
    eps: float = _EPS,
) -> torch.Tensor:
    """BPDA quantile: hard quantile forward, soft quantile backward."""
    # hard forward quantile using torch
    hard = torch.quantile(values_2d, q, dim=-1)
    soft = soft_quantile_window(values_2d, q, reg_strength=reg_strength, pick_strength=pick_strength, eps=eps)
    return (hard - soft).detach() + soft


__all__ = [
    "DEFAULT_REG_STRENGTH",
    "DEFAULT_PICK_STRENGTH",
    "soft_quantile_window",
    "bpda_quantile_window",
]
