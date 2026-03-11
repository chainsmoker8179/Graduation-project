"""IdxMax/IdxMin BPDA utilities (module 5)."""

from __future__ import annotations

from typing import Union

import torch


DEFAULT_TAU_IDX = 0.5


def _one_hot(indices: torch.Tensor, num_classes: int, dim: int = -1) -> torch.Tensor:
    """Create one-hot along `dim` for indices tensor."""
    y = torch.nn.functional.one_hot(indices, num_classes=num_classes).to(dtype=torch.float32)
    if dim != -1 and dim != y.ndim - 1:
        perm = list(range(y.ndim))
        perm.insert(dim, perm.pop(-1))
        y = y.permute(perm)
    return y


def hard_idxmax_st_onehot(x: torch.Tensor, tau: float = DEFAULT_TAU_IDX, dim: int = -1):
    """Hard argmax forward, softmax backward (straight-through).

    Returns:
        idx_long: hard argmax index (0-based)
        y_st: straight-through one-hot (hard forward, soft backward)
        y_soft: softmax weights
    """
    y_soft = torch.softmax(x / tau, dim=dim)
    idx_long = torch.argmax(x, dim=dim)
    y_hard = _one_hot(idx_long, num_classes=x.size(dim), dim=dim).to(dtype=x.dtype, device=x.device)
    if y_soft.shape != y_hard.shape:
        y_soft = y_soft.expand_as(y_hard)
    y_st = (y_hard - y_soft).detach() + y_soft
    return idx_long, y_st, y_soft


def hard_idxmin_st_onehot(x: torch.Tensor, tau: float = DEFAULT_TAU_IDX, dim: int = -1):
    """Hard argmin forward, softmax backward (straight-through)."""
    idx_long, y_st, y_soft = hard_idxmax_st_onehot(-x, tau=tau, dim=dim)
    return idx_long, y_st, y_soft


def bpda_idxmax(x: torch.Tensor, tau: float = DEFAULT_TAU_IDX, dim: int = -1) -> torch.Tensor:
    """BPDA idxmax returning 1-based index (float)."""
    idx_long, y_st, _ = hard_idxmax_st_onehot(x, tau=tau, dim=dim)
    # forward: hard index (1-based)
    hard = idx_long.to(dtype=x.dtype) + 1.0
    # backward: use soft weights to build a differentiable index
    idx_soft = (y_st * torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)).sum(dim=dim)
    return (hard - idx_soft).detach() + idx_soft


def bpda_idxmin(x: torch.Tensor, tau: float = DEFAULT_TAU_IDX, dim: int = -1) -> torch.Tensor:
    """BPDA idxmin returning 1-based index (float)."""
    idx_long, y_st, _ = hard_idxmin_st_onehot(x, tau=tau, dim=dim)
    hard = idx_long.to(dtype=x.dtype) + 1.0
    idx_soft = (y_st * torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)).sum(dim=dim)
    return (hard - idx_soft).detach() + idx_soft


__all__ = [
    "DEFAULT_TAU_IDX",
    "hard_idxmax_st_onehot",
    "hard_idxmin_st_onehot",
    "bpda_idxmax",
    "bpda_idxmin",
]
