"""Rolling/window utilities for alpha158 (module 2)."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import torch


def rolling_unfold(x: torch.Tensor, window: int, dim: int = 1) -> torch.Tensor:
    """Return a sliding window view using unfold.

    Args:
        x: input tensor.
        window: window size (>0).
        dim: dimension to roll over.

    Returns:
        Tensor with an extra last dimension of size `window`, length reduced by window-1.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if x.size(dim) < window:
        raise ValueError(f"window ({window}) is larger than length ({x.size(dim)})")
    return x.unfold(dim, size=window, step=1)


def rolling_apply(x: torch.Tensor, window: int, reduce_fn: Callable[[torch.Tensor], torch.Tensor], dim: int = 1) -> torch.Tensor:
    """Apply reduce_fn over rolling windows."""
    xu = rolling_unfold(x, window, dim=dim)
    return reduce_fn(xu)


def rolling_sum(x: torch.Tensor, window: int, dim: int = 1) -> torch.Tensor:
    return rolling_apply(x, window, lambda t: t.sum(dim=-1), dim=dim)


def rolling_mean(x: torch.Tensor, window: int, dim: int = 1) -> torch.Tensor:
    return rolling_apply(x, window, lambda t: t.mean(dim=-1), dim=dim)


def rolling_std(x: torch.Tensor, window: int, dim: int = 1, unbiased: bool = False) -> torch.Tensor:
    return rolling_apply(x, window, lambda t: t.std(dim=-1, unbiased=unbiased), dim=dim)


def rolling_var(x: torch.Tensor, window: int, dim: int = 1, unbiased: bool = False) -> torch.Tensor:
    return rolling_apply(x, window, lambda t: t.var(dim=-1, unbiased=unbiased), dim=dim)


def rolling_max(x: torch.Tensor, window: int, dim: int = 1) -> torch.Tensor:
    return rolling_apply(x, window, lambda t: t.max(dim=-1).values, dim=dim)


def rolling_min(x: torch.Tensor, window: int, dim: int = 1) -> torch.Tensor:
    return rolling_apply(x, window, lambda t: t.min(dim=-1).values, dim=dim)


def right_align(*tensors: torch.Tensor, dim: int = 1) -> List[torch.Tensor]:
    """Right-align tensors by trimming the leading part along `dim` to the shortest length."""
    if not tensors:
        return []
    lengths = [t.size(dim) for t in tensors]
    min_len = min(lengths)
    if min_len <= 0:
        raise ValueError("min length must be > 0")
    out = []
    for t in tensors:
        slices = [slice(None)] * t.ndim
        slices[dim] = slice(t.size(dim) - min_len, t.size(dim))
        out.append(t[tuple(slices)])
    return out


def align_to_length(x: torch.Tensor, length: int, dim: int = 1) -> torch.Tensor:
    """Trim tensor along `dim` to the last `length` elements."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if x.size(dim) < length:
        raise ValueError("length larger than tensor size")
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(x.size(dim) - length, x.size(dim))
    return x[tuple(slices)]


__all__ = [
    "rolling_unfold",
    "rolling_apply",
    "rolling_sum",
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "rolling_max",
    "rolling_min",
    "right_align",
    "align_to_length",
]
