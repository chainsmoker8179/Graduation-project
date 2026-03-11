"""NeuralSort-based soft sort and soft rank (GPU-friendly, no external deps)."""

from __future__ import annotations

from typing import Literal

import torch


Direction = Literal["ASCENDING", "DESCENDING"]


def _check_input(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"values must be 2D (B, N), got shape {tuple(values.shape)}")
    return values


def soft_permutation_matrix(
    values: torch.Tensor,
    tau: float = 1.0,
    direction: Direction = "ASCENDING",
) -> torch.Tensor:
    """Compute NeuralSort soft permutation matrix.

    Args:
        values: (B, N)
        tau: temperature (>0). smaller -> closer to hard sort.
        direction: ASCENDING or DESCENDING

    Returns:
        P: (B, N, N) soft permutation matrix, rows correspond to sorted positions (1..N).
    """
    x = _check_input(values)
    if tau <= 0:
        raise ValueError("tau must be > 0")

    if direction == "ASCENDING":
        s = -x
    elif direction == "DESCENDING":
        s = x
    else:
        raise ValueError("direction must be 'ASCENDING' or 'DESCENDING'")

    B, N = s.shape
    device = s.device
    dtype = s.dtype

    # Pairwise absolute differences |s_i - s_j|
    diff = torch.abs(s.unsqueeze(-1) - s.unsqueeze(-2))  # (B, N, N)
    # Row sums
    Bsum = diff.sum(dim=-1)  # (B, N)

    # Construct scores for each position k (1..N)
    k = torch.arange(1, N + 1, device=device, dtype=dtype)  # (N,)
    # (B, N, N): for each k (row), score over elements i
    scores = (N + 1 - 2 * k).view(1, N, 1) * s.unsqueeze(1) - Bsum.unsqueeze(1)
    P = torch.softmax(scores / tau, dim=-1)
    return P


def soft_sort(
    values: torch.Tensor,
    tau: float = 1.0,
    direction: Direction = "ASCENDING",
) -> torch.Tensor:
    """Soft sort values along last dimension.

    Returns:
        soft sorted values of shape (B, N)
    """
    P = soft_permutation_matrix(values, tau=tau, direction=direction)
    return torch.bmm(P, values.unsqueeze(-1)).squeeze(-1)


def soft_rank(
    values: torch.Tensor,
    tau: float = 1.0,
    direction: Direction = "ASCENDING",
    pct: bool = False,
) -> torch.Tensor:
    """Soft rank values using NeuralSort permutation.

    Args:
        pct: if True, return percentile rank in [0,1].
    """
    x = _check_input(values)
    P = soft_permutation_matrix(x, tau=tau, direction=direction)
    B, N = x.shape
    device = x.device
    dtype = x.dtype
    # position index k for each sorted row (1..N)
    k = torch.arange(1, N + 1, device=device, dtype=dtype).view(1, 1, N)  # (1, 1, N)
    # rank of element i is sum_k P_{k,i} * k
    rank = torch.sum(P.transpose(1, 2) * k, dim=-1)  # (B, N)
    if pct:
        if N == 1:
            return torch.zeros_like(rank)
        rank = (rank - 1.0) / (N - 1.0)
    return rank


__all__ = [
    "soft_permutation_matrix",
    "soft_sort",
    "soft_rank",
]
