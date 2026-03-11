"""Execution helpers for Alpha158 template graphs (module 8)."""

from __future__ import annotations

from typing import Dict, Any

import torch

from alpha158_diff_ops import (
    bpda_max_pair,
    bpda_min_pair,
    bpda_greater,
    bpda_less,
    DEFAULT_TAU_MAXMIN,
)
from alpha158_rolling import (
    rolling_unfold,
    rolling_sum,
    rolling_mean,
    rolling_std,
    right_align,
    align_to_length,
)
from alpha158_softsort import soft_rank
from alpha158_quantile import bpda_quantile_window
from alpha158_idx import bpda_idxmax, bpda_idxmin, DEFAULT_TAU_IDX
from alpha158_regression import rolling_slope, rolling_rsquare, rolling_resi, rolling_corr


_EPS = 1e-12
_RANK_TAU = 1.0


def _align_binary(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.ndim >= 2 and b.ndim >= 2:
            a, b = right_align(a, b, dim=1)
    return a, b


def _to_int(x):
    if torch.is_tensor(x):
        x = x.item()
    return int(x)


def _smooth_max_window(xu: torch.Tensor, tau: float = DEFAULT_TAU_MAXMIN) -> torch.Tensor:
    return torch.logsumexp(xu * tau, dim=-1) / tau


def _smooth_min_window(xu: torch.Tensor, tau: float = DEFAULT_TAU_MAXMIN) -> torch.Tensor:
    return -_smooth_max_window(-xu, tau=tau)


def op_Add(a, b):
    a, b = _align_binary(a, b)
    return a + b


def op_Sub(a, b):
    a, b = _align_binary(a, b)
    return a - b


def op_Mul(a, b):
    a, b = _align_binary(a, b)
    return a * b


def op_Div(a, b):
    a, b = _align_binary(a, b)
    return a / (b + 0.0)


def op_Pow(a, b):
    a, b = _align_binary(a, b)
    return torch.pow(a, b)


def op_Abs(a):
    return torch.abs(a)


def op_Log(a):
    return torch.log(a)


def op_Greater(a, b):
    a, b = _align_binary(a, b)
    return bpda_max_pair(a, b)


def op_Less(a, b):
    a, b = _align_binary(a, b)
    return bpda_min_pair(a, b)


def op_Gt(a, b):
    a, b = _align_binary(a, b)
    return bpda_greater(a, b)


def op_Lt(a, b):
    a, b = _align_binary(a, b)
    return bpda_less(a, b)


def op_Mean(a, N):
    N = _to_int(N)
    return rolling_mean(a, N, dim=1)


def op_Sum(a, N):
    N = _to_int(N)
    return rolling_sum(a, N, dim=1)


def op_Std(a, N):
    N = _to_int(N)
    return rolling_std(a, N, dim=1, unbiased=False)


def op_Max(a, N):
    N = _to_int(N)
    xu = rolling_unfold(a, N, dim=1)
    hard = xu.max(dim=-1).values
    soft = _smooth_max_window(xu)
    return (hard - soft).detach() + soft


def op_Min(a, N):
    N = _to_int(N)
    xu = rolling_unfold(a, N, dim=1)
    hard = xu.min(dim=-1).values
    soft = _smooth_min_window(xu)
    return (hard - soft).detach() + soft


def op_Ref(a, N):
    N = _to_int(N)
    if N <= 0:
        return a[:, :1]
    return a[:, :-N]


def op_Rank(a, N):
    N = _to_int(N)
    xu = rolling_unfold(a, N, dim=1)  # (B, L-N+1, N)
    B, Lp, _ = xu.shape
    last = xu[..., -1]  # (B, Lp)
    less = (xu < last.unsqueeze(-1)).sum(dim=-1)
    equal = (xu == last.unsqueeze(-1)).sum(dim=-1)
    hard_rank = less + (equal + 1) / 2.0
    hard_pct = hard_rank / float(N)

    xu2 = xu.reshape(-1, N)
    soft = soft_rank(xu2, tau=_RANK_TAU, direction="ASCENDING", pct=True)
    soft = soft.reshape(B, Lp, N)[..., -1]

    return (hard_pct - soft).detach() + soft


def op_Quantile(a, N, Q):
    N = _to_int(N)
    q = float(Q)
    xu = rolling_unfold(a, N, dim=1)
    B, Lp, _ = xu.shape
    xu2 = xu.reshape(-1, N)
    out = bpda_quantile_window(xu2, q)
    return out.reshape(B, Lp)


def op_IdxMax(a, N):
    N = _to_int(N)
    xu = rolling_unfold(a, N, dim=1)
    B, Lp, _ = xu.shape
    out = bpda_idxmax(xu.reshape(-1, N), tau=DEFAULT_TAU_IDX)
    return out.reshape(B, Lp)


def op_IdxMin(a, N):
    N = _to_int(N)
    xu = rolling_unfold(a, N, dim=1)
    B, Lp, _ = xu.shape
    out = bpda_idxmin(xu.reshape(-1, N), tau=DEFAULT_TAU_IDX)
    return out.reshape(B, Lp)


def op_Slope(a, N):
    N = _to_int(N)
    return rolling_slope(a, N, dim=1)


def op_Rsquare(a, N):
    N = _to_int(N)
    return rolling_rsquare(a, N, dim=1)


def op_Resi(a, N):
    N = _to_int(N)
    return rolling_resi(a, N, dim=1)


def op_Corr(a, b, N):
    N = _to_int(N)
    a, b = _align_binary(a, b)
    return rolling_corr(a, b, N, dim=1)


def eval_graph(graph, variables: Dict[str, torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
    values: Dict[str, Any] = {}
    for n in graph.nodes:
        op = n["op"]
        nid = n["id"]
        if op == "var":
            values[nid] = variables[n["name"]]
        elif op == "const":
            values[nid] = torch.tensor(n["value"], device=next(iter(variables.values())).device,
                                       dtype=next(iter(variables.values())).dtype)
        elif op == "param":
            values[nid] = params.get(n["name"], None)
            if values[nid] is None:
                raise ValueError(f"Missing param {n['name']}")
        else:
            args = [values[i] for i in n.get("inputs", [])]
            fn = globals().get(f"op_{op}")
            if fn is None:
                raise ValueError(f"Unsupported op: {op}")
            values[nid] = fn(*args)
    return values[graph.output]


__all__ = ["eval_graph"]
