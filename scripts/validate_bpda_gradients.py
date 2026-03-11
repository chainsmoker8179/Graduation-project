#!/usr/bin/env python3
"""BPDA 梯度验证套件（完整版）。

本脚本用于验证离散/BPDA 算子的三个核心目标：
1) 前向结果与硬算子语义一致。
2) 反向梯度替代正确（BPDA 梯度应等于软替代算子梯度）。
3) 梯度语义可信（方向一致性与边界敏感性符合预期）。

脚本不依赖 pytest；只要任一检查失败就返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha158_diff_ops import (
    DEFAULT_TAU_MAXMIN,
    DEFAULT_TEMP_CMP,
    bpda_greater,
    bpda_less,
    bpda_max_pair,
    bpda_min_pair,
    smooth_max_pair,
    smooth_min_pair,
    soft_greater,
    soft_less,
)
from alpha158_idx import DEFAULT_TAU_IDX, bpda_idxmax, bpda_idxmin
from alpha158_ops import op_Max, op_Min, op_Rank
from alpha158_quantile import (
    DEFAULT_PICK_STRENGTH,
    DEFAULT_REG_STRENGTH,
    bpda_quantile_window,
    soft_quantile_window,
)
from alpha158_rolling import rolling_unfold
from alpha158_softsort import soft_rank


@dataclass
class CheckResult:
    name: str
    group: str
    passed: bool
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    note: str = ""


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _samplewise_sum(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y
    return y.reshape(y.shape[0], -1).sum(dim=1)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    av = a.reshape(-1).double()
    bv = b.reshape(-1).double()
    denom = av.norm() * bv.norm()
    if float(denom) < eps:
        max_diff = float((av - bv).abs().max())
        return 1.0 if max_diff < 1e-12 else 0.0
    return float(torch.dot(av, bv) / (denom + eps))


def _value_and_grad(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    y = fn(x_req)
    grad = torch.autograd.grad(y.sum(), x_req, create_graph=False, retain_graph=False)[0]
    return y.detach(), grad.detach()


def _finite_diff_grad_subset(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-4,
    max_coords: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = x.detach().reshape(-1)
    total = flat.numel()
    count = min(max_coords, total)
    idx = torch.randperm(total, device=x.device)[:count]
    fd = torch.zeros(count, device=x.device, dtype=x.dtype)
    for k, i in enumerate(idx):
        xp = flat.clone()
        xm = flat.clone()
        xp[i] += eps
        xm[i] -= eps
        yp = fn(xp.reshape_as(x)).sum()
        ym = fn(xm.reshape_as(x)).sum()
        fd[k] = (yp - ym) / (2.0 * eps)
    return idx.detach(), fd.detach()


def _directional_sign_agreement(
    hard_fn: Callable[[torch.Tensor], torch.Tensor],
    bpda_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    delta: float = 1e-3,
    informative_tol: float = 1e-9,
) -> Tuple[float, float]:
    x_req = x.detach().clone().requires_grad_(True)
    y_bpda = bpda_fn(x_req)
    g = torch.autograd.grad(y_bpda.sum(), x_req, create_graph=False, retain_graph=False)[0].detach()

    v = torch.randn_like(x_req)
    v_norm = v.reshape(v.shape[0], -1).norm(dim=1, keepdim=True).clamp_min(1e-12)
    v = v / v_norm.view(-1, *([1] * (x_req.ndim - 1)))

    yp = _samplewise_sum(hard_fn(x_req.detach() + delta * v))
    ym = _samplewise_sum(hard_fn(x_req.detach() - delta * v))
    dy = yp - ym
    pred = (g * v).reshape(g.shape[0], -1).sum(dim=1)

    mask = dy.abs() > informative_tol
    info_ratio = float(mask.float().mean().item())
    if bool(mask.any()):
        agree = (torch.sign(pred[mask]) == torch.sign(dy[mask])).float().mean().item()
        return float(agree), info_ratio
    return 1.0, info_ratio


def _sample_pair(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(batch, device=device, dtype=dtype)
    b = torch.randn(batch, device=device, dtype=dtype)
    return torch.stack([a, b], dim=1)


def _sample_pair_near_boundary(
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 0.03,
) -> torch.Tensor:
    base = torch.randn(batch, 1, device=device, dtype=dtype)
    delta = scale * torch.randn(batch, 1, device=device, dtype=dtype)
    a = base + delta
    b = base - delta
    return torch.cat([a, b], dim=1)


def _sample_pair_far_boundary(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.randn(batch, 1, device=device, dtype=dtype)
    mag = 1.0 + 2.0 * torch.rand(batch, 1, device=device, dtype=dtype)
    sign = torch.where(torch.rand(batch, 1, device=device, dtype=dtype) > 0.5, 1.0, -1.0)
    delta = sign * mag
    a = base + delta
    b = base - delta
    return torch.cat([a, b], dim=1)


def _sample_vector(
    batch: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    near_tie: bool = False,
) -> torch.Tensor:
    if near_tie:
        base = torch.randn(batch, 1, device=device, dtype=dtype)
        return base + 0.02 * torch.randn(batch, width, device=device, dtype=dtype)
    return torch.randn(batch, width, device=device, dtype=dtype)


def _sample_series(
    batch: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    near_tie: bool = False,
) -> torch.Tensor:
    if near_tie:
        base = torch.randn(batch, 1, device=device, dtype=dtype)
        return base + 0.02 * torch.randn(batch, length, device=device, dtype=dtype)
    noise = 0.25 * torch.randn(batch, length, device=device, dtype=dtype)
    drift = 0.02 * torch.randn(batch, 1, device=device, dtype=dtype)
    walk = torch.cumsum(noise + drift, dim=1)
    t = torch.linspace(0.0, 6.283185307179586, length, device=device, dtype=dtype)
    seasonal = 0.2 * torch.sin(t).unsqueeze(0)
    return walk + seasonal


def _evaluate_operator_core(
    name: str,
    group: str,
    sample_fn: Callable[[], torch.Tensor],
    hard_fn: Callable[[torch.Tensor], torch.Tensor],
    soft_fn: Callable[[torch.Tensor], torch.Tensor],
    bpda_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    directional_delta: float = 1e-3,
    forward_tol: float = 1e-9,
    cosine_tol: float = 0.999,
    grad_err_tol: float = 5e-6,
    fd_rel_tol: float = 5e-2,
    sign_acc_tol: float = 0.75,
) -> CheckResult:
    x = sample_fn()
    y_hard = hard_fn(x)
    y_bpda, g_bpda = _value_and_grad(bpda_fn, x)
    _, g_soft = _value_and_grad(soft_fn, x)

    forward_max_abs_err = float((y_bpda - y_hard).abs().max().item())
    grad_cosine = _cosine_similarity(g_bpda, g_soft)
    grad_max_abs_err = float((g_bpda - g_soft).abs().max().item())
    grad_finite_rate = float(torch.isfinite(g_bpda).float().mean().item())

    idx, g_fd = _finite_diff_grad_subset(soft_fn, x, eps=1e-4, max_coords=24)
    g_soft_flat = g_soft.reshape(-1)
    g_soft_subset = g_soft_flat[idx]
    soft_fd_rel_err = float(((g_fd - g_soft_subset).abs() / (g_soft_subset.abs() + 1e-8)).mean().item())

    sign_acc, info_ratio = _directional_sign_agreement(
        hard_fn=hard_fn,
        bpda_fn=bpda_fn,
        x=x,
        delta=directional_delta,
    )

    passed = (
        (forward_max_abs_err <= forward_tol)
        and (grad_cosine >= cosine_tol)
        and (grad_max_abs_err <= grad_err_tol)
        and (grad_finite_rate == 1.0)
        and (soft_fd_rel_err <= fd_rel_tol)
        and ((info_ratio < 0.10) or (sign_acc >= sign_acc_tol))
    )

    return CheckResult(
        name=name,
        group=group,
        passed=passed,
        metrics={
            "forward_max_abs_err": forward_max_abs_err,
            "grad_cosine_bpda_vs_soft": grad_cosine,
            "grad_max_abs_err_bpda_vs_soft": grad_max_abs_err,
            "grad_finite_rate": grad_finite_rate,
            "soft_fd_rel_err": soft_fd_rel_err,
            "direction_sign_acc": sign_acc,
            "direction_info_ratio": info_ratio,
        },
        thresholds={
            "forward_tol_le": forward_tol,
            "grad_cosine_ge": cosine_tol,
            "grad_err_tol_le": grad_err_tol,
            "grad_finite_rate_eq": 1.0,
            "soft_fd_rel_err_le": fd_rel_tol,
            "sign_acc_ge_if_info_ratio>=0.10": sign_acc_tol,
        },
    )


def _check_semantics_maxmin(
    *,
    name: str,
    group: str,
    bpda_fn: Callable[[torch.Tensor], torch.Tensor],
    expect_a_dominant_when_a_gt_b: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> CheckResult:
    x = _sample_pair(512, device, dtype)
    _, g = _value_and_grad(bpda_fn, x)
    ga = g[:, 0]
    gb = g[:, 1]
    diff = x[:, 0] - x[:, 1]

    if expect_a_dominant_when_a_gt_b:
        dom = torch.where(diff > 0, ga > gb, ga < gb)
    else:
        dom = torch.where(diff > 0, ga < gb, ga > gb)
    dom = torch.where(diff.abs() < 1e-12, torch.ones_like(dom, dtype=torch.bool), dom)

    dominance_acc = float(dom.float().mean().item())
    grad_pair_sum_mae = float((ga + gb - 1.0).abs().mean().item())

    x_tie = _sample_pair_near_boundary(512, device, dtype, scale=1e-4)
    _, g_tie = _value_and_grad(bpda_fn, x_tie)
    tie_balance_mae = float((g_tie[:, 0] - 0.5).abs().mean().item())

    thresholds = {
        "dominance_acc_ge": 0.95,
        "grad_pair_sum_mae_le": 1e-3,
        "tie_balance_mae_le": 0.05,
    }
    passed = (
        (dominance_acc >= thresholds["dominance_acc_ge"])
        and (grad_pair_sum_mae <= thresholds["grad_pair_sum_mae_le"])
        and (tie_balance_mae <= thresholds["tie_balance_mae_le"])
    )
    return CheckResult(
        name=name,
        group=group,
        passed=passed,
        metrics={
            "dominance_acc": dominance_acc,
            "grad_pair_sum_mae": grad_pair_sum_mae,
            "tie_balance_mae": tie_balance_mae,
        },
        thresholds=thresholds,
    )


def _check_semantics_cmp(
    *,
    name: str,
    group: str,
    bpda_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> CheckResult:
    x_near = _sample_pair_near_boundary(512, device, dtype, scale=0.02)
    _, g_near = _value_and_grad(bpda_fn, x_near)
    x_far = _sample_pair_far_boundary(512, device, dtype)
    _, g_far = _value_and_grad(bpda_fn, x_far)

    near_mag = float(g_near.abs().mean().item())
    far_mag = float(g_far.abs().mean().item())
    boundary_sensitivity_ratio = near_mag / (far_mag + 1e-12)
    anti_sym_mae = float((g_near[:, 0] + g_near[:, 1]).abs().mean().item())

    thresholds = {
        "boundary_sensitivity_ratio_ge": 3.0,
        "anti_sym_mae_le": 1e-6,
    }
    passed = (
        (boundary_sensitivity_ratio >= thresholds["boundary_sensitivity_ratio_ge"])
        and (anti_sym_mae <= thresholds["anti_sym_mae_le"])
    )
    return CheckResult(
        name=name,
        group=group,
        passed=passed,
        metrics={
            "near_grad_mag": near_mag,
            "far_grad_mag": far_mag,
            "boundary_sensitivity_ratio": boundary_sensitivity_ratio,
            "anti_sym_mae": anti_sym_mae,
        },
        thresholds=thresholds,
    )


def _check_semantics_idx(
    *,
    name: str,
    group: str,
    bpda_fn: Callable[[torch.Tensor], torch.Tensor],
    expect_grad_nondecreasing: bool,
    device: torch.device,
    dtype: torch.dtype,
    width: int = 20,
) -> CheckResult:
    x = torch.zeros(256, width, device=device, dtype=dtype)
    _, g = _value_and_grad(bpda_fn, x)
    diff = g[:, 1:] - g[:, :-1]
    if expect_grad_nondecreasing:
        monotonic_ratio = float((diff >= -1e-9).float().mean().item())
    else:
        monotonic_ratio = float((diff <= 1e-9).float().mean().item())
    translation_invariance_mae = float(g.sum(dim=1).abs().mean().item())

    thresholds = {
        "monotonic_ratio_ge": 0.999,
        "translation_invariance_mae_le": 1e-6,
    }
    passed = (
        (monotonic_ratio >= thresholds["monotonic_ratio_ge"])
        and (translation_invariance_mae <= thresholds["translation_invariance_mae_le"])
    )
    return CheckResult(
        name=name,
        group=group,
        passed=passed,
        metrics={
            "monotonic_ratio": monotonic_ratio,
            "translation_invariance_mae": translation_invariance_mae,
        },
        thresholds=thresholds,
    )


def _check_semantics_quantile(device: torch.device, dtype: torch.dtype, width: int = 20) -> CheckResult:
    x = _sample_vector(512, width, device, dtype, near_tie=False)
    y20 = bpda_quantile_window(x, 0.2, reg_strength=DEFAULT_REG_STRENGTH, pick_strength=DEFAULT_PICK_STRENGTH)
    y80 = bpda_quantile_window(x, 0.8, reg_strength=DEFAULT_REG_STRENGTH, pick_strength=DEFAULT_PICK_STRENGTH)
    monotonic_q_ratio = float((y80 >= y20 - 1e-9).float().mean().item())

    _, g50 = _value_and_grad(
        lambda z: bpda_quantile_window(
            z,
            0.5,
            reg_strength=DEFAULT_REG_STRENGTH,
            pick_strength=DEFAULT_PICK_STRENGTH,
        ),
        x,
    )
    translation_invariance_mae = float((g50.sum(dim=1) - 1.0).abs().mean().item())

    thresholds = {
        "monotonic_q_ratio_ge": 0.999,
        "translation_invariance_mae_le": 1e-4,
    }
    passed = (
        (monotonic_q_ratio >= thresholds["monotonic_q_ratio_ge"])
        and (translation_invariance_mae <= thresholds["translation_invariance_mae_le"])
    )
    return CheckResult(
        name="semantic_quantile",
        group="semantic",
        passed=passed,
        metrics={
            "monotonic_q_ratio": monotonic_q_ratio,
            "translation_invariance_mae": translation_invariance_mae,
        },
        thresholds=thresholds,
    )


def _check_semantics_rank(device: torch.device, dtype: torch.dtype, window: int = 20) -> CheckResult:
    near = _sample_series(256, 80, device, dtype, near_tie=True)
    _, g_near = _value_and_grad(lambda z: op_Rank(z, window), near)

    base = torch.arange(80, device=device, dtype=dtype).unsqueeze(0).repeat(256, 1)
    far = base + 0.01 * torch.randn(256, 80, device=device, dtype=dtype)
    _, g_far = _value_and_grad(lambda z: op_Rank(z, window), far)

    near_mag = float(g_near.abs().mean().item())
    far_mag = float(g_far.abs().mean().item())
    boundary_sensitivity_ratio = near_mag / (far_mag + 1e-12)

    thresholds = {"boundary_sensitivity_ratio_ge": 5.0}
    passed = boundary_sensitivity_ratio >= thresholds["boundary_sensitivity_ratio_ge"]
    return CheckResult(
        name="semantic_rank_boundary_sensitivity",
        group="semantic",
        passed=passed,
        metrics={
            "near_grad_mag": near_mag,
            "far_grad_mag": far_mag,
            "boundary_sensitivity_ratio": boundary_sensitivity_ratio,
        },
        thresholds=thresholds,
    )


def _print_result(r: CheckResult) -> None:
    status = "通过" if r.passed else "失败"
    print(f"[{status}] {r.group}/{r.name}")
    for k, v in r.metrics.items():
        print(f"  {k}: {v:.8f}")
    if r.note:
        print(f"  note: {r.note}")


def _serialize_results(results: List[CheckResult]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(description="BPDA 梯度完整验证。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch 运行设备。")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"], help="Torch 数值类型。")
    parser.add_argument("--json-out", type=str, default="reports/bpda_grad_validation_report.json", help="JSON 报告输出路径。")
    parser.add_argument("--quiet", action="store_true", help="仅打印摘要。")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    _set_seed(args.seed)

    print(f"[env] device={device} dtype={dtype} seed={args.seed}")
    print("[环境] 正在验证 BPDA 前向/反向机制与梯度语义一致性")

    N_VEC = 20
    L_SERIES = 80
    W_RANK = 20

    pair_max_hard = lambda x: torch.maximum(x[:, 0], x[:, 1])
    pair_max_soft = lambda x: smooth_max_pair(x[:, 0], x[:, 1], tau=DEFAULT_TAU_MAXMIN)
    pair_max_bpda = lambda x: bpda_max_pair(x[:, 0], x[:, 1], tau=DEFAULT_TAU_MAXMIN)

    pair_min_hard = lambda x: torch.minimum(x[:, 0], x[:, 1])
    pair_min_soft = lambda x: smooth_min_pair(x[:, 0], x[:, 1], tau=DEFAULT_TAU_MAXMIN)
    pair_min_bpda = lambda x: bpda_min_pair(x[:, 0], x[:, 1], tau=DEFAULT_TAU_MAXMIN)

    pair_gt_hard = lambda x: (x[:, 0] > x[:, 1]).to(dtype=x.dtype)
    pair_gt_soft = lambda x: soft_greater(x[:, 0], x[:, 1], temperature=DEFAULT_TEMP_CMP)
    pair_gt_bpda = lambda x: bpda_greater(x[:, 0], x[:, 1], temperature=DEFAULT_TEMP_CMP)

    pair_lt_hard = lambda x: (x[:, 0] < x[:, 1]).to(dtype=x.dtype)
    pair_lt_soft = lambda x: soft_less(x[:, 0], x[:, 1], temperature=DEFAULT_TEMP_CMP)
    pair_lt_bpda = lambda x: bpda_less(x[:, 0], x[:, 1], temperature=DEFAULT_TEMP_CMP)

    quantile_hard = lambda x: torch.quantile(x, 0.5, dim=-1)
    quantile_soft = lambda x: soft_quantile_window(
        x,
        0.5,
        reg_strength=DEFAULT_REG_STRENGTH,
        pick_strength=DEFAULT_PICK_STRENGTH,
    )
    quantile_bpda = lambda x: bpda_quantile_window(
        x,
        0.5,
        reg_strength=DEFAULT_REG_STRENGTH,
        pick_strength=DEFAULT_PICK_STRENGTH,
    )

    idxmax_hard = lambda x: (torch.argmax(x, dim=-1).to(x.dtype) + 1.0)
    idxmax_soft = lambda x: (
        torch.softmax(x / DEFAULT_TAU_IDX, dim=-1)
        * torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype).view(1, -1)
    ).sum(dim=-1)
    idxmax_bpda = lambda x: bpda_idxmax(x, tau=DEFAULT_TAU_IDX, dim=-1)

    idxmin_hard = lambda x: (torch.argmin(x, dim=-1).to(x.dtype) + 1.0)
    idxmin_soft = lambda x: (
        torch.softmax(-x / DEFAULT_TAU_IDX, dim=-1)
        * torch.arange(1, x.size(-1) + 1, device=x.device, dtype=x.dtype).view(1, -1)
    ).sum(dim=-1)
    idxmin_bpda = lambda x: bpda_idxmin(x, tau=DEFAULT_TAU_IDX, dim=-1)

    win_max_hard = lambda x: rolling_unfold(x, W_RANK, dim=1).max(dim=-1).values
    win_max_soft = lambda x: torch.logsumexp(rolling_unfold(x, W_RANK, dim=1) * DEFAULT_TAU_MAXMIN, dim=-1) / DEFAULT_TAU_MAXMIN
    win_max_bpda = lambda x: op_Max(x, W_RANK)

    win_min_hard = lambda x: rolling_unfold(x, W_RANK, dim=1).min(dim=-1).values
    win_min_soft = lambda x: -torch.logsumexp(-rolling_unfold(x, W_RANK, dim=1) * DEFAULT_TAU_MAXMIN, dim=-1) / DEFAULT_TAU_MAXMIN
    win_min_bpda = lambda x: op_Min(x, W_RANK)

    rank_hard = lambda x: (
        (lambda xu: (((xu < xu[..., -1:].expand_as(xu)).sum(dim=-1) + ((xu == xu[..., -1:].expand_as(xu)).sum(dim=-1) + 1) / 2.0) / float(W_RANK)))(
            rolling_unfold(x, W_RANK, dim=1)
        )
    )
    rank_soft = lambda x: (
        (lambda xu: soft_rank(xu.reshape(-1, W_RANK), tau=1.0, direction="ASCENDING", pct=True).reshape(xu.size(0), xu.size(1), W_RANK)[..., -1])(
            rolling_unfold(x, W_RANK, dim=1)
        )
    )
    rank_bpda = lambda x: op_Rank(x, W_RANK)

    results: List[CheckResult] = []

    results.append(
        _evaluate_operator_core(
            name="pair_max",
            group="core",
            sample_fn=lambda: _sample_pair(256, device, dtype),
            hard_fn=pair_max_hard,
            soft_fn=pair_max_soft,
            bpda_fn=pair_max_bpda,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="pair_min",
            group="core",
            sample_fn=lambda: _sample_pair(256, device, dtype),
            hard_fn=pair_min_hard,
            soft_fn=pair_min_soft,
            bpda_fn=pair_min_bpda,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="pair_gt",
            group="core",
            sample_fn=lambda: _sample_pair_near_boundary(256, device, dtype, scale=0.03),
            hard_fn=pair_gt_hard,
            soft_fn=pair_gt_soft,
            bpda_fn=pair_gt_bpda,
            directional_delta=5e-3,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="pair_lt",
            group="core",
            sample_fn=lambda: _sample_pair_near_boundary(256, device, dtype, scale=0.03),
            hard_fn=pair_lt_hard,
            soft_fn=pair_lt_soft,
            bpda_fn=pair_lt_bpda,
            directional_delta=5e-3,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="quantile_q50",
            group="core",
            sample_fn=lambda: _sample_vector(128, N_VEC, device, dtype, near_tie=True),
            hard_fn=quantile_hard,
            soft_fn=quantile_soft,
            bpda_fn=quantile_bpda,
            directional_delta=2e-3,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="idxmax",
            group="core",
            sample_fn=lambda: _sample_vector(128, N_VEC, device, dtype, near_tie=True),
            hard_fn=idxmax_hard,
            soft_fn=idxmax_soft,
            bpda_fn=idxmax_bpda,
            directional_delta=1e-2,
            sign_acc_tol=0.65,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="idxmin",
            group="core",
            sample_fn=lambda: _sample_vector(128, N_VEC, device, dtype, near_tie=True),
            hard_fn=idxmin_hard,
            soft_fn=idxmin_soft,
            bpda_fn=idxmin_bpda,
            directional_delta=1e-2,
            sign_acc_tol=0.65,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="window_max_n20",
            group="core",
            sample_fn=lambda: _sample_series(64, L_SERIES, device, dtype, near_tie=False),
            hard_fn=win_max_hard,
            soft_fn=win_max_soft,
            bpda_fn=win_max_bpda,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="window_min_n20",
            group="core",
            sample_fn=lambda: _sample_series(64, L_SERIES, device, dtype, near_tie=False),
            hard_fn=win_min_hard,
            soft_fn=win_min_soft,
            bpda_fn=win_min_bpda,
        )
    )
    results.append(
        _evaluate_operator_core(
            name="window_rank_n20",
            group="core",
            sample_fn=lambda: _sample_series(64, L_SERIES, device, dtype, near_tie=True),
            hard_fn=rank_hard,
            soft_fn=rank_soft,
            bpda_fn=rank_bpda,
            directional_delta=1e-2,
            sign_acc_tol=0.50,
        )
    )

    results.append(
        _check_semantics_maxmin(
            name="semantic_max_pair",
            group="semantic",
            bpda_fn=pair_max_bpda,
            expect_a_dominant_when_a_gt_b=True,
            device=device,
            dtype=dtype,
        )
    )
    results.append(
        _check_semantics_maxmin(
            name="semantic_min_pair",
            group="semantic",
            bpda_fn=pair_min_bpda,
            expect_a_dominant_when_a_gt_b=False,
            device=device,
            dtype=dtype,
        )
    )
    results.append(
        _check_semantics_cmp(
            name="semantic_gt_boundary",
            group="semantic",
            bpda_fn=pair_gt_bpda,
            device=device,
            dtype=dtype,
        )
    )
    results.append(
        _check_semantics_cmp(
            name="semantic_lt_boundary",
            group="semantic",
            bpda_fn=pair_lt_bpda,
            device=device,
            dtype=dtype,
        )
    )
    results.append(
        _check_semantics_idx(
            name="semantic_idxmax_order",
            group="semantic",
            bpda_fn=idxmax_bpda,
            expect_grad_nondecreasing=True,
            device=device,
            dtype=dtype,
            width=N_VEC,
        )
    )
    results.append(
        _check_semantics_idx(
            name="semantic_idxmin_order",
            group="semantic",
            bpda_fn=idxmin_bpda,
            expect_grad_nondecreasing=False,
            device=device,
            dtype=dtype,
            width=N_VEC,
        )
    )
    results.append(_check_semantics_quantile(device=device, dtype=dtype, width=N_VEC))
    results.append(_check_semantics_rank(device=device, dtype=dtype, window=W_RANK))

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    failed = [r for r in results if not r.passed]

    if not args.quiet:
        for r in results:
            _print_result(r)
            print("")

    print(f"[summary] passed={passed} failed={total - passed} total={total}")
    if failed:
        print("[摘要] 未通过的检查项：")
        for r in failed:
            print(f"  - {r.group}/{r.name}")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "env": {
            "device": str(device),
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total,
        },
        "results": _serialize_results(results),
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[摘要] 已写入报告: {out_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
