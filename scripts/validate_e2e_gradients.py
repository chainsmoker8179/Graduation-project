#!/usr/bin/env python3
"""End-to-end gradient validation for OHLCV-based differentiable factors.

This script validates three gradient paths on a real Qlib dataset:
1. Full E2E: MSE loss through an OHLCV-only extractor and a lightweight predictor.
2. Subset E2E: MSE loss through a discrete-operator-focused factor subset.
3. Factor Probe: direct per-factor probe losses to isolate extractor-side gradients.

The script does not modify the production extractor. It builds an OHLCV-only
extractor locally by dropping factors that require VWAP.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpha158_ops import eval_graph
from alpha158_rolling import align_to_length
from alpha158_templates import build_templates


INPUT_COLUMNS = ["open", "high", "low", "close", "volume"]
FACTOR_GROUPS = {
    "greater_less_pair": ["KUP", "KUP2", "KLOW", "KLOW2"],
    "greater_relu_close": ["SUMP5", "SUMN5", "SUMD5"],
    "greater_relu_volume": ["VSUMP5", "VSUMN5", "VSUMD5"],
    "rolling_maxmin": ["MAX5", "MIN5", "RSV5"],
    "quantile": ["QTLU5", "QTLD5"],
    "rank": ["RANK5"],
    "idx": ["IMAX5", "IMIN5", "IMXD5"],
}
FACTOR_EXPECTED_INPUTS = {
    "KUP": ["open", "high", "close"],
    "KUP2": ["open", "high", "close"],
    "KLOW": ["open", "low", "close"],
    "KLOW2": ["open", "low", "close"],
    "SUMP5": ["close"],
    "SUMN5": ["close"],
    "SUMD5": ["close"],
    "VSUMP5": ["volume"],
    "VSUMN5": ["volume"],
    "VSUMD5": ["volume"],
    "MAX5": ["high", "close"],
    "MIN5": ["low", "close"],
    "RSV5": ["high", "low", "close"],
    "QTLU5": ["close"],
    "QTLD5": ["close"],
    "RANK5": ["close"],
    "IMAX5": ["high"],
    "IMIN5": ["low"],
    "IMXD5": ["high", "low"],
}


@dataclass
class BatchStats:
    unit: str
    split: str
    batch_idx: int
    loss: float
    grad_is_none: bool
    grad_finite_rate: float
    grad_mean_abs: float
    grad_max_abs: float


class QlibOHLCVDataset(Dataset):
    """Wrap a qlib TSDataset split into torch tensors."""

    def __init__(self, qlib_ts_set: Any, valid_idx: np.ndarray, ohlcv_dim: int = 5, label_col: int = 5):
        self.ds = qlib_ts_set
        self.valid_idx = valid_idx
        self.ohlcv_dim = ohlcv_dim
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, j: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.valid_idx[j])
        s = self.ds[i]
        x = torch.from_numpy(s[:, : self.ohlcv_dim].astype(np.float32))
        y = torch.tensor(np.float32(s[-1, self.label_col]), dtype=torch.float32)
        return x, y


class OHLCVFactorExtractor(nn.Module):
    """Alpha158-style factor extractor that only depends on OHLCV."""

    def __init__(self, factor_names: Sequence[str], csv_path: str = "alpha158_name_expression.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.names = list(factor_names)
        self.name_to_graph = self._build_graph_map(csv_path, self.names)
        self.feat_dim = len(self.names)

    @staticmethod
    def _build_graph_map(csv_path: str, factor_names: Sequence[str]) -> Dict[str, tuple[Any, Dict[str, Any]]]:
        wanted = set(factor_names)
        templates = build_templates(csv_path)
        mapping: Dict[str, tuple[Any, Dict[str, Any]]] = {}
        for t in templates:
            graph = t["graph"]
            for name in t["names"]:
                if name not in wanted:
                    continue
                params = t.get("name_params", {}).get(name, {})
                mapping[name] = (graph, params)
        missing = [name for name in factor_names if name not in mapping]
        if missing:
            raise ValueError(f"Missing factor graphs for: {missing}")
        return mapping

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        open_ = x_raw[..., 0]
        high_ = x_raw[..., 1]
        low_ = x_raw[..., 2]
        close_ = x_raw[..., 3]
        volume_ = x_raw[..., 4]

        variables = {
            "open_": open_,
            "high_": high_,
            "low_": low_,
            "close_": close_,
            "volume_": volume_,
        }

        feats = []
        lengths = []
        for name in self.names:
            graph, params = self.name_to_graph[name]
            out = eval_graph(graph, variables, params)
            feats.append(out)
            lengths.append(out.size(1))

        min_len = min(lengths)
        feats = [align_to_length(f, min_len, dim=1) for f in feats]
        return torch.stack(feats, dim=-1)


class MeanLinearPredictor(nn.Module):
    """Lightweight prediction head for gradient-path validation."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = feats.mean(dim=1)
        return self.fc(pooled).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate end-to-end OHLCV gradients on a Qlib dataset.")
    parser.add_argument("--provider-uri", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--market", type=str, default="all")
    parser.add_argument("--start-time", type=str, default="2025-01-01")
    parser.add_argument("--end-time", type=str, default="2025-12-31")
    parser.add_argument("--fit-start-time", type=str, default="2025-01-01")
    parser.add_argument("--fit-end-time", type=str, default="2025-08-31")
    parser.add_argument("--test-start-time", type=str, default="2025-09-01")
    parser.add_argument("--test-end-time", type=str, default="2025-10-31")
    parser.add_argument("--label-expr", type=str, default="Ref($close, -2) / Ref($close, -1) - 1")
    parser.add_argument("--step-len", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full-train-batches", type=int, default=20)
    parser.add_argument("--full-test-batches", type=int, default=10)
    parser.add_argument("--subset-train-batches", type=int, default=20)
    parser.add_argument("--subset-test-batches", type=int, default=10)
    parser.add_argument("--probe-train-batches", type=int, default=8)
    parser.add_argument("--probe-test-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv-path", type=str, default="alpha158_name_expression.csv")
    parser.add_argument("--out-dir", type=str, default="reports/e2e_grad")
    parser.add_argument("--eps", type=float, default=1e-8, help="Probe pass threshold for mean(abs(input_grad)).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_factor_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_factor_sets(csv_path: str) -> tuple[List[str], List[str], Dict[str, List[str]]]:
    rows = load_factor_rows(csv_path)
    expr_by_name = {row["name"]: row["expression"] for row in rows}

    full_names = [row["name"] for row in rows if "$vwap" not in row["expression"]]

    subset_names: List[str] = []
    for names in FACTOR_GROUPS.values():
        subset_names.extend(names)
    subset_names = [name for name in subset_names if name in expr_by_name]

    missing = [name for name in subset_names if "$vwap" in expr_by_name[name]]
    if missing:
        raise ValueError(f"Subset factors unexpectedly depend on VWAP: {missing}")

    return full_names, subset_names, FACTOR_GROUPS


def build_valid_indices(qlib_ts_set: Any, ohlcv_dim: int = 5, label_col: int = 5) -> np.ndarray:
    valid = []
    for i in range(len(qlib_ts_set)):
        s = qlib_ts_set[i]
        x = s[:, :ohlcv_dim]
        y = s[-1, label_col]
        if np.isfinite(x).all() and np.isfinite(y):
            valid.append(i)
    return np.array(valid, dtype=np.int64)


def build_split_loaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    provider_uri = os.path.expanduser(args.provider_uri)
    qlib.init(provider_uri=provider_uri)

    instruments = D.instruments(market=args.market)
    data_loader = QlibDataLoader(
        config={
            "feature": ["$open", "$high", "$low", "$close", "$volume"],
            "label": [args.label_expr],
        }
    )
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=args.start_time,
        end_time=args.end_time,
        data_loader=data_loader,
    )
    dataset = TSDatasetH(
        handler=handler,
        segments={
            "train": (args.fit_start_time, args.fit_end_time),
            "test": (args.test_start_time, args.test_end_time),
        },
        step_len=args.step_len,
    )

    split_loaders: Dict[str, DataLoader] = {}
    split_seed_offsets = {"train": 0, "test": 1}
    for split in ("train", "test"):
        split_set = dataset.prepare(split)
        valid_idx = build_valid_indices(split_set)
        torch_ds = QlibOHLCVDataset(split_set, valid_idx)
        generator = torch.Generator()
        generator.manual_seed(args.seed + split_seed_offsets[split])
        split_loaders[split] = DataLoader(
            torch_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
    return split_loaders


def collect_batches(loader: DataLoader, num_batches: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
    batches: List[tuple[torch.Tensor, torch.Tensor]] = []
    for xb, yb in loader:
        batches.append((xb.clone(), yb.clone()))
        if len(batches) >= num_batches:
            break
    return batches


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def make_relative_scores(values: List[float]) -> List[float]:
    positive = [v for v in values if math.isfinite(v)]
    if not positive:
        return [0.0 for _ in values]
    median = float(np.median(np.array(positive, dtype=np.float64)))
    median = max(median, 1e-20)
    return [v / median for v in values]


def summarize_columns(grad: torch.Tensor, x: torch.Tensor) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    raw_values = []
    norm_values = []
    for col_idx, name in enumerate(INPUT_COLUMNS):
        col_grad = grad[..., col_idx]
        raw = float(col_grad.abs().mean().item())
        scale = float(x[..., col_idx].detach().std().item())
        norm = raw * max(scale, 1e-12)
        raw_values.append(raw)
        norm_values.append(norm)
        stats.append(
            {
                "column": name,
                "grad_mean_abs": raw,
                "grad_mean_abs_scaled": norm,
            }
        )

    raw_rel = make_relative_scores(raw_values)
    norm_rel = make_relative_scores(norm_values)
    for item, raw_score, norm_score in zip(stats, raw_rel, norm_rel):
        item["relative_raw_score"] = raw_score
        item["relative_scaled_score"] = norm_score
    return stats


def run_e2e_unit(
    *,
    unit_name: str,
    batches_by_split: Dict[str, List[tuple[torch.Tensor, torch.Tensor]]],
    extractor: OHLCVFactorExtractor,
    predictor: MeanLinearPredictor,
    device: torch.device,
    out_factor_limit: int | None = None,
) -> tuple[List[BatchStats], List[Dict[str, Any]], List[Dict[str, Any]]]:
    batch_stats: List[BatchStats] = []
    column_rows: List[Dict[str, Any]] = []
    factor_rows: List[Dict[str, Any]] = []

    extractor.eval()
    predictor.eval()
    freeze_module(extractor)
    freeze_module(predictor)

    for split, batches in batches_by_split.items():
        for batch_idx, (xb, yb) in enumerate(batches):
            x = xb.to(device).detach().requires_grad_(True)
            y = yb.to(device)

            feats = extractor(x)
            feats.retain_grad()
            pred = predictor(feats)
            loss = F.mse_loss(pred, y)
            loss.backward()

            grad = x.grad
            grad_is_none = grad is None
            if grad_is_none:
                finite_rate = 0.0
                mean_abs = 0.0
                max_abs = 0.0
                column_stats = []
            else:
                finite_rate = float(torch.isfinite(grad).float().mean().item())
                mean_abs = float(grad.abs().mean().item())
                max_abs = float(grad.abs().max().item())
                column_stats = summarize_columns(grad, x)

            batch_stats.append(
                BatchStats(
                    unit=unit_name,
                    split=split,
                    batch_idx=batch_idx,
                    loss=float(loss.detach().item()),
                    grad_is_none=grad_is_none,
                    grad_finite_rate=finite_rate,
                    grad_mean_abs=mean_abs,
                    grad_max_abs=max_abs,
                )
            )

            for item in column_stats:
                row = {
                    "unit": unit_name,
                    "split": split,
                    "batch_idx": batch_idx,
                }
                row.update(item)
                column_rows.append(row)

            if feats.grad is not None:
                feat_grad_mean = feats.grad.abs().mean(dim=(0, 1)).detach().cpu().tolist()
                names = extractor.names
                if out_factor_limit is not None:
                    names = names[:out_factor_limit]
                    feat_grad_mean = feat_grad_mean[:out_factor_limit]
                for name, value in zip(names, feat_grad_mean):
                    factor_rows.append(
                        {
                            "unit": unit_name,
                            "split": split,
                            "batch_idx": batch_idx,
                            "factor_name": name,
                            "factor_grad_mean_abs": float(value),
                        }
                    )

    return batch_stats, column_rows, factor_rows


def run_factor_probes(
    *,
    batches_by_split: Dict[str, List[tuple[torch.Tensor, torch.Tensor]]],
    extractor: OHLCVFactorExtractor,
    group_map: Dict[str, List[str]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    probe_rows: List[Dict[str, Any]] = []
    extractor.eval()
    freeze_module(extractor)

    name_to_group = {name: group for group, names in group_map.items() for name in names}
    name_to_index = {name: i for i, name in enumerate(extractor.names)}

    for split, batches in batches_by_split.items():
        for factor_name in extractor.names:
            factor_idx = name_to_index[factor_name]
            group_name = name_to_group.get(factor_name, "unknown")
            for batch_idx, (xb, _) in enumerate(batches):
                x = xb.to(device).detach().requires_grad_(True)
                feats = extractor(x)
                target = feats[..., factor_idx]
                loss = 0.5 * target.pow(2).mean()
                loss.backward()
                grad = x.grad

                if grad is None:
                    finite_rate = 0.0
                    grad_mean_abs = 0.0
                    grad_max_abs = 0.0
                    column_stats = summarize_columns(torch.zeros_like(x), x)
                else:
                    finite_rate = float(torch.isfinite(grad).float().mean().item())
                    grad_mean_abs = float(grad.abs().mean().item())
                    grad_max_abs = float(grad.abs().max().item())
                    column_stats = summarize_columns(grad, x)

                row = {
                    "split": split,
                    "batch_idx": batch_idx,
                    "group_name": group_name,
                    "factor_name": factor_name,
                    "loss": float(loss.detach().item()),
                    "grad_is_none": grad is None,
                    "grad_finite_rate": finite_rate,
                    "grad_mean_abs": grad_mean_abs,
                    "grad_max_abs": grad_max_abs,
                }
                for item in column_stats:
                    col = item["column"]
                    row[f"{col}_grad_mean_abs"] = item["grad_mean_abs"]
                    row[f"{col}_grad_mean_abs_scaled"] = item["grad_mean_abs_scaled"]
                    row[f"{col}_relative_scaled_score"] = item["relative_scaled_score"]
                probe_rows.append(row)
    return probe_rows


def aggregate_batch_summary(batch_stats: Sequence[BatchStats]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for row in batch_stats:
        unit = row.unit
        summary.setdefault(unit, {"batches": 0, "pass_batches": 0, "splits": {}})
        summary[unit]["batches"] += 1
        passed = (not row.grad_is_none) and (row.grad_finite_rate == 1.0) and (row.grad_mean_abs > 0.0)
        summary[unit]["pass_batches"] += int(passed)

        split_info = summary[unit]["splits"].setdefault(
            row.split,
            {
                "batches": 0,
                "pass_batches": 0,
                "mean_loss": 0.0,
                "mean_grad_mean_abs": 0.0,
            },
        )
        split_info["batches"] += 1
        split_info["pass_batches"] += int(passed)
        split_info["mean_loss"] += row.loss
        split_info["mean_grad_mean_abs"] += row.grad_mean_abs

    for unit_info in summary.values():
        for split_info in unit_info["splits"].values():
            denom = max(split_info["batches"], 1)
            split_info["mean_loss"] /= denom
            split_info["mean_grad_mean_abs"] /= denom
            split_info["pass_rate"] = split_info["pass_batches"] / denom
        unit_info["pass_rate"] = unit_info["pass_batches"] / max(unit_info["batches"], 1)
    return summary


def aggregate_suspicious_columns(column_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in column_rows:
        unit = row["unit"]
        column = row["column"]
        grouped.setdefault(unit, {}).setdefault(column, []).append(float(row["relative_scaled_score"]))

    out: Dict[str, Dict[str, Any]] = {}
    for unit, col_map in grouped.items():
        out[unit] = {}
        for column, scores in col_map.items():
            if not scores:
                continue
            low_ratio = float(np.mean(np.array(scores) < 1e-3))
            out[unit][column] = {
                "num_batches": len(scores),
                "low_ratio": low_ratio,
                "is_suspicious": low_ratio >= 0.8,
                "mean_relative_scaled_score": float(np.mean(scores)),
            }
    return out


def aggregate_probe_summary(probe_rows: Sequence[Dict[str, Any]], eps: float) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in probe_rows:
        key = row["factor_name"]
        grouped.setdefault(key, []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for factor_name, rows in grouped.items():
        total = len(rows)
        pass_count = 0
        group_name = rows[0]["group_name"]
        col_scores = {col: [] for col in INPUT_COLUMNS}
        expected_inputs = FACTOR_EXPECTED_INPUTS.get(factor_name, INPUT_COLUMNS)
        expected_grad_values: List[float] = []
        for row in rows:
            expected_grad_mean_abs = float(
                np.mean([float(row[f"{col}_grad_mean_abs"]) for col in expected_inputs])
            )
            passed = (
                (not row["grad_is_none"])
                and (row["grad_finite_rate"] == 1.0)
                and (expected_grad_mean_abs > eps)
            )
            pass_count += int(passed)
            expected_grad_values.append(expected_grad_mean_abs)
            for col in INPUT_COLUMNS:
                col_scores[col].append(float(row[f"{col}_relative_scaled_score"]))
        out[factor_name] = {
            "group_name": group_name,
            "num_batches": total,
            "pass_rate": pass_count / max(total, 1),
            "mean_grad_mean_abs": float(np.mean([row["grad_mean_abs"] for row in rows])),
            "mean_expected_grad_mean_abs": float(np.mean(expected_grad_values)),
            "expected_inputs": expected_inputs,
            "suspicious_expected_columns": [
                col
                for col, scores in col_scores.items()
                if col in expected_inputs and scores and float(np.mean(np.array(scores) < 1e-3)) >= 0.8
            ],
        }
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    full_names: Sequence[str],
    subset_names: Sequence[str],
    batch_summary: Dict[str, Dict[str, Any]],
    suspicious_columns: Dict[str, Dict[str, Any]],
    probe_summary: Dict[str, Dict[str, Any]],
) -> None:
    lines = [
        "# E2E Gradient Validation Report",
        "",
        "## Setup",
        "",
        f"- provider_uri: `{os.path.expanduser(args.provider_uri)}`",
        f"- market: `{args.market}`",
        f"- label_expr: `{args.label_expr}`",
        f"- step_len: `{args.step_len}`",
        f"- batch_size: `{args.batch_size}`",
        f"- seed: `{args.seed}`",
        f"- full extractor factors: `{len(full_names)}`",
        f"- subset factors: `{len(subset_names)}`",
        "",
        "## Batch Summary",
        "",
    ]

    for unit, info in batch_summary.items():
        lines.append(f"### {unit}")
        lines.append("")
        lines.append(f"- pass_rate: `{info['pass_rate']:.4f}`")
        for split, split_info in info["splits"].items():
            lines.append(
                f"- {split}: batches=`{split_info['batches']}`, pass_rate=`{split_info['pass_rate']:.4f}`, "
                f"mean_loss=`{split_info['mean_loss']:.6e}`, mean_grad_mean_abs=`{split_info['mean_grad_mean_abs']:.6e}`"
            )
        lines.append("")

    lines.extend(["## Suspicious Columns", ""])
    for unit, col_map in suspicious_columns.items():
        flagged = [col for col, stats in col_map.items() if stats["is_suspicious"]]
        lines.append(f"### {unit}")
        lines.append("")
        if flagged:
            for col in flagged:
                stats = col_map[col]
                lines.append(
                    f"- {col}: low_ratio=`{stats['low_ratio']:.4f}`, "
                    f"mean_relative_scaled_score=`{stats['mean_relative_scaled_score']:.6e}`"
                )
        else:
            lines.append("- no suspicious columns detected under the current threshold")
        lines.append("")

    lines.extend(["## Factor Probe Summary", ""])
    weak = [(name, info) for name, info in probe_summary.items() if info["pass_rate"] < 0.9]
    if weak:
        for factor_name, info in sorted(weak):
            lines.append(
                f"- {factor_name} ({info['group_name']}): pass_rate=`{info['pass_rate']:.4f}`, "
                f"mean_grad_mean_abs=`{info['mean_grad_mean_abs']:.6e}`, "
                f"mean_expected_grad_mean_abs=`{info['mean_expected_grad_mean_abs']:.6e}`, "
                f"suspicious_expected_columns=`{','.join(info['suspicious_expected_columns']) or 'none'}`"
            )
    else:
        lines.append("- all factor probes pass the current threshold")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.step_len < 61:
        raise SystemExit("--step-len must be at least 61 to keep 60-day factors meaningful.")

    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_names, subset_names, group_map = build_factor_sets(args.csv_path)
    split_loaders = build_split_loaders(args)

    max_train_batches = max(args.full_train_batches, args.subset_train_batches, args.probe_train_batches)
    max_test_batches = max(args.full_test_batches, args.subset_test_batches, args.probe_test_batches)
    cached_batches = {
        "train": collect_batches(split_loaders["train"], max_train_batches),
        "test": collect_batches(split_loaders["test"], max_test_batches),
    }

    full_batches = {
        "train": cached_batches["train"][: args.full_train_batches],
        "test": cached_batches["test"][: args.full_test_batches],
    }
    subset_batches = {
        "train": cached_batches["train"][: args.subset_train_batches],
        "test": cached_batches["test"][: args.subset_test_batches],
    }
    probe_batches = {
        "train": cached_batches["train"][: args.probe_train_batches],
        "test": cached_batches["test"][: args.probe_test_batches],
    }

    full_extractor = OHLCVFactorExtractor(full_names, csv_path=args.csv_path).to(device)
    subset_extractor = OHLCVFactorExtractor(subset_names, csv_path=args.csv_path).to(device)

    torch.manual_seed(args.seed)
    full_predictor = MeanLinearPredictor(full_extractor.feat_dim).to(device)
    torch.manual_seed(args.seed + 1)
    subset_predictor = MeanLinearPredictor(subset_extractor.feat_dim).to(device)

    full_batch_stats, full_column_rows, full_factor_rows = run_e2e_unit(
        unit_name="full_e2e",
        batches_by_split=full_batches,
        extractor=full_extractor,
        predictor=full_predictor,
        device=device,
    )
    subset_batch_stats, subset_column_rows, subset_factor_rows = run_e2e_unit(
        unit_name="subset_e2e",
        batches_by_split=subset_batches,
        extractor=subset_extractor,
        predictor=subset_predictor,
        device=device,
    )
    probe_rows = run_factor_probes(
        batches_by_split=probe_batches,
        extractor=subset_extractor,
        group_map=group_map,
        device=device,
    )

    batch_stats = full_batch_stats + subset_batch_stats
    column_rows = full_column_rows + subset_column_rows
    factor_rows = full_factor_rows + subset_factor_rows

    batch_summary = aggregate_batch_summary(batch_stats)
    suspicious_columns = aggregate_suspicious_columns(column_rows)
    probe_summary = aggregate_probe_summary(probe_rows, args.eps)

    summary_payload = {
        "setup": {
            "provider_uri": os.path.expanduser(args.provider_uri),
            "market": args.market,
            "label_expr": args.label_expr,
            "step_len": args.step_len,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
            "full_factor_count": len(full_names),
            "subset_factor_count": len(subset_names),
        },
        "batch_summary": batch_summary,
        "suspicious_columns": suspicious_columns,
        "probe_summary": probe_summary,
    }

    (out_dir / "e2e_grad_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_csv(out_dir / "e2e_grad_batch_stats.csv", [vars(row) for row in batch_stats])
    write_csv(out_dir / "e2e_grad_column_stats.csv", column_rows)
    write_csv(out_dir / "e2e_grad_factor_stats.csv", factor_rows)
    write_csv(out_dir / "e2e_grad_probe_stats.csv", probe_rows)
    write_markdown_summary(
        path=out_dir / "e2e_grad_validation_report.md",
        args=args,
        full_names=full_names,
        subset_names=subset_names,
        batch_summary=batch_summary,
        suspicious_columns=suspicious_columns,
        probe_summary=probe_summary,
    )

    weak_probes = [name for name, info in probe_summary.items() if info["pass_rate"] < 0.9]
    print(f"[summary] full_factors={len(full_names)} subset_factors={len(subset_names)}")
    print(f"[summary] weak_probe_factors={len(weak_probes)}")
    print(f"[summary] outputs={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
