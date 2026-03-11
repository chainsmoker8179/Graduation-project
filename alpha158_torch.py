"""TorchFactorExtractor based on Alpha158 templates (module 8)."""

from __future__ import annotations

import csv
from typing import Dict, Tuple, Any, List

import torch
import torch.nn as nn

from alpha158_templates import build_templates
from alpha158_ops import eval_graph
from alpha158_rolling import align_to_length


class TorchFactorExtractor(nn.Module):
    """Compute Alpha158 factors with differentiable approximations.

    Input: x_raw (B, L, 6) -> [open, high, low, close, volume, vwap]
    Output: feats (B, L', 158)
    """

    def __init__(self, csv_path: str = "alpha158_name_expression.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.names = self._load_names(csv_path)
        self.name_to_graph = self._build_graph_map(csv_path)

    @staticmethod
    def _load_names(csv_path: str) -> List[str]:
        names = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["name"])
        return names

    @staticmethod
    def _build_graph_map(csv_path: str) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        templates = build_templates(csv_path)
        mapping = {}
        for t in templates:
            graph = t["graph"]
            for name in t["names"]:
                params = t.get("name_params", {}).get(name, {})
                mapping[name] = (graph, params)
        return mapping

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # x_raw: (B, L, 6)
        open_ = x_raw[..., 0]
        high_ = x_raw[..., 1]
        low_ = x_raw[..., 2]
        close_ = x_raw[..., 3]
        volume_ = x_raw[..., 4]
        vwap_ = x_raw[..., 5]

        variables = {
            "open_": open_,
            "high_": high_,
            "low_": low_,
            "close_": close_,
            "volume_": volume_,
            "vwap_": vwap_,
        }

        feats = []
        lengths = []
        for name in self.names:
            graph, params = self.name_to_graph[name]
            out = eval_graph(graph, variables, params)
            feats.append(out)
            lengths.append(out.size(1))

        Lp = min(lengths)
        feats = [align_to_length(f, Lp, dim=1) for f in feats]
        feat_tensor = torch.stack(feats, dim=-1)
        return feat_tensor


__all__ = ["TorchFactorExtractor"]
