from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from alpha158_ops import eval_graph
from alpha158_rolling import align_to_length
from alpha158_templates import build_templates


LEGACY_LSTM_FEATURES = [
    "KLEN",
    "KLOW",
    "ROC60",
    "STD5",
    "RSQR5",
    "RSQR10",
    "RSQR20",
    "RSQR60",
    "RESI5",
    "RESI10",
    "CORR5",
    "CORR10",
    "CORR20",
    "CORR60",
    "CORD5",
    "CORD10",
    "CORD60",
    "VSTD5",
    "WVMA5",
    "WVMA60",
]


class LegacyLSTMFeatureBridge(nn.Module):
    def __init__(self, csv_path: str | Path | None = None, input_window_len: int = 80) -> None:
        super().__init__()
        self.input_window_len = int(input_window_len)
        self.feature_names = list(LEGACY_LSTM_FEATURES)
        self.csv_path = Path(csv_path) if csv_path is not None else Path(__file__).resolve().with_name("alpha158_name_expression.csv")
        self.name_to_graph = self._build_graph_map(self.csv_path, self.feature_names)

    @staticmethod
    def _build_graph_map(csv_path: Path, feature_names: list[str]) -> dict[str, tuple[Any, dict[str, Any]]]:
        wanted = set(feature_names)
        mapping: dict[str, tuple[Any, dict[str, Any]]] = {}
        for template in build_templates(str(csv_path)):
            graph = template["graph"]
            for name in template["names"]:
                if name not in wanted:
                    continue
                params = template.get("name_params", {}).get(name, {})
                mapping[name] = (graph, params)
        missing = [name for name in feature_names if name not in mapping]
        if missing:
            raise ValueError(f"missing legacy feature graphs: {missing}")
        return mapping

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim != 3 or x_raw.size(-1) != 5:
            raise ValueError(f"expected input shape (B, L, 5), got {tuple(x_raw.shape)}")
        if x_raw.size(1) != self.input_window_len:
            raise ValueError(f"legacy LSTM bridge expects raw window length {self.input_window_len}, got {x_raw.size(1)}")

        variables = {
            "open_": x_raw[..., 0],
            "high_": x_raw[..., 1],
            "low_": x_raw[..., 2],
            "close_": x_raw[..., 3],
            "volume_": x_raw[..., 4],
        }

        outputs = []
        lengths = []
        for name in self.feature_names:
            graph, params = self.name_to_graph[name]
            out = eval_graph(graph, variables, params)
            outputs.append(out)
            lengths.append(out.size(1))

        min_len = min(lengths)
        outputs = [align_to_length(out, min_len, dim=1) for out in outputs]
        return torch.stack(outputs, dim=-1)


__all__ = ["LEGACY_LSTM_FEATURES", "LegacyLSTMFeatureBridge"]
