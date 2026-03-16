from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LegacyLSTMPredictor(nn.Module):
    def __init__(self, d_feat: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.d_feat = int(d_feat)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.rnn = nn.LSTM(
            input_size=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_legacy_lstm_from_files(
    config_path: str | Path,
    state_dict_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LegacyLSTMPredictor:
    config = _load_json(config_path)
    model = LegacyLSTMPredictor(
        d_feat=config["d_feat"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    state_dict = torch.load(state_dict_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


__all__ = ["LegacyLSTMPredictor", "load_legacy_lstm_from_files"]
