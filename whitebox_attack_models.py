from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from legacy_lstm_predictor import LegacyLSTMPredictor


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class LSTMAdapter(nn.Module):
    def __init__(self, *, state_dict_path: Path, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        kwargs = dict(config["model_kwargs"])
        self.model = LegacyLSTMPredictor(
            d_feat=kwargs["d_feat"],
            hidden_size=kwargs["hidden_size"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
        )
        state_dict = torch.load(state_dict_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class _QlibWrapperAdapter(nn.Module):
    inner_attr_name: str

    def __init__(self, *, state_dict_path: Path, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        module = importlib.import_module(config["qlib_model_module"])
        wrapper_cls = getattr(module, config["qlib_model_class"])
        kwargs = dict(config["model_kwargs"])
        kwargs["GPU"] = -1 if device.type == "cpu" else kwargs.get("GPU", 0)
        wrapper = wrapper_cls(**kwargs)
        inner = getattr(wrapper, self.inner_attr_name)
        state_dict = torch.load(state_dict_path, map_location=device)
        inner.load_state_dict(state_dict)
        inner.to(device)
        self.model = inner

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class TransformerAdapter(_QlibWrapperAdapter):
    inner_attr_name = "model"


class TCNAdapter(_QlibWrapperAdapter):
    inner_attr_name = "TCN_model"


MODEL_ADAPTERS: dict[str, type[nn.Module]] = {
    "lstm": LSTMAdapter,
    "transformer": TransformerAdapter,
    "tcn": TCNAdapter,
}


def get_model_adapter_class(model_name: str) -> type[nn.Module]:
    try:
        return MODEL_ADAPTERS[model_name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported model_name: {model_name}") from exc


def load_model_adapter(
    *,
    model_name: str,
    model_root: str | Path,
    device: torch.device | str | None = None,
) -> nn.Module:
    resolved_device = _resolve_device(device)
    model_dir = Path(model_root) / model_name.lower() / "model"
    config = _load_json(model_dir / "model_config.json")
    adapter_cls = get_model_adapter_class(model_name)
    adapter = adapter_cls(
        state_dict_path=model_dir / "state_dict.pt",
        config=config,
        device=resolved_device,
    )
    return adapter.to(resolved_device)
