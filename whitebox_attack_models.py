from __future__ import annotations

import importlib
import inspect
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


def _normalize_predictions(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    return pred.reshape(-1)


def _resolve_model_dir(model_root: Path, model_name: str) -> Path:
    candidates = [
        model_root / model_name.lower() / "model",
        model_root / model_name / "model",
        model_root / model_name.upper() / "model",
        model_root / model_name.capitalize() / "model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_state_dict_path(model_dir: Path) -> Path:
    default_path = model_dir / "state_dict.pt"
    if default_path.exists():
        return default_path

    pt_files = sorted(model_dir.glob("*.pt"))
    if len(pt_files) == 1:
        return pt_files[0]
    raise FileNotFoundError(f"could not uniquely resolve state_dict under {model_dir}")


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
        return _normalize_predictions(self.model(features))


class _QlibWrapperAdapter(nn.Module):
    inner_attr_name: str

    def __init__(self, *, state_dict_path: Path, config: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        module_name = config.get("qlib_wrapper_module") or config.get("qlib_model_module")
        class_name = config.get("qlib_wrapper_class") or config.get("qlib_model_class")
        if module_name is None or class_name is None:
            raise KeyError("config must provide qlib_wrapper_* or qlib_model_* fields")
        module = importlib.import_module(module_name)
        wrapper_cls = getattr(module, class_name)
        kwargs = dict(config["model_kwargs"])
        signature = inspect.signature(wrapper_cls.__init__)
        if "GPU" in signature.parameters:
            kwargs["GPU"] = -1 if device.type == "cpu" else kwargs.get("GPU", 0)
        wrapper = wrapper_cls(**kwargs)
        inner_attr_name = config.get("torch_submodule_attr", self.inner_attr_name)
        inner = getattr(wrapper, inner_attr_name)
        state_dict = torch.load(state_dict_path, map_location=device)
        inner.load_state_dict(state_dict)
        inner.to(device)
        inner.eval()
        self.model = inner

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return _normalize_predictions(self.model(features))


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
    model_dir = _resolve_model_dir(Path(model_root), model_name)
    return load_model_adapter_from_paths(
        config_path=model_dir / "model_config.json",
        state_dict_path=_resolve_state_dict_path(model_dir),
        device=resolved_device,
        model_name=model_name,
    )


def load_model_adapter_from_paths(
    *,
    config_path: str | Path,
    state_dict_path: str | Path,
    device: torch.device | str | None = None,
    model_name: str | None = None,
) -> nn.Module:
    resolved_device = _resolve_device(device)
    config = _load_json(Path(config_path))
    resolved_model_name = (model_name or config.get("model_name") or "").lower()
    if not resolved_model_name:
        raise ValueError("model_name must be provided explicitly or in config")
    adapter_cls = get_model_adapter_class(resolved_model_name)
    adapter = adapter_cls(
        state_dict_path=Path(state_dict_path),
        config=config,
        device=resolved_device,
    )
    return adapter.to(resolved_device)
