from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def _canonicalize_key(datetime_value: Any, instrument_value: Any) -> tuple[str, str]:
    return str(pd.Timestamp(datetime_value)), str(instrument_value)


def load_probe_config(config_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def instantiate_wrapper_from_config(config: dict[str, Any]) -> object:
    module = importlib.import_module(config["qlib_wrapper_module"])
    wrapper_class = getattr(module, config["qlib_wrapper_class"])
    model_kwargs = dict(config.get("model_kwargs", {}))
    signature = inspect.signature(wrapper_class.__init__)
    if "GPU" in signature.parameters and "GPU" not in model_kwargs:
        model_kwargs["GPU"] = -1
    return wrapper_class(**model_kwargs)


def extract_torch_submodule(wrapper: object, attr_name: str) -> torch.nn.Module:
    submodule = getattr(wrapper, attr_name, None)
    if submodule is None or not isinstance(submodule, torch.nn.Module):
        raise AttributeError(f"wrapper does not expose torch.nn.Module attr '{attr_name}'")
    return submodule


def _normalize_predictions(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    return pred.reshape(-1)


def _forward_with_shape_fallbacks(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    candidates = [
        features,
        features.transpose(1, 2),
        features.reshape(features.shape[0], -1),
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            pred = model(candidate)
            return _normalize_predictions(pred)
        except Exception as exc:  # pragma: no cover - exercised on real models, not dummy tests
            last_error = exc
    assert last_error is not None
    raise last_error


def load_feature_model_from_config(
    *,
    config_path: str | Path,
    state_dict_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    config = load_probe_config(config_path)
    wrapper = instantiate_wrapper_from_config(config)
    model = extract_torch_submodule(wrapper, config["torch_submodule_attr"])
    state_dict = torch.load(Path(state_dict_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_probe_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu().reshape(-1).float()
    ref = ref.detach().cpu().reshape(-1).float()
    finite_mask = torch.isfinite(pred)
    finite_rate = float(finite_mask.float().mean().item()) if pred.numel() > 0 else 0.0
    valid_pred = pred[finite_mask]
    valid_ref = ref[finite_mask]

    pred_series = pd.Series(valid_pred.numpy())
    ref_series = pd.Series(valid_ref.numpy())
    spearman = pred_series.corr(ref_series, method="spearman")
    if pd.isna(spearman):
        spearman = 0.0

    abs_err = (valid_pred - valid_ref).abs()
    sq_err = (valid_pred - valid_ref).square()
    return {
        "sample_count": int(pred.numel()),
        "pred_finite_rate": finite_rate,
        "pred_mean": float(valid_pred.mean().item()) if valid_pred.numel() else 0.0,
        "pred_std": float(valid_pred.std(unbiased=False).item()) if valid_pred.numel() else 0.0,
        "mae_to_reference": float(abs_err.mean().item()) if abs_err.numel() else 0.0,
        "mse_to_reference": float(sq_err.mean().item()) if sq_err.numel() else 0.0,
        "spearman_to_reference": float(spearman),
    }


def load_probe_asset(
    asset_dir: str | Path,
    *,
    max_samples: int | None = None,
) -> tuple[list[tuple[str, str]], torch.Tensor, torch.Tensor]:
    asset_dir = Path(asset_dir)
    feature_asset = torch.load(asset_dir / "matched_feature_windows.pt", map_location="cpu")
    reference_df = pd.read_csv(asset_dir / "matched_reference.csv")
    ref_lookup = {
        _canonicalize_key(row["datetime"], row["instrument"]): float(row["score"])
        for _, row in reference_df.iterrows()
    }

    keys: list[tuple[str, str]] = []
    ref_scores: list[float] = []
    keep_indices: list[int] = []
    for idx, raw_key in enumerate(feature_asset["keys"]):
        key = _canonicalize_key(raw_key[0], raw_key[1])
        if key in ref_lookup:
            keep_indices.append(idx)
            keys.append(key)
            ref_scores.append(ref_lookup[key])
        if max_samples is not None and len(keys) >= max_samples:
            break

    if not keys:
        raise ValueError(f"no matched keys with reference scores under {asset_dir}")

    features = feature_asset["features"][keep_indices].float()
    reference_scores = torch.tensor(ref_scores, dtype=torch.float32)
    return keys, features, reference_scores


def run_clean_forward_probe(
    *,
    model: torch.nn.Module,
    keys: list[tuple[str, str]],
    feature_windows: torch.Tensor,
    reference_scores: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    feature_windows = feature_windows.to(device)
    with torch.no_grad():
        pred = _forward_with_shape_fallbacks(model, feature_windows).detach().cpu()
    summary = compute_probe_metrics(pred, reference_scores)

    rows = []
    for i, key in enumerate(keys):
        rows.append(
            {
                "datetime": key[0],
                "instrument": key[1],
                "reference_score": float(reference_scores[i].item()),
                "pred_score": float(pred[i].item()),
                "abs_error": float(abs(pred[i].item() - reference_scores[i].item())),
            }
        )
    return summary, pd.DataFrame(rows)
