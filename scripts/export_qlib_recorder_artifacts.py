#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STATE_DICT_CANDIDATES = ["LSTM_model", "Transformer_model", "GRU_model", "model"]
MODEL_ATTR_KEYS = ["d_feat", "hidden_size", "num_layers", "dropout", "loss", "device"]


def _copy_artifact(recorder: Any, artifact_name: str, destination: Path) -> Path:
    source_path = Path(recorder.download_artifact(artifact_name))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _extract_state_dict(model: Any) -> tuple[dict[str, torch.Tensor], str]:
    for attr_name in STATE_DICT_CANDIDATES:
        candidate = getattr(model, attr_name, None)
        if candidate is not None and hasattr(candidate, "state_dict"):
            return candidate.state_dict(), attr_name
    if hasattr(model, "state_dict"):
        return model.state_dict(), "self"
    raise ValueError("could not find a state_dict-capable model attribute")


def _collect_model_attrs(model: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in MODEL_ATTR_KEYS:
        value = getattr(model, key, None)
        if value is None:
            continue
        if isinstance(value, torch.device):
            attrs[key] = str(value)
        else:
            attrs[key] = value
    return attrs


def export_recorder_artifacts(
    *,
    recorder: Any,
    experiment_id: str,
    recorder_id: str,
    model_dir: str | Path,
    prediction_dir: str | Path,
) -> dict[str, str]:
    model_dir = Path(model_dir)
    prediction_dir = Path(prediction_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model_pickle_path = _copy_artifact(recorder, "trained_model", model_dir / "trained_model.pkl")
    pred_path = _copy_artifact(recorder, "pred.pkl", prediction_dir / "pred.pkl")
    label_path = _copy_artifact(recorder, "label.pkl", prediction_dir / "label.pkl")

    model = _load_pickle(model_pickle_path)
    state_dict, state_dict_source = _extract_state_dict(model)

    state_dict_path = model_dir / "trained_model_state_dict.pt"
    torch.save(state_dict, state_dict_path)

    model_config = {
        "model_class": type(model).__name__,
        "state_dict_source": state_dict_source,
        "model_attrs": _collect_model_attrs(model),
    }
    model_config_path = model_dir / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8")

    recorder_meta = {
        "experiment_id": experiment_id,
        "recorder_id": recorder_id,
        "params": recorder.list_params(),
        "metrics": recorder.list_metrics(),
        "tags": recorder.list_tags(),
        "artifacts": recorder.list_artifacts(),
    }
    recorder_meta_path = model_dir / "recorder_meta.json"
    recorder_meta_path.write_text(json.dumps(recorder_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model_pickle_path": str(model_pickle_path),
        "state_dict_path": str(state_dict_path),
        "model_config_path": str(model_config_path),
        "recorder_meta_path": str(recorder_meta_path),
        "pred_path": str(pred_path),
        "label_path": str(label_path),
    }


def _resolve_recorder(experiment_id: str, recorder_id: str) -> Any:
    from qlib.workflow import R

    experiment = R.get_exp(experiment_id=experiment_id)
    return experiment.get_recorder(recorder_id=recorder_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model and prediction artifacts from a Qlib recorder.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--recorder-id", required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("prediction"))
    args = parser.parse_args()

    recorder = _resolve_recorder(args.experiment_id, args.recorder_id)
    summary = export_recorder_artifacts(
        recorder=recorder,
        experiment_id=args.experiment_id,
        recorder_id=args.recorder_id,
        model_dir=args.model_dir,
        prediction_dir=args.prediction_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
