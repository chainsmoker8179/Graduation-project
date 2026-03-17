import json
from pathlib import Path

import torch

from whitebox_attack_models import get_model_adapter_class, load_model_adapter


def _write_model_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_temp_lstm_assets(root: Path) -> None:
    from qlib.contrib.model.pytorch_lstm_ts import LSTM

    wrapper = LSTM(d_feat=20, hidden_size=64, num_layers=2, dropout=0.0, n_jobs=1, GPU=-1)
    model_dir = root / "lstm" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.LSTM_model.state_dict(), model_dir / "state_dict.pt")
    _write_model_config(
        model_dir / "model_config.json",
        {
            "model_name": "lstm",
            "qlib_model_class": "LSTM",
            "qlib_model_module": "qlib.contrib.model.pytorch_lstm_ts",
            "model_kwargs": {
                "d_feat": 20,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "feature_spec": {"d_feat": 20, "step_len": 20},
        },
    )


def _build_temp_transformer_assets(root: Path) -> None:
    from qlib.contrib.model.pytorch_transformer_ts import TransformerModel

    wrapper = TransformerModel(d_feat=20, d_model=64, nhead=2, num_layers=2, dropout=0.0, n_jobs=1, GPU=-1)
    model_dir = root / "transformer" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.model.state_dict(), model_dir / "state_dict.pt")
    _write_model_config(
        model_dir / "model_config.json",
        {
            "model_name": "transformer",
            "qlib_model_class": "TransformerModel",
            "qlib_model_module": "qlib.contrib.model.pytorch_transformer_ts",
            "model_kwargs": {
                "d_feat": 20,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "feature_spec": {"d_feat": 20, "step_len": 20},
        },
    )


def _build_temp_tcn_assets(root: Path) -> None:
    from qlib.contrib.model.pytorch_tcn_ts import TCN

    wrapper = TCN(d_feat=20, n_chans=32, kernel_size=7, num_layers=5, dropout=0.5, n_jobs=1, GPU=-1)
    model_dir = root / "tcn" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.TCN_model.state_dict(), model_dir / "state_dict.pt")
    _write_model_config(
        model_dir / "model_config.json",
        {
            "model_name": "tcn",
            "qlib_model_class": "TCN",
            "qlib_model_module": "qlib.contrib.model.pytorch_tcn_ts",
            "model_kwargs": {
                "d_feat": 20,
                "n_chans": 32,
                "kernel_size": 7,
                "num_layers": 5,
                "dropout": 0.5,
            },
            "feature_spec": {"d_feat": 20, "step_len": 20},
        },
    )


def test_registry_supports_lstm_transformer_tcn() -> None:
    assert get_model_adapter_class("lstm") is not None
    assert get_model_adapter_class("transformer") is not None
    assert get_model_adapter_class("tcn") is not None


def test_load_model_adapter_returns_batch_scores(tmp_path: Path) -> None:
    _build_temp_lstm_assets(tmp_path)
    _build_temp_transformer_assets(tmp_path)
    _build_temp_tcn_assets(tmp_path)

    x = torch.randn(4, 20, 20)
    for model_name in ["lstm", "transformer", "tcn"]:
        adapter = load_model_adapter(model_name=model_name, model_root=tmp_path, device=torch.device("cpu"))
        y = adapter(x)
        assert y.shape == (4,)
        assert torch.isfinite(y).all()
