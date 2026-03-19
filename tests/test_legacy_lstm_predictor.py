import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_lstm_predictor import LegacyLSTMPredictor, load_legacy_lstm_from_files


def test_load_legacy_lstm_from_files_accepts_nested_model_kwargs_config(tmp_path) -> None:
    config_path = tmp_path / "model_config.json"
    state_dict_path = tmp_path / "state_dict.pt"

    config_path.write_text(
        json.dumps(
            {
                "model_name": "lstm",
                "model_kwargs": {
                    "d_feat": 20,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                },
                "feature_spec": {
                    "d_feat": 20,
                    "step_len": 20,
                },
            }
        ),
        encoding="utf-8",
    )

    model = LegacyLSTMPredictor(d_feat=20, hidden_size=64, num_layers=2, dropout=0.0)
    torch.save(model.state_dict(), state_dict_path)

    loaded = load_legacy_lstm_from_files(config_path=config_path, state_dict_path=state_dict_path)

    assert isinstance(loaded, LegacyLSTMPredictor)
    assert loaded.d_feat == 20
    assert loaded.hidden_size == 64
    assert loaded.num_layers == 2
    assert loaded.dropout == 0.0
