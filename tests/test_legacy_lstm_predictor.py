from pathlib import Path

import torch

from legacy_lstm_predictor import LegacyLSTMPredictor, load_legacy_lstm_from_files


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"


def test_legacy_lstm_predictor_output_shape():
    model = LegacyLSTMPredictor(d_feat=20, hidden_size=64, num_layers=2, dropout=0.0)
    x = torch.randn(4, 20, 20)
    y = model(x)
    assert y.shape == (4,)


def test_legacy_lstm_predictor_loads_exported_state_dict():
    state_dict_path = MODEL_DIR / "lstm_state_dict.pt"
    config_path = MODEL_DIR / "lstm_state_dict_config.json"

    model = load_legacy_lstm_from_files(config_path=config_path, state_dict_path=state_dict_path)

    exported_state = torch.load(state_dict_path, map_location="cpu")
    loaded_state = model.state_dict()

    assert set(loaded_state.keys()) == set(exported_state.keys())
    for key, value in exported_state.items():
        assert torch.equal(loaded_state[key], value), key
