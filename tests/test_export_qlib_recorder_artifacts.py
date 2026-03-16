import json
import pickle
from pathlib import Path

import pandas as pd
import torch

from scripts.export_qlib_recorder_artifacts import export_recorder_artifacts


class FakeWrappedModel:
    def __init__(self):
        self.d_feat = 20
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.0
        self.device = torch.device("cpu")
        self.loss = "mse"
        self.LSTM_model = torch.nn.Linear(3, 1)


class FakeRecorder:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir

    def download_artifact(self, path: str, dst_path: str | None = None) -> str:
        candidate = self.artifact_dir / path
        if not candidate.exists():
            raise FileNotFoundError(path)
        return str(candidate)

    def list_params(self):
        return {
            "model.class": "qlib.contrib.model.pytorch_lstm_ts.LSTM",
            "model.kwargs.d_feat": "20",
        }

    def list_metrics(self):
        return {"IC": 0.123}

    def list_tags(self):
        return {"hostname": "unit-test"}

    def list_artifacts(self, artifact_path: str | None = None):
        return ["trained_model", "pred.pkl", "label.pkl"]


def test_export_recorder_artifacts_writes_model_prediction_and_state_dict(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    trained_model = FakeWrappedModel()
    pred = pd.DataFrame({"score": [0.1, 0.2]})
    label = pd.DataFrame({"label": [0.3, 0.4]})

    with (artifact_dir / "trained_model").open("wb") as f:
        pickle.dump(trained_model, f)
    with (artifact_dir / "pred.pkl").open("wb") as f:
        pickle.dump(pred, f)
    with (artifact_dir / "label.pkl").open("wb") as f:
        pickle.dump(label, f)

    summary = export_recorder_artifacts(
        recorder=FakeRecorder(artifact_dir),
        experiment_id="exp-1",
        recorder_id="rec-1",
        model_dir=tmp_path / "model",
        prediction_dir=tmp_path / "prediction",
    )

    model_pkl_path = tmp_path / "model" / "trained_model.pkl"
    state_dict_path = tmp_path / "model" / "trained_model_state_dict.pt"
    model_config_path = tmp_path / "model" / "model_config.json"
    recorder_meta_path = tmp_path / "model" / "recorder_meta.json"
    pred_path = tmp_path / "prediction" / "pred.pkl"
    label_path = tmp_path / "prediction" / "label.pkl"

    assert model_pkl_path.exists()
    assert state_dict_path.exists()
    assert model_config_path.exists()
    assert recorder_meta_path.exists()
    assert pred_path.exists()
    assert label_path.exists()

    exported_state_dict = torch.load(state_dict_path, map_location="cpu")
    expected_state_dict = trained_model.LSTM_model.state_dict()
    assert set(exported_state_dict.keys()) == set(expected_state_dict.keys())
    for key, value in expected_state_dict.items():
        assert torch.equal(exported_state_dict[key], value)

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    assert model_config["model_class"] == "FakeWrappedModel"
    assert model_config["state_dict_source"] == "LSTM_model"
    assert model_config["model_attrs"]["d_feat"] == 20

    recorder_meta = json.loads(recorder_meta_path.read_text(encoding="utf-8"))
    assert recorder_meta["experiment_id"] == "exp-1"
    assert recorder_meta["recorder_id"] == "rec-1"
    assert recorder_meta["params"]["model.class"] == "qlib.contrib.model.pytorch_lstm_ts.LSTM"

    loaded_pred = pd.read_pickle(pred_path)
    loaded_label = pd.read_pickle(label_path)
    pd.testing.assert_frame_equal(loaded_pred, pred)
    pd.testing.assert_frame_equal(loaded_label, label)

    assert summary["model_pickle_path"] == str(model_pkl_path)
    assert summary["state_dict_path"] == str(state_dict_path)
    assert summary["pred_path"] == str(pred_path)
