import json
from pathlib import Path

import pandas as pd
import torch

from legacy_lstm_feature_bridge import LegacyLSTMFeatureBridge
from legacy_lstm_preprocess import FillnaLayer, RobustZScoreNormLayer


class DummyFeatureModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, -1, :].sum(dim=-1)


def _build_dummy_asset_dir(tmp_path: Path) -> Path:
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()

    x = torch.linspace(1.0, 400.0, steps=2 * 80 * 5, dtype=torch.float32).reshape(2, 80, 5)
    keys = [("2025-01-02 00:00:00", "AAA"), ("2025-01-03 00:00:00", "BBB")]
    torch.save({"keys": keys, "ohlcv": x}, asset_dir / "matched_ohlcv_windows.pt")

    bridge = LegacyLSTMFeatureBridge()
    norm = RobustZScoreNormLayer(center=torch.zeros(20), scale=torch.ones(20), clip_outlier=False)
    fillna = FillnaLayer()
    with torch.no_grad():
        features = fillna(norm(bridge(x)))
        scores = features[:, -1, :].sum(dim=-1)
    torch.save({"keys": keys, "features": features, "missing_keys": []}, asset_dir / "matched_feature_windows.pt")
    pd.DataFrame(
        {
            "datetime": [k[0] for k in keys],
            "instrument": [k[1] for k in keys],
            "score": scores.tolist(),
            "label": [0.0, 0.0],
        }
    ).to_csv(asset_dir / "matched_reference.csv", index=False)
    (asset_dir / "normalization_stats.json").write_text(
        json.dumps(
            {
                "feature_names": bridge.feature_names,
                "center": [0.0] * 20,
                "scale": [1.0] * 20,
                "clip_outlier": False,
            }
        ),
        encoding="utf-8",
    )
    return asset_dir


def test_parse_args_accepts_model_name_and_paths() -> None:
    from scripts.run_whitebox_attack import parse_args

    args = parse_args(
        [
            "--model-name",
            "transformer",
            "--config-path",
            "origin_model_pred/Transformer/model/model_config.json",
            "--state-dict-path",
            "/tmp/model.pt",
            "--asset-dir",
            "/tmp/asset",
            "--out-dir",
            "/tmp/out",
        ]
    )

    assert args.model_name == "transformer"
    assert args.asset_dir == Path("/tmp/asset")


def test_main_writes_attack_outputs_for_dummy_assets(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_whitebox_attack

    asset_dir = _build_dummy_asset_dir(tmp_path)
    out_dir = tmp_path / "report"

    monkeypatch.setattr(run_whitebox_attack, "load_model_adapter_from_paths", lambda **_: DummyFeatureModel())

    run_whitebox_attack.main(
        [
            "--model-name",
            "dummy",
            "--config-path",
            str(tmp_path / "dummy_config.json"),
            "--state-dict-path",
            str(tmp_path / "dummy_state_dict.pt"),
            "--asset-dir",
            str(asset_dir),
            "--out-dir",
            str(out_dir),
            "--max-samples",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert (out_dir / "attack_summary.json").exists()
    assert (out_dir / "sample_metrics.csv").exists()
    assert (out_dir / "attack_report.md").exists()

    summary = json.loads((out_dir / "attack_summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 2
    assert summary["clean_gate"]["feature_mae_to_reference"] == 0.0
    assert "fgsm_loss" in summary
    assert "pgd_loss" in summary
