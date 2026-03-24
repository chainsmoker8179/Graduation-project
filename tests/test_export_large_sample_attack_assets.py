import json
from pathlib import Path

import pandas as pd
import torch

from scripts.export_lstm_attack_assets import build_matched_reference as legacy_build_matched_reference
from scripts.export_large_sample_attack_assets import (
    build_export_summary,
    build_feature_windows_from_raw,
    main,
)
from scripts.export_whitebox_attack_assets import build_matched_reference as compat_build_matched_reference


def _build_sample_asset() -> dict:
    x = torch.linspace(1.0, 2.0 * 80 * 5, steps=2 * 80 * 5, dtype=torch.float32).reshape(2, 80, 5)
    return {
        "keys": [("2025-01-02 00:00:00", "AAA"), ("2025-01-03 00:00:00", "BBB")],
        "ohlcv": x,
        "label": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "score": torch.tensor([0.3, 0.4], dtype=torch.float32),
        "missing_keys": [("2025-01-04 00:00:00", "CCC")],
    }


def _build_stats() -> dict:
    return {
        "feature_names": [f"F{i}" for i in range(20)],
        "center": [0.0] * 20,
        "scale": [1.0] * 20,
        "clip_outlier": False,
    }


def test_build_feature_windows_from_raw_preserves_keys_and_marks_source() -> None:
    sample_asset = _build_sample_asset()

    feature_asset = build_feature_windows_from_raw(sample_asset, normalization_stats=_build_stats())

    assert feature_asset["keys"] == sample_asset["keys"]
    assert feature_asset["features"].shape == (2, 20, 20)
    assert torch.isfinite(feature_asset["features"]).all()
    assert feature_asset["missing_keys"] == sample_asset["missing_keys"]
    assert feature_asset["feature_source"] == "torch_bridge_from_raw"


def test_build_export_summary_contains_fast_export_fields() -> None:
    sample_asset = _build_sample_asset()
    feature_asset = build_feature_windows_from_raw(sample_asset, normalization_stats=_build_stats())

    summary = build_export_summary(
        matched_reference_rows=2,
        sample_asset=sample_asset,
        feature_asset=feature_asset,
        normalization_stats_source=Path("/tmp/stats.json"),
    )

    assert summary["matched_reference_rows"] == 2
    assert summary["exported_sample_rows"] == 2
    assert summary["exported_feature_rows"] == 2
    assert summary["feature_source"] == "torch_bridge_from_raw"
    assert summary["normalization_stats_source"] == "/tmp/stats.json"


def test_main_writes_fast_export_contract(tmp_path: Path, monkeypatch) -> None:
    pred_index = pd.MultiIndex.from_tuples(
        [("2025-01-02", "AAA"), ("2025-01-03", "BBB")],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [0.3, 0.4]}, index=pred_index)
    label = pd.DataFrame({"label": [0.1, 0.2]}, index=pred_index)
    pred_path = tmp_path / "pred.pkl"
    label_path = tmp_path / "label.pkl"
    pred.to_pickle(pred_path)
    label.to_pickle(label_path)

    stats = _build_stats()
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.export_large_sample_attack_assets._build_raw_test_split",
        lambda args: object(),
    )
    monkeypatch.setattr(
        "scripts.export_large_sample_attack_assets.export_matched_raw_windows",
        lambda matched_reference, raw_test_split: _build_sample_asset(),
    )

    out_dir = tmp_path / "out"
    main(
        [
            "--pred-pkl",
            str(pred_path),
            "--label-pkl",
            str(label_path),
            "--out-dir",
            str(out_dir),
            "--normalization-stats",
            str(stats_path),
            "--max-samples",
            "2",
        ]
    )

    assert (out_dir / "matched_reference.csv").exists()
    assert (out_dir / "matched_ohlcv_windows.pt").exists()
    assert (out_dir / "matched_feature_windows.pt").exists()
    assert (out_dir / "normalization_stats.json").exists()
    assert (out_dir / "export_summary.json").exists()

    written_stats = json.loads((out_dir / "normalization_stats.json").read_text(encoding="utf-8"))
    assert written_stats == stats

    summary = json.loads((out_dir / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_source"] == "torch_bridge_from_raw"
    assert summary["normalization_stats_source"] == str(stats_path)


def test_export_whitebox_attack_assets_reuses_legacy_builder() -> None:
    pred_index = pd.MultiIndex.from_tuples(
        [("2025-01-02", "AAA"), ("2025-01-02", "BBB")],
        names=["datetime", "instrument"],
    )
    label_index = pd.MultiIndex.from_tuples(
        [("2025-01-02", "BBB"), ("2025-01-03", "CCC")],
        names=["datetime", "instrument"],
    )
    pred = pd.DataFrame({"score": [0.1, 0.2]}, index=pred_index)
    label = pd.DataFrame({"label": [1.2, 1.3]}, index=label_index)

    expected = legacy_build_matched_reference(pred, label)
    actual = compat_build_matched_reference(pred, label)

    pd.testing.assert_frame_equal(actual, expected)
