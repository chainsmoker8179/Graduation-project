import json
import sys
import types
from pathlib import Path

import pandas as pd
import torch


def _install_dummy_probe_module(module_name: str = "dummy_probe_cli_module") -> str:
    module = types.ModuleType(module_name)

    class DummyInner(torch.nn.Module):
        def __init__(self, d_feat: int = 3) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(d_feat, 1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x[:, -1, :]).squeeze(-1)

    class DummyWrapper:
        def __init__(self, d_feat: int = 3) -> None:
            self.model = DummyInner(d_feat=d_feat)

    module.DummyWrapper = DummyWrapper
    sys.modules[module_name] = module
    return module_name


def test_parse_args_accepts_explicit_model_asset_paths() -> None:
    from scripts.run_model_rebuild_probe import parse_args

    args = parse_args(
        [
            "--model-name",
            "transformer",
            "--config-path",
            "origin_model_pred/Transformer/model/model_config.json",
            "--state-dict-path",
            "/home/chainsmoker/qlib_test/origin_model_pred/Transformer/model/transformer_state_dict.pt",
            "--asset-dir",
            "artifacts/transformer_probe_assets",
        ]
    )

    assert args.model_name == "transformer"
    assert args.asset_dir == Path("artifacts/transformer_probe_assets")


def test_main_writes_probe_outputs_for_dummy_assets(tmp_path: Path) -> None:
    from scripts.run_model_rebuild_probe import main

    module_name = _install_dummy_probe_module()

    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "qlib_wrapper_module": module_name,
                "qlib_wrapper_class": "DummyWrapper",
                "torch_submodule_attr": "model",
                "model_kwargs": {"d_feat": 3},
                "feature_spec": {"d_feat": 3, "step_len": 4},
            }
        ),
        encoding="utf-8",
    )

    module = sys.modules[module_name]
    wrapper = module.DummyWrapper(d_feat=3)
    with torch.no_grad():
        wrapper.model.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))
    state_dict_path = tmp_path / "dummy_state_dict.pt"
    torch.save(wrapper.model.state_dict(), state_dict_path)

    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    feature_asset = {
        "keys": [("2025-01-02 00:00:00", "AAA"), ("2025-01-03 00:00:00", "BBB")],
        "features": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0], [0.1, 2.0, 3.0]],
                [[0.0, 0.0, 0.0], [0.4, 0.0, 1.0], [0.0, 0.0, 0.0], [0.3, 1.0, 2.0]],
            ],
            dtype=torch.float32,
        ),
        "missing_keys": [],
    }
    torch.save(feature_asset, asset_dir / "matched_feature_windows.pt")
    pd.DataFrame(
        {
            "datetime": ["2025-01-02 00:00:00", "2025-01-03 00:00:00"],
            "instrument": ["AAA", "BBB"],
            "score": [0.1, 0.3],
            "label": [0.0, 0.0],
        }
    ).to_csv(asset_dir / "matched_reference.csv", index=False)

    out_dir = tmp_path / "report"
    main(
        [
            "--model-name",
            "dummy",
            "--config-path",
            str(config_path),
            "--state-dict-path",
            str(state_dict_path),
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

    assert (out_dir / "probe_summary.json").exists()
    assert (out_dir / "probe_predictions.csv").exists()
    assert (out_dir / "README.md").exists()

    summary = json.loads((out_dir / "probe_summary.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 2
    assert summary["pred_finite_rate"] == 1.0
