import json
import sys
import types
from pathlib import Path

import torch


def _install_dummy_probe_module(module_name: str = "dummy_probe_module") -> str:
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


def test_load_probe_config_reads_json(tmp_path: Path) -> None:
    from whitebox_model_probe import load_probe_config

    config_path = tmp_path / "model_config.json"
    config_path.write_text(json.dumps({"model_name": "dummy"}), encoding="utf-8")

    out = load_probe_config(config_path)

    assert out["model_name"] == "dummy"


def test_compute_probe_metrics_returns_shape_and_finite_summary() -> None:
    from whitebox_model_probe import compute_probe_metrics

    pred = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    ref = torch.tensor([0.1, 0.1, 0.4], dtype=torch.float32)

    out = compute_probe_metrics(pred, ref)

    assert out["sample_count"] == 3
    assert out["pred_finite_rate"] == 1.0
    assert out["mae_to_reference"] > 0.0
    assert out["mse_to_reference"] > 0.0
    assert out["spearman_to_reference"] > 0.0


def test_load_feature_model_from_config_restores_state_dict_for_dummy_wrapper(tmp_path: Path) -> None:
    from whitebox_model_probe import load_feature_model_from_config

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
        wrapper.model.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
    state_dict_path = tmp_path / "dummy_state_dict.pt"
    torch.save(wrapper.model.state_dict(), state_dict_path)

    model = load_feature_model_from_config(
        config_path=config_path,
        state_dict_path=state_dict_path,
        device=torch.device("cpu"),
    )
    x = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    pred = model(x)

    assert pred.shape == (2,)
    assert torch.allclose(pred, torch.tensor([14.0, 4.0], dtype=torch.float32))
