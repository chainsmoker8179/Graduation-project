# Transformer / TCN 重建探针 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本地 `qlib` 环境中验证 `Transformer` 与 `TCN` 能否基于当前导出资产稳定重建并完成 clean forward probe。

**Architecture:** 不直接进入统一攻击框架，而是先补齐两模型的 `model_config.json`，再通过一个小型 probe 模块完成“实例化 wrapper -> 提取 torch 子模块 -> 加载 state_dict -> 对标准化特征窗口做 clean forward -> 与参考 pred 对齐评估”。这一阶段只处理模型重建，不引入 `FGSM / PGD`。

**Tech Stack:** Python, PyTorch, Qlib, pandas, pytest, JSON

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-19-transformer-tcn-rebuild-probe-design.md`
  - 已批准设计文档
- Create: `origin_model_pred/Transformer/model/model_config.json`
  - Transformer 本地重建配置
- Create: `origin_model_pred/TCN/model/model_config.json`
  - TCN 本地重建配置
- Create: `whitebox_model_probe.py`
  - 模型重建与 clean forward probe 核心逻辑
- Create: `scripts/run_model_rebuild_probe.py`
  - CLI 入口
- Create: `tests/test_whitebox_model_probe.py`
  - probe 核心测试
- Create: `tests/test_run_model_rebuild_probe.py`
  - CLI 测试

> 注意：本计划不要求重命名 `origin_model_pred/` 下的二进制资产目录，也不要求提交大文件；仅新增 `model_config.json`。

## Chunk 1: 配置与资产探查

### Task 1: 明确 Transformer / TCN 可重建配置

**Files:**
- Create: `origin_model_pred/Transformer/model/model_config.json`
- Create: `origin_model_pred/TCN/model/model_config.json`

- [ ] **Step 1: 用 `trained_model.pkl` 探查 Transformer wrapper 类型与 torch 子模块属性**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python - <<'PY'\nimport pickle\nfrom pathlib import Path\np = Path('origin_model_pred/Transformer/model/transformer_trained_model.pkl')\nwith p.open('rb') as f:\n    m = pickle.load(f)\nprint(type(m))\nprint(sorted([k for k in dir(m) if k.endswith('_model') or k.endswith('_Model') or 'model' in k.lower()][:20]))\nprint(getattr(m, '__dict__', {}).keys())\nPY`
Expected: 能看到 wrapper 类名，以及类似 `Transformer_model` 的候选属性

- [ ] **Step 2: 用同样方式探查 TCN**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python - <<'PY'\nimport pickle\nfrom pathlib import Path\np = Path('origin_model_pred/TCN/model/tcn_trained_model.pkl')\nwith p.open('rb') as f:\n    m = pickle.load(f)\nprint(type(m))\nprint(sorted([k for k in dir(m) if k.endswith('_model') or k.endswith('_Model') or 'model' in k.lower()][:20]))\nprint(getattr(m, '__dict__', {}).keys())\nPY`
Expected: 能看到 wrapper 类名，以及底层 torch 子模块候选属性

- [ ] **Step 3: 写出 Transformer `model_config.json`**

至少包含：

```json
{
  "model_name": "transformer",
  "qlib_wrapper_module": "...",
  "qlib_wrapper_class": "...",
  "torch_submodule_attr": "...",
  "model_kwargs": {
    "d_feat": 20
  },
  "feature_spec": {
    "d_feat": 20,
    "step_len": 20
  }
}
```

- [ ] **Step 4: 写出 TCN `model_config.json`**

同样至少包含：

```json
{
  "model_name": "tcn",
  "qlib_wrapper_module": "...",
  "qlib_wrapper_class": "...",
  "torch_submodule_attr": "...",
  "model_kwargs": {
    "d_feat": 20
  },
  "feature_spec": {
    "d_feat": 20,
    "step_len": 20
  }
}
```

- [ ] **Step 5: 用最小 JSON 检查确认两份配置可解析**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\nfor p in [Path('origin_model_pred/Transformer/model/model_config.json'), Path('origin_model_pred/TCN/model/model_config.json')]:\n    payload = json.loads(p.read_text(encoding='utf-8'))\n    assert 'qlib_wrapper_module' in payload\n    assert 'qlib_wrapper_class' in payload\n    assert 'torch_submodule_attr' in payload\nprint('ok')\nPY`
Expected: 输出 `ok`

## Chunk 2: Probe 核心

### Task 2: 先写 probe 核心失败测试

**Files:**
- Create: `whitebox_model_probe.py`
- Create: `tests/test_whitebox_model_probe.py`

- [ ] **Step 1: 写配置加载与形状契约失败测试**

```python
from pathlib import Path

import torch

from whitebox_model_probe import load_probe_config, compute_probe_metrics


def test_compute_probe_metrics_returns_shape_and_finite_summary():
    pred = torch.tensor([0.1, 0.2, 0.3])
    ref = torch.tensor([0.1, 0.0, 0.4])
    out = compute_probe_metrics(pred, ref)
    assert out["sample_count"] == 3
    assert out["pred_finite_rate"] == 1.0
    assert "spearman_to_reference" in out
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_model_probe.py -q`
Expected: FAIL because `whitebox_model_probe.py` does not exist yet

- [ ] **Step 3: 实现最小 probe 核心**

至少实现：

```python
def load_probe_config(config_path: Path) -> dict: ...
def instantiate_wrapper_from_config(config: dict) -> object: ...
def extract_torch_submodule(wrapper: object, attr_name: str) -> torch.nn.Module: ...
def compute_probe_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]: ...
```

- [ ] **Step 4: 补写 state_dict 加载和前向测试**

```python
def test_extract_torch_submodule_loads_state_dict_for_dummy_wrapper():
    ...
```

- [ ] **Step 5: 运行 probe 核心测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_model_probe.py -q`
Expected: PASS

## Chunk 3: CLI 与真实资产 probe

### Task 3: 写 CLI 入口并在小样本资产上验证

**Files:**
- Create: `scripts/run_model_rebuild_probe.py`
- Create: `tests/test_run_model_rebuild_probe.py`

- [ ] **Step 1: 写 CLI 参数失败测试**

```python
from scripts.run_model_rebuild_probe import parse_args


def test_parse_args_accepts_explicit_model_asset_paths():
    args = parse_args(
        [
            "--model-name", "transformer",
            "--config-path", "origin_model_pred/Transformer/model/model_config.json",
            "--state-dict-path", "origin_model_pred/Transformer/model/transformer_state_dict.pt",
            "--asset-dir", "artifacts/lstm_attack_expanded_v7_512",
        ]
    )
    assert args.model_name == "transformer"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_run_model_rebuild_probe.py -q`
Expected: FAIL because script does not exist yet

- [ ] **Step 3: 实现 CLI**

CLI 至少支持：

```text
--model-name
--config-path
--state-dict-path
--asset-dir
--out-dir
--max-samples
--device
```

并完成：
- 读取 `matched_feature_windows.pt`
- 读取参考 `score`
- 跑 clean forward
- 写出 `probe_summary.json`
- 写出 `probe_predictions.csv`
- 写出 `README.md`

- [ ] **Step 4: 跑 CLI 测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_run_model_rebuild_probe.py -q`
Expected: PASS

- [ ] **Step 5: 为 Transformer 生成小样本 probe 资产**

如果现有 `artifacts/lstm_attack_expanded_v7_512` 不能直接复用模型自身参考分数，则重新导出：

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_lstm_attack_assets.py --pred-pkl origin_model_pred/Transformer/pred/pred.pkl --label-pkl origin_model_pred/Transformer/pred/label.pkl --out-dir artifacts/transformer_probe_assets --max-samples 128 --test-end-time 2025-10-31`
Expected: 生成 `matched_feature_windows.pt`、`matched_ohlcv_windows.pt` 与 `normalization_stats.json`

- [ ] **Step 6: 跑 Transformer probe**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_model_rebuild_probe.py --model-name transformer --config-path origin_model_pred/Transformer/model/model_config.json --state-dict-path origin_model_pred/Transformer/model/transformer_state_dict.pt --asset-dir artifacts/transformer_probe_assets --out-dir reports/transformer_model_probe --max-samples 64 --device cpu`
Expected: 成功输出 `probe_summary.json` 且 `pred_finite_rate = 1.0`

- [ ] **Step 7: 为 TCN 生成小样本 probe 资产**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_lstm_attack_assets.py --pred-pkl origin_model_pred/TCN/pred/pred.pkl --label-pkl origin_model_pred/TCN/pred/label.pkl --out-dir artifacts/tcn_probe_assets --max-samples 128 --test-end-time 2025-10-31`
Expected: 生成 `matched_feature_windows.pt`、`matched_ohlcv_windows.pt` 与 `normalization_stats.json`

- [ ] **Step 8: 跑 TCN probe**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_model_rebuild_probe.py --model-name tcn --config-path origin_model_pred/TCN/model/model_config.json --state-dict-path origin_model_pred/TCN/model/tcn_state_dict.pt --asset-dir artifacts/tcn_probe_assets --out-dir reports/tcn_model_probe --max-samples 64 --device cpu`
Expected: 成功输出 `probe_summary.json` 且 `pred_finite_rate = 1.0`

## Chunk 4: 收口与进入下一阶段

### Task 4: 形成结论并决定是否进入统一攻击框架

**Files:**
- Verify: `reports/transformer_model_probe/probe_summary.json`
- Verify: `reports/tcn_model_probe/probe_summary.json`
- Optional next: `docs/superpowers/specs/...` for unified attack core

- [ ] **Step 1: 运行 probe 相关测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_model_probe.py tests/test_run_model_rebuild_probe.py -q`
Expected: PASS

- [ ] **Step 2: 检查 Transformer probe 指标**

重点看：
- `pred_finite_rate == 1.0`
- `sample_count > 0`
- `spearman_to_reference > 0`

- [ ] **Step 3: 检查 TCN probe 指标**

重点看：
- `pred_finite_rate == 1.0`
- `sample_count > 0`
- `spearman_to_reference > 0`

- [ ] **Step 4: 只有在双模型 probe 都通过后，才进入下一步规划**

下一步才应切回旧的多模型总计划中的后续部分：
- 共享 `whitebox_attack_core.py`
- 多模型 adapter registry
- 统一 `run_whitebox_attack.py`
- `FGSM / PGD` 多模型 smoke
