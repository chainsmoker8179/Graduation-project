# 多模型样本级白盒攻击迁移 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留现有 LSTM 攻击结论与攻击算法口径不变的前提下，把样本级白盒攻击统一迁移到 `lstm`、`transformer` 和 `tcn` 三个旧基线模型。

**Architecture:** 先统一本地模型资产契约，并为三模型补齐结构化 `model_config.json`；随后把现有 LSTM 专用攻击主干抽成共享攻击骨架，再通过模型适配器封装 `LSTM`、`Transformer`、`TCN` 的特征到分数前向。统一入口脚本按 `--model-name` 路由模型，统一复用 clean gate、`FGSM` / `PGD`、预算投影和报告输出。

**Tech Stack:** Python、PyTorch、Qlib 原生模型类、现有 `legacy_lstm_*` 模块、pytest、pandas、JSON 配置文件、仓库内本地模型资产。

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-17-multi-model-whitebox-attack-design.md`
  - 已批准设计文档
- Create: `origin_model_pred/lstm/model/model_config.json`
  - LSTM 统一模型元数据
- Create: `origin_model_pred/transformer/model/model_config.json`
  - Transformer 统一模型元数据
- Create: `origin_model_pred/tcn/model/model_config.json`
  - TCN 统一模型元数据
- Create: `whitebox_attack_core.py`
  - 共享 attack pipeline、clean gate、FGSM / PGD、预算与投影逻辑
- Create: `whitebox_attack_models.py`
  - 三模型 adapter、registry、`model_config.json` 解析与加载入口
- Modify: `legacy_lstm_attack_core.py`
  - 视情况改为轻量兼容层，复用共享核心逻辑
- Create: `scripts/export_whitebox_attack_assets.py`
  - 统一模型无关的样本级资产导出入口
- Modify: `scripts/export_lstm_attack_assets.py`
  - 变成兼容壳层或复用统一导出逻辑
- Create: `scripts/run_whitebox_attack.py`
  - 统一样本级攻击入口，按 `--model-name` 路由
- Modify: `scripts/run_lstm_whitebox_attack.py`
  - 变成兼容壳层，内部转调统一入口
- Create: `tests/test_whitebox_attack_core.py`
  - 共享核心与预算投影测试
- Create: `tests/test_whitebox_attack_models.py`
  - adapter 与 registry 测试
- Create: `tests/test_run_whitebox_attack_smoke.py`
  - 统一入口的 mock smoke test
- Modify: `tests/test_export_lstm_attack_assets.py`
  - 确认旧导出入口仍可用，或转向验证统一导出入口

> 注意：`origin_model_pred/` 下的二进制模型与预测文件属于本地实验资产，不纳入 git commit；仅新增或修改其中的 `model_config.json`。

## Chunk 1: 统一资产契约

### Task 1: 规范化 `origin_model_pred/` 目录结构

**Files:**
- Modify: `origin_model_pred/LSTM/...`
- Modify: `origin_model_pred/Transformer/...`
- Modify: `origin_model_pred/TCN/...`

- [ ] **Step 1: 将模型目录名统一改成小写**

目标目录应变成：

```text
origin_model_pred/lstm
origin_model_pred/transformer
origin_model_pred/tcn
```

- [ ] **Step 2: 将预测目录名统一改成 `prediction/`**

要求三模型都满足：

```text
origin_model_pred/<model>/prediction/pred.pkl
origin_model_pred/<model>/prediction/label.pkl
```

- [ ] **Step 3: 将模型文件名统一改成固定契约**

要求三模型都满足：

```text
origin_model_pred/<model>/model/state_dict.pt
origin_model_pred/<model>/model/trained_model.pkl
```

- [ ] **Step 4: 检查目录结构是否全部符合约定**

Run: `find origin_model_pred -maxdepth 3 -type f | sort`
Expected: 只看到三套小写模型目录，且每套都包含 `state_dict.pt`、`trained_model.pkl`、`prediction/pred.pkl`、`prediction/label.pkl`

- [ ] **Step 5: 不提交二进制资产**

不要执行 `git add origin_model_pred/*/model/state_dict.pt` 或 `git add origin_model_pred/*/prediction/*.pkl`。这些文件保持本地存在即可。

### Task 2: 为三模型补齐 `model_config.json`

**Files:**
- Create: `origin_model_pred/lstm/model/model_config.json`
- Create: `origin_model_pred/transformer/model/model_config.json`
- Create: `origin_model_pred/tcn/model/model_config.json`

- [ ] **Step 1: 先写出 LSTM 配置文件**

内容至少包含：

```json
{
  "model_name": "lstm",
  "qlib_model_class": "LSTM",
  "qlib_model_module": "qlib.contrib.model.pytorch_lstm_ts",
  "model_kwargs": {
    "d_feat": 20,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.0
  },
  "feature_spec": {
    "d_feat": 20,
    "step_len": 20
  }
}
```

- [ ] **Step 2: 写出 Transformer 配置文件**

优先从旧配置与可用资产中补齐：`d_feat`、`d_model`、`nhead`、`num_layers`、`dropout`、`batch_size`、`seed` 等必需字段。

- [ ] **Step 3: 写出 TCN 配置文件**

至少包含：`d_feat`、`n_chans`、`kernel_size`、`num_layers`、`dropout`、`batch_size`、`seed`。

- [ ] **Step 4: 用最小 JSON 读取脚本验证三份配置都合法**

Run: `python -c "import json; from pathlib import Path; [json.loads(Path(p).read_text()) for p in ['origin_model_pred/lstm/model/model_config.json','origin_model_pred/transformer/model/model_config.json','origin_model_pred/tcn/model/model_config.json']]; print('ok')"`
Expected: 输出 `ok`

- [ ] **Step 5: Commit 仅配置文件**

```bash
git add origin_model_pred/lstm/model/model_config.json origin_model_pred/transformer/model/model_config.json origin_model_pred/tcn/model/model_config.json
git commit -m "chore: add normalized model configs for whitebox attacks"
```

## Chunk 2: 共享攻击骨架

### Task 3: 抽取共享 attack core

**Files:**
- Create: `whitebox_attack_core.py`
- Modify: `legacy_lstm_attack_core.py`
- Test: `tests/test_whitebox_attack_core.py`

- [ ] **Step 1: 先写共享 pipeline 的失败测试**

```python
import torch
import torch.nn as nn

from whitebox_attack_core import RawFeatureAttackPipeline


class DummyBridge(nn.Module):
    def forward(self, x):
        return x[..., :2]


class DummyNorm(nn.Module):
    def forward(self, x):
        return x


class DummyFillna(nn.Module):
    def forward(self, x):
        return x


class DummyAdapter(nn.Module):
    def forward(self, features):
        return features[:, -1, :].sum(dim=-1)


def test_pipeline_backward_reaches_raw_ohlcv():
    pipe = RawFeatureAttackPipeline(
        bridge=DummyBridge(),
        norm=DummyNorm(),
        fillna=DummyFillna(),
        model=DummyAdapter(),
    )
    x = torch.randn(4, 20, 5, requires_grad=True)
    y = pipe(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
```

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_attack_core.py::test_pipeline_backward_reaches_raw_ohlcv -v`
Expected: FAIL，提示 `whitebox_attack_core` 或 `RawFeatureAttackPipeline` 不存在

- [ ] **Step 3: 在 `whitebox_attack_core.py` 中实现最小共享 pipeline**

实现：
- `RawFeatureAttackPipeline`
- `forward_features`
- `forward_from_features`
- `forward`

- [ ] **Step 4: 补写预算与攻击失败测试**

```python
def test_project_relative_box_keeps_values_within_budget():
    ...


def test_fgsm_uses_shared_pipeline_and_preserves_shape():
    ...
```

- [ ] **Step 5: 运行共享 core 全部测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_attack_core.py -v`
Expected: PASS

- [ ] **Step 6: 把现有 `legacy_lstm_attack_core.py` 改成复用共享实现**

要求：
- 旧接口尽量保留
- 不影响后续 `partial attack backtest` 现有 import

- [ ] **Step 7: 重新运行旧 LSTM core 测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_legacy_lstm_attack_core.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add whitebox_attack_core.py legacy_lstm_attack_core.py tests/test_whitebox_attack_core.py
git commit -m "refactor: extract shared whitebox attack core"
```

### Task 4: 实现三模型 adapter 与 registry

**Files:**
- Create: `whitebox_attack_models.py`
- Test: `tests/test_whitebox_attack_models.py`

- [ ] **Step 1: 先写 registry 失败测试**

```python
from whitebox_attack_models import get_model_adapter_class


def test_registry_supports_lstm_transformer_tcn():
    assert get_model_adapter_class("lstm") is not None
    assert get_model_adapter_class("transformer") is not None
    assert get_model_adapter_class("tcn") is not None
```

- [ ] **Step 2: 运行测试，确认它先失败**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_attack_models.py::test_registry_supports_lstm_transformer_tcn -v`
Expected: FAIL

- [ ] **Step 3: 实现 `LSTMAdapter`**

要求：
- 复用 `legacy_lstm_predictor.py`
- 读取 `origin_model_pred/lstm/model/model_config.json`
- 读取 `state_dict.pt`

- [ ] **Step 4: 为 `TransformerAdapter` 和 `TCNAdapter` 写形状契约测试**

```python
def test_adapter_forward_returns_batch_scores():
    ...
```

测试要求：
- 输入形状 `(batch, 20, 20)`
- 输出形状 `(batch,)`

- [ ] **Step 5: 实现 `TransformerAdapter` 和 `TCNAdapter`**

要求：
- 从 `model_config.json` 读取 Qlib 类路径与参数
- 在 `qlib` 环境中实例化 wrapper
- 提取真正的 torch 子模块
- 加载 `state_dict.pt`

- [ ] **Step 6: 实现统一加载入口**

至少提供：
- `load_model_adapter(model_name, model_root, device)`
- `get_model_adapter_class(model_name)`

- [ ] **Step 7: 运行 adapter 全部测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_attack_models.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add whitebox_attack_models.py tests/test_whitebox_attack_models.py
git commit -m "feat: add multi-model whitebox attack adapters"
```

## Chunk 3: 统一导出与统一运行入口

### Task 5: 抽出统一资产导出入口

**Files:**
- Create: `scripts/export_whitebox_attack_assets.py`
- Modify: `scripts/export_lstm_attack_assets.py`
- Modify: `tests/test_export_lstm_attack_assets.py`

- [ ] **Step 1: 先写帮助输出检查**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_whitebox_attack_assets.py --help`
Expected: 正常打印 usage，包含 `--pred-pkl`、`--label-pkl`、`--out-dir`

- [ ] **Step 2: 在新脚本中复用现有导出逻辑**

要求：
- 输出结构与 LSTM 版本保持兼容
- 不在导出逻辑里写死模型名

- [ ] **Step 3: 将旧 `export_lstm_attack_assets.py` 改成兼容壳层**

要求：
- 保持旧命令仍可使用
- 内部转调统一实现

- [ ] **Step 4: 更新或补充测试**

至少覆盖：
- 匹配 reference 的构建
- 旧入口仍可导入核心 helper

- [ ] **Step 5: 运行导出相关测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_export_lstm_attack_assets.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/export_whitebox_attack_assets.py scripts/export_lstm_attack_assets.py tests/test_export_lstm_attack_assets.py
git commit -m "refactor: add unified whitebox asset export entry"
```

### Task 6: 实现统一样本级攻击入口

**Files:**
- Create: `scripts/run_whitebox_attack.py`
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Test: `tests/test_run_whitebox_attack_smoke.py`

- [ ] **Step 1: 先写统一入口 smoke 失败测试**

```python
from pathlib import Path

from scripts.run_whitebox_attack import parse_args


def test_run_whitebox_attack_accepts_model_name():
    args = parse_args([
        "--model-name", "lstm",
        "--asset-dir", "artifacts/demo",
        "--model-root", "origin_model_pred",
    ])
    assert args.model_name == "lstm"
```

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_run_whitebox_attack_smoke.py::test_run_whitebox_attack_accepts_model_name -v`
Expected: FAIL

- [ ] **Step 3: 实现最小 CLI 与统一装配逻辑**

要求：
- 读取 `--model-name`
- 解析 `origin_model_pred/<model>/model/model_config.json`
- 装配 `RawFeatureAttackPipeline`
- 复用共享 clean gate 与 `FGSM` / `PGD`

- [ ] **Step 4: 补写 mock 产物测试**

要求验证：
- 运行后生成 `sample_metrics.csv`
- 运行后生成 `attack_summary.json`
- 运行后生成 `attack_report.md`

- [ ] **Step 5: 将旧 `run_lstm_whitebox_attack.py` 改为兼容壳层**

要求：
- 旧命令仍然可跑
- 默认内部等价于 `--model-name lstm`

- [ ] **Step 6: 运行统一入口测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_run_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/run_whitebox_attack.py scripts/run_lstm_whitebox_attack.py tests/test_run_whitebox_attack_smoke.py
git commit -m "feat: add unified sample-level whitebox attack runner"
```

### Task 7: 运行三模型真实 smoke 验证

**Files:**
- Output: `reports/lstm_whitebox_attack_*`
- Output: `reports/transformer_whitebox_attack_*`
- Output: `reports/tcn_whitebox_attack_*`

- [ ] **Step 1: 用统一入口回归 LSTM 小样本**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py --model-name lstm --model-root origin_model_pred --asset-dir artifacts/lstm_attack_expanded_v6 --out-dir reports/lstm_whitebox_attack_multimodel_smoke --max-samples 8 --device cpu`
Expected: 生成 CSV、JSON、Markdown，且 `clean_loss`、`fgsm_loss`、`pgd_loss` 全部为有限值

- [ ] **Step 2: 运行 Transformer 小样本 clean + attack**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py --model-name transformer --model-root origin_model_pred --asset-dir artifacts/lstm_attack_expanded_v6 --out-dir reports/transformer_whitebox_attack_smoke --max-samples 8 --device cpu`
Expected: loader 正常；clean gate 指标可读；`fgsm_loss > clean_loss`

- [ ] **Step 3: 运行 TCN 小样本 clean + attack**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py --model-name tcn --model-root origin_model_pred --asset-dir artifacts/lstm_attack_expanded_v6 --out-dir reports/tcn_whitebox_attack_smoke --max-samples 8 --device cpu`
Expected: loader 正常；clean gate 指标可读；`fgsm_loss > clean_loss`

- [ ] **Step 4: 汇总 smoke 运行结果，确认三模型输出格式一致**

Run: `find reports -maxdepth 2 \\( -path 'reports/*whitebox_attack*' -o -path 'reports/*whitebox_attack*/*' \\) -type f | sort`
Expected: 三个输出目录都包含 `sample_metrics.csv`、`attack_summary.json`、`attack_report.md`

- [ ] **Step 5: Commit 仅代码与文档，不提交大体积中间产物**

```bash
git add whitebox_attack_core.py whitebox_attack_models.py scripts/export_whitebox_attack_assets.py scripts/export_lstm_attack_assets.py scripts/run_whitebox_attack.py scripts/run_lstm_whitebox_attack.py tests origin_model_pred/lstm/model/model_config.json origin_model_pred/transformer/model/model_config.json origin_model_pred/tcn/model/model_config.json docs/superpowers/specs/2026-03-17-multi-model-whitebox-attack-design.md docs/superpowers/plans/2026-03-17-multi-model-whitebox-attack.md
git commit -m "feat: support sample-level whitebox attacks for transformer and tcn"
```

## Verification

- [ ] **Step 1: 跑新增测试集**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_whitebox_attack_core.py tests/test_whitebox_attack_models.py tests/test_run_whitebox_attack_smoke.py tests/test_export_lstm_attack_assets.py -v`
Expected: PASS

- [ ] **Step 2: 回归旧 LSTM 相关测试**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_legacy_lstm_attack_core.py tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 3: 抽查统一入口在 LSTM 上的对齐情况**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py --model-name lstm --model-root origin_model_pred --asset-dir artifacts/lstm_attack_expanded_v6 --out-dir reports/lstm_whitebox_attack_verify --max-samples 8 --device cpu`
Expected: `spearman_to_reference` 为正，输入梯度非空，`fgsm_loss > clean_loss`

- [ ] **Step 4: 抽查统一入口在 Transformer 与 TCN 上的攻击方向**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py --model-name transformer --model-root origin_model_pred --asset-dir artifacts/lstm_attack_expanded_v6 --out-dir reports/transformer_whitebox_attack_verify --max-samples 8 --device cpu && /home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py --model-name tcn --model-root origin_model_pred --asset-dir artifacts/lstm_attack_expanded_v6 --out-dir reports/tcn_whitebox_attack_verify --max-samples 8 --device cpu`
Expected: 两个模型都能生成完整报告，且攻击后 `MSE` 不低于 clean

Plan complete and saved to `docs/superpowers/plans/2026-03-17-multi-model-whitebox-attack.md`. Ready to execute?
