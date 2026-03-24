# Fast Large-Sample Attack Export Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个正式的快导出脚本，用于稳定构建大样本 white-box attack 资产，同时保留旧导出脚本行为不变。

**Architecture:** 在新脚本中复用 `build_matched_reference`、`_build_raw_test_split` 与 `export_matched_raw_windows`，将 Qlib 慢路径替换为 `LegacyLSTMFeatureBridge + RobustZScoreNormLayer + FillnaLayer` 的本地特征重建；另加一个极薄的兼容壳层 `export_whitebox_attack_assets.py`，只暴露旧计划中已经引用的入口。

**Tech Stack:** Python、PyTorch、现有 `scripts/export_lstm_attack_assets.py` 公共函数、`legacy_lstm_feature_bridge.py`、`legacy_lstm_preprocess.py`、pytest。

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-20-fast-large-sample-attack-export-design.md`
  - 本次快导出脚本设计
- Create: `scripts/export_large_sample_attack_assets.py`
  - 新的正式快导出脚本
- Create: `scripts/export_whitebox_attack_assets.py`
  - 兼容壳层，提供历史引用入口
- Create: `tests/test_export_large_sample_attack_assets.py`
  - 新脚本的轻量测试

## Chunk 1: 测试优先定义接口

### Task 1: 为快导出核心函数写失败测试

**Files:**
- Create: `tests/test_export_large_sample_attack_assets.py`
- Test: `tests/test_export_large_sample_attack_assets.py`

- [ ] **Step 1: 先写 feature window 重建测试**

测试最小行为：

```python
def test_build_feature_windows_from_raw_preserves_keys_and_marks_source():
    ...
```

断言至少包括：
- 输出 `keys` 与输入 raw asset 一致
- 输出 `features.shape[0] == len(keys)`
- 输出 `feature_source == "torch_bridge_from_raw"`

- [ ] **Step 2: 写 normalization stats 复用测试**

测试最小行为：

```python
def test_copy_normalization_stats_writes_same_payload(tmp_path):
    ...
```

- [ ] **Step 3: 写 summary 生成测试**

测试最小行为：

```python
def test_build_export_summary_contains_fast_export_fields():
    ...
```

- [ ] **Step 4: 写兼容壳层测试**

测试：

```python
from scripts.export_whitebox_attack_assets import build_matched_reference
```

并验证它与旧实现行为一致。

- [ ] **Step 5: 运行测试确认先失败**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_export_large_sample_attack_assets.py -q
```

Expected: FAIL，提示新脚本或目标函数不存在。

## Chunk 2: 最小实现新脚本

### Task 2: 实现 `export_large_sample_attack_assets.py`

**Files:**
- Create: `scripts/export_large_sample_attack_assets.py`
- Test: `tests/test_export_large_sample_attack_assets.py`

- [ ] **Step 1: 实现参数解析**

至少支持：
- `--pred-pkl`
- `--label-pkl`
- `--out-dir`
- `--normalization-stats`
- `--test-start-time`
- `--test-end-time`
- `--max-samples`
- `--seed`

- [ ] **Step 2: 实现 `build_feature_windows_from_raw`**

内部流程：

```python
bridge = LegacyLSTMFeatureBridge(...)
norm = RobustZScoreNormLayer(...)
fillna = FillnaLayer()
features = fillna(norm(bridge(raw_ohlcv)))
```

- [ ] **Step 3: 实现 `build_export_summary`**

明确写出：
- `feature_source`
- `normalization_stats_source`

- [ ] **Step 4: 实现 `main(argv=None)`**

按顺序执行：
- 读取 `pred.pkl` / `label.pkl`
- 生成 `matched_reference`
- 导出 raw windows
- 复制 normalization stats
- 从 raw window 重建 feature windows
- 写 `export_summary.json`

- [ ] **Step 5: 运行新测试确认转绿**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_export_large_sample_attack_assets.py -q
```

Expected: PASS

## Chunk 3: 兼容壳层

### Task 3: 实现 `export_whitebox_attack_assets.py`

**Files:**
- Create: `scripts/export_whitebox_attack_assets.py`
- Test: `tests/test_export_large_sample_attack_assets.py`

- [ ] **Step 1: 创建极薄兼容壳层**

至少提供：

```python
from scripts.export_lstm_attack_assets import build_matched_reference
```

必要时也可导出 `main` 或说明该文件仅作兼容入口。

- [ ] **Step 2: 运行兼容测试**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_export_large_sample_attack_assets.py -q
```

Expected: PASS

## Chunk 4: 最终核查

### Task 4: 运行聚焦验证

**Files:**
- Verify: `scripts/export_large_sample_attack_assets.py`
- Verify: `scripts/export_whitebox_attack_assets.py`
- Verify: `tests/test_export_large_sample_attack_assets.py`

- [ ] **Step 1: 只跑新增测试和相关旧测试**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest \
  tests/test_export_large_sample_attack_assets.py \
  tests/test_export_lstm_attack_assets.py -q
```

Expected: 新脚本测试通过；若旧测试因既有脏修改导致不稳定，至少确保新增测试独立通过。

- [ ] **Step 2: 记录使用方式**

最终说明至少给出：
- 新脚本的最小调用命令
- 何时应使用新脚本
- 何时仍应保留旧脚本
