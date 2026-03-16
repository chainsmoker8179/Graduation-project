# Partial Attack Backtest Evaluation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一条“每日随机攻击 5% 股票并重放完整 Qlib 回测”的实验链路，输出 `reference_clean / partial_clean / partial_fgsm / partial_pgd` 四组组合表现及其比较结果。

**Architecture:** 先把现有原始 `OHLCV` LSTM 攻击脚本中的可复用攻击核心抽取成模块，再新增部分股票攻击与分数混合逻辑，最后用统一的 Qlib 回测配置对四组分数表分别重放并输出汇总指标、日度差值和覆盖率诊断。实现上继续复用现有 `pred.pkl`、`label.pkl`、LSTM `state_dict`、可微特征桥接层和 `expanded_v6` 攻击参数，避免重新定义模型或攻击口径。

**Tech Stack:** Python、PyTorch、pandas、Qlib、本仓库现有 legacy LSTM attack pipeline、pytest

---

## 文件结构

- Create: `legacy_lstm_attack_core.py`
  - 提取并复用当前 `scripts/run_lstm_whitebox_attack.py` 中的 pipeline、预算投影、FGSM/PGD 与 clean gate 逻辑。
- Create: `partial_attack_backtest.py`
  - 封装每日随机抽样、可攻击样本过滤、四组分数表组装、回测结果整理与日度差值分析。
- Create: `scripts/run_partial_attack_backtest.py`
  - 端到端运行入口，负责读取资产、生成分数、触发 Qlib 回测并落盘结果。
- Create: `tests/test_legacy_lstm_attack_core.py`
  - 约束抽取后的攻击核心与现有脚本行为一致。
- Create: `tests/test_partial_attack_backtest.py`
  - 约束 mask 生成、局部替换、回测汇总与日度差值逻辑。
- Modify: `scripts/run_lstm_whitebox_attack.py`
  - 改为复用 `legacy_lstm_attack_core.py`，避免攻击逻辑重复维护。
- Modify: `scripts/export_lstm_attack_assets.py`
  - 暴露共享的数据准备工具，或补充支持按指定 key 集导出 raw windows / 参考标签，供部分股票攻击实验复用。
- Create: `reports/partial_attack_backtest/`
  - 存放首轮实验输出与中文报告。

## Chunk 1: 抽取可复用攻击核心

### Task 1: 提取 legacy LSTM 攻击核心模块

**Files:**
- Create: `legacy_lstm_attack_core.py`
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Test: `tests/test_legacy_lstm_attack_core.py`

- [ ] **Step 1: 写失败测试，约束抽取后的 `LegacyRawLSTMPipeline`、预算投影、FGSM、PGD 和 clean gate 与现有脚本返回相同形状与关键数值**

```python
def test_fgsm_and_pgd_match_existing_attack_core():
    assert torch.allclose(new_fgsm, old_fgsm)
    assert torch.allclose(new_pgd, old_pgd)
```

- [ ] **Step 2: 运行测试确认先失败**

Run: `pytest tests/test_legacy_lstm_attack_core.py -v`
Expected: FAIL with import or symbol not found errors

- [ ] **Step 3: 创建 `legacy_lstm_attack_core.py`，把当前脚本中的攻击核心提取成可 import 的模块**

```python
class LegacyRawLSTMPipeline(nn.Module):
    ...

def fgsm_maximize_mse(...):
    ...

def pgd_maximize_mse(...):
    ...

def run_clean_gate(...):
    ...
```

- [ ] **Step 4: 修改 `scripts/run_lstm_whitebox_attack.py`，让其改用新核心模块**

```python
from legacy_lstm_attack_core import (
    LegacyRawLSTMPipeline,
    fgsm_maximize_mse,
    pgd_maximize_mse,
    run_clean_gate,
)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_legacy_lstm_attack_core.py -v`
Expected: PASS

## Chunk 2: 构建部分股票攻击与分数混合逻辑

### Task 2: 实现每日随机 5% 攻击 mask 和局部替换分数表

**Files:**
- Create: `partial_attack_backtest.py`
- Test: `tests/test_partial_attack_backtest.py`

- [ ] **Step 1: 写失败测试，约束每日随机抽样固定 `seed` 时可复现，且每日至少选出 1 个样本**

```python
def test_build_daily_attack_mask_is_reproducible():
    mask_a = build_daily_attack_mask(index_df, ratio=0.05, seed=0)
    mask_b = build_daily_attack_mask(index_df, ratio=0.05, seed=0)
    assert mask_a.equals(mask_b)
```

- [ ] **Step 2: 运行测试确认先失败**

Run: `pytest tests/test_partial_attack_backtest.py::test_build_daily_attack_mask_is_reproducible -v`
Expected: FAIL with function not defined

- [ ] **Step 3: 实现 mask 构造、可攻击 key 过滤与四组分数表混合逻辑**

```python
def build_daily_attack_mask(index_df, ratio, seed):
    ...

def merge_partial_scores(reference_scores, replacement_scores, attackable_keys):
    ...

def build_score_tables(...):
    return {
        "reference_clean": ...,
        "partial_clean": ...,
        "partial_fgsm": ...,
        "partial_pgd": ...,
    }
```

- [ ] **Step 4: 补充失败测试，约束 `partial_clean / fgsm / pgd` 只替换指定 key，其余 key 必须与 `pred.pkl` 完全一致**

```python
def test_build_score_tables_only_replaces_attackable_keys():
    assert unchanged_scores.equals(reference_scores.loc[unchanged_index])
```

- [ ] **Step 5: 运行对应测试并确认通过**

Run: `pytest tests/test_partial_attack_backtest.py -v`
Expected: PASS

### Task 3: 复用数据准备工具，支持按 key 集加载攻击样本

**Files:**
- Modify: `scripts/export_lstm_attack_assets.py`
- Modify: `partial_attack_backtest.py`
- Test: `tests/test_partial_attack_backtest.py`

- [ ] **Step 1: 写失败测试，约束指定 key 集能拿到对应 raw window、label 和 reference score**

```python
def test_load_attack_subset_returns_requested_keys_only():
    assert list(result.index) == requested_keys
```

- [ ] **Step 2: 运行测试确认先失败**

Run: `pytest tests/test_partial_attack_backtest.py::test_load_attack_subset_returns_requested_keys_only -v`
Expected: FAIL because helper is missing

- [ ] **Step 3: 在 `scripts/export_lstm_attack_assets.py` 中整理出可复用 helper，供部分股票攻击脚本按 key 集加载测试样本**

```python
def load_raw_windows_for_keys(...):
    ...

def load_reference_scores_for_keys(...):
    ...
```

- [ ] **Step 4: 在 `partial_attack_backtest.py` 中接入这些 helper，统计 `selected_keys`、`attackable_keys` 和失败原因**

```python
summary = {
    "selected_count": ...,
    "attackable_count": ...,
    "failure_reasons": ...,
}
```

- [ ] **Step 5: 运行目标测试并确认通过**

Run: `pytest tests/test_partial_attack_backtest.py -v`
Expected: PASS

## Chunk 3: 完整回测重放与结果整理

### Task 4: 实现四组分数表的 Qlib 回测重放

**Files:**
- Create: `scripts/run_partial_attack_backtest.py`
- Create: `partial_attack_backtest.py`
- Test: `tests/test_partial_attack_backtest.py`

- [ ] **Step 1: 写失败测试，约束脚本层或模块层能接受四组分数表并产出统一结构的回测汇总**

```python
def test_backtest_summary_contains_all_score_groups():
    assert sorted(summary.keys()) == [
        "partial_clean",
        "partial_fgsm",
        "partial_pgd",
        "reference_clean",
    ]
```

- [ ] **Step 2: 运行测试确认先失败**

Run: `pytest tests/test_partial_attack_backtest.py::test_backtest_summary_contains_all_score_groups -v`
Expected: FAIL with backtest runner missing

- [ ] **Step 3: 在 `partial_attack_backtest.py` 中实现回测适配层，复用现有 notebook 使用的 Qlib 回测配置**

```python
def run_score_backtests(score_tables, backtest_config):
    ...

def summarize_backtest_outputs(results):
    ...
```

- [ ] **Step 4: 创建 `scripts/run_partial_attack_backtest.py`，串联 mask、攻击分数生成、四组回测与结果落盘**

```python
def main():
    mask = build_daily_attack_mask(...)
    score_tables = build_score_tables(...)
    backtest_results = run_score_backtests(score_tables, backtest_config)
    save_outputs(...)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_partial_attack_backtest.py -v`
Expected: PASS

### Task 5: 输出主比较差值、日度对照表与实验报告

**Files:**
- Modify: `partial_attack_backtest.py`
- Modify: `scripts/run_partial_attack_backtest.py`
- Create: `reports/partial_attack_backtest/README.md`
- Test: `tests/test_partial_attack_backtest.py`

- [ ] **Step 1: 写失败测试，约束输出包含 `partial_fgsm - partial_clean`、`partial_pgd - partial_clean` 和 `partial_clean - reference_clean` 三组主差值**

```python
def test_comparison_table_contains_primary_deltas():
    assert "partial_fgsm_minus_partial_clean" in comparison.index
```

- [ ] **Step 2: 运行测试确认先失败**

Run: `pytest tests/test_partial_attack_backtest.py::test_comparison_table_contains_primary_deltas -v`
Expected: FAIL with comparison helper missing

- [ ] **Step 3: 实现汇总比较表、日度差值表和攻击覆盖率摘要输出**

```python
def build_comparison_table(summary):
    ...

def build_daily_comparison_table(...):
    ...
```

- [ ] **Step 4: 生成中文结果说明模板，至少包含实验设置、攻击覆盖率、四组回测结果与结论**

```markdown
# 部分股票攻击回测实验报告

- 攻击比例
- 可攻击覆盖率
- 超额收益变化
- 最大回撤变化
- RankIC 变化
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_partial_attack_backtest.py -v`
Expected: PASS

## Chunk 4: 验证与首轮实验运行

### Task 6: 跑新增测试并执行首轮单种子实验

**Files:**
- Test: `tests/test_legacy_lstm_attack_core.py`
- Test: `tests/test_partial_attack_backtest.py`
- Run: `scripts/run_partial_attack_backtest.py`

- [ ] **Step 1: 运行新增测试集合**

Run: `pytest tests/test_legacy_lstm_attack_core.py tests/test_partial_attack_backtest.py -v`
Expected: PASS

- [ ] **Step 2: 在 Qlib 环境中运行首轮单种子部分股票攻击回测**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_partial_attack_backtest.py --seed 0 --attack-ratio 0.05`
Expected: 写出四组分数、回测汇总、日度比较表和中文报告

- [ ] **Step 3: 核对三类关键结果**

Run: `python - <<'PY'\nfrom pathlib import Path\nimport json\nprint(json.loads(Path('reports/partial_attack_backtest/backtest_summary.json').read_text())['comparison'])\nPY`
Expected: 能看到 `partial_clean - reference_clean`、`partial_fgsm - partial_clean`、`partial_pgd - partial_clean` 三组差值

- [ ] **Step 4: 根据结果判断是否满足第一版成功标准**

Run: `python - <<'PY'\nfrom pathlib import Path\nimport pandas as pd\np = Path('reports/partial_attack_backtest/daily_comparison.csv')\ndf = pd.read_csv(p)\nprint(df[['fgsm_minus_partial_clean_excess_return','pgd_minus_partial_clean_excess_return']].mean())\nPY`
Expected: 为后续报告撰写提供日度差值方向证据

---

Plan complete and saved to `docs/superpowers/plans/2026-03-13-partial-attack-backtest-evaluation.md`. Ready to execute?
