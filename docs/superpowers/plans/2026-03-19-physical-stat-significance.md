# Physical-Stat 多随机种子显著性分析 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/` 生成显著性统计文件和中文显著性报告。

**Architecture:** 复用现有 `scripts/analysis/lstm_attack_significance.py` 统计口径，仅把多 seed 日度 panel 的读取逻辑参数化，并新增一个面向当前实验目录的中文报告脚本。输出全部写回当前实验目录，避免污染旧的无约束证据链目录。

**Tech Stack:** Python, pandas, scipy, pytest

---

## Chunk 1: 参数化日度面板与统计入口

### Task 1: 先写失败测试

**Files:**
- Modify: `tests/test_lstm_attack_significance.py`
- Modify: `scripts/analysis/lstm_attack_daily_panel.py`
- Modify: `scripts/run_lstm_attack_significance.py`

- [ ] **Step 1: 为自定义多 seed 目录加载写失败测试**

```python
def test_load_multiseed_daily_panel_accepts_custom_seed_root(tmp_path: Path) -> None:
    ...
```

- [ ] **Step 2: 运行定向测试确认失败**

Run: `pytest tests/test_lstm_attack_significance.py -q`
Expected: FAIL because custom root is not supported yet

- [ ] **Step 3: 实现最小改造**

```python
def load_multiseed_daily_panel(report_root: str | Path, *, seed_root_name: str | None = None, seed_root_path: str | Path | None = None) -> pd.DataFrame:
    ...
```

- [ ] **Step 4: 运行定向测试确认通过**

Run: `pytest tests/test_lstm_attack_significance.py -q`
Expected: PASS

### Task 2: 为显著性脚本输出目标目录写失败测试

**Files:**
- Modify: `tests/test_lstm_attack_significance.py`
- Modify: `scripts/run_lstm_attack_significance.py`

- [ ] **Step 1: 新增 CLI 级测试，验证脚本可对指定实验目录输出统计文件**

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_lstm_attack_significance.py -q`
Expected: FAIL because the script still hardcodes the old union directory

- [ ] **Step 3: 实现最小参数化**

```python
parser.add_argument("--seed-root-path", type=Path, default=None)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_lstm_attack_significance.py -q`
Expected: PASS

## Chunk 2: 中文显著性报告

### Task 3: 先写失败测试，再补报告脚本

**Files:**
- Create: `scripts/build_physical_stat_significance_report.py`
- Create: `tests/test_build_physical_stat_significance_report.py`

- [ ] **Step 1: 为中文报告最小渲染写失败测试**

```python
def test_render_report_includes_daily_significance_and_bootstrap_interval() -> None:
    ...
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_build_physical_stat_significance_report.py -q`
Expected: FAIL because report script does not exist yet

- [ ] **Step 3: 实现最小报告生成**

```python
def render_report(...)-> str:
    ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_build_physical_stat_significance_report.py -q`
Expected: PASS

## Chunk 3: 实际生成与回归验证

### Task 4: 跑测试与实际统计

**Files:**
- Verify: `tests/test_lstm_attack_significance.py`
- Verify: `tests/test_build_physical_stat_significance_report.py`
- Generate: `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/significance_daily_metrics.csv`
- Generate: `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/significance_block_bootstrap.json`
- Generate: `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/显著性检验报告.md`

- [ ] **Step 1: 运行测试**

Run: `pytest tests/test_lstm_attack_significance.py tests/test_build_physical_stat_significance_report.py -q`
Expected: PASS

- [ ] **Step 2: 实际生成统计文件**

Run: `python scripts/run_lstm_attack_significance.py --seed-root-path reports/partial_attack_backtest_multiseed_ratio5_physical_stat --out-dir reports/partial_attack_backtest_multiseed_ratio5_physical_stat`
Expected: 输出 `significance_daily_metrics.csv` 和 `significance_block_bootstrap.json`

- [ ] **Step 3: 生成中文报告**

Run: `python scripts/build_physical_stat_significance_report.py --report-dir reports/partial_attack_backtest_multiseed_ratio5_physical_stat`
Expected: 输出 `显著性检验报告.md`

- [ ] **Step 4: 做最终回归验证**

Run: `pytest tests/test_lstm_attack_significance.py tests/test_build_physical_stat_significance_report.py tests/test_partial_attack_backtest_multiseed.py -q`
Expected: PASS
