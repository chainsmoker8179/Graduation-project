# 组合层 5% 多随机种子 `physical_stat` 回测 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 LSTM 组合层“每日随机攻击 5% 股票”的多随机种子回测，切换到当前默认的 `physical_stat` 受约束攻击口径，并输出新的多 seed 汇总结果。

**Architecture:** 保留现有单次组合层回测主流程，只在 [scripts/run_partial_attack_backtest.py](/home/chainsmoker/qlib_test/scripts/run_partial_attack_backtest.py) 中增量接入 `constraint_mode / tau / lambda` 参数透传；再新增一个“共享 raw-only 资产构建”脚本和一个“多 seed orchestration”脚本，分别负责输入资产与批量实验驱动。结果目录结构沿用旧多 seed 结果格式，只更换目录名，不改主字段名。

**Tech Stack:** Python、PyTorch、pandas、Qlib、pytest、现有 `legacy_lstm_attack_core.py`、现有组合层 backtest 模块。

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-19-partial-attack-physical-stat-multiseed-design.md`
  - 已批准设计文档
- Modify: `scripts/run_partial_attack_backtest.py`
  - 接入 `constraint_mode / tau / lambda` 参数并写入输出摘要
- Create: `scripts/build_partial_attack_union_asset.py`
  - 生成 `seed=0..4`、`ratio=5%` 的 key 并集并导出 raw-only 资产
- Create: `scripts/run_partial_attack_backtest_multiseed.py`
  - 循环执行单 seed 回测并生成多 seed 汇总
- Modify: `tests/test_partial_attack_backtest.py`
  - 覆盖组合层脚本的约束参数透传
- Create: `tests/test_partial_attack_backtest_multiseed.py`
  - 覆盖多 seed 汇总逻辑
- Optionally Create: `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/`
  - 新实验结果目录

## Chunk 1: 单次组合层脚本接入 `physical_stat`

### Task 1: 为约束参数透传写失败测试

**Files:**
- Modify: `tests/test_partial_attack_backtest.py`
- Modify: `scripts/run_partial_attack_backtest.py`

- [ ] **Step 1: 写失败测试，验证 `_build_attack_fn(...)` 会把约束参数传给 `fgsm_maximize_mse / pgd_maximize_mse`**

```python
def test_attack_fn_passes_constraint_arguments(monkeypatch):
    calls = []

    def fake_fgsm_maximize_mse(**kwargs):
        calls.append(("fgsm", kwargs["constraint_mode"], kwargs["lambda_ret"]))
        return kwargs["x"]

    def fake_pgd_maximize_mse(**kwargs):
        calls.append(("pgd", kwargs["constraint_mode"], kwargs["lambda_ret"]))
        return kwargs["x"]

    ...

    assert calls == [
        ("fgsm", "physical_stat", 0.8),
        ("pgd", "physical_stat", 0.8),
    ]
```

- [ ] **Step 2: 运行该测试，确认在实现前失败**

Run: `pytest tests/test_partial_attack_backtest.py -k constraint_arguments -v`
Expected: FAIL，提示脚本未透传约束参数

- [ ] **Step 3: 在 `scripts/run_partial_attack_backtest.py` 中为 `argparse` 新增参数**

要求新增：
- `--constraint-mode`
- `--tau-ret`
- `--tau-body`
- `--tau-range`
- `--tau-vol`
- `--lambda-ret`
- `--lambda-candle`
- `--lambda-vol`

默认值：
- `constraint_mode=physical_stat`
- `tau_ret=0.005`
- `tau_body=0.005`
- `tau_range=0.01`
- `tau_vol=0.05`
- `lambda_ret=0.8`
- `lambda_candle=0.4`
- `lambda_vol=0.3`

- [ ] **Step 4: 在 `_build_attack_fn(...)` 中把这些参数透传给 `fgsm_maximize_mse` 和 `pgd_maximize_mse`**

- [ ] **Step 5: 重新运行该测试，确认通过**

Run: `pytest tests/test_partial_attack_backtest.py -k constraint_arguments -v`
Expected: PASS

- [ ] **Step 6: 扩展输出摘要**

在 `backtest_summary.json` 和 `README.md` 中新增：
- `constraint_mode`
- `tau_*`
- `lambda_*`

- [ ] **Step 7: 跑组合层相关测试子集**

Run: `pytest tests/test_partial_attack_backtest.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_partial_attack_backtest.py scripts/run_partial_attack_backtest.py
git commit -m "feat: support constrained partial attack backtests"
```

## Chunk 2: 构建多 seed 5% 共享 raw-only 资产

### Task 2: 新增并集 key 与 raw-only 资产导出脚本

**Files:**
- Create: `scripts/build_partial_attack_union_asset.py`
- Test: `tests/test_partial_attack_backtest_multiseed.py`

- [ ] **Step 1: 写失败测试，验证多 seed 选股 key 并集生成正确**

```python
def test_build_union_requested_keys_merges_multiple_seed_masks(tmp_path):
    ...
    assert len(result_keys) >= max(len(seed0_keys), len(seed1_keys))
    assert result_keys == sorted(result_keys)
```

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `pytest tests/test_partial_attack_backtest_multiseed.py -k union_requested_keys -v`
Expected: FAIL，提示脚本或函数不存在

- [ ] **Step 3: 实现脚本的核心逻辑**

要求：
- 读取 `pred.pkl`
- 基于 `build_daily_attack_mask(...)` 对 `seeds=0..4`、`ratio=0.05` 生成每日选股 key
- 对所有 key 取并集
- 落盘 `requested_keys.csv`
- 调用现有 exporter，使用 `--requested-keys-csv` 导出 raw-only 资产

输出目录固定包含：
- `matched_reference.csv`
- `normalization_stats.json`
- `matched_ohlcv_windows.pt`
- `export_summary.json`

- [ ] **Step 4: 重新运行并集逻辑测试**

Run: `pytest tests/test_partial_attack_backtest_multiseed.py -k union_requested_keys -v`
Expected: PASS

- [ ] **Step 5: 用小样本 dry-run 脚本**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/build_partial_attack_union_asset.py \
  --pred-pkl origin_model_pred/LSTM/prediction/pred.pkl \
  --label-pkl origin_model_pred/LSTM/prediction/label.pkl \
  --state-dict-path origin_model_pred/LSTM/model/lstm_state_dict.pt \
  --config-path .worktrees/multi-model-whitebox-attack/origin_model_pred/lstm/model/model_config.json \
  --seeds 0 1 \
  --attack-ratio 0.05 \
  --max-keys 2048 \
  --out-dir artifacts/partial_attack_union_smoke
```

Expected:
- 脚本成功退出
- 生成 `export_summary.json`

- [ ] **Step 6: Commit**

```bash
git add scripts/build_partial_attack_union_asset.py tests/test_partial_attack_backtest_multiseed.py
git commit -m "feat: add shared raw-only asset builder for multiseed partial attacks"
```

## Chunk 3: 多 seed orchestration 与统计汇总

### Task 3: 新增多 seed 驱动脚本

**Files:**
- Create: `scripts/run_partial_attack_backtest_multiseed.py`
- Modify: `tests/test_partial_attack_backtest_multiseed.py`

- [ ] **Step 1: 写失败测试，验证多 seed 汇总表结构**

```python
def test_multiseed_summary_contains_expected_delta_rows(tmp_path):
    ...
    assert "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost" in summary["metric"].tolist()
```

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `pytest tests/test_partial_attack_backtest_multiseed.py -k multiseed_summary -v`
Expected: FAIL，提示汇总脚本或函数不存在

- [ ] **Step 3: 实现多 seed 脚本**

要求：
- 循环 `seed=0..4`
- 调用 `scripts/run_partial_attack_backtest.py`
- 输出到：
  - `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/seed_<n>/`
- 汇总每个 seed 的：
  - `attackable_count`
  - `selected_available_count`
  - `partial_clean_minus_reference_clean`
  - `partial_fgsm_minus_partial_clean`
  - `partial_pgd_minus_partial_clean`
- 生成：
  - `multiseed_seed_metrics.csv`
  - `multiseed_summary_stats.csv`
  - `multiseed_pgd_vs_fgsm.json`

- [ ] **Step 4: 生成中文报告**

文件名：
- `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/多随机种子稳定性报告.md`

报告内容至少包含：
- 资产覆盖率
- 各 seed 的 `attackable_count`
- 四项主指标的均值 / 标准差
- `FGSM` 与 `PGD` 的相对强弱

- [ ] **Step 5: 重新运行汇总测试**

Run: `pytest tests/test_partial_attack_backtest_multiseed.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_partial_attack_backtest_multiseed.py tests/test_partial_attack_backtest_multiseed.py
git commit -m "feat: add multiseed constrained partial attack backtest runner"
```

## Chunk 4: 真实实验执行与结果核验

### Task 4: 重建 `seed=0..4`、`ratio=5%` 共享 raw-only 资产

**Files:**
- Create: `artifacts/lstm_attack_partial_seed0to4_ratio5_union_rawonly/`

- [ ] **Step 1: 运行共享资产构建脚本**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/build_partial_attack_union_asset.py \
  --pred-pkl origin_model_pred/LSTM/prediction/pred.pkl \
  --label-pkl origin_model_pred/LSTM/prediction/label.pkl \
  --attack-ratio 0.05 \
  --seeds 0 1 2 3 4 \
  --out-dir artifacts/lstm_attack_partial_seed0to4_ratio5_union_rawonly \
  --test-end-time 2025-10-31
```

Expected:
- 生成 raw-only 资产文件
- `export_summary.json` 中 `exported_sample_rows` 为正

- [ ] **Step 2: 记录资产统计**

从 `export_summary.json` 提取：
- `matched_reference_rows`
- `exported_sample_rows`
- `missing_raw_keys`

- [ ] **Step 3: Commit（若你们此阶段希望保留实验资产脚本状态）**

```bash
git add scripts/build_partial_attack_union_asset.py
git commit -m "chore: prepare shared raw-only asset for multiseed constrained backtests"
```

### Task 5: 运行 `physical_stat` 多 seed 组合层回测

**Files:**
- Create: `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/`

- [ ] **Step 1: 运行多 seed 驱动脚本**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_partial_attack_backtest_multiseed.py \
  --pred-pkl origin_model_pred/LSTM/prediction/pred.pkl \
  --label-pkl origin_model_pred/LSTM/prediction/label.pkl \
  --asset-dir artifacts/lstm_attack_partial_seed0to4_ratio5_union_rawonly \
  --state-dict-path origin_model_pred/LSTM/model/lstm_state_dict.pt \
  --config-path .worktrees/multi-model-whitebox-attack/origin_model_pred/lstm/model/model_config.json \
  --attack-ratio 0.05 \
  --seeds 0 1 2 3 4 \
  --constraint-mode physical_stat \
  --lambda-ret 0.8 \
  --lambda-candle 0.4 \
  --lambda-vol 0.3 \
  --out-dir reports/partial_attack_backtest_multiseed_ratio5_physical_stat
```

Expected:
- 五个 `seed_n/` 子目录生成成功
- 顶层汇总文件全部生成

- [ ] **Step 2: 检查多 seed 覆盖率**

重点检查：
- `attackable_count`
- `selected_available_count`

Expected:
- 五个 seed 波动较小

- [ ] **Step 3: 检查主结论**

重点读取：
- `multiseed_summary_stats.csv`
- `multiseed_pgd_vs_fgsm.json`

Expected:
- `partial_pgd_minus_partial_clean__annualized_excess_return_with_cost` 均值为负
- `partial_pgd_minus_partial_clean__rank_ic_mean` 均值为负
- `negative_ratio` 高，最好接近或等于 `1.0`

- [ ] **Step 4: 若结果稳定，补中文报告中的最终数值**

- [ ] **Step 5: 跑相关分析/可视化兼容性检查**

Run:

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python -m pytest tests/test_lstm_attack_daily_panel.py tests/plotting/test_next_figure_outputs.py -q
```

Expected:
- 与旧目录结构兼容的测试通过；若测试写死旧目录，则补最小必要修改

- [ ] **Step 6: Commit**

```bash
git add scripts/run_partial_attack_backtest.py scripts/run_partial_attack_backtest_multiseed.py \
  reports/partial_attack_backtest_multiseed_ratio5_physical_stat \
  tests/test_partial_attack_backtest.py tests/test_partial_attack_backtest_multiseed.py
git commit -m "feat: run multiseed partial backtests with physical-stat constrained attacks"
```
