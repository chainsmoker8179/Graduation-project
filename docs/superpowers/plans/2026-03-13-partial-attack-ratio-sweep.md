# Partial Attack Ratio Sweep Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `seeds=0..4` 下比较 `attack_ratio=1% / 5% / 10%` 的组合级退化趋势，并输出中文报告与可复用汇总表。

**Architecture:** 不修改现有攻击与回测脚本，直接复用 `scripts/run_partial_attack_backtest.py`。先对三档比例和五个种子的候选 key 求并集并导出一份共享 raw-only 资产，再跑新增的 `1%` 与 `10%` 十组实验，复用已有 `5%` 多 seed 结果做统一聚合统计，最后生成中文报告。

**Tech Stack:** Python、PyTorch、pandas、Qlib、现有 partial attack backtest 脚本

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-13-partial-attack-ratio-sweep-design.md`
  - 记录攻击比例对比的实验目的、共享资产策略和判定口径。
- Create: `docs/superpowers/plans/2026-03-13-partial-attack-ratio-sweep.md`
  - 记录本轮执行步骤。
- Create: `artifacts/partial_attack_requested_keys_seed0to4_ratio1_5_10_union_2025_01_01_2025_10_31.csv`
  - 三档比例、五个种子的候选 key 并集。
- Create: `artifacts/lstm_attack_partial_seed0to4_ratio1_5_10_union_rawonly/`
  - 共享 raw-only 资产目录。
- Create: `reports/partial_attack_ratio_sweep_multiseed/`
  - 存放比例对比的明细、汇总和中文报告。

## Chunk 1: 构建共享资产

### Task 1: 生成 `1% / 5% / 10%` 的多 seed 候选 key 并集

**Files:**
- Create: `artifacts/partial_attack_requested_keys_seed0to4_ratio1_5_10_union_2025_01_01_2025_10_31.csv`

- [ ] **Step 1: 读取测试窗口内的 `pred.pkl` 索引，并为 `ratio=0.01 / 0.05 / 0.10`、`seed=0..4` 构造 daily attack mask**

- [ ] **Step 2: 对所有候选 key 去重并保存 CSV**

- [ ] **Step 3: 记录三档比例各自的 union 规模，以及总 union 规模**

### Task 2: 导出共享 raw-only 资产

**Files:**
- Create: `artifacts/lstm_attack_partial_seed0to4_ratio1_5_10_union_rawonly/matched_reference.csv`
- Create: `artifacts/lstm_attack_partial_seed0to4_ratio1_5_10_union_rawonly/normalization_stats.json`
- Create: `artifacts/lstm_attack_partial_seed0to4_ratio1_5_10_union_rawonly/matched_ohlcv_windows.pt`
- Create: `artifacts/lstm_attack_partial_seed0to4_ratio1_5_10_union_rawonly/export_summary.json`

- [ ] **Step 1: 复用 `pred.pkl`、`label.pkl` 与并集 key，构造 filtered `matched_reference`**

- [ ] **Step 2: 直接复用现有 `normalization_stats.json` 作为共享资产统计量**

- [ ] **Step 3: 仅导出 raw windows，不导出 feature windows**

- [ ] **Step 4: 保存导出汇总，至少记录 `matched_reference_rows`、`exported_sample_rows` 与 `missing_raw_keys`**

## Chunk 2: 执行新增比例实验

### Task 3: 运行 `1%` 多 seed 复跑

**Files:**
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_1/seed_0/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_1/seed_1/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_1/seed_2/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_1/seed_3/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_1/seed_4/`

- [ ] **Step 1: 使用共享资产目录运行 `ratio=0.01`、`seed=0..4` 的五组 partial backtest**

- [ ] **Step 2: 检查每个 seed 的 `attackable_count` 是否处于合理量级**

### Task 4: 运行 `10%` 多 seed 复跑

**Files:**
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_10/seed_0/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_10/seed_1/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_10/seed_2/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_10/seed_3/`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_10/seed_4/`

- [ ] **Step 1: 使用共享资产目录运行 `ratio=0.10`、`seed=0..4` 的五组 partial backtest**

- [ ] **Step 2: 检查每个 seed 的 `attackable_count` 是否处于合理量级**

## Chunk 3: 汇总比例趋势并写报告

### Task 5: 聚合 `1% / 5% / 10%` 三档比例结果

**Files:**
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_sweep_seed_metrics.csv`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_sweep_summary_stats.csv`
- Create: `reports/partial_attack_ratio_sweep_multiseed/ratio_sweep_trend_summary.json`

- [ ] **Step 1: 读取 `1%`、已有 `5%`、`10%` 的每 seed `comparison.csv` 与 `backtest_summary.json`**

- [ ] **Step 2: 输出统一 seed 级汇总表**

- [ ] **Step 3: 计算三档比例在 FGSM/PGD 下的均值、标准差、方向一致性与近似单调趋势**

### Task 6: 写中文实验报告

**Files:**
- Create: `reports/partial_attack_ratio_sweep_multiseed/攻击比例对比报告.md`

- [ ] **Step 1: 记录共享资产策略与并集规模**

- [ ] **Step 2: 总结 `1% / 5% / 10%` 在 FGSM 和 PGD 下的平均退化幅度**

- [ ] **Step 3: 解释比例上升是否带来更强退化，以及哪些指标最稳定**

- [ ] **Step 4: 给出限制与下一步建议**
