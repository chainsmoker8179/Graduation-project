# Attack Clean Gate And Expanded Smoke Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收紧原始 OHLCV 白盒攻击的 clean gate，引入 Qlib 参考特征序列对齐检查，并在更大样本上重新运行 FGSM/PGD 统计与中文报告生成。

**Architecture:** 在离线导出侧补充旧版 LSTM 真正消费的参考特征窗口资产；在在线攻击侧增加特征级对齐指标与可配置阈值，并将 clean gate 从“可计算”提升为“可计算且与参考输出足够接近”。最后基于扩大后的 matched 样本重新运行攻击脚本，并把设置、阈值和结果整理为中文实验报告。

**Tech Stack:** Python、PyTorch、Qlib、pytest、pandas、现有 Legacy LSTM attack scripts

---

## Chunk 1: 参考特征资产导出

### Task 1: 导出 matched 的参考特征时序窗口

**Files:**
- Modify: `scripts/export_lstm_attack_assets.py`
- Test: `tests/test_export_lstm_attack_assets.py`

- [ ] **Step 1: 写失败测试，约束导出脚本内部函数能输出 matched feature windows**
- [ ] **Step 2: 运行该测试，确认先失败**
- [ ] **Step 3: 实现基于 Qlib 特征 handler + TSDatasetH(step_len=20) 的参考特征窗口导出**
- [ ] **Step 4: 运行测试确认通过**

## Chunk 2: Clean Gate 收紧

### Task 2: 增加特征对齐指标与阈值校验

**Files:**
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Test: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 写失败测试，约束 clean gate 能报告 feature MAE / RMSE 和 Spearman 阈值失败**
- [ ] **Step 2: 运行该测试，确认先失败**
- [ ] **Step 3: 实现 reference feature comparison、阈值参数与 fail-fast clean gate**
- [ ] **Step 4: 运行测试确认通过**

## Chunk 3: 更大样本攻击与报告

### Task 3: 扩大导出样本并生成中文实验报告

**Files:**
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Create: `reports/lstm_whitebox_attack_expanded_report.md`

- [ ] **Step 1: 使用扩大后的 max_samples 重跑导出脚本**
- [ ] **Step 2: 使用新 clean gate 重跑攻击脚本**
- [ ] **Step 3: 将设置、clean gate 指标和 FGSM/PGD 统计整理为中文报告**
- [ ] **Step 4: 运行 `pytest tests -v` 作为最终回归验证**
