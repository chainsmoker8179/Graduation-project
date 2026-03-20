# Transformer / TCN 大样本白盒攻击正式实验 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `Transformer` 与 `TCN` 各自构建 `4096` 样本的大样本白盒攻击正式实验，并生成可直接引用的结果文件与中文总结。

**Architecture:** 不新增攻击算法，直接复用现有 `export_lstm_attack_assets.py` 和 `run_whitebox_attack.py`。先在共同时间区间 `2025-01-01` 到 `2025-10-31` 内，从完整 `pred.pkl` / `label.pkl` 为两个模型分别导出 `4096` 样本攻击资产；随后复跑统一 white-box runner，并将导出规模、clean gate 与 FGSM / PGD 结果整理成正式报告。

**Tech Stack:** Python、Qlib、PyTorch、现有资产导出脚本、统一 white-box attack runner、pandas、JSON、Markdown 报告。

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-20-multi-model-large-sample-whitebox-eval-design.md`
  - 本次正式实验设计文档
- Create: `artifacts/transformer_attack_4096_v1/`
  - Transformer 大样本攻击资产
- Create: `artifacts/tcn_attack_4096_v1/`
  - TCN 大样本攻击资产
- Create: `reports/transformer_whitebox_attack_4096_v1/`
  - Transformer 大样本攻击结果
- Create: `reports/tcn_whitebox_attack_4096_v1/`
  - TCN 大样本攻击结果
- Create: `reports/transformer_tcn_4096白盒攻击实验报告.md`
  - 两模型中文总结报告

## Chunk 1: 导出大样本攻击资产

### Task 1: 导出 Transformer 的 4096 样本攻击资产

**Files:**
- Use: `scripts/export_lstm_attack_assets.py`
- Output: `artifacts/transformer_attack_4096_v1/`

- [ ] **Step 1: 先确认导出命令只使用共同时间区间**

命令必须包含：

```bash
--test-start-time 2025-01-01
--test-end-time 2025-10-31
--max-samples 4096
```

- [ ] **Step 2: 运行 Transformer 资产导出**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl /home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_lstm_attack_assets.py \
  --pred-pkl origin_model_pred/Transformer/pred/pred.pkl \
  --label-pkl origin_model_pred/Transformer/pred/label.pkl \
  --out-dir artifacts/transformer_attack_4096_v1 \
  --test-start-time 2025-01-01 \
  --test-end-time 2025-10-31 \
  --max-samples 4096 \
  --seed 0
```

Expected: 正常输出 `matched_reference_rows`、`exported_sample_rows` 与资产路径。

- [ ] **Step 3: 检查导出摘要**

Run:

```bash
cat artifacts/transformer_attack_4096_v1/export_summary.json
```

Expected:
- `matched_reference_rows` 大于 0
- `exported_sample_rows` 大于 0
- `raw_window_len = 80`
- `feature_dim = 20`

### Task 2: 导出 TCN 的 4096 样本攻击资产

**Files:**
- Use: `scripts/export_lstm_attack_assets.py`
- Output: `artifacts/tcn_attack_4096_v1/`

- [ ] **Step 1: 运行 TCN 资产导出**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl /home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_lstm_attack_assets.py \
  --pred-pkl origin_model_pred/TCN/pred/pred.pkl \
  --label-pkl origin_model_pred/TCN/pred/label.pkl \
  --out-dir artifacts/tcn_attack_4096_v1 \
  --test-start-time 2025-01-01 \
  --test-end-time 2025-10-31 \
  --max-samples 4096 \
  --seed 0
```

Expected: 正常输出 `matched_reference_rows`、`exported_sample_rows` 与资产路径。

- [ ] **Step 2: 检查导出摘要**

Run:

```bash
cat artifacts/tcn_attack_4096_v1/export_summary.json
```

Expected:
- `matched_reference_rows` 大于 0
- `exported_sample_rows` 大于 0
- `raw_window_len = 80`
- `feature_dim = 20`

## Chunk 2: 运行大样本白盒攻击

### Task 3: 运行 Transformer 大样本 white-box attack

**Files:**
- Use: `scripts/run_whitebox_attack.py`
- Input: `artifacts/transformer_attack_4096_v1/`
- Output: `reports/transformer_whitebox_attack_4096_v1/`

- [ ] **Step 1: 运行 Transformer 攻击**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl /home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py \
  --model-name transformer \
  --config-path origin_model_pred/Transformer/model/model_config.json \
  --state-dict-path /home/chainsmoker/qlib_test/origin_model_pred/Transformer/model/transformer_state_dict.pt \
  --asset-dir artifacts/transformer_attack_4096_v1 \
  --out-dir reports/transformer_whitebox_attack_4096_v1 \
  --max-samples 1000000 \
  --device cpu
```

Expected: 成功写出 `attack_summary.json`。

- [ ] **Step 2: 检查 clean gate 与攻击结果**

Run:

```bash
cat reports/transformer_whitebox_attack_4096_v1/attack_summary.json
```

Expected:
- `sample_count > 0`
- `clean_gate.clean_grad_finite_rate = 1.0`
- `fgsm_loss > clean_loss`
- `pgd_loss > clean_loss`

### Task 4: 运行 TCN 大样本 white-box attack

**Files:**
- Use: `scripts/run_whitebox_attack.py`
- Input: `artifacts/tcn_attack_4096_v1/`
- Output: `reports/tcn_whitebox_attack_4096_v1/`

- [ ] **Step 1: 运行 TCN 攻击**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl /home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_whitebox_attack.py \
  --model-name tcn \
  --config-path origin_model_pred/TCN/model/model_config.json \
  --state-dict-path /home/chainsmoker/qlib_test/origin_model_pred/TCN/model/tcn_state_dict.pt \
  --asset-dir artifacts/tcn_attack_4096_v1 \
  --out-dir reports/tcn_whitebox_attack_4096_v1 \
  --max-samples 1000000 \
  --device cpu
```

Expected: 成功写出 `attack_summary.json`。

- [ ] **Step 2: 检查 clean gate 与攻击结果**

Run:

```bash
cat reports/tcn_whitebox_attack_4096_v1/attack_summary.json
```

Expected:
- `sample_count > 0`
- `clean_gate.clean_grad_finite_rate = 1.0`
- `fgsm_loss > clean_loss`
- `pgd_loss > clean_loss`

## Chunk 3: 汇总正式实验结果

### Task 5: 生成两模型中文总报告

**Files:**
- Create: `reports/transformer_tcn_4096白盒攻击实验报告.md`
- Use: `reports/transformer_whitebox_attack_4096_v1/attack_summary.json`
- Use: `reports/tcn_whitebox_attack_4096_v1/attack_summary.json`
- Use: `artifacts/transformer_attack_4096_v1/export_summary.json`
- Use: `artifacts/tcn_attack_4096_v1/export_summary.json`

- [ ] **Step 1: 抽取核心指标**

至少整理：
- `matched_reference_rows`
- `exported_sample_rows`
- 最终 `sample_count`
- `clean_loss`
- `fgsm_loss`
- `pgd_loss`
- `FGSM / clean`
- `PGD / clean`
- `clean_grad_finite_rate`
- `spearman_to_reference`

- [ ] **Step 2: 写报告**

报告至少包含四节：
- 实验设置
- 资产导出结果
- 白盒攻击结果
- 结论与当前限制

- [ ] **Step 3: 明确限制**

必须显式写明：
- 本阶段是 `4096` 固定上限大样本，不是全预测集全量攻击；
- `Transformer` 与 `TCN` 共用 `2025-01-01` 到 `2025-10-31` 时间区间；
- 结果仍是样本级攻击，不代表组合层回测结论。

## Chunk 4: 最终核查

### Task 6: 核查结果文件完整性

**Files:**
- Verify: `artifacts/transformer_attack_4096_v1/`
- Verify: `artifacts/tcn_attack_4096_v1/`
- Verify: `reports/transformer_whitebox_attack_4096_v1/`
- Verify: `reports/tcn_whitebox_attack_4096_v1/`
- Verify: `reports/transformer_tcn_4096白盒攻击实验报告.md`

- [ ] **Step 1: 列出结果文件**

Run:

```bash
find artifacts/transformer_attack_4096_v1 artifacts/tcn_attack_4096_v1 reports/transformer_whitebox_attack_4096_v1 reports/tcn_whitebox_attack_4096_v1 -maxdepth 2 -type f | sort
```

Expected: 四个目录下都能看到预期的 `.json` / `.pt` / `.csv` / `.md` 文件。

- [ ] **Step 2: 记录最终结论**

结论必须能回答三件事：
- 两个模型的大样本攻击是否跑通；
- 攻击强度相对小样本版是否稳定；
- 后续是否需要继续扩大到 `8192+` 或进入约束攻击迁移。

