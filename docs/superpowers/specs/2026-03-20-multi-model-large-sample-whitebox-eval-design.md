# Transformer / TCN 大样本白盒攻击正式实验设计

## 背景

当前仓库已经完成两步前置工作：

- `Transformer` 与 `TCN` 的本地重建路径已经通过 clean forward probe 验证；
- 基于统一入口 `scripts/run_whitebox_attack.py` 的样本级 white-box attack 已在两个模型上完成 smoke 验证，并在 `62` 样本与当前 probe 资产全量 `124` 样本上证明 `FGSM` / `PGD` 可以稳定抬高 `MSE`。

但这两版结果仍然属于“小样本或当前探查资产范围内的验证”。如果要把 `Transformer` 与 `TCN` 的白盒攻击结果提升到“正式实验”级别，还需要从完整 `pred.pkl` / `label.pkl` 重新构建更大规模的 raw-window 攻击资产，并按统一口径复跑攻击。

## 用户确认后的实验口径

用户已明确确认本阶段采用以下口径：

- 不直接导出“全部可重建样本”；
- 先做固定上限的大样本子集；
- 首版每个模型使用 `4096` 条样本；
- 与已有 LSTM 白盒攻击链路保持相同攻击目标与预算设定；
- 本阶段优先产出正式实验结果，不做新的攻击算法设计。

## 目标

在不修改攻击算法定义的前提下，为 `Transformer` 与 `TCN` 各自构建一版 `4096` 样本的大样本白盒攻击正式实验，形成可直接引用的：

- 攻击资产导出结果；
- clean gate 指标；
- `FGSM` / `PGD` 攻击结果；
- 样本级结果文件；
- 中文实验总结材料。

## 关键设计取舍

### 1. 固定大样本上限，而不是全量导出

完整 `pred.pkl` / `label.pkl` 的规模已经达到百万级，直接导出全部 raw-window 资产会带来明显的 CPU、I/O 和磁盘成本，也会让首版正式实验变得过重。

因此本阶段取固定上限 `4096`，其优点是：

- 足以比当前 `124` 样本版本更稳定；
- 成本可控，便于在本地 `qlib` 环境中完成；
- 后续若需要 `8192` 或更大规模，可在同一链路上直接扩展。

### 2. 两模型统一到共同时间区间

当前资产的时间覆盖存在差异：

- `Transformer pred.pkl` 覆盖到 `2025-12-31`；
- `TCN pred.pkl` 只覆盖到 `2025-10-31`。

为了保证横向对比口径一致，本阶段统一采用共同测试区间：

- `test_start_time = 2025-01-01`
- `test_end_time = 2025-10-31`

这样可以避免“模型结果差异实际上来自测试时间范围不同”的干扰。

### 3. 保持攻击定义不变

本阶段不引入任何新的攻击约束或新目标函数，统一复用现有样本级 white-box 定义：

- 攻击目标：最大化 `MSE`
- 输入空间：原始 `OHLCV`
- 扰动预算：价格列 `1%`、成交量列 `2%`
- 攻击方法：`FGSM` 与 `PGD(steps=5, step_size=0.25)`
- 预处理链路：`LegacyLSTMFeatureBridge -> RobustZScoreNorm -> Fillna -> model`

这样得到的结果可以直接与当前 LSTM 无约束白盒攻击基线对齐。

## 范围内内容

- 复用 `scripts/export_lstm_attack_assets.py`，从完整 `pred.pkl` / `label.pkl` 重导 `Transformer` 与 `TCN` 的 `4096` 样本攻击资产；
- 统一使用 `2025-01-01` 到 `2025-10-31` 时间区间；
- 运行 `scripts/run_whitebox_attack.py`，得到新的大样本攻击结果；
- 汇总 clean、FGSM、PGD 指标；
- 产出一份中文实验报告，便于后续论文与答辩材料复用。

## 范围外内容

- 不在本阶段实现跨模型 transfer attack；
- 不在本阶段迁移金融物理约束或统计约束攻击；
- 不在本阶段进行组合层回测；
- 不在本阶段进行显著性检验与 block bootstrap；
- 不在本阶段继续扩大到 `8192+` 样本。

## 产物约定

### 攻击资产

新增两份大样本资产目录：

```text
artifacts/
  transformer_attack_4096_v1/
  tcn_attack_4096_v1/
```

每个目录至少包含：

- `matched_reference.csv`
- `matched_ohlcv_windows.pt`
- `matched_feature_windows.pt`
- `normalization_stats.json`
- `export_summary.json`

### 攻击结果

新增两份正式实验结果目录：

```text
reports/
  transformer_whitebox_attack_4096_v1/
  tcn_whitebox_attack_4096_v1/
```

每个目录至少包含：

- `attack_summary.json`
- `sample_metrics.csv`
- `attack_report.md`

### 中文总结

新增一份总报告，例如：

```text
reports/transformer_tcn_4096白盒攻击实验报告.md
```

用于汇总：

- 资产导出规模；
- clean gate 稳定性；
- FGSM / PGD 抬高误差幅度；
- 与小样本版本相比是否稳定；
- 两模型之间的攻击强弱差异。

## 验收标准

本阶段完成的判定标准如下：

- 两个模型都成功导出 `4096` 条或接近该上限的可攻击样本；
- `clean_grad_finite_rate = 1.0`；
- clean gate 通过；
- `fgsm_loss > clean_loss`；
- `pgd_loss > fgsm_loss` 或至少 `pgd_loss > clean_loss` 且总体攻击强度不弱于 FGSM；
- 新结果成功写入 `reports/`；
- 中文总结能够说明样本规模、攻击增幅与口径限制。

## 风险与应对

### 风险 1：大样本导出过慢

应对：

- 先按 `4096` 固定上限执行；
- 如导出耗时过长，不改口径，只记录耗时并保持现有方案。

### 风险 2：导出后存在非有限 label 或特征

应对：

- 保持当前 runner 的 finite 过滤逻辑；
- 报告中显式记录最终 usable sample count，而不是盲目声称固定为 `4096`。

### 风险 3：两模型最终 usable sample count 不完全一致

应对：

- 允许因 finite 过滤产生少量偏差；
- 报告中同时给出 `matched_reference_rows`、`exported_sample_rows` 与最终 `sample_count`。

## 结论

这次工作不是新的算法开发，而是把已经跑通的 `Transformer` / `TCN` 白盒攻击链路提升到“大样本正式实验”级别。首版采用共同时间区间 `2025-01-01` 到 `2025-10-31`、每模型 `4096` 条样本的固定口径，在计算成本、可比性和论文可用性之间取得平衡。
