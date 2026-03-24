# 组合层 5% 多随机种子 `physical_stat` 回测设计

## 背景

当前仓库已经具备一条可运行的组合层部分股票攻击回测链路，其核心流程为：

1. 从完整 `pred.pkl` 上按交易日随机选择固定比例股票；
2. 对可攻击样本执行样本级 `FGSM` / `PGD`；
3. 用攻击后分数替换被选股票的 clean 分数；
4. 运行四组回测：
   - `reference_clean`
   - `partial_clean`
   - `partial_fgsm`
   - `partial_pgd`
5. 汇总年化超额收益、最大回撤、`rank_ic_mean`、信息比率等指标。

此前这条组合层链路已经在“每日随机攻击 `5%` 股票、随机种子 `0..4`”的口径下验证过跨 seed 稳定性，但使用的仍是旧版无约束攻击器，即样本级攻击目标仅最大化 `MSE`，不带金融物理约束和统计约束。

而在最近完成的样本级工作中，`physical_stat` 受约束攻击已经在 `497`、`1978` 与 `3967` 三档有效样本规模上稳定达到 `strict_attack_success_pgd = True`。因此，当前下一步的自然目标，不是继续沿用旧口径，而是把组合层 5% 多随机种子回测切换到当前默认的 `physical_stat` 约束版攻击器上，验证在更合理的金融约束下，组合退化是否仍然稳定存在。

## 目标

在不推翻现有组合层回测框架的前提下，将“每日随机攻击 `5%` 股票、随机种子 `0..4`”的组合层实验切换到 `physical_stat` 受约束攻击口径，并输出与现有旧实验同结构的新结果目录和汇总文件。

具体目标包括：

1. 让组合层脚本支持 `constraint_mode`、统计约束阈值和惩罚系数；
2. 重建本地可用的多 seed `5%` 共享 raw-only 攻击资产；
3. 在 `seed=0..4` 上批量运行 `physical_stat` 版 `FGSM` / `PGD` 回测；
4. 生成与当前 plotting / significance / daily panel 工具兼容的新多 seed 结果目录；
5. 给出中文实验报告，回答“组合层退化在 `physical_stat` 口径下是否仍稳定成立”。

## 范围内内容

- 将 [scripts/run_partial_attack_backtest.py](/home/chainsmoker/qlib_test/scripts/run_partial_attack_backtest.py) 的样本级攻击器扩展到支持：
  - `none`
  - `physical`
  - `physical_stat`
- 透传以下攻击参数：
  - `constraint_mode`
  - `tau_ret`
  - `tau_body`
  - `tau_range`
  - `tau_vol`
  - `lambda_ret`
  - `lambda_candle`
  - `lambda_vol`
- 新增多随机种子驱动脚本，统一执行 `seed=0..4` 的单次回测并汇总统计结果。
- 新增或重建“多 seed、`ratio=5%`、共享 key 并集”的 raw-only 攻击资产。
- 生成新的结果目录、汇总 CSV / JSON 和中文报告。

## 范围外内容

- 不在本阶段扩展到 `Transformer` 或 `TCN`。
- 不在本阶段扩展到 `1% / 10%` 比例 sweep。
- 不在本阶段接入显著性检验、论文图重绘或答辩图更新。
- 不在本阶段引入组合级风险约束或横截面统计约束。
- 不在本阶段修改已有旧实验目录或覆盖其结果。

## 设计原则

### 1. 复用现有组合层框架

组合层已有的分组、回测、对比和汇总框架已经足够稳定，因此本轮不新造一套回测主流程，而是在现有框架上做增量改造。

### 2. 样本级与组合层口径一致

组合层攻击器必须与当前样本级默认攻击器保持一致，即使用 `physical_stat` 约束目标和当前已验证稳定的默认超参。这是为了避免样本级结论与组合层结论使用两套不同攻击定义。

### 3. 结果目录不覆盖旧实验

旧的无约束多 seed 结果仍有参考价值，因此本轮新结果必须写入新的报告目录，避免把不同攻击口径的结果混写到同一路径下。

### 4. 尽量复用现有分析工具格式

新的多 seed 结果目录结构应尽量与当前：

- `reports/partial_attack_backtest_multiseed_ratio5_union/`

保持一致，从而兼容现有 `daily_panel`、plotting 和证据汇总脚本。

## 推荐方案

推荐采用“现有单次脚本增量改造 + 新增多 seed 驱动脚本 + 重建共享 raw-only 资产”的方案。

相比“完全重写组合回测链路”或“仅用 shell 循环包装旧脚本”，这一方案的优势是：

1. 改动集中，主逻辑仍留在现有入口脚本中；
2. 后续无论是继续做比例 sweep、显著性检验还是图表复用，数据格式都更稳定；
3. 风险清晰，出问题时能在“资产构造 / 单 seed 攻击回测 / 多 seed 汇总”三个边界上快速定位。

## 总体方案

### 1. 单次回测脚本支持受约束攻击

在 [scripts/run_partial_attack_backtest.py](/home/chainsmoker/qlib_test/scripts/run_partial_attack_backtest.py) 中新增攻击参数，使 `_build_attack_fn(...)` 在样本级生成 `FGSM` 与 `PGD` 分数时，可以调用：

- `fgsm_maximize_mse(..., constraint_mode=..., tau_*=..., lambda_*=...)`
- `pgd_maximize_mse(..., constraint_mode=..., tau_*=..., lambda_*=...)`

默认值直接对齐当前样本级稳定配置：

- `constraint_mode = physical_stat`
- `tau_ret = 0.005`
- `tau_body = 0.005`
- `tau_range = 0.01`
- `tau_vol = 0.05`
- `lambda_ret = 0.8`
- `lambda_candle = 0.4`
- `lambda_vol = 0.3`

同时保留 `none` 和 `physical` 作为可选模式，避免把组合层脚本再次写死为单一路径。

### 2. 重建多 seed `5%` 共享 raw-only 资产

由于当前工作区中的旧 union 资产目录已经不存在，而组合层多 seed 回测的输入覆盖率又高度依赖“所选 key 的并集资产”，因此需要先重建共享资产。

推荐新增一个辅助脚本，完成以下两步：

1. 对完整 `reference_scores` 按 `seed=0..4`、`ratio=5%` 生成每日攻击 key；
2. 取这些 key 的并集后，调用现有 exporter，用 `requested_keys_csv` 导出 raw-only 资产。

该资产仅保留：

- `matched_reference.csv`
- `normalization_stats.json`
- `matched_ohlcv_windows.pt`
- `export_summary.json`

不要求导出 `matched_feature_windows.pt`，因为当前组合层回测更关注“可攻击覆盖率 + 回测退化方向”，而不是再次进行 feature-level clean gate 收紧。

### 3. 新增多随机种子驱动脚本

推荐新增单独的多 seed 驱动脚本，职责固定为：

1. 循环执行 `seed=0..4` 的单次组合层回测；
2. 将每个 seed 的 `backtest_summary.json` 与 `comparison.csv` 提取为结构化行；
3. 生成：
   - `multiseed_seed_metrics.csv`
   - `multiseed_summary_stats.csv`
   - `multiseed_pgd_vs_fgsm.json`
4. 生成中文报告。

这一脚本不应重新实现攻击或回测主流程，而只作为 orchestration 层，保证逻辑清晰。

## 数据流

新的完整数据流如下：

1. 读取 `origin_model_pred/LSTM/prediction/pred.pkl`
2. 按 `seed=0..4`、`ratio=5%` 生成每日攻击 key
3. 对所有 key 取并集并导出共享 raw-only 攻击资产
4. 对每个 seed：
   - 基于完整 `pred.pkl` 构造当日 `5%` 攻击 mask
   - 从共享资产中选取对应可攻击子集
   - 运行 `physical_stat` 版 `FGSM` / `PGD`
   - 生成 `partial_clean / partial_fgsm / partial_pgd`
   - 运行四组回测并落盘
5. 聚合所有 seeds 的指标差值
6. 生成总表、稳定性摘要与中文报告

## 输出目录

推荐使用新的结果目录：

- `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/`

目录结构建议为：

- `seed_0/`
- `seed_1/`
- `seed_2/`
- `seed_3/`
- `seed_4/`
- `multiseed_seed_metrics.csv`
- `multiseed_summary_stats.csv`
- `multiseed_pgd_vs_fgsm.json`
- `多随机种子稳定性报告.md`

共享资产目录建议为：

- `artifacts/lstm_attack_partial_seed0to4_ratio5_union_rawonly/`

## 关键文件边界

### [scripts/run_partial_attack_backtest.py](/home/chainsmoker/qlib_test/scripts/run_partial_attack_backtest.py)

职责：

- 单个 `seed`、单个 `ratio`、单个模型的组合层回测入口
- 负责单次攻击、分数组装、回测、结果写盘

本轮改动：

- 新增约束攻击参数解析与透传
- 报告中记录攻击模式与约束超参

### [partial_attack_backtest.py](/home/chainsmoker/qlib_test/partial_attack_backtest.py)

职责：

- 组合层 mask、subset 选择、分数替换、回测结果汇总

本轮原则：

- 尽量不改核心逻辑
- 若需要，则仅补轻量辅助函数，不引入新的攻击逻辑

### 新增 `union keys / raw-only asset` 辅助脚本

职责：

- 生成多 seed `5%` 的 key 并集
- 调用现有 exporter 导出共享 raw-only 资产

### 新增多 seed 驱动脚本

职责：

- 循环运行 `seed=0..4`
- 汇总单 seed 结果为总表和报告

## 验证方案

### 单元测试

至少补以下测试：

1. 单次组合层脚本会把 `constraint_mode / tau / lambda` 正确传给攻击器；
2. 多 seed 汇总脚本能正确聚合多个 seed 的指标差值；
3. 如果新增 raw-only 资产导出辅助脚本，则验证其生成的 key 并集规模和路径写盘逻辑。

### 实验验证

主实验验证指标固定为：

- `attackable_count`
- `partial_clean - reference_clean`
- `partial_fgsm - partial_clean`
- `partial_pgd - partial_clean`

重点观察：

- 年化超额收益(含费)
- 最大回撤(含费)
- `rank_ic_mean`
- 信息比率(含费)

成功标准是：

1. 多个 seed 的 `attackable_count` 保持稳定；
2. `partial_pgd - partial_clean` 在上述主指标上退化方向一致；
3. `PGD` 在收益和排序能力上的破坏强度不弱于 `FGSM`。

## 风险与应对

### 风险 1：共享资产覆盖率不足

如果共享 raw-only 资产缺失过多 key，多 seed 之间的 `attackable_count` 会出现额外波动。

应对：

- 在资产导出摘要中记录 `matched_reference_rows`、`exported_sample_rows` 和 `missing_raw_keys`
- 在多 seed 汇总中单独记录每个 seed 的 `selected_available_count` 与 `attackable_count`

### 风险 2：大批量 clean gate 梯度阈值不适配

当前 `clean_grad_mean_abs` 会随 batch 大小增大而下降。组合层脚本如果继续使用固定阈值，可能在大 batch 下误判失败。

应对：

- 沿用当前经验：优先通过 `attack_batch_size` 控制单次攻击 batch
- 若需要，仅对组合层实验临时降低 `min_clean_grad_mean_abs`
- 本阶段不顺手修改全局 clean gate 设计

### 风险 3：结果目录格式与现有分析脚本不兼容

如果多 seed 新目录与旧目录结构差异过大，后续 plotting / significance 工具会失效。

应对：

- 保持 `seed_i/` 子目录和顶层汇总文件命名与旧目录尽量一致
- 新目录名只区分攻击口径，不改字段名

## 结论

本阶段最合理的推进方式，是在现有组合层回测框架上增量接入 `physical_stat` 约束攻击器，并重建一个服务于 `seed=0..4`、`ratio=5%` 的共享 raw-only 资产。这样既能保持与当前样本级主结论一致，又能最大程度复用已有组合层回测、统计和可视化工具链，快速回答“在更合理的金融约束下，组合层退化是否仍稳定存在”这一核心问题。
