# Physical-Stat 多随机种子显著性分析设计

## 背景

当前 `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/` 已完成 `5%` 覆盖率、`seed=0..4` 的组合层回测，并形成主结果与中文稳定性报告。下一步需要补齐统计显著性证据，回答两个问题：

1. 在日度层面，攻击后相对 `partial_clean` 的收益与 `RankIC` 退化是否具有统计显著性。
2. 在路径指标层面，年化超额收益、信息比率与最大回撤的平均退化区间是否稳定落在不利方向。

## 目标

在不改动现有攻击与回测主流程的前提下，复用现有 LSTM 显著性分析模块，为 `physical_stat` 多随机种子结果目录生成：

- `significance_daily_metrics.csv`
- `significance_block_bootstrap.json`
- 中文显著性报告 `显著性检验报告.md`

以上文件统一写入 `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/`。

## 方案

### 方案 A：复用现有统计模块并参数化目录

- 将日度 panel 加载器从硬编码目录改为“接受任意多 seed 结果根目录”。
- 将显著性脚本从默认 `partial_attack_backtest_multiseed_ratio5_union` 扩展为可传入目标目录。
- 新增一个针对当前实验目录的中文报告脚本，直接读取统计产物与 `multiseed_summary_stats.csv` 生成结论。

优点：
- 改动最小，直接复用已有 `Wilcoxon + block bootstrap` 统计口径。
- 后续其他模型或其他约束目录也可复用。

缺点：
- 仍沿用当前统计框架，不在本轮引入更复杂的多重检验或层级模型。

### 方案 B：单独写一版 physical_stat 专用统计脚本

- 不动旧脚本，直接为 `physical_stat` 新建一套读取、统计、报告逻辑。

优点：
- 实现快，局部隔离。

缺点：
- 代码重复，后续维护成本更高。
- 统计口径和已有无约束分析容易漂移。

## 选择

采用方案 A。原因是这一步的核心不是发明新统计方法，而是把当前 `physical_stat` 主线接入已有、已验证的统计口径，保持无约束与受约束实验之间的可比性。

## 数据流

1. 从 `reports/partial_attack_backtest_multiseed_ratio5_physical_stat/seed_*/daily_comparison.csv` 读取五个 seed 的日度面板。
2. 对 `partial_fgsm` / `partial_pgd` 相对 `partial_clean` 的日度超额收益与日度 `RankIC` 做配对 `Wilcoxon` 检验。
3. 以 seed 内 block bootstrap、seed 间均值聚合的方式，估计年化超额收益、信息比率、最大回撤的退化区间。
4. 结合 `multiseed_summary_stats.csv` 与 bootstrap 结果输出中文显著性报告。

## 验证

- 单元测试覆盖：
  - 日度 panel 加载器支持自定义结果目录。
  - 显著性入口脚本能向指定目录输出统计文件。
  - 中文报告脚本能从最小输入生成关键结论。
- 实际验证：
  - 对 `partial_attack_backtest_multiseed_ratio5_physical_stat` 跑一遍统计生成流程。
  - 核对 `CSV/JSON/Markdown` 三类产物均落盘，且数值方向与既有主结果一致。
