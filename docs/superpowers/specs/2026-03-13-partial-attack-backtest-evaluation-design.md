# 部分股票白盒攻击下的回测退化验证设计

## 1. 背景

当前项目已经完成三项关键前置工作。第一，可导 Alpha158 因子模块与 BPDA 算子验证已经建立，说明离散算子在前向硬语义一致、反向软近似可导这一策略下是可用的。第二，端到端梯度回传实验已经证明，在真实 Qlib 数据上，`OHLCV -> 因子 -> loss` 的整体梯度链路可以稳定回传。第三，针对历史已训练 LSTM 基线模型的原始 `OHLCV` 白盒攻击链路已经搭建完成，并在 `expanded_v6` 版本上获得了相对稳定的 clean 对齐与 FGSM/PGD 攻击响应。

到这一步，当前实验回答的仍然主要是“预测误差是否会被放大”。它还没有回答另一个更接近实际量化使用场景的问题：如果对测试集中的一部分股票施加对抗扰动，那么下游组合层面的回测指标是否会出现明显退化，例如超额收益下降、最大回撤上升、RankIC 下降。

本设计文档就是为这一问题定义第一版验证方案。

## 2. 目标

第一版实验只针对当前唯一已迁移完成的已训练 `LSTM` 基线模型，在完整测试区间上验证如下命题：

当每个交易日随机抽取固定比例的股票，并仅对这部分股票施加原始 `OHLCV` 空间中的白盒 `FGSM` 与 `PGD` 扰动时，使用这些 adversarial score 重新构建组合并回测，组合层面的关键指标是否相对 clean 对照组发生退化。

本轮重点观察三类指标：

1. 超额收益率是否下降。
2. 最大回撤是否上升。
3. `RankIC` 是否下降。

## 3. 非目标

本轮设计明确不做以下内容：

1. 不同时比较多个模型，当前只评估一个已训练 `LSTM`。
2. 不在第一版中做多随机种子显著性检验。
3. 不在第一版中做多攻击覆盖率扫描，固定为每日 `5%`。
4. 不改变现有攻击目标，继续沿用“最大化预测误差（MSE）”的白盒攻击目标。
5. 不把“单次固定随机种子下的明显退化”直接表述为严格统计学意义上的显著性结论。

## 4. 核心实验设计

### 4.1 攻击单位

攻击单位定义为测试集中的 `(datetime, instrument)` 样本键。实验以完整测试区间的 `pred.pkl` 为主索引，对每个交易日的横截面股票池单独处理。

### 4.2 攻击覆盖方式

每个交易日随机抽取固定 `5%` 的股票作为攻击目标，随机种子固定为 `seed=0`。该抽样结果会保存为一份固定的 `attack_mask`，后续所有攻击器与对照组都复用这同一份 mask。

每日随机 `5%` 的设计意图有两点：

1. 避免只攻击最容易影响组合的极端股票，从而获得一种更中性的局部攻击评估。
2. 允许后续自然扩展到多覆盖率或多随机种子版本。

### 4.3 四组对照

本轮不只比较 `clean` 和 `adv` 两组，而是构造四组完整分数表：

1. `reference_clean`
   - 全部股票分数直接来自历史 `pred.pkl`。
   - 它代表旧版 Qlib 运行下的原始基线。

2. `partial_clean`
   - 每日随机抽中的 `5%` 股票，不再使用 `pred.pkl`，而是使用当前攻击链路在线重建得到的 clean score。
   - 其余 `95%` 股票仍使用 `pred.pkl`。
   - 它的作用是隔离“在线重建链路残余误差”本身对回测的影响。

3. `partial_fgsm`
   - 在与 `partial_clean` 完全相同的攻击股票子集上，将该 `5%` 股票的分数替换为 `FGSM adversarial score`。
   - 其余股票保持 `pred.pkl`。

4. `partial_pgd`
   - 同上，只是替换为 `PGD adversarial score`。

### 4.4 主比较口径

主比较定义为：

1. `partial_fgsm` 与 `partial_clean`
2. `partial_pgd` 与 `partial_clean`

辅助比较定义为：

1. `partial_clean` 与 `reference_clean`

之所以这样设计，是因为当前原始 `OHLCV -> Torch bridge -> LSTM` 在线链路即使在 `expanded_v6` 上已经明显改进，仍不能默认与旧版 `pred.pkl` 完全等价。若直接用 `adversarial` 结果去对比 `reference_clean`，就会把两种效应混在一起：

1. 在线重建链本身与历史 `pred.pkl` 的残余差异。
2. 真实由 adversarial perturbation 带来的额外退化。

只有引入 `partial_clean` 作为局部替换对照，才能更清楚地回答“攻击额外造成了多少组合层退化”。

## 5. 数据流

第一版完整数据流如下：

1. 读取完整测试区间的 `pred.pkl` 与 `label.pkl`。
2. 按交易日构建随机 `5%` 的 `attack_mask`。
3. 对 `attack_mask` 中的样本尝试重建原始 `80x5` `OHLCV` 窗口。
4. 对可成功重建且通过 clean gate 的样本，分别生成：
   - `bridge_clean_score`
   - `fgsm_score`
   - `pgd_score`
5. 以 `pred.pkl` 为底表，组装四份完整分数表：
   - `reference_clean_scores`
   - `partial_clean_scores`
   - `partial_fgsm_scores`
   - `partial_pgd_scores`
6. 将这四份分数表在完全相同的 Qlib 回测配置下分别重放。
7. 汇总四组回测指标，并计算主比较差值与日度差值。

## 6. 样本覆盖与可攻击性处理

由于不是所有被抽中的样本都一定能成功生成对抗分数，因此设计中区分两层集合：

1. `selected_keys`
   - 每日随机 `5%` 抽样得到的目标样本。

2. `attackable_keys`
   - 在 `selected_keys` 中，能够完成原始窗口重建、通过 clean gate，并成功生成 `clean / FGSM / PGD` 三类分数的样本。

若某个被选中的样本最终不可攻击，则其分数回退为 `pred.pkl` 原始分数，整轮回测不会因此中断。同时，实验必须显式记录：

1. 总体 `selected_ratio`
2. 总体 `attackable_ratio`
3. 每日可攻击比例
4. clean gate 失败原因统计

这一点非常关键。若实际 `attackable_ratio` 明显低于预设的 `5%`，则即便组合层退化不强，也不能直接解释为“模型稳健”，因为攻击覆盖本身可能已经被样本可用性稀释。

## 7. 攻击配置

第一版攻击器并行比较 `FGSM` 与 `PGD` 两种方法，但不改变当前已验证通过的攻击链路配置。换言之，攻击阶段应直接沿用当前 `expanded_v6` 所使用的：

1. 模型权重与配置。
2. `OHLCV -> 20 特征 -> RobustZScoreNorm -> Fillna -> LSTM` 在线链路。
3. 相对幅度约束与投影方式。
4. `FGSM` 与 `PGD` 的预算、步数和步长设置。

这样设计的原因是，本轮要回答的是“组合层面会不会退化”，而不是重新寻找新的攻击超参数。若本轮再同时改变攻击预算或 clean gate 口径，就会导致回测结果难以和当前 `expanded_v6` 攻击结论对齐。

## 8. 回测协议

四组分数表必须使用完全一致的回测配置，包括但不限于：

1. 相同测试区间。
2. 相同选股策略与调仓频率。
3. 相同 benchmark。
4. 相同交易成本设置。
5. 相同 executor / exchange 配置。

否则，指标差异将无法明确归因到攻击分数本身。

本轮建议直接复用现有 notebook 与 Qlib 回测资产中已经使用过的那套回测口径，以确保结果与历史基线可比。

## 9. 评估指标

### 9.1 主指标

第一版锁定以下三项主结论指标：

1. 超额收益率
2. 最大回撤
3. `RankIC mean`

### 9.2 辅助指标

除主指标外，再记录以下辅助指标用于诊断：

1. `IC mean`
2. `RankICIR`
3. 年化波动率
4. 换手率
5. 日度超额收益序列
6. 日度 `RankIC` 序列

## 10. 第一版结果解释与“显著变化”口径

由于本轮只使用单次固定随机种子，因此本轮不直接宣称“统计显著变化”。更合理的表述是：在单次固定种子的完整测试回测中，是否观察到明确的退化证据。

第一版将采用“双层判定”：

### 10.1 汇总指标方向判定

相对于 `partial_clean`，若出现下列方向性变化，则认为攻击方向符合预期：

1. `partial_fgsm` 或 `partial_pgd` 的超额收益下降。
2. `partial_fgsm` 或 `partial_pgd` 的最大回撤上升。
3. `partial_fgsm` 或 `partial_pgd` 的 `RankIC mean` 下降。

### 10.2 日度诊断支持

对每个交易日构造：

1. `daily_excess_return_diff = adv - partial_clean`
2. `daily_rankic_diff = adv - partial_clean`

如果这些日度差值同时满足以下趋势，则认为单次实验对“退化”提供了较强支撑：

1. 日度差值均值为负。
2. 日度差值中位数为负。
3. 负值天数占比明显超过 `50%`。

对于最大回撤，因为其不是日度可加指标，本轮仍以汇总值变化为准。

## 11. 第一版结果解读框架

本设计预先定义四种可能结果：

### 11.1 理想情形

若 `partial_clean` 与 `reference_clean` 非常接近，而 `partial_fgsm / partial_pgd` 相对 `partial_clean` 明显退化，则可以较有力地说明：

1. 在线攻击链路本身没有引入太大额外偏差。
2. 回测退化主要来自 adversarial perturbation。

### 11.2 clean 残差偏大的情形

若 `partial_clean` 相对 `reference_clean` 已经出现明显偏移，则说明在线链路残余误差仍不可忽略。此时即便 `FGSM/PGD` 继续变差，也应更谨慎地表述为：

“在当前在线攻击链近似下，局部 adversarial score 会进一步恶化回测指标。”

而不应直接把该结论等同于“原始历史基线在相同攻击下必然出现同等幅度退化”。

### 11.3 攻击效果偏弱的情形

若 `FGSM/PGD` 相对 `partial_clean` 只有轻微退化，则优先检查：

1. `attackable_ratio` 是否太低。
2. 被攻击股票对组合排序的影响是否有限。

因此，结果弱不应第一时间被解释为“模型稳健”。

### 11.4 PGD 显著强于 FGSM 的情形

若 `PGD` 在回测层面显著强于 `FGSM`，这是合理结果，说明多步优化更能把局部输入扰动转化为横截面排序破坏。后续若继续放大实验规模，可优先以 `PGD` 作为主攻击器。

## 12. 必要输出物

第一版实验至少应输出以下文件：

1. `attack_mask`
2. `attack_generation_summary.json`
3. `scores_reference`
4. `scores_partial_clean`
5. `scores_partial_fgsm`
6. `scores_partial_pgd`
7. `backtest_summary.json`
8. `daily_comparison.csv`
9. 中文实验报告

其中 `attack_generation_summary.json` 至少记录：

1. 目标攻击样本数
2. 实际可攻击样本数
3. 每日可攻击比例
4. clean gate 失败原因

## 13. 第一版成功标准

本轮第一版可以被认为达标，当且仅当同时满足以下三点：

1. `partial_clean` 相对 `reference_clean` 的回测漂移处于可解释范围内。
2. `partial_fgsm` 或 `partial_pgd` 相对 `partial_clean` 在主指标上出现预期方向退化。
3. 日度超额收益与日度 `RankIC` 的差值统计支持上述方向。

## 14. 第二版升级方向

若第一版跑通，下一阶段自然升级为真正意义上的统计检验版本，包括：

1. 多随机种子重复实验。
2. 多攻击覆盖率，例如 `5% / 10% / 20%`。
3. 对跨种子指标差值做 bootstrap 置信区间或配对检验。
4. 将当前“最大化 MSE”的攻击目标逐步扩展到更接近组合构建目标的排序型攻击目标。

## 15. 一句话总结

第一版方案的核心是：在完整测试区间上，每日随机攻击 `5%` 股票，构造 `reference_clean / partial_clean / partial_fgsm / partial_pgd` 四组分数，使用统一回测配置重放，并以 `partial_clean` 为主对照，判断 adversarial score 是否会在组合层面带来超额收益下降、最大回撤上升和 `RankIC` 下降。
