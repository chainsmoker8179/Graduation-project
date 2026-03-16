# 部分股票白盒攻击回测实验报告

## 1. 实验目的

本轮实验的目标是验证这样一个更接近实盘扰动场景的设定：攻击并不作用于全部股票，而是仅对每个交易日随机抽取的一部分股票施加白盒扰动，再观察组合层面的回测指标是否出现退化。相较于前面的样本级攻击实验，这一设定更关注局部输入扰动能否穿透选股与组合构建流程，最终反映到超额收益、回撤和排序相关指标上。

## 2. 实验设置

实验对象为当前已迁移完成的 legacy LSTM 基线模型。攻击空间仍然是原始 `OHLCV` 窗口，攻击目标仍定义为最大化预测误差的均方误差（MSE）。测试区间覆盖 `2025-01-01` 至 `2025-10-31` 的完整测试期，并采用固定随机种子 `seed=0`。在每个交易日内，脚本从当日可预测股票中随机抽取 `5%` 作为候选攻击对象，并并行比较 `FGSM` 与 `PGD` 两种梯度型攻击。

为避免把“局部替换 clean 分数”本身带来的偏移误认为攻击效果，本轮同时构造四组分数表并重放同一套 Qlib 回测配置：`reference_clean` 表示原始 `pred.pkl`；`partial_clean` 仅将被抽中的股票替换为攻击链路下的 clean 前向预测；`partial_fgsm` 与 `partial_pgd` 则分别将同一批股票替换为 FGSM 和 PGD 扰动后的预测。主比较口径为 `partial_fgsm - partial_clean` 与 `partial_pgd - partial_clean`，辅助比较口径为 `partial_clean - reference_clean`。

需要说明的是，本轮使用的 partial asset 目录只包含 `matched_ohlcv_windows.pt`、`matched_reference.csv` 和 `normalization_stats.json`，不包含 `matched_feature_windows.pt`。因此 clean gate 在本轮主要承担“输入梯度有限且非空、clean 预测与 reference score 排序关系不过度偏离”的作用，而不额外使用 feature-level 对齐约束。正式运行前还修复了一处 clean gate 逻辑问题，即当未提供 `reference_features` 时，不再把 `feature_* = None` 误判为 gate 失败。

## 3. 样本覆盖率与可攻击性

按每日 `5%` 随机抽样后，候选攻击键总数为 `52,422`，对应完整测试集的 `5.009%`。其中，`50,874` 条样本可以在本轮 partial asset 中找到对应原始窗口，缺失样本为 `1,548` 条；最终 `50,874` 条可用样本全部通过 clean gate 并进入攻击阶段，对应的实际可攻击比例为 `4.861%`。这一结果说明，当前“按指定 key 定向导出原始窗口并回放局部攻击”的链路已经可以在完整测试区间上稳定工作，不再停留于此前仅覆盖几十条样本的 smoke 级验证。

## 4. 回测主结果

四组组合主指标如下表所示。

| 组合组别 | 年化超额收益(含费) | 最大回撤(含费) | RankIC 均值 | 信息比率(含费) |
| --- | ---: | ---: | ---: | ---: |
| `reference_clean` | 0.378584 | -0.092310 | 0.071965 | 2.777257 |
| `partial_clean` | 0.383355 | -0.093232 | 0.072007 | 2.866649 |
| `partial_fgsm` | 0.103333 | -0.143246 | 0.059019 | 0.891190 |
| `partial_pgd` | -0.014963 | -0.153809 | 0.057629 | -0.129149 |

将 `partial_clean` 视作主对照后，可以得到更直接的攻击退化量：

| 差值组别 | 年化超额收益(含费) | 最大回撤(含费) | RankIC 均值 | 信息比率(含费) |
| --- | ---: | ---: | ---: | ---: |
| `partial_clean - reference_clean` | 0.004771 | -0.000922 | 0.000043 | 0.089392 |
| `partial_fgsm - partial_clean` | -0.280023 | -0.050014 | -0.012988 | -1.975459 |
| `partial_pgd - partial_clean` | -0.398318 | -0.060577 | -0.014378 | -2.995798 |

从相对变化幅度看，FGSM 使年化超额收益相对 `partial_clean` 下降约 `73.05%`，最大回撤恶化约 `53.64%`，RankIC 均值下降约 `18.04%`；PGD 的作用更强，其年化超额收益相对下降约 `103.90%`，信息比率由正转负，最大回撤恶化约 `64.97%`，RankIC 均值下降约 `19.97%`。

## 5. 结果分析

首先，`partial_clean - reference_clean` 的偏移总体可控。虽然两者并非逐点完全相同，但其差值远小于攻击组带来的退化幅度：年化超额收益仅增加 `0.004771`，最大回撤仅恶化 `0.000922`，RankIC 均值变化仅为 `0.000043`。从日度层面看，`partial_clean - reference_clean` 的日度超额收益差值平均仅为 `2.00e-05`，日度 RankIC 差值平均仅为 `4.27e-05`。这说明，本轮“局部替换 clean 分数”本身并未改写组合结论，因此后续 `partial_fgsm` 和 `partial_pgd` 的退化可以主要归因于扰动样本，而不是归因于回测构造方式。

其次，局部攻击已经能够在组合层面产生稳定退化。相较于 `partial_clean`，FGSM 将年化超额收益从 `0.383355` 拉低到 `0.103333`，同时把最大回撤从 `-0.093232` 扩大到 `-0.143246`；PGD 则进一步把年化超额收益压到 `-0.014963`，并将最大回撤扩大到 `-0.153809`。这表明，即便每天只攻击 `5%` 的股票，扰动也已经足以改变组合排名与持仓结果，进而在累计收益和风险指标上形成可观测影响。

再次，排序相关指标的退化比日度收益更稳定。`fgsm_minus_partial_clean_rank_ic` 的日度均值为 `-0.01299`，其中 `99.0%` 的交易日为负；`pgd_minus_partial_clean_rank_ic` 的日度均值为 `-0.01438`，其中 `99.5%` 的交易日为负，且全样本最优值仍为负数。这说明，攻击对横截面排序质量的破坏并不是少数极端日驱动的偶发现象，而是在绝大多数交易日都存在一致方向的弱化。相比之下，日度超额收益差值虽然均值也为负，但其符号波动更大，这与组合收益同时受市场噪声、换仓路径和交易成本影响的事实一致。

最后，PGD 在当前预算约束下明显强于 FGSM。无论是年化超额收益、信息比率，还是最大回撤和 RankIC，PGD 均表现出更大的退化幅度。这与前面样本级攻击实验中的观察一致，即多步投影更新能够比单步 FGSM 更充分地利用输入预算，从而在保持同类约束的前提下制造更强的预测偏移。

## 6. 当前结论与后续工作

本轮结果给出两个当前可用的结论。第一，部分股票攻击回测链路已经打通，并且可以在完整测试区间上稳定运行，`partial_clean` 与 `reference_clean` 的差异仍处于可接受范围。第二，在“每日随机攻击 `5%` 股票、固定种子、固定预算”的设定下，FGSM 和 PGD 都已经能够把样本级扰动传导到组合层面，其中 PGD 的破坏效应更强。

当前版本仍有两个边界。其一，本轮尚未做多随机种子重复实验，因此暂不对统计显著性做结论，只能将其视为单次固定种子下的明确退化证据。其二，本轮 clean gate 没有接入 `matched_feature_windows.pt` 的中间特征比对，因此下一步可以在补齐 feature-level 参考窗口后，再做一轮更严格的 partial attack backtest 复核。如果后续需要进一步接近论文口径，最自然的扩展方向是增加多随机种子重复、不同攻击比例对比，以及对组合换手和行业暴露变化的补充分析。

## 7. 结果文件

- 回测汇总 JSON：[reports/partial_attack_backtest_seed0_ratio5_v1/backtest_summary.json](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/partial_attack_backtest_seed0_ratio5_v1/backtest_summary.json)
- 主比较表：[reports/partial_attack_backtest_seed0_ratio5_v1/comparison.csv](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/partial_attack_backtest_seed0_ratio5_v1/comparison.csv)
- 日度比较表：[reports/partial_attack_backtest_seed0_ratio5_v1/daily_comparison.csv](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/partial_attack_backtest_seed0_ratio5_v1/daily_comparison.csv)
- 运行说明：[reports/partial_attack_backtest_seed0_ratio5_v1/README.md](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/partial_attack_backtest_seed0_ratio5_v1/README.md)
