# LSTM 样本级受约束白盒攻击扩样实验报告

## 1. 实验目标

此前样本级白盒攻击链路已经在较小样本规模上跑通，但原始可用攻击样本仅有 `62` 个。该规模更适合作为 smoke 验证，不足以支撑更稳定的样本级攻击统计。因此本轮工作的目标是：

1. 将 LSTM 样本级攻击资产扩展到更大的测试样本集。
2. 在扩样资产上重新验证 `none / physical / physical_stat` 三种攻击模式。
3. 检查扩样后 clean gate 是否稳定通过，并评估约束攻击在更大样本规模下是否仍然有效。

## 2. 导出阶段的问题定位与修复

### 2.1 初次扩样失败现象

在首次尝试导出 `512` 个候选样本后，导出脚本得到：

- `matched_reference_rows = 512`
- `exported_sample_rows = 498`

但随后运行攻击脚本时，clean gate 失败，失败指标为：

- `clean_grad_finite_rate = 0.9990662932395935`

该指标没有达到 clean gate 对“梯度全部有限”的要求，说明扩样样本中至少有一个样本会在损失或梯度计算中产生 `NaN`。

### 2.2 根因分析

对扩样资产逐样本排查后，定位到异常样本键值为：

- `('2025-03-20 00:00:00', 'SH600590')`

进一步检查发现：

- 原始 `OHLCV` 输入是有限值；
- 前向提取的特征是有限值；
- 模型预测值是有限值；
- 但该样本对应的 `label` 为 `NaN`。

由于攻击目标采用 `MSE(pred, y)`，因此只要 `y` 中存在 `NaN`，clean loss 与输入梯度都会被污染，最终导致 clean gate 失败。

### 2.3 修复方式

问题出在攻击资产导出流程的 `build_matched_reference(...)`：此前该函数只做了 `pred.pkl` 与 `label.pkl` 的交集匹配，但没有显式去除非有限的 `score` 或 `label` 行。

本轮修复在 [scripts/export_lstm_attack_assets.py](/home/chainsmoker/qlib_test/scripts/export_lstm_attack_assets.py) 中加入了如下过滤逻辑：

```python
merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
```

同时，在 [tests/test_export_lstm_attack_assets.py](/home/chainsmoker/qlib_test/tests/test_export_lstm_attack_assets.py) 中新增了针对非有限 `score/label` 的单元测试，确保导出前会主动剔除这些异常行。修复后，导出相关测试共 `11` 项，已全部通过。

## 3. 扩样后攻击资产情况

修复后重新执行扩样导出，结果如下：

- `matched_reference_rows = 512`
- `exported_sample_rows = 497`
- `missing_raw_keys = 15`
- `exported_feature_rows = 512`

这说明：

1. 共有 `512` 个候选参考样本满足 `pred.pkl` 与 `label.pkl` 的交集匹配，且 `score`、`label` 均为有限值。
2. 最终只有 `497` 个样本成功导出了原始 `OHLCV` 窗口。
3. 缺失的 `15` 个样本并非标签异常，而是原始行情窗口在当前导出条件下不可用，因此被正常跳过。

从 clean gate 指标看，扩样后的攻击资产已经恢复正常：

- `clean_grad_finite_rate = 1.0`
- `spearman_to_reference = 0.9960`
- `feature_mae_to_reference = 0.0352`

这说明修复后的扩样资产在预测对齐和梯度有限性上都满足攻击前提。

## 4. 攻击设置

本轮实验沿用此前已经验证过的 LSTM 样本级攻击链路，主要设置如下：

- 模型：原始训练好的 LSTM
- 输入：`80 x 5` 的原始 `OHLCV` 时间窗
- 攻击目标：最大化预测误差 `MSE(pred, y)`
- 扰动方式：
  - `FGSM`
  - `PGD`
- 扰动约束：
  - 价格列按当前值百分比限制，`price_epsilon = 0.01`
  - 成交量列按当前值百分比限制，`volume_epsilon = 0.02`
- PGD 参数：
  - `pgd_steps = 5`
  - `pgd_step_size = 0.25`
- 约束模式：
  - `none`：仅做幅度约束，不保证金融物理合法
  - `physical`：加入 K 线与成交量物理合法投影
  - `physical_stat`：在 `physical` 基础上继续加入统计约束惩罚

其中 `physical_stat` 的默认惩罚系数沿用当前项目中已调好的推荐值：

- `lambda_ret = 0.8`
- `lambda_candle = 0.4`
- `lambda_vol = 0.3`

## 5. 扩样后的主要结果

### 5.1 三种模式在 497 个样本上的对比结果

| 模式 | 样本数 | clean_loss | fgsm_loss | pgd_loss | PGD / clean | objective_pgd | PGD strict success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| none | 497 | 0.01187 | 0.12151 | 0.24905 | 20.98x | 0.24905 | False |
| physical | 497 | 0.01187 | 0.08354 | 0.13225 | 11.14x | 0.13225 | True |
| physical_stat | 497 | 0.01187 | 0.08354 | 0.08472 | 7.14x | 0.02760 | True |

### 5.2 结果解读

可以看到，三种模式都显著放大了预测误差，但其含义并不相同。

`none` 模式下，PGD 将 `clean_loss` 从 `0.01187` 提升到了 `0.24905`，放大量达到 `20.98x`，是三种模式中攻击最强的一种。但该模式没有金融物理约束，因此生成的扰动不保证满足 `high >= max(open, close)`、`low <= min(open, close)`、成交量非负等条件，所以其 `strict_attack_success_pgd = False` 是符合预期的。也就是说，它证明“梯度驱动的误差放大能力”很强，但不能直接视为金融场景下可接受的攻击样本。

`physical` 模式下，PGD 仍能把损失提升到 `0.13225`，相对于 clean 提升 `11.14x`，并且满足物理合法性，因此 `strict_attack_success_pgd = True`。这说明仅加入金融物理约束后，攻击强度虽然下降，但仍然保持显著有效。

`physical_stat` 模式下，PGD 的 `pgd_loss = 0.08472`，相对于 clean 仍达到 `7.14x` 的误差放大；同时 `objective_pgd = 0.02760 > objective_clean = 0.01187`，并保持 `strict_attack_success_pgd = True`。这表明在进一步加入统计约束后，攻击虽然被明显收紧，但并未失效，而是转化为“更保守但仍可成功”的受约束对抗扰动。

## 6. 统计约束的实际作用

从 `physical_stat` 结果可以更清楚地看到统计约束的效果：

- `ret_penalty_pgd = 0.05651`
- `candle_penalty_pgd = 0.02978`
- `vol_penalty_pgd = 0.0`
- `mean_abs_ret_shift_pgd = 0.00372`
- `mean_abs_dlogvol_shift_pgd = 0.01950`

这些结果表明，统计约束主要压制了收益率与 K 线形态相关的扰动，而成交量方向在当前默认参数下并未形成额外惩罚压力。与 `physical` 模式相比，`physical_stat` 的 PGD 损失从 `0.13225` 下降到 `0.08472`，预测偏移幅度也从 `0.2360` 降到 `0.1664`，说明统计约束确实有效收紧了攻击空间。

同时，`FGSM + physical_stat` 的结果显示：

- `fgsm_loss = 0.08354`
- 但 `objective_fgsm = -4.20804`
- `strict_attack_success_fgsm = False`

这说明单步攻击虽然能把 `MSE` 拉高，但它在一步更新内无法兼顾统计惩罚项，导致整体优化目标反而下降。相较之下，PGD 由于有多步迭代与多次投影，能够逐步在“误差放大”和“约束满足”之间找到更平衡的解，因此最终恢复了严格成功。

## 7. 与此前 62 样本结果的对比

此前在 `62` 个样本上的 `PGD + physical_stat` 结果为：

- `clean_loss = 0.01981`
- `pgd_loss = 0.08003`
- `PGD / clean = 4.04x`
- `objective_pgd = 0.02754`
- `strict_attack_success_pgd = True`

本轮扩样到 `497` 个样本后，对应结果变为：

- `clean_loss = 0.01187`
- `pgd_loss = 0.08472`
- `PGD / clean = 7.14x`
- `objective_pgd = 0.02760`
- `strict_attack_success_pgd = True`

这组对比有两个重要结论。

第一，扩样后 `objective_pgd` 与此前的 `62` 样本结果几乎一致，分别为 `0.02754` 与 `0.02760`。这说明当前 `physical_stat` 约束下的 PGD 解并没有因为样本规模放大而出现明显退化，整体攻击目标仍然稳定成立。

第二，扩样后 `clean_loss` 更低，但 `pgd_loss` 没有同步下降，因此相对攻击强度 `PGD / clean` 反而从 `4.04x` 提高到了 `7.14x`。这说明在更大样本集上，受约束攻击并没有因为样本异质性增加而失效，反而显示出更明确的误差放大效应。

## 8. 本轮实验结论

本轮扩样实验表明：

1. LSTM 样本级白盒攻击资产可以从原先的 `62` 个样本稳定扩展到 `497` 个有效样本。
2. clean gate 失败的根因并不是模型或梯度链路本身，而是导出阶段混入了 `label = NaN` 的异常匹配行；修复导出逻辑后，`clean_grad_finite_rate` 已恢复到 `1.0`。
3. 在 `497` 个样本上，`physical` 与 `physical_stat` 两种受约束模式都仍然能够使 PGD 达到严格成功。
4. `physical_stat` 模式虽然显著降低了攻击强度，但仍然保留了稳定的误差放大能力，说明其更适合作为后续金融场景回测退化实验的默认样本级攻击方案。

## 9. 后续建议

基于本轮结果，后续可以优先推进两项工作：

1. 以 `physical_stat` 为默认样本级攻击器，继续扩展到组合层部分股票攻击与回测退化验证。
2. 在更大样本集上补充显著性检验、分日期统计和可视化图表，使论文中的证据链从“攻击可行”进一步提升为“攻击退化稳定且统计显著”。

## 10. 进一步扩样到 2048 候选样本的验证

在完成 `497` 样本规模的稳定验证后，本轮进一步将候选参考样本上限提升到 `2048`。新的导出结果为：

- `matched_reference_rows = 2048`
- `exported_sample_rows = 1978`
- `missing_raw_keys = 70`

这说明在更大候选集下，原始 `OHLCV` 窗口的缺失比例仍然维持在较低水平，数据导出本身没有出现新的系统性问题。

### 10.1 扩样过程中新增暴露出的两个边界问题

在直接复用原攻击链路运行 `1978` 个样本时，clean gate 起初没有立即通过。与前一轮 `label = NaN` 导致的导出污染不同，这次暴露的是 Torch 复现层与 Qlib 参考实现之间的两个更细粒度边界差异。

第一类问题出现在 `RSQR5` 与 `CORR5` 这类基于方差或协方差定义的滚动算子上。对部分长期横盘样本，例如：

- `('2025-06-17 00:00:00', 'SH600157')`

其局部 5 日收盘价窗口会出现完全常数序列。例如在 `2025-06-05` 与 `2025-06-06` 两个时点，`SH600157` 的 5 日窗口收盘价分别为：

- `1.35, 1.35, 1.35, 1.35, 1.35`

在这种零方差窗口上，Qlib 的：

- `Rsquare($close, 5)`
- `Corr($close, Log($volume+1), 5)`

都会返回 `NaN`，随后在 `Fillna` 阶段被填成 `0`。而此前的 Torch 实现为了避免除零，直接在分母上加 `eps`，从而把这些退化窗口映射为了有效数值 `0`。经过离线拟合的 `RobustZScoreNorm` 之后，这些 `0` 会进一步变成负的标准化值，例如 `RSQR5` 上会得到 `-1.2786`，从而与 Qlib 参考特征中的 `0.0` 产生明显偏差。

第二类问题出现在反向传播阶段。即使把这些退化窗口显式改写为 `NaN`，如果内部仍先执行 `0/0` 或 `sqrt(0)` 等不安全计算，再在输出端用 `where` 或 `Fillna` 把它们屏蔽掉，反向链路中仍然可能残留 `NaN` 梯度。因此，单纯“把输出改成 `NaN`”还不足以保证 clean gate 中的 `clean_grad_finite_rate = 1.0`。

### 10.2 针对边界问题的修复

本轮对 Torch 复现链路做了两类最小修复。

第一，在 [alpha158_regression.py](/home/chainsmoker/qlib_test/alpha158_regression.py) 中修正了退化窗口语义：

- `rolling_rsquare(...)` 在零方差窗口上返回 `NaN`
- `rolling_corr(...)` 在任一侧零方差窗口上返回 `NaN`

同时实现上不再先执行不安全除法，而是先构造安全分母做有限值计算，再仅在最终输出层将退化窗口标记为 `NaN`。这样既保留了与 Qlib 一致的前向语义，又避免了中间图中出现不可控的 `NaN` 梯度。

第二，在 [legacy_lstm_preprocess.py](/home/chainsmoker/qlib_test/legacy_lstm_preprocess.py) 中调整了 `RobustZScoreNormLayer` 的缺失值处理方式：当输入原始特征为 `NaN` 时，不再让其直接参与标准化，而是先用对应特征的离线拟合中心值替换，再做归一化。这样归一化后的结果恰好为 `0`，与“Qlib 中先得到 `NaN`，再经过 `Fillna` 变为 `0`”的最终输出完全等价，但反向链路保持有限。

为确保修复针对的确实是根因，本轮新增并通过了如下测试：

- [test_alpha158_regression.py](/home/chainsmoker/qlib_test/tests/test_alpha158_regression.py)
- [test_legacy_lstm_preprocess.py](/home/chainsmoker/qlib_test/tests/test_legacy_lstm_preprocess.py)

这些测试覆盖了：

- 常数窗口下 `rolling_rsquare` 返回 `NaN`
- 零方差窗口下 `rolling_corr` 返回 `NaN`
- 上述 `NaN` 在被下游掩蔽使用时，梯度仍保持有限
- `RobustZScoreNormLayer` 在 `NaN` 输入上输出与 Qlib 最终行为一致的 `0`

### 10.3 大样本 clean gate 的批量尺度效应

完成上述修复后，`1978` 样本规模上的 clean gate 已恢复为：

- `clean_grad_finite_rate = 1.0`
- `feature_max_abs_to_reference = 0.61249`
- `spearman_to_reference = 0.99577`

此时唯一仍与默认门槛发生冲突的指标是：

- `clean_grad_mean_abs = 6.94e-07`

它略低于脚本中的默认阈值 `1e-6`。进一步按不同 batch 规模检查后可以看到，该值会随样本数量扩大而近似按 `1/B` 下降：

| 样本数 | clean_grad_mean_abs |
| ---: | ---: |
| 128 | `1.04e-05` |
| 256 | `5.63e-06` |
| 497 | `3.26e-06` |
| 512 | `3.13e-06` |
| 1024 | `1.52e-06` |
| 1978 | `6.94e-07` |

因此，这里的下降并不意味着梯度链路断裂，而是因为当前攻击目标使用 batch-mean 形式的 `MSE`，随着 batch 变大，每个输入元素分到的平均梯度自然会被稀释。基于这一点，本轮没有修改默认 clean gate 逻辑，而是在 `1978` 样本这次实验中临时将 `min_clean_grad_mean_abs` 调整为 `5e-7`，仅用于大批量扩样验证。

### 10.4 `1978` 样本上的最终攻击结果

在上述边界修复与阈值说明后，`PGD + physical_stat` 在 `1978` 个样本上的结果为：

- `clean_loss = 0.011838`
- `fgsm_loss = 0.079009`
- `pgd_loss = 0.082323`
- `PGD / clean = 6.95x`
- `objective_pgd = 0.026521`
- `strict_attack_success_pgd = True`

同时，统计约束相关指标为：

- `ret_penalty_pgd = 0.05509`
- `candle_penalty_pgd = 0.02933`
- `vol_penalty_pgd = 0.0`
- `mean_abs_ret_shift_pgd = 0.00371`
- `mean_abs_dlogvol_shift_pgd = 0.01921`

### 10.5 与 497 样本结果的对比

与前一轮 `497` 样本的 `physical_stat` 结果相比：

| 指标 | 497 样本 | 1978 样本 |
| --- | ---: | ---: |
| `clean_loss` | 0.01187 | 0.01184 |
| `pgd_loss` | 0.08472 | 0.08232 |
| `PGD / clean` | 7.14x | 6.95x |
| `objective_pgd` | 0.02760 | 0.02652 |
| `strict_attack_success_pgd` | True | True |
| `ret_penalty_pgd` | 0.05651 | 0.05509 |
| `candle_penalty_pgd` | 0.02978 | 0.02933 |

可以看到，扩样到 `1978` 个样本后，攻击强度与统计惩罚几乎没有发生结构性变化：

1. `objective_pgd` 仅从 `0.02760` 轻微下降到 `0.02652`，仍显著高于 `objective_clean`。
2. `PGD / clean` 从 `7.14x` 变为 `6.95x`，说明误差放大量基本稳定。
3. `strict_attack_success_pgd` 在更大样本集上依旧保持为 `True`。

这意味着，当前 `physical_stat` 受约束攻击并不是仅在小样本上偶然成立，而是在接近 `2000` 个样本规模下仍保持稳定有效。

## 11. 继续扩样到 4096 候选样本的验证

在 `2048 -> 1978` 这一档验证通过后，本轮继续把候选参考样本上限提升到 `4096`。新的导出结果为：

- `matched_reference_rows = 4096`
- `exported_sample_rows = 3967`
- `missing_raw_keys = 129`

从缺失比例看，这一档的原始窗口可恢复率依然稳定，说明扩样并没有引入新的系统性导出失败。

### 11.1 clean gate 预检查

在 `3967` 个样本上，clean gate 的主要指标为：

- `clean_loss = 0.011583`
- `clean_grad_mean_abs = 3.38e-07`
- `clean_grad_max_abs = 4.46e-04`
- `clean_grad_finite_rate = 1.0`
- `feature_finite_rate = 1.0`
- `spearman_to_reference = 0.99589`
- `feature_mae_to_reference = 0.03511`
- `feature_rmse_to_reference = 0.09941`
- `feature_max_abs_to_reference = 0.61249`

可以看到，经过前一轮对 `Rsquare/Corr` 边界语义与 `RobustZScoreNormLayer` 缺失值处理方式的修复后，`3967` 样本规模下的特征对齐质量依然稳定，且梯度有限率保持为 `1.0`。此时唯一仍随 batch 扩大而下降的指标仍然是 `clean_grad_mean_abs`。因此本轮延续与 `1978` 档相同的处理策略，不改默认脚本阈值，只在本次实验命令中临时将：

- `min_clean_grad_mean_abs = 2.5e-7`

用于大批量验证。

### 11.2 `3967` 样本上的最终攻击结果

在 `PGD + physical_stat` 设置下，本轮结果为：

- `clean_loss = 0.011583`
- `fgsm_loss = 0.077909`
- `pgd_loss = 0.080698`
- `PGD / clean = 6.97x`
- `objective_pgd = 0.024214`
- `strict_attack_success_pgd = True`

统计约束相关指标为：

- `ret_penalty_pgd = 0.05587`
- `candle_penalty_pgd = 0.02946`
- `vol_penalty_pgd = 0.0`
- `mean_abs_ret_shift_pgd = 0.00371`
- `mean_abs_dlogvol_shift_pgd = 0.01920`

### 11.3 与前两档扩样结果的对比

将 `497`、`1978` 与 `3967` 三档 `physical_stat` 结果放在一起比较，可得到：

| 有效样本数 | clean_loss | pgd_loss | PGD / clean | objective_pgd | strict success |
| ---: | ---: | ---: | ---: | ---: | --- |
| 497 | 0.01187 | 0.08472 | 7.14x | 0.02760 | True |
| 1978 | 0.01184 | 0.08232 | 6.95x | 0.02652 | True |
| 3967 | 0.01158 | 0.08070 | 6.97x | 0.02421 | True |

从这个对比可以看到：

1. 三档样本规模上的 `PGD / clean` 始终稳定在 `7x` 左右，没有随着样本规模翻倍而突然塌陷。
2. `objective_pgd` 虽然随样本增多出现了轻微下降，但下降幅度较小，且始终显著高于 `objective_clean`。
3. `strict_attack_success_pgd` 在 `497 -> 1978 -> 3967` 三档下全部保持为 `True`。

这说明当前 `PGD + physical_stat` 受约束攻击已经不再是“小样本可行”的结论，而是在接近 `4000` 个样本规模下仍表现出稳定的误差放大能力与约束满足能力。

### 11.4 当前阶段结论

扩样到 `4096` 候选、`3967` 有效样本后，可以得到更强的阶段性结论：

1. LSTM 样本级 `physical_stat` 白盒攻击已经在接近 `4000` 个有效样本上稳定成立。
2. 经过对退化窗口算子和缺失值标准化边界的修复后，clean gate 在大样本下依然能够保持特征对齐与梯度有限。
3. 当前唯一需要随 batch 规模调整的是 `clean_grad_mean_abs` 门槛，而不是攻击链路本身的正确性。
4. 因此，现阶段完全可以将 `physical_stat` 视为后续组合层部分股票攻击与回测退化实验的默认样本级攻击器。
