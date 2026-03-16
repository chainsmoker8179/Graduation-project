# LSTM 原始 OHLCV 白盒攻击扩大样本实验报告

## 1. 目的

本轮实验在前一版 smoke attack 的基础上推进三项工作：第一，收紧 clean gate，不再仅检查梯度是否存在，而是加入对旧版参考输出和中间特征序列的定量对齐约束；第二，将导出样本数从极小规模扩展到更大的测试子集，以获得更稳定的 FGSM/PGD 统计；第三，将新的导出逻辑、clean gate 指标和攻击结果整理为可复用的中文实验记录。

## 2. 本轮实现改动

### 2.1 clean gate 收紧

攻击脚本新增了带阈值的 clean gate。除了原有的 `feature_finite_rate`、`clean_grad_finite_rate` 和 `clean_grad_mean_abs` 之外，本轮新增了三类中间特征对齐指标：

- `feature_mae_to_reference`：Torch 在线重建特征与 Qlib 参考特征窗口之间的平均绝对误差
- `feature_rmse_to_reference`：Torch 在线重建特征与 Qlib 参考特征窗口之间的均方根误差
- `feature_max_abs_to_reference`：Torch 在线重建特征与 Qlib 参考特征窗口之间的最大绝对误差

同时，clean gate 继续保留最终预测与旧版 `pred.pkl` 之间的 Spearman 排序相关检查。考虑到当前阶段的目标是优先验证攻击图是否在“旧版模型真实输入空间”上语义接近，因此本轮 gate 采用“特征对齐优先、分数排序辅助”的约束方式。最终使用的阈值为：

- `min_clean_grad_mean_abs = 1e-6`
- `min_spearman_to_reference = 0.09`
- `max_feature_mae_to_reference = 0.05`
- `max_feature_rmse_to_reference = 0.12`
- `max_feature_max_abs_to_reference = 0.7`

### 2.2 离线导出逻辑增强

离线资产导出脚本新增了 `matched_feature_windows.pt`，用于保存 Qlib 侧经 `RobustZScoreNorm + Fillna` 处理后的旧版 20 特征参考窗口，从而支持在线攻击脚本在 clean gate 中直接比较“模型真实输入”而不是只比较最终分数。

在扩大样本时，导出流程还暴露出两个真实问题。其一，`pred.pkl` 与 `label.pkl` 的交集样本原本会跨越 `2025-11` 和 `2025-12`，但本轮攻击设置只导出到 `2025-10-31`，因此部分样本键在原始 `OHLCV` 测试 split 中不存在。其二，即使样本键处于测试区间内，部分股票对应的 `80` 日原始窗口前段仍然包含全列 NaN，原因是有效历史不足。针对这两个问题，导出脚本现在会先按 `test_start_time` 和 `test_end_time` 过滤 matched reference，再在导出原始窗口和参考特征窗口时跳过非有限样本，并将其计入 `missing_*_keys` 统计，而不是让整个导出流程失败。

## 3. 导出结果

本轮扩大样本实验使用如下命令运行导出：

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_lstm_attack_assets.py \
  --out-dir artifacts/lstm_attack_expanded_v2 \
  --max-samples 64 \
  --test-end-time 2025-10-31
```

导出汇总结果如下：

- `matched_reference_rows = 64`
- `exported_sample_rows = 62`
- `raw_window_len = 80`
- `raw_feature_dim = 5`
- `missing_raw_keys = 2`
- `exported_feature_rows = 64`
- `feature_window_len = 20`
- `feature_dim = 20`
- `missing_feature_keys = 0`

这表明，在 64 个匹配键中，有 62 个样本可以成功重建为有限的原始 `OHLCV` 窗口，并同时与参考特征窗口对齐；另有 2 个样本因为原始 `OHLCV` 窗口中存在非有限值而被过滤。过滤后的 62 个样本构成了本轮 expanded smoke attack 的最终评估子集。

## 4. 收紧后的 clean gate 结果

在 62 个有效样本上，新的 clean gate 指标如下：

- `clean_loss = 0.01622363`
- `clean_grad_mean_abs = 1.1748e-05`
- `clean_grad_max_abs = 0.00601645`
- `clean_grad_finite_rate = 1.0`
- `feature_finite_rate = 1.0`
- `spearman_to_reference = 0.10035`
- `feature_mae_to_reference = 0.03452`
- `feature_rmse_to_reference = 0.09824`
- `feature_max_abs_to_reference = 0.60877`

将这些数值与 clean gate 阈值对照可以看到，五项约束全部通过。特别是，虽然最终预测分数与旧版 `pred.pkl` 的 Spearman 仍然只有 `0.10035`，但其已经超过本轮设定的 `0.09` 下界；与此同时，三项中间特征对齐误差均处于较低水平，其中特征 MAE 为 `0.03452`，特征 RMSE 为 `0.09824`，最大绝对误差为 `0.60877`。这说明当前 Torch 在线重建链在“模型实际输入张量”层面已经与 Qlib 参考特征保持较接近的一致性，clean gate 的通过不再仅仅依赖“梯度存在”这一弱条件。

## 5. FGSM/PGD 扩大样本结果

本轮攻击使用如下命令运行：

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_lstm_whitebox_attack.py \
  --asset-dir artifacts/lstm_attack_expanded_v2 \
  --out-dir reports/lstm_whitebox_attack_expanded_v2 \
  --max-samples 62
```

主结果如下：

- `clean_loss = 0.01622363`
- `fgsm_loss = 0.07544520`
- `pgd_loss = 0.10530549`
- `fgsm_mean_abs_pred_shift = 0.15217`
- `pgd_mean_abs_pred_shift = 0.21399`

相对于 clean 推理，FGSM 将均方误差提升到原来的约 `4.65` 倍，PGD 将均方误差提升到原来的约 `6.49` 倍。就预测偏移而言，FGSM 的平均绝对预测偏移为 `0.15217`，中位数为 `0.14756`，范围为 `[0.03912, 0.47216]`；PGD 的平均绝对预测偏移为 `0.21399`，中位数为 `0.20331`，范围为 `[0.07964, 0.51332]`。按样本级 `MSE_adv > MSE_clean` 的判据计算，FGSM 和 PGD 在 62 个样本上的成功率均为 `1.0`。因此，在扩大后的样本子集上，当前白盒攻击链路并未出现只对个别样本有效的退化现象，攻击效应在样本层面仍保持稳定。

从预算使用情况看，FGSM 和 PGD 依旧表现出对成交量通道更高的边界利用率：

- FGSM：`price_ratio_mean = 0.42823`，`volume_ratio_mean = 1.0`
- PGD：`price_ratio_mean = 0.32752`，`volume_ratio_mean = 0.81628`

这与前一版小样本实验中的现象保持一致，即在相对幅度约束下，攻击器更容易将成交量通道推向预算边界，而价格通道上的扰动则更倾向于集中在部分时间步与部分股票上。

## 6. 结论与当前判断

本轮实验说明，原始 `OHLCV` 白盒攻击流程已经从“功能性 smoke”推进到“带中间特征对齐约束的扩大样本 smoke”。一方面，新增的 Qlib 参考特征窗口导出使 clean gate 可以直接检查旧版模型真实输入的一致性，从而显著提高了 clean 验证的约束强度。另一方面，在 62 个有效样本上，FGSM 与 PGD 仍然能够稳定放大预测误差，并保持 1.0 的样本级成功率，说明当前攻击通路不仅可微，而且已具备可重复的优化效应。

不过，这一结果仍应被理解为“扩大样本下的稳定 smoke 验证”，而不是最终的完整攻击评估。当前仍有两个边界尚未解决。第一，最终分数与旧版 `pred.pkl` 的排序一致性仍然偏弱，`spearman_to_reference` 只有 `0.10035`，表明最终输出层面仍存在残余语义偏差。第二，本轮结果依然停留在模型预测误差层面，尚未接入组合收益、换手、风险暴露和回测指标。因此，下一步更合适的推进方向是：继续压缩 clean 语义偏差，并在此基础上将当前攻击扩展到组合级评价。

## 7. 结果文件

- 导出汇总：[artifacts/lstm_attack_expanded_v2/export_summary.json](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/artifacts/lstm_attack_expanded_v2/export_summary.json)
- 攻击汇总：[reports/lstm_whitebox_attack_expanded_v2/attack_summary.json](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/lstm_whitebox_attack_expanded_v2/attack_summary.json)
- 样本级结果：[reports/lstm_whitebox_attack_expanded_v2/sample_metrics.csv](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/lstm_whitebox_attack_expanded_v2/sample_metrics.csv)

## 8. clean 偏弱根因修正与正式重跑结果

在上述扩大样本实验完成后，又继续对 clean 对齐偏弱的根因进行了逐层排查。排查结论表明，主导问题并不在 `pred.pkl`、LSTM 权重或原始 `OHLCV -> 特征公式` 本身，而在于“训练时模型真实消费的 20 维特征顺序”与当前攻击链路默认使用的特征顺序并不一致。旧 notebook 中虽然将 20 个特征名写成了一个 `feature_cols` 列表，但 Qlib 的 `FilterCol` 处理器只负责筛列，并不会按照 `col_list` 重新排列列顺序；它保留的是 Alpha158 原始列顺序。此前攻击链路把手写列表顺序直接当作模型输入顺序，导致在线 bridge、离线归一化统计量和参考特征窗口在语义上一致、但在维度排列上错位，从而把 clean 对齐压低到了 `Spearman = 0.10035` 的水平。

围绕这一根因，本轮修正做了两项关键调整。第一，将 `legacy_lstm_feature_bridge.py` 中的 20 维特征输出顺序改为 Alpha158 经过 `FilterCol` 后的真实保留顺序，即 `KLEN, KLOW, ROC60, STD5, RSQR5, RSQR10, RSQR20, RSQR60, RESI5, RESI10, CORR5, CORR10, CORR20, CORR60, CORD5, CORD10, CORD60, VSTD5, WVMA5, WVMA60`。第二，将离线 reference feature 导出逻辑改为直接走 `Alpha158 + FilterCol + RobustZScoreNorm + Fillna` 的旧 notebook 等价路径，并显式使用 `dataset.prepare("test", col_set=["feature", "label"], data_key=DK_I)` 与 `fillna_type="ffill+bfill"`，从而让 reference window 的列顺序、统计量顺序和在线攻击图中的顺序重新对齐。相关单元测试已经补充并通过。

在完成顺序修正后，首先使用一版快速验证资产确认修正方向是否正确。结果显示，只要同步修正特征顺序和归一化统计量顺序，clean 预测与 `pred.pkl` 的排序一致性就会立刻显著提升。随后，又基于当前修正后的代码正式重导了一版资产并重新运行攻击。正式导出命令如下：

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/export_lstm_attack_assets.py \
  --out-dir artifacts/lstm_attack_expanded_v6 \
  --max-samples 64 \
  --test-end-time 2025-10-31
```

正式攻击命令如下：

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python scripts/run_lstm_whitebox_attack.py \
  --asset-dir artifacts/lstm_attack_expanded_v6 \
  --out-dir reports/lstm_whitebox_attack_expanded_v6 \
  --max-samples 62
```

正式 `v6` 导出结果与此前扩大样本设置保持一致：

- `matched_reference_rows = 64`
- `exported_sample_rows = 62`
- `missing_raw_keys = 2`
- `exported_feature_rows = 64`
- `missing_feature_keys = 0`

在这 62 个有效样本上，修正后的 clean gate 指标如下：

- `clean_loss = 0.01431117`
- `clean_grad_mean_abs = 1.4190e-05`
- `clean_grad_max_abs = 0.00388841`
- `feature_finite_rate = 1.0`
- `clean_grad_finite_rate = 1.0`
- `spearman_to_reference = 0.75130`
- `feature_mae_to_reference = 0.03452`
- `feature_rmse_to_reference = 0.09824`
- `feature_max_abs_to_reference = 0.60877`

与修正前的 `expanded_v3` 结果相比，最关键的变化是 `spearman_to_reference` 从 `0.10035` 提升到了 `0.75130`，增幅达到 `0.65095`。与此同时，`clean_loss` 从 `0.01622363` 降到 `0.01431117`。值得注意的是，三项特征级误差指标几乎没有变化，这说明此前的主要问题确实不是“特征公式整体错误”，而是“同一批特征张量在输入维度上的排列顺序错位”。一旦顺序对齐，模型输出会立即更接近旧版 `pred.pkl`。

修正后的 FGSM/PGD 结果如下：

- `clean_loss = 0.01431117`
- `fgsm_loss = 0.10467355`
- `pgd_loss = 0.28165913`
- `fgsm_mean_abs_pred_shift = 0.19707`
- `pgd_mean_abs_pred_shift = 0.36676`

相对于修正前的 `expanded_v3`，FGSM 的 loss 进一步提高了 `0.02923`，PGD 的 loss 提高了 `0.17635`；FGSM 的平均绝对预测偏移增加了 `0.04490`，PGD 的平均绝对预测偏移增加了 `0.15277`。样本级统计也同步增强：`fgsm_abs_shift > 0.05` 的样本数为 `61/62`，`pgd_abs_shift > 0.05` 的样本数为 `62/62`；FGSM 中位数偏移为 `0.15352`，PGD 中位数偏移为 `0.24372`。这表明，在 clean 对齐得到显著修复后，白盒攻击在同一预算约束下能够更有效地放大预测误差，说明当前攻击图与旧版模型真实输入空间之间的语义一致性明显增强。

综合来看，这轮正式重跑给出了两个更稳固的判断。第一，clean 偏弱的主导根因已经定位并修复，即 LSTM 实际输入的 20 维特征顺序必须遵循 Alpha158 经 `FilterCol` 筛列后的保留顺序，而不能直接使用 notebook 中手写的 `feature_cols` 顺序。第二，在顺序修正后，当前原始 `OHLCV -> 可微特征 -> RobustZScoreNorm -> LSTM` 攻击链路已经具备更高的 clean 对齐质量和更强的 FGSM/PGD 攻击响应。因此，后续如果要继续推进到组合收益或回测层面的攻击评估，应当以 `expanded_v6` 这一版结果作为新的基线，而不应再使用修正前的 `expanded_v2/v3` 结果作为最终 clean 参考。

## 9. 修正后结果文件

- 正式导出汇总：[artifacts/lstm_attack_expanded_v6/export_summary.json](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/artifacts/lstm_attack_expanded_v6/export_summary.json)
- 正式攻击汇总：[reports/lstm_whitebox_attack_expanded_v6/attack_summary.json](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/lstm_whitebox_attack_expanded_v6/attack_summary.json)
- 正式样本级结果：[reports/lstm_whitebox_attack_expanded_v6/sample_metrics.csv](/home/chainsmoker/qlib_test/.worktrees/raw-ohlcv-lstm-attack/reports/lstm_whitebox_attack_expanded_v6/sample_metrics.csv)
