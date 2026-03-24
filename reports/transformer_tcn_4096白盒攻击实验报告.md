# Transformer / TCN 4096 样本白盒攻击实验报告

## 1. 实验目的

本实验的目标是在 `Transformer` 与 `TCN` 两个旧基线模型上，将此前已经跑通的样本级 white-box attack 从小样本验证提升到大样本正式实验口径。实验仍然采用原始 `OHLCV` 输入空间上的 `FGSM` 与 `PGD`，攻击目标保持为最大化 `MSE`，不引入新的物理约束或统计约束。

## 2. 实验口径

- 时间区间：`2025-01-01` 至 `2025-10-31`
- 参考预测来源：各模型完整 `pred.pkl` / `label.pkl`
- 初始采样上限：每个模型 `4096` 条 matched reference
- 预算设置：价格列 `1%`，成交量列 `2%`
- `PGD` 参数：`steps=5`，`step_size=0.25`
- 设备：`cpu`

之所以统一截断到 `2025-10-31`，是因为 `TCN` 的原始预测文件只覆盖到该日期；若保留 `Transformer` 到 `2025-12-31` 的额外样本，则横向比较口径不一致。

## 3. 资产导出过程与问题定位

最初尝试直接复用 `scripts/export_lstm_attack_assets.py` 导出 `4096` 样本攻击资产，但排查后发现，脚本中的 `Alpha158` 路径在当前环境下极慢且不稳定。进一步拆解发现：

- `raw test split` 构造本身是可用的，约 `10` 秒量级即可完成；
- 真正的瓶颈出现在 `Alpha158 -> RobustZScoreNorm` 统计量拟合以及 Qlib 特征窗口导出链路；
- 该链路并非攻击定义本身所必需，因为归一化统计量已经在历史资产中稳定存在，且对 `LSTM`、`Transformer`、`TCN` 三模型是一致的。

因此本次实验最终固化为正式大样本导出脚本 `scripts/export_large_sample_attack_assets.py`，其核心仍然采用快导出路径：

1. 直接从完整 `pred.pkl` / `label.pkl` 构造 `4096` 条 `matched_reference`；
2. 复用现有已验证的 `normalization_stats.json`；
3. 通过 `raw test split` 导出匹配的 `matched_ohlcv_windows.pt`；
4. 使用 `LegacyLSTMFeatureBridge + RobustZScoreNorm + Fillna` 从 raw window 直接重建 `matched_feature_windows.pt`。

这一方案不改变攻击图，只是绕开了最慢的 Qlib 特征导出环节。

## 4. 资产导出结果

两模型的快导出结果一致：

| 模型 | matched reference | raw window 导出成功数 | missing raw keys | feature window 导出数 |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 4096 | 3985 | 111 | 3985 |
| TCN | 4096 | 3985 | 111 | 3985 |

本轮正式复跑对应的导出摘要文件为：

- `artifacts/transformer_attack_4096_v1/export_summary.json`
- `artifacts/tcn_attack_4096_v1/export_summary.json`

需要注意的是，进入攻击脚本前后又有极少量样本因 finite 过滤被剔除，因此最终参与攻击的样本数为 `3983`。

## 5. Clean Gate 调整

在 `3985` 量级的大 batch 上，`clean_grad_mean_abs` 明显低于小样本版。这并不是梯度链路断裂，而是因为当前攻击目标采用 `MSE(mean)`，当 batch 规模显著放大时，输入梯度均值会被平均操作自然摊薄。因此原先小样本阶段使用的固定阈值 `1e-6` 会误判失败。

本轮正式实验将 `min_clean_grad_mean_abs` 下调为 `1e-7`，其余阈值保持不变。调整后两模型 clean gate 均通过，且：

- `clean_grad_finite_rate = 1.0`
- `feature_finite_rate = 1.0`
- `spearman_to_reference` 保持在很高水平

因此这次阈值调整属于 batch 规模变化下的口径修正，而不是数值稳定性退化。

另外，正式版目录 `*_4096_v1` 的攻击摘要与此前临时 `_fast` 目录结果逐项一致，说明脚本固化并未引入新的数值偏差。

## 6. 大样本攻击结果

### 6.1 Transformer

- 最终样本数：`3983`
- `clean_loss = 0.00698`
- `fgsm_loss = 0.03768`
- `pgd_loss = 0.04981`
- `FGSM / clean = 5.40x`
- `PGD / clean = 7.14x`
- `clean_grad_mean_abs = 7.24e-07`
- `clean_grad_finite_rate = 1.0`
- `spearman_to_reference = 0.9982`
- `fgsm_abs_shift_mean = 0.1186`
- `pgd_abs_shift_mean = 0.1483`

结论：`Transformer` 在大样本下依然保持稳定可攻击。`FGSM` 和 `PGD` 都能显著抬高 `MSE`，且 `PGD` 仍强于 `FGSM`。

### 6.2 TCN

- 最终样本数：`3983`
- `clean_loss = 0.02479`
- `fgsm_loss = 0.06502`
- `pgd_loss = 0.14440`
- `FGSM / clean = 2.62x`
- `PGD / clean = 5.82x`
- `clean_grad_mean_abs = 7.01e-07`
- `clean_grad_finite_rate = 1.0`
- `spearman_to_reference = 0.9966`
- `fgsm_abs_shift_mean = 0.1108`
- `pgd_abs_shift_mean = 0.2318`

结论：`TCN` 在大样本下同样保持稳定可攻击，且 `PGD` 的攻击强度明显高于 `FGSM`。

## 7. 与 124 样本版的稳定性比较

### Transformer

`124` 样本版与 `3983` 样本版对比显示，`Transformer` 的结果非常稳定：

- `FGSM / clean` 从 `5.07x` 上升到 `5.40x`
- `PGD / clean` 从 `6.79x` 上升到 `7.14x`
- 平均预测偏移几乎不变
- `spearman_to_reference` 维持在 `0.998` 附近

这说明 `Transformer` 的白盒攻击结论并不是由小样本偶然性造成的。

### TCN

`TCN` 在大样本下也保持同样结论方向：

- `FGSM / clean` 从 `2.34x` 上升到 `2.62x`
- `PGD / clean` 从 `5.36x` 上升到 `5.82x`
- `PGD` 始终明显强于 `FGSM`
- `spearman_to_reference` 从 `0.9923` 提升到 `0.9966`

这说明 `TCN` 的白盒攻击有效性在扩样后同样成立，且稳定性比 124 样本版更高。

## 8. 当前阶段结论

截至本轮实验，可以得到以下结论：

1. `Transformer` 与 `TCN` 的样本级 white-box attack 已经完成从 smoke 到大样本正式实验的迁移。
2. 在约 `4k` 样本规模上，两模型 clean gate 均通过，输入梯度链路完整且有限。
3. `FGSM` 与 `PGD` 均能稳定放大预测误差，其中 `PGD` 一贯强于 `FGSM`。
4. 与 `124` 样本版相比，扩样后的结果没有出现方向性反转，反而表现出更好的稳定性。

## 9. 当前限制

本次实验仍有两个需要明确说明的边界：

1. 这是一轮固定上限 `4096` 的大样本实验，不是对完整百万级预测集合做全量攻击。
2. 正式脚本中的 `matched_feature_windows.pt` 来自 `torch_bridge_from_raw`，即通过当前桥接层与复用统计量从 raw window 直接重建，而不是重新经过慢速 Qlib `Alpha158` 特征导出路径。

这两个限制不影响“攻击链路成立、攻击效果稳定”这一结论，但在论文写作中应当如实说明。
