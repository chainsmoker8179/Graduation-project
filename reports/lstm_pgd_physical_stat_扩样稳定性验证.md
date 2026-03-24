# LSTM `PGD + physical_stat` 扩样稳定性验证

## 1. 实验目的

在前一轮小样本验证中，调优后的默认约束参数

- `lambda_ret = 0.8`
- `lambda_candle = 0.4`
- `lambda_vol = 0.3`

已经能够使 `PGD + physical_stat` 在 `16` 个样本上恢复 `strict_attack_success_pgd = True`。

本轮实验的目标是进一步回答一个更关键的问题：

当攻击样本数从小规模扩展到更大规模时，这一结论是否仍然稳定成立。

## 2. 实验设置

### 2.1 数据与模型

- 攻击资产目录：
  - `/home/chainsmoker/.qlib_test_archive/2026-03-16-worktree-archive/artifacts/lstm_attack_expanded_v6`
- 可用匹配样本总数：
  - `62`
- 模型：
  - legacy raw `OHLCV -> feature bridge -> RobustZScoreNorm -> Fillna -> LSTM`

### 2.2 攻击设置

- 攻击模式：
  - `physical_stat`
- 攻击器：
  - `PGD`
- 使用当前默认约束参数：
  - `tau_ret = 0.005`
  - `tau_body = 0.005`
  - `tau_range = 0.01`
  - `tau_vol = 0.05`
  - `lambda_ret = 0.8`
  - `lambda_candle = 0.4`
  - `lambda_vol = 0.3`

### 2.3 扩样规模

本轮选取三档样本规模做一致口径对照：

- `16`
- `32`
- `62`（全量可用样本）

## 3. 结果

结果汇总文件：

- [stability_summary.md](/home/chainsmoker/qlib_test/reports/lstm_whitebox_attack_constraints_stability/stability_summary.md)
- [stability_summary.csv](/home/chainsmoker/qlib_test/reports/lstm_whitebox_attack_constraints_stability/stability_summary.csv)

核心结果如下：

| 样本数 | clean_loss | pgd_loss | pgd_delta | pgd_ratio | objective_margin | ret_penalty | candle_penalty | 物理合法 | 严格成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 16 | 0.03384351 | 0.10394776 | 0.07010425 | 3.0714 | 0.00608887 | 0.06690290 | 0.02623265 | True | True |
| 32 | 0.02537336 | 0.09884812 | 0.07347476 | 3.8957 | 0.01248855 | 0.06319687 | 0.02607222 | True | True |
| 62 | 0.01981449 | 0.08002781 | 0.06021332 | 4.0389 | 0.00772817 | 0.05266451 | 0.02588159 | True | True |

## 4. 结果分析

### 4.1 扩样后 `PGD + physical_stat` 仍保持严格成功

三档样本规模下均满足：

- `physical_constraints_satisfied_pgd = True`
- `strict_attack_success_pgd = True`

这说明当前调优后的默认超参并不是只对 `16` 个样本偶然有效，而是在扩展到全量 `62` 个样本后仍能保持：

1. 对模型误差的稳定放大；
2. 对金融物理约束的严格满足；
3. 对约束目标 `objective` 的正向增益。

### 4.2 从绝对误差看，`pgd_loss` 随样本量增加略有下降，但相对攻击强度并未减弱

如果只看 `pgd_loss` 的绝对值，会观察到：

- `16 -> 32 -> 62` 时，`pgd_loss` 从 `0.1039` 下降到 `0.0800`

但与此同时：

- `clean_loss` 也在下降，从 `0.0338` 下降到 `0.0198`

因此更合理的比较方式是看相对放大量 `pgd_ratio = pgd_loss / clean_loss`：

- `16` 样本：`3.0714`
- `32` 样本：`3.8957`
- `62` 样本：`4.0389`

这表明随着样本规模增大，攻击在相对意义上的误差放大能力并没有削弱，反而更稳定地维持在 `3x ~ 4x` 的区间。

### 4.3 objective 裕量保持为正，且没有在扩样时塌陷

衡量受约束攻击是否真正成立，关键不仅是 `pgd_loss > clean_loss`，还要看：

\[
objective\_margin = objective\_{pgd} - objective\_{clean}
\]

本轮结果为：

- `16` 样本：`0.00609`
- `32` 样本：`0.01249`
- `62` 样本：`0.00773`

三档都为正，且数量级相近，没有出现扩样后 margin 直接跌回负值的情况。

这说明当前默认超参下，`PGD + physical_stat` 在更大样本规模上仍然保持“受约束意义下的有效攻击”，而不是仅仅通过牺牲统计真实性来换取误差增大。

### 4.4 统计 penalty 较稳定，未出现扩样后的异常放大

`PGD` 的两项主要 penalty 为：

- `ret_penalty_pgd`
- `candle_penalty_pgd`

扩样结果分别为：

- `ret_penalty_pgd`：`0.0669 -> 0.0632 -> 0.0527`
- `candle_penalty_pgd`：`0.0262 -> 0.0261 -> 0.0259`

可以看到：

- 收益率 penalty 没有上升，反而略有下降；
- K 线形态 penalty 基本稳定在 `0.026` 附近；
- `vol_penalty_pgd` 仍为 `0.0`

因此当前扩样并未引入新的统计失真风险，说明这组默认超参在该数据集上的 penalty 行为是可控的。

### 4.5 攻击预算使用率也保持稳定

`PGD` 的平均预算使用率在三档样本规模下变化很小：

- `price_ratio_mean`：`0.2426 -> 0.2453 -> 0.2442`
- `volume_ratio_mean`：`0.8795 -> 0.8808 -> 0.8687`

这意味着扩样后攻击器没有表现出新的异常行为，例如：

- 被迫把价格通道推得更满；
- 或者在成交量通道上出现额外的预算堆积。

## 5. 结论

本轮扩样实验表明：

1. 当前默认参数下，`PGD + physical_stat` 在 `16 / 32 / 62` 三档样本规模上均保持 `strict_attack_success_pgd = True`。
2. 扩样后物理合法性始终满足，说明硬投影机制稳定。
3. `objective_margin` 在三档规模上均为正，说明受约束攻击目标没有因样本量增加而失效。
4. `ret_penalty` 与 `candle_penalty` 保持稳定，没有出现扩样后的 penalty 爆炸。
5. 因此可以认为：对当前数据集而言，调优后的 `PGD + physical_stat` 已经具备初步的扩样稳定性。

## 6. 当前判断与下一步

基于这轮结果，后续更值得继续推进的是：

1. 把 `PGD + physical_stat` 作为论文中的主攻击设定；
2. 在更高层面继续验证其组合收益与回测退化是否也保持稳定；
3. 若要进一步提升说服力，可以补多随机种子或多时间段切片的重复实验，而不必优先继续下调约束强度。
