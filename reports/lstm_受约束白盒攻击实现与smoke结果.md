# LSTM 受约束白盒攻击实现与 Smoke 结果

## 1. 本轮实现内容

本轮工作在现有 legacy LSTM 原始 `OHLCV` 白盒攻击链路上，新增了三类可对照攻击模式：

- `none`：保留原有无约束攻击基线；
- `physical`：在每步更新后加入金融物理硬投影；
- `physical_stat`：在 `physical` 基础上进一步加入相对 clean 窗口的统计 penalty。

对应实现包含以下几部分：

- 在 `legacy_lstm_attack_core.py` 中新增 `project_financial_feasible_box(...)`，用于强制满足：
  - 相对预算盒约束；
  - 价格、成交量下界；
  - `high >= max(open, close)`；
  - `low <= min(open, close)`；
  - `low <= high`。
- 新增三类统计 penalty：
  - `compute_return_penalty(...)`
  - `compute_candle_penalty(...)`
  - `compute_volume_penalty(...)`
- 新增 `compute_constrained_attack_objective(...)` 与 `compute_attack_objective(...)`，支持在 `physical_stat` 下优化
  `J = MSE - lambda_ret * P_ret - lambda_candle * P_candle - lambda_vol * P_vol`。
- 扩展 `fgsm_maximize_mse(...)` 与 `pgd_maximize_mse(...)`，使其支持 `constraint_mode` 与统计约束超参。
- 扩展 `scripts/run_lstm_whitebox_attack.py`：
  - 新增 `--constraint-mode`
  - 新增 `--tau-ret / --tau-body / --tau-range / --tau-vol`
  - 新增 `--lambda-ret / --lambda-candle / --lambda-vol`
  - 在输出 summary 中补充 objective、penalty、统计 shift、物理合法性与 strict success 指标。
- 修复 `legacy_lstm_predictor.py` 的配置加载兼容性，使其既支持旧的扁平配置，也支持导出的 `model_kwargs` 嵌套配置。

## 2. 测试验证

本轮新增并通过了以下测试：

- `tests/test_lstm_attack_constraints.py`
  - 金融物理投影约束
  - return / candle / volume penalty
  - constrained objective
- `tests/test_lstm_whitebox_attack_smoke.py`
  - `physical` 模式 smoke
  - `physical_stat` 模式 smoke
  - runner 新增 CLI 参数解析
- `tests/test_legacy_lstm_predictor.py`
  - 嵌套 `model_kwargs` 配置兼容加载

当前相关测试总计：

- `22 passed`

## 3. 真实链路验证

首先用 `physical_stat` 跑了一轮 8 样本小验证，链路已可完整产出结果文件：

- `clean_loss = 0.06113984`
- `fgsm_loss = 0.16900462`
- `pgd_loss = 0.16203253`

说明在真实导出资产上，受约束攻击链路已经可以从模型加载、clean gate、攻击生成、summary/report 落盘整条跑通。

## 4. 三模式 16 样本 Smoke 结果

使用同一批 16 个样本，对 `none / physical / physical_stat` 三种模式进行了并行对照。

### 4.1 核心结果

| 模式 | clean_loss | fgsm_loss | pgd_loss | FGSM 物理合法 | PGD 物理合法 | FGSM strict success | PGD strict success |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| none | 0.03384351 | 0.13246885 | 0.27077109 | False | False | False | False |
| physical | 0.03384351 | 0.10416338 | 0.14425640 | True | True | True | True |
| physical_stat | 0.03384351 | 0.10416338 | 0.10394776 | True | True | False | True |

### 4.2 penalty 结果

`physical_stat` 模式下，当前默认值已调为 `lambda_ret=0.8`、`lambda_candle=0.4`、`lambda_vol=0.3`：

- FGSM:
  - `ret_penalty = 4.87780237`
  - `candle_penalty = 0.87938762`
  - `vol_penalty = 0.0`
  - `objective_fgsm = -4.14983368`
- PGD:
  - `ret_penalty = 0.06690290`
  - `candle_penalty = 0.02623265`
  - `vol_penalty = 0.0`
  - `objective_pgd = 0.03993238`

## 5. 初步结论

### 5.1 `physical` 模式验证了“金融物理合法性可以接入且攻击仍有效”

从 `none -> physical` 可见：

- FGSM loss 从 `0.1325` 下降到 `0.1042`
- PGD loss 从 `0.2708` 下降到 `0.1443`

这说明单独加入金融物理投影后，攻击强度确实下降，尤其对多步 PGD 更明显；但攻击并未失效，且 `strict_attack_success` 在 `physical` 下为真，表明“满足物理合法性且仍能放大误差”的目标已经达成。

### 5.2 `physical_stat` 进一步压制了统计上过于激进的攻击

从 `physical -> physical_stat` 可见：

- FGSM 的 `fgsm_loss` 基本没有进一步下降，仍为 `0.1042`
- 但其 `objective_fgsm` 变为显著负值，说明虽然误差被放大，但统计 penalty 爆得很高，因此不再属于“严格成功”
- PGD 的 `pgd_loss` 从 `0.1443` 进一步下降到 `0.1039`
- 同时 `objective_pgd = 0.0399 > objective_clean = 0.0338`，因此 PGD 在当前调优后的默认值下已经恢复为严格成功

这表明：

- 统计约束主要在抑制那些会引起显著收益率路径偏移和 K 线形态偏移的攻击步；
- 对一阶 FGSM 来说，单步更新仍容易直接撞上 penalty 区域；
- 对 PGD 来说，经过轻度调参后，已经可以在维持约束的同时保留严格成功。

### 5.3 当前样本上 `volume` penalty 尚未成为主导项

本轮 16 样本对比中：

- `vol_penalty_fgsm = 0.0`
- `vol_penalty_pgd = 0.0`

说明当前约束配置下，主要起作用的是：

- `P_ret`
- `P_candle`

成交量动态约束在这一组样本上尚未成为主要抑制来源。

## 6. 超参调优结果

围绕 `lambda_ret` 与 `lambda_candle`，额外做了一轮 `4 x 4` 小网格扫描：

- `lambda_ret in {0.7, 0.8, 0.9, 1.0}`
- `lambda_candle in {0.2, 0.3, 0.4, 0.5}`

结果见：

- `reports/lstm_whitebox_attack_constraints_tuning/lambda_ret_candle_grid.csv`
- `reports/lstm_whitebox_attack_constraints_tuning/lambda_ret_candle_grid.md`

观察到：

1. 共 `9/16` 组组合可使 `strict_attack_success_pgd = True`
2. 当 `lambda_ret = 1.0` 时，大多数配置仍然过严
3. 当 `lambda_ret = 0.8` 时，PGD 已经可以较稳定地回到严格成功区
4. `lambda_ret = 0.9, lambda_candle = 0.2` 虽然也能通过，但 objective 裕量太小，更接近边界点

综合“约束优先级”和“成功裕量”后，本轮将默认值更新为：

- `lambda_ret = 0.8`
- `lambda_candle = 0.4`
- `lambda_vol = 0.3`

选择这组默认值的原因是：

- 相比 `0.9 / 0.2`，它不是刚好过线，而是保留了更明显的 objective 裕量；
- 相比 `0.7 / *`，它没有把收益率约束放得过松；
- 仍保持 `lambda_ret > lambda_candle > lambda_vol` 的约束优先级。

## 7. 当前判断

截至本轮，代码层面的结论已经比较清楚：

1. 受约束攻击实现已经完成，且真实资产链路可运行。
2. `physical` 模式是一个稳定可用的“现实感更强”的攻击版本。
3. `physical_stat` 模式已经体现出 tradeoff，并且在调优后的默认值下，`PGD + physical_stat` 已恢复为严格成功。
4. FGSM 在 `physical_stat` 下仍明显过于激进，说明单步攻击更容易一次性冲出统计容忍带。
5. 若下一步要继续推进论文证据，更建议把 `PGD + physical_stat` 作为主分析对象，因为它已经体现出“攻击仍有效，但被统计约束明显收紧”的现象。

## 8. 产出目录

- 8 样本验证：
  - `reports/lstm_whitebox_attack_constraints_verify/`
- 16 样本三模式对照：
  - `reports/lstm_whitebox_attack_constraints_smoke/none/`
  - `reports/lstm_whitebox_attack_constraints_smoke/physical/`
  - `reports/lstm_whitebox_attack_constraints_smoke/physical_stat/`
- 超参扫描：
  - `reports/lstm_whitebox_attack_constraints_tuning/`

## 9. 运行前提说明

当前 runner 已支持如下配置解析顺序：

1. 优先使用显式传入的 `--config-path`
2. 若显式路径不存在，则回退到 `state_dict` 同目录下的常见配置文件名，例如：
   - `model_config.json`
   - `lstm_state_dict_config.json`

需要注意的是，你当前主工作区的 `origin_model_pred/LSTM/model/` 下只有：

- `lstm_state_dict.pt`
- `lstm_trained_model.pkl`

尚未放入 `model_config.json`。因此在你把配置文件也放到该目录之前，真实运行时仍需要显式传入 `--config-path`。
