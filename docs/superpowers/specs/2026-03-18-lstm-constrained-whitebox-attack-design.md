# LSTM 金融物理约束与统计约束白盒攻击设计

## 背景

当前仓库已经具备一条可运行的 legacy LSTM 原始 `OHLCV` 白盒攻击链路，其核心形式为：

`raw OHLCV -> 20 维旧版特征 -> RobustZScoreNorm -> Fillna -> legacy LSTM -> MSE`

在这一链路上，现有 `FGSM` 与 `PGD` 已经能够在样本级与组合级实验中稳定放大预测误差，并进一步传导到组合回测退化。然而，当前攻击器的约束仍然比较弱，主要只包含：

1. 相对 `L_inf` 扰动预算；
2. 非负域约束。

这意味着当前攻击样本虽然对模型有效，但不一定满足更贴近金融真实数据风控的要求。尤其是以下两类问题仍未被正式纳入攻击定义：

- 金融物理约束：例如 `high >= max(open, close)`、`low <= min(open, close)`、`volume >= 0`；
- 统计约束：例如收益率路径、K 线形态和成交量动态是否相对 clean 窗口出现过大偏移。

用户希望下一阶段暂不扩展到多模型迁移，而是先在 LSTM 上完善一版“更像真实金融数据、能更好通过风控视角审视”的受约束白盒攻击。

## 目标

在不推翻现有 LSTM 攻击主链路的前提下，扩展出一版受约束白盒攻击器，使其同时具备：

1. 金融物理合法性；
2. 相对 clean 窗口的统计稳定性；
3. 与现有无约束攻击可直接对照的实验口径。

具体目标不是完全模拟真实券商或基金公司的全套风控系统，而是在当前样本级攻击框架中加入一组有明确金融含义、可微且可审计的约束机制，使得攻击样本既能继续提高预测误差，又尽量避免生成显著不自然的市场数据。

## 范围内内容

- 在 legacy LSTM 原始 `OHLCV` 白盒攻击链路中新增金融物理约束投影。
- 在攻击目标中新增相对 clean 窗口的统计惩罚项。
- 保留原有 `FGSM` / `PGD` 两类攻击器，但允许其优化目标从单纯 `MSE` 扩展为“攻击收益减去约束代价”。
- 在样本级输出中增加物理合法性与统计偏移的诊断指标。
- 支持对比三种模式：
  - `none`
  - `physical`
  - `physical_stat`

## 范围外内容

- 不在本阶段扩展到 `Transformer`、`TCN` 等多模型。
- 不在本阶段接入横截面统计约束或组合级约束。
- 不在本阶段使用训练集或全市场历史统计量作为主要统计参照。
- 不在本阶段把该攻击器直接接到 partial backtest 或论文图表主线。
- 不在本阶段实现更复杂的价量相关性、行业分布或市场状态条件约束。

## 设计原则

本方案固定采用以下三条原则：

### 1. 旧基线不破坏

当前无约束攻击器必须作为 `none` 模式保留，以保证与现有 `expanded_v6` 及后续组合实验口径可直接对照。

### 2. 物理约束使用硬投影

对于那些“明显不合法”的数据状态，不采用软 penalty，而采用每步更新后的硬投影。这包括：

- 价格与成交量非负；
- `high >= max(open, close)`；
- `low <= min(open, close)`；
- `low <= high`。

### 3. 统计约束使用软惩罚

统计约束不要求攻击样本与 clean 完全一致，而是允许在一定容忍带内偏离，只有超过阈值后才开始惩罚。这更接近实际风控中的告警带思路，而不是“任何偏移都视为异常”。

## 总体方案

最终攻击器分为三种模式：

### 模式 A：`none`

保持当前定义：

- 目标函数：最大化 `MSE`
- 更新后投影：相对预算盒 + 非负域

该模式作为对照组。

### 模式 B：`physical`

保持目标函数仍为最大化 `MSE`，但每步更新后投影到更严格的金融物理可行域：

- 相对预算盒；
- 非负价格和成交量；
- K 线一致性。

该模式用于回答：仅靠物理合法性约束，会损失多少攻击强度。

### 模式 C：`physical_stat`

在 `physical` 的基础上，再把统计惩罚项加入优化目标：

\[
J(x)=\mathrm{MSE}(f(x),y)-\lambda_{ret}P_{ret}-\lambda_{candle}P_{candle}-\lambda_{vol}P_{vol}
\]

该模式用于回答：在满足物理合法性的前提下，如果进一步要求统计特征不要明显偏离 clean，攻击还能保留多少有效性。

## 金融物理约束

### 1. 基础预算约束

保留当前相对预算盒定义：

\[
|\delta_{t,c}| \le \epsilon_c \cdot \max(|x_{t,c}|, floor_c)
\]

其中：

- `open`、`high`、`low`、`close` 共用价格组预算；
- `volume` 使用单独预算。

### 2. 非负约束

每步更新后需满足：

- `open >= price_floor`
- `high >= price_floor`
- `low >= price_floor`
- `close >= price_floor`
- `volume >= volume_floor`

### 3. K 线一致性约束

每个时间步都应满足：

- `high >= max(open, close)`
- `low <= min(open, close)`
- `low <= high`

### 4. 推荐投影顺序

每一步 FGSM 或 PGD 更新后，投影顺序固定为：

1. 先投影回相对预算盒；
2. 再裁剪价格和成交量的下界；
3. 再修正 `high` 和 `low` 以满足 K 线关系。

原因是：预算盒是外层攻击可行域，K 线关系是内部金融合法性约束。先回预算盒，再做 K 线修复，可以避免“修完 K 线又被预算截断破坏”的顺序反复。

## 统计约束

本阶段统计约束全部采用“相对 clean 窗口”的局部约束，不依赖训练集、历史全市场或跨截面的统计量。

### 为什么以 clean 为 reference

这是当前样本级攻击链路最稳妥的第一版选择，原因有三点：

1. clean 窗口与 adversarial 窗口一一对应，定义简单；
2. 不需要再单独导出训练集全局统计量；
3. 更容易解释攻击 tradeoff：攻击到底是“放大误差”，还是“靠把数据扭曲得很不自然”。

### 统计 penalty 的统一形式

所有统计 penalty 都采用带容忍带的 hinge-squared 形式：

\[
P(z_{adv}, z_{clean}; \tau)=\mathrm{mean}\left[\mathrm{ReLU}\left(\frac{|z_{adv}-z_{clean}|}{\tau}-1\right)^2\right]
\]

含义是：

- 当统计量偏移小于阈值 `\tau` 时，不处罚；
- 当偏移超过阈值时，超出部分按平方增长。

这能显著降低“约束一加上攻击就完全失效”的风险。

## 三类统计惩罚项

### 1. 收益率惩罚 `P_ret`

使用收盘价的对数收益率：

\[
r_t(x)=\log(c_t+\epsilon)-\log(c_{t-1}+\epsilon)
\]

再定义：

\[
P_{ret}=P(r_{adv}, r_{clean}; \tau_{ret})
\]

推荐初值：

- `tau_ret = 0.005`

解释：

- 允许单步对数收益率与 clean 存在小幅偏移；
- 若相邻时间步收益率变化过大，则开始惩罚。

### 2. K 线形态惩罚 `P_candle`

定义两个统计量：

\[
body_t=\frac{close_t-open_t}{open_t+\epsilon}
\]

\[
range_t=\frac{high_t-low_t}{open_t+\epsilon}
\]

分别定义：

\[
P_{body}=P(body_{adv}, body_{clean}; \tau_{body})
\]

\[
P_{range}=P(range_{adv}, range_{clean}; \tau_{range})
\]

再合成为：

\[
P_{candle}=0.5P_{body}+0.5P_{range}
\]

推荐初值：

- `tau_body = 0.005`
- `tau_range = 0.01`

解释：

- `body` 控制开收实体变化；
- `range` 控制蜡烛振幅变化；
- 两者共同限制“虽然 K 线合法，但形态已经明显走样”的攻击样本。

### 3. 成交量动态惩罚 `P_vol`

对成交量先取对数，再看相邻时刻变化：

\[
u_t(x)=\log(volume_t+1)
\]

\[
\Delta u_t=u_t-u_{t-1}
\]

\[
P_{vol}=P(\Delta u_{adv}, \Delta u_{clean}; \tau_{vol})
\]

推荐初值：

- `tau_vol = 0.05`

解释：

- 该项不约束绝对成交量值，而是约束时间动态；
- 主要用于抑制攻击器把成交量通道过度顶满预算。

## 约束攻击目标

### `none` 模式

\[
J=\mathrm{MSE}
\]

### `physical` 模式

\[
J=\mathrm{MSE}
\]

但更新后增加金融物理投影。

### `physical_stat` 模式

\[
J=\mathrm{MSE}-\lambda_{ret}P_{ret}-\lambda_{candle}P_{candle}-\lambda_{vol}P_{vol}
\]

推荐初值：

- `lambda_ret = 1.0`
- `lambda_candle = 0.5`
- `lambda_vol = 0.3`

这组初值体现如下优先级：

1. 收益率路径最重要；
2. K 线形态次之；
3. 成交量动态第三。

## FGSM 与 PGD 的修改方式

### FGSM

在 `none` 模式下仍按当前方式，对 `MSE` 做一次梯度上升。

在 `physical_stat` 模式下，改为对约束目标 `J` 做一次梯度上升，然后投影到“预算盒 + 金融物理可行域”。

### PGD

在 `physical` 或 `physical_stat` 模式下，每一步执行：

1. 计算当前目标 `J` 的梯度；
2. 沿梯度方向更新；
3. 投影到金融物理可行域。

本阶段仍保持：

- 不使用随机初始化；
- 步长相对预算盒定义；
- PGD 步数和单步比率与当前实验口径兼容。

## 输出与诊断

为了支持后续 tradeoff 分析，样本级攻击输出除当前指标外，还应新增：

- `constraint_mode`
- `objective_clean`
- `objective_fgsm`
- `objective_pgd`
- `ret_penalty_fgsm`
- `ret_penalty_pgd`
- `candle_penalty_fgsm`
- `candle_penalty_pgd`
- `vol_penalty_fgsm`
- `vol_penalty_pgd`
- `mean_abs_ret_shift_fgsm`
- `mean_abs_ret_shift_pgd`
- `mean_abs_body_shift_fgsm`
- `mean_abs_body_shift_pgd`
- `mean_abs_range_shift_fgsm`
- `mean_abs_range_shift_pgd`
- `mean_abs_dlogvol_shift_fgsm`
- `mean_abs_dlogvol_shift_pgd`
- `physical_constraints_satisfied_fgsm`
- `physical_constraints_satisfied_pgd`
- `strict_attack_success_fgsm`
- `strict_attack_success_pgd`

### `strict_attack_success` 建议定义

攻击样本被视为“严格成功”，需同时满足：

1. `adv_loss > clean_loss`
2. 金融物理约束全部满足
3. 统计 penalty 没有明显爆掉

这样可以区分：

- 单纯让模型误差变大；
- 在看起来仍像真实金融样本的前提下，让模型误差变大。

## 实验顺序

本设计建议按以下顺序推进，而不是一次性把所有约束全部接入：

### 阶段 A：`none` vs `physical`

先只加入金融物理投影，回答：

- K 线合法化后，攻击是否仍然有效；
- 有效性损失有多大。

### 阶段 B：`physical` vs `physical_stat(ret)`

只加 `P_ret`，回答：

- 收益率路径约束是否已经显著压制攻击。

### 阶段 C：完整 `physical_stat`

再补 `P_candle` 与 `P_vol`，回答：

- 样本是否进一步变得更自然；
- 攻击强度进一步下降了多少。

## 风险与缓解

### 风险 1：约束过强，攻击几乎完全失效

缓解方式：

- 先调低 `lambda`，再考虑放宽 `tau`
- 先单独接入 `P_ret`，不要一开始就三项全上

### 风险 2：投影实现破坏了预算约束

缓解方式：

- 将“预算盒是否仍满足”写成独立单元测试
- 每次投影后单独检查价格列和成交量列的相对偏移

### 风险 3：统计约束与 clean 对齐误差混淆

缓解方式：

- 所有 penalty 都相对在线 clean 窗口定义，而不是相对离线参考特征
- 在报告中显式区分 clean 对齐误差和 adversarial 偏移

### 风险 4：成交量 penalty 主导目标函数

缓解方式：

- 将 `lambda_vol` 设为最低优先级；
- 若攻击显著失效，优先下调 `lambda_vol`。

## 决策记录

本设计在版本 1 中固定以下决策：

- 只在 LSTM 攻击链路中实现受约束攻击；
- 统计约束以 clean 窗口为 reference；
- 金融物理约束使用硬投影；
- 统计约束使用容忍带 hinge penalty；
- 第一版只包含三类统计约束：收益率、K 线形态、成交量动态；
- 第一版固定三种模式：`none`、`physical`、`physical_stat`；
- 优先先做 `physical`，再逐项加统计 penalty，而不是一次性全上。
