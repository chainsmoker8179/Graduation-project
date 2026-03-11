# Alpha158 可导因子模块实施计划

本计划将“可导 Alpha158 因子计算模块”的实现按模块化方式推进，保证每个模块都有明确目标、可验证的测试与可追踪的产出文件。

---

## 总体目标

- 将 Alpha158 因子计算重构为**可导的 PyTorch 模块**，实现端到端梯度回传。
- 对齐 Qlib 0.8 官方语义：前向硬算子一致，反向使用可微近似（BPDA）。
- 形成可复用的离散算子近似与 rolling 底座，并最终实现完整 158 因子的 `TorchFactorExtractor`。

---

## 已完成模块

### ✅ 模块 1：BPDA 基础与逐元素离散算子
**内容**
- `smooth_max_pair/smooth_min_pair`
- `soft_greater/soft_less`
- `bpda_max_pair/bpda_min_pair`
- `bpda_greater/bpda_less`

**文件**
- `alpha158_diff_ops.py`
- `scripts/validate_module1.py`

**验证结果**
- 前向与硬算子一致
- 梯度可回传

---

### ✅ 模块 2：rolling/unfold 工具层
**内容**
- `rolling_unfold` + `rolling_sum/mean/std/max/min`
- 右对齐裁剪 `right_align`

**文件**
- `alpha158_rolling.py`
- `scripts/validate_module2.py`

**验证结果**
- rolling 输出正确
- 对齐裁剪正确

---

### ✅ 模块 3：GPU Soft-Rank / Soft-Sort
**内容**
- NeuralSort 软置换矩阵
- `soft_sort`, `soft_rank(pct=True)`

**文件**
- `alpha158_softsort.py`
- `scripts/validate_module3.py`

**验证结果**
- soft_sort 逼近硬排序
- soft_rank 输出百分位
- 梯度可回传

---

### ✅ 模块 4：Soft-Quantile + BPDA
**内容**
- soft-sort + soft-pick quantile
- BPDA quantile

**文件**
- `alpha158_quantile.py`
- `scripts/validate_module4.py`

**验证结果**
- 前向与硬 quantile 一致（BPDA）
- 梯度可回传

---

### ✅ 模块 5：IdxMax / IdxMin（1-based）
**内容**
- `bpda_idxmax`, `bpda_idxmin`（前向硬 index + 1-based）
- 反向 soft one-hot

**文件**
- `alpha158_idx.py`
- `scripts/validate_module5.py`

**验证结果**
- 前向 index 与 Qlib 1-based 一致
- 梯度可回传

---

## 待完成模块

### 🔜 模块 6：Slope / Rsquare / Resi / Corr
**目标**
- 实现滚动回归类算子（对齐 Qlib 语义）
- 处理 `std≈0` 的稳定性（训练友好）

**计划内容**
- 窗口内线性回归公式（无需依赖 qlib cython）
- `Corr` 需与 Qlib 一致：若任一 std≈0 → 输出 NaN（或用 eps 替代）

**验证**
- 小样本对比 numpy 公式
- 梯度可回传

---

### 🔜 模块 7：表达式解析器
**目标**
- 将 `alpha158_name_expression.csv` 自动解析为 PyTorch 计算图

**计划内容**
- 变量替换 `$close → close_` 等
- `>` / `<` 替换为 `Gt/Lt`
- 构造安全 eval / AST

**验证**
- 选取 3~5 条表达式做数值检查

---

### 🔜 模块 8：TorchFactorExtractor（完整 158 因子）
**目标**
- 输出 shape `(B, L', 158)`
- 统一右对齐裁剪

**计划内容**
- 使用模块 1~7 的算子实现因子表达式
- `vwap` 输入假定在 `x_raw` 中

**验证**
- 输出维度正确
- 因子无 NaN 爆炸
- 梯度回传到输入

---

### 🔜 模块 9：端到端联调（可选）
**目标**
- raw → factor → predictor 梯度链路验证

---

## 已更新论文文档

- `可导特征模块章节.md`
  - 模块 1~4 的论文描述已补充
  - 表述风格已调整为顶会论文逻辑

---

## 关键语义对齐确认（来自 Qlib）

- `Greater/Less` = 元素级 `max/min`（非比较）
- `Rank` = 百分位 (0~1)
- `IdxMax/IdxMin` = 1-based
- rolling 统一 `min_periods=1`

---

## 当前默认参数

- `DEFAULT_TAU_MAXMIN = 5.0`
- `DEFAULT_TEMP_CMP = 0.2`
- `DEFAULT_TAU_IDX = 0.5`
- `DEFAULT_REG_STRENGTH = 0.3`
- `DEFAULT_PICK_STRENGTH = 0.3`

---

如需调整计划顺序或补充细节，请直接告诉我。
