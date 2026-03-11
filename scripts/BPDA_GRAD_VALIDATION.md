# BPDA 梯度验证方案说明

本文档说明 `scripts/validate_bpda_gradients.py` 的验证逻辑与指标含义。

## 目标

验证 BPDA 离散算子是否同时满足：

1. 前向保持硬算子语义一致。
2. 反向梯度替代正确（BPDA 梯度与软替代算子梯度一致）。
3. 梯度具备真实优化意义（方向性与边界敏感性符合预期）。

## 覆盖算子

核心检查（core）：

- `bpda_max_pair`
- `bpda_min_pair`
- `bpda_greater`
- `bpda_less`
- `bpda_quantile_window`（`q=0.5`）
- `bpda_idxmax`
- `bpda_idxmin`
- `op_Max`（滚动窗口）
- `op_Min`（滚动窗口）
- `op_Rank`（滚动窗口）

语义检查（semantic）：

- max/min 梯度主导性与 tie（并列）行为
- gt/lt 边界敏感性与反对称性
- idxmax/idxmin 梯度顺序单调性
- quantile 关于 `q` 的单调性与平移不变性
- rank 的边界敏感性

## 核心指标

每个 core 算子会输出：

- `forward_max_abs_err`
- `grad_cosine_bpda_vs_soft`
- `grad_max_abs_err_bpda_vs_soft`
- `grad_finite_rate`
- `soft_fd_rel_err`
- `direction_sign_acc`
- `direction_info_ratio`

## 通过规则

默认 core 阈值：

- 前向误差 <= `1e-9`
- 梯度余弦相似度 >= `0.999`
- BPDA 与 soft 梯度最大绝对误差 <= `5e-6`
- 有限值比例 == `1.0`
- soft 有限差分相对误差 <= `5e-2`
- 当 `direction_info_ratio >= 0.10` 时，`direction_sign_acc` 需高于算子阈值

算子级方向阈值：

- idxmax: `0.65`
- idxmin: `0.65`
- rank: `0.50`
- 其它: `0.75`

## 运行方式

在 WSL 仓库根目录执行：

```bash
conda activate qlib
python scripts/validate_bpda_gradients.py --device cpu --dtype float64
```

仅输出摘要：

```bash
python scripts/validate_bpda_gradients.py --device cpu --dtype float64 --quiet
```

## 输出结果

脚本会生成 JSON 报告：

- `reports/bpda_grad_validation_report.json`

只要存在未通过项，脚本会返回非零退出码，可直接接入 CI 或合并前检查。
