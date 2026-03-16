# 攻击结果可视化设计

## 1. 目标

为当前 LSTM 原始 OHLCV 白盒攻击实验生成一组适用于论文展示与答辩投屏的科研风格图像。图像需要同时支持：

1. `slide` 版本：优先适配 `16:9` 答辩投屏，可读性和视觉冲击优先。
2. `paper` 版本：优先适配论文排版，紧凑、克制、可在缩小后保持清晰。

整体目标不是简单“把结果画出来”，而是建立一套统一的视觉系统，使攻击链路修正、样本级攻击有效性、组合级回测退化和攻击比例敏感性能够被连贯地讲述。

## 2. 图像范围

本轮确认的主图共 6 张：

1. `Clean 对齐修正图`
2. `样本级攻击偏移分布图`
3. `样本级攻击强度 CDF 图`
4. `组合级累计收益曲线图`
5. `多随机种子稳定性图`
6. `攻击比例敏感性图`

同时保留 2 张备份图，用于附录或答辩备用：

1. `PGD 相对 FGSM 优势热力图`
2. `partial_clean 残余偏差图`

## 3. 视觉风格

整体风格采用“论文风 + 答辩友好”的折中方案：

- 白底
- 低饱和主色系
- 暖色表达攻击器，冷色表达 clean / baseline
- 统一字体、线宽、图例顺序和标题规范
- 不使用渐变、3D 效果、重阴影或装饰性图元

颜色语义固定如下：

- `reference_clean`：`#4C566A`
- `partial_clean`：`#A7B1C2`
- `FGSM`：`#E69F00`
- `PGD`：`#D55E00`
- `修正前 / old baseline`：`#B8C1CC`
- `修正后 / new baseline`：`#2F6DB3`

## 4. 图文规范

### 4.1 标题规范

- 图标题只描述“对象 + 指标”，不直接写结论。
- 子图标题尽量简短，例如 `排序对齐`、`年化超额收益退化`。
- 坐标轴文案优先使用中文。

### 4.2 指标方向规范

为避免答辩中出现“有的指标越大越差，有的指标越小越差”的解释负担，所有退化类指标在画图时统一转换为“越高表示攻击越强”的方向，例如：

- `-Δ annualized_excess_return_with_cost`
- `-Δ rank_ic_mean`
- `-Δ information_ratio_with_cost`
- `-Δ max_drawdown_with_cost`

### 4.3 图例规范

图例语义顺序在全局保持一致：

1. `reference_clean`
2. `partial_clean`
3. `FGSM`
4. `PGD`

如果某张图不包含全部对象，则保持剩余对象的相对顺序不变。

## 5. 数据来源

每张图都绑定到已有实验结果，不新增实验口径：

### 图 1：Clean 对齐修正图

- `reports/lstm_whitebox_attack_expanded_v3/attack_summary.json`
- `reports/lstm_whitebox_attack_expanded_v6/attack_summary.json`

### 图 2 / 图 3：样本级攻击图

- `reports/lstm_whitebox_attack_expanded_v6/sample_metrics.csv`

### 图 4：组合级累计收益曲线图

- `reports/partial_attack_backtest_multiseed_ratio5_union/seed_0/daily_comparison.csv`

### 图 5：多随机种子稳定性图

- `reports/partial_attack_backtest_multiseed_ratio5_union/multiseed_summary_stats.csv`

### 图 6：攻击比例敏感性图

- `reports/partial_attack_ratio_sweep_multiseed/ratio_sweep_summary_stats.csv`

### 备份图

- `reports/partial_attack_ratio_sweep_multiseed/ratio_sweep_trend_summary.json`
- `reports/partial_attack_ratio_sweep_multiseed/ratio_sweep_summary_stats.csv`

## 6. 技术设计

### 6.1 绘图库

使用 `matplotlib` 作为主绘图库，必要时使用 `seaborn` 辅助统计图形绘制。这样做的原因是：

- `matplotlib` 对 PDF/PNG 双导出最稳定
- 更适合严格控制科研图的细节
- 便于统一 slide/paper 两种 profile

### 6.2 目录结构

建议新增如下结构：

- `scripts/plotting/style.py`
- `scripts/plotting/loaders.py`
- `scripts/plotting/fig01_clean_alignment.py`
- `scripts/plotting/fig02_sample_shift_box.py`
- `scripts/plotting/fig03_sample_shift_cdf.py`
- `scripts/plotting/fig04_cumulative_return.py`
- `scripts/plotting/fig05_multiseed_stability.py`
- `scripts/plotting/fig06_ratio_sensitivity.py`
- `scripts/plotting/build_all_figures.py`
- `reports/figures/data/`
- `reports/figures/slide/`
- `reports/figures/paper/`

其中：

- `style.py` 负责全局风格、颜色、字体、尺寸 profile
- `loaders.py` 负责读取并整理现有实验文件
- 单图脚本负责各自图像逻辑
- `build_all_figures.py` 负责批量生成主图

### 6.3 profile 机制

所有图脚本统一接受：

- `--profile slide`
- `--profile paper`

这样可以复用一份绘图逻辑，仅在尺寸、字号、线宽和布局参数上切换，不复制代码。

## 7. 实现顺序

第一轮实现优先完成 4 张最关键的主图：

1. `fig01_clean_alignment_repair`
2. `fig06_ratio_sensitivity`
3. `fig04_cumulative_return_ratio5_seed0`
4. `fig02_sample_shift_distribution`

理由：

- 这四张图已经足够支撑答辩主线
- 可以快速得到可展示产物
- 后续 `fig03` 和 `fig05` 作为增强图补上即可

## 8. 验证标准

每张图都需要通过以下检查：

1. `slide` 版在大屏展示下刻度、图例、标题清晰可读。
2. `paper` 版在缩小后仍能看清主趋势。
3. 不同图之间颜色语义一致。
4. 图中不存在方向口径混乱的问题。
5. 输出同时包含 `png` 和 `pdf`。

此外，需要至少做一次“端到端批量出图”验证，确保：

- 作图脚本能成功读取现有实验结果
- 主图文件都按预期写入对应目录
- 文件名与图号一一对应
