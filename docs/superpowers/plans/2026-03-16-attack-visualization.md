# Attack Visualization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前 LSTM 白盒攻击实验构建一套可复用的科研风格出图脚本，生成适配答辩投屏和论文排版的双版本图像。

**Architecture:** 将出图能力拆分为统一视觉样式层、统一数据读取层、单图绘制脚本和总构建脚本。所有图复用同一语义配色与 profile 机制，输出到 `reports/figures/slide/` 和 `reports/figures/paper/`，同时保留中间作图数据于 `reports/figures/data/`。

**Tech Stack:** Python、matplotlib、seaborn、pandas、现有实验结果 CSV/JSON 文件

---

## 文件结构

- Create: `scripts/plotting/style.py`
  - 统一定义 `slide` / `paper` 两套视觉 profile、颜色、字体、线宽和导出参数。
- Create: `scripts/plotting/loaders.py`
  - 负责从现有 `reports/` 结果中读取并整理图表所需数据。
- Create: `scripts/plotting/fig01_clean_alignment.py`
  - 绘制 clean 对齐修正图。
- Create: `scripts/plotting/fig02_sample_shift_box.py`
  - 绘制样本级攻击偏移箱线图。
- Create: `scripts/plotting/fig03_sample_shift_cdf.py`
  - 绘制样本级攻击偏移累计分布图。
- Create: `scripts/plotting/fig04_cumulative_return.py`
  - 绘制组合级累计收益曲线图。
- Create: `scripts/plotting/fig05_multiseed_stability.py`
  - 绘制 5% 多随机种子稳定性图。
- Create: `scripts/plotting/fig06_ratio_sensitivity.py`
  - 绘制攻击比例敏感性图。
- Create: `scripts/plotting/build_all_figures.py`
  - 统一生成主图与可选备份图。
- Create: `reports/figures/data/`
  - 存放中间作图表。
- Create: `reports/figures/slide/`
  - 存放答辩版图像。
- Create: `reports/figures/paper/`
  - 存放论文版图像。
- Test: `tests/plotting/test_figure_loaders.py`
  - 验证 loader 输出字段和关键聚合结果。
- Test: `tests/plotting/test_next_figure_outputs.py`
  - 验证图像构建脚本会生成预期文件。

## Chunk 1: 样式和数据层

### Task 1: 为 loader 写失败测试

**Files:**
- Create: `tests/plotting/test_figure_loaders.py`

- [ ] **Step 1: 写测试，验证 clean alignment loader 能读出 `v3` 和 `v6` 的关键字段**
- [ ] **Step 2: 写测试，验证 ratio sensitivity loader 能返回 `1% / 5% / 10%` 三档比例与 FGSM/PGD 四项指标**
- [ ] **Step 3: 运行测试，确认在 loader 尚未实现前失败**

Run: `pytest tests/plotting/test_figure_loaders.py -v`
Expected: FAIL，提示 loader 模块或函数不存在

### Task 2: 实现统一样式层

**Files:**
- Create: `scripts/plotting/style.py`

- [ ] **Step 1: 定义全局颜色映射和图例标签映射**
- [ ] **Step 2: 定义 `slide` 和 `paper` 两套尺寸、字号、线宽配置**
- [ ] **Step 3: 提供统一的 figure/save helper，支持同时导出 `png` 和 `pdf`**

### Task 3: 实现统一数据读取层

**Files:**
- Create: `scripts/plotting/loaders.py`
- Create: `reports/figures/data/`

- [ ] **Step 1: 实现 clean alignment、sample shift、cumulative return、multiseed、ratio sensitivity 的 loader**
- [ ] **Step 2: 将关键中间表保存到 `reports/figures/data/`**
- [ ] **Step 3: 重新运行 loader 测试，确认通过**

Run: `pytest tests/plotting/test_figure_loaders.py -v`
Expected: PASS

## Chunk 2: 先实现四张主图

### Task 4: 实现图 1 和图 6

**Files:**
- Create: `scripts/plotting/fig01_clean_alignment.py`
- Create: `scripts/plotting/fig06_ratio_sensitivity.py`

- [ ] **Step 1: 为图 1 写失败测试，验证脚本会生成 `slide` 和 `paper` 两套输出文件**
- [ ] **Step 2: 实现图 1 的 `2x2` 分组柱状图**
- [ ] **Step 3: 为图 6 写失败测试，验证脚本会生成 `slide` 和 `paper` 两套输出文件**
- [ ] **Step 4: 实现图 6 的 `2x2` 折线图 + 误差条**
- [ ] **Step 5: 运行对应测试，确认通过**

Run: `pytest tests/plotting/test_next_figure_outputs.py -k 'fig01 or fig06' -v`
Expected: PASS

### Task 5: 实现图 4 和图 2

**Files:**
- Create: `scripts/plotting/fig04_cumulative_return.py`
- Create: `scripts/plotting/fig02_sample_shift_box.py`

- [ ] **Step 1: 为图 4 写失败测试，验证累计收益曲线图输出存在**
- [ ] **Step 2: 实现图 4 的四条累计收益曲线**
- [ ] **Step 3: 为图 2 写失败测试，验证箱线图输出存在**
- [ ] **Step 4: 实现图 2 的箱线图，可选叠加轻量散点**
- [ ] **Step 5: 运行对应测试，确认通过**

Run: `pytest tests/plotting/test_next_figure_outputs.py -k 'fig02 or fig04' -v`
Expected: PASS

## Chunk 3: 补充增强图与总构建

### Task 6: 实现图 3 和图 5

**Files:**
- Create: `scripts/plotting/fig03_sample_shift_cdf.py`
- Create: `scripts/plotting/fig05_multiseed_stability.py`

- [ ] **Step 1: 实现图 3 的 CDF 曲线图**
- [ ] **Step 2: 实现图 5 的多随机种子 `2x2` 柱状图 + 误差条**
- [ ] **Step 3: 为两张图补测试并运行**

Run: `pytest tests/plotting/test_next_figure_outputs.py -k 'fig03 or fig05' -v`
Expected: PASS

### Task 7: 实现总构建脚本

**Files:**
- Create: `scripts/plotting/build_all_figures.py`

- [ ] **Step 1: 提供按 profile 批量生成主图的入口**
- [ ] **Step 2: 支持 `--profile slide`、`--profile paper` 和 `--all`**
- [ ] **Step 3: 运行一次 `slide` 和一次 `paper` 构建**

Run: `python scripts/plotting/build_all_figures.py --profile slide`
Expected: `reports/figures/slide/` 下出现 4-6 张主图的 `png` 和 `pdf`

Run: `python scripts/plotting/build_all_figures.py --profile paper`
Expected: `reports/figures/paper/` 下出现 4-6 张主图的 `png` 和 `pdf`

## Chunk 4: 验证与文档

### Task 8: 做最小端到端验证

**Files:**
- Verify: `reports/figures/slide/`
- Verify: `reports/figures/paper/`
- Verify: `reports/figures/data/`

- [ ] **Step 1: 检查图像文件名与图号对应**
- [ ] **Step 2: 抽查图 1、图 4、图 6 的 `slide` 与 `paper` 输出是否同时存在**
- [ ] **Step 3: 运行测试目录和主构建脚本，确认全部成功**

Run: `pytest tests/plotting -v`
Expected: PASS

Run: `python scripts/plotting/build_all_figures.py --all`
Expected: 所有主图生成成功，无异常退出

### Task 9: 补最小使用说明

**Files:**
- Modify: `reports/partial_attack_ratio_sweep_multiseed/攻击比例对比报告.md`
  - 仅在适当位置补一句“配套图像位置”说明；如果不适合则不改。

- [ ] **Step 1: 记录图像输出目录和主构建命令**
- [ ] **Step 2: 仅在确有必要时补文档引用，避免文档膨胀**
