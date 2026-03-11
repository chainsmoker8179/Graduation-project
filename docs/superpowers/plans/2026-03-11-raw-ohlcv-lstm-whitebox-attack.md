# 原始 OHLCV LSTM 白盒攻击实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个小样本白盒 `FGSM/PGD` 工作流，在不先重新训练迁移来的基线 LSTM 的前提下，攻击重建后的 `原始 OHLCV -> 20 个旧版特征 -> RobustZScoreNorm -> 旧版 LSTM` 流水线。

**Architecture:** 实现上保留 Qlib 作为离线侧的资产导出工具，用于导出拟合后的归一化统计量和匹配样本参照；所有攻击时计算则全部保留在 Torch 中。攻击图由可微 20 特征提取器、Torch 版 `RobustZScoreNorm`、加载旧 `state_dict` 的 LSTM，以及负责 clean、FGSM、PGD 评估的小样本攻击脚本构成。

**Tech Stack:** Python、PyTorch、当前仓库中已有的可微 Alpha158 模块、Qlib 离线资产导出、pytest 或最小脚本级验证、pandas 对齐参考数据。

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-11-raw-ohlcv-lstm-whitebox-attack-design.md`
  - 已批准设计文档
- Create: `scripts/export_lstm_attack_assets.py`
  - 在具备 Qlib 环境的 Python 下运行，导出归一化统计量和匹配 clean 参照
- Create: `legacy_lstm_predictor.py`
  - 旧版 LSTM 头部的纯 Torch 重建及 `state_dict` 加载逻辑
- Create: `legacy_lstm_preprocess.py`
  - Torch 版 `RobustZScoreNorm` 层和轻量预处理辅助逻辑
- Create: `legacy_lstm_feature_bridge.py`
  - 从原始 `OHLCV` 计算 20 个旧版特征的可微桥接层
- Create: `scripts/run_lstm_whitebox_attack.py`
  - clean 闸门验证与 FGSM/PGD smoke 攻击运行脚本
- Create: `reports/lstm_whitebox_attack/`
  - 样本级结果与汇总输出目录
- Create: `tests/test_legacy_lstm_preprocess.py`
  - 归一化层行为单元测试
- Create: `tests/test_legacy_lstm_predictor.py`
  - LSTM 重建与权重加载单元测试
- Create: `tests/test_legacy_lstm_feature_bridge.py`
  - 特征桥接层形状与顺序契约测试
- Create: `tests/test_lstm_whitebox_attack_smoke.py`
  - 基于极小合成数据的 clean 前向/反向与攻击单调性 smoke test

## Chunk 1: 离线资产导出

### Task 1: 定义导出资产契约

**Files:**
- Create: `scripts/export_lstm_attack_assets.py`
- Test: `scripts/export_lstm_attack_assets.py --help`

- [ ] **Step 1: 先写出导出脚本骨架，并定义清晰的 CLI 参数**

参数至少应包括：
- 旧版 `pred.pkl`
- 旧版 `label.pkl`
- 输出目录
- provider URI
- market
- 时间区间
- 特征列表路径或内置旧版 20 特征列表
- 匹配样本最大导出数量

- [ ] **Step 2: 运行帮助输出，确认 CLI 能正常解析**

Run: `python scripts/export_lstm_attack_assets.py --help`
Expected: 在 Qlib 可用环境下打印 usage 文本并返回 0

- [ ] **Step 3: 实现旧版特征列表与匹配索引加载**

脚本需要：
- 读取 `pred.pkl`
- 读取 `label.pkl`
- 求交集索引
- 只保留精确匹配的行

- [ ] **Step 4: 导出一份匹配参考表**

输出紧凑表，例如 `matched_reference.parquet` 或 `matched_reference.csv`，包含：
- `datetime`
- `instrument`
- 旧版 prediction
- 旧版 label

- [ ] **Step 5: 验证匹配参考表已生成且非空**

Run: `python scripts/export_lstm_attack_assets.py ...`
Expected: 输出文件生成，且命令打印匹配样本数

- [ ] **Step 6: Commit**

```bash
git add scripts/export_lstm_attack_assets.py
git commit -m "feat: add LSTM attack asset export skeleton"
```

### Task 2: 导出拟合后的归一化统计量

**Files:**
- Modify: `scripts/export_lstm_attack_assets.py`
- Test: 生成的归一化统计量 JSON/NPZ

- [ ] **Step 1: 先写失败检查，约束必须导出全部 20 个特征的归一化统计量**

在脚本执行路径中加入断言：若 handler 无法提供全部 20 个特征的 `RobustZScoreNorm` 拟合统计量，则直接报错。

- [ ] **Step 2: 运行导出路径，确认在实现前该检查会失败**

Run: `python scripts/export_lstm_attack_assets.py ...`
Expected: 失败信息明确指出归一化统计量导出尚未实现

- [ ] **Step 3: 实现对拟合 `median` 和稳健尺度的提取**

导出内容包括：
- 有序特征名
- 每个特征的拟合中心
- 每个特征的拟合稳健尺度
- clipping 标志

- [ ] **Step 4: 将归一化统计量写成稳定格式的产物**

输出 `normalization_stats.json` 或 `normalization_stats.npz` 到指定目录。

- [ ] **Step 5: 重新运行导出并检查统计量文件内容**

Expected: 文件存在；恰好包含 20 个有序特征；不存在缺失尺度

- [ ] **Step 6: Commit**

```bash
git add scripts/export_lstm_attack_assets.py
git commit -m "feat: export legacy normalization statistics"
```

### Task 3: 导出匹配后的原始 OHLCV 窗口

**Files:**
- Modify: `scripts/export_lstm_attack_assets.py`
- Test: 生成的原始样本资产

- [ ] **Step 1: 先写失败检查，要求脚本必须输出匹配样本对应的原始 OHLCV 序列窗口**

- [ ] **Step 2: 运行导出，确认该检查在实现前会失败**

Run: `python scripts/export_lstm_attack_assets.py ...`
Expected: 失败信息指出原始 `OHLCV` 窗口导出尚未实现

- [ ] **Step 3: 实现与 `step_len=20` 对齐的窗口提取逻辑**

每条记录需要保留：
- `datetime`
- `instrument`
- 该样本窗口的原始 `OHLCV` 张量或数组
- 对齐标签
- 旧版 prediction

- [ ] **Step 4: 将原始窗口资产落盘**

使用易于 Torch 读取的格式，例如 `.pt`、`.npz` 或 `.pkl`。

- [ ] **Step 5: 重新运行导出并验证样本数量与张量形状**

Expected: 窗口资产存在；值全部有限；序列长度等于 20；特征宽度等于 5

- [ ] **Step 6: Commit**

```bash
git add scripts/export_lstm_attack_assets.py
git commit -m "feat: export matched OHLCV attack windows"
```

## Chunk 2: Torch 重建

### Task 4: 重建旧版 LSTM 预测器

**Files:**
- Create: `legacy_lstm_predictor.py`
- Test: `tests/test_legacy_lstm_predictor.py`

- [ ] **Step 1: 先写模型构造失败测试**

```python
from legacy_lstm_predictor import LegacyLSTMPredictor

def test_legacy_lstm_predictor_output_shape():
    model = LegacyLSTMPredictor(d_feat=20, hidden_size=64, num_layers=2, dropout=0.0)
    x = torch.randn(4, 20, 20)
    y = model(x)
    assert y.shape == (4,)
```

- [ ] **Step 2: 运行测试，确认它先失败**

Run: `pytest tests/test_legacy_lstm_predictor.py::test_legacy_lstm_predictor_output_shape -v`
Expected: 因类不存在或导入失败而 FAIL

- [ ] **Step 3: 实现最小预测器模块**

需要重建：
- 堆叠 LSTM
- 最后一步读出
- 线性输出头

- [ ] **Step 4: 重新运行测试，确认通过**

Run: `pytest tests/test_legacy_lstm_predictor.py::test_legacy_lstm_predictor_output_shape -v`
Expected: PASS

- [ ] **Step 5: 再写一个权重加载失败测试**

```python
def test_legacy_lstm_predictor_loads_exported_state_dict():
    model = load_legacy_lstm_from_files(config_path, state_dict_path)
    assert isinstance(model, torch.nn.Module)
```

- [ ] **Step 6: 运行权重加载测试，确认在 loader 实现前失败**

Run: `pytest tests/test_legacy_lstm_predictor.py::test_legacy_lstm_predictor_loads_exported_state_dict -v`
Expected: FAIL

- [ ] **Step 7: 实现 loader 工具函数**

加载：
- `/home/chainsmoker/qlib_test/model/lstm_state_dict_config.json`
- `/home/chainsmoker/qlib_test/model/lstm_state_dict.pt`

- [ ] **Step 8: 重新运行全部 predictor 测试**

Run: `pytest tests/test_legacy_lstm_predictor.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add legacy_lstm_predictor.py tests/test_legacy_lstm_predictor.py
git commit -m "feat: rebuild legacy LSTM predictor"
```

### Task 5: 在 Torch 中复现 `RobustZScoreNorm`

**Files:**
- Create: `legacy_lstm_preprocess.py`
- Test: `tests/test_legacy_lstm_preprocess.py`

- [ ] **Step 1: 先写中心-尺度归一化失败测试**

```python
from legacy_lstm_preprocess import RobustZScoreNormLayer

def test_robust_zscore_layer_matches_expected_formula():
    layer = RobustZScoreNormLayer(
        center=torch.tensor([1.0, 2.0]),
        scale=torch.tensor([2.0, 4.0]),
        clip_outlier=False,
    )
    x = torch.tensor([[[3.0, 6.0]]])
    y = layer(x)
    assert torch.allclose(y, torch.tensor([[[1.0, 1.0]]]))
```

- [ ] **Step 2: 运行归一化测试，确认先失败**

Run: `pytest tests/test_legacy_lstm_preprocess.py::test_robust_zscore_layer_matches_expected_formula -v`
Expected: FAIL

- [ ] **Step 3: 实现归一化层**

该层应当：
- 将中心和尺度注册为冻结 buffer
- 对有序特征张量做归一化
- 按需裁剪到 `[-3, 3]`

- [ ] **Step 4: 重新运行归一化测试**

Run: `pytest tests/test_legacy_lstm_preprocess.py::test_robust_zscore_layer_matches_expected_formula -v`
Expected: PASS

- [ ] **Step 5: 增加一个截断失败测试**

```python
def test_robust_zscore_layer_clips_outliers():
    layer = RobustZScoreNormLayer(
        center=torch.tensor([0.0]),
        scale=torch.tensor([1.0]),
        clip_outlier=True,
    )
    x = torch.tensor([[[10.0]]])
    y = layer(x)
    assert y.item() == 3.0
```

- [ ] **Step 6: 运行截断测试，确认若无 clipping 实现则失败**

Run: `pytest tests/test_legacy_lstm_preprocess.py::test_robust_zscore_layer_clips_outliers -v`
Expected: 在 clipping 未实现前 FAIL

- [ ] **Step 7: 实现 clipping 和可选的 NaN 修复辅助逻辑**

- [ ] **Step 8: 重新运行全部 preprocess 测试**

Run: `pytest tests/test_legacy_lstm_preprocess.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add legacy_lstm_preprocess.py tests/test_legacy_lstm_preprocess.py
git commit -m "feat: add Torch robust z-score preprocessing"
```

### Task 6: 构建 20 特征可微桥接层

**Files:**
- Create: `legacy_lstm_feature_bridge.py`
- Test: `tests/test_legacy_lstm_feature_bridge.py`

- [ ] **Step 1: 先写特征顺序与维度失败测试**

```python
from legacy_lstm_feature_bridge import LegacyLSTMFeatureBridge

def test_feature_bridge_emits_20_ordered_features():
    bridge = LegacyLSTMFeatureBridge()
    x = torch.randn(2, 80, 5)
    feats = bridge(x)
    assert feats.shape[-1] == 20
```

- [ ] **Step 2: 运行桥接层测试，确认先失败**

Run: `pytest tests/test_legacy_lstm_feature_bridge.py::test_feature_bridge_emits_20_ordered_features -v`
Expected: FAIL

- [ ] **Step 3: 利用现有 Alpha158 图执行器实现最小桥接层**

桥接层应当：
- 复用现有因子模板机制
- 只计算这 20 个旧版特征
- 固定特征顺序

- [ ] **Step 4: 重新运行形状测试**

Run: `pytest tests/test_legacy_lstm_feature_bridge.py::test_feature_bridge_emits_20_ordered_features -v`
Expected: PASS

- [ ] **Step 5: 增加有限值输出失败测试**

```python
def test_feature_bridge_outputs_finite_values_on_finite_input():
    bridge = LegacyLSTMFeatureBridge()
    x = torch.rand(2, 80, 5).abs() + 1e-3
    feats = bridge(x)
    assert torch.isfinite(feats).all()
```

- [ ] **Step 6: 运行有限值测试，确认若存在数值不稳定则失败**

Run: `pytest tests/test_legacy_lstm_feature_bridge.py::test_feature_bridge_outputs_finite_values_on_finite_input -v`
Expected: FAIL 或因未实现而失败

- [ ] **Step 7: 修正对齐或特征处理，直到两个测试都通过**

- [ ] **Step 8: 重新运行全部 bridge 测试**

Run: `pytest tests/test_legacy_lstm_feature_bridge.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add legacy_lstm_feature_bridge.py tests/test_legacy_lstm_feature_bridge.py
git commit -m "feat: add legacy LSTM feature bridge"
```

## Chunk 3: 端到端攻击运行器

### Task 7: 组装端到端 clean 推理图

**Files:**
- Modify: `legacy_lstm_predictor.py`
- Modify: `legacy_lstm_preprocess.py`
- Modify: `legacy_lstm_feature_bridge.py`
- Create: `scripts/run_lstm_whitebox_attack.py`
- Test: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 先写 clean 前向失败 smoke test**

```python
def test_clean_pipeline_forward_returns_finite_predictions():
    batch = build_tiny_fake_ohlcv_batch()
    model = build_attack_pipeline(...)
    pred = model(batch["x"])
    assert pred.shape[0] == batch["x"].shape[0]
    assert torch.isfinite(pred).all()
```

- [ ] **Step 2: 运行 smoke test，确认先失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_clean_pipeline_forward_returns_finite_predictions -v`
Expected: FAIL

- [ ] **Step 3: 在攻击脚本中实现组合后的 clean 流水线**

组合：
- 特征桥接层
- 稳健归一化层
- 必要的 NaN 修复
- 旧版 LSTM 预测器

- [ ] **Step 4: 重新运行 clean 前向 smoke test**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_clean_pipeline_forward_returns_finite_predictions -v`
Expected: PASS

- [ ] **Step 5: 再写一个反向链路失败 smoke test**

```python
def test_clean_pipeline_backward_reaches_raw_ohlcv():
    batch = build_tiny_fake_ohlcv_batch()
    x = batch["x"].clone().requires_grad_(True)
    model = build_attack_pipeline(...)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, batch["y"])
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().mean().item() > 0
```

- [ ] **Step 6: 运行反向链路测试，确认在图未完整前失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_clean_pipeline_backward_reaches_raw_ohlcv -v`
Expected: 若梯度无法回传则 FAIL

- [ ] **Step 7: 修正流水线，直到两个 smoke test 都通过**

- [ ] **Step 8: 重新运行 smoke test 文件**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/run_lstm_whitebox_attack.py tests/test_lstm_whitebox_attack_smoke.py legacy_lstm_predictor.py legacy_lstm_preprocess.py legacy_lstm_feature_bridge.py
git commit -m "feat: assemble clean LSTM attack pipeline"
```

### Task 8: 实现 clean 对齐闸门

**Files:**
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Test: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 先写对齐指标输出失败测试**

```python
def test_attack_runner_reports_clean_alignment_metrics():
    result = run_clean_gate(...)
    assert "clean_loss_mean" in result
    assert "prediction_rank_corr" in result
```

- [ ] **Step 2: 运行对齐测试，确认先失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_attack_runner_reports_clean_alignment_metrics -v`
Expected: FAIL

- [ ] **Step 3: 实现小样本匹配加载与 clean 闸门检查**

运行器应当：
- 读取匹配参考资产
- 以确定性方式随机采样小子集
- 计算 clean 预测
- 与旧版预测做比较
- 计算梯度有效性统计

- [ ] **Step 4: 重新运行对齐测试**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_attack_runner_reports_clean_alignment_metrics -v`
Expected: PASS

- [ ] **Step 5: 为闸门失败和成功增加 CLI 级别报告**

- [ ] **Step 6: 重新运行 smoke test 文件**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/run_lstm_whitebox_attack.py tests/test_lstm_whitebox_attack_smoke.py
git commit -m "feat: add clean alignment gate for attack runner"
```

### Task 9: 实现 FGSM 和 PGD

**Files:**
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Test: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 先写 FGSM loss 增大失败测试**

```python
def test_fgsm_does_not_reduce_loss_on_tiny_batch():
    result = run_fgsm_attack(...)
    assert result["adv_loss_mean"] >= result["clean_loss_mean"]
```

- [ ] **Step 2: 运行 FGSM 测试，确认先失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_fgsm_does_not_reduce_loss_on_tiny_batch -v`
Expected: FAIL

- [ ] **Step 3: 实现带相对预算与投影的 FGSM**

支持：
- 统一价格预算
- 独立成交量预算
- 按元素 floor
- 非负投影

- [ ] **Step 4: 重新运行 FGSM 测试**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_fgsm_does_not_reduce_loss_on_tiny_batch -v`
Expected: PASS

- [ ] **Step 5: 再写一个 PGD loss 增大失败测试**

```python
def test_pgd_does_not_reduce_loss_on_tiny_batch():
    result = run_pgd_attack(...)
    assert result["adv_loss_mean"] >= result["clean_loss_mean"]
```

- [ ] **Step 6: 运行 PGD 测试，确认先失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_pgd_does_not_reduce_loss_on_tiny_batch -v`
Expected: FAIL

- [ ] **Step 7: 实现无随机初始化、逐步投影的 PGD**

- [ ] **Step 8: 重新运行 PGD 测试**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_pgd_does_not_reduce_loss_on_tiny_batch -v`
Expected: PASS

- [ ] **Step 9: 增加扰动使用情况和成功率结果汇报**

- [ ] **Step 10: 重新运行 smoke test 文件**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add scripts/run_lstm_whitebox_attack.py tests/test_lstm_whitebox_attack_smoke.py
git commit -m "feat: add FGSM and PGD white-box attacks"
```

## Chunk 4: 端到端验证

### Task 10: 在一小段参考样本上运行离线导出

**Files:**
- Modify: `scripts/export_lstm_attack_assets.py` if needed
- Output: `reports/lstm_whitebox_attack/` 或配置指定的资产目录

- [ ] **Step 1: 在具备 Qlib 的环境中运行导出命令**

Run: `python scripts/export_lstm_attack_assets.py --pred-path prediction/pred.pkl --label-path prediction/label.pkl --out-dir reports/lstm_whitebox_attack/assets --max-samples 64 ...`
Expected: 生成匹配参考、归一化统计量和原始窗口资产

- [ ] **Step 2: 检查导出文件清单**

Run: `find reports/lstm_whitebox_attack/assets -maxdepth 2 -type f | sort`
Expected: 至少存在参考表、归一化统计量、原始窗口资产

- [ ] **Step 3: Commit**

```bash
git add reports/lstm_whitebox_attack/assets
git commit -m "chore: export small-sample LSTM attack assets"
```

### Task 11: 运行 clean 闸门与 smoke attack

**Files:**
- Use: `scripts/run_lstm_whitebox_attack.py`
- Output: `reports/lstm_whitebox_attack/`

- [ ] **Step 1: 在匹配子集上运行 clean 闸门评估**

Run: `python scripts/run_lstm_whitebox_attack.py --assets-dir reports/lstm_whitebox_attack/assets --mode clean --sample-size 16 --seed 0 --out-dir reports/lstm_whitebox_attack/run01`
Expected: 生成 clean 指标和梯度统计，且闸门通过

- [ ] **Step 2: 运行 FGSM smoke attack**

Run: `python scripts/run_lstm_whitebox_attack.py --assets-dir reports/lstm_whitebox_attack/assets --mode fgsm --sample-size 16 --seed 0 --out-dir reports/lstm_whitebox_attack/run01 ...`
Expected: 输出样本级与汇总级结果

- [ ] **Step 3: 运行 PGD smoke attack**

Run: `python scripts/run_lstm_whitebox_attack.py --assets-dir reports/lstm_whitebox_attack/assets --mode pgd --sample-size 16 --seed 0 --out-dir reports/lstm_whitebox_attack/run01 ...`
Expected: 输出样本级与汇总级结果

- [ ] **Step 4: 检查报告产物**

Run: `find reports/lstm_whitebox_attack/run01 -maxdepth 2 -type f | sort`
Expected: 至少存在 clean summary、FGSM summary、PGD summary 和样本级结果表

- [ ] **Step 5: Commit**

```bash
git add reports/lstm_whitebox_attack/run01
git commit -m "chore: run LSTM white-box attack smoke validation"
```

### Task 12: 最终验证

**Files:**
- Use: 所有新建文件

- [ ] **Step 1: 运行所有新模块的单元测试**

Run: `pytest tests/test_legacy_lstm_preprocess.py tests/test_legacy_lstm_predictor.py tests/test_legacy_lstm_feature_bridge.py tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 2: 重新从头运行一条 clean 命令**

Run: `python scripts/run_lstm_whitebox_attack.py --assets-dir reports/lstm_whitebox_attack/assets --mode clean --sample-size 8 --seed 1 --out-dir reports/lstm_whitebox_attack/verify`
Expected: PASS，生成 clean 汇总

- [ ] **Step 3: 重新从头运行一条 adversarial 命令**

Run: `python scripts/run_lstm_whitebox_attack.py --assets-dir reports/lstm_whitebox_attack/assets --mode fgsm --sample-size 8 --seed 1 --out-dir reports/lstm_whitebox_attack/verify ...`
Expected: PASS，生成 adversarial 汇总

- [ ] **Step 4: 在交接前检查 diff 范围**

Run: `git diff -- docs/superpowers/specs/2026-03-11-raw-ohlcv-lstm-whitebox-attack-design.md docs/superpowers/plans/2026-03-11-raw-ohlcv-lstm-whitebox-attack.md legacy_lstm_predictor.py legacy_lstm_preprocess.py legacy_lstm_feature_bridge.py scripts/export_lstm_attack_assets.py scripts/run_lstm_whitebox_attack.py tests/`
Expected: diff 仅覆盖计划范围内的文件与改动

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-03-11-raw-ohlcv-lstm-whitebox-attack-design.md docs/superpowers/plans/2026-03-11-raw-ohlcv-lstm-whitebox-attack.md legacy_lstm_predictor.py legacy_lstm_preprocess.py legacy_lstm_feature_bridge.py scripts/export_lstm_attack_assets.py scripts/run_lstm_whitebox_attack.py tests reports/lstm_whitebox_attack
git commit -m "feat: add raw OHLCV white-box attack workflow for legacy LSTM"
```
