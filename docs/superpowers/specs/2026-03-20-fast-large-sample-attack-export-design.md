# 大样本攻击资产快导出脚本设计

## 背景

在 `Transformer` 与 `TCN` 的大样本白盒攻击实验中，原始 `scripts/export_lstm_attack_assets.py` 虽然功能完整，但其 `Alpha158 -> RobustZScoreNorm` 统计量拟合与 Qlib 特征窗口导出路径在当前环境下过慢，不适合作为 `4096+` 样本正式实验的默认导出方案。

本轮实验已经验证了一个更轻的快导出流程：

1. 从 `pred.pkl` / `label.pkl` 构造 `matched_reference`；
2. 用 `raw test split` 导出 `matched_ohlcv_windows.pt`；
3. 复用现有 `normalization_stats.json`；
4. 通过 `LegacyLSTMFeatureBridge + RobustZScoreNormLayer + FillnaLayer` 从 raw window 直接重建 `matched_feature_windows.pt`。

该流程不改变攻击图定义，只是绕开了最慢的 Qlib 特征导出路径。

## 目标

将这条已经验证可用的快导出流程正式脚本化，形成一个独立入口，供后续 `Transformer`、`TCN`、甚至其他模型的大样本攻击实验直接复用。

## 用户确认后的约束

用户已经明确要求：

- 采用独立新脚本，而不是修改现有 `export_lstm_attack_assets.py` 的行为；
- 旧脚本保持原样，避免影响历史实验；
- 快导出流程要能直接生成统一攻击资产契约；
- 后续能直接喂给 `scripts/run_whitebox_attack.py` 使用。

## 方案选择

### 方案 A：新增独立脚本

新增 `scripts/export_large_sample_attack_assets.py`，专门负责快导出。

优点：

- 风险最小；
- 不污染旧脚本语义；
- 适合大样本正式实验场景；
- 后续可在文档中明确区分“Qlib 慢导出”与“快导出”。

缺点：

- 会多一个脚本文件；
- 与旧脚本存在一定逻辑重复。

### 方案 B：给旧脚本增加模式开关

例如在 `export_lstm_attack_assets.py` 增加 `--fast-feature-export`。

优点：

- 入口更少；
- 共享更多参数解析逻辑。

缺点：

- 旧脚本已经较重，再加模式开关后更难维护；
- 容易让历史实验口径变得模糊；
- 增加回归风险。

### 推荐方案

采用方案 A。

## 脚本边界

### 新增正式脚本

创建：

```text
scripts/export_large_sample_attack_assets.py
```

职责：

- 读取 `pred.pkl` / `label.pkl`
- 构造 `matched_reference`
- 通过 `raw test split` 导出 `matched_ohlcv_windows.pt`
- 复用外部传入的 `normalization_stats.json`
- 从 raw window 直接构造 `matched_feature_windows.pt`
- 输出统一 `export_summary.json`

### 兼容壳层

额外创建一个很薄的兼容入口：

```text
scripts/export_whitebox_attack_assets.py
```

该文件不承载完整逻辑，只负责：

- 复用 `build_matched_reference`
- 作为历史计划中引用的轻量兼容入口

这样可以避免已有测试或文档中的 `export_whitebox_attack_assets.py` 引用继续悬空。

## 输入参数

新脚本至少支持：

- `--pred-pkl`
- `--label-pkl`
- `--out-dir`
- `--normalization-stats`
- `--provider-uri`
- `--market`
- `--start-time`
- `--end-time`
- `--fit-start-time`
- `--fit-end-time`
- `--test-start-time`
- `--test-end-time`
- `--label-expr`
- `--raw-window-len`
- `--max-samples`
- `--seed`

## 输出契约

输出目录与旧脚本保持一致：

- `matched_reference.csv`
- `matched_ohlcv_windows.pt`
- `matched_feature_windows.pt`
- `normalization_stats.json`
- `export_summary.json`

其中 `export_summary.json` 至少包含：

- `matched_reference_rows`
- `exported_sample_rows`
- `raw_window_len`
- `raw_feature_dim`
- `missing_raw_keys`
- `exported_feature_rows`
- `feature_window_len`
- `feature_dim`
- `feature_source`
- `normalization_stats_source`

其中：

- `feature_source` 固定写为 `torch_bridge_from_raw`
- `normalization_stats_source` 记录复用的统计量文件路径

## 核心数据流

### 1. matched reference

直接复用现有：

- `build_matched_reference`

这样可以保证与旧脚本保持相同的 `pred/label` 对齐逻辑。

### 2. raw window 导出

直接复用现有：

- `_build_raw_test_split`
- `export_matched_raw_windows`

这部分已经验证为稳定且速度可接受。

### 3. feature window 重建

新脚本不再走 Qlib `Alpha158` 特征窗口导出，而是使用：

- `LegacyLSTMFeatureBridge`
- `RobustZScoreNormLayer`
- `FillnaLayer`

对 `sample_asset["ohlcv"]` 直接生成特征。

这样可以保证：

- 特征维度仍为 `20`
- 序列长度仍为 `20`
- 攻击时使用的特征变换与导出阶段完全一致

## 测试策略

新增：

```text
tests/test_export_large_sample_attack_assets.py
```

最小测试覆盖：

1. `build_feature_windows_from_raw` 输出的 key 与输入 raw asset 一致；
2. 输出 `feature_source = torch_bridge_from_raw`；
3. `normalization_stats.json` 能被原样复用写出；
4. `build_export_summary` 或 `main` 结果包含完整字段；
5. `scripts/export_whitebox_attack_assets.py` 能复用 `build_matched_reference`。

测试应保持轻量，不依赖真实 Qlib 大数据集。

## 验收标准

完成后应满足：

- 可以用新脚本稳定导出大样本攻击资产；
- 导出结果能直接被 `scripts/run_whitebox_attack.py` 消费；
- 不修改旧脚本行为；
- 新测试通过；
- 兼容壳层可消除现有悬空引用。
