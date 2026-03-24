# Transformer / TCN 样本级白盒攻击迁移设计

## 背景

当前仓库已经完成了基于原始 `OHLCV` 输入的 LSTM 样本级白盒攻击链路，核心结论包括：

- `原始 OHLCV -> 20 维旧版特征 -> RobustZScoreNorm -> Fillna -> LSTM -> loss` 的可微攻击图已经打通。
- 基于 `MSE` 最大化的 `FGSM` 与 `PGD` 在小样本和扩大样本上都已经形成可复用基线。
- clean gate、相对扰动预算、样本级报告格式都已经稳定。

接下来的目标不是重新设计攻击算法，而是把这一条已经验证过的样本级攻击链路迁移到其他旧基线模型，首先覆盖 `Transformer` 和 `TCN`。用户明确要求：

- 只做“样本级 white-box attack”的迁移；
- 不在本阶段扩展到 `partial attack backtest`；
- 尽量复用 LSTM 版本中已经验证过的攻击逻辑；
- 如果现有实现把模型写死成了 LSTM，需要抽象出统一模型接口；
- 第一版允许把模型加载运行环境限定为 `qlib` conda 环境。

## 目标

构建一个统一的样本级白盒攻击框架，使得 `LSTM`、`Transformer` 与 `TCN` 都能复用同一套：

- 原始 `OHLCV` 输入桥接层；
- 20 维旧版特征提取与归一化；
- clean gate 验证；
- `FGSM` / `PGD` 扰动生成；
- 样本级结果导出与 Markdown 报告。

第一版以“跑通且结果方向正确”为主，不要求在本阶段完成多模型横向统计、组合级回测或论文主文图表。

## 范围内内容

- 统一三套旧模型资产的本地目录契约。
- 为三套模型补齐结构化 `model_config.json`。
- 抽象出共享攻击骨架，使其不再依赖 `LSTM` 特定类名。
- 为 `LSTM`、`Transformer`、`TCN` 分别实现统一接口的模型适配器。
- 新增统一攻击入口脚本，按 `--model-name` 路由模型。
- 保留旧的 LSTM 专用脚本作为兼容入口或轻量壳层。
- 在小样本上验证三模型 clean、FGSM、PGD 的基本可用性。

## 范围外内容

- 不在本阶段接入 `partial attack backtest`。
- 不在本阶段扩展到 `GRU`、`ALSTM` 等更多模型。
- 不在本阶段实现非 `qlib` 环境下的 `Transformer/TCN` 兼容加载。
- 不在本阶段统一三模型的最优 clean gate 阈值。
- 不在本阶段补做显著性检验、组合收益图或论文写作段落。

## 基线约束

三套模型当前共享以下前置条件：

- 输入特征维度：`20`
- 序列长度：`20`
- 特征选择：旧版固定 20 特征
- 预处理：`FilterCol -> RobustZScoreNorm(clip_outlier=True) -> Fillna`
- 攻击目标：最大化 `MSE`
- 扰动空间：原始 `OHLCV`
- 扰动预算：按当前值百分比约束，价格列与成交量列分组

这意味着：

- `原始 OHLCV -> 20 特征` 的桥接逻辑可以完全共享；
- `RobustZScoreNorm` 与 `Fillna` 的 Torch 复现可以完全共享；
- `FGSM` / `PGD` 与投影逻辑可以完全共享；
- 唯一需要按模型区分的是“最后一段特征到分数的预测头”。

## 统一资产目录契约

为了避免入口脚本继续兼容大小写目录名、`pred/` 与 `prediction/` 混用、不同模型不同文件名等历史差异，三套模型资产统一为：

```text
origin_model_pred/
  lstm/
    model/
      state_dict.pt
      trained_model.pkl
      model_config.json
    prediction/
      pred.pkl
      label.pkl
  transformer/
    model/
      state_dict.pt
      trained_model.pkl
      model_config.json
    prediction/
      pred.pkl
      label.pkl
  tcn/
    model/
      state_dict.pt
      trained_model.pkl
      model_config.json
    prediction/
      pred.pkl
      label.pkl
```

其中：

- `state_dict.pt` 是正式加载入口；
- `trained_model.pkl` 仅保留作兜底排障，不作为第一版主加载源；
- `model_config.json` 是重建模型结构的必需配置；
- `pred.pkl` 和 `label.pkl` 是 clean 对齐与样本匹配参照。

## `model_config.json` 最小契约

每个模型目录下都必须包含一份结构化配置：

```json
{
  "model_name": "transformer",
  "qlib_model_class": "TransformerModel",
  "qlib_model_module": "qlib.contrib.model.pytorch_transformer_ts",
  "model_kwargs": {
    "d_feat": 20,
    "num_layers": 2
  },
  "feature_spec": {
    "d_feat": 20,
    "step_len": 20
  }
}
```

设计约束如下：

- `model_name` 用于统一入口脚本路由；
- `qlib_model_class` 和 `qlib_model_module` 用于在 `qlib` 环境里重建 `Transformer/TCN`；
- `model_kwargs` 必须足以独立构造模型，不依赖反序列化 `trained_model.pkl`；
- `feature_spec` 用于显式校验 `d_feat=20`、`step_len=20` 等输入契约。

## 为什么第一版不以 `trained_model.pkl` 为主加载源

在现有环境中，直接反序列化 `trained_model.pkl` 已经暴露出一个现实问题：部分对象是在 CUDA 环境下序列化的，在当前 CPU-only 或不同 CUDA 配置的环境里恢复时会产生设备映射错误。

因此第一版做如下取舍：

- `LSTM` 继续复用已经验证过的纯 Torch 重建逻辑；
- `Transformer` 和 `TCN` 不手写结构，而是在 `qlib` 环境中使用原生 Qlib 模型类 + `state_dict.pt` 重建；
- `trained_model.pkl` 只保留作人工排障、配置补全与未来导出脚本的参考材料。

这个取舍比“直接依赖 pickle 恢复旧对象”更稳，也更适合后续多模型扩展。

## 拟采用的整体架构

整体架构分成两层：共享攻击骨架，以及模型适配器层。

### 共享攻击骨架

共享层只负责模型无关的部分：

1. 从原始 `OHLCV` 计算 20 维旧版特征；
2. 施加 Torch 版 `RobustZScoreNorm`；
3. 施加 `Fillna`；
4. 运行 clean gate；
5. 计算输入梯度；
6. 运行 `FGSM` / `PGD`；
7. 生成样本级 CSV / JSON / Markdown 报告。

该层的核心约束是：不应再出现 `LSTM` 特化命名，不应直接依赖具体模型类。

### 模型适配器层

模型适配器层只负责把“归一化后的 20 维特征序列”映射到最终分数。

统一接口可以收敛为：

```python
class FeatureModelAdapter(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ...
```

攻击骨架通过这个接口调用模型，不再区分底层是 `LSTM`、`Transformer` 还是 `TCN`。

## Pipeline 设计

统一后的攻击图为：

```text
raw ohlcv
  -> feature bridge
  -> RobustZScoreNorm
  -> Fillna
  -> model adapter
  -> score
  -> MSE loss
```

建议将整条链路封装为一个统一 pipeline：

```python
class RawFeatureAttackPipeline(nn.Module):
    def forward_features(self, raw_ohlcv) -> torch.Tensor: ...
    def forward_from_features(self, features) -> torch.Tensor: ...
    def forward(self, raw_ohlcv) -> torch.Tensor: ...
```

其中：

- `forward_features` 用于 clean gate 中的中间特征对比；
- `forward_from_features` 只交给模型适配器；
- `forward` 供 `FGSM` / `PGD` 与 `MSE` 损失直接调用。

## 三个模型适配器

### `LSTMAdapter`

`LSTM` 继续使用当前已经验证过的纯 Torch 重建逻辑：

- 复用 `legacy_lstm_predictor.py`；
- 直接从 `state_dict.pt` 加载；
- 不重新回退到 Qlib wrapper 路线。

这样可以最大限度保持现有 LSTM 基线不变，降低回归风险。

### `TransformerAdapter`

`Transformer` 第一版不手写网络结构，而是：

1. 从 `model_config.json` 读取 `qlib_model_module`、`qlib_model_class` 和 `model_kwargs`；
2. 在 `qlib` 环境中实例化对应 wrapper；
3. 从 wrapper 中拿到真正的 torch 子模块；
4. 加载 `state_dict.pt`；
5. 暴露统一的 `forward(features)` 接口。

### `TCNAdapter`

`TCN` 与 `Transformer` 路线一致：

1. 在 `qlib` 环境中根据 `model_config.json` 重建；
2. 加载 `state_dict.pt`；
3. 通过统一接口输出分数。

## 脚本边界

### 资产导出脚本

当前的 `scripts/export_lstm_attack_assets.py` 已经基本承担了“模型无关的样本匹配与资产导出”职责，但命名仍然带有 LSTM 偏见。

第一版建议：

- 新增统一入口脚本 `scripts/export_whitebox_attack_assets.py`；
- 其内部复用现有导出逻辑；
- 保留 `scripts/export_lstm_attack_assets.py` 作为兼容壳层或历史别名。

导出脚本职责仅限于：

- 读取 `pred.pkl` 与 `label.pkl`；
- 生成 `matched_reference`；
- 导出匹配的 `OHLCV` 窗口；
- 导出参考特征窗口；
- 导出 `normalization_stats.json`。

该脚本不关心具体模型结构。

### 攻击执行脚本

新增统一入口脚本 `scripts/run_whitebox_attack.py`，推荐主要参数如下：

- `--model-name {lstm,transformer,tcn}`
- `--model-root origin_model_pred`
- `--asset-dir`
- `--out-dir`
- `--device`
- clean gate 阈值参数
- `FGSM` / `PGD` 超参数

入口逻辑固定为：

1. 根据 `--model-name` 读取对应目录；
2. 解析 `model_config.json`；
3. 构造 model adapter；
4. 组装统一 pipeline；
5. 执行 clean gate；
6. 执行 `FGSM` 和 `PGD`；
7. 产出统一格式结果。

旧的 `scripts/run_lstm_whitebox_attack.py` 保留，但改为轻量兼容壳层，内部转调统一入口。

## clean gate 设计

第一版 clean gate 分成两层。

### 共享硬门槛

所有模型共享下面几项硬条件：

- `feature_finite_rate == 1`
- `clean_grad_finite_rate == 1`
- `clean_grad_mean_abs > min_clean_grad_mean_abs`
- clean 前向与 loss 均可计算

### 模型相关软门槛

以下阈值允许按模型分别配置：

- `spearman_to_reference`
- `feature_mae_to_reference`
- `feature_rmse_to_reference`
- `feature_max_abs_to_reference`

原因是：

- 三模型虽然输入特征相同，但预测头结构不同；
- clean 对齐难度与分数分布可能不同；
- 第一版目标是先跑通并确认攻击方向，而不是强行把三模型压成完全相同阈值。

## 攻击与报告口径

攻击定义保持与现有 LSTM 版本一致：

- 目标函数：最大化 `MSE`
- 扰动预算：按当前值百分比约束
- 价格列和成交量列分组约束
- 非负投影
- 第一版不强制 K 线一致性约束

输出格式也与现有 LSTM 保持一致：

- `sample_metrics.csv`
- `attack_summary.json`
- `attack_report.md`

Markdown 标题与结果目录只需参数化模型名，不再写死为 `LSTM`。

## 第一版成功标准

对 `Transformer` 和 `TCN`，第一版只要求满足以下 4 条：

1. clean 可运行；
2. clean 与参考分数存在正排序相关，且中间特征误差可接受；
3. 输入梯度非空、非 NaN；
4. 小样本上 `FGSM` 和 `PGD` 能提升 `MSE`，且扰动预算不越界。

这意味着第一版重点是“迁移成功”，而不是“横向比较已经完备”。

## 测试与验证口径

验证分三层：

### 单元测试

- 模型 registry 能正确返回三类 adapter；
- 统一 pipeline 对 mock adapter 可正常反传梯度；
- `FGSM` / `PGD` 不越预算。

### 兼容性 smoke test

- 使用 tiny mock assets 跑统一入口；
- 验证输出文件存在、字段齐全。

### 脚本级真实 smoke

在 `qlib` 环境下分别对：

- `lstm`
- `transformer`
- `tcn`

运行小样本 clean、FGSM、PGD，确认：

- loader 正常；
- clean gate 字段合理；
- 攻击后 `MSE` 方向正确。

## 风险与缓解

### 风险 1：`Transformer/TCN` 的 wrapper 内部子模块名与预期不一致

缓解方式：

- 在 adapter 层集中处理 wrapper 到 torch 子模块的映射；
- 为每个模型写独立的最小前向单元测试；
- 把 “内部子模块解析失败” 设计成显式异常，而不是静默 fallback。

### 风险 2：`model_config.json` 缺字段或与 `state_dict.pt` 不匹配

缓解方式：

- 在加载阶段显式校验 `feature_spec.d_feat` 和 `step_len`；
- 对关键 `model_kwargs` 做缺失字段报错；
- 在第一版中把 `model_config.json` 设为必需资产。

### 风险 3：旧脚本与新入口并存时产生维护分叉

缓解方式：

- 旧 `run_lstm_whitebox_attack.py` 只做兼容壳层，不再承载新逻辑；
- 共享攻击逻辑只保留一份正式实现。

### 风险 4：用户本地资产目录继续混用大小写与不同目录名

缓解方式：

- 统一入口严格只接受规范化目录；
- 把目录契约写入文档与错误信息；
- 不在主逻辑里再兼容历史命名。

## 决策记录

本设计在版本 1 中固定以下决策：

- 第一版只做样本级攻击迁移，不做组合级攻击回测；
- `LSTM`、`Transformer`、`TCN` 共享同一套攻击算法与预处理骨架；
- 只把“模型预测头”抽象成 adapter；
- `Transformer` 和 `TCN` 第一版依赖 `qlib` 原生模型类 + `state_dict` 重建；
- `trained_model.pkl` 不作为正式主加载源；
- `model_config.json` 为强制资产；
- clean gate 采用“共享硬门槛 + 模型相关软门槛”；
- 旧 LSTM 专用脚本保留为兼容壳层，而不是继续扩展。
