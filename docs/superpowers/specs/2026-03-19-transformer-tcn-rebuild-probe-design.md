# Transformer / TCN 重建探针设计

## 背景

当前仓库已经完成了 legacy LSTM 的原始 `OHLCV` 样本级白盒攻击链路，但“将攻击迁移到 Transformer 和 TCN”这条线还停留在设计阶段，尚未进入可运行状态。实际代码现状如下：

- 攻击核心仍是 LSTM 专用实现：`legacy_lstm_attack_core.py`
- 攻击入口仍是 LSTM 专用脚本：`scripts/run_lstm_whitebox_attack.py`
- 计划中的统一骨架与统一入口尚不存在：
  - `whitebox_attack_core.py`
  - `whitebox_attack_models.py`
  - `scripts/run_whitebox_attack.py`
- 本地模型资产也还没有整理成统一契约：
  - 目录大小写混用：`LSTM/`、`Transformer/`、`TCN/`
  - `pred/` 与 `prediction/` 混用
  - 没有任何 `model_config.json`

这说明当前真正的首要问题不是“怎么抽象统一攻击框架”，而是“Transformer 和 TCN 能否被本地稳定重建并前向”。

## 问题定义

如果直接跳到统一攻击框架，会把三类风险绑在一起：

1. 模型资产契约不统一；
2. Transformer / TCN 的 Qlib wrapper 与底层 torch 子模块结构不明确；
3. `state_dict` 是否能在本地 `qlib` 环境里正确加载尚未验证。

在上述问题都未消解前，直接实现共享攻击骨架会导致调试成本很高，而且很难判断失败到底来自“模型重建”还是“攻击逻辑抽象”。

## 目标

把“多模型样本级攻击迁移”切成更小的 phase 0，只验证以下命题：

在当前本地 `qlib` 环境中，`Transformer` 与 `TCN` 能否基于已有导出资产完成：

1. 读取结构化 `model_config.json`
2. 重建 Qlib wrapper 或其底层 torch 模块
3. 加载 `state_dict`
4. 对一小批已导出的标准化 20 维特征窗口做 clean forward
5. 输出有限、形状正确、与 `pred.pkl` 对应参考分数具有合理一致性的结果

如果这一阶段通过，下一阶段才进入统一攻击骨架与 `FGSM / PGD` 迁移。

## 范围内内容

- 明确 Transformer / TCN 当前可用资产路径
- 为 Transformer / TCN 补齐最小 `model_config.json`
- 新增一个模型重建与 clean forward 探针模块
- 新增一个 probe CLI 脚本
- 为 Transformer / TCN 生成各自的小样本 probe 资产
- 在 `qlib` conda 环境中跑通两模型 probe

## 范围外内容

- 不在本阶段抽取统一攻击骨架
- 不在本阶段修改 `legacy_lstm_attack_core.py`
- 不在本阶段实现统一的 `run_whitebox_attack.py`
- 不在本阶段接入 `FGSM` / `PGD`
- 不在本阶段接入 `partial attack backtest`
- 不在本阶段重构 `origin_model_pred/` 的目录大小写和命名

## 为什么这一阶段不先重构目录契约

旧计划中的目录统一是合理的，但当前更重要的是尽快验证“模型是否可重建”。如果一开始就重命名目录、移动大文件、统一大小写，会把“路径整理”和“模型重建”两个问题混在一起。

因此 phase 0 的策略是：

- 允许继续使用当前现有目录：
  - `origin_model_pred/Transformer/...`
  - `origin_model_pred/TCN/...`
- probe 脚本通过显式参数接收：
  - `--state-dict-path`
  - `--trained-model-pkl`
  - `--pred-pkl`
  - `--asset-dir`
  - `--config-path`

等 probe 通过后，再决定是否统一整理为新的 `origin_model_pred/transformer/` 与 `origin_model_pred/tcn/` 契约。

## 可复用资产与策略

当前仓库已经存在可直接复用的特征窗口资产：

- `artifacts/lstm_attack_expanded_v7_512/`
- `artifacts/lstm_attack_expanded_v8_2048/`
- `artifacts/lstm_attack_expanded_v9_4096/`

其中前两者同时包含：

- `matched_feature_windows.pt`
- `matched_ohlcv_windows.pt`
- `normalization_stats.json`
- `matched_reference.csv`

为了降低 probe 成本，第一版默认使用较小资产做验证。更稳妥的策略是：

1. 继续复用 `scripts/export_lstm_attack_assets.py`
2. 仅替换 `--pred-pkl` 与 `--label-pkl` 为 Transformer / TCN 对应文件
3. 为每个模型各自导出一份小样本 probe 资产

这样可以确保 probe 使用的是“与该模型自身 `pred.pkl` 对齐”的样本，而不是借用 LSTM 的参考分数。

## `model_config.json` 最小契约

phase 0 不要求一次性设计完整多模型配置体系，但至少要显式记录足以重建模型的字段：

```json
{
  "model_name": "transformer",
  "qlib_wrapper_module": "qlib.contrib.model.xxx",
  "qlib_wrapper_class": "TransformerModel",
  "torch_submodule_attr": "Transformer_model",
  "model_kwargs": {
    "d_feat": 20
  },
  "feature_spec": {
    "d_feat": 20,
    "step_len": 20
  }
}
```

设计要求：

- `qlib_wrapper_module` / `qlib_wrapper_class`：用于在 `qlib` 环境中重建 wrapper
- `torch_submodule_attr`：指出真正持有 `state_dict` 的 torch 子模块属性
- `model_kwargs`：必须足以实例化模型
- `feature_spec`：显式约束输入 `(batch, 20, 20)`

## Probe 核心接口

建议把 probe 核心封装成一个独立模块，而不是直接堆在脚本里。接口形态可收敛为：

```python
def load_feature_model_from_config(
    config_path: Path,
    state_dict_path: Path,
    device: torch.device,
) -> nn.Module:
    ...


def run_clean_forward_probe(
    model: nn.Module,
    feature_windows: torch.Tensor,
    reference_scores: torch.Tensor,
) -> dict[str, float]:
    ...
```

其中 probe 输出至少包含：

- `sample_count`
- `pred_finite_rate`
- `pred_mean`
- `pred_std`
- `mse_to_reference`
- `mae_to_reference`
- `spearman_to_reference`

## 成功标准

Transformer 或 TCN probe 通过，至少满足：

1. 模型可在本地 `qlib` 环境中实例化
2. `state_dict` 加载成功
3. 输入 `(batch, 20, 20)` 的特征窗口时，输出形状为 `(batch,)`
4. 输出全部有限
5. 与参考 `pred.pkl` 分数相比，`spearman_to_reference` 为正且明显高于随机

这里不要求 phase 0 就达到 LSTM clean gate 的标准，因为当前目标只是证明“本地重建路线可行”，不是直接进入正式攻击实验。

## 三种可选推进方式

### 方案 A：直接实现完整统一攻击框架

优点：
- 一次性完成终局结构

缺点：
- 把“模型重建不确定性”和“攻击抽象”绑在一起
- 调试成本最高

### 方案 B：先做模型重建 probe，再做统一攻击框架

优点：
- 先消除最大技术不确定性
- 后续攻击框架抽象更稳

缺点：
- 需要先接受一次“只验证前向、不做攻击”的过渡阶段

### 方案 C：先复制两份模型专用攻击脚本

优点：
- 短期可能更快看到结果

缺点：
- 会制造三套分叉逻辑
- 后续重构成本更高

## 选择

采用方案 B。

原因是当前最关键的硬阻塞不是攻击算法，而是模型重建。只要 Transformer / TCN 尚未在本地稳定重建，继续讨论统一 `FGSM / PGD` 框架没有意义。

## 输出物

phase 0 完成后，应至少产生：

- `origin_model_pred/Transformer/model/model_config.json`
- `origin_model_pred/TCN/model/model_config.json`
- `scripts/run_model_rebuild_probe.py`
- `whitebox_model_probe.py`
- `reports/transformer_model_probe/`
- `reports/tcn_model_probe/`

其中每个 probe 报告目录下至少包含：

- `probe_summary.json`
- `probe_predictions.csv`
- `README.md`

## 与后续阶段的关系

phase 0 的意义不是替代旧的多模型迁移设计，而是把它切成一个可以快速验证、快速止损的前置子项目。

只有当 Transformer / TCN 两个 probe 都通过后，才进入下一阶段：

1. 抽共享 `whitebox_attack_core.py`
2. 抽 `whitebox_attack_models.py`
3. 新增 `scripts/run_whitebox_attack.py`
4. 把 `FGSM / PGD` 正式迁移到多模型框架
