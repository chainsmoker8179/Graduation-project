# Graduation-project

## 项目简介

本项目围绕“面向量化金融时序模型的可导特征构建与白盒对抗攻击”展开，核心目标是把传统依赖 Qlib/Alpha158 的因子处理链路改造成可回传梯度的 PyTorch 图，并在此基础上对已训练的量化预测模型实施样本级与组合级白盒攻击。

当前仓库已经形成三条主线：

1. 可导 Alpha158 与 BPDA 梯度验证
2. legacy LSTM 的样本级与组合级白盒攻击
3. Transformer / TCN 的样本级 white-box attack 迁移与大样本正式实验

## 当前进展

截至目前，仓库中已经完成以下关键工作：

- 对 `max/min`、比较、分位数、`rank`、`idxmax/idxmin` 等离散算子完成 BPDA 算子级验证
- 在真实 Qlib 数据上完成 `OHLCV -> 因子 -> loss` 的端到端梯度回传验证
- 搭建 legacy LSTM 原始 `OHLCV` 白盒攻击链路，并完成从 smoke、扩样、clean 偏弱定位到正式基线的完整闭环
- 完成 LSTM 的 partial attack backtest、多随机种子稳定性验证、`1% / 5% / 10%` 比例趋势验证，以及显著性与排序机制补强
- 将样本级 white-box attack 迁移到 `Transformer` 和 `TCN`，并完成 `4096` 样本正式实验

详细实验阶段、文档对应关系与推荐阅读顺序，请直接查看：

- [项目实验总览与文档索引](reports/项目实验总览与文档索引.md)

## 仓库结构

```text
.
├── alpha158_*.py                  # Alpha158 可导算子与模块实现
├── legacy_lstm_*.py               # legacy LSTM 攻击链路相关模块
├── whitebox_attack_*.py           # 通用 white-box attack 核心与多模型适配
├── scripts/                       # 核心实验脚本、导出脚本与分析脚本
├── tests/                         # 单元测试与 smoke 测试
├── reports/                       # 中文实验报告、结果摘要与论文段落
├── docs/superpowers/              # 设计文档与实施计划
├── origin_model_pred/             # 本地模型配置、权重与预测文件入口
└── artifacts/                     # 大体积中间资产（默认不纳入 git）
```

重点目录说明：

- `scripts/`
  - 主要实验入口与结果构建脚本
- `tests/`
  - 回归测试、导出测试、攻击 smoke 测试
- `reports/`
  - 当前最重要的实验结论与中文说明文档
- `docs/superpowers/specs/` 与 `docs/superpowers/plans/`
  - 设计决策与实现计划留档
- `origin_model_pred/`
  - 模型配置文件受 git 管理；本地权重与 `pred.pkl`/`label.pkl` 由使用者自行准备

## 环境与运行

当前实验主要在本地 `qlib` conda 环境中执行。仓库中的测试和脚本默认按如下 Python 解释器调用：

```bash
/home/chainsmoker/miniconda3/envs/qlib/bin/python
```

建议使用方式：

1. 准备好本地 Qlib 环境
2. 确保 `origin_model_pred/` 下存在对应模型的配置文件、权重和预测结果
3. 在仓库根目录运行相关脚本或测试

## 常用脚本入口

以下脚本是当前最常用的实验入口：

- `scripts/validate_bpda_gradients.py`
  - BPDA 算子级梯度验证
- `scripts/validate_e2e_gradients.py`
  - 端到端梯度验证
- `scripts/run_lstm_whitebox_attack.py`
  - legacy LSTM 样本级白盒攻击
- `scripts/run_whitebox_attack.py`
  - Transformer / TCN 等模型的统一样本级攻击入口
- `scripts/run_model_rebuild_probe.py`
  - 多模型 clean rebuild probe
- `scripts/export_lstm_attack_assets.py`
  - LSTM 攻击资产导出
- `scripts/export_large_sample_attack_assets.py`
  - 大样本攻击资产快导出
- `scripts/run_partial_attack_backtest.py`
  - 单次 partial attack backtest
- `scripts/run_partial_attack_backtest_multiseed.py`
  - 多随机种子 partial backtest
- `scripts/run_lstm_attack_significance.py`
  - LSTM 攻击显著性统计

## 关键文档入口

如果你是第一次阅读本仓库，建议从下面这些文档开始：

1. [项目实验总览与文档索引](reports/项目实验总览与文档索引.md)
2. [端到端梯度验证总结](reports/e2e_grad_validation.md)
3. [LSTM 扩样攻击总报告](reports/lstm_whitebox_attack_expanded_report.md)
4. [LSTM 单模型证据补强报告](reports/lstm_single_model_evidence_report.md)
5. [Transformer / TCN 4096 样本白盒攻击实验报告](reports/transformer_tcn_4096白盒攻击实验报告.md)

## 注意事项

- 本仓库同时包含代码、实验报告和一部分结果快照，因此 `reports/` 中存在大量已纳入版本管理的结果文件。
- `artifacts/` 默认不纳入 git；它们主要是导出中间资产，体积较大。
- `origin_model_pred/` 中只有配置文件适合长期纳入版本管理，本地模型权重与预测文件需要按实际实验环境准备。
- 根目录下的 `fast-soft-sort-master/` 为第三方依赖代码，不属于本项目核心实验逻辑。

## GitHub

- 仓库地址：<https://github.com/chainsmoker8179/Graduation-project>
