# 剩余攻击实验内容分批合并设计

## 1. 目标

将 `feat/raw-ohlcv-lstm-attack` worktree 中尚未合入 `master` 的攻击实验相关代码、测试与文档，按风险和依赖关系拆分为三个可验证批次，逐批整理、提交并合入主分支，同时避免把超大中间产物和不必要的实验缓存带入仓库。

## 2. 分批策略

### 第一批：核心攻击链路

范围：
- `legacy_lstm_feature_bridge.py`
- `legacy_lstm_preprocess.py`
- `legacy_lstm_predictor.py`
- `legacy_lstm_attack_core.py`
- `scripts/run_lstm_whitebox_attack.py`
- 对应 `tests/test_legacy_lstm_*.py`
- `tests/test_lstm_whitebox_attack_smoke.py`

目的：
- 固化原始 OHLCV 到 legacy LSTM 的可微链路
- 固化 FGSM/PGD 的输入约束、clean gate 与 smoke 验证逻辑
- 为后续局部攻击回测提供稳定基础模块

约束：
- 只引入运行这些模块所必需的最小依赖文件
- 若引用尚未合入的脚本或文件，优先补齐缺失项，而不是在测试里绕过

### 第二批：局部攻击回测与导出链路

范围：
- `scripts/export_lstm_attack_assets.py`
- `partial_attack_backtest.py`
- `scripts/run_partial_attack_backtest.py`
- `tests/test_export_lstm_attack_assets.py`
- `tests/test_partial_attack_backtest.py`
- 若实际被测试引用，则补入 `scripts/export_qlib_recorder_artifacts.py` 与其测试

目的：
- 固化从预测结果/标签导出攻击资产的流程
- 固化局部股票攻击、分数替换和组合回测比较逻辑
- 将“攻击样本生成”与“组合层指标退化评估”打通

约束：
- 不把大体积 `artifacts/*.pt`、`reports/*/attack_mask.csv`、逐样本 `scores.pkl` 纳入本批代码提交
- 若需要实验结果，只保留测试或 plotting 真正需要的摘要文件

### 第三批：实验报告与补充结果

范围：
- 剩余 `docs/superpowers/plans/*.md` / `specs/*.md`
- 中文实验报告 `.md`
- `reports/` 下的摘要 JSON/CSV/README/报告文档
- 必要时补充少量中间结果，但排除大文件和可再生缓存

目的：
- 补全项目实验链路的文字记录和结论材料
- 建立“代码 -> 实验 -> 图表 -> 报告”的完整可追溯关系

约束：
- 明确排除：超大 `attack_mask.csv`、`*.pkl` 分数字典、`artifacts/` 中的大张量窗口文件
- 优先提交可读摘要与结论文档，而不是原始大体积中间产物

## 3. 合并顺序与原因

1. 先合并核心攻击链路：这是后续导出、回测和报告解释的基础。
2. 再合并局部攻击回测与导出链路：这部分依赖第一批提供的攻击内核与 legacy pipeline。
3. 最后整理报告与结果：等代码边界稳定后再纳入文档，能减少文档引用失效和重复改写。

## 4. 验证策略

每一批都遵守同一套门槛：
- 只跑与该批相关的最小测试集合
- 必须在 feature worktree 中先提交为独立 commit
- 再合并到 `master`
- 合并后在 `master` 重新跑同一组验证命令

## 5. 风险点

1. `tests/` 可能引用尚未合入的脚本，导致首轮最小提交不完整。
2. 旧 run 目录中的报告和结果文件体量差异很大，容易误带入大文件。
3. `master` 当前有用户本地改动，必须继续通过 worktree + 最小 merge 的方式推进。

## 6. 本轮执行决策

用户已确认按三批顺序逐一整理。因此本轮直接从“第一批：核心攻击链路”开始，先完成文件盘点、最小提交集合筛选、feature 分支验证、提交，再合并到 `master`。
