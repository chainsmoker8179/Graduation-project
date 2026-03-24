# LSTM 受约束白盒攻击 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 legacy LSTM 原始 `OHLCV` 白盒攻击链路上，引入金融物理约束和相对 clean 的统计约束，并支持 `none / physical / physical_stat` 三种可对照攻击模式。

**Architecture:** 保留当前无约束攻击器作为 `none` 基线，在 `legacy_lstm_attack_core.py` 中增量加入金融物理投影、统计 penalty 和受约束目标函数；`scripts/run_lstm_whitebox_attack.py` 只做模式路由、参数解析和结果落盘。实现顺序严格按照 `physical -> physical+ret -> full physical_stat` 推进，避免一次性加满约束导致难以定位攻击失效原因。

**Tech Stack:** Python、PyTorch、现有 `legacy_lstm_attack_core.py`、`scripts/run_lstm_whitebox_attack.py`、pytest、pandas、JSON 报告输出。

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-18-lstm-constrained-whitebox-attack-design.md`
  - 已批准设计文档
- Modify: `legacy_lstm_attack_core.py`
  - 新增金融物理投影、统计 penalty、受约束目标与新攻击器
- Modify: `scripts/run_lstm_whitebox_attack.py`
  - 新增 `constraint_mode`、统计约束参数、扩展 summary/report
- Create: `tests/test_lstm_attack_constraints.py`
  - 金融物理投影和统计 penalty 单元测试
- Modify: `tests/test_lstm_whitebox_attack_smoke.py`
  - 覆盖受约束攻击模式的 smoke 验证
- Optionally Create: `reports/lstm_whitebox_attack_constraints_smoke/`
  - 第一轮三模式小样本 smoke 结果目录

## Chunk 1: 金融物理硬约束

### Task 1: 为金融物理投影写失败测试

**Files:**
- Create: `tests/test_lstm_attack_constraints.py`
- Modify: `legacy_lstm_attack_core.py`

- [ ] **Step 1: 写 `K` 线一致性投影失败测试**

```python
import torch

from legacy_lstm_attack_core import project_financial_feasible_box


def test_project_financial_feasible_box_enforces_kline_constraints():
    x_clean = torch.tensor([[[10.0, 11.0, 9.0, 10.5, 1000.0]]], dtype=torch.float32)
    x_adv = torch.tensor([[[10.0, 9.0, 12.0, 10.5, -5.0]]], dtype=torch.float32)

    projected = project_financial_feasible_box(
        x_adv,
        x_clean,
        price_epsilon=0.05,
        volume_epsilon=0.1,
        price_floor=1e-6,
        volume_floor=0.0,
    )

    open_, high_, low_, close_, volume_ = projected[0, 0]
    assert high_ >= max(open_, close_)
    assert low_ <= min(open_, close_)
    assert low_ <= high_
    assert volume_ >= 0
```

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_attack_constraints.py::test_project_financial_feasible_box_enforces_kline_constraints -v`
Expected: FAIL，提示 `project_financial_feasible_box` 不存在

- [ ] **Step 3: 在 `legacy_lstm_attack_core.py` 中实现最小金融物理投影**

要求：
- 先复用现有 `project_relative_box`
- 再做价格和成交量下界裁剪
- 最后修正 `high` / `low`

- [ ] **Step 4: 重新运行该测试，确认通过**

Run: `pytest tests/test_lstm_attack_constraints.py::test_project_financial_feasible_box_enforces_kline_constraints -v`
Expected: PASS

- [ ] **Step 5: 再补一个预算不越界测试**

```python
def test_project_financial_feasible_box_preserves_relative_budget():
    ...
```

- [ ] **Step 6: 运行 constraints 测试子集**

Run: `pytest tests/test_lstm_attack_constraints.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_lstm_attack_constraints.py legacy_lstm_attack_core.py
git commit -m "feat: add financial feasibility projection for lstm attacks"
```

### Task 2: 接入 `physical` 模式攻击器

**Files:**
- Modify: `legacy_lstm_attack_core.py`
- Modify: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 写 `physical` 模式 FGSM 失败测试**

```python
def test_physical_fgsm_preserves_kline_constraints():
    ...
```

测试要求：
- 运行一次 `physical` 模式攻击
- 检查输出满足预算约束和 K 线合法性

- [ ] **Step 2: 运行该测试，确认在实现前失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py::test_physical_fgsm_preserves_kline_constraints -v`
Expected: FAIL，提示 `physical` 模式接口不存在

- [ ] **Step 3: 实现 `physical` 模式下的 FGSM / PGD**

要求：
- 保持目标函数仍为 `MSE`
- 每步更新后改用 `project_financial_feasible_box`

- [ ] **Step 4: 运行相关 smoke 测试**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS，且旧 `none` 口径测试不回归

- [ ] **Step 5: Commit**

```bash
git add legacy_lstm_attack_core.py tests/test_lstm_whitebox_attack_smoke.py
git commit -m "feat: support physical-only constrained lstm attacks"
```

## Chunk 2: 统计惩罚项

### Task 3: 为收益率 penalty 写失败测试并实现

**Files:**
- Modify: `tests/test_lstm_attack_constraints.py`
- Modify: `legacy_lstm_attack_core.py`

- [ ] **Step 1: 写 clean 输入 penalty 为零的失败测试**

```python
def test_return_penalty_is_zero_on_clean_input():
    x = torch.tensor(...)
    penalty = compute_return_penalty(x, x, tau_ret=0.005)
    assert torch.allclose(penalty, torch.tensor(0.0))
```

- [ ] **Step 2: 写阈值内不处罚的测试**

```python
def test_return_penalty_stays_zero_within_tolerance():
    ...
```

- [ ] **Step 3: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_attack_constraints.py -k return_penalty -v`
Expected: FAIL

- [ ] **Step 4: 实现 `compute_return_penalty`**

要求：
- 基于 `close` 的对数收益率
- 使用 hinge-squared 容忍带形式

- [ ] **Step 5: 重新运行收益率 penalty 测试**

Run: `pytest tests/test_lstm_attack_constraints.py -k return_penalty -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_lstm_attack_constraints.py legacy_lstm_attack_core.py
git commit -m "feat: add return-based statistical penalty for lstm attacks"
```

### Task 4: 为 K 线形态 penalty 写失败测试并实现

**Files:**
- Modify: `tests/test_lstm_attack_constraints.py`
- Modify: `legacy_lstm_attack_core.py`

- [ ] **Step 1: 写 `body` / `range` penalty 为零测试**

```python
def test_candle_penalty_is_zero_on_clean_input():
    ...
```

- [ ] **Step 2: 写阈值内不处罚测试**

```python
def test_candle_penalty_stays_zero_within_tolerance():
    ...
```

- [ ] **Step 3: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_attack_constraints.py -k candle_penalty -v`
Expected: FAIL

- [ ] **Step 4: 实现 `compute_candle_penalty`**

要求：
- 同时返回或内部计算 `body` 与 `range`
- 使用 `tau_body=0.005`、`tau_range=0.01` 作为默认初值

- [ ] **Step 5: 重新运行 K 线形态 penalty 测试**

Run: `pytest tests/test_lstm_attack_constraints.py -k candle_penalty -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_lstm_attack_constraints.py legacy_lstm_attack_core.py
git commit -m "feat: add candle-shape penalty for lstm attacks"
```

### Task 5: 为成交量动态 penalty 写失败测试并实现

**Files:**
- Modify: `tests/test_lstm_attack_constraints.py`
- Modify: `legacy_lstm_attack_core.py`

- [ ] **Step 1: 写 clean 输入成交量 penalty 为零测试**

```python
def test_volume_penalty_is_zero_on_clean_input():
    ...
```

- [ ] **Step 2: 写阈值内不处罚测试**

```python
def test_volume_penalty_stays_zero_within_tolerance():
    ...
```

- [ ] **Step 3: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_attack_constraints.py -k volume_penalty -v`
Expected: FAIL

- [ ] **Step 4: 实现 `compute_volume_penalty`**

要求：
- 使用 `log(volume + 1)` 的一阶差分
- 使用 hinge-squared 容忍带

- [ ] **Step 5: 重新运行成交量 penalty 测试**

Run: `pytest tests/test_lstm_attack_constraints.py -k volume_penalty -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_lstm_attack_constraints.py legacy_lstm_attack_core.py
git commit -m "feat: add volume-dynamics penalty for lstm attacks"
```

## Chunk 3: 受约束目标函数与脚本入口

### Task 6: 实现 `physical_stat` 目标函数

**Files:**
- Modify: `legacy_lstm_attack_core.py`
- Modify: `tests/test_lstm_attack_constraints.py`

- [ ] **Step 1: 写受约束目标函数失败测试**

```python
def test_constrained_objective_matches_mse_on_clean_input():
    ...
```

要求：
- 当 `adv == clean` 时
- penalty 为零
- `objective == mse`

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_attack_constraints.py -k constrained_objective -v`
Expected: FAIL

- [ ] **Step 3: 实现 `compute_constrained_attack_objective`**

要求：
- 返回结构化结果，至少包含：
  - `mse_loss`
  - `ret_penalty`
  - `candle_penalty`
  - `vol_penalty`
  - `objective`

- [ ] **Step 4: 重新运行目标函数测试**

Run: `pytest tests/test_lstm_attack_constraints.py -k constrained_objective -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_lstm_attack_constraints.py legacy_lstm_attack_core.py
git commit -m "feat: add constrained objective for lstm attacks"
```

### Task 7: 扩展 FGSM / PGD 到 `physical_stat`

**Files:**
- Modify: `legacy_lstm_attack_core.py`
- Modify: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 写 `physical_stat` 模式失败测试**

```python
def test_physical_stat_attack_preserves_constraints_and_changes_loss():
    ...
```

要求：
- 攻击输出 shape 正确
- 物理约束满足
- 输入梯度非零

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -k physical_stat -v`
Expected: FAIL

- [ ] **Step 3: 实现受约束版 FGSM / PGD**

要求：
- 在 `physical_stat` 下优化约束目标 `J`
- 每步仍使用金融物理投影

- [ ] **Step 4: 重新运行 smoke 测试**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add legacy_lstm_attack_core.py tests/test_lstm_whitebox_attack_smoke.py
git commit -m "feat: add physical-stat constrained fgsm and pgd"
```

### Task 8: 扩展运行脚本与输出指标

**Files:**
- Modify: `scripts/run_lstm_whitebox_attack.py`
- Modify: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 写 CLI 失败测试**

```python
def test_parse_args_accepts_constraint_mode_and_penalty_hparams():
    ...
```

- [ ] **Step 2: 运行测试，确认在实现前失败**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -k constraint_mode -v`
Expected: FAIL

- [ ] **Step 3: 在运行脚本中新增参数**

新增：
- `--constraint-mode`
- `--tau-ret`
- `--tau-body`
- `--tau-range`
- `--tau-vol`
- `--lambda-ret`
- `--lambda-candle`
- `--lambda-vol`

- [ ] **Step 4: 扩展 summary 和 report 输出**

要求新增：
- 各类 penalty 指标
- 统计 shift 指标
- `physical_constraints_satisfied_*`
- `strict_attack_success_*`

- [ ] **Step 5: 重新运行脚本级 smoke 测试**

Run: `pytest tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_lstm_whitebox_attack.py tests/test_lstm_whitebox_attack_smoke.py
git commit -m "feat: expose constrained attack modes in lstm runner"
```

## Chunk 4: 三模式 smoke 实验

### Task 9: 运行 `none / physical / physical_stat` 三模式小样本对比

**Files:**
- Output: `reports/lstm_whitebox_attack_constraints_smoke/`

- [ ] **Step 1: 跑 `none` 模式基线**

Run: `python scripts/run_lstm_whitebox_attack.py --asset-dir artifacts/lstm_attack_expanded_v6 --state-dict-path origin_model_pred/lstm/model/state_dict.pt --config-path origin_model_pred/lstm/model/model_config.json --constraint-mode none --max-samples 16 --out-dir reports/lstm_whitebox_attack_constraints_smoke/none`
Expected: 正常输出 `sample_metrics.csv`、`attack_summary.json`、`attack_report.md`

- [ ] **Step 2: 跑 `physical` 模式**

Run: `python scripts/run_lstm_whitebox_attack.py --asset-dir artifacts/lstm_attack_expanded_v6 --state-dict-path origin_model_pred/lstm/model/state_dict.pt --config-path origin_model_pred/lstm/model/model_config.json --constraint-mode physical --max-samples 16 --out-dir reports/lstm_whitebox_attack_constraints_smoke/physical`
Expected: 攻击仍可运行，且 `physical_constraints_satisfied_*` 为真

- [ ] **Step 3: 跑 `physical_stat` 模式**

Run: `python scripts/run_lstm_whitebox_attack.py --asset-dir artifacts/lstm_attack_expanded_v6 --state-dict-path origin_model_pred/lstm/model/state_dict.pt --config-path origin_model_pred/lstm/model/model_config.json --constraint-mode physical_stat --max-samples 16 --out-dir reports/lstm_whitebox_attack_constraints_smoke/physical_stat`
Expected: 攻击仍可运行，统计 penalty 与 shift 指标可读

- [ ] **Step 4: 对比三模式结果**

检查重点：
- `clean_loss`、`fgsm_loss`、`pgd_loss`
- 物理合法性指标
- 各 penalty 大小
- 攻击强度是否按 `none -> physical -> physical_stat` 逐步下降

- [ ] **Step 5: 视结果决定是否需要微调 `lambda`**

仅当 `physical_stat` 几乎完全失效时，再优先下调：
1. `lambda_vol`
2. `lambda_candle`
3. `lambda_ret`

## Verification

- [ ] **Step 1: 跑新增约束测试**

Run: `pytest tests/test_lstm_attack_constraints.py -v`
Expected: PASS

- [ ] **Step 2: 跑旧 LSTM 回归测试**

Run: `pytest tests/test_legacy_lstm_attack_core.py tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS，旧 `none` 逻辑不回归

- [ ] **Step 3: 跑脚本级 smoke**

Run: `python scripts/run_lstm_whitebox_attack.py --asset-dir artifacts/lstm_attack_expanded_v6 --state-dict-path origin_model_pred/lstm/model/state_dict.pt --config-path origin_model_pred/lstm/model/model_config.json --constraint-mode physical_stat --max-samples 8 --out-dir reports/lstm_whitebox_attack_constraints_verify`
Expected: 输出完整，且物理合法性和 penalty 字段合理

Plan complete and saved to `docs/superpowers/plans/2026-03-18-lstm-constrained-whitebox-attack.md`. Ready to execute?
