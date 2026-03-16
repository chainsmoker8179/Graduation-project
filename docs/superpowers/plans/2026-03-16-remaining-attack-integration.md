# Remaining Attack Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 分三批将剩余攻击实验代码、测试和文档从 feature worktree 整理并合入 `master`。

**Architecture:** 采用“盘点 -> 最小提交集合 -> feature 验证 -> 独立 commit -> merge 到 master -> master 回归验证”的重复流程。每一批单独控制范围，避免把大体积实验缓存和可再生中间产物一并提交。

**Tech Stack:** git、pytest、Python、现有攻击/回测脚本、实验摘要 CSV/JSON/Markdown

---

## 文件结构

- Create: `docs/superpowers/specs/2026-03-16-remaining-attack-integration-design.md`
- Create: `docs/superpowers/plans/2026-03-16-remaining-attack-integration.md`
- Modify: `legacy_lstm_feature_bridge.py` / `legacy_lstm_preprocess.py` / `legacy_lstm_predictor.py` / `legacy_lstm_attack_core.py` / `scripts/run_lstm_whitebox_attack.py`（如第一批整理时需要）
- Modify: `scripts/export_lstm_attack_assets.py` / `partial_attack_backtest.py` / `scripts/run_partial_attack_backtest.py`（第二批）
- Modify: `reports/` 与 `docs/` 下筛选后的报告文件（第三批）
- Test: `tests/test_legacy_lstm_*.py`
- Test: `tests/test_lstm_whitebox_attack_smoke.py`
- Test: `tests/test_export_lstm_attack_assets.py`
- Test: `tests/test_partial_attack_backtest.py`

## Chunk 1: 第一批核心攻击链路

### Task 1: 盘点第一批最小提交集合

**Files:**
- Inspect: `legacy_lstm_feature_bridge.py`
- Inspect: `legacy_lstm_preprocess.py`
- Inspect: `legacy_lstm_predictor.py`
- Inspect: `legacy_lstm_attack_core.py`
- Inspect: `scripts/run_lstm_whitebox_attack.py`
- Inspect: `tests/test_legacy_lstm_feature_bridge.py`
- Inspect: `tests/test_legacy_lstm_preprocess.py`
- Inspect: `tests/test_legacy_lstm_predictor.py`
- Inspect: `tests/test_legacy_lstm_attack_core.py`
- Inspect: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 检查第一批代码和测试的导入依赖，列出额外必须纳入的文件**
- [ ] **Step 2: 确认是否需要补入配置文件、模板 CSV 或脚本级依赖**
- [ ] **Step 3: 形成第一批 git add 清单，避免误带 `artifacts/` 和大报告目录**

Run: `rg -n "from |import " legacy_lstm_*.py scripts/run_lstm_whitebox_attack.py tests/test_legacy_lstm_*.py tests/test_lstm_whitebox_attack_smoke.py`
Expected: 明确第一批依赖边界

### Task 2: 在 feature worktree 中验证第一批

**Files:**
- Modify: 第一批代码文件（仅在缺失依赖或测试失败时修复）
- Test: `tests/test_legacy_lstm_feature_bridge.py`
- Test: `tests/test_legacy_lstm_preprocess.py`
- Test: `tests/test_legacy_lstm_predictor.py`
- Test: `tests/test_legacy_lstm_attack_core.py`
- Test: `tests/test_lstm_whitebox_attack_smoke.py`

- [ ] **Step 1: 将第一批最小文件集合加入暂存区**
- [ ] **Step 2: 先运行第一批测试，观察缺口**
- [ ] **Step 3: 若失败，按最小修复原则补齐缺失文件或代码**
- [ ] **Step 4: 重新运行第一批测试直到通过**
- [ ] **Step 5: 提交第一批 commit**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/pytest tests/test_legacy_lstm_feature_bridge.py tests/test_legacy_lstm_preprocess.py tests/test_legacy_lstm_predictor.py tests/test_legacy_lstm_attack_core.py tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

### Task 3: 将第一批合并到 master 并复验

**Files:**
- Verify: `master`

- [ ] **Step 1: 确认 `master` 当前用户本地改动状态，必要时临时 stash**
- [ ] **Step 2: merge 第一批 commit 到 `master`**
- [ ] **Step 3: 在 `master` 重新运行第一批测试**
- [ ] **Step 4: 恢复用户原本的本地改动**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/pytest tests/test_legacy_lstm_feature_bridge.py tests/test_legacy_lstm_preprocess.py tests/test_legacy_lstm_predictor.py tests/test_legacy_lstm_attack_core.py tests/test_lstm_whitebox_attack_smoke.py -v`
Expected: PASS

## Chunk 2: 第二批局部攻击回测与导出链路

### Task 4: 盘点第二批代码边界

**Files:**
- Inspect: `scripts/export_lstm_attack_assets.py`
- Inspect: `partial_attack_backtest.py`
- Inspect: `scripts/run_partial_attack_backtest.py`
- Inspect: `tests/test_export_lstm_attack_assets.py`
- Inspect: `tests/test_partial_attack_backtest.py`
- Inspect: `scripts/export_qlib_recorder_artifacts.py`（若存在于 feature 分支待纳入）
- Inspect: `tests/test_export_qlib_recorder_artifacts.py`（若需要）

- [ ] **Step 1: 识别第二批测试所需的代码与数据依赖**
- [ ] **Step 2: 明确哪些结果文件必须纳入，哪些大文件必须排除**
- [ ] **Step 3: 形成第二批最小提交清单**

### Task 5: 在 feature worktree 中验证第二批

**Files:**
- Modify: 第二批相关代码文件（仅在修复依赖或测试失败时）
- Test: `tests/test_export_lstm_attack_assets.py`
- Test: `tests/test_partial_attack_backtest.py`
- Test: `tests/test_export_qlib_recorder_artifacts.py`（若纳入）

- [ ] **Step 1: 暂存第二批文件**
- [ ] **Step 2: 运行第二批测试并定位失败**
- [ ] **Step 3: 做最小修复**
- [ ] **Step 4: 重新运行第二批测试直到通过**
- [ ] **Step 5: 提交第二批 commit**

Run: `/home/chainsmoker/miniconda3/envs/qlib/bin/pytest tests/test_export_lstm_attack_assets.py tests/test_partial_attack_backtest.py tests/test_export_qlib_recorder_artifacts.py -v`
Expected: PASS 或明确说明某测试因文件不在本批而不纳入

### Task 6: 将第二批合并到 master 并复验

**Files:**
- Verify: `master`

- [ ] **Step 1: 暂存并保护 `master` 上用户本地改动**
- [ ] **Step 2: merge 第二批 commit 到 `master`**
- [ ] **Step 3: 在 `master` 重新运行第二批测试**
- [ ] **Step 4: 恢复用户原本本地改动**

## Chunk 3: 第三批实验报告与补充结果

### Task 7: 筛选可进仓库的报告产物

**Files:**
- Inspect: `reports/`
- Inspect: `docs/superpowers/plans/2026-03-11-attack-clean-gate-and-expanded-smoke.md`
- Inspect: `docs/superpowers/plans/2026-03-13-partial-attack-backtest-evaluation.md`
- Inspect: `docs/superpowers/plans/2026-03-13-partial-attack-ratio-sweep.md`
- Inspect: `docs/superpowers/specs/2026-03-13-partial-attack-backtest-evaluation-design.md`
- Inspect: `docs/superpowers/specs/2026-03-13-partial-attack-ratio-sweep-design.md`

- [ ] **Step 1: 按大小和可再生性筛出必须排除的文件**
- [ ] **Step 2: 为第三批生成“保留/排除”清单**
- [ ] **Step 3: 仅将摘要 CSV/JSON、README、中文报告、设计/计划文档加入暂存区**

### Task 8: 提交第三批并合并到 master

**Files:**
- Modify: 第三批筛选后的 `reports/` 与 `docs/`

- [ ] **Step 1: 在 feature worktree 检查第三批提交范围**
- [ ] **Step 2: 提交第三批 commit**
- [ ] **Step 3: merge 到 `master`**
- [ ] **Step 4: 抽查关键文档和摘要文件存在性**

Run: `find reports docs/superpowers -maxdepth 3 -type f | sort`
Expected: 第三批只包含摘要和文档，不含超大缓存
