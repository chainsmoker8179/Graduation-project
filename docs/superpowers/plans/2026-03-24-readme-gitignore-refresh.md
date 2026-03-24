# README 与 .gitignore 补充 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为仓库补充一个中文根目录 README，并以保守策略完善根 .gitignore。

**Architecture:** README 作为顶层入口文档，只描述项目概览、实验主线、常用脚本和文档索引，不重复报告正文内容；.gitignore 只新增本地缓存、日志和编辑器临时文件忽略规则，避免影响当前已跟踪实验结果。

**Tech Stack:** Markdown、gitignore 规则、现有 `reports/` 与 `scripts/` 目录结构。

---

## 文件结构

- Create: `README.md`
  - 根目录中文项目说明
- Modify: `.gitignore`
  - 补充保守型忽略规则
- Create: `docs/superpowers/specs/2026-03-24-readme-gitignore-refresh-design.md`
  - 本次设计说明

## Chunk 1: 补充根目录 README

### Task 1: 编写项目入口 README

**Files:**
- Create: `README.md`

- [ ] **Step 1: 基于当前仓库结构列出 README 所需章节**

章节至少包括：

- 项目简介
- 当前实验进展
- 仓库结构
- 环境与运行
- 常用脚本
- 文档索引
- 注意事项

- [ ] **Step 2: 编写中文 README 内容**

要求：

- 中文为主
- 不重复抄录各实验报告
- 明确指向 `reports/项目实验总览与文档索引.md`

- [ ] **Step 3: 回读 README 检查路径与脚本名**

重点检查：

- 目录名是否真实存在
- 文档路径是否准确
- 脚本名是否与当前仓库一致

## Chunk 2: 保守补充 .gitignore

### Task 2: 新增明确的本地临时文件忽略规则

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 保留现有实验结果忽略策略**

不得删除当前已有规则：

- `.worktrees/`
- `artifacts/`
- `reports/**/attack_mask.csv`
- `reports/**/*.pkl`

- [ ] **Step 2: 仅补充保守规则**

新增范围限定为：

- notebook 检查点
- 日志文件
- 临时文件
- 编辑器交换文件
- 系统杂项文件

- [ ] **Step 3: 回读 .gitignore，确认没有误伤 `reports/` 与 `origin_model_pred/`**

## Chunk 3: 最小验证

### Task 3: 做最小一致性检查

**Files:**
- Verify: `README.md`
- Verify: `.gitignore`

- [ ] **Step 1: 检查 README 已创建且非空**

Run:

```bash
test -s README.md
```

Expected: PASS

- [ ] **Step 2: 检查 .gitignore 包含新增规则**

Run:

```bash
rg -n "ipynb_checkpoints|\\.log|\\.tmp|\\.swp|\\.DS_Store" .gitignore
```

Expected: 至少匹配新增规则

- [ ] **Step 3: 检查 git 状态只包含预期文件**

Run:

```bash
git status --short README.md .gitignore docs/superpowers/specs/2026-03-24-readme-gitignore-refresh-design.md docs/superpowers/plans/2026-03-24-readme-gitignore-refresh.md
```

Expected: 仅出现本轮相关文件
