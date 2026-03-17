# Weekly Report Writer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个全局可用的 `weekly-report-writer` skill，支持基于当前对话、聊天记录或实验目录生成中文周报，并默认写入 `/mnt/d/毕业设计/周报/`。

**Architecture:** 使用 `SKILL.md` 定义工作流与写作约束，使用两个小脚本分别处理“下一个周报文件名推断”和“周报骨架初始化”，并通过参考模板统一周报结构。实现先在全局 skill 目录初始化，再补齐脚本与文档，最后通过命名逻辑和一次真实落盘进行验证。

**Tech Stack:** Markdown、Python、Codex skill folder、现有 skill 初始化与校验脚本

---

## Chunk 1: 文档与脚手架

### Task 1: 写 spec 和 plan

**Files:**
- Create: `docs/superpowers/specs/2026-03-15-weekly-report-writer-design.md`
- Create: `docs/superpowers/plans/2026-03-15-weekly-report-writer.md`

- [ ] **Step 1: 写设计文档**
- [ ] **Step 2: 写实现计划**

### Task 2: 初始化全局 skill 目录

**Files:**
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/`
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/SKILL.md`
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/agents/openai.yaml`
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/scripts/`
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/references/`

- [ ] **Step 1: 用 `init_skill.py` 初始化 `weekly-report-writer` skill**
- [ ] **Step 2: 确认模板目录与 `agents/openai.yaml` 已生成**

## Chunk 2: 先测后写辅助脚本

### Task 3: 为命名逻辑写最小失败验证

**Files:**
- Create: `/tmp/weekly_report_writer_name_test.py`

- [ ] **Step 1: 写一个最小测试脚本，模拟从 `周报01-1-18.md`、`周报02-01-25.pdf`、`周报03-01-31.pdf` 推断下一个文件名**
- [ ] **Step 2: 运行测试，确认在脚本未实现前失败**

### Task 4: 实现 `next_report_name.py`

**Files:**
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/scripts/next_report_name.py`

- [ ] **Step 1: 写最小实现，支持扫描目录、解析已有周报编号、生成默认新文件名**
- [ ] **Step 2: 重新运行命名测试，确认通过**

### Task 5: 实现 `init_weekly_report.py`

**Files:**
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/scripts/init_weekly_report.py`
- Create: `/home/chainsmoker/.codex/skills/weekly-report-writer/references/template.md`

- [ ] **Step 1: 写最小实现，支持根据目录和日期创建新周报文件并写入模板**
- [ ] **Step 2: 用临时目录或真实周报目录执行一次，确认文件生成成功**

## Chunk 3: Skill 文档与验证

### Task 6: 编写 `SKILL.md`

**Files:**
- Modify: `/home/chainsmoker/.codex/skills/weekly-report-writer/SKILL.md`

- [ ] **Step 1: 写 frontmatter，明确触发条件**
- [ ] **Step 2: 写正文，覆盖三种输入模式、事实抽取规则、写作模板和落盘流程**
- [ ] **Step 3: 写明何时需要提升权限写入 `/mnt/d/毕业设计/周报/`**

### Task 7: 校验并做最小真实验证

**Files:**
- Verify: `/home/chainsmoker/.codex/skills/weekly-report-writer/`
- Verify: `/mnt/d/毕业设计/周报/周报*.md`

- [ ] **Step 1: 运行 `quick_validate.py` 校验 skill 目录**
- [ ] **Step 2: 运行 `next_report_name.py` 对真实周报目录推断文件名**
- [ ] **Step 3: 运行 `init_weekly_report.py` 在真实目录创建一份新周报 Markdown**
- [ ] **Step 4: 将当前对话的本周内容整理进该文件，检查结构完整**
