# Project Index Refresh Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `reports/项目实验总览与文档索引.md` 改造成面向维护者的导航型总览文档。

**Architecture:** 保留同一文件路径，重写其整体结构与内容组织；以当前主仓库中的实验文档、设计文档和归档索引为输入，输出一份明确区分正式基线、过程版本和辅助材料的导航页。

**Tech Stack:** Markdown

---

## Chunk 1: 内容重组

### Task 1: 盘点当前应纳入索引的文档

**Files:**
- Read: `reports/项目实验总览与文档索引.md`
- Read: `reports/*.md`
- Read: `docs/superpowers/specs/*.md`
- Read: `docs/superpowers/plans/*.md`

- [ ] 识别当前正式基线文档
- [ ] 识别过程性中间版本文档
- [ ] 识别辅助设计、可视化与归档文档

### Task 2: 重写目标文档

**Files:**
- Modify: `reports/项目实验总览与文档索引.md`

- [ ] 重写为导航型结构
- [ ] 补入 `LSTM 单模型证据补强` 阶段与相关文档
- [ ] 去除过时的 hidden worktree 默认口径
- [ ] 补入当前推荐阅读路径与正式/过程版本区分

### Task 3: 自检

**Files:**
- Verify: `reports/项目实验总览与文档索引.md`

- [ ] 检查结构完整性
- [ ] 检查关键路径与文件名是否准确
- [ ] 检查新阶段与当前基线是否被纳入
