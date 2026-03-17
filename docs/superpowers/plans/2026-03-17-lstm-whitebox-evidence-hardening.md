# LSTM 白盒攻击单模型证据补强 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于现有 legacy LSTM 攻击产物，补齐统计显著性、排序机制和论文级图表，形成主文聚焦 `5%` 多 seed、附录补充比例趋势的单模型证据链。

**Architecture:** 新增 `scripts/analysis/` 纯分析模块，读取现有 `reports/` 目录中的日度与 seed 级结果，输出结构化的显著性统计和排序机制诊断；在此基础上扩展 `scripts/plotting/` 的 loader 与 figure builder，最后由统一脚本生成中文实验报告和图表索引。

**Tech Stack:** Python, pandas, numpy, scipy.stats, matplotlib, pytest.

---

## Chunk 1: 数据契约与共享分析输入

### Task 1: 建立日度分析面板构造器

**Files:**
- Create: `scripts/analysis/__init__.py`
- Create: `scripts/analysis/lstm_attack_daily_panel.py`
- Test: `tests/test_lstm_attack_daily_panel.py`

- [ ] **Step 1: 写失败测试，固定输入输出契约**

```python
from pathlib import Path

from scripts.analysis.lstm_attack_daily_panel import load_multiseed_daily_panel


def test_load_multiseed_daily_panel_stacks_seed_daily_files(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    seed0 = report_root / "partial_attack_backtest_multiseed_ratio5_union" / "seed_0"
    seed1 = report_root / "partial_attack_backtest_multiseed_ratio5_union" / "seed_1"
    seed0.mkdir(parents=True)
    seed1.mkdir(parents=True)
    payload = (
        "datetime,partial_clean_excess_return_with_cost,partial_fgsm_excess_return_with_cost,"
        "partial_clean_rank_ic,partial_fgsm_rank_ic\n"
        "2025-01-02,0.01,-0.02,0.10,0.06\n"
    )
    (seed0 / "daily_comparison.csv").write_text(payload, encoding="utf-8")
    (seed1 / "daily_comparison.csv").write_text(payload, encoding="utf-8")

    df = load_multiseed_daily_panel(report_root)

    assert list(df["seed"]) == [0, 1]
    assert "fgsm_minus_partial_clean_excess_return_with_cost" in df.columns
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_lstm_attack_daily_panel.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function error.

- [ ] **Step 3: 实现最小日度面板加载逻辑**

```python
def load_multiseed_daily_panel(report_root: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted((report_root / "partial_attack_backtest_multiseed_ratio5_union").glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        df = pd.read_csv(seed_dir / "daily_comparison.csv")
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: 再跑测试确认通过**

Run: `pytest tests/test_lstm_attack_daily_panel.py -v`
Expected: PASS.

- [ ] **Step 5: 提交该任务**

```bash
git add scripts/analysis/__init__.py scripts/analysis/lstm_attack_daily_panel.py tests/test_lstm_attack_daily_panel.py
git commit -m "feat: add daily panel loader for lstm attack evidence"
```

## Chunk 2: 显著性检验与区间估计

### Task 2: 计算日度配对显著性统计

**Files:**
- Create: `scripts/analysis/lstm_attack_significance.py`
- Create: `scripts/run_lstm_attack_significance.py`
- Test: `tests/test_lstm_attack_significance.py`

- [ ] **Step 1: 写失败测试，锁定配对检验输出字段**

```python
import pandas as pd

from scripts.analysis.lstm_attack_significance import summarize_paired_significance


def test_summarize_paired_significance_reports_effect_and_pvalue() -> None:
    df = pd.DataFrame(
        {
            "seed": [0, 0, 0, 1, 1, 1],
            "datetime": ["2025-01-02", "2025-01-03", "2025-01-06"] * 2,
            "partial_clean_excess_return_with_cost": [0.02, 0.01, 0.03, 0.01, 0.00, 0.02],
            "partial_fgsm_excess_return_with_cost": [0.00, -0.01, 0.01, -0.01, -0.02, 0.00],
        }
    )

    out = summarize_paired_significance(
        df,
        baseline_col="partial_clean_excess_return_with_cost",
        attacked_col="partial_fgsm_excess_return_with_cost",
        metric_name="excess_return_with_cost",
    )

    assert out["metric"] == "excess_return_with_cost"
    assert out["comparison"] == "partial_fgsm_vs_partial_clean"
    assert out["delta_mean"] < 0
    assert 0.0 <= out["p_value"] <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_lstm_attack_significance.py -v`
Expected: FAIL with missing module/function error.

- [ ] **Step 3: 实现最小配对检验与 bootstrap 汇总**

```python
def summarize_paired_significance(df: pd.DataFrame, baseline_col: str, attacked_col: str, metric_name: str) -> dict[str, float | str]:
    deltas = df[attacked_col] - df[baseline_col]
    stat = wilcoxon(deltas)
    return {
        "metric": metric_name,
        "comparison": f"{attacked_col.split('_')[1]}_vs_partial_clean",
        "delta_mean": float(deltas.mean()),
        "delta_median": float(deltas.median()),
        "p_value": float(stat.pvalue),
    }
```

- [ ] **Step 4: 扩展为 CLI，输出结构化结果**

Run target script should write:
- `reports/lstm_single_model_evidence/significance_daily_metrics.csv`
- `reports/lstm_single_model_evidence/significance_block_bootstrap.json`

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_lstm_attack_significance.py -v`
Expected: PASS.

- [ ] **Step 6: 提交该任务**

```bash
git add scripts/analysis/lstm_attack_significance.py scripts/run_lstm_attack_significance.py tests/test_lstm_attack_significance.py
git commit -m "feat: add significance analysis for lstm attack outputs"
```

## Chunk 3: 排序机制诊断

### Task 3: 计算分数相关性、Top-K 重合率与排名位移

**Files:**
- Create: `scripts/analysis/lstm_attack_ranking_diagnostics.py`
- Create: `scripts/run_lstm_attack_ranking_diagnostics.py`
- Test: `tests/test_lstm_attack_ranking_diagnostics.py`

- [ ] **Step 1: 写失败测试，锁定排序诊断输出**

```python
import pandas as pd

from scripts.analysis.lstm_attack_ranking_diagnostics import compute_daily_topk_overlap


def test_compute_daily_topk_overlap_returns_fraction_per_day() -> None:
    scores = pd.DataFrame(
        {
            "datetime": ["2025-01-02"] * 4,
            "instrument": ["A", "B", "C", "D"],
            "baseline_score": [4.0, 3.0, 2.0, 1.0],
            "attacked_score": [4.0, 1.0, 3.0, 2.0],
        }
    )

    out = compute_daily_topk_overlap(scores, topk=2)

    assert out.loc[0, "topk_overlap"] == 0.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_lstm_attack_ranking_diagnostics.py -v`
Expected: FAIL with missing module/function error.

- [ ] **Step 3: 实现最小排序诊断函数**

```python
def compute_daily_topk_overlap(scores: pd.DataFrame, topk: int) -> pd.DataFrame:
    rows = []
    for dt, part in scores.groupby("datetime"):
        base_top = set(part.nlargest(topk, "baseline_score")["instrument"])
        adv_top = set(part.nlargest(topk, "attacked_score")["instrument"])
        rows.append({"datetime": dt, "topk_overlap": len(base_top & adv_top) / float(topk)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 扩展输出完整诊断结果**

Run target script should write:
- `reports/lstm_single_model_evidence/ranking_overlap_daily.csv`
- `reports/lstm_single_model_evidence/ranking_correlation_daily.csv`
- `reports/lstm_single_model_evidence/rank_shift_summary.csv`

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_lstm_attack_ranking_diagnostics.py -v`
Expected: PASS.

- [ ] **Step 6: 提交该任务**

```bash
git add scripts/analysis/lstm_attack_ranking_diagnostics.py scripts/run_lstm_attack_ranking_diagnostics.py tests/test_lstm_attack_ranking_diagnostics.py
git commit -m "feat: add ranking diagnostics for lstm attack analysis"
```

## Chunk 4: 图表接入现有 plotting 体系

### Task 4: 新增显著性与排序机制图

**Files:**
- Modify: `scripts/plotting/loaders.py`
- Modify: `scripts/plotting/build_all_figures.py`
- Create: `scripts/plotting/fig07_significance_summary.py`
- Create: `scripts/plotting/fig08_ranking_mechanism.py`
- Test: `tests/test_plotting_loaders.py`

- [ ] **Step 1: 写失败测试，验证 loader 能读取新分析产物**

```python
from pathlib import Path

from scripts.plotting.loaders import load_significance_summary_data


def test_load_significance_summary_data_reads_generated_csv(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    target = report_dir / "lstm_single_model_evidence"
    target.mkdir(parents=True)
    (target / "significance_daily_metrics.csv").write_text(
        "metric,comparison,delta_mean,p_value\nexcess_return,FGSM,-0.1,0.01\n",
        encoding="utf-8",
    )

    df = load_significance_summary_data(report_dir)

    assert list(df["metric"]) == ["excess_return"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_plotting_loaders.py -v`
Expected: FAIL with missing loader error.

- [ ] **Step 3: 实现 loader 与 figure builder**

```python
FIGURE_BUILDERS = {
    "fig01": build_fig01,
    "fig02": build_fig02,
    "fig03": build_fig03,
    "fig04": build_fig04,
    "fig05": build_fig05,
    "fig06": build_fig06,
    "fig07": build_fig07,
    "fig08": build_fig08,
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_plotting_loaders.py -v`
Expected: PASS.

- [ ] **Step 5: 手工构建双尺寸图表**

Run: `python scripts/plotting/build_all_figures.py --all`
Expected: `reports/figures/paper` and `reports/figures/slide` include `fig07_*` and `fig08_*` outputs.

- [ ] **Step 6: 提交该任务**

```bash
git add scripts/plotting/loaders.py scripts/plotting/build_all_figures.py scripts/plotting/fig07_significance_summary.py scripts/plotting/fig08_ranking_mechanism.py tests/test_plotting_loaders.py
git commit -m "feat: add paper figures for lstm attack evidence"
```

## Chunk 5: 中文实验报告与论文段落生成

### Task 5: 统一生成单模型证据报告

**Files:**
- Create: `scripts/build_lstm_single_model_evidence_report.py`
- Create: `reports/lstm_single_model_evidence_report.md`
- Test: `tests/test_build_lstm_single_model_evidence_report.py`

- [ ] **Step 1: 写失败测试，锁定报告必须包含的章节**

```python
from scripts.build_lstm_single_model_evidence_report import render_report


def test_render_report_contains_core_sections() -> None:
    text = render_report(
        significance_rows=[{"metric": "rank_ic", "p_value": 0.001}],
        ranking_rows=[{"metric": "topk_overlap", "delta_mean": -0.2}],
    )

    assert "统计显著性" in text
    assert "排序机制" in text
    assert "主文放置建议" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_build_lstm_single_model_evidence_report.py -v`
Expected: FAIL with missing module/function error.

- [ ] **Step 3: 实现最小报告渲染与文件写出**

```python
def render_report(significance_rows: list[dict], ranking_rows: list[dict]) -> str:
    return "\n".join([
        "# LSTM 单模型攻击证据补强报告",
        "## 统计显著性",
        "## 排序机制",
        "## 主文放置建议",
    ])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_build_lstm_single_model_evidence_report.py -v`
Expected: PASS.

- [ ] **Step 5: 生成正式报告**

Run: `python scripts/build_lstm_single_model_evidence_report.py`
Expected: `reports/lstm_single_model_evidence_report.md` regenerated from structured outputs.

- [ ] **Step 6: 提交该任务**

```bash
git add scripts/build_lstm_single_model_evidence_report.py reports/lstm_single_model_evidence_report.md tests/test_build_lstm_single_model_evidence_report.py
git commit -m "docs: add lstm single-model evidence report"
```

## Chunk 6: 端到端校验

### Task 6: 跑新增 tests 与指定脚本验证结果

**Files:**
- Verify: `tests/test_lstm_attack_daily_panel.py`
- Verify: `tests/test_lstm_attack_significance.py`
- Verify: `tests/test_lstm_attack_ranking_diagnostics.py`
- Verify: `tests/test_plotting_loaders.py`
- Verify: `tests/test_build_lstm_single_model_evidence_report.py`

- [ ] **Step 1: 跑新增 tests**

Run: `pytest tests/test_lstm_attack_daily_panel.py tests/test_lstm_attack_significance.py tests/test_lstm_attack_ranking_diagnostics.py tests/test_plotting_loaders.py tests/test_build_lstm_single_model_evidence_report.py -v`
Expected: PASS.

- [ ] **Step 2: 跑分析脚本**

Run: `python scripts/run_lstm_attack_significance.py && python scripts/run_lstm_attack_ranking_diagnostics.py && python scripts/build_lstm_single_model_evidence_report.py`
Expected: fresh CSV/JSON/Markdown outputs under `reports/lstm_single_model_evidence/` and `reports/lstm_single_model_evidence_report.md`.

- [ ] **Step 3: 跑图表构建**

Run: `python scripts/plotting/build_all_figures.py --all`
Expected: new `fig07` and `fig08` appear under both paper and slide output roots.

- [ ] **Step 4: 记录验证结论并提交**

```bash
git add reports/lstm_single_model_evidence reports/lstm_single_model_evidence_report.md reports/figures/paper reports/figures/slide
git commit -m "analysis: harden lstm single-model attack evidence"
```
