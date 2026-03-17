from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _pick_mean(rows: list[dict], metric: str) -> float | None:
    for row in rows:
        if row.get("metric") == metric:
            return float(row["mean"])
    return None


def _daily_sig_line(rows: list[dict], comparison: str, metric: str, metric_label: str) -> str:
    for row in rows:
        if row.get("comparison") == comparison and row.get("metric") == metric:
            return f"- `{comparison}` 在 `{metric_label}` 上的日度平均差值为 `{row['delta_mean']:.6f}`，`p={row['p_value']:.2e}`。"
    return f"- `{comparison}` 的 `{metric_label}` 显著性结果缺失。"


def _ranking_line(rows: list[dict], attacker: str, metric: str, label: str) -> str:
    for row in rows:
        if row.get("attacker") == attacker and row.get("metric") == metric:
            return f"- `{attacker}` 的 `{label}` 平均值为 `{row['value_mean']:.4f}`。"
    return f"- `{attacker}` 的 `{label}` 结果缺失。"


def _bootstrap_line(payload: dict, comparison: str, metric: str, label: str) -> str:
    summary = payload.get(comparison, {}).get(metric)
    if summary is None:
        return f"- `{comparison}` 的 `{label}` bootstrap 区间结果缺失。"
    return f"- `{comparison}` 在 `{label}` 上的 block bootstrap 区间为 `[{summary['ci95_lower']:.4f}, {summary['ci95_upper']:.4f}]`。"


def render_report(
    *,
    significance_rows: list[dict],
    ranking_rows: list[dict],
    main_rows: list[dict],
    ratio_rows: list[dict],
    bootstrap_payload: dict,
) -> str:
    fgsm_ann = _pick_mean(main_rows, "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost")
    pgd_ann = _pick_mean(main_rows, "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost")
    fgsm_rank = _pick_mean(main_rows, "partial_fgsm_minus_partial_clean__rank_ic_mean")
    pgd_rank = _pick_mean(main_rows, "partial_pgd_minus_partial_clean__rank_ic_mean")

    lines = [
        "# LSTM 单模型攻击证据补强报告",
        "",
        "## 1. 主结果回顾",
        f"- 在 `5%` 每日局部攻击、`seed=0..4` 的主口径下，FGSM 相对 `partial_clean` 的年化超额收益均值退化为 `{fgsm_ann:.4f}`，`RankIC` 均值退化为 `{fgsm_rank:.6f}`。"
        if fgsm_ann is not None and fgsm_rank is not None
        else "- FGSM 主结果缺失。",
        f"- 在相同口径下，PGD 相对 `partial_clean` 的年化超额收益均值退化为 `{pgd_ann:.4f}`，`RankIC` 均值退化为 `{pgd_rank:.6f}`。"
        if pgd_ann is not None and pgd_rank is not None
        else "- PGD 主结果缺失。",
        "- 这些结果与已有多随机种子回测结论一致，说明当前主文仍应以 `5%` 多 seed 作为核心证据。",
        "",
        "## 2. 统计显著性",
        _daily_sig_line(significance_rows, "partial_fgsm_vs_partial_clean", "excess_return_with_cost", "超额收益"),
        _daily_sig_line(significance_rows, "partial_pgd_vs_partial_clean", "excess_return_with_cost", "超额收益"),
        _daily_sig_line(significance_rows, "partial_fgsm_vs_partial_clean", "rank_ic", "RankIC"),
        _daily_sig_line(significance_rows, "partial_pgd_vs_partial_clean", "rank_ic", "RankIC"),
        _bootstrap_line(bootstrap_payload, "partial_fgsm_vs_partial_clean", "annualized_excess_return_with_cost", "年化超额收益"),
        _bootstrap_line(bootstrap_payload, "partial_pgd_vs_partial_clean", "annualized_excess_return_with_cost", "年化超额收益"),
        "",
        "## 3. 排序机制",
        _ranking_line(ranking_rows, "FGSM", "topk_overlap", "Top-K 重合率"),
        _ranking_line(ranking_rows, "PGD", "topk_overlap", "Top-K 重合率"),
        _ranking_line(ranking_rows, "FGSM", "spearman", "分数 Spearman"),
        _ranking_line(ranking_rows, "PGD", "spearman", "分数 Spearman"),
        _ranking_line(ranking_rows, "FGSM", "rank_shift_abs_mean", "平均绝对排名位移"),
        _ranking_line(ranking_rows, "PGD", "rank_shift_abs_mean", "平均绝对排名位移"),
        "- 这说明攻击并非仅在少量样本上抬高损失，而是会系统性扰动横截面排序，从而进一步传导到组合构建层面。",
        "",
        "## 4. 附录稳健性",
        "- `1% / 5% / 10%` 的比例扫描结果应继续保留在附录，用于说明攻击覆盖率提升时，收益、排序与风险退化同步增强。",
    ]
    if ratio_rows:
        lines.append(f"- 当前附录结果条目数为 `{len(ratio_rows)}`，足以支持比例趋势图和补充表格。")
    lines.extend(
        [
            "",
            "## 5. 主文放置建议",
            "- 主文：保留 `5%` 多 seed 的主结果表、显著性图表、排序机制图表。",
            "- 附录：保留 `1% / 5% / 10%` 比例扫描表、seed 级明细和补充区间估计。",
            "- 答辩时可优先展示累计超额收益曲线、显著性图和排序机制图三类结果，形成“现象-统计-机制”三层证据链。",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_ranking_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (attacker, metric), part in df.groupby(["attacker", "metric"], sort=True):
        rows.append({"attacker": attacker, "metric": metric, "value_mean": float(part["value"].mean())})
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report_root = repo_root / "reports"
    evidence_dir = report_root / "lstm_single_model_evidence"

    significance_rows = pd.read_csv(evidence_dir / "significance_daily_metrics.csv").to_dict(orient="records")
    ranking_parts = [
        pd.read_csv(evidence_dir / "ranking_overlap_daily.csv").assign(metric="topk_overlap", value=lambda df: df["topk_overlap"]),
        pd.read_csv(evidence_dir / "ranking_correlation_daily.csv").assign(metric="spearman", value=lambda df: df["spearman"]),
        pd.read_csv(evidence_dir / "rank_shift_summary.csv").assign(metric="rank_shift_abs_mean", value=lambda df: df["rank_shift_abs_mean"]),
    ]
    for part in ranking_parts:
        part["attacker"] = part["comparison"].str.contains("fgsm", case=False).map({True: "FGSM", False: "PGD"})
    ranking_rows = _aggregate_ranking_rows(pd.concat(ranking_parts, ignore_index=True))
    main_rows = pd.read_csv(report_root / "partial_attack_backtest_multiseed_ratio5_union" / "multiseed_summary_stats.csv").to_dict(orient="records")
    ratio_rows = pd.read_csv(report_root / "partial_attack_ratio_sweep_multiseed" / "ratio_sweep_summary_stats.csv").to_dict(orient="records")
    bootstrap_payload = json.loads((evidence_dir / "significance_block_bootstrap.json").read_text(encoding="utf-8"))

    text = render_report(
        significance_rows=significance_rows,
        ranking_rows=ranking_rows,
        main_rows=main_rows,
        ratio_rows=ratio_rows,
        bootstrap_payload=bootstrap_payload,
    )
    output_path = report_root / "lstm_single_model_evidence_report.md"
    output_path.write_text(text, encoding="utf-8")
    print(f"report_md={output_path}")


if __name__ == "__main__":
    main()
