from __future__ import annotations

import argparse
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


def _bootstrap_line(payload: dict, comparison: str, metric: str, label: str) -> str:
    summary = payload.get(comparison, {}).get(metric)
    if summary is None:
        return f"- `{comparison}` 的 `{label}` bootstrap 区间结果缺失。"
    return f"- `{comparison}` 在 `{label}` 上的 block bootstrap 区间为 `[{summary['ci95_lower']:.4f}, {summary['ci95_upper']:.4f}]`。"


def _find_p_value(rows: list[dict], comparison: str, metric: str) -> float | None:
    for row in rows:
        if row.get("comparison") == comparison and row.get("metric") == metric:
            return float(row["p_value"])
    return None


def _find_ci(payload: dict, comparison: str, metric: str) -> tuple[float, float] | None:
    summary = payload.get(comparison, {}).get(metric)
    if summary is None:
        return None
    return float(summary["ci95_lower"]), float(summary["ci95_upper"])


def _contains_zero(interval: tuple[float, float] | None) -> bool | None:
    if interval is None:
        return None
    lower, upper = interval
    return lower <= 0.0 <= upper


def _conclusion_lines(significance_rows: list[dict], bootstrap_payload: dict) -> list[str]:
    fgsm_ret_p = _find_p_value(significance_rows, "partial_fgsm_vs_partial_clean", "excess_return_with_cost")
    fgsm_rank_p = _find_p_value(significance_rows, "partial_fgsm_vs_partial_clean", "rank_ic")
    pgd_ret_p = _find_p_value(significance_rows, "partial_pgd_vs_partial_clean", "excess_return_with_cost")
    pgd_rank_p = _find_p_value(significance_rows, "partial_pgd_vs_partial_clean", "rank_ic")
    fgsm_ann_ci = _find_ci(bootstrap_payload, "partial_fgsm_vs_partial_clean", "annualized_excess_return_with_cost")
    pgd_ann_ci = _find_ci(bootstrap_payload, "partial_pgd_vs_partial_clean", "annualized_excess_return_with_cost")

    lines: list[str] = []
    if fgsm_rank_p is not None and fgsm_rank_p < 0.05:
        if fgsm_ret_p is not None and fgsm_ret_p >= 0.05 and _contains_zero(fgsm_ann_ci):
            lines.append("- FGSM 在排序层面的扰动显著，但收益打击不具有统计显著性。")
        else:
            lines.append("- FGSM 在排序与收益两个层面都表现出统计显著退化。")
    else:
        lines.append("- FGSM 当前没有形成稳定的统计显著退化证据。")

    if pgd_ret_p is not None and pgd_ret_p < 0.05 and pgd_rank_p is not None and pgd_rank_p < 0.05:
        if pgd_ann_ci is not None and not _contains_zero(pgd_ann_ci):
            lines.append("- PGD 在收益与排序两个层面都具有统计显著性，且年化收益退化区间稳定落在负向区间。")
        else:
            lines.append("- PGD 在收益与排序两个层面都具有统计显著性，但收益区间稳定性仍需结合更多重复实验观察。")
    else:
        lines.append("- PGD 当前尚未在收益与排序两个层面同时形成完整的统计显著证据。")
    return lines


def render_report(
    *,
    summary_rows: list[dict],
    significance_rows: list[dict],
    bootstrap_payload: dict,
) -> str:
    fgsm_ann = _pick_mean(summary_rows, "partial_fgsm_minus_partial_clean__annualized_excess_return_with_cost")
    pgd_ann = _pick_mean(summary_rows, "partial_pgd_minus_partial_clean__annualized_excess_return_with_cost")
    fgsm_rank = _pick_mean(summary_rows, "partial_fgsm_minus_partial_clean__rank_ic_mean")
    pgd_rank = _pick_mean(summary_rows, "partial_pgd_minus_partial_clean__rank_ic_mean")

    lines = [
        "# Physical-Stat 多随机种子显著性检验报告",
        "",
        "## 1. 检验对象",
        "- 目标目录：`partial_attack_backtest_multiseed_ratio5_physical_stat`",
        "- 对照组：`partial_clean`",
        "- 攻击组：`partial_fgsm`、`partial_pgd`",
        "",
        "## 2. 主结果回顾",
        f"- `FGSM` 相对 `partial_clean` 的年化超额收益均值差值为 `{fgsm_ann:.4f}`，`RankIC` 均值差值为 `{fgsm_rank:.6f}`。"
        if fgsm_ann is not None and fgsm_rank is not None
        else "- `FGSM` 主结果缺失。",
        f"- `PGD` 相对 `partial_clean` 的年化超额收益均值差值为 `{pgd_ann:.4f}`，`RankIC` 均值差值为 `{pgd_rank:.6f}`。"
        if pgd_ann is not None and pgd_rank is not None
        else "- `PGD` 主结果缺失。",
        "",
        "## 3. 日度配对显著性",
        _daily_sig_line(significance_rows, "partial_fgsm_vs_partial_clean", "excess_return_with_cost", "超额收益"),
        _daily_sig_line(significance_rows, "partial_pgd_vs_partial_clean", "excess_return_with_cost", "超额收益"),
        _daily_sig_line(significance_rows, "partial_fgsm_vs_partial_clean", "rank_ic", "RankIC"),
        _daily_sig_line(significance_rows, "partial_pgd_vs_partial_clean", "rank_ic", "RankIC"),
        "",
        "## 4. Block Bootstrap 区间",
        _bootstrap_line(bootstrap_payload, "partial_fgsm_vs_partial_clean", "annualized_excess_return_with_cost", "年化超额收益"),
        _bootstrap_line(bootstrap_payload, "partial_pgd_vs_partial_clean", "annualized_excess_return_with_cost", "年化超额收益"),
        _bootstrap_line(bootstrap_payload, "partial_fgsm_vs_partial_clean", "information_ratio_with_cost", "信息比率"),
        _bootstrap_line(bootstrap_payload, "partial_pgd_vs_partial_clean", "information_ratio_with_cost", "信息比率"),
        _bootstrap_line(bootstrap_payload, "partial_fgsm_vs_partial_clean", "max_drawdown_with_cost", "最大回撤"),
        _bootstrap_line(bootstrap_payload, "partial_pgd_vs_partial_clean", "max_drawdown_with_cost", "最大回撤"),
        "",
        "## 5. 结论",
    ]
    lines.extend(_conclusion_lines(significance_rows, bootstrap_payload))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Chinese significance report for physical_stat multiseed backtest.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/partial_attack_backtest_multiseed_ratio5_physical_stat"),
    )
    args = parser.parse_args()

    report_dir = args.report_dir
    summary_rows = pd.read_csv(report_dir / "multiseed_summary_stats.csv").to_dict(orient="records")
    significance_rows = pd.read_csv(report_dir / "significance_daily_metrics.csv").to_dict(orient="records")
    bootstrap_payload = json.loads((report_dir / "significance_block_bootstrap.json").read_text(encoding="utf-8"))

    text = render_report(
        summary_rows=summary_rows,
        significance_rows=significance_rows,
        bootstrap_payload=bootstrap_payload,
    )
    output_path = report_dir / "显著性检验报告.md"
    output_path.write_text(text, encoding="utf-8")
    print(f"report_md={output_path}")


if __name__ == "__main__":
    main()
