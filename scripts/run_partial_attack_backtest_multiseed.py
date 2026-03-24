#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PRIMARY_COMPARISON_METRICS = [
    "annualized_excess_return_with_cost",
    "max_drawdown_with_cost",
    "rank_ic_mean",
    "information_ratio_with_cost",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiseed partial-attack backtests and summarize results.")
    parser.add_argument("--pred-pkl", type=Path, default=Path("origin_model_pred/LSTM/prediction/pred.pkl"))
    parser.add_argument("--label-pkl", type=Path, default=Path("origin_model_pred/LSTM/prediction/label.pkl"))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--state-dict-path", type=Path, default=Path("origin_model_pred/LSTM/model/lstm_state_dict.pt"))
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provider-uri", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--attack-ratio", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-drop", type=int, default=5)
    parser.add_argument("--benchmark", type=str, default="SH000300")
    parser.add_argument("--account", type=float, default=100000000)
    parser.add_argument("--start-time", type=str, default=None)
    parser.add_argument("--end-time", type=str, default=None)
    parser.add_argument("--price-epsilon", type=float, default=0.01)
    parser.add_argument("--volume-epsilon", type=float, default=0.02)
    parser.add_argument("--price-floor", type=float, default=1e-6)
    parser.add_argument("--volume-floor", type=float, default=1.0)
    parser.add_argument("--pgd-steps", type=int, default=5)
    parser.add_argument("--pgd-step-size", type=float, default=0.25)
    parser.add_argument("--attack-batch-size", type=int, default=256)
    parser.add_argument("--constraint-mode", type=str, default="physical_stat", choices=["none", "physical", "physical_stat"])
    parser.add_argument("--tau-ret", type=float, default=0.005)
    parser.add_argument("--tau-body", type=float, default=0.005)
    parser.add_argument("--tau-range", type=float, default=0.01)
    parser.add_argument("--tau-vol", type=float, default=0.05)
    parser.add_argument("--lambda-ret", type=float, default=0.8)
    parser.add_argument("--lambda-candle", type=float, default=0.4)
    parser.add_argument("--lambda-vol", type=float, default=0.3)
    parser.add_argument("--min-clean-grad-mean-abs", type=float, default=1e-6)
    parser.add_argument("--min-spearman-to-reference", type=float, default=0.09)
    parser.add_argument("--max-feature-mae-to-reference", type=float, default=0.05)
    parser.add_argument("--max-feature-rmse-to-reference", type=float, default=0.12)
    parser.add_argument("--max-feature-max-abs-to-reference", type=float, default=0.7)
    parser.add_argument("--deal-price", type=str, default="close")
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0015)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    return parser.parse_args(argv)


def build_seed_metric_row(seed: int, summary: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {
        "seed": int(seed),
        "attackable_count": float(summary["generation"]["attackable_count"]),
        "selected_available_count": float(summary["generation"]["selected_available_count"]),
    }
    for delta_name, metrics in summary["comparison"].items():
        for metric_name, value in metrics.items():
            if metric_name not in PRIMARY_COMPARISON_METRICS:
                continue
            row[f"{delta_name}__{metric_name}"] = float(value)
    return row


def build_multiseed_summary_stats(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [col for col in seed_metrics.columns if col != "seed"]:
        series = seed_metrics[column].astype(float)
        rows.append(
            {
                "metric": column,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "negative_ratio": float((series < 0).mean()),
                "positive_ratio": float((series > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_multiseed_pgd_vs_fgsm_summary(
    seed_metrics: pd.DataFrame,
    *,
    metrics: list[str] = PRIMARY_COMPARISON_METRICS,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        fgsm_col = f"partial_fgsm_minus_partial_clean__{metric}"
        pgd_col = f"partial_pgd_minus_partial_clean__{metric}"
        fgsm = seed_metrics[fgsm_col].astype(float)
        pgd = seed_metrics[pgd_col].astype(float)
        summary[metric] = {
            "pgd_more_negative_ratio": float((pgd < fgsm).mean()),
            "fgsm_mean": float(fgsm.mean()),
            "pgd_mean": float(pgd.mean()),
        }
    return summary


def write_multiseed_report(
    out_dir: Path,
    *,
    summary_stats: pd.DataFrame,
    pgd_vs_fgsm: dict[str, dict[str, float]],
    seeds: list[int],
    attack_ratio: float,
) -> None:
    stats = summary_stats.set_index("metric")
    lines = [
        "# 部分股票白盒攻击回测多随机种子稳定性报告",
        "",
        "## 1. 实验设置",
        "",
        f"- 攻击比例：{attack_ratio:.2%}",
        f"- 随机种子：{', '.join(str(seed) for seed in seeds)}",
        "",
        "## 2. 主统计结果",
        "",
        "| 差值组别 | 年化超额收益(含费) | 最大回撤(含费) | RankIC 均值 | 信息比率(含费) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for prefix, label in [
        ("partial_fgsm_minus_partial_clean", "`partial_fgsm - partial_clean` 平均值"),
        ("partial_pgd_minus_partial_clean", "`partial_pgd - partial_clean` 平均值"),
    ]:
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{stats.loc[f'{prefix}__{metric}', 'mean']:.8f}"
                for metric in PRIMARY_COMPARISON_METRICS
            )
            + " |"
        )
        lines.append(
            "| "
            + label.replace("平均值", "标准差")
            + " | "
            + " | ".join(
                f"{stats.loc[f'{prefix}__{metric}', 'std']:.8f}"
                for metric in PRIMARY_COMPARISON_METRICS
            )
            + " |"
        )
    lines.extend(["", "## 3. PGD 与 FGSM 对比", "", json.dumps(pgd_vs_fgsm, ensure_ascii=False, indent=2), ""])
    (out_dir / "多随机种子稳定性报告.md").write_text("\n".join(lines), encoding="utf-8")


def build_single_seed_command(args: argparse.Namespace, seed: int, seed_out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_partial_attack_backtest.py"),
        "--pred-pkl",
        str(args.pred_pkl),
        "--label-pkl",
        str(args.label_pkl),
        "--asset-dir",
        str(args.asset_dir),
        "--state-dict-path",
        str(args.state_dict_path),
        "--config-path",
        str(args.config_path),
        "--out-dir",
        str(seed_out_dir),
        "--provider-uri",
        str(args.provider_uri),
        "--device",
        str(args.device),
        "--seed",
        str(seed),
        "--attack-ratio",
        str(args.attack_ratio),
        "--topk",
        str(args.topk),
        "--n-drop",
        str(args.n_drop),
        "--benchmark",
        str(args.benchmark),
        "--account",
        str(args.account),
        "--price-epsilon",
        str(args.price_epsilon),
        "--volume-epsilon",
        str(args.volume_epsilon),
        "--price-floor",
        str(args.price_floor),
        "--volume-floor",
        str(args.volume_floor),
        "--pgd-steps",
        str(args.pgd_steps),
        "--pgd-step-size",
        str(args.pgd_step_size),
        "--attack-batch-size",
        str(args.attack_batch_size),
        "--constraint-mode",
        str(args.constraint_mode),
        "--tau-ret",
        str(args.tau_ret),
        "--tau-body",
        str(args.tau_body),
        "--tau-range",
        str(args.tau_range),
        "--tau-vol",
        str(args.tau_vol),
        "--lambda-ret",
        str(args.lambda_ret),
        "--lambda-candle",
        str(args.lambda_candle),
        "--lambda-vol",
        str(args.lambda_vol),
        "--min-clean-grad-mean-abs",
        str(args.min_clean_grad_mean_abs),
        "--min-spearman-to-reference",
        str(args.min_spearman_to_reference),
        "--max-feature-mae-to-reference",
        str(args.max_feature_mae_to_reference),
        "--max-feature-rmse-to-reference",
        str(args.max_feature_rmse_to_reference),
        "--max-feature-max-abs-to-reference",
        str(args.max_feature_max_abs_to_reference),
        "--deal-price",
        str(args.deal_price),
        "--open-cost",
        str(args.open_cost),
        "--close-cost",
        str(args.close_cost),
        "--min-cost",
        str(args.min_cost),
        "--limit-threshold",
        str(args.limit_threshold),
    ]
    if args.start_time is not None:
        cmd.extend(["--start-time", str(args.start_time)])
    if args.end_time is not None:
        cmd.extend(["--end-time", str(args.end_time)])
    return cmd


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    for seed in args.seeds:
        seed_out_dir = args.out_dir / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_single_seed_command(args, seed, seed_out_dir), check=True)
        summary = json.loads((seed_out_dir / "backtest_summary.json").read_text(encoding="utf-8"))
        seed_rows.append(build_seed_metric_row(seed, summary))

    seed_metrics = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    summary_stats = build_multiseed_summary_stats(seed_metrics)
    pgd_vs_fgsm = build_multiseed_pgd_vs_fgsm_summary(seed_metrics)

    seed_metrics.to_csv(args.out_dir / "multiseed_seed_metrics.csv", index=False)
    summary_stats.to_csv(args.out_dir / "multiseed_summary_stats.csv", index=False)
    (args.out_dir / "multiseed_pgd_vs_fgsm.json").write_text(
        json.dumps(pgd_vs_fgsm, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_multiseed_report(
        args.out_dir,
        summary_stats=summary_stats,
        pgd_vs_fgsm=pgd_vs_fgsm,
        seeds=list(args.seeds),
        attack_ratio=float(args.attack_ratio),
    )

    print(f"seed_metrics_csv={args.out_dir / 'multiseed_seed_metrics.csv'}")
    print(f"summary_stats_csv={args.out_dir / 'multiseed_summary_stats.csv'}")
    print(f"pgd_vs_fgsm_json={args.out_dir / 'multiseed_pgd_vs_fgsm.json'}")


if __name__ == "__main__":
    main()
