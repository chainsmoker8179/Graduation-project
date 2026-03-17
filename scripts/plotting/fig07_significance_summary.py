from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_significance_summary_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_significance_summary_data(repo_root / "reports")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2) if profile == "slide" else (7.0, 3.6))
    daily_df = df[df["source"] == "daily"].copy()
    bootstrap_df = df[df["source"] == "bootstrap"].copy()

    daily_labels = [f"{row.attacker}\n{row.metric}" for row in daily_df.itertuples()]
    daily_colors = [COLOR_MAP[row.attacker] for row in daily_df.itertuples()]
    y_pos = np.arange(len(daily_df))
    axes[0].barh(y_pos, daily_df["delta_mean"], color=daily_colors, alpha=0.85)
    axes[0].set_yticks(y_pos, daily_labels)
    axes[0].axvline(0.0, color="#7F8A9A", linewidth=1.0)
    axes[0].set_title("日度配对显著性结果")
    axes[0].set_xlabel("平均差值")
    axes[0].grid(axis="x")
    for idx, row in enumerate(daily_df.itertuples()):
        axes[0].text(row.delta_mean, idx, f"  p={row.p_value:.2e}", va="center", ha="left" if row.delta_mean >= 0 else "right")

    boot_labels = [f"{row.attacker}\n{row.metric}" for row in bootstrap_df.itertuples()]
    x_pos = np.arange(len(bootstrap_df))
    for idx, row in enumerate(bootstrap_df.itertuples()):
        axes[1].errorbar(
            idx,
            row.delta_mean,
            yerr=[[row.delta_mean - row.ci95_lower], [row.ci95_upper - row.delta_mean]],
            fmt="o",
            color=COLOR_MAP[row.attacker],
            capsize=4,
        )
    axes[1].axhline(0.0, color="#7F8A9A", linewidth=1.0)
    axes[1].set_xticks(x_pos, boot_labels)
    axes[1].set_title("路径指标 Bootstrap 区间")
    axes[1].set_ylabel("差值估计")
    axes[1].grid(axis="y")

    fig.suptitle("LSTM 白盒攻击的统计显著性证据")
    fig.tight_layout()
    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig07_significance_summary_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig07 significance summary chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
