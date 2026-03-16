from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_multiseed_stability_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


METRICS = [
    ("annualized_excess_return_with_cost", "年化超额收益退化"),
    ("max_drawdown_with_cost", "最大回撤恶化"),
    ("rank_ic_mean", "RankIC 下降"),
    ("information_ratio_with_cost", "信息比率下降"),
]


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_multiseed_stability_data(repo_root / "reports")

    fig, axes = plt.subplots(2, 2)
    fig.suptitle("5% 攻击比例下的多随机种子稳定性")
    x_positions = np.arange(2)
    labels = [LABEL_MAP["FGSM"], LABEL_MAP["PGD"]]
    colors = [COLOR_MAP["FGSM"], COLOR_MAP["PGD"]]
    for ax, (metric, title) in zip(axes.flatten(), METRICS, strict=True):
        subset = df[df["metric"] == metric].set_index("attacker").loc[["FGSM", "PGD"]]
        bars = ax.bar(
            x_positions,
            subset["degradation_mean"],
            yerr=subset["degradation_std"],
            color=colors,
            width=0.58,
            capsize=5,
        )
        for bar, color in zip(bars, colors, strict=True):
            bar.set_edgecolor(color)
            bar.set_alpha(0.8)
        ax.set_title(title)
        ax.set_ylabel("退化幅度")
        ax.set_xticks(x_positions, labels)
        ax.grid(axis="y")

    fig.tight_layout()
    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig05_multiseed_stability_ratio5_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig05 multiseed stability chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
