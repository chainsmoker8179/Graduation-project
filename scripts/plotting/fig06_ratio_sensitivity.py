from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_ratio_sensitivity_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


METRICS = [
    ("annualized_excess_return_with_cost", "年化超额收益退化"),
    ("max_drawdown_with_cost", "最大回撤恶化"),
    ("rank_ic_mean", "RankIC 下降"),
    ("information_ratio_with_cost", "信息比率下降"),
]


LAYOUT_MAP = {
    "slide": {
        "suptitle_y": 0.985,
        "legend_anchor_y": 0.955,
        "tight_rect_top": 0.92,
    },
    "paper": {
        "suptitle_y": 0.994,
        "legend_anchor_y": 0.960,
        "tight_rect_top": 0.90,
    },
}


def create_figure(repo_root: Path, profile: str):
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_ratio_sensitivity_data(repo_root / "reports")
    layout = LAYOUT_MAP[profile]

    fig, axes = plt.subplots(2, 2)
    fig.suptitle("攻击覆盖率对组合退化的影响", y=layout["suptitle_y"])
    x_order = [1, 5, 10]
    x_labels = ["1%", "5%", "10%"]
    for ax, (metric, title) in zip(axes.flatten(), METRICS, strict=True):
        for attacker in ["FGSM", "PGD"]:
            subset = df[(df["metric"] == metric) & (df["attacker"] == attacker)].sort_values("ratio_pct")
            ax.errorbar(
                x_order,
                subset["degradation_mean"],
                yerr=subset["degradation_std"],
                label=LABEL_MAP[attacker],
                color=COLOR_MAP[attacker],
                marker="o",
                capsize=4,
            )
        ax.set_title(title)
        ax.set_xticks(x_order, x_labels)
        ax.set_xlabel("攻击比例")
        ax.set_ylabel("退化幅度")
        ax.grid(axis="y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, layout["legend_anchor_y"]))
    fig.tight_layout(rect=(0, 0, 1, layout["tight_rect_top"]))
    return fig


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    fig = create_figure(repo_root=repo_root, profile=profile)
    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig06_ratio_sensitivity_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig06 ratio sensitivity chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
