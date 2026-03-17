from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_ranking_mechanism_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


METRIC_LABELS = {
    "topk_overlap": "Top-K 重合率",
    "spearman": "分数 Spearman",
    "rank_shift_abs_mean": "平均绝对排名位移",
}


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_ranking_mechanism_data(repo_root / "reports")

    metrics = ["topk_overlap", "spearman", "rank_shift_abs_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2) if profile == "slide" else (7.2, 3.2))
    for ax, metric in zip(axes, metrics, strict=True):
        subsets = []
        colors = []
        labels = []
        for attacker in ["FGSM", "PGD"]:
            subset = df[(df["metric"] == metric) & (df["attacker"] == attacker)]["value"].dropna().tolist()
            subsets.append(subset)
            colors.append(COLOR_MAP[attacker])
            labels.append(LABEL_MAP[attacker])
        box = ax.boxplot(subsets, patch_artist=True, tick_labels=labels)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor(color)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y")

    fig.suptitle("LSTM 白盒攻击的排序机制破坏")
    fig.tight_layout()
    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig08_ranking_mechanism_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig08 ranking mechanism chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
