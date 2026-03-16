from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_clean_alignment_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


METRICS = [
    ("spearman_to_reference", "排序对齐", "Spearman 相关"),
    ("clean_loss", "Clean 损失", "损失值"),
    ("fgsm_loss", "FGSM 攻击损失", "损失值"),
    ("pgd_loss", "PGD 攻击损失", "损失值"),
]


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_clean_alignment_data(repo_root / "reports")
    label_values = [LABEL_MAP["old"], LABEL_MAP["new"]]
    version_order = ["expanded_v3", "expanded_v6"]
    x_positions = range(len(version_order))

    fig, axes = plt.subplots(2, 2)
    fig.suptitle("攻击链路修正前后的 clean 对齐与攻击响应")
    for ax, (metric, title, ylabel) in zip(axes.flatten(), METRICS, strict=True):
        values = [df.loc[df["version"] == version, metric].item() for version in version_order]
        colors = [COLOR_MAP["old"], COLOR_MAP["new"]]
        ax.bar(list(x_positions), values, color=colors, width=0.58)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x_positions), label_values)
        ax.grid(axis="y")

    fig.tight_layout()
    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig01_clean_alignment_repair_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig01 clean alignment repair chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
