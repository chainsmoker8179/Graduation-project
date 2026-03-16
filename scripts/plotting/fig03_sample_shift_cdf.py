from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_sample_shift_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_sample_shift_data(repo_root / "reports")

    fig, ax = plt.subplots()
    for attacker in ["FGSM", "PGD"]:
        values = np.sort(df.loc[df["attacker"] == attacker, "abs_pred_shift"].to_numpy())
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, color=COLOR_MAP[attacker], label=LABEL_MAP[attacker], marker="o", markevery=max(len(values) // 10, 1))

    ax.set_title("样本级预测偏移的累计分布")
    ax.set_xlabel(r"绝对预测偏移 $|\hat{y}_{adv} - \hat{y}_{clean}|$")
    ax.set_ylabel("累计样本比例")
    ax.grid(axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()

    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig03_sample_shift_cdf_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig03 sample shift CDF chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
