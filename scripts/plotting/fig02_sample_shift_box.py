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
    attackers = ["FGSM", "PGD"]
    positions = np.arange(1, len(attackers) + 1)
    box_data = [df.loc[df["attacker"] == attacker, "abs_pred_shift"].to_numpy() for attacker in attackers]

    box = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#2E3440", "linewidth": 2},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    for patch, attacker in zip(box["boxes"], attackers, strict=True):
        patch.set_facecolor(COLOR_MAP[attacker])
        patch.set_alpha(0.55)
        patch.set_edgecolor(COLOR_MAP[attacker])

    rng = np.random.default_rng(7)
    for pos, attacker in zip(positions, attackers, strict=True):
        values = df.loc[df["attacker"] == attacker, "abs_pred_shift"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=20 if profile == "slide" else 10,
            alpha=0.45,
            color=COLOR_MAP[attacker],
            edgecolors="none",
        )

    ax.set_xticks(positions, [LABEL_MAP["FGSM"], LABEL_MAP["PGD"]])
    ax.set_title("样本级预测偏移分布")
    ax.set_xlabel("攻击方法")
    ax.set_ylabel(r"绝对预测偏移 $|\hat{y}_{adv} - \hat{y}_{clean}|$")
    ax.grid(axis="y")
    fig.tight_layout()

    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig02_sample_shift_distribution_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig02 sample shift boxplot.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
