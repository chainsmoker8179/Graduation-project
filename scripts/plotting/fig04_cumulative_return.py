from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.loaders import load_cumulative_return_data
from scripts.plotting.style import COLOR_MAP, LABEL_MAP, apply_style, figure_output_dir, save_figure


SERIES_MAP = [
    ("reference_clean_excess_return_with_cost", "reference_clean"),
    ("partial_clean_excess_return_with_cost", "partial_clean"),
    ("partial_fgsm_excess_return_with_cost", "FGSM"),
    ("partial_pgd_excess_return_with_cost", "PGD"),
]


def build_figure(repo_root: Path, profile: str, output_dir: Path | None = None) -> dict[str, str]:
    repo_root = Path(repo_root).resolve()
    apply_style(profile)
    df = load_cumulative_return_data(repo_root / "reports").copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    fig, ax = plt.subplots()
    for column, key in SERIES_MAP:
        cumulative = df[column].cumsum()
        ax.plot(
            df["datetime"],
            cumulative,
            label=LABEL_MAP[key],
            color=COLOR_MAP[key],
        )

    ax.set_title("局部攻击下的组合级累计超额收益")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计超额收益（含成本）")
    ax.grid(axis="y")
    ax.legend(loc="best")
    fig.tight_layout()

    target_dir = Path(output_dir) if output_dir is not None else figure_output_dir(repo_root, profile)
    return save_figure(fig, target_dir, f"fig04_cum_return_ratio5_seed0_{profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fig04 cumulative return chart.")
    parser.add_argument("--profile", choices=["slide", "paper"], required=True)
    args = parser.parse_args()
    build_figure(repo_root=REPO_ROOT, profile=args.profile)


if __name__ == "__main__":
    main()
