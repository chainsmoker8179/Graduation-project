from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plotting.fig01_clean_alignment import build_figure as build_fig01
from scripts.plotting.fig02_sample_shift_box import build_figure as build_fig02
from scripts.plotting.fig03_sample_shift_cdf import build_figure as build_fig03
from scripts.plotting.fig04_cumulative_return import build_figure as build_fig04
from scripts.plotting.fig05_multiseed_stability import build_figure as build_fig05
from scripts.plotting.fig06_ratio_sensitivity import build_figure as build_fig06


FIGURE_BUILDERS = {
    "fig01": build_fig01,
    "fig02": build_fig02,
    "fig03": build_fig03,
    "fig04": build_fig04,
    "fig05": build_fig05,
    "fig06": build_fig06,
}


def build_all_figures(repo_root: Path, profile: str, output_root: Path | None = None) -> dict[str, dict[str, str]]:
    repo_root = Path(repo_root).resolve()
    base_output = Path(output_root) if output_root is not None else (repo_root / "reports" / "figures")
    profile_output = base_output / profile
    profile_output.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    for figure_name, builder in FIGURE_BUILDERS.items():
        outputs[figure_name] = builder(repo_root=repo_root, profile=profile, output_dir=profile_output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all attack visualization figures.")
    parser.add_argument("--profile", choices=["slide", "paper"], help="Single profile to build")
    parser.add_argument("--all", action="store_true", help="Build both slide and paper profiles")
    args = parser.parse_args()

    if not args.all and not args.profile:
        raise SystemExit("Either --profile or --all must be provided.")

    repo_root = REPO_ROOT
    if args.all:
        for profile in ["slide", "paper"]:
            build_all_figures(repo_root=repo_root, profile=profile)
    else:
        build_all_figures(repo_root=repo_root, profile=args.profile)


if __name__ == "__main__":
    main()
