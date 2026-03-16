from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"{module_name} missing: {exc}")


def test_fig01_outputs_exist_for_slide_and_paper(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.fig01_clean_alignment")

    slide_paths = module.build_figure(repo_root=REPO_ROOT, profile="slide", output_dir=tmp_path)
    paper_paths = module.build_figure(repo_root=REPO_ROOT, profile="paper", output_dir=tmp_path)

    assert Path(slide_paths["png"]).name == "fig01_clean_alignment_repair_slide.png"
    assert Path(slide_paths["pdf"]).name == "fig01_clean_alignment_repair_slide.pdf"
    assert Path(paper_paths["png"]).name == "fig01_clean_alignment_repair_paper.png"
    assert Path(paper_paths["pdf"]).name == "fig01_clean_alignment_repair_paper.pdf"
    assert Path(slide_paths["png"]).exists()
    assert Path(slide_paths["pdf"]).exists()
    assert Path(paper_paths["png"]).exists()
    assert Path(paper_paths["pdf"]).exists()


def test_fig06_outputs_exist_for_slide_and_paper(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.fig06_ratio_sensitivity")

    slide_paths = module.build_figure(repo_root=REPO_ROOT, profile="slide", output_dir=tmp_path)
    paper_paths = module.build_figure(repo_root=REPO_ROOT, profile="paper", output_dir=tmp_path)

    assert Path(slide_paths["png"]).name == "fig06_ratio_sensitivity_slide.png"
    assert Path(slide_paths["pdf"]).name == "fig06_ratio_sensitivity_slide.pdf"
    assert Path(paper_paths["png"]).name == "fig06_ratio_sensitivity_paper.png"
    assert Path(paper_paths["pdf"]).name == "fig06_ratio_sensitivity_paper.pdf"
    assert Path(slide_paths["png"]).exists()
    assert Path(slide_paths["pdf"]).exists()
    assert Path(paper_paths["png"]).exists()
    assert Path(paper_paths["pdf"]).exists()


@pytest.mark.parametrize("profile", ["slide", "paper"])
def test_fig06_legend_and_suptitle_do_not_overlap(profile: str) -> None:
    module = _import_module("scripts.plotting.fig06_ratio_sensitivity")

    fig = module.create_figure(repo_root=REPO_ROOT, profile=profile)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0]
    suptitle = fig._suptitle
    inv = fig.transFigure.inverted()
    legend_bbox = inv.transform_bbox(legend.get_window_extent(renderer=renderer))
    title_bbox = inv.transform_bbox(suptitle.get_window_extent(renderer=renderer))

    assert title_bbox.y1 + 0.01 <= legend_bbox.y0 or legend_bbox.y1 + 0.01 <= title_bbox.y0


def test_fig04_outputs_exist_for_slide_and_paper(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.fig04_cumulative_return")

    slide_paths = module.build_figure(repo_root=REPO_ROOT, profile="slide", output_dir=tmp_path)
    paper_paths = module.build_figure(repo_root=REPO_ROOT, profile="paper", output_dir=tmp_path)

    assert Path(slide_paths["png"]).name == "fig04_cum_return_ratio5_seed0_slide.png"
    assert Path(slide_paths["pdf"]).name == "fig04_cum_return_ratio5_seed0_slide.pdf"
    assert Path(paper_paths["png"]).name == "fig04_cum_return_ratio5_seed0_paper.png"
    assert Path(paper_paths["pdf"]).name == "fig04_cum_return_ratio5_seed0_paper.pdf"
    assert Path(slide_paths["png"]).exists()
    assert Path(slide_paths["pdf"]).exists()
    assert Path(paper_paths["png"]).exists()
    assert Path(paper_paths["pdf"]).exists()


def test_fig02_outputs_exist_for_slide_and_paper(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.fig02_sample_shift_box")

    slide_paths = module.build_figure(repo_root=REPO_ROOT, profile="slide", output_dir=tmp_path)
    paper_paths = module.build_figure(repo_root=REPO_ROOT, profile="paper", output_dir=tmp_path)

    assert Path(slide_paths["png"]).name == "fig02_sample_shift_distribution_slide.png"
    assert Path(slide_paths["pdf"]).name == "fig02_sample_shift_distribution_slide.pdf"
    assert Path(paper_paths["png"]).name == "fig02_sample_shift_distribution_paper.png"
    assert Path(paper_paths["pdf"]).name == "fig02_sample_shift_distribution_paper.pdf"
    assert Path(slide_paths["png"]).exists()
    assert Path(slide_paths["pdf"]).exists()
    assert Path(paper_paths["png"]).exists()
    assert Path(paper_paths["pdf"]).exists()


def test_fig03_outputs_exist_for_slide_and_paper(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.fig03_sample_shift_cdf")

    slide_paths = module.build_figure(repo_root=REPO_ROOT, profile="slide", output_dir=tmp_path)
    paper_paths = module.build_figure(repo_root=REPO_ROOT, profile="paper", output_dir=tmp_path)

    assert Path(slide_paths["png"]).name == "fig03_sample_shift_cdf_slide.png"
    assert Path(slide_paths["pdf"]).name == "fig03_sample_shift_cdf_slide.pdf"
    assert Path(paper_paths["png"]).name == "fig03_sample_shift_cdf_paper.png"
    assert Path(paper_paths["pdf"]).name == "fig03_sample_shift_cdf_paper.pdf"
    assert Path(slide_paths["png"]).exists()
    assert Path(slide_paths["pdf"]).exists()
    assert Path(paper_paths["png"]).exists()
    assert Path(paper_paths["pdf"]).exists()


def test_fig05_outputs_exist_for_slide_and_paper(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.fig05_multiseed_stability")

    slide_paths = module.build_figure(repo_root=REPO_ROOT, profile="slide", output_dir=tmp_path)
    paper_paths = module.build_figure(repo_root=REPO_ROOT, profile="paper", output_dir=tmp_path)

    assert Path(slide_paths["png"]).name == "fig05_multiseed_stability_ratio5_slide.png"
    assert Path(slide_paths["pdf"]).name == "fig05_multiseed_stability_ratio5_slide.pdf"
    assert Path(paper_paths["png"]).name == "fig05_multiseed_stability_ratio5_paper.png"
    assert Path(paper_paths["pdf"]).name == "fig05_multiseed_stability_ratio5_paper.pdf"
    assert Path(slide_paths["png"]).exists()
    assert Path(slide_paths["pdf"]).exists()
    assert Path(paper_paths["png"]).exists()
    assert Path(paper_paths["pdf"]).exists()


def test_build_all_figures_generates_slide_outputs(tmp_path: Path) -> None:
    module = _import_module("scripts.plotting.build_all_figures")

    outputs = module.build_all_figures(repo_root=REPO_ROOT, profile="slide", output_root=tmp_path)

    expected_keys = {"fig01", "fig02", "fig03", "fig04", "fig05", "fig06"}
    assert set(outputs.keys()) == expected_keys
    assert (tmp_path / "slide" / "fig01_clean_alignment_repair_slide.png").exists()
    assert (tmp_path / "slide" / "fig06_ratio_sensitivity_slide.pdf").exists()
