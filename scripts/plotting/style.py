from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

mpl.use("Agg")


def _detect_preferred_fonts() -> list[str]:
    candidate_paths = [
        Path("/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    ]
    font_names: list[str] = []
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            if font_name not in font_names:
                font_names.append(font_name)
        except RuntimeError:
            continue
    if "DejaVu Sans" not in font_names:
        font_names.append("DejaVu Sans")
    return font_names


FONT_FAMILIES = _detect_preferred_fonts()


COLOR_MAP = {
    "reference_clean": "#4C566A",
    "partial_clean": "#A7B1C2",
    "FGSM": "#E69F00",
    "PGD": "#D55E00",
    "old": "#B8C1CC",
    "new": "#2F6DB3",
}

LABEL_MAP = {
    "reference_clean": "参考 clean",
    "partial_clean": "局部 clean",
    "FGSM": "FGSM",
    "PGD": "PGD",
    "old": "修正前",
    "new": "修正后",
}

PROFILE_MAP = {
    "slide": {
        "figure.figsize": (16, 9),
        "figure.dpi": 220,
        "savefig.dpi": 240,
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 3.0,
        "lines.markersize": 7,
    },
    "paper": {
        "figure.figsize": (6.9, 4.2),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.5,
    },
}

BASE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILIES,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#BFC7D5",
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "grid.color": "#E8ECF2",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style(profile: str) -> dict[str, object]:
    if profile not in PROFILE_MAP:
        raise ValueError(f"Unknown profile: {profile}")
    params = {**BASE_STYLE, **PROFILE_MAP[profile]}
    mpl.rcParams.update(params)
    return params


def figure_output_dir(repo_root: Path, profile: str) -> Path:
    out_dir = repo_root / "reports" / "figures" / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def figure_data_dir(repo_root: Path) -> Path:
    data_dir = repo_root / "reports" / "figures" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def save_figure(fig, output_dir: Path, stem: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
