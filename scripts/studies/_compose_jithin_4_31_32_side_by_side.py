"""Side-by-side comparison: Jithin's Fig 4.31/4.32 vs our reproductions.

Extracts Fig 4.31 (top) and Fig 4.32 (bottom) from page 135 of the Jithin
thesis PDF at 200 dpi, pairs each with the matching output from our solver,
and writes both per-figure pairs and a combined 2-row panel.

Inputs (must exist before running):
    docs/papers/Jithin Thesis.pdf
    StudyResults/jithin_fig_4_31_4_32/fig_4_31_ratio.png
    StudyResults/jithin_fig_4_31_4_32/fig_4_32_jv.png

Outputs:
    StudyResults/jithin_fig_4_31_4_32/jithin_pdf_pages/fig_4_31_crop.png
    StudyResults/jithin_fig_4_31_4_32/jithin_pdf_pages/fig_4_32_crop.png
    StudyResults/jithin_fig_4_31_4_32/compare_fig_4_31.png
    StudyResults/jithin_fig_4_31_4_32/compare_fig_4_32.png
    StudyResults/jithin_fig_4_31_4_32/compare_all.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
PDF_PATH = _ROOT / "docs" / "papers" / "Jithin Thesis.pdf"
OUT_DIR = _ROOT / "StudyResults" / "jithin_fig_4_31_4_32"
PDF_PAGES_DIR = OUT_DIR / "jithin_pdf_pages"

# Page 135 carries Fig 4.31 (top) and Fig 4.32 (bottom).  Crop boxes in
# (left, upper, right, lower) at 200 dpi on a letter page (~1700×2200 px).
# Generous top/bottom split so neither plot+caption is clipped; tighten if
# the render DPI changes.
PDF_PAGE: str = "page-135.png"
CROPS = {
    "fig_4_31": (PDF_PAGE, (180, 250, 1520, 1110)),
    "fig_4_32": (PDF_PAGE, (180, 1110, 1520, 1980)),
}

OUR_FIGS = {
    "fig_4_31": OUT_DIR / "fig_4_31_ratio.png",
    "fig_4_32": OUT_DIR / "fig_4_32_jv.png",
}

FIG_TITLES = {
    "fig_4_31": "Fig 4.31 — c*_O₂ / c^b_O₂ vs V (volume sweep)",
    "fig_4_32": "Fig 4.32 — simulated jV (volume sweep)",
}


def _ensure_pdf_page() -> None:
    """Render PDF page 135 at 200 dpi if not already present."""
    PDF_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    target = PDF_PAGES_DIR / PDF_PAGE
    if target.exists():
        return
    cmd = [
        "pdftoppm", "-r", "200", "-f", "135", "-l", "135",
        str(PDF_PATH), str(PDF_PAGES_DIR / "page"), "-png",
    ]
    subprocess.run(cmd, check=True)


def crop_jithin_figs() -> dict[str, Path]:
    """Crop each Jithin figure out of page 135 and save as a tight PNG."""
    _ensure_pdf_page()
    out_paths: dict[str, Path] = {}
    for key, (src, box) in CROPS.items():
        src_path = PDF_PAGES_DIR / src
        if not src_path.exists():
            raise FileNotFoundError(src_path)
        im = Image.open(src_path).crop(box)
        out = PDF_PAGES_DIR / f"{key}_crop.png"
        im.save(out)
        out_paths[key] = out
        print(f"  cropped {src} → {out.name} ({im.size})")
    return out_paths


def make_pair(key: str, jithin_png: Path, ours_png: Path) -> Path:
    """Build a left/right pair: Jithin original vs our reproduction."""
    jim = mpimg.imread(jithin_png)
    oim = mpimg.imread(ours_png)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    axes[0].imshow(jim)
    axes[0].set_axis_off()
    axes[0].set_title(f"Jithin (thesis): {FIG_TITLES[key]}", fontsize=10)
    axes[1].imshow(oim)
    axes[1].set_axis_off()
    axes[1].set_title(f"Our solver: {FIG_TITLES[key]}", fontsize=10)
    out = OUT_DIR / f"compare_{key}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def make_combined(jithin_pngs: dict[str, Path]) -> Path:
    """Stack both comparisons in a 2-row × 2-col grid."""
    keys = ["fig_4_31", "fig_4_32"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for r, key in enumerate(keys):
        jim = mpimg.imread(jithin_pngs[key])
        oim = mpimg.imread(OUR_FIGS[key])
        axes[r, 0].imshow(jim)
        axes[r, 0].set_axis_off()
        axes[r, 0].set_title(
            f"Jithin (thesis): {FIG_TITLES[key]}", fontsize=10,
        )
        axes[r, 1].imshow(oim)
        axes[r, 1].set_axis_off()
        axes[r, 1].set_title(
            f"Our solver: {FIG_TITLES[key]}", fontsize=10,
        )
    fig.suptitle(
        "Jithin Thesis §4.8 volume sweep — side-by-side reproduction\n"
        "(Cs⁺/SO₄²⁻ pH 2, L_eff=50 µm, C_S=1.738 F/m², "
        "A_Tafel=142 mV/dec, single-Tafel R2e, steric via PDE)",
        fontsize=11, y=1.005,
    )
    out = OUT_DIR / "compare_all.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: {PDF_PATH} not found", file=sys.stderr)
        return 1
    for key, path in OUR_FIGS.items():
        if not path.exists():
            print(f"ERROR: missing our figure {path}; run "
                  "_run_jithin_fig_4_31_32.py + _plot_jithin_fig_4_31_32.py "
                  "first.", file=sys.stderr)
            return 1

    crops = crop_jithin_figs()
    pair_paths = [make_pair(k, crops[k], OUR_FIGS[k]) for k in crops]
    combined = make_combined(crops)
    for p in pair_paths:
        print(f"  wrote {p}")
    print(f"  wrote {combined}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
