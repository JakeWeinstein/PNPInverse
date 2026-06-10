"""Side-by-side comparison: Jithin's original figures vs our reproductions.

Extracts Fig 4.26/4.27/4.28 from the Jithin thesis PDF (pages 131-132 at
200 dpi), pairs each with the matching output from our solver, and writes
both per-figure pairs and a combined 3-row panel.

Inputs (must exist before running):
    docs/papers/Jithin Thesis.pdf
    StudyResults/jithin_fig_4_26_4_27_4_28/fig_4_26_ratio.png
    StudyResults/jithin_fig_4_26_4_27_4_28/fig_4_27_O2.png
    StudyResults/jithin_fig_4_26_4_27_4_28/fig_4_28_H.png

Outputs:
    StudyResults/jithin_fig_4_26_4_27_4_28/jithin_pdf_pages/fig_4_26_crop.png
    StudyResults/jithin_fig_4_26_4_27_4_28/jithin_pdf_pages/fig_4_27_crop.png
    StudyResults/jithin_fig_4_26_4_27_4_28/jithin_pdf_pages/fig_4_28_crop.png
    StudyResults/jithin_fig_4_26_4_27_4_28/compare_fig_4_26.png
    StudyResults/jithin_fig_4_26_4_27_4_28/compare_fig_4_27.png
    StudyResults/jithin_fig_4_26_4_27_4_28/compare_fig_4_28.png
    StudyResults/jithin_fig_4_26_4_27_4_28/compare_all.png
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
OUT_DIR = _ROOT / "StudyResults" / "jithin_fig_4_26_4_27_4_28"
PDF_PAGES_DIR = OUT_DIR / "jithin_pdf_pages"

# Crop boxes in (left, upper, right, lower) at 200 dpi on letter-sized pages.
# Page 131 carries Fig 4.26 (top) and Fig 4.27 (bottom); page 132 carries
# Fig 4.28 at the top.  Boxes are tight on plot+caption; tweak if PDF DPI
# changes.
CROPS = {
    "fig_4_26": ("page-131.png", (260, 280, 1450, 1010)),
    "fig_4_27": ("page-131.png", (200, 1080, 1450, 1900)),
    "fig_4_28": ("page-132.png", (260, 230, 1450, 1010)),
}

OUR_FIGS = {
    "fig_4_26": OUT_DIR / "fig_4_26_ratio.png",
    "fig_4_27": OUT_DIR / "fig_4_27_O2.png",
    "fig_4_28": OUT_DIR / "fig_4_28_H.png",
}

FIG_TITLES = {
    "fig_4_26": "Fig 4.26 — c*_O₂ / c^b_O₂ vs V_RHE",
    "fig_4_27": "Fig 4.27 — O₂(x) profile at 2 voltages",
    "fig_4_28": "Fig 4.28 — H⁺(x) profile at 2 voltages",
}


def _ensure_pdf_pages() -> None:
    """Render PDF pages 131-132 at 200 dpi if not already present."""
    PDF_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    needed = [PDF_PAGES_DIR / "page-131.png", PDF_PAGES_DIR / "page-132.png"]
    if all(p.exists() for p in needed):
        return
    cmd = [
        "pdftoppm", "-r", "200", "-f", "131", "-l", "132",
        str(PDF_PATH), str(PDF_PAGES_DIR / "page"), "-png",
    ]
    subprocess.run(cmd, check=True)


def crop_jithin_figs() -> dict[str, Path]:
    """Crop each Jithin figure out of its page and save as a tight PNG."""
    _ensure_pdf_pages()
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
    """Stack all three comparisons in a 3-row × 2-col grid."""
    keys = ["fig_4_26", "fig_4_27", "fig_4_28"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
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
        "Jithin Thesis §4.8 — side-by-side reproduction\n"
        "(Cs⁺/SO₄²⁻ pH 2, L_eff=50 µm, C_S=1.738 F/m², "
        "A_Tafel=142 mV/dec, single-Tafel R2e)",
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
                  "_run_jithin_fig_4_26_28.py + _plot_jithin_fig_4_26_28.py first.",
                  file=sys.stderr)
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
