"""Shared styling + save helper for the group-meeting talk figures.

Pure matplotlib (no Firedrake). Figures are written to
``writeups/GroupMeetingTalk/figures/`` as both PNG (for slides) and PDF
(vector). Run each ``fig_*.py`` from the repo root, e.g.::

    MPLCONFIGDIR=/tmp ../venv-firedrake/bin/python \\
        scripts/studies/talk_figs/fig_s22_mms_rates.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
FIG_DIR = _REPO / "writeups" / "GroupMeetingTalk" / "figures"
STUDY = _REPO / "StudyResults"

# Palette — restrained, consistent across the deck.
INK = "#1b2433"      # near-black: text, primary lines
MUTE = "#6b7685"     # secondary text / guides
HERO = "#1f6feb"     # blue: the solver / "hero"
OK = "#2a9d54"       # green: converged / easy
FAIL = "#d1495b"     # red: villain / failure / cliff
ACCENT = "#e08a1e"   # amber: highlights, H2O2
STERN = "#7b5ea7"    # purple: Stern / compact layer
GRID = "#d7dce3"

SPECIES = {
    "O2": "#1f6feb",
    "H2O2": "#e08a1e",
    "H": "#d1495b",
    "K": "#2a9d54",
    "SO4": "#6b7685",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 13,
            "font.family": "sans-serif",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "svg.fonttype": "none",
        }
    )


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    print("wrote", FIG_DIR / f"{name}.png")
    plt.close(fig)


# --- schematic drawing helpers -------------------------------------------
def schematic_ax(fig_w=9.0, fig_h=5.0, title=None):
    """A blank axes (no spines/ticks) on a unit square for box/arrow art."""
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig, ax


def box(ax, cx, cy, w, h, text, *, fc="white", ec=INK, tc=None, fs=12,
        lw=1.8, weight="normal"):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            fc=fc, ec=ec, lw=lw, zorder=3,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=(tc or INK), weight=weight, zorder=4)


def arrow(ax, p0, p1, *, color=INK, lw=2.2, style="-|>", ls="-", rad=0.0):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=2, shrinkB=2),
        zorder=2,
    )


def tint(hex_color, alpha_over_white=0.18):
    """Light tint of a palette color for box fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c * alpha_over_white + 255 * (1 - alpha_over_white)))
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"
