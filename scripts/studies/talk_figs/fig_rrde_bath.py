"""RRDE in the electrolyte bath — side-view cell schematic.

Companion to the top-down ``fig_s4_rrde``: shows the rotating ring-disk
electrode dipped into the electrolyte, the counter + reference electrodes,
and labels every species present in the bath (Ruggiero 2022 deck:
aqueous M2SO4 + H2SO4/MOH, O2-saturated).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

import _style as S

S.setup()
fig, ax = plt.subplots(figsize=(11.8, 7.4))
ax.set_aspect("equal")
ax.set_xlim(0, 1.46)
ax.set_ylim(0, 1.10)
ax.axis("off")

OHC = S.STERN  # OH- color (reuse Stern purple)

# ============================ electrolyte bath ============================
lx0, lx1, ly0, ly1 = 0.11, 0.99, 0.07, 0.66
ax.add_patch(mp.Rectangle((lx0, ly0), lx1 - lx0, ly1 - ly0,
             fc="#e9f2fb", ec="none", zorder=0))
# wavy liquid surface
xs = np.linspace(lx0, lx1, 240)
ax.plot(xs, ly1 + 0.007 * np.sin(2 * np.pi * xs / 0.07), color=S.HERO, lw=1.5, zorder=1)
# beaker glass walls (U shape, open top)
ax.plot([lx0 - 0.018, lx0 - 0.018, lx1 + 0.018, lx1 + 0.018],
        [ly1 + 0.07, ly0 - 0.018, ly0 - 0.018, ly1 + 0.07],
        color=S.MUTE, lw=2.6, zorder=1, solid_capstyle="round", solid_joinstyle="round")
ax.text(lx0 + 0.02, ly1 - 0.04, "electrolyte bath", ha="left", va="top",
        fontsize=11, style="italic", color=S.HERO, zorder=5)

# ============================ RRDE assembly ==============================
cx = 0.55
# insulating shaft
ax.add_patch(mp.FancyBboxPatch((cx - 0.045, 0.40), 0.09, 0.55,
             boxstyle="round,pad=0.002,rounding_size=0.012",
             fc="#cfd6df", ec=S.INK, lw=1.5, zorder=6))
# PTFE shroud (wider tip)
ax.add_patch(mp.FancyBboxPatch((cx - 0.12, 0.305), 0.24, 0.105,
             boxstyle="round,pad=0.002,rounding_size=0.012",
             fc="#e6e9ee", ec=S.INK, lw=1.5, zorder=6))
# active face: ring | disk | ring (concentric -> two ring segments in section)
fy = 0.305
ax.add_patch(mp.Rectangle((cx - 0.045, fy - 0.014), 0.09, 0.016,
             fc=S.tint(S.HERO), ec=S.HERO, lw=1.4, zorder=8))          # disk
for sx in (cx - 0.105, cx + 0.07):
    ax.add_patch(mp.Rectangle((sx, fy - 0.014), 0.035, 0.016,
                 fc=S.ACCENT, ec=S.ACCENT, lw=1.4, zorder=8))          # ring
# rotation
S.arrow(ax, (cx - 0.11, 0.905), (cx + 0.11, 0.905), color=S.INK, lw=2.3, rad=-0.5)
ax.text(cx, 0.965, r"$\omega$  (rotation)", ha="center", fontsize=12.5, weight="bold")

# disk / ring callouts
ax.annotate("carbon disk\nORR $\\to$ H$_2$O$_2$", xy=(cx, fy - 0.012),
            xytext=(0.275, 0.165), ha="center", fontsize=10, weight="bold",
            color=S.HERO, zorder=9,
            arrowprops=dict(arrowstyle="-|>", color=S.HERO, lw=1.6))
ax.annotate("Pt ring\ncollects H$_2$O$_2$", xy=(cx + 0.088, fy - 0.012),
            xytext=(0.83, 0.165), ha="center", fontsize=10, weight="bold",
            color=S.ACCENT, zorder=9,
            arrowprops=dict(arrowstyle="-|>", color=S.ACCENT, lw=1.6))

# convective flow: axial up to the disk, flung radially out toward the ring
S.arrow(ax, (0.50, 0.12), (0.535, 0.275), color=S.HERO, lw=1.6, style="-|>", rad=0.15)
S.arrow(ax, (0.61, 0.12), (0.565, 0.275), color=S.HERO, lw=1.6, style="-|>", rad=-0.15)
S.arrow(ax, (cx - 0.06, fy - 0.03), (0.36, 0.235), color=S.HERO, lw=1.6, style="-|>", rad=0.3)
S.arrow(ax, (cx + 0.06, fy - 0.03), (0.74, 0.235), color=S.HERO, lw=1.6, style="-|>", rad=-0.3)

# ============================ electrolyte ions ===========================
SP = {"O2": S.HERO, "H2O2": S.ACCENT, "H": S.FAIL, "K": S.OK, "SO4": S.MUTE, "OH": OHC}


def ion(x, y, sp, r=0.0115, z=4):
    ax.add_patch(mp.Circle((x, y), r, fc=SP[sp], ec="white", lw=0.7, zorder=z))


def labeled(x, y, sp, txt, dx=0.028, dy=0.0, ha="left"):
    ion(x, y, sp, r=0.017, z=5)
    ax.text(x + dx, y + dy, txt, ha=ha, va="center", fontsize=10.5,
            weight="bold", color=SP[sp], zorder=6)


# one labeled representative per species
labeled(0.175, 0.50, "H", r"H$^+$")
labeled(0.34, 0.56, "O2", r"O$_2$")
labeled(0.20, 0.38, "OH", r"OH$^-$")
labeled(0.33, 0.40, "K", r"K$^+$")
labeled(0.74, 0.40, "SO4", r"SO$_4^{2-}$", dx=0.028)
labeled(0.70, 0.52, "H2O2", r"H$_2$O$_2$")

# unlabeled "soup" for texture (majority carriers K+, SO4^2-)
rng = np.random.default_rng(3)
soup = [
    (0.30, 0.27, "SO4"), (0.16, 0.30, "SO4"),
    (0.385, 0.47, "K"), (0.20, 0.60, "SO4"), (0.31, 0.62, "K"),
    (0.78, 0.55, "K"), (0.83, 0.46, "SO4"), (0.92, 0.52, "K"),
    (0.90, 0.34, "SO4"), (0.78, 0.27, "K"), (0.70, 0.34, "SO4"),
    (0.83, 0.60, "H"), (0.66, 0.45, "H2O2"), (0.62, 0.22, "O2"),
    (0.50, 0.19, "O2"), (0.40, 0.22, "H"), (0.93, 0.20, "K"),
    (0.155, 0.43, "K"), (0.27, 0.49, "SO4"),
]
for x, y, sp in soup:
    ion(x, y, sp)

# ============================ species legend =============================
lgx = 1.045
ax.text(lgx, 0.905, "Species in the bath", fontsize=12.5, weight="bold", color=S.INK)
ax.plot([lgx, 1.45], [0.875, 0.875], color=S.GRID, lw=1.0)
rows = [
    ("O2",  r"O$_2$",        "dissolved reactant (O$_2$-saturated)"),
    ("H",   r"H$^+$",        "acidic medium; ORR co-reactant"),
    ("H2O2", r"H$_2$O$_2$",  "2e$^-$ product (swept to the ring)"),
    ("K",   r"K$^+$",        r"support cation; M$^+\!=$ Li/Na/K/Cs"),
    ("SO4", r"SO$_4^{2-}$",  r"support anion (K$_2$SO$_4$ + H$_2$SO$_4$)"),
    ("OH",  r"OH$^-$",       "from water / KOH (minor, acidic)"),
]
yy = 0.80
for sp, formula, role in rows:
    ax.add_patch(mp.Circle((lgx + 0.018, yy + 0.012), 0.015, fc=SP[sp], ec="white", lw=0.8))
    ax.text(lgx + 0.055, yy + 0.012, formula, fontsize=11, weight="bold",
            va="center", color=SP[sp])
    ax.text(lgx + 0.055, yy - 0.028, role, fontsize=8.3, color=S.MUTE, va="center")
    yy -= 0.092
# solvent row (no ion marker)
ax.text(lgx + 0.055, yy + 0.012, r"H$_2$O", fontsize=11, weight="bold", va="center", color=S.HERO)
ax.text(lgx + 0.055, yy - 0.028, "solvent (the bath itself)", fontsize=8.3, color=S.MUTE, va="center")

# ============================ caption + title ============================
ax.text(cx, 1.05, "The RRDE cell: rotating ring–disk electrode in the electrolyte bath",
        ha="center", va="center", fontsize=15, fontweight="bold")
ax.text((lx0 + lx1) / 2, 0.005,
        r"Aqueous K$_2$SO$_4$ (M$_2$SO$_4$, M$\in\{$Li,Na,K,Cs$\}$) $+$ H$_2$SO$_4$/KOH to set pH "
        r"($\approx$1–6) · O$_2$-saturated · $I\approx0.3$ M  (Ruggiero 2022)",
        ha="center", va="center", fontsize=9.5, style="italic", color=S.MUTE)

S.save(fig, "rrde_electrolyte_bath")
