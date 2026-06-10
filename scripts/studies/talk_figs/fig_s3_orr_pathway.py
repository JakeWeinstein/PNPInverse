"""S3 — oxygen reduction: H2O2 is the product we want (parallel topology)."""
from __future__ import annotations

import matplotlib.pyplot as plt

import _style as S

S.setup()
fig, ax = S.schematic_ax(9.4, 4.8,
                         "Oxygen reduction: two competing pathways")

# O2 source
S.box(ax, 0.12, 0.52, 0.16, 0.20, r"$O_2$", fc=S.tint(S.HERO), ec=S.HERO,
      fs=18, weight="bold")

# 2e branch (up) -> H2O2  (the product we want)
S.arrow(ax, (0.21, 0.60), (0.52, 0.74), color=S.ACCENT, lw=2.6, rad=-0.18)
ax.text(0.35, 0.83, r"$+\,2H^+ +\,2e^-$", ha="center", color=S.ACCENT, fontsize=12)
S.box(ax, 0.66, 0.76, 0.30, 0.18, r"$H_2O_2$" + "\nthe product we want",
      fc=S.tint(S.ACCENT), ec=S.ACCENT, fs=13.5, weight="bold")
ax.text(0.66, 0.62, r"$E^\circ = 0.695$ V vs RHE", ha="center",
        color=S.ACCENT, fontsize=11)

# 4e branch (down) -> H2O  (the competing loss)
S.arrow(ax, (0.21, 0.44), (0.52, 0.28), color=S.OK, lw=2.6, rad=0.18)
ax.text(0.35, 0.20, r"$+\,4H^+ +\,4e^-$", ha="center", color=S.OK, fontsize=12)
S.box(ax, 0.66, 0.26, 0.30, 0.18, r"$2\,H_2O$" + "\n4e$^-$ — the loss",
      fc=S.tint(S.OK), ec=S.OK, fs=13.5, weight="bold")
ax.text(0.66, 0.12, r"$E^\circ = 1.23$ V vs RHE", ha="center",
        color=S.OK, fontsize=11)

# parasitic further reduction of H2O2 -> H2O
S.arrow(ax, (0.66, 0.66), (0.66, 0.36), color=S.MUTE, lw=1.4, ls=":", style="->")
ax.text(0.71, 0.51, "(parasitic\nloss)", ha="left", va="center",
        color=S.MUTE, fontsize=9)

ax.text(0.5, 0.005,
        "This work: a better electrochemical route to H$_2$O$_2$ — the 2e$^-$ "
        "product.  Selectivity = keeping O$_2$ on the 2e$^-$ path, not the "
        "4e$^-$ path to water.",
        ha="center", va="bottom", fontsize=11, style="italic", color=S.INK)
S.save(fig, "s3_orr_pathway")
