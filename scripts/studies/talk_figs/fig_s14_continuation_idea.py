"""S14 — continuation: solve easy, deform into hard (the big idea)."""
from __future__ import annotations

import numpy as np

import _style as S

S.setup()
fig, ax = S.schematic_ax(9.6, 4.6, "Continuation: solve easy, deform into hard")

S.box(ax, 0.16, 0.66, 0.26, 0.24,
      "EASY problem\n\ntiny $k_0$, small $C_S$,\nfar from the cliff",
      fc=S.tint(S.OK), ec=S.OK, fs=11.5, weight="bold")
S.box(ax, 0.84, 0.66, 0.26, 0.24,
      "HARD target\n\nphysical $k_0$, $C_S{=}0.20$,\nfull V window",
      fc=S.tint(S.FAIL), ec=S.FAIL, fs=11.5, weight="bold")
S.arrow(ax, (0.30, 0.66), (0.70, 0.66), color=S.INK, lw=3.0)
ax.text(0.5, 0.74, "deform (continuation)", ha="center", fontsize=12.5,
        weight="bold", color=S.INK)
ax.text(0.5, 0.585, "ramp $k_0$  •  ramp $C_S$", ha="center", fontsize=11,
        color=S.MUTE)

# the path: warm-started dots from easy to hard
xs = np.linspace(0.18, 0.82, 9)
y = 0.30
for i, xx in enumerate(xs):
    c = S.OK if i == 0 else (S.FAIL if i == len(xs) - 1 else S.HERO)
    ax.scatter([xx], [y], s=130, color=c, edgecolor=S.INK, zorder=3)
    if i:
        S.arrow(ax, (xs[i - 1] + 0.012, y), (xx - 0.012, y), color=S.MUTE,
                lw=1.4, style="->")
ax.text(0.5, 0.40, "each step warm-started from the previous solution",
        ha="center", fontsize=11, color=S.INK)
ax.text(0.5, 0.16,
        "Never solve the hard problem cold — walk to it from one you can solve.",
        ha="center", fontsize=11.5, style="italic", color=S.INK)
S.save(fig, "s14_continuation_idea")
