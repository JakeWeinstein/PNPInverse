"""S13b — cold start fails; anchor+continuation succeeds.

Visualizes the documented outcome (solver_demo grid, multi-ion + Stern
stack): a naive cold Newton solve cannot traverse the window, while the
anchor + continuation orchestrator converges every voltage.
"""
from __future__ import annotations

import numpy as np

import _style as S

S.setup()
fig, ax = S.schematic_ax(9.6, 3.8, "Cold start fails; continuation succeeds")

vgrid = np.linspace(-0.4, 0.55, 13)


def to_x(v):
    return 0.22 + 0.70 * (v - vgrid[0]) / (vgrid[-1] - vgrid[0])


# voltage axis
S.arrow(ax, (0.18, 0.15), (0.97, 0.15), color=S.INK, lw=1.6)
ax.text(0.57, 0.05, r"$V_{\mathrm{RHE}}$  (−0.4 → +0.55 V)", ha="center", fontsize=11.5)
for v in (-0.4, 0.0, 0.55):
    ax.text(to_x(v), 0.105, f"{v:+.1f}", ha="center", fontsize=9, color=S.MUTE)

# Row 1: cold start — all fail
y1 = 0.70
ax.text(0.02, y1, "cold\nNewton\n13/13 fail", ha="left", va="center", fontsize=10.5,
        weight="bold", color=S.FAIL)
ax.scatter([to_x(v) for v in vgrid], [y1] * len(vgrid), marker="X", s=140,
           color=S.FAIL, edgecolor=S.INK, linewidth=0.8, zorder=3)

# Row 2: anchor + continuation — all converge
y2 = 0.42
ax.text(0.02, y2, "anchor +\ncontinuation\n25/25 ok", ha="left", va="center",
        fontsize=10.5, weight="bold", color=S.OK)
ax.scatter([to_x(v) for v in vgrid], [y2] * len(vgrid), marker="o", s=130,
           color=S.OK, edgecolor=S.INK, linewidth=0.8, zorder=3)

ax.text(0.59, 0.93, "(documented outcome — solver_demo grid)", ha="center",
        fontsize=10, color=S.MUTE)
S.save(fig, "s13_coldstart_map")
