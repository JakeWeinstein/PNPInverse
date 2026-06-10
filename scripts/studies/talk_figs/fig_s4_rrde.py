"""S4 — rotating ring-disk electrode (RRDE) schematic (top-down)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _style as S

S.setup()
fig, ax = plt.subplots(figsize=(7.6, 6.2))
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.25, 1.2)
ax.axis("off")
ax.set_title("Rotating ring–disk electrode (RRDE)")

# annulus: draw outer (ring) then gap then disk
ax.add_patch(mpatches.Circle((0, 0), 0.88, fc=S.tint(S.ACCENT), ec=S.ACCENT, lw=2))
ax.add_patch(mpatches.Circle((0, 0), 0.64, fc="white", ec=S.MUTE, lw=1.2))
ax.add_patch(mpatches.Circle((0, 0), 0.52, fc=S.tint(S.HERO), ec=S.HERO, lw=2))

ax.text(0, 0, "DISK\nORR\n(makes H$_2$O$_2$)", ha="center", va="center",
        fontsize=12, weight="bold", color=S.HERO)
ax.text(0, 0.76, "RING (Pt): detects H$_2$O$_2$", ha="center", va="center",
        fontsize=11, weight="bold", color=S.ACCENT)

# radial product flux: disk -> ring
for th in np.linspace(0, 2 * np.pi, 10, endpoint=False):
    x0, y0 = 0.53 * np.cos(th), 0.53 * np.sin(th)
    x1, y1 = 0.63 * np.cos(th), 0.63 * np.sin(th)
    S.arrow(ax, (x0, y0), (x1, y1), color=S.MUTE, lw=1.3, style="->")

# rotation arrow (curved, outside)
S.arrow(ax, (-0.62, 0.95), (0.62, 0.95), color=S.INK, lw=2.2, rad=-0.45)
ax.text(0, 1.12, r"$\omega$  (rotation)", ha="center", fontsize=12, weight="bold")

ax.text(0, -1.12,
        "O$_2$ reduced at the disk; H$_2$O$_2$ swept radially out and quantified\n"
        "at the ring.  Collection efficiency $N = 0.224$ (Ruggiero 2022).",
        ha="center", va="center", fontsize=10.5, style="italic")
S.save(fig, "s4_rrde_schematic")
