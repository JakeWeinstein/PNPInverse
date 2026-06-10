"""Continuation — the three knobs we ramp (each step warm-started)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.setup()
fig, ax = plt.subplots(figsize=(11.0, 5.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Continuation: what we ramp")

# (y, name, start, end, note, color)
ramps = [
    (0.78, r"reaction rate $k_0$", r"$10^{-12}$", r"$1$ (physical)", "turn the reaction on gradually", S.HERO),
    (0.55, r"Stern capacitance $C_S$", r"$0.10$", r"$0.20\ \mathrm{F/m^2}$", "stiffen the compact layer", S.STERN),
    (0.32, r"voltage $V_{\mathrm{RHE}}$", r"$-0.5$", r"$+1.0\ \mathrm{V}$", "walk the operating window", S.ACCENT),
]
x0, x1 = 0.40, 0.86
for y, name, lo, hi, note, col in ramps:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
    for xx in np.linspace(x0, x1, 6):
        ax.scatter([xx], [y], s=46, color=col, edgecolor=S.INK, zorder=3, lw=0.6)
    ax.text(0.36, y + 0.005, name, ha="right", va="center", fontsize=12.5,
            weight="bold", color=col)
    ax.text(x0 - 0.005, y + 0.058, lo, ha="center", va="center", fontsize=10.5, color=S.INK)
    ax.text(x1 + 0.005, y + 0.058, hi, ha="center", va="center", fontsize=10.5, color=S.INK)
    ax.text((x0 + x1) / 2, y - 0.06, note, ha="center", va="center", fontsize=10,
            color=S.MUTE, style="italic")

ax.text(0.5, 0.11,
        "every step is warm-started from the previous converged solution",
        ha="center", fontsize=11.5, weight="bold", color=S.INK)
S.save(fig, "continuation_ramps")
