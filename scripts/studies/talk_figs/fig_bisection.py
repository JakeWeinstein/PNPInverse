"""Continuation — how a failed step recovers (bisection), two mechanisms."""
from __future__ import annotations

import matplotlib.pyplot as plt

import _style as S

S.setup()
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.7))

# ---- Panel A: geometric (sqrt) midpoint retry on the k0 ladder ----
axL.set_xlim(0, 1)
axL.set_ylim(0, 1)
axL.axis("off")
axL.set_title("1 · failed rung → geometric-midpoint retry")
axL.text(0.5, 0.86,
         "step too big in log-space?\nhalve it (geometric midpoint), then go on",
         ha="center", fontsize=10.5, color=S.INK)
axL.annotate("", xy=(0.95, 0.50), xytext=(0.05, 0.50),
             arrowprops=dict(arrowstyle="->", color=S.INK, lw=1.5))
axL.text(0.5, 0.40, r"$k_0$  (log scale)", ha="center", fontsize=10.5, color=S.MUTE)
xa, xb, ya = 0.40, 0.82, 0.50
axL.scatter([xa], [ya], s=170, color=S.OK, edgecolor=S.INK, zorder=3)
axL.text(xa, 0.60, "last ok\n$10^{-3}$", ha="center", fontsize=10, color=S.OK)
axL.scatter([xb], [ya], s=200, marker="X", color=S.FAIL, edgecolor=S.INK, zorder=3)
axL.text(xb, 0.60, "target $1.0$\n(fails)", ha="center", fontsize=10, color=S.FAIL)
xm = (xa + xb) / 2
axL.scatter([xm], [ya], s=170, color=S.HERO, edgecolor=S.INK, zorder=3)
axL.annotate(r"insert $\sqrt{a\,b}\approx 0.03$" "\n" "retry (ok)",
             xy=(xm, ya), xytext=(xm, 0.24), ha="center", fontsize=10, color=S.HERO,
             arrowprops=dict(arrowstyle="->", color=S.HERO, lw=1.4))

# ---- Panel B: recursive voltage-step bisection across the cliff ----
axR.set_xlim(0, 1)
axR.set_ylim(0, 1)
axR.axis("off")
axR.set_title("2 · cliff: recursively halve the voltage step")
axR.text(0.5, 0.90, r"warm-walk: 8 substeps $\times$ depth-5 (up to 32$\times$ refine)",
         ha="center", fontsize=10.5, color=S.INK)
levels = [
    (0.72, [0.10, 0.78], "full step  (fails)"),
    (0.55, [0.10, 0.44, 0.78], "halve  (fails)"),
    (0.38, [0.10, 0.27, 0.44, 0.61, 0.78], "halve again  (ok)"),
]
for y, xs, lab in levels:
    axR.annotate("", xy=(0.80, y), xytext=(0.08, y),
                 arrowprops=dict(arrowstyle="-", color=S.MUTE, lw=1.0))
    for xx in xs:
        axR.scatter([xx], [y], s=58, color=S.HERO, edgecolor=S.INK, zorder=3)
    axR.text(0.84, y, lab, ha="left", va="center", fontsize=9.5, color=S.INK)
axR.text(0.5, 0.18,
         "each sub-step warm-started → walks across the near-discontinuity",
         ha="center", fontsize=10, color=S.MUTE, style="italic")

fig.suptitle("How a failed continuation step recovers", fontsize=15, weight="bold")
fig.subplots_adjust(top=0.80, wspace=0.12, left=0.04, right=0.98)
S.save(fig, "continuation_bisection")
