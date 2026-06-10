"""S28 — cation hydrolysis at the interface (EXPLORATORY future direction).

Mechanism schematic only — no results, no fitted parameters. Frames the
direction the general FEM solver can reach but Jithin's analytic closures
cannot. Motivation traces to the group's reaction-modeling write-up.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _style as S

S.setup()
fig, ax = plt.subplots(figsize=(9.8, 5.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Future direction: cation hydrolysis at the interface (exploratory)")

# electrode (cathodic, negatively charged)
ax.add_patch(mpatches.Rectangle((0.0, 0.18), 0.09, 0.66, fc="#b8c0cc", ec=S.INK, lw=1.5))
ax.text(0.045, 0.51, "electrode", rotation=90, ha="center", va="center",
        fontsize=11, weight="bold")
for yy in np.linspace(0.26, 0.76, 6):
    ax.text(0.075, yy, "–", ha="center", va="center", fontsize=15, color=S.HERO,
            weight="bold")
# field arrows
for yy in (0.34, 0.51, 0.68):
    S.arrow(ax, (0.10, yy), (0.30, yy), color=S.HERO, lw=1.8, style="->")
ax.text(0.20, 0.80, "strong interfacial field $E$", ha="center", color=S.HERO,
        fontsize=11)

# hydrated cation near OHP
ax.add_patch(mpatches.Circle((0.45, 0.55), 0.055, fc=S.tint(S.OK, 0.5), ec=S.OK, lw=1.8))
ax.text(0.45, 0.55, r"$M^+$", ha="center", va="center", fontsize=13, weight="bold",
        color=S.OK)
for th in np.linspace(0, 2 * np.pi, 6, endpoint=False):
    ax.add_patch(mpatches.Circle((0.45 + 0.085 * np.cos(th), 0.55 + 0.085 * np.sin(th)),
                 0.018, fc="#cfe3ff", ec=S.HERO, lw=0.8))
ax.text(0.45, 0.40, "hydrated cation\nat the OHP", ha="center", va="top",
        fontsize=9.5, color=S.MUTE)

# reaction
S.arrow(ax, (0.53, 0.62), (0.66, 0.66), color=S.INK, lw=2.0)
ax.text(0.595, 0.71, "field-shifted\npK$_a$ (Singh 2016)", ha="center",
        fontsize=9.5, color=S.INK)
ax.text(0.80, 0.62, r"$M^+\!\cdot\!(H_2O)\ \rightarrow\ M\text{-}OH\ +\ H^+$",
        ha="center", va="center", fontsize=12, weight="bold")

# released proton -> local source
ax.add_patch(mpatches.Circle((0.80, 0.46), 0.030, fc=S.tint(S.FAIL, 0.5), ec=S.FAIL, lw=1.6))
ax.text(0.80, 0.46, r"$H^+$", ha="center", va="center", fontsize=11, weight="bold",
        color=S.FAIL)
S.arrow(ax, (0.80, 0.42), (0.80, 0.30), color=S.FAIL, lw=1.8, style="->")
ax.text(0.80, 0.26, "local proton source\n(could reshape selectivity)", ha="center",
        va="top", fontsize=9.5, color=S.FAIL)

ax.text(0.30, 0.20, r"surface coverage $\Gamma$ (Langmuir-capped)", ha="center",
        fontsize=10, color=S.OK)

# exploratory banner
ax.add_patch(mpatches.FancyBboxPatch((0.06, 0.025), 0.88, 0.085,
             boxstyle="round,pad=0.006,rounding_size=0.02",
             fc=S.tint(S.ACCENT, 0.35), ec=S.ACCENT, lw=1.8))
ax.text(0.5, 0.068, "EXPLORATORY · FUTURE WORK — the solver handles it "
        "numerically; the physics is not yet validated", ha="center", va="center",
        fontsize=11, weight="bold", color="#8a5200")
S.save(fig, "s28_hydrolysis_mechanism")
