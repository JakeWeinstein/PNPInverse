"""S15-S17 — the continuation trajectory (defense in depth).

Three stages of the anchor+grid orchestrator:
  1. k0 AdaptiveLadder  — REAL ladder_history from the solver_demo anchor
     JSON, including the sqrt-midpoint failure recovery.
  2. Stern bump ladder  — the verified rung list 0.10 -> ... -> 100.
  3. grid walk + cliff warm-walk — recursive bisection across V ~ 0.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _style as S

S.setup()
anc = json.load(
    open(S.STUDY / "solver_demo_slide15_no_speculative_cs" / "factor_1e-18" / "iv_curve.json")
)["anchor"]
lh = anc["ladder_history"]  # [[k0, "ok"/"fail"], ...]

fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4))

# ---- Stage 1: k0 AdaptiveLadder (REAL data) ----
ax = axes[0]
ax.grid(False)
ax.grid(axis="y", color=S.GRID)
ks = [step[0] for step in lh]
status = [step[1] for step in lh]
xs = np.arange(len(lh))
ax.set_yscale("log")
ax.plot(xs, ks, "-", color=S.MUTE, lw=1.4, zorder=1)
for i, (k, st) in enumerate(zip(ks, status)):
    ax.scatter([i], [k], s=150, zorder=3,
               color=(S.OK if st == "ok" else S.FAIL),
               marker=("o" if st == "ok" else "X"),
               edgecolor=S.INK, linewidth=1.0)
fail_i = status.index("fail")
ax.annotate(r"jump to $k_0=1$ fails", xy=(fail_i, ks[fail_i]),
            xytext=(fail_i - 2.7, ks[fail_i] * 8), color=S.FAIL, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=S.FAIL, lw=1.5))
ax.annotate("insert geometric\nmidpoint, retry → ok",
            xy=(fail_i + 1, ks[fail_i + 1]),
            xytext=(fail_i - 3.0, ks[fail_i + 1] * 0.015), color=S.OK, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=S.OK, lw=1.5))
ax.set_xlabel("ladder step")
ax.set_ylabel(r"reaction rate $k_0$ (nondim)")
ax.set_xticks(xs)
ax.set_title(r"1 · $k_0$ continuation" "\n" r"(real ladder, $\sqrt{\,}$-midpoint recovery)")

# ---- Stage 2: Stern bump ladder ----
ax = axes[1]
ax.grid(False)
ax.grid(axis="y", color=S.GRID)
rungs = [0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0]
xs2 = np.arange(len(rungs))
ax.set_yscale("log")
ax.plot(xs2, rungs, "-o", color=S.STERN, lw=2.0, ms=8, mec=S.INK, zorder=2)
ax.scatter([0], [0.10], s=170, color=S.OK, zorder=4, edgecolor=S.INK,
           label="cold-built anchor")
ax.scatter([1], [0.20], s=240, color=S.HERO, marker="*", zorder=5, edgecolor=S.INK,
           label="production target")
ax.annotate("build cheap here", xy=(0, 0.10), xytext=(0.25, 0.028),
            color=S.OK, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=S.OK, lw=1.5))
ax.annotate("ramp $C_S$, reuse solver\n(no form rebuild)", xy=(4, rungs[4]),
            xytext=(1.3, 22), color=S.STERN, fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color=S.STERN, lw=1.5))
ax.set_xlabel("bump rung")
ax.set_ylabel(r"Stern capacitance $C_S$ (F/m$^2$)")
ax.set_xticks(xs2)
ax.set_xticklabels([f"{r:g}" for r in rungs], rotation=45)
ax.legend(loc="lower right")
ax.set_title("2 · Stern bump ladder" "\n" "(0.10 → 0.20 → … → 100)")

# ---- Stage 3: grid walk + cliff warm-walk (schematic) ----
ax = axes[2]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("3 · grid walk + cliff warm-walk" "\n" "(recursive bisection across V ≈ 0)")
y0 = 0.66
ax.annotate("", xy=(0.97, y0), xytext=(0.03, y0),
            arrowprops=dict(arrowstyle="->", color=S.INK, lw=1.6))
ax.text(0.5, y0 + 0.09, "voltage grid, warm-started point→point",
        ha="center", fontsize=10.5)
for xx in np.linspace(0.1, 0.9, 9):
    ax.scatter([xx], [y0], s=70, color=S.HERO, zorder=3, edgecolor=S.INK)
cliffx = 0.5
ax.add_patch(mpatches.FancyBboxPatch(
    (cliffx - 0.07, y0 - 0.03), 0.14, 0.06, boxstyle="round,pad=0.004",
    fc="none", ec=S.FAIL, lw=1.8))
ax.text(cliffx, y0 - 0.11, "Frumkin cliff", ha="center", color=S.FAIL, fontsize=10.5)
yb = 0.30
ax.annotate("", xy=(0.78, yb), xytext=(0.22, yb),
            arrowprops=dict(arrowstyle="->", color=S.MUTE, lw=1.2))
ax.text(0.5, yb + 0.07, "zoom: 8 substeps × depth-5\nbisection (up to 32× refine)",
        ha="center", fontsize=9.5, color=S.MUTE)
for xx in np.linspace(0.25, 0.75, 9):
    ax.scatter([xx], [yb], s=38, color=S.ACCENT, zorder=3, edgecolor=S.INK)
ax.plot([cliffx - 0.07, 0.22], [y0 - 0.03, yb + 0.04], ls=":", color=S.MUTE, lw=1.0)
ax.plot([cliffx + 0.07, 0.78], [y0 - 0.03, yb + 0.04], ls=":", color=S.MUTE, lw=1.0)
ax.text(0.5, 0.08, "result: 25/25 voltages converge",
        ha="center", fontsize=11.5, color=S.OK, fontweight="bold")

fig.suptitle("Defense in depth: never solve the hard problem cold — deform into it",
             fontsize=15, fontweight="bold")
fig.subplots_adjust(top=0.80, wspace=0.30, left=0.06, right=0.98, bottom=0.12)
S.save(fig, "s15_17_continuation_trajectory")
