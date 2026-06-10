"""S8 + S10 — model domain and the double layer (where the difficulty lives)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _style as S

S.setup()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.2))

# ============================ Panel A: domain ============================
axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
axA.axis("off")
axA.set_title("Quasi-1D diffusion layer")
# electrode
axA.add_patch(mpatches.Rectangle((0.0, 0.14), 0.07, 0.72, fc="#b8c0cc", ec=S.INK, lw=1.5))
axA.text(0.035, 0.5, "electrode", rotation=90, ha="center", va="center",
         fontsize=11, weight="bold")
# domain background
axA.add_patch(mpatches.Rectangle((0.07, 0.14), 0.86, 0.72, fc="#f4f7fb", ec=S.MUTE, lw=1.0))
# O2 profile (depleted at electrode)
x = np.linspace(0.07, 0.93, 100)
o2 = 0.78 - 0.34 * np.exp(-(x - 0.07) / 0.28)
axA.plot(x, o2, color=S.SPECIES["O2"], lw=2.6)
axA.text(0.9, 0.80, r"$c_{O_2}(x)$", color=S.SPECIES["O2"], fontsize=12, ha="right")
# species transported toward electrode
for yy, name, col in [(0.62, r"$O_2$", "O2"), (0.50, r"$H^+$", "H"),
                      (0.38, r"$H_2O_2$", "H2O2")]:
    S.arrow(axA, (0.62, yy), (0.20, yy), color=S.SPECIES[col], lw=2.0, style="->")
    axA.text(0.66, yy, name, color=S.SPECIES[col], fontsize=12, va="center")
axA.text(0.80, 0.26, r"counterions $K^+,\ SO_4^{2-}$", color=S.MUTE,
         fontsize=10.5, ha="center")
# axis labels
axA.text(0.07, 0.08, "electrode / OHP", ha="center", fontsize=10)
axA.text(0.93, 0.08, "bulk", ha="center", fontsize=10)
S.arrow(axA, (0.10, 0.10), (0.90, 0.10), color=S.INK, lw=1.2, style="<->")
axA.text(0.5, 0.04, r"$L_{\mathrm{eff}} \approx 100\ \mu$m", ha="center", fontsize=11)

# ===================== Panel B: double-layer zoom =====================
axB.set_xlim(0, 1)
axB.set_ylim(0, 1)
axB.axis("off")
axB.set_title("Compact + diffuse layer (the zoom)")
# metal
axB.add_patch(mpatches.Rectangle((0.0, 0.10), 0.10, 0.80, fc="#b8c0cc", ec=S.INK, lw=1.5))
axB.text(0.05, 0.5, "metal", rotation=90, ha="center", va="center", fontsize=10.5,
         weight="bold")
# Stern slab (hatched)
axB.add_patch(mpatches.Rectangle((0.10, 0.10), 0.12, 0.80, fc=S.tint(S.STERN),
              ec=S.STERN, lw=1.4, hatch="///"))
axB.text(0.16, 0.16, "Stern\n$C_S$", ha="center", va="center", color=S.STERN,
         fontsize=10.5, weight="bold")
# OHP line
axB.plot([0.22, 0.22], [0.10, 0.90], ls="--", color=S.INK, lw=1.6)
axB.text(0.22, 0.93, "OHP", ha="center", fontsize=10.5, weight="bold")
# potential profile phi(x): drop across Stern, exp decay in diffuse
xs = np.linspace(0.22, 0.92, 100)
phi = 0.55 + 0.18 * np.exp(-(xs - 0.22) / 0.16)
axB.plot([0.10, 0.22], [0.88, 0.73], color=S.FAIL, lw=2.6)   # linear drop across Stern
axB.plot(xs, phi, color=S.FAIL, lw=2.6)                       # diffuse decay
axB.text(0.105, 0.90, r"$V_{\mathrm{app}}$", color=S.FAIL, fontsize=11, ha="left")
axB.text(0.235, 0.74, r"$\varphi_{\mathrm{OHP}}$", color=S.FAIL, fontsize=11, ha="left")
axB.text(0.78, 0.60, r"$\psi_{\mathrm{diffuse}}(x)$", color=S.FAIL, fontsize=11)
# Bikerman-saturated counterion pile-up just past OHP
rng = np.random.default_rng(0)
for i in range(5):
    for j in range(4):
        cx = 0.245 + i * 0.028
        cy = 0.20 + j * 0.052
        axB.add_patch(mpatches.Circle((cx, cy), 0.013, fc=S.tint(S.OK, 0.5),
                      ec=S.OK, lw=1.0))
axB.text(0.45, 0.30, "counterions packed to\nBikerman saturation\n"
         r"$c_{\max}\approx 1/a$", ha="left", va="center", color=S.OK, fontsize=10.5)

fig.suptitle("Model domain & the double layer — where the numerical difficulty lives",
             fontsize=15, fontweight="bold")
fig.subplots_adjust(top=0.84, wspace=0.06, left=0.03, right=0.98, bottom=0.05)
S.save(fig, "s8_s10_domain_double_layer")
