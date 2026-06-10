"""S11 — the full boundary-value problem on the domain.

Shows exactly what is modeled everywhere: the PNP-BV system solved in the
diffuse domain, the Bikerman counterion closure, and every boundary
condition (Stern Robin BC + Butler-Volmer flux at the electrode; Dirichlet
at the bulk).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as mp

import _style as S

S.setup()
fig, ax = plt.subplots(figsize=(12.4, 6.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# --- meshed diffuse domain (the box) ---
dx0, dx1, dy0, dy1 = 0.22, 0.80, 0.36, 0.70
ax.add_patch(mp.FancyBboxPatch((dx0, dy0), dx1 - dx0, dy1 - dy0,
             boxstyle="round,pad=0.004,rounding_size=0.008",
             fc="#f3f7fe", ec=S.HERO, lw=1.8))
ax.text((dx0 + dx1) / 2, dy1 - 0.045,
        "Diffuse domain — the PNP–BV system is solved here",
        ha="center", va="top", fontsize=12.5, weight="bold", color=S.HERO)
ax.text((dx0 + dx1) / 2, (dy0 + dy1) / 2 - 0.005,
        r"$\partial_t c_i = \nabla\!\cdot\!\left(D_i\left[\nabla c_i + (z_i F/RT)\,c_i\,\nabla\varphi\right]\right)$"
        "\n"
        r"$-\nabla\!\cdot(\varepsilon\nabla\varphi) = \sum_i z_i F\,c_i$",
        ha="center", va="center", fontsize=12.5, color=S.INK)
ax.text((dx0 + dx1) / 2, dy0 + 0.035,
        r"counterions $K^+/SO_4^{2-}$ : analytic Bikerman $\to$ Poisson source",
        ha="center", va="bottom", fontsize=9.5, color=S.MUTE)

# --- electrode + Stern strip at the left (outside the mesh) ---
ax.add_patch(mp.Rectangle((0.05, dy0), 0.045, dy1 - dy0, fc="#b8c0cc", ec=S.INK, lw=1.4))
ax.text(0.0725, (dy0 + dy1) / 2, "electrode", rotation=90, ha="center", va="center",
        fontsize=9, weight="bold")
ax.add_patch(mp.Rectangle((0.095, dy0), 0.125, dy1 - dy0, fc=S.tint(S.STERN),
             ec=S.STERN, lw=1.2, hatch="///"))
ax.text(0.1575, dy0 - 0.105, "Stern\n(not meshed)", ha="center", va="top",
        fontsize=8.5, color=S.STERN)

# --- boundaries ---
ax.plot([dx0, dx0], [dy0, dy1], color=S.INK, lw=1.6)
ax.plot([dx1, dx1], [dy0, dy1], ls="--", color=S.INK, lw=1.6)
ax.text(dx0, dy0 - 0.03, "electrode / OHP\n$x=0$", ha="center", va="top", fontsize=8.5, color=S.MUTE)
ax.text(dx1, dy0 - 0.03, "bulk\n$x = L \\approx 100\\,\\mu$m", ha="center", va="top", fontsize=8.5, color=S.MUTE)


def callout(xytext, text, xy, color, fs=10.5):
    ax.annotate(text, xy=xy, xytext=xytext, ha="center", va="center", fontsize=fs,
                color=color, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


# Stern Robin BC (upper-left)
callout((0.30, 0.90),
        r"Stern: Robin BC on $\varphi$" "\n" r"$\varepsilon\,\partial_n\varphi = C_S\,(V_{\mathrm{app}}-\varphi_{\mathrm{OHP}})$",
        (dx0, dy1 - 0.04), S.STERN)
# Butler-Volmer flux BC (lower-left) — explicit rate law, "PNP Equation Formulations.tex" form
callout((0.46, 0.11),
        r"Electrode: Butler–Volmer flux  (O$_2$, H$_2$O$_2$, H$^+$)" "\n"
        r"$-D_i\left(\nabla c_i + (z_i F/RT)\,c_i\,\nabla\varphi\right)\cdot\mathbf{n} = \sum_r \nu_{i,r}\,R_r$" "\n"
        r"$R_r = k_{0,r}\left[c_O\,e^{-\alpha_r F\eta_r/RT} - c_R\,e^{(1-\alpha_r)F\eta_r/RT}\right]$" "\n"
        r"$\eta_r = \varphi_m - \varphi_s - E_{\mathrm{eq},r}$",
        (dx0, dy0 + 0.04), S.ACCENT, fs=9.5)
# Bulk Dirichlet (upper-right)
callout((0.74, 0.90),
        r"Bulk: Dirichlet" "\n" r"$c_i = c_i^{\mathrm{bulk}},\ \ \varphi = 0$",
        (dx1, dy1 - 0.04), S.HERO)

ax.set_title("The full model on the domain: governing equations + every boundary condition")
S.save(fig, "double_layer_model")
