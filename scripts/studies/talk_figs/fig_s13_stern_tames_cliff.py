"""S13 — the Stern compact layer tames the Frumkin cliff.

Overlays total disk current density vs V_RHE for the production Stern
(C_S = 0.20 F/m^2, smooth) against near-no-Stern (C_S = 100 F/m^2, which
exposes the Frumkin cliff), at the deck-like k0_R4e/k0_R2e factor 1e-18.
Data are read from the existing solver_demo runs (no new solve).
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.setup()
FACTOR = "1e-18"


def load(sub):
    d = json.load(open(S.STUDY / sub / f"factor_{FACTOR}" / "iv_curve.json"))
    return np.array(d["v_rhe"]), np.array(d["cd_mA_cm2"])


vn, cdn = load("solver_demo_slide15_no_speculative_cs_noStern")
vw, cdw = load("solver_demo_slide15_no_speculative_cs")

fig, ax = plt.subplots(figsize=(8.4, 5.1))
ax.grid(False)
ax.grid(axis="y", color=S.GRID)

ax.plot(vn, cdn, "-o", color=S.FAIL, lw=2.4, ms=5,
        label=r"$C_S=100\ \mathrm{F/m^2}$  (no compact layer)")
ax.plot(vw, cdw, "-o", color=S.HERO, lw=2.4, ms=5,
        label=r"$C_S=0.20\ \mathrm{F/m^2}$  (production Stern)")

# steepest point of the no-Stern curve = the cliff
icliff = 1 + int(np.argmax(np.abs(np.diff(cdn) / np.diff(vn))))
ax.annotate(
    "Frumkin cliff:\nnear-discontinuous in V\n→ Newton chokes cold",
    xy=(vn[icliff], cdn[icliff]), xytext=(-0.17, -0.047),
    ha="center", va="center", color=S.FAIL, fontsize=10.5,
    arrowprops=dict(arrowstyle="->", color=S.FAIL, lw=1.7),
)
ax.annotate(
    "production Stern stays\nsmooth & convergent",
    xy=(0.50, float(np.interp(0.50, vw, cdw))), xytext=(0.30, -0.052),
    ha="center", va="center", color=S.HERO, fontsize=10.5,
    arrowprops=dict(arrowstyle="->", color=S.HERO, lw=1.7),
)

ax.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
ax.set_ylabel(r"total disk current  $j_{\mathrm{disk}}$ (mA/cm$^2$)")
ax.set_title("The Stern compact layer tames the Frumkin cliff")
ax.legend(loc="upper left")
S.save(fig, "s13_stern_tames_cliff")
