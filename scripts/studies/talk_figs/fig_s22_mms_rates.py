"""S22 — MMS convergence rates for the production stack.

Grouped bar chart of the measured L2 and H1 convergence rates per field
(logc_muh + Cs+/SO4 multi-ion + Stern Robin), with theory reference lines
at 2.0 (L2) and 1.0 (H1). Data:
StudyResults/mms_logc_muh_multi_ion_stern/convergence_data.json.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.setup()
d = json.load(open(S.STUDY / "mms_logc_muh_multi_ion_stern" / "convergence_data.json"))
res = d["results"]
h = np.array(res["h"])

fields = [("u_O2", r"$u_{O_2}$"), ("u_H2O2", r"$u_{H_2O_2}$"),
          ("mu_H", r"$\mu_H$"), ("phi", r"$\varphi$")]


def rate(key):
    return float(np.polyfit(np.log(h), np.log(np.array(res[key])), 1)[0])


l2 = [rate(f + "_L2") for f, _ in fields]
h1 = [rate(f + "_H1") for f, _ in fields]
labels = [lab for _, lab in fields]
x = np.arange(len(fields))
w = 0.38

fig, ax = plt.subplots(figsize=(8.4, 5.1))
ax.grid(False)
ax.grid(axis="y", color=S.GRID)

b1 = ax.bar(x - w / 2, l2, w, color=S.HERO, label=r"$L^2$ rate")
b2 = ax.bar(x + w / 2, h1, w, color=S.ACCENT, label=r"$H^1$ rate")
ax.set_xlim(-0.75, len(fields) - 0.4)
ax.axhline(2.0, ls="--", color=S.HERO, lw=1.3)
ax.axhline(1.0, ls="--", color=S.ACCENT, lw=1.3)
ax.text(-0.68, 2.05, r"theory $L^2=2$", color=S.HERO,
        ha="left", va="bottom", fontsize=9.5)
ax.text(-0.68, 1.05, r"theory $H^1=1$", color=S.ACCENT,
        ha="left", va="bottom", fontsize=9.5)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("measured convergence rate")
ax.set_ylim(0, 2.4)
ax.set_title("MMS recovers textbook CG1 rates\n"
             r"(production stack: logc_muh + Cs$^+$/SO$_4^{2-}$ + Stern)")
ax.legend(loc="upper center", ncol=2)
S.save(fig, "s22_mms_rates")
