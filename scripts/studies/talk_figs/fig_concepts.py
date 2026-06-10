"""Small concept diagrams: forward/inverse map (S5/6), MMS idea (S21),
spectral-vs-FEM independence (S24)."""
from __future__ import annotations

import _style as S


def forward_inverse():
    fig, ax = S.schematic_ax(9.4, 4.2, "Forward map vs. the inverse problem")
    S.box(ax, 0.17, 0.62, 0.26, 0.26,
          "kinetic params\n" r"$[\log k_0,\ \alpha]$" "\n2e$^-$ and 4e$^-$",
          fc=S.tint(S.MUTE), ec=S.MUTE, fs=12)
    S.box(ax, 0.5, 0.62, 0.20, 0.20, "forward\nsolver", fc=S.tint(S.HERO),
          ec=S.HERO, fs=13, weight="bold")
    S.box(ax, 0.83, 0.62, 0.26, 0.26,
          "IV curve +\n" r"H$_2$O$_2$ % vs $V$", fc=S.tint(S.ACCENT),
          ec=S.ACCENT, fs=12)
    S.arrow(ax, (0.30, 0.62), (0.40, 0.62), color=S.HERO, lw=2.6)
    S.arrow(ax, (0.60, 0.62), (0.70, 0.62), color=S.HERO, lw=2.6)
    S.arrow(ax, (0.70, 0.40), (0.30, 0.40), color=S.FAIL, lw=2.4, ls="--")
    ax.text(0.5, 0.34, "inverse problem (the eventual goal)", ha="center",
            color=S.FAIL, fontsize=11.5, weight="bold")
    ax.text(0.5, 0.12, "This talk builds and verifies the forward arrow.",
            ha="center", style="italic", fontsize=11.5)
    S.save(fig, "s5_6_forward_inverse")


def mms_idea():
    fig, ax = S.schematic_ax(11.0, 3.6, "Method of manufactured solutions: a self-test")
    steps = [
        (0.13, "pick an exact\n" r"solution $u^*$"),
        (0.38, "plug into PDE\n" r"$\Rightarrow$ source $f$"),
        (0.63, "solve with $f$\n" r"$\Rightarrow u_h$"),
        (0.88, r"$\|u_h-u^*\|\to 0$" "\n" r"at rate $h^p$"),
    ]
    for i, (x, t) in enumerate(steps):
        last = i == len(steps) - 1
        S.box(ax, x, 0.55, 0.20, 0.30, t,
              fc=S.tint(S.OK if last else S.HERO),
              ec=S.OK if last else S.HERO, fs=11.5,
              weight="bold" if last else "normal")
        if i:
            S.arrow(ax, (steps[i - 1][0] + 0.10, 0.55), (x - 0.10, 0.55),
                    color=S.INK, lw=2.2)
    ax.text(0.5, 0.18, "If the measured rate matches theory, the discretization "
            "is implemented correctly.", ha="center", style="italic", fontsize=11.5)
    S.save(fig, "s21_mms_idea")


def spectral_vs_fem():
    fig, ax = S.schematic_ax(9.4, 4.2, "Two independent codes, same physics")
    S.box(ax, 0.22, 0.70, 0.34, 0.22,
          "Jithin (thesis)\nspectral / Chebyshev\nanalytic closure",
          fc=S.tint(S.MUTE), ec=S.MUTE, fs=11.5)
    S.box(ax, 0.22, 0.34, 0.34, 0.22,
          "Ours\nweak-form FEM\n(Firedrake)", fc=S.tint(S.HERO), ec=S.HERO,
          fs=11.5, weight="bold")
    S.box(ax, 0.78, 0.52, 0.30, 0.24,
          "same physical\nanswer\n→ cross-check", fc=S.tint(S.OK), ec=S.OK,
          fs=11.5, weight="bold")
    S.arrow(ax, (0.39, 0.66), (0.63, 0.56), color=S.INK, lw=2.2)
    S.arrow(ax, (0.39, 0.38), (0.63, 0.48), color=S.INK, lw=2.2)
    ax.text(0.5, 0.08, "Different discretizations agreeing is exactly what "
            "cross-validation should show.", ha="center", style="italic",
            fontsize=11)
    S.save(fig, "s24_spectral_vs_fem")


if __name__ == "__main__":
    S.setup()
    forward_inverse()
    mms_idea()
    spectral_vs_fem()
