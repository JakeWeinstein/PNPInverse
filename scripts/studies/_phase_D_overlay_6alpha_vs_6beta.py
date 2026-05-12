"""Overlay slide-15 digitised data with two model runs:
   - Phase 6α multi-ion (no cation hydrolysis, kw on, V10A kinetics, Cs+, Stern=0.10)
   - Phase D Δ_β=0 (full 6β stack: cation hydrolysis + kw + V10B + K+, Stern=0.20)

Tests the user's recollection that 6α was qualitatively closer to deck.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_slide15():
    v, j = [], []
    with open("data/mangan_deck_p15_h2o2_current.csv") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or row[0].startswith("V_"):
                continue
            v.append(float(row[0]))
            j.append(float(row[1]))
    return np.array(v), np.array(j)


def load_6alpha(ratio_dir: str):
    with open(ratio_dir + "/iv_curve.json") as f:
        d = json.load(f)
    return (np.array(d["v_rhe"]), np.array(d["pc_mA_cm2"]),
            np.array(d["cd_mA_cm2"]), d["ratio"])


def load_phase_D():
    with open("StudyResults/phase6b_step10_phase_D/eval_db_0p0_stern_baseline.json") as f:
        d = json.load(f)
    recs = [r for r in d["per_v_records"] if r["snes_converged"]]
    v = np.array([r["v_rhe"] for r in recs])
    pc = np.array([r["pc_mA_cm2"] for r in recs])  # positive in this convention
    cd = np.array([r["cd_mA_cm2"] for r in recs])
    return v, -pc, cd  # flip to slide-15 sign convention


def main():
    v_deck, j_deck = load_slide15()
    v_6a18, pc_6a18, cd_6a18, _ = load_6alpha(
        "StudyResults/fast_realignment/fast_realignment_2026-05-08/mangan_full_grid/ratio_1e-18")
    v_6a24, pc_6a24, cd_6a24, _ = load_6alpha(
        "StudyResults/fast_realignment/fast_realignment_2026-05-08/mangan_full_grid/ratio_1e-24")
    v_D, pc_D, cd_D = load_phase_D()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)

    # ===== Top: peroxide current density =====
    ax = axes[0]
    ax.plot(v_deck, j_deck, "x-", color="#8e44ad", lw=1.8, ms=6, alpha=0.85,
            label="Deck slide-15 (digitised, Cs⁺ pH 4)")
    ax.plot(v_6a18, pc_6a18, "s-", color="#27ae60", lw=1.6, ms=5,
            label="Phase 6α multi-ion (Cs⁺/SO₄²⁻, ratio=1e-18, Stern=0.10, no cation-hyd)")
    ax.plot(v_6a24, pc_6a24, "v--", color="#16a085", lw=1.4, ms=4, alpha=0.6,
            label="Phase 6α multi-ion (ratio=1e-24, pure-2e Levich)")
    ax.plot(v_D, pc_D, "o-", color="#c0392b", lw=1.6, ms=5,
            label="Phase D Δ_β=0 (K⁺/SO₄²⁻, V10B + cation-hyd + Stern=0.20)")

    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0.10, ls=":", color="#8e44ad", alpha=0.5, lw=0.8)
    ax.axvline(0.45, ls=":", color="#8e44ad", alpha=0.5, lw=0.8)
    ax.text(0.10, -0.42, "deck peak\n~+0.10V", fontsize=8, ha="center", color="#8e44ad")
    ax.text(0.45, -0.05, "deck onset\n~+0.45V", fontsize=8, ha="left", color="#8e44ad")

    ax.set_xlim(-0.55, 1.05)
    ax.set_ylim(-1.10, 0.05)
    ax.set_ylabel("Peroxide Current Density (mA/cm²)\n[cathodic = negative]")
    ax.set_title("Slide-15 deck vs Phase 6α (no cation hyd) vs Phase D (full 6β stack)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.3, frameon=True)

    # ===== Bottom: total cathodic disk current =====
    ax = axes[1]
    ax.plot(v_6a18, cd_6a18, "s-", color="#27ae60", lw=1.6, ms=5,
            label="Phase 6α (ratio=1e-18) j_disk")
    ax.plot(v_6a24, cd_6a24, "v--", color="#16a085", lw=1.4, ms=4, alpha=0.6,
            label="Phase 6α (ratio=1e-24) j_disk")
    ax.plot(v_D, cd_D, "o-", color="#c0392b", lw=1.6, ms=5,
            label="Phase D Δ_β=0 j_disk")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(-0.55, 1.05)
    ax.set_xlabel("V vs RHE (V)")
    ax.set_ylabel("Total disk current density (mA/cm²)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.3, frameon=True)

    plt.tight_layout()
    out = "StudyResults/phase6b_step10_phase_D/overlay_slide15_vs_6alpha_vs_phase_D.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(f"wrote {out}")
    # Quick stats
    print("\nDeck slide-15 (digitised):")
    i_peak = np.argmin(j_deck)
    print(f"  peak |j_per| = {-j_deck[i_peak]:.3f} mA/cm² at V = {v_deck[i_peak]:+.3f} V")
    print(f"  cathodic-end (V=-0.40): j_per = {j_deck[np.argmin(np.abs(v_deck+0.40))]:.3f}")
    print("\nPhase 6α ratio=1e-18:")
    i_p = np.argmin(pc_6a18)
    print(f"  max |j_per| = {-pc_6a18[i_p]:.3f} mA/cm² at V = {v_6a18[i_p]:+.3f} V")
    print(f"  has volcano? max minus cathodic-end = {-pc_6a18[i_p] - (-pc_6a18[0]):.3f} mA/cm² (positive=volcano)")
    print("\nPhase D Δ_β=0:")
    i_p = np.argmin(pc_D)
    print(f"  max |j_per| = {-pc_D[i_p]:.3f} mA/cm² at V = {v_D[i_p]:+.3f} V")


if __name__ == "__main__":
    main()
