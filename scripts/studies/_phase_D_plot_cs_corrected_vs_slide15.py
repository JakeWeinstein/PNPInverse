"""Plot Cs⁺ corrected-a bridge result against slide-15 (and uncorrected baseline)."""
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


def load_bridge(path: str):
    with open(path) as f:
        d = json.load(f)
    v = np.array(d["v_rhe"])
    pc = np.array([x if x is not None else np.nan for x in d["pc_mA_cm2"]])
    cd = np.array([x if x is not None else np.nan for x in d["cd_mA_cm2"]])
    ring = np.array([x if x is not None else np.nan for x in d["j_ring_mA_cm2"]])
    sel = np.array([x if x is not None else np.nan for x in d["S_H2O2_percent"]])
    pH = np.array([x if x is not None else np.nan for x in d["surface_pH_proxy"]])
    return {"v": v, "pc": pc, "cd": cd, "ring": ring, "sel": sel, "pH": pH}


def load_phase_D():
    with open("StudyResults/phase6b_step10_phase_D/eval_db_0p0_stern_baseline.json") as f:
        d = json.load(f)
    recs = [r for r in d["per_v_records"] if r["snes_converged"]]
    return (np.array([r["v_rhe"] for r in recs]),
            np.array([r["pc_mA_cm2"] for r in recs]),
            np.array([r["cd_mA_cm2"] for r in recs]))


def main():
    v_deck, j_deck = load_slide15()
    cs_corr = load_bridge("StudyResults/phase6b_step10_phase_D_bridge_corrected_a_Cs/iv_curve.json")
    cs_unco = load_bridge("StudyResults/phase6b_step10_phase_D_no_hydrolysis_bridge_Cs/iv_curve.json")
    v_D, pc_D, cd_D = load_phase_D()

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # ---- Top: peroxide current (slide-15 convention) ----
    ax = axes[0]
    ax.plot(v_deck, j_deck, "x-", color="#8e44ad", lw=1.8, ms=6, alpha=0.85,
            label="Deck slide-15 (digitised, Cs⁺ pH 4)")
    ax.plot(cs_corr["v"], cs_corr["pc"], "o-", color="#c0392b", lw=1.8, ms=6,
            label="Cs⁺ bridge, PHYSICAL a (r_HP=2.8 Å, r_O2=1.7, r_H2O2=2.0)")
    ax.plot(cs_unco["v"], cs_unco["pc"], "s--", color="#e67e22", lw=1.2, ms=4, alpha=0.7,
            label="Cs⁺ bridge, LEGACY A_DEFAULT (r≈14.9 Å)")
    ax.plot(v_D, -pc_D, "v:", color="#7f8c8d", lw=1.2, ms=4, alpha=0.6,
            label="Phase D Δ_β=0 (full 6β stack, K⁺ — for context)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(-1.10, 0.05)
    ax.set_ylabel("Peroxide j_per (mA/cm²)\n[cathodic = negative]")
    ax.set_title("Cs⁺ corrected-a bridge vs slide-15 (no cation hyd, no kw, Stern=0.20, k0_R4e=1e-14)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)

    # ---- Middle: total disk current ----
    ax = axes[1]
    ax.plot(cs_corr["v"], cs_corr["cd"], "o-", color="#c0392b", lw=1.8, ms=5,
            label="Cs⁺ corrected-a j_disk")
    ax.plot(cs_unco["v"], cs_unco["cd"], "s--", color="#e67e22", lw=1.2, ms=4, alpha=0.7,
            label="Cs⁺ legacy A_DEFAULT j_disk")
    ax.plot(v_D, cd_D, "v:", color="#7f8c8d", lw=1.2, ms=4, alpha=0.6,
            label="Phase D Δ_β=0 j_disk (K⁺, V10B + hyd)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Total disk j (mA/cm²)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    # ---- Bottom: surface pH proxy + selectivity ----
    ax = axes[2]
    ax.plot(cs_corr["v"], cs_corr["pH"], "o-", color="#16a085", lw=1.5, ms=5,
            label="surface pH proxy (corrected a)")
    ax.plot(cs_unco["v"], cs_unco["pH"], "s--", color="#16a085", lw=1.0, ms=3, alpha=0.6,
            label="surface pH proxy (legacy a)")
    ax.set_ylabel("surface pH proxy", color="#16a085")
    ax.tick_params(axis="y", labelcolor="#16a085")
    ax.axhline(4.0, color="#16a085", ls=":", lw=0.6, alpha=0.5)
    ax.text(0.6, 4.0, "bulk pH 4", fontsize=8, color="#16a085", va="bottom")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(cs_corr["v"], cs_corr["sel"], "^-", color="#8e44ad", lw=1.5, ms=5,
             label="Selectivity % (corrected a)")
    ax2.set_ylabel("H₂O₂ selectivity (%)", color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    ax2.set_ylim(0, 105)

    ax.set_xlabel("V vs RHE (V)")
    ax.set_xlim(-0.55, 1.05)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right",
              fontsize=8, frameon=True)

    plt.tight_layout()
    out = "StudyResults/phase6b_step10_phase_D_bridge_corrected_a_Cs/plot_vs_slide15.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    print(f"wrote {out}")

    # Key numerical findings
    print(f"\nCs⁺ corrected-a vs LEGACY a (max abs differences):")
    print(f"  cd:    Δ = {float(np.nanmax(np.abs(cs_corr['cd'] - cs_unco['cd']))):.4f} mA/cm²")
    print(f"  pc:    Δ = {float(np.nanmax(np.abs(cs_corr['pc'] - cs_unco['pc']))):.4f} mA/cm²")
    print(f"  sel:   Δ = {float(np.nanmax(np.abs(cs_corr['sel'] - cs_unco['sel']))):.4f} pp")
    print(f"  pH:    Δ = {float(np.nanmax(np.abs(cs_corr['pH'] - cs_unco['pH']))):.4f}")
    print(f"\nAt V = +0.10 V (slide-15 peak position):")
    i = int(np.argmin(np.abs(cs_corr['v'] - 0.10)))
    iD = int(np.argmin(np.abs(np.array(v_deck) - 0.10)))
    print(f"  Cs⁺ corrected-a: pc = {cs_corr['pc'][i]:.4f} mA/cm²")
    print(f"  slide-15:        pc = {j_deck[iD]:.4f} mA/cm²")
    print(f"  ratio (slide/model) = {abs(j_deck[iD] / cs_corr['pc'][i]):.1f}×")


if __name__ == "__main__":
    main()
