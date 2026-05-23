"""Plot Cs⁺/SO₄²⁻ pH 4 K0=1e-15 model run vs the digitized slide-15
envelope (deck reference, also Cs⁺ pH 4).

Outputs ``writeups/HydrolysisMeetingPrep/cs_pH4_vs_slide15.png``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))


def main() -> int:
    eval_path = os.path.join(
        _ROOT, "StudyResults", "phase6b_cs_pH4_K0fit_vs_slide15",
        "eval_lambda_1p0000_cs_K0_1e-15.json",
    )
    if not os.path.exists(eval_path):
        print(f"[error] {eval_path} not found — run the simulation first")
        return 1
    with open(eval_path) as f:
        j = json.load(f)

    recs = [r for r in j["per_v_records"]
            if r.get("v_rhe") is not None
            and r.get("pc_gross_mA_cm2") is not None
            and r.get("snes_converged")]
    recs.sort(key=lambda r: float(r["v_rhe"]))

    v_model = np.array([float(r["v_rhe"]) for r in recs])
    pc_gross_model = np.array([float(r["pc_gross_mA_cm2"]) for r in recs])
    cd_model = np.array([float(r["cd_mA_cm2"]) for r in recs])
    h2o2_pct_model = np.array([
        float(r["gross_h2o2_pct"]) if r.get("gross_h2o2_pct") is not None else np.nan
        for r in recs
    ])

    # Digitized slide-15 envelope (Cs⁺ pH 4, eyeballed from figure;
    # peroxide current density vs V_RHE, cathodic = negative).
    v_slide = np.array([
        -0.45, -0.30, -0.15, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0
    ])
    pc_slide = np.array([
        -0.18, -0.20, -0.30, -0.45, -0.55, -0.50, -0.30, -0.10, -0.02, 0.0, 0.0
    ])

    # 2-panel figure: pc (slide-15 compat) + gross H₂O₂%
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Cs⁺/SO₄²⁻ pH 4 forward run at K0=1e-15 (corrected deck-fit), "
        "λ=1, Δβ=0  vs  slide-15 envelope",
        fontsize=13, fontweight="bold",
    )

    ax_pc, ax_sel = axes

    # === Panel A: peroxide current vs V (slide-15 comparison) ===
    ax_pc.plot(
        v_model, pc_gross_model,
        "o-", color="#1f4e79", lw=2.5, ms=7,
        label=r"Model: gross pc = $-I_\mathrm{scale} \cdot R_{2e}$  (K0=10⁻¹⁵, Cs⁺/SO₄²⁻)",
    )
    ax_pc.plot(
        v_slide, pc_slide,
        "--s", color="#b7202c", lw=2.5, ms=8,
        label="Deck slide-15 envelope (Cs⁺ pH 4, eyeballed)",
    )
    ax_pc.axhline(0, color="black", lw=0.5)
    ax_pc.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax_pc.set_ylabel("Peroxide current density (mA/cm²)\n(slide-15 convention: cathodic = negative)")
    ax_pc.set_title("(a) Gross peroxide current vs slide-15 envelope")
    ax_pc.grid(True, alpha=0.3)
    ax_pc.set_xlim(-0.5, 1.05)
    ax_pc.legend(loc="lower right", fontsize=10, framealpha=0.95)

    # === Panel B: total disk current AND gross H₂O₂% ===
    ax_sel.plot(
        v_model, cd_model,
        "o-", color="#2e7d32", lw=2.5, ms=7,
        label="Model: total disk current cd",
    )
    ax_sel.plot(
        v_model, pc_gross_model,
        "o-", color="#1f4e79", lw=2.5, ms=7,
        label=r"Model: gross peroxide $-I_\mathrm{scale} \cdot R_{2e}$",
    )
    ax_sel.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax_sel.set_ylabel("Current density (mA/cm²)")
    ax_sel.set_title("(b) Model cd vs pc — peroxide as fraction of total")
    ax_sel.grid(True, alpha=0.3)
    ax_sel.set_xlim(-0.15, 1.0)
    ax_sel.axhline(0, color="black", lw=0.5)
    ax_sel.legend(loc="lower right", fontsize=10, framealpha=0.95)

    # Twin axis with H₂O₂%
    ax_sel2 = ax_sel.twinx()
    ax_sel2.plot(
        v_model, h2o2_pct_model,
        "^-", color="#c77700", lw=2.0, ms=6, alpha=0.8,
        label=r"gross H$_2$O$_2$% = 100·R$_{2e}$/(R$_{2e}$+R$_{4e}$)",
    )
    ax_sel2.set_ylabel(r"gross H$_2$O$_2$% (right axis)", color="#c77700")
    ax_sel2.tick_params(axis="y", labelcolor="#c77700")
    ax_sel2.set_ylim(0, 100)
    # Deck mean 50.95
    ax_sel2.axhline(50.95, color="#c77700", ls=":", lw=1.5, alpha=0.7)
    ax_sel2.text(
        0.7, 53, "deck K@pH4 mean (50.95%)", color="#c77700",
        fontsize=9, ha="left",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_dir = Path(_ROOT) / "writeups" / "HydrolysisMeetingPrep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cs_pH4_vs_slide15.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")

    # Print summary
    print()
    print("Summary:")
    print(f"  n converged V points: {len(recs)}")
    print(f"  pc_gross range: [{pc_gross_model.min():+.4f}, {pc_gross_model.max():+.4f}] mA/cm²")
    print(f"  cd range: [{cd_model.min():+.4f}, {cd_model.max():+.4f}] mA/cm²")
    print(f"  H₂O₂% (filtered): [{np.nanmin(h2o2_pct_model):.1f}, {np.nanmax(h2o2_pct_model):.1f}]")
    print(f"  Slide-15 envelope peak: {pc_slide.min():+.2f} mA/cm² at V={v_slide[pc_slide.argmin()]:+.2f} V")
    print(f"  Model peak: {pc_gross_model.min():+.4f} mA/cm² at V={v_model[pc_gross_model.argmin()]:+.3f} V")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
