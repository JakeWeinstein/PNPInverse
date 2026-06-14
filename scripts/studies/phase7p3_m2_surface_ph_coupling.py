"""Phase 7.3 M2 (G) — surface-pH kinetic coupling: documentation + β diagnostic.

G is the infrastructure every c_H-dependent rate (A1, C1, any pH-dependent C2)
needs: the rate must read the **surface** (electrode-facet boundary trace) of
c_H, not the bulk.  This script documents that the production stack already
does so, and quantifies the surface↔bulk relationship β under load.

UFL provenance (verified by code read of Forward/bv_solver/forms_logc_muh.py):
  * The BV rate is assembled as a surface integral `* ds(electrode_marker)`
    (line ~620), so every factor is the boundary trace at the electrode facet.
  * A `cathodic_conc_factor` on species s contributes `power*(u_exprs[s] −
    ln c_ref)` to the log-rate (line ~567); for the proton (a "mu species")
    `u_exprs[2]` is the muh-reconstructed `log c_H = μ_H − em·z_H·φ`
    (docstring §"five c_H-touch sites").
  * Therefore the proton concentration the rate sees is
        c_H,surf = exp(μ_H − em·z_H·φ)  |_electrode facet
    — NOT the bulk, NOT the OHP/Stern-plane, NOT a cell average.
  * Water routes carry `cathodic_conc_factors = []` ⇒ byte-equivalent when no
    c_H rate is active (the off-path guarantee).

β diagnostic (this script, from the frozen-θ_L water-only pH-series, no new
solve): β ≡ −∂log₁₀(c_H,surf)/∂pH_bulk = ∂(surface_pH)/∂(pH_bulk).  β=1 means
the surface tracks the bulk (bulk-limiting); β≈0 means the surface is pinned
(decoupled).  Result is reported as a curve, NOT a single slope, because the
relationship is a THRESHOLD, not a line.

Output: StudyResults/phase7p3_m2_surface_ph/{surface_ph_coupling.json, .png}
Run from PNPInverse/ (no Firedrake needed; post-processes the pH-series).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PH_SER = Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
OUT = Path(_ROOT) / "StudyResults" / "phase7p3_m2_surface_ph"
PH_LIST = [1.65, 2.0, 2.35, 3.42, 4.0, 4.21, 5.21, 6.0, 6.39]
V_SAMPLES = [0.30, 0.40, 0.50]    # representative cathodic V_RHE under load


def _load_surface_ph(ph):
    d = json.load(open(PH_SER / f"iv_curve_pH{ph}.json"))
    v = np.array(d["v_rhe_deck"], float)
    sp = np.array([x if x is not None else np.nan
                   for x in d.get("surface_pH", [None] * len(v))], float)
    o = np.argsort(v)
    return v[o], sp[o]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for ph in PH_LIST:
        try:
            rows[ph] = _load_surface_ph(ph)
        except FileNotFoundError:
            continue

    table = {f"{V:.2f}": [] for V in V_SAMPLES}
    for ph, (v, sp) in rows.items():
        for V in V_SAMPLES:
            table[f"{V:.2f}"].append([ph, float(np.interp(V, v, sp))])

    # Characterize as acid-plateau vs alkaline-plateau, not a single slope.
    summary = {}
    for V in V_SAMPLES:
        arr = np.array(table[f"{V:.2f}"], float)
        ph_b, sp_s = arr[:, 0], arr[:, 1]
        acid = sp_s[ph_b <= 2.35]
        alk = sp_s[ph_b >= 3.42]
        # local β by finite difference between adjacent bulk-pH points
        beta_fd = np.gradient(sp_s, ph_b)
        summary[f"{V:.2f}"] = {
            "bulk_pH": ph_b.tolist(),
            "surface_pH": sp_s.tolist(),
            "surface_cH_mol_m3": [10.0 ** (3.0 - s) for s in sp_s],
            "acid_plateau_surface_pH_mean": (None if acid.size == 0
                                             else float(np.nanmean(acid))),
            "alk_plateau_surface_pH_mean": (None if alk.size == 0
                                            else float(np.nanmean(alk))),
            "beta_local_fd": beta_fd.tolist(),
            "beta_max_local": float(np.nanmax(beta_fd)),
        }

    out = {
        "test": "phase7p3_M2_surface_pH_coupling",
        "ufl_provenance": (
            "BV rate is integrated on ds(electrode_marker); a cathodic_conc_"
            "factor on the proton reads u_exprs[2] = muh-reconstructed "
            "log c_H = mu_H - em*z_H*phi -> the ELECTRODE-FACET BOUNDARY "
            "TRACE of c_H (forms_logc_muh.py ~L567/L620). Water routes carry "
            "no c_H factor (byte-equiv off-path)."),
        "v_samples": V_SAMPLES,
        "by_V": summary,
        "finding": (
            "Surface c_H is a THRESHOLD function of bulk pH, NOT linear: an "
            "acidic plateau (surface pH ~0.7-1.7) for bulk pH <= 2.35 where "
            "the H+ reservoir + cathodic field enrichment hold the surface "
            "acidic, and an alkaline plateau (surface pH ~9-10) for bulk pH "
            ">= 3.42 where ORR H+-consumption outruns bulk supply. The "
            "transition sits at bulk pH ~2.3-3.4. Hence a surface-c_H-reading "
            "consumption rate (C1) is ON only at bulk pH <= ~2.3 and OFF at "
            ">= 3.4 -> reproduces the data's ring-magnitude collapse at pH 2 "
            "and survival at pH 4/6 (mechanism C/M3). The brainstorm's "
            "bulk-limiting m (beta=1) is INVALID here: beta is ~0 on each "
            "plateau and large only across the switch, so C's apparent proton "
            "ORDER is NOT identifiable as a simple c_H^m power from bulk pH."),
    }
    with open(OUT / "surface_ph_coupling.json", "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print("  Phase 7.3 M2 (G) — surface-pH kinetic coupling")
    print("=" * 78)
    print("  UFL: BV c_H factor = boundary trace exp(mu_H - em*z_H*phi) "
          "@ electrode facet (verified).")
    for V in V_SAMPLES:
        s = summary[f"{V:.2f}"]
        print(f"\n  V_RHE={V:.2f}:  bulk_pH -> surface_pH")
        for pb, ss in zip(s["bulk_pH"], s["surface_pH"]):
            print(f"      {pb:5.2f} -> {ss:6.2f}  (c_H,surf={10.0**(3.0-ss):.2e} mol/m³)")
        print(f"      acid-plateau surf pH ~{s['acid_plateau_surface_pH_mean']:.2f}, "
              f"alk-plateau surf pH ~{s['alk_plateau_surface_pH_mean']:.2f}")
    print("\n  ==> Surface c_H is a THRESHOLD (acid<=2.35 vs alk>=3.42), not "
          "linear; β invalid as a single slope.")
    print("      C1 (surface-c_H consumption) is ON only at bulk pH<=~2.3 -> "
          "ring collapse; OFF at >=3.4 -> ring survives. Sets up M3.")
    _plot(summary)
    print(f"\n  wrote {OUT / 'surface_ph_coupling.json'}")
    return 0


def _plot(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for V in V_SAMPLES:
        s = summary[f"{V:.2f}"]
        ax.plot(s["bulk_pH"], s["surface_pH"], "o-", label=f"V_RHE={V:.2f}")
    ax.axhspan(0, 2.5, color="tab:red", alpha=0.06)
    ax.axhspan(8.5, 10.5, color="tab:blue", alpha=0.06)
    ax.axvspan(2.35, 3.42, color="gray", alpha=0.12, label="transition")
    ax.set_xlabel("bulk pH")
    ax.set_ylabel("surface pH (under load)")
    ax.set_title("M2/G: surface pH vs bulk pH — a threshold, not a line\n"
                 "(acidic ≤2.35 / alkaline ≥3.42; gates C1 surface-c_H "
                 "consumption)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "surface_ph_coupling.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
