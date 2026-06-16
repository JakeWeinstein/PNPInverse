"""Phase 7.2 generalization — frozen θ_L water-route model across pH.

PREDICTION (no refit): take the K₂SO₄ pH-6.39 accepted kinetics θ_L
(log f_2w=−1.009, log f_4w=−12.309, α_2w=0.577, α_4w=0.305; L_eff
21.7 µm; K⁺/SO₄²⁻) and run the model UNCHANGED at other bulk pH,
changing ONLY the bulk H⁺ concentration and the per-pH OCP shift.

Two deliberate choices (see campaign log):
  * water routes ONLY (acid k0=0) — a clean "same mechanism
    everywhere" prediction; low-pH deviations expose the acid /
    local-pH physics the frozen model omits.
  * per-pH OCP shift V_OCP = 0.664 + 0.059·pH (theoretical
    0.47 + 0.197 + 0.059·pH; the measured Ag/AgCl cal is only known
    for the 6.39 run — using the consistent theoretical form across
    pH, ~22 mV offset from the 6.39 fit's 1.019 V, documented).

Targets we have on disk to compare against:
  * Exp Info summary metrics (same electrode/cation/run) for pH
    {6.39, 5.21, 4.21, 3.42, 2.35, 1.65}: ring-onset potential,
    max ring current, peak H₂O₂ selectivity.
  * ACS deck 3-panel (pH 2/4/6/10/12) full ring/disk/selectivity
    curves (raster; compared qualitatively / by the same metrics).

Output: StudyResults/phase7p2_ph_series_generalization/
  iv_curve_pH{..}.json per pH + metrics.json (model onset/peak/sel
  vs Exp Info) + the run is plotted by the companion _plot script.

Run from PNPInverse/ in the firedrake venv (background; ~1 run/pH,
sequential).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# frozen θ_L kinetics (K₂SO₄ pH 6.39 accepted fit)
F2W, F4W = 10.0 ** -1.008699731156705, 10.0 ** -12.308926782786854
A2W, A4W = 0.5770144526758703, 0.304853786169721
L_EFF_UM = 21.7

# pH conditions: Exp Info series + 3-panel range (deduped)
PH_LIST = [1.65, 2.0, 2.35, 3.42, 4.0, 4.21, 5.21, 6.0, 6.39,
           10.0, 12.0]

# Exp Info summary metrics (V_RHE onset @ j_ring=0.01 mA/cm²_ring;
# max ring mA/cm²_ring; peak H₂O₂ %) — same electrode/cation/run.
EXP_INFO = {
    6.39: {"onset": 0.472, "max_ring": 0.355, "peak_sel": 73.2},
    5.21: {"onset": 0.458, "max_ring": 0.504, "peak_sel": 75.33},
    4.21: {"onset": 0.434, "max_ring": 0.353, "peak_sel": 91.2},
    3.42: {"onset": 0.417, "max_ring": 0.406, "peak_sel": 72.1},
    2.35: {"onset": 0.363, "max_ring": 0.416, "peak_sel": 47.6},
    1.65: {"onset": 0.251, "max_ring": 0.0791, "peak_sel": 19.9},
}

A_DISK_CM2, A_RING_CM2, N_COLL = 0.19635, 0.109956, 0.224
OUT = Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"


def v_ocp(ph):
    return 0.47 + 0.197 + 0.059 * ph


def _opts(ph):
    return SimpleNamespace(
        routes="water",
        k0_water_2e_factor=F2W, k0_water_4e_factor=F4W,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=A2W, alpha_water_4e=A4W,
        l_eff_um=L_EFF_UM, bulk_h_mol_m3=10.0 ** (3.0 - ph),
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_ocp_rhe=v_ocp(ph),
        v_grid_lo=-0.10, v_grid_hi=0.85,
        out_name=f"_phser_pH{ph}",
    )


def _metrics(report):
    v = np.array(report["v_rhe_deck"], float)
    pc = np.array([p if p is not None else np.nan
                   for p in report["pc_mA_cm2"]], float)
    cd = np.array([c if c is not None else np.nan
                   for c in report.get("cd_mA_cm2", [None] * len(v))],
                  float)
    # model ring current density (ring area) from peroxide production
    jr = -pc * N_COLL * A_DISK_CM2 / A_RING_CM2
    order = np.argsort(v)
    v, jr, pc, cd = v[order], jr[order], pc[order], cd[order]
    ok = np.isfinite(jr)
    v, jr, pc, cd = v[ok], jr[ok], pc[ok], cd[ok]
    # ring onset: most ANODIC V where jr crosses 0.01 mA/cm²_ring
    onset = None
    idx = np.where((jr[:-1] - 0.01) * (jr[1:] - 0.01) <= 0)[0]
    if len(idx):
        i = idx[-1]
        onset = float(v[i] + (0.01 - jr[i]) * (v[i + 1] - v[i])
                      / (jr[i + 1] - jr[i]))
    max_ring = float(np.nanmax(jr))
    # peak H₂O₂ selectivity (molecule fraction): R2/(R2+R4);
    # electron form pc & cd: R2∝|pc|, 2*R4∝|cd|-|pc| -> R4∝(|cd|-|pc|)/2
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.abs(pc)
        r4 = np.maximum(0.0, (np.abs(cd) - np.abs(pc)) / 2.0)
        sel = 100.0 * r2 / (r2 + r4)
    peak_sel = float(np.nanmax(sel[np.abs(cd) > 0.02])) \
        if np.any(np.abs(cd) > 0.02) else float("nan")
    return {"onset": onset, "max_ring": max_ring,
            "peak_sel": peak_sel,
            "curve": {"v_rhe": v.tolist(), "jr": jr.tolist(),
                      "cd": cd.tolist(), "pc": pc.tolist()}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    import scripts.studies.drivers.solver_demo_slide15_dual_pathway_cs as dp

    out = {"theta_L": {"f2w": F2W, "f4w": F4W, "a2w": A2W,
                       "a4w": A4W, "l_eff_um": L_EFF_UM},
           "convention": "water-only frozen; V_OCP=0.664+0.059·pH",
           "by_pH": {}}
    for ph in PH_LIST:
        t0 = time.time()
        print(f"=== pH {ph}  c_H={10.0**(3.0-ph):.3e} mol/m³  "
              f"V_OCP={v_ocp(ph):.3f} ===", flush=True)
        try:
            report = dp._run(_opts(ph))
        except Exception as exc:
            print(f"  pH {ph} FAILED: {exc}", flush=True)
            out["by_pH"][str(ph)] = {"error": str(exc)}
            continue
        nconv = report.get("n_converged", 0)
        m = _metrics(report)
        m["n_converged"] = nconv
        m["n_total"] = report.get("n_total")
        m["exp_info"] = EXP_INFO.get(ph)
        out["by_pH"][str(ph)] = m
        with open(OUT / f"iv_curve_pH{ph}.json", "w") as f:
            json.dump(report, f)
        on = m["onset"]
        print(f"  {nconv}/{report.get('n_total')} conv  "
              f"onset={on if on is None else round(on,3)}  "
              f"max_ring={m['max_ring']:.3f}  "
              f"peak_sel={m['peak_sel']:.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        with open(OUT / "metrics.json", "w") as f:
            json.dump(out, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
