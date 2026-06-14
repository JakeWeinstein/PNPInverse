"""Phase 7.3 M3 — C1 smoke: mechanism + lock-preservation qualitative check.

Before any fitting, verify the two qualitative predictions that make C1
(electrochemical H₂O₂ reduction, reading SURFACE c_H) the lead mechanism for
the ring-magnitude collapse (plan §5):

  1. LOCK PRESERVED at pH 6.39 — surface c_H ≈ 10⁻⁷ mol/m³ (alkaline under
     load, M2/G) ⇒ C1 ≈ off ⇒ net ring ≈ the locked N0 ring. (G2 sub-gate.)
  2. RING COLLAPSE at pH 2 — surface c_H ≈ 10² mol/m³ (stays acidic) ⇒ C1
     consumes peroxide ⇒ net ring drops toward the data (~0.03–0.08 mA/cm²).

The model ring with C1 is the NET peroxide escape = gross 2e production −
C1 consumption, computed from the per-reaction rates (reaction_sum only
counts producers):
    net_peroxide_current = Σ_j stoich_H2O2[j] · rate_2e_units[j]
(recovers the gross `pc` when C1 is absent; subtracts C1's consumption).

This is a SMOKE (a few runs), not the fit. k0_c1_factor / p are swept
coarsely to show the ring responds and the lock holds.

Run from PNPInverse/ in the firedrake venv (background; ~3 runs):
    python -u scripts/studies/phase7p3_m3_c1_smoke.py
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

F2W, F4W = 10.0 ** -1.008699731156705, 10.0 ** -12.308926782786854
A2W, A4W = 0.5770144526758703, 0.304853786169721
L_EFF_UM = 21.7
A_DISK_CM2, A_RING_CM2, N_COLL = 0.19635, 0.109956, 0.224
STOICH_H2O2 = {"R2e_acid": +1, "R2e_water": +1, "R4e_acid": 0,
               "R4e_water": 0, "C1_h2o2_reduction": -1}

PH_SER = Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
BYTETEST = Path(_ROOT) / "StudyResults" / "phase7p3_p0_1_frame_byte_test"
DIGITIZED = PH_SER / "digitized_experimental_3panel.json"
OUT = Path(_ROOT) / "StudyResults" / "phase7p3_m3_c1_smoke"

# (pH, V_OCP, N0-reuse path) — pH 6.39 reuses the P0.1 byte-test RHE run.
# C1 with E°=1.765 V saturates (peroxide-transport-limited) at k0_factor~1;
# the c_H gating spans ~8.5 decades (pH2 factor ~720 vs pH6.39 ~2.1e-6), so a
# log-k0 sweep ~10⁻²²..10⁻¹⁰ brackets the window where C1 is OFF at pH 6.39
# (lock preserved) yet ON at pH 2 (ring collapses).  Full-range ring peak
# (the digitized ring peaks migrate to LOW V at low pH: pH2 @ V≈0.01).
CASES = [
    {"pH": 6.39, "v_ocp": 1.019, "bulk_h": 4.07e-4,
     "n0": BYTETEST / "iv_curve_rhe.json",
     "c1_factors": [1e-22, 1e-18, 1e-14, 1e-10], "v_lo": 0.0, "v_hi": 0.85},
    {"pH": 2.0, "v_ocp": 0.47 + 0.197 + 0.059 * 2.0, "bulk_h": 10.0,
     "n0": PH_SER / "iv_curve_pH2.0.json",
     "c1_factors": [1e-18, 1e-14], "v_lo": -0.10, "v_hi": 0.85},
]


def _opts(case, k0_c1_factor):
    return SimpleNamespace(
        routes="water,c1",
        k0_water_2e_factor=F2W, k0_water_4e_factor=F4W,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=A2W, alpha_water_4e=A4W,
        k0_c1_factor=k0_c1_factor, alpha_c1=None, c1_h_order=1.0,
        l_eff_um=L_EFF_UM, bulk_h_mol_m3=case["bulk_h"],
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_ocp_rhe=case["v_ocp"],
        v_grid_lo=case["v_lo"], v_grid_hi=case["v_hi"],
        proton_frame="rhe",
        out_name=f"_m3c1_pH{case['pH']}_k{k0_c1_factor}",
    )


def _net_ring(report):
    """disk=cd and ring=-net_peroxide·N·A_d/A_r on the reported axis."""
    v = np.array(report["v_rhe_deck"], float)
    cd = np.array([c if c is not None else np.nan
                   for c in report["cd_mA_cm2"]], float)
    per = report.get("per_reaction", [None] * len(v))
    netp = np.full(len(v), np.nan)
    for i, pr in enumerate(per):
        if not pr:
            continue
        netp[i] = sum(STOICH_H2O2.get(p["label"], 0)
                      * (p["rate_2e_units_mA_cm2"] or 0.0) for p in pr)
    o = np.argsort(v)
    v, cd, netp = v[o], cd[o], netp[o]
    ring = -netp * N_COLL * A_DISK_CM2 / A_RING_CM2
    return v, cd, ring


def _gross_ring(report):
    v = np.array(report["v_rhe_deck"], float)
    pc = np.array([p if p is not None else np.nan
                   for p in report["pc_mA_cm2"]], float)
    o = np.argsort(v)
    return v[o], (-pc[o] * N_COLL * A_DISK_CM2 / A_RING_CM2)


def _data_ring_peak(dig, ph, v_lo, v_hi):
    key = str(int(ph)) if str(int(ph)) in dig["ring"] else None
    if key is None:
        return None
    rv = np.array(dig["ring"][key]["v_rhe"]); rr = np.array(dig["ring"][key]["value"])
    m = (rv >= v_lo) & (rv <= v_hi)
    return float(np.nanmax(rr[m])) if m.any() else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
    dig = json.load(open(DIGITIZED)) if DIGITIZED.exists() else {"ring": {}}

    print("=" * 78, flush=True)
    print("  Phase 7.3 M3 — C1 smoke (mechanism + lock-preservation)", flush=True)
    print("=" * 78, flush=True)
    results = {"test": "phase7p3_M3_c1_smoke", "by_case": {}}

    for case in CASES:
        ph = case["pH"]
        rec = {"pH": ph, "v_ocp": case["v_ocp"]}
        # N0 baseline ring (gross == net; no C1)
        n0 = json.load(open(case["n0"])) if Path(case["n0"]).exists() else None
        if n0 is not None:
            vg, ring0 = _gross_ring(n0)
            win = (vg >= case["v_lo"]) & (vg <= case["v_hi"])
            rec["N0_ring_peak"] = float(np.nanmax(ring0[win])) if win.any() else None
        rec["data_ring_peak"] = _data_ring_peak(dig, ph, case["v_lo"], case["v_hi"])

        rec["c1"] = {}
        for kf in case["c1_factors"]:
            print(f"\n  pH {ph}  C1 k0_factor={kf} ...", flush=True)
            t0 = time.time()
            try:
                rep = dp._run(_opts(case, kf))
            except Exception as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
                rec["c1"][str(kf)] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            nc = rep.get("n_converged")
            print(f"    {nc}/{rep.get('n_total')} conv in {time.time()-t0:.0f}s",
                  flush=True)
            with open(OUT / f"iv_C1_pH{ph}_k{kf}.json", "w") as f:
                json.dump(rep, f)
            if not (rep.get("anchor_converged") and nc):
                rec["c1"][str(kf)] = {"anchor": rep.get("anchor_converged"),
                                      "n_converged": nc}
                continue
            v, cd, ring_net = _net_ring(rep)
            win = (v >= case["v_lo"]) & (v <= case["v_hi"])
            ring_peak = float(np.nanmax(ring_net[win])) if win.any() else None
            disk_peak = float(np.nanmin(cd[win])) if win.any() else None
            rec["c1"][str(kf)] = {
                "n_converged": nc,
                "net_ring_peak": ring_peak,
                "disk_min": disk_peak,
            }
            print(f"    net ring peak={ring_peak:.3f} (N0={rec.get('N0_ring_peak')}, "
                  f"data={rec.get('data_ring_peak')})", flush=True)
        results["by_case"][str(ph)] = rec
        with open(OUT / "c1_smoke.json", "w") as f:
            json.dump(results, f, indent=2)

    # verdict
    print("\n" + "=" * 78, flush=True)
    print("  C1 SMOKE VERDICT", flush=True)
    for ph_s, rec in results["by_case"].items():
        n0p = rec.get("N0_ring_peak")
        for kf, c in rec.get("c1", {}).items():
            if "net_ring_peak" not in c:
                continue
            rp = c["net_ring_peak"]
            if abs(float(ph_s) - 6.39) < 0.01:
                preserved = (n0p and abs(rp - n0p) / n0p < 0.10)
                print(f"  pH {ph_s} k={kf}: net ring {rp:.3f} vs N0 {n0p:.3f} "
                      f"-> lock {'PRESERVED' if preserved else 'BROKEN'}", flush=True)
            else:
                collapsed = (n0p and rp < 0.6 * n0p)
                print(f"  pH {ph_s} k={kf}: net ring {rp:.3f} vs N0 {n0p:.3f} "
                      f"(data {rec.get('data_ring_peak')}) -> "
                      f"{'COLLAPSES' if collapsed else 'no collapse'}", flush=True)
    with open(OUT / "c1_smoke.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {OUT / 'c1_smoke.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
