"""Phase 7.3 M3 — is the pH-2 ring shape a COARSE-GRID artifact?

Re-runs pH 2 (N0 and C1 at the exploratory k0*) on a FINER V grid (25 pts over
a focused window [-0.05, 0.65] ≈ 0.029 V spacing, ~2.7× finer than the 13-pt
coarse grid over [-0.10, 0.85]) and overlays vs the coarse curves + data.

Question: does finer V-sampling (a) smooth/relocate the C1 "spike", and
(b) move the model volcano peak toward the data's low-V peroxide bump?
Expectation: the spike sharpens/smooths (it's an output-sampling feature) but
the volcano PEAK position (~0.30 V) does NOT move toward the data (~0.01 V) —
that offset is structural (too-anodic onset), not resolution.

Run from PNPInverse/ in the firedrake venv (background; 2 runs):
    python -u scripts/studies/phase7p3_m3_fine_grid_check.py
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
K0_C1_STAR = 1.2190220362539797e-21          # exploratory pH-2 fit
A_DISK_CM2, A_RING_CM2, N_COLL = 0.19635, 0.109956, 0.224
STOICH_H2O2 = {"R2e_acid": +1, "R2e_water": +1, "R4e_acid": 0,
               "R4e_water": 0, "C1_h2o2_reduction": -1}
V_OCP_PH2 = 0.47 + 0.197 + 0.059 * 2.0

PH_SER = Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
FIT = Path(_ROOT) / "StudyResults" / "phase7p3_m3_c1_exploratory_fit"
DIGITIZED = PH_SER / "digitized_experimental_3panel.json"
OUT = Path(_ROOT) / "StudyResults" / "phase7p3_m3_fine_grid_check"


def _opts(routes, k0_c1):
    return SimpleNamespace(
        routes=routes,
        k0_water_2e_factor=F2W, k0_water_4e_factor=F4W, k0_acid_4e_factor=1e-15,
        alpha_water_2e=A2W, alpha_water_4e=A4W,
        k0_c1_factor=k0_c1, alpha_c1=None, c1_h_order=1.0,
        l_eff_um=L_EFF_UM, bulk_h_mol_m3=10.0,
        enable_water_ionization=True, coarse_grid=False,    # 25 pts
        cation="k", v_ocp_rhe=V_OCP_PH2,
        v_grid_lo=-0.05, v_grid_hi=0.65, proton_frame="rhe",
        out_name="_finegrid",
    )


def _disk_ring(rep, net=True):
    v = np.array(rep["v_rhe_deck"], float)
    cd = np.array([c if c is not None else np.nan for c in rep["cd_mA_cm2"]], float)
    if net and rep.get("per_reaction"):
        per = rep["per_reaction"]
        r2 = np.full(len(v), np.nan)
        for i, pr in enumerate(per):
            if pr:
                r2[i] = sum(STOICH_H2O2.get(p["label"], 0)
                            * (p["rate_2e_units_mA_cm2"] or 0.0) for p in pr)
        ringsrc = r2
    else:
        ringsrc = np.array([p if p is not None else np.nan
                            for p in rep["pc_mA_cm2"]], float)
    o = np.argsort(v)
    return v[o], cd[o], -ringsrc[o] * N_COLL * A_DISK_CM2 / A_RING_CM2


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    import scripts.studies.drivers.solver_demo_slide15_dual_pathway_cs as dp

    runs = {}
    for tag, routes, k0c1 in [("N0_fine", "water", 1.0),
                              ("C1_fine", "water,c1", K0_C1_STAR)]:
        print(f"\n=== {tag} (25-pt fine grid, pH 2) ===", flush=True)
        t0 = time.time()
        rep = dp._run(_opts(routes, k0c1))
        print(f"  {rep.get('n_converged')}/{rep.get('n_total')} in "
              f"{time.time()-t0:.0f}s", flush=True)
        with open(OUT / f"iv_{tag}_pH2.json", "w") as f:
            json.dump(rep, f)
        runs[tag] = rep

    _plot(runs)
    # report peak positions
    summary = {}
    for tag, rep in runs.items():
        v, cd, ring = _disk_ring(rep, net=(tag == "C1_fine"))
        ip = int(np.nanargmax(ring))
        summary[tag] = {"ring_peak": float(ring[ip]), "ring_peak_V": float(v[ip]),
                        "n_pts": int(np.isfinite(ring).sum())}
    dig = json.load(open(DIGITIZED))
    rv = np.array(dig["ring"]["2"]["v_rhe"]); rr = np.array(dig["ring"]["2"]["value"])
    ip = int(np.argmax(rr))
    summary["data"] = {"ring_peak": float(rr[ip]), "ring_peak_V": float(rv[ip])}
    with open(OUT / "fine_grid_check.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  ring peak (mA/cm²) @ V_RHE:")
    for k, s in summary.items():
        print(f"    {k:10s}: {s['ring_peak']:.3f} @ V={s['ring_peak_V']:.3f}"
              + (f"  ({s.get('n_pts')} pts)" if "n_pts" in s else ""))
    print(f"\n  wrote {OUT / 'fine_grid_check.json'}", flush=True)
    return 0


def _plot(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    dig = json.load(open(DIGITIZED))
    rv = np.array(dig["ring"]["2"]["v_rhe"]); rr = np.array(dig["ring"]["2"]["value"])
    dv = np.array(dig["disk"]["2"]["v_rhe"]); dd = np.array(dig["disk"]["2"]["value"])
    # coarse refs
    n0c = json.load(open(PH_SER / "iv_curve_pH2.0.json"))
    c1c = json.load(open(FIT / "iv_C1fit_pH2.0.json"))
    vN0c, cdN0c, rN0c = _disk_ring(n0c, net=False)
    vC1c, cdC1c, rC1c = _disk_ring(c1c, net=True)
    vN0f, cdN0f, rN0f = _disk_ring(runs["N0_fine"], net=False)
    vC1f, cdC1f, rC1f = _disk_ring(runs["C1_fine"], net=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(dv, dd, "k-", lw=2, label="data")
    ax[0].plot(vN0c, cdN0c, "b--", alpha=.5, label="N0 coarse(13)")
    ax[0].plot(vN0f, cdN0f, "b-", label="N0 fine(25)")
    ax[0].plot(vC1c, cdC1c, "r--", alpha=.5, label="C1 coarse(13)")
    ax[0].plot(vC1f, cdC1f, "r-", label="C1 fine(25)")
    ax[0].set_title("pH 2 disk — coarse vs fine"); ax[0].set_ylabel("mA/cm²")
    ax[1].plot(rv, rr, "k-", lw=2, label="data")
    ax[1].plot(vN0c, rN0c, "b--", alpha=.5, label="N0 coarse(13)")
    ax[1].plot(vN0f, rN0f, "b-", label="N0 fine(25)")
    ax[1].plot(vC1c, rC1c, "r--", alpha=.5, label="C1 coarse(13)")
    ax[1].plot(vC1f, rC1f, "r-", label="C1 fine(25)")
    ax[1].set_title("pH 2 ring — coarse vs fine")
    for a in ax:
        a.set_xlabel("V_RHE (V)"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle("pH-2 grid-resolution check: does finer V-sampling change the "
                 "shape?", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fine_grid_check.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
