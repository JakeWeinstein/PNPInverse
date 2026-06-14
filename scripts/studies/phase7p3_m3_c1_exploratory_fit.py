"""Phase 7.3 M3 — EXPLORATORY C1 fit (NOT a defensible prediction).

⚠️  This tunes C1's k0 to the HELD-OUT pH-2 ring (0.069), which violates the
P0.3 held-out protocol on purpose — it answers a narrower question the user
asked: *can a single c_H-gated peroxide-reduction rate actually hit the pH-2
ring magnitude, and what does the DISK do when it does?*  The result is an
exploratory illustration, not a calibrated/transferable model and not a
prediction (C1 is unconstrained at pH 4/6, so k0 is pinned only by this
held-out point — see docs/phase7/phase7p3_summary.md "degenerate/conditional").

Procedure:
  1. pH 2: sweep k0_factor, measure the NET ring peak (gross − C1), and
     log-interpolate the k0* that lands it at the digitized pH-2 ring (0.069).
  2. Re-run pH 2 at k0* (refined), and run pH 4 & pH 6 at the SAME k0*.
  3. Compute disk + net-ring for all; overlay vs the digitized data and the
     C1-OFF baseline (N0).  Report the pH-2 disk (the discriminator) and
     whether pH 4/6 stay ≈ N0 (C1 off there).

Run from PNPInverse/ in the firedrake venv (background; ~8 runs):
    python -u scripts/studies/phase7p3_m3_c1_exploratory_fit.py
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
DIGITIZED = PH_SER / "digitized_experimental_3panel.json"
OUT = Path(_ROOT) / "StudyResults" / "phase7p3_m3_c1_exploratory_fit"

PH2_SWEEP_K0 = [1e-23, 1e-22, 1e-21, 1e-20, 1e-19]   # bracket the collapse
TARGET_RING_PH2 = 0.069                               # digitized pH-2 ring peak


def _v_ocp(ph):
    return 0.47 + 0.197 + 0.059 * ph


def _opts(ph, k0_c1_factor):
    return SimpleNamespace(
        routes="water,c1",
        k0_water_2e_factor=F2W, k0_water_4e_factor=F4W,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=A2W, alpha_water_4e=A4W,
        k0_c1_factor=k0_c1_factor, alpha_c1=None, c1_h_order=1.0,
        l_eff_um=L_EFF_UM, bulk_h_mol_m3=10.0 ** (3.0 - ph),
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_ocp_rhe=_v_ocp(ph),
        v_grid_lo=-0.10, v_grid_hi=0.85, proton_frame="rhe",
        out_name=f"_m3fit_pH{ph}",
    )


def _disk_ring(report):
    v = np.array(report["v_rhe_deck"], float)
    cd = np.array([c if c is not None else np.nan
                   for c in report["cd_mA_cm2"]], float)
    per = report.get("per_reaction", [None] * len(v))
    netp = np.full(len(v), np.nan)
    for i, pr in enumerate(per):
        if pr:
            netp[i] = sum(STOICH_H2O2.get(p["label"], 0)
                          * (p["rate_2e_units_mA_cm2"] or 0.0) for p in pr)
    o = np.argsort(v)
    v, cd, netp = v[o], cd[o], netp[o]
    ring = -netp * N_COLL * A_DISK_CM2 / A_RING_CM2
    return v, cd, ring


def _ring_peak(report):
    _, _, ring = _disk_ring(report)
    return float(np.nanmax(ring))


def _interp_k0_for_target(ring_by_k0, target):
    """Log-linear interpolate k0 where the net ring peak == target.
    ring decreases as k0 increases."""
    items = sorted(ring_by_k0.items())            # by k0 ascending
    lk = np.array([np.log10(k) for k, _ in items])
    rg = np.array([r for _, r in items])
    # find adjacent pair bracketing target (rg is descending in k0)
    for i in range(len(items) - 1):
        r_hi, r_lo = rg[i], rg[i + 1]             # r_hi >= r_lo (k ascending)
        if (r_hi - target) * (r_lo - target) <= 0 and r_hi != r_lo:
            frac = (r_hi - target) / (r_hi - r_lo)
            return 10.0 ** (lk[i] + frac * (lk[i + 1] - lk[i]))
    # not bracketed: clamp to nearest end
    return items[int(np.argmin(np.abs(rg - target)))][0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
    dig = json.load(open(DIGITIZED))

    print("=" * 78, flush=True)
    print("  Phase 7.3 M3 — EXPLORATORY C1 fit (tunes k0 to HELD-OUT pH-2 ring)",
          flush=True)
    print(f"  target pH-2 ring peak = {TARGET_RING_PH2}", flush=True)
    print("=" * 78, flush=True)

    out = {"warning": "EXPLORATORY — k0 tuned to held-out pH-2; NOT a prediction",
           "target_ring_pH2": TARGET_RING_PH2, "pH2_sweep": {}}

    # Stage 1: pH-2 k0 sweep
    ring_by_k0 = {}
    for kf in PH2_SWEEP_K0:
        print(f"\n  [sweep] pH 2  k0_factor={kf:g} ...", flush=True)
        t0 = time.time()
        try:
            rep = dp._run(_opts(2.0, kf))
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            continue
        if not (rep.get("anchor_converged") and rep.get("n_converged")):
            print("    not converged", flush=True)
            continue
        rp = _ring_peak(rep)
        ring_by_k0[kf] = rp
        out["pH2_sweep"][f"{kf:g}"] = rp
        print(f"    net ring peak={rp:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(OUT / "exploratory_fit.json", "w") as f:
            json.dump(out, f, indent=2)

    if len(ring_by_k0) < 2:
        print("  sweep failed to produce a curve; abort", flush=True)
        return 1
    k0_star = _interp_k0_for_target(ring_by_k0, TARGET_RING_PH2)
    out["k0_star_factor"] = k0_star
    print(f"\n  ==> interpolated k0*_factor = {k0_star:.3e} "
          f"(targets pH-2 ring {TARGET_RING_PH2})", flush=True)

    # Stage 2: run pH 2 (refined), 4, 6 at k0*
    out["fit_at_k0_star"] = {}
    curves = {}
    for ph in [2.0, 4.0, 6.0]:
        print(f"\n  [fit] pH {ph}  k0*={k0_star:.3e} ...", flush=True)
        t0 = time.time()
        try:
            rep = dp._run(_opts(ph, k0_star))
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            out["fit_at_k0_star"][str(ph)] = {"error": str(exc)}
            continue
        with open(OUT / f"iv_C1fit_pH{ph}.json", "w") as f:
            json.dump(rep, f)
        v, cd, ring = _disk_ring(rep)
        curves[ph] = (v, cd, ring)
        # N0 (C1 off) baseline from the pH-series
        n0 = json.load(open(PH_SER / f"iv_curve_pH{ph}.json"))
        vn = np.array(n0["v_rhe_deck"], float); on = np.argsort(vn)
        cdn = np.array([c if c is not None else np.nan
                        for c in n0["cd_mA_cm2"]], float)[on]
        pcn = np.array([p if p is not None else np.nan
                        for p in n0["pc_mA_cm2"]], float)[on]
        vn = vn[on]; ringn = -pcn * N_COLL * A_DISK_CM2 / A_RING_CM2
        # data
        k = str(int(ph))
        dv = np.array(dig["disk"][k]["v_rhe"]); dd = np.array(dig["disk"][k]["value"])
        rv = np.array(dig["ring"][k]["v_rhe"]); rr = np.array(dig["ring"][k]["value"])
        rec = {
            "ring_peak_fit": float(np.nanmax(ring)),
            "ring_peak_N0": float(np.nanmax(ringn)),
            "ring_peak_data": float(np.nanmax(rr)),
            "disk_min_fit": float(np.nanmin(cd)),
            "disk_min_N0": float(np.nanmin(cdn)),
            "disk_min_data": float(np.nanmin(dd)),
        }
        out["fit_at_k0_star"][str(ph)] = rec
        print(f"    ring peak: fit={rec['ring_peak_fit']:.3f} "
              f"N0={rec['ring_peak_N0']:.3f} data={rec['ring_peak_data']:.3f}",
              flush=True)
        print(f"    disk min : fit={rec['disk_min_fit']:.3f} "
              f"N0={rec['disk_min_N0']:.3f} data={rec['disk_min_data']:.3f}",
              flush=True)
        with open(OUT / "exploratory_fit.json", "w") as f:
            json.dump(out, f, indent=2)

    _plot(curves, dig, k0_star)
    print(f"\n  wrote {OUT / 'exploratory_fit.json'}", flush=True)
    return 0


def _plot(curves, dig, k0_star):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    phs = [2.0, 4.0, 6.0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for j, ph in enumerate(phs):
        k = str(int(ph))
        dv = np.array(dig["disk"][k]["v_rhe"]); dd = np.array(dig["disk"][k]["value"])
        rv = np.array(dig["ring"][k]["v_rhe"]); rr = np.array(dig["ring"][k]["value"])
        # N0
        n0 = json.load(open(PH_SER / f"iv_curve_pH{ph}.json"))
        vn = np.array(n0["v_rhe_deck"], float); on = np.argsort(vn)
        cdn = np.array([c if c is not None else np.nan for c in n0["cd_mA_cm2"]], float)[on]
        pcn = np.array([p if p is not None else np.nan for p in n0["pc_mA_cm2"]], float)[on]
        vn = vn[on]; ringn = -pcn * N_COLL * A_DISK_CM2 / A_RING_CM2
        axes[0][j].plot(dv, dd, "k-", lw=2, label="data")
        axes[0][j].plot(vn, cdn, "b--", label="N0 (C1 off)")
        axes[1][j].plot(rv, rr, "k-", lw=2, label="data")
        axes[1][j].plot(vn, ringn, "b--", label="N0 (C1 off)")
        if ph in curves:
            v, cd, ring = curves[ph]
            axes[0][j].plot(v, cd, "r-", label="C1 fit")
            axes[1][j].plot(v, ring, "r-", label="C1 fit")
        axes[0][j].set_title(f"pH {ph} — disk")
        axes[1][j].set_title(f"pH {ph} — ring")
        for r in (0, 1):
            axes[r][j].grid(alpha=0.3)
            if j == 0:
                axes[r][j].legend(fontsize=8)
            if r == 1:
                axes[r][j].set_xlabel("V_RHE (V)")
    axes[0][0].set_ylabel("disk mA/cm²")
    axes[1][0].set_ylabel("ring mA/cm²")
    fig.suptitle(f"EXPLORATORY C1 fit (k0*_factor={k0_star:.2e}, tuned to "
                 f"held-out pH-2 ring) — NOT a prediction", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "exploratory_3panel.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
