"""Phase 7.3 M1a — onset selection N0 / N1a / N1b / A (no G).

Tests whether the SHE-anchored first-ET hypothesis (A) is NEW PHYSICS or
merely a potential-FRAME choice, by scoring four frozen-θ_L water-route
predictions against the digitized disk+ring curves at pH 2/4/6:

  * N0  — RHE-flat status quo (the production model; OCP cancels in η).
  * N1a — pure voltage RELABEL of N0 by Δ(pH)=S·(pH−6.39) (anchored at 6.39).
          Ring magnitude invariant BY CONSTRUCTION. No solver run.
  * N1b — solver-side OCP shift V_OCP→V_OCP−Δ (η-invariant; exercises the
          Stern/diffuse-layer/surface fields). Tests NUMERICALLY whether an
          OCP/frame shift moves the ring magnitude. Solver run.
  * A   — SHE-anchored E_eq (E_eq += Δ; --proton-frame she), frozen k0/α.
          Curve shifts by Δ AND re-solves the kinetics. Solver run.

Decision (plan §3): credit A as physics ONLY if it beats the frame nulls
(N1a/N1b) on FULL disk+ring SHAPE residuals. A and N1a are gauge-equivalent
on onset alone (#9), so onset position cannot decide it. The frame-INVARIANT
signal is the RING MAGNITUDE — reported explicitly per variant.

θ_L (locked pH-6.39 fit), water routes only, L_eff 21.7, K⁺, coarse grid.
N0 is reused from StudyResults/phase7p2_ph_series_generalization/ when present
(identical config); only N1b + A are solved here (~6 runs).

Run from PNPInverse/ in the firedrake venv (background; ~25 min):
    python -u scripts/studies/phase7p3_m1a_onset_selection.py
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

# Locked θ_L (pH-6.39 dual-series accepted fit).
F2W, F4W = 10.0 ** -1.008699731156705, 10.0 ** -12.308926782786854
A2W, A4W = 0.5770144526758703, 0.304853786169721
L_EFF_UM = 21.7
PH_ANCHOR = 6.39
BULK_H_ANCHOR = 4.07e-4
A_DISK_CM2, A_RING_CM2, N_COLL = 0.19635, 0.109956, 0.224

PH_LIST = [2.0, 4.0, 6.0]
PH_SER = Path(_ROOT) / "StudyResults" / "phase7p2_ph_series_generalization"
DIGITIZED = PH_SER / "digitized_experimental_3panel.json"
OUT = Path(_ROOT) / "StudyResults" / "phase7p3_m1a_onset_selection"

# Score on the ORR-active overlap (avoid the extreme cathodic plateau and
# the near-OCP zero-current tail where digitization is least reliable).
SCORE_V_LO, SCORE_V_HI = 0.20, 0.75


def _v_ocp_base(ph):                       # = phase7p2_ph_series_generalization
    return 0.47 + 0.197 + 0.059 * ph


def _bulk_h(ph):
    return 10.0 ** (3.0 - ph)


def _delta(ph, S):                         # Nernstian frame shift, anchored
    return S * (ph - PH_ANCHOR)


def _base_opts(ph):
    return dict(
        routes="water",
        k0_water_2e_factor=F2W, k0_water_4e_factor=F4W,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=A2W, alpha_water_4e=A4W,
        l_eff_um=L_EFF_UM, bulk_h_mol_m3=_bulk_h(ph),
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_grid_lo=-0.10, v_grid_hi=0.85,
        out_name=f"_m1a_pH{ph}",
    )


def _opts(ph, variant, S):
    o = _base_opts(ph)
    if variant in ("N0", "N1a"):           # N1a is a post-hoc relabel of N0
        o.update(proton_frame="rhe", v_ocp_rhe=_v_ocp_base(ph))
    elif variant == "N1b":                 # solver-side OCP shift (η-invariant)
        o.update(proton_frame="rhe", v_ocp_rhe=_v_ocp_base(ph) - _delta(ph, S))
    elif variant == "A":                   # SHE-anchored E_eq
        o.update(proton_frame="she", bulk_h_anchor_mol_m3=BULK_H_ANCHOR,
                 v_ocp_rhe=_v_ocp_base(ph))
    else:
        raise ValueError(variant)
    return SimpleNamespace(**o)


def _observables(report):
    """disk=cd and ring=-pc·N·A_d/A_r on the reported V_RHE axis (sorted)."""
    v = np.array(report["v_rhe_deck"], float)
    cd = np.array([c if c is not None else np.nan
                   for c in report["cd_mA_cm2"]], float)
    pc = np.array([p if p is not None else np.nan
                   for p in report["pc_mA_cm2"]], float)
    o = np.argsort(v)
    v, cd, pc = v[o], cd[o], pc[o]
    ring = -pc * N_COLL * A_DISK_CM2 / A_RING_CM2
    return v, cd, ring


def _digit(dig, ph):
    key = str(int(ph))
    dv = np.array(dig["disk"][key]["v_rhe"]); dd = np.array(dig["disk"][key]["value"])
    rv = np.array(dig["ring"][key]["v_rhe"]); rr = np.array(dig["ring"][key]["value"])
    od = np.argsort(dv); orr = np.argsort(rv)
    return dv[od], dd[od], rv[orr], rr[orr]


def _rms_on_overlap(mv, my, dv, dy):
    """RMS(model−data) on the data's V grid within the scored window and the
    model's V span. Returns (rms, n)."""
    lo = max(SCORE_V_LO, mv.min(), dv.min())
    hi = min(SCORE_V_HI, mv.max(), dv.max())
    m = (dv >= lo) & (dv <= hi)
    if m.sum() < 3:
        return None, int(m.sum())
    mi = np.interp(dv[m], mv, my)
    r = mi - dy[m]
    return float(np.sqrt(np.mean(r ** 2))), int(m.sum())


def _score_variant(v, cd, ring, dig, ph):
    dv, dd, rv, rr = _digit(dig, ph)
    disk_rms, n_d = _rms_on_overlap(v, cd, dv, dd)
    ring_rms, n_r = _rms_on_overlap(v, ring, rv, rr)
    # ring peak height (frame-invariant magnitude signal)
    win = (v >= SCORE_V_LO) & (v <= SCORE_V_HI)
    model_ring_peak = float(np.nanmax(ring[win])) if win.any() else None
    dwin = (rv >= SCORE_V_LO) & (rv <= SCORE_V_HI)
    data_ring_peak = float(np.nanmax(rr[dwin])) if dwin.any() else None
    combined = (None if (disk_rms is None or ring_rms is None)
                else float(disk_rms + ring_rms))
    return {
        "disk_rms": disk_rms, "ring_rms": ring_rms,
        "combined_rms": combined, "n_disk": n_d, "n_ring": n_r,
        "model_ring_peak": model_ring_peak, "data_ring_peak": data_ring_peak,
        "ring_peak_resid": (None if (model_ring_peak is None
                                     or data_ring_peak is None)
                            else model_ring_peak - data_ring_peak),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
    S = dp._nernst_slope_v_per_ph()
    dig = json.load(open(DIGITIZED))

    print("=" * 78, flush=True)
    print(f"  Phase 7.3 M1a onset selection — S={S:.5f} V/pH, anchor pH {PH_ANCHOR}",
          flush=True)
    print("=" * 78, flush=True)

    results = {"S_v_per_ph": S, "anchor_pH": PH_ANCHOR,
               "score_window_v": [SCORE_V_LO, SCORE_V_HI], "by_pH": {}}

    for ph in PH_LIST:
        d_ph = {}
        curves = {}
        # --- N0: reuse pH-series if config matches, else run ---
        n0_cached = PH_SER / f"iv_curve_pH{ph}.json"
        rep_n0 = None
        if n0_cached.exists():
            r = json.load(open(n0_cached))
            c = r.get("config", {})
            if (c.get("l_eff_um") == L_EFF_UM and c.get("cation") == "k"
                    and c.get("routes") == "water"
                    and abs(float(c.get("k0_water_2e_factor", 0)) - F2W) < 1e-9
                    and r.get("n_converged") == r.get("n_total")):
                rep_n0 = r
                print(f"  pH {ph} N0: reused cached pH-series run "
                      f"({r['n_converged']}/{r['n_total']})", flush=True)
        if rep_n0 is None:
            print(f"  pH {ph} N0: running...", flush=True)
            t0 = time.time()
            rep_n0 = dp._run(_opts(ph, "N0", S))
            print(f"    N0 {rep_n0.get('n_converged')}/{rep_n0.get('n_total')} "
                  f"in {time.time()-t0:.0f}s", flush=True)
        curves["N0"] = _observables(rep_n0)

        # --- N1b, A: solve ---
        for variant in ("N1b", "A"):
            print(f"  pH {ph} {variant}: running...", flush=True)
            t0 = time.time()
            try:
                rep = dp._run(_opts(ph, variant, S))
            except Exception as exc:
                print(f"    {variant} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                d_ph[variant] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            nc = rep.get("n_converged")
            print(f"    {variant} {nc}/{rep.get('n_total')} in "
                  f"{time.time()-t0:.0f}s", flush=True)
            with open(OUT / f"iv_{variant}_pH{ph}.json", "w") as f:
                json.dump(rep, f)
            if rep.get("anchor_converged") and nc:
                curves[variant] = _observables(rep)

        # --- N1a: relabel N0 by Δ(pH) ---
        v0, cd0, ring0 = curves["N0"]
        dlt = _delta(ph, S)
        curves["N1a"] = (v0 + dlt, cd0, ring0)

        # --- score all available variants ---
        for variant, (v, cd, ring) in curves.items():
            sc = _score_variant(v, cd, ring, dig, ph)
            d_ph[variant] = sc
        results["by_pH"][str(ph)] = d_ph

        # console
        print(f"  --- pH {ph} scores (Δ={dlt:+.3f} V) ---", flush=True)
        for variant in ("N0", "N1a", "N1b", "A"):
            sc = d_ph.get(variant)
            if not sc or "disk_rms" not in sc:
                print(f"      {variant}: (no score)", flush=True)
                continue
            print(f"      {variant}: disk_rms={sc['disk_rms']:.3f} "
                  f"ring_rms={sc['ring_rms']:.3f} "
                  f"comb={sc['combined_rms']:.3f} "
                  f"ring_peak model={sc['model_ring_peak']:.3f} "
                  f"data={sc['data_ring_peak']:.3f}", flush=True)
        with open(OUT / "m1a_scores.json", "w") as f:
            json.dump(results, f, indent=2)

    _verdict(results)
    _plot(results, dig, S)
    with open(OUT / "m1a_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT / 'm1a_scores.json'}", flush=True)
    return 0


def _verdict(results):
    """A is physics only if it beats the frame nulls (N1a/N1b) on combined
    disk+ring shape; and report whether ANY variant matches the ring-peak
    magnitude trend (the frame-invariant signal)."""
    print("\n" + "=" * 78, flush=True)
    print("  M1a VERDICT", flush=True)
    print("=" * 78, flush=True)
    a_beats_null = []
    ring_unmatched = []
    for ph, d in results["by_pH"].items():
        def comb(k):
            return (d.get(k, {}) or {}).get("combined_rms")
        a, n1a, n1b = comb("A"), comb("N1a"), comb("N1b")
        nulls = [x for x in (n1a, n1b) if x is not None]
        if a is not None and nulls:
            best_null = min(nulls)
            beats = a < best_null - 1e-3
            a_beats_null.append(beats)
            print(f"  pH {ph}: A comb={a:.3f} vs best frame-null "
                  f"{best_null:.3f} -> A {'BEATS' if beats else 'ties/loses'}",
                  flush=True)
        # ring peak: does any variant reproduce the data peak within 25%?
        for variant in ("N0", "N1a", "N1b", "A"):
            sc = d.get(variant, {}) or {}
            mp, dp_ = sc.get("model_ring_peak"), sc.get("data_ring_peak")
            if mp is not None and dp_ is not None and dp_ > 0:
                if abs(mp - dp_) / dp_ <= 0.25:
                    break
        else:
            ring_unmatched.append(ph)
    a_is_physics = bool(a_beats_null) and all(a_beats_null)
    print(f"\n  A beats frame-nulls on shape at all pH: {a_is_physics}", flush=True)
    print(f"  pH where NO variant matches ring-peak magnitude (±25%): "
          f"{ring_unmatched}", flush=True)
    results["verdict"] = {
        "A_beats_frame_nulls_all_pH": a_is_physics,
        "ring_peak_unmatched_pH": ring_unmatched,
        "interpretation": (
            "A is gauge-equivalent to the frame nulls (onset is a frame "
            "question); the frame-invariant ring MAGNITUDE is unmatched by "
            "all frozen-θ_L frame variants -> mechanism C (M3) is the "
            "load-bearing new physics."
            if not a_is_physics else
            "A improves the full disk+ring shape beyond the frame nulls -> "
            "SHE-anchored kinetics carry shape information; investigate A1.")
    }
    print(f"  ==> {results['verdict']['interpretation']}", flush=True)


def _plot(results, dig, S):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(2, len(PH_LIST), figsize=(4 * len(PH_LIST), 8),
                             sharex=True)
    col = {"N0": "tab:blue", "N1a": "tab:green", "N1b": "tab:purple",
           "A": "tab:red"}
    for j, ph in enumerate(PH_LIST):
        dv, dd, rv, rr = _digit(dig, ph)
        for row, (obs, dvv, dyy, lab) in enumerate(
                [("disk", dv, dd, "disk mA/cm²"), ("ring", rv, rr, "ring mA/cm²")]):
            ax = axes[row][j]
            ax.plot(dvv, dyy, "k-", lw=2, label="data")
            ax.set_title(f"pH {ph} {obs}")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel("V_RHE (V)")
    # overlay model curves from saved iv jsons (+ N0 cache, N1a relabel)
    for j, ph in enumerate(PH_LIST):
        dlt = _delta(ph, S)
        srcs = {
            "N0": PH_SER / f"iv_curve_pH{ph}.json",
            "N1b": OUT / f"iv_N1b_pH{ph}.json",
            "A": OUT / f"iv_A_pH{ph}.json",
        }
        n0v = None
        for variant, path in srcs.items():
            if not Path(path).exists():
                continue
            v, cd, ring = _observables(json.load(open(path)))
            if variant == "N0":
                n0v = (v, cd, ring)
            axes[0][j].plot(v, cd, "--", color=col[variant], label=variant)
            axes[1][j].plot(v, ring, "--", color=col[variant], label=variant)
        if n0v is not None:
            v, cd, ring = n0v
            axes[0][j].plot(v + dlt, cd, ":", color=col["N1a"], label="N1a")
            axes[1][j].plot(v + dlt, ring, ":", color=col["N1a"], label="N1a")
        axes[0][j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "m1a_disk_ring_overlay.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
