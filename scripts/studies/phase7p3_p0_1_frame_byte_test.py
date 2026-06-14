"""Phase 7.3 P0.1 — end-to-end frame byte-test (the central falsification guard).

Runs the LOCKED pH-6.39 dual-series forward model (θ_L, water routes, V_OCP
1.019, K⁺, L_eff 21.7 µm) twice through the SAME driver (``dp._run``):

  * RHE frame  (proton_frame='rhe')  — the production / locked-fit path.
  * SHE frame  (proton_frame='she', anchored on the bulk c_H 4.07e-4).

At the anchor condition the SHE formal-potential offset is EXACTLY 0.0, so
the SHE-anchored driver must reproduce the locked fit's disk current (cd) and
ring/peroxide current (pc) at every voltage BYTE-FOR-BYTE.  If it does not,
the frame math is wrong and nothing downstream of the Phase 7.3 plan is
trustworthy (plan §0 / §2 / milestone 1 — STOP gate).

The reaction-level proof (instant, Firedrake-free) lives in
``tests/test_phase7p3_frame_byte.py``; this script is the solver-level
demonstration the plan's P0.1 asks for.

Run from PNPInverse/ in the firedrake venv:
    python -u scripts/studies/phase7p3_p0_1_frame_byte_test.py
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

# Locked pH-6.39 θ_L (= phase7p2_dual_series_summary.md accepted fit).
F2W, F4W = 10.0 ** -1.008699731156705, 10.0 ** -12.308926782786854
A2W, A4W = 0.5770144526758703, 0.304853786169721
L_EFF_UM = 21.7
V_OCP_PH639 = 1.019
BULK_H_PH639 = 4.07e-4   # mol/m³ -> pH 6.39 (the anchor)

OUT = Path(_ROOT) / "StudyResults" / "phase7p3_p0_1_frame_byte_test"


def _opts(proton_frame: str) -> SimpleNamespace:
    return SimpleNamespace(
        routes="water",
        k0_water_2e_factor=F2W, k0_water_4e_factor=F4W,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=A2W, alpha_water_4e=A4W,
        l_eff_um=L_EFF_UM, bulk_h_mol_m3=BULK_H_PH639,
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_ocp_rhe=V_OCP_PH639,
        v_grid_lo=0.10, v_grid_hi=0.60,        # pH-6.39 ORR window
        proton_frame=proton_frame,
        bulk_h_anchor_mol_m3=BULK_H_PH639,
        out_name=f"_p0_1_{proton_frame}",
    )


def _max_abs_diff(a_list, b_list):
    """Max |a−b| over voltages where BOTH are finite; None if no overlap."""
    worst = None
    n_cmp = 0
    for a, b in zip(a_list, b_list):
        if a is None or b is None:
            continue
        n_cmp += 1
        d = abs(float(a) - float(b))
        worst = d if worst is None else max(worst, d)
    return worst, n_cmp


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()
    import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp

    print("=" * 78, flush=True)
    print("  Phase 7.3 P0.1 frame byte-test — RHE (lock) vs SHE-anchored "
          "@ pH 6.39", flush=True)
    print(f"  she_eeq_shift_v(anchor) = "
          f"{dp._she_eeq_shift_v(_opts('she')):.17g} V (must be 0.0)",
          flush=True)
    print("=" * 78, flush=True)

    reports = {}
    for frame in ("rhe", "she"):
        t0 = time.time()
        print(f"\n--- {frame.upper()} frame run ---", flush=True)
        rep = dp._run(_opts(frame))
        rep["_wall_s"] = time.time() - t0
        reports[frame] = rep
        with open(OUT / f"iv_curve_{frame}.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(f"  {frame}: {rep.get('n_converged')}/{rep.get('n_total')} "
              f"converged in {rep['_wall_s']:.0f}s", flush=True)

    r, s = reports["rhe"], reports["she"]
    cd_worst, cd_n = _max_abs_diff(r["cd_mA_cm2"], s["cd_mA_cm2"])
    pc_worst, pc_n = _max_abs_diff(r["pc_mA_cm2"], s["pc_mA_cm2"])
    conv_match = r["converged"] == s["converged"]
    grid_match = r["v_rhe"] == s["v_rhe"] and r["v_rhe_deck"] == s["v_rhe_deck"]

    # Byte-exact at the anchor: identical inputs through a deterministic
    # solver -> identical outputs. Tiny tol guards only against incidental
    # nondeterminism; the expectation is EXACTLY 0.0.
    byte_exact = (conv_match and grid_match
                  and cd_worst is not None and pc_worst is not None
                  and cd_worst == 0.0 and pc_worst == 0.0)
    TOL = 1e-12
    within_tol = (conv_match and grid_match
                  and cd_worst is not None and pc_worst is not None
                  and cd_worst <= TOL and pc_worst <= TOL)

    verdict = {
        "test": "phase7p3_P0.1_frame_byte_test",
        "anchor_pH": dp._ph_from_bulk_h(BULK_H_PH639),
        "she_eeq_shift_v_at_anchor": dp._she_eeq_shift_v(_opts("she")),
        "nernst_slope_v_per_ph": dp._nernst_slope_v_per_ph(),
        "grid_match": grid_match,
        "converged_match": conv_match,
        "n_compared_cd": cd_n, "n_compared_pc": pc_n,
        "max_abs_diff_cd_mA_cm2": cd_worst,
        "max_abs_diff_pc_mA_cm2": pc_worst,
        "byte_exact": bool(byte_exact),
        "within_tol_1e-12": bool(within_tol),
        "PASS": bool(byte_exact),
    }
    with open(OUT / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"  grid_match={grid_match}  converged_match={conv_match}", flush=True)
    print(f"  max|Δcd| = {cd_worst!r} mA/cm²  (over {cd_n} pts)", flush=True)
    print(f"  max|Δpc| = {pc_worst!r} mA/cm²  (over {pc_n} pts)", flush=True)
    print(f"  BYTE-EXACT: {byte_exact}   (within 1e-12: {within_tol})", flush=True)
    print(f"  ==> P0.1 {'PASS — proceed to P0.2' if byte_exact else 'FAIL — STOP, frame math is wrong'}",
          flush=True)
    print("=" * 78, flush=True)
    print(f"wrote {OUT / 'verdict.json'}", flush=True)
    return 0 if byte_exact else 1


if __name__ == "__main__":
    sys.exit(main())
