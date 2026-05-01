"""Test: does the hybrid solver make k0 identifiable?

Compare curves at different k0 values to see if the onset shape changes.
If the onset shifts with k0, the inverse problem is well-posed.
"""
from __future__ import annotations
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def main():
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE, K_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78
    from Forward.bv_solver.hybrid_forward import solve_curve_hybrid

    V_RHE = np.sort(np.array([
        -0.40, -0.20, 0.00, 0.10,
        0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
    ]))[::-1]
    PHI_HAT = V_RHE / V_T

    SNES = {
        "snes_type": "newtonls", "snes_max_it": 400,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
        "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
    }

    k0_multipliers = [0.5, 1.0, 2.0, 5.0]

    print(f"{'='*70}")
    print(f"  K0 SENSITIVITY TEST (hybrid solver)")
    print(f"  k0_r1 baseline: {K0_HAT_R1:.4e} ({K0_HAT_R1*K_SCALE:.3e} m/s)")
    print(f"{'='*70}\n")

    results = {}
    for mult in k0_multipliers:
        k0_r1 = K0_HAT_R1 * mult
        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=80.0,
            species=FOUR_SPECIES_CHARGED, snes_opts=SNES,
            k0_hat_r1=k0_r1,
            E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
        )
        print(f"  k0_r1 x{mult}: solving...")
        result = solve_curve_hybrid(sp, PHI_HAT, -I_SCALE, v_transition=0.10, n_workers=4)
        results[mult] = result
        print()

    # Compare
    print(f"\n{'='*70}")
    print(f"  SENSITIVITY COMPARISON")
    print(f"{'='*70}")

    sort_idx = np.argsort(V_RHE)
    header = f"  {'V_RHE':>8}"
    for mult in k0_multipliers:
        header += f"  {'x'+str(mult)+' cd':>10}  {'x'+str(mult)+' pc':>10}"
    print(header)

    for i in sort_idx:
        row = f"  {V_RHE[i]:+8.3f}"
        for mult in k0_multipliers:
            r = results[mult]
            row += f"  {r.cd[i]:+10.4f}  {r.pc[i]:+10.4f}"
        print(row)

    # Compute sensitivity: how much does cd/pc change per unit change in k0?
    print(f"\n  Sensitivity: max |Δcd| and |Δpc| between k0 x0.5 and k0 x5.0:")
    r_lo = results[0.5]
    r_hi = results[5.0]
    for i in sort_idx:
        if np.isnan(r_lo.cd[i]) or np.isnan(r_hi.cd[i]):
            continue
        dcd = abs(r_hi.cd[i] - r_lo.cd[i])
        dpc = abs(r_hi.pc[i] - r_lo.pc[i])
        z_label = "z=0" if V_RHE[i] > 0.10 else "z=1"
        star = " ***" if dcd > 0.01 or dpc > 0.01 else ""
        print(f"    V={V_RHE[i]:+.3f}V [{z_label}]: Δcd={dcd:.4f}, Δpc={dpc:.4f}{star}")


if __name__ == "__main__":
    main()
