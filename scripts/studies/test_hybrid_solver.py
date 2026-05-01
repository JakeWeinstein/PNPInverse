"""Test hybrid solver across full voltage range."""
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
        setup_firedrake_env, V_T, I_SCALE,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    V_RHE = np.sort(np.array([
        -0.50, -0.40, -0.30, -0.20, -0.10,
        0.00, 0.05, 0.10,
        0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.50, 0.60, 0.70,
    ]))[::-1]
    PHI_HAT = V_RHE / V_T

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 400,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
        },
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    print(f"{'='*70}")
    print(f"  HYBRID SOLVER TEST")
    print(f"  {len(V_RHE)} points, V_RHE: [{V_RHE.min():.2f}, {V_RHE.max():.2f}]")
    print(f"  z=0 for V>0.10V, z=1 for V<=0.10V")
    print(f"{'='*70}\n")

    from Forward.bv_solver.hybrid_forward import solve_curve_hybrid

    result = solve_curve_hybrid(
        sp, PHI_HAT, observable_scale=-I_SCALE,
        v_transition=0.10, n_workers=8,
    )

    print(f"\n{'='*70}")
    print(f"  RESULTS: {result.n_converged}/{result.n_total} usable")
    print(f"{'='*70}")

    sort_idx = np.argsort(V_RHE)
    for i in sort_idx:
        z_label = "z=0" if result.z_achieved[i] < 0.5 else "z=1"
        print(f"  V={V_RHE[i]:+.3f}V: cd={result.cd[i]:+.6f}  pc={result.pc[i]:+.6f}  [{z_label}]")


if __name__ == "__main__":
    main()
