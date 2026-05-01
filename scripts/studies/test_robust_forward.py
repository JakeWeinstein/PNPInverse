"""Quick test of the robust parallel forward solver."""
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

    V_RHE = np.array([
        -0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10,
    ])
    PHI_HAT = np.sort(V_RHE / V_T)[::-1]

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 400,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
            "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
        },
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    print("=== Testing robust_forward.solve_curve_robust ===\n")
    print(f"V_RHE: {V_RHE.tolist()}")
    print(f"phi_hat: [{PHI_HAT.min():.1f}, {PHI_HAT.max():.1f}]")
    print(f"E_eq: ({E_EQ_R1}, {E_EQ_R2})")

    from Forward.bv_solver.robust_forward import solve_curve_robust

    t0 = time.time()
    result = solve_curve_robust(
        sp, PHI_HAT, observable_scale=-I_SCALE,
        n_workers=8, charge_steps=15, max_eta_gap=2.0,
    )
    total = time.time() - t0

    print(f"\n=== RESULTS ({total:.1f}s) ===")
    print(f"Converged: {result.n_converged}/{result.n_total}")
    print(f"Phase 1: {result.phase1_time:.1f}s, Phase 2: {result.phase2_time:.1f}s")

    sort_idx = np.argsort(V_RHE)
    for i in sort_idx:
        z_str = f"z={result.z_achieved[i]:.3f}" if result.z_achieved[i] < 0.999 else "OK"
        print(f"  V={V_RHE[i]:+.3f}V: cd={result.cd[i]:+.4f}, pc={result.pc[i]:+.4f} ({z_str})")

    # Compare with sequential charge continuation
    print(f"\n=== Comparing with sequential solve_grid_with_charge_continuation ===\n")
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    seq_cd = np.full(len(PHI_HAT), np.nan)
    seq_pc = np.full(len(PHI_HAT), np.nan)
    obs_scale = -I_SCALE

    def _extract(idx, phi, ctx):
        seq_cd[idx] = float(fd.assemble(_build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=obs_scale)))
        seq_pc[idx] = float(fd.assemble(_build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=obs_scale)))

    t_seq = time.time()
    with adj.stop_annotating():
        solve_grid_with_charge_continuation(
            sp, phi_applied_values=PHI_HAT,
            charge_steps=20, mesh=mesh, max_eta_gap=2.0, min_delta_z=0.002,
            per_point_callback=_extract,
        )
    seq_time = time.time() - t_seq

    print(f"Sequential time: {seq_time:.1f}s")
    print(f"Parallel time:   {total:.1f}s")
    print(f"Speedup:         {seq_time/total:.1f}x\n")

    max_cd_diff = np.nanmax(np.abs(result.cd - seq_cd))
    max_pc_diff = np.nanmax(np.abs(result.pc - seq_pc))
    print(f"Max |cd diff|: {max_cd_diff:.2e}")
    print(f"Max |pc diff|: {max_pc_diff:.2e}")

    if max_cd_diff < 1e-3 and max_pc_diff < 1e-3:
        print("PASS: Results match within tolerance")
    else:
        print("WARNING: Results differ significantly")
        for i in sort_idx:
            cd_d = abs(result.cd[i] - seq_cd[i]) if not np.isnan(seq_cd[i]) else float('nan')
            pc_d = abs(result.pc[i] - seq_pc[i]) if not np.isnan(seq_pc[i]) else float('nan')
            print(f"  V={V_RHE[i]:+.3f}: cd_diff={cd_d:.2e}, pc_diff={pc_d:.2e}")


if __name__ == "__main__":
    main()
