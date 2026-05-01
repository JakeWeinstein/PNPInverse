"""SNES diagnostics: identify exact failure mode at the convergence boundary.

Tests what happens at phi_hat=+5.84 (V=+0.15V) where z=1 fails.
Enables full SNES/KSP/line-search monitoring.
Also tests fieldsplit preconditioning.
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
        setup_firedrake_env, V_T, I_SCALE,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.robust_forward import _phase1_neutral_sweep

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    # Single target point at the failure boundary
    V_TARGET = 0.15  # V vs RHE — where z=1 fails
    PHI_TARGET = V_TARGET / V_T

    # Also need nearby converging point for warm-start
    V_RHE = np.sort(np.array([-0.10, 0.00, 0.05, 0.10, 0.12, 0.15]))[::-1]
    PHI_HAT = V_RHE / V_T

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.1, t_end=100.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 500,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-14,
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 100,
        },
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list

    print(f"{'='*70}")
    print(f"  SNES DIAGNOSTICS at V={V_TARGET}V (phi_hat={PHI_TARGET:.2f})")
    print(f"{'='*70}\n")

    # Phase 1: get neutral ICs
    print("Phase 1: neutral sweep...")
    neutrals, _, _ = _phase1_neutral_sweep(sp_list, PHI_HAT, max_eta_gap=2.0)
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    n = n_species
    z_nominal = [float(z_vals[i]) for i in range(n)]

    # ---- Test A: Standard solver with SNES monitoring ----
    print("="*60)
    print("  Test A: Standard solver at V=+0.15V with SNES monitor")
    print("="*60)

    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]; W = ctx["W"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func")
    dtc = ctx.get("dt_const"); dti = float(dt)

    # Load neutral IC for the target point
    target_idx = list(V_RHE).index(V_TARGET)
    for s, d in zip(neutrals[target_idx], U.dat):
        d.data[:] = s
    Up.assign(U)

    # Set z=0.8 (partial, where standard solver starts failing)
    for i in range(n):
        zc[i].assign(z_nominal[i] * 0.85)
    paf.assign(PHI_TARGET)

    # Build solver WITH monitoring
    sp_dict_monitor = {
        "snes_type": "newtonls",
        "snes_max_it": 20,
        "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_linesearch_monitor": None,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict_monitor)

    print(f"\n  Attempting z=0.85 at phi_hat={PHI_TARGET:.2f}...")
    dtc.assign(dti)
    try:
        sol.solve()
        Up.assign(U)
        print("  Step 1: CONVERGED")
    except Exception as e:
        print(f"  Step 1: FAILED - {str(e)[:120]}")

    # ---- Test B: Fieldsplit preconditioning ----
    print("\n" + "="*60)
    print("  Test B: FIELDSPLIT preconditioning (phi | species)")
    print("="*60)

    # Rebuild context fresh
    with adj.stop_annotating():
        ctx2 = build_context(sp_list, mesh=mesh)
        ctx2 = build_forms(ctx2, sp_list)
    U2 = ctx2["U"]; U2p = ctx2["U_prev"]; W2 = ctx2["W"]
    zc2 = ctx2.get("z_consts"); paf2 = ctx2.get("phi_applied_func")
    dtc2 = ctx2.get("dt_const")

    for s, d in zip(neutrals[target_idx], U2.dat):
        d.data[:] = s
    U2p.assign(U2)
    for i in range(n):
        zc2[i].assign(z_nominal[i] * 0.85)
    paf2.assign(PHI_TARGET)

    # Fieldsplit: split phi (field n) from species (fields 0..n-1)
    # species_fields = "0,1,2,3" for 4-species
    species_str = ",".join(str(i) for i in range(n))

    sp_fieldsplit = {
        "snes_type": "newtonls",
        "snes_max_it": 30,
        "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_linesearch_type": "bt",  # backtracking instead of l2
        "snes_linesearch_maxstep": 1.0,
        "snes_divergence_tolerance": 1e10,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "fgmres",
        "ksp_max_it": 200,
        "ksp_rtol": 1e-6,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
        f"pc_fieldsplit_0_fields": species_str,
        f"pc_fieldsplit_1_fields": str(n),
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "lu",
        "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
    }

    prob2 = fd.NonlinearVariationalProblem(
        ctx2["F_res"], U2, bcs=ctx2["bcs"], J=fd.derivative(ctx2["F_res"], U2))
    sol2 = fd.NonlinearVariationalSolver(prob2, solver_parameters=sp_fieldsplit)

    print(f"\n  Attempting z=0.85 at phi_hat={PHI_TARGET:.2f} with fieldsplit...")
    dtc2.assign(dti)
    try:
        sol2.solve()
        U2p.assign(U2)
        print("  Step 1: CONVERGED")
    except Exception as e:
        print(f"  Step 1: FAILED - {str(e)[:120]}")

    # ---- Test C: Smaller initial dt ----
    print("\n" + "="*60)
    print("  Test C: Very small dt=0.001 at z=0.85")
    print("="*60)

    with adj.stop_annotating():
        ctx3 = build_context(sp_list, mesh=mesh)
        ctx3 = build_forms(ctx3, sp_list)
    U3 = ctx3["U"]; U3p = ctx3["U_prev"]
    zc3 = ctx3.get("z_consts"); paf3 = ctx3.get("phi_applied_func")
    dtc3 = ctx3.get("dt_const")

    for s, d in zip(neutrals[target_idx], U3.dat):
        d.data[:] = s
    U3p.assign(U3)
    for i in range(n):
        zc3[i].assign(z_nominal[i] * 0.85)
    paf3.assign(PHI_TARGET)

    sp_small_dt = {
        "snes_type": "newtonls", "snes_max_it": 30,
        "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_linesearch_type": "bt",
        "snes_divergence_tolerance": 1e10,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    prob3 = fd.NonlinearVariationalProblem(
        ctx3["F_res"], U3, bcs=ctx3["bcs"], J=fd.derivative(ctx3["F_res"], U3))
    sol3 = fd.NonlinearVariationalSolver(prob3, solver_parameters=sp_small_dt)

    print(f"\n  Attempting 5 steps with dt=0.001 at z=0.85, phi_hat={PHI_TARGET:.2f}...")
    tiny_dt = 0.001
    dtc3.assign(tiny_dt)
    for step in range(1, 6):
        try:
            sol3.solve()
            U3p.assign(U3)
            print(f"  Step {step}: OK (dt={tiny_dt:.4f})")
            tiny_dt *= 2  # Grow dt
            dtc3.assign(tiny_dt)
        except Exception as e:
            print(f"  Step {step}: FAILED at dt={tiny_dt:.4f} - {str(e)[:80]}")
            break

    # ---- Test D: z=0.95 with warm-start from z=0.85 ----
    print("\n" + "="*60)
    print("  Test D: Sequential z-steps 0.85→0.90→0.95→1.0, tiny dt")
    print("="*60)

    with adj.stop_annotating():
        ctx4 = build_context(sp_list, mesh=mesh)
        ctx4 = build_forms(ctx4, sp_list)
    U4 = ctx4["U"]; U4p = ctx4["U_prev"]
    zc4 = ctx4.get("z_consts"); paf4 = ctx4.get("phi_applied_func")
    dtc4 = ctx4.get("dt_const")

    for s, d in zip(neutrals[target_idx], U4.dat):
        d.data[:] = s
    U4p.assign(U4)
    paf4.assign(PHI_TARGET)

    sp_d = {
        "snes_type": "newtonls", "snes_max_it": 50,
        "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    prob4 = fd.NonlinearVariationalProblem(
        ctx4["F_res"], U4, bcs=ctx4["bcs"], J=fd.derivative(ctx4["F_res"], U4))
    sol4 = fd.NonlinearVariationalSolver(prob4, solver_parameters=sp_d)

    of4 = _build_bv_observable_form(ctx4, mode="current_density", reaction_index=None, scale=1.0)

    for z_target in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        for i in range(n):
            zc4[i].assign(z_nominal[i] * z_target)

        # Multiple small time steps
        dt_val = 0.001
        dtc4.assign(dt_val)
        ok = False
        for step in range(1, 201):
            try:
                sol4.solve()
                U4p.assign(U4)
                # SER: grow dt
                dt_val = min(dt_val * 2.0, 1.0)
                dtc4.assign(dt_val)
                # Check convergence
                fv = float(fd.assemble(of4))
                if step > 4:
                    ok = True
                    break
            except:
                print(f"  z={z_target:.2f}: FAILED at step {step}, dt={dt_val:.4f}")
                break

        if ok:
            print(f"  z={z_target:.2f}: OK ({step} steps)")
        else:
            if not ok and step > 1:
                print(f"  z={z_target:.2f}: budget ({step} steps)")
            break


if __name__ == "__main__":
    main()
