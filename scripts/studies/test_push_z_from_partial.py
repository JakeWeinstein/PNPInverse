"""Push z from the partial z=0.75 solution that the standard solver achieves.

Use solve_grid_with_charge_continuation to get to z=0.75, then try
to push further with fine z-steps, tiny dt, and backtracking line search.
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
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.forms import build_context, build_forms
    from Forward.bv_solver.observables import _build_bv_observable_form

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    # Single target voltage
    V_TARGET = 0.15
    PHI_TARGET = V_TARGET / V_T
    phi_arr = np.array([PHI_TARGET])

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.1, t_end=100.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 400,
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
    z_nominal = [float(z_vals[i]) for i in range(n_species)]
    n = n_species

    print(f"{'='*70}")
    print(f"  PUSH Z from partial solution at V=+{V_TARGET}V")
    print(f"{'='*70}\n")

    # Step 1: Get the partial-z solution from charge continuation
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    print("Step 1: Running charge continuation to get partial-z solution...")

    partial_U_data = None
    achieved_z = 0.0

    def _capture(idx, phi, ctx):
        nonlocal partial_U_data, achieved_z
        partial_U_data = tuple(d.data_ro.copy() for d in ctx["U"].dat)

    with adj.stop_annotating():
        result = solve_grid_with_charge_continuation(
            sp_list, phi_applied_values=phi_arr, charge_steps=20, mesh=mesh,
            max_eta_gap=2.0, min_delta_z=0.002,
            per_point_callback=_capture,
        )

    pt = result.points[0]
    achieved_z = pt.achieved_z_factor
    partial_U_data = pt.U_data
    print(f"  Charge continuation achieved: z={achieved_z:.4f}")
    print(f"  Converged: {pt.converged}")

    if achieved_z >= 0.999:
        print("  Already at z=1.0! No push needed.")
        return

    # Step 2: Build fresh context and load partial solution
    print(f"\nStep 2: Pushing from z={achieved_z:.4f} toward z=1.0...")

    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]; W = ctx["W"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func")
    dtc = ctx.get("dt_const"); dti = float(dt)

    # Load partial solution
    for src, dst in zip(partial_U_data, U.dat):
        dst.data[:] = src
    Up.assign(U)
    paf.assign(PHI_TARGET)

    # Set z to the achieved level
    for i in range(n):
        zc[i].assign(z_nominal[i] * achieved_z)

    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}

    # Build solver with BT line search (cubic backtracking)
    sp_push = dict(sp_dict)
    sp_push.update({
        "snes_type": "newtonls",
        "snes_max_it": 100,
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 3,
    })

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_push)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)

    dt_max = dti * 30.0

    def _ss(max_steps, dt_start=None):
        dc = dt_start or dti; dtc.assign(dc)
        pf = pd = None; sc = 0
        for s in range(1, max_steps+1):
            try: sol.solve()
            except: return False, s-1
            Up.assign(U)
            fv = float(fd.assemble(of))
            if pf is not None:
                d = abs(fv-pf); sv = max(abs(fv),abs(pf),1e-8)
                sc = sc+1 if (d/sv<=1e-4 or d<=1e-8) else 0
                if pd and d>0:
                    r = pd/d
                    dc = min(dc*min(r,4.0),dt_max) if r>1 else max(dc*0.5,0.0001)
                    dtc.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, max_steps

    Uc = fd.Function(W); Upc = fd.Function(W)
    def _ck(): Uc.assign(U); Upc.assign(Up)
    def _rs(): U.assign(Uc); Up.assign(Upc)

    # Fine z-push from achieved_z toward 1.0
    current_z = achieved_z
    z_step = 0.01  # Very fine steps

    with adj.stop_annotating():
        while current_z < 1.0 - 1e-6:
            target_z = min(current_z + z_step, 1.0)
            _ck()

            for i in range(n):
                zc[i].assign(z_nominal[i] * target_z)

            # Try with small dt first, then normal
            ok = False
            for dt_try in [0.001, 0.01, 0.1]:
                ok, steps = _ss(300, dt_start=dt_try)
                if ok or steps > 10:
                    current_z = target_z
                    _ck()
                    ok = True
                    break
                _rs()

            if ok:
                if current_z >= 0.999 or (current_z * 100) % 5 < z_step * 100 + 0.1:
                    print(f"  z={current_z:.3f}: OK ({steps} steps, dt_start={dt_try})")
            else:
                print(f"  z={target_z:.3f}: FAILED (all dt tried)")
                # Try even finer step
                z_step /= 2
                if z_step < 0.001:
                    print(f"  STUCK at z={current_z:.4f} (step size < 0.001)")
                    break
                print(f"  Halving z_step to {z_step:.4f}")

    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
    cd_val = float(fd.assemble(ocd))
    pc_val = float(fd.assemble(opc))

    print(f"\n{'='*70}")
    print(f"  RESULT: z={current_z:.4f} at V=+{V_TARGET}V")
    print(f"  cd={cd_val:.6f} mA/cm², pc={pc_val:.6f} mA/cm²")
    print(f"  Started at z={achieved_z:.4f}, pushed to z={current_z:.4f}")
    improvement = current_z - achieved_z
    if improvement > 0.01:
        print(f"  IMPROVEMENT: Δz = +{improvement:.4f}")
    else:
        print(f"  No significant improvement")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
