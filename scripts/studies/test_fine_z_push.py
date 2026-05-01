"""Push past z=0.75 at V=+0.15V with very fine z-steps and tiny dt.

From our earlier tests, the standard solver reaches z=0.75 at V=+0.15V.
Can we push to z=1.0 with extremely fine z-steps (0.01) and tiny dt?
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
    from Forward.bv_solver.sweep_order import _build_sweep_order

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    V_RHE = np.sort(np.array([-0.10, 0.00, 0.05, 0.10, 0.12, 0.14, 0.15]))[::-1]
    PHI_HAT = V_RHE / V_T

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.1, t_end=100.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 100,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-14,
            "snes_linesearch_type": "bt", "snes_linesearch_order": 3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 100,
        },
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    z_nominal = [float(z_vals[i]) for i in range(n_species)]
    n = n_species

    print(f"{'='*70}")
    print(f"  FINE Z-PUSH: V=+0.15V, z from 0 to 1.0 in 0.01 steps")
    print(f"{'='*70}\n")

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # Phase 1: neutral sweep
    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)
        set_initial_conditions(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]; W = ctx["W"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func")
    dtc = ctx.get("dt_const"); dti = float(dt)

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)

    dt_max = dti * 30.0

    def _sz(z):
        for i in range(n): zc[i].assign(z_nominal[i]*z)

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
                    dc = min(dc*min(r,4.0),dt_max) if r>1 else max(dc*0.5,dti*0.001)
                    dtc.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, max_steps

    Uc = fd.Function(W); Upc = fd.Function(W)
    def _ck(): Uc.assign(U); Upc.assign(Up)
    def _rs(): U.assign(Uc); Up.assign(Upc)

    # Neutral sweep at z=0
    print("Phase 1: neutral sweep at z=0")
    _sz(0.0)
    sweep = _build_sweep_order(PHI_HAT)
    hub = None; pe = 0.0

    with adj.stop_annotating():
        for p, oi in enumerate(sweep):
            ei = float(PHI_HAT[oi])
            if p > 0 and np.sign(ei) != np.sign(pe) and np.sign(ei) != 0 and hub:
                for s, d in zip(hub, U.dat): d.data[:] = s
                Up.assign(U)
            if p > 0 and abs(ei-pe) > 2.0:
                for br in np.linspace(pe, ei, max(2,int(abs(ei-pe)/2.0))+1)[1:-1]:
                    paf.assign(br); _ss(20)
            paf.assign(ei); _ss(100 if p==0 else 20)
            if not hub: hub = tuple(d.data_ro.copy() for d in U.dat)
            pe = ei

    # Now at the target voltage V=+0.15V
    target_idx = list(V_RHE).index(0.15)
    paf.assign(float(PHI_HAT[target_idx]))

    print(f"\nPhase 2: Fine z-ramp at V=+0.15V (phi_hat={PHI_HAT[target_idx]:.2f})")
    print(f"  z-step = 0.01, dt starts at 0.01, BT line search\n")

    _ck()  # checkpoint neutral solution

    # Fine z-ramp: 0.0 → 1.0 in steps of 0.01
    achieved = 0.0
    z_step = 0.01
    z_targets = np.arange(z_step, 1.0 + z_step/2, z_step)

    with adj.stop_annotating():
        for zt in z_targets:
            _ck()
            _sz(zt)
            # Start with small dt, allow SER to grow
            ok, steps = _ss(200, dt_start=0.01)
            if ok or steps > 5:
                achieved = zt
                _ck()
                if zt % 0.1 < z_step/2 or zt >= 0.99:
                    print(f"  z={zt:.2f}: OK ({steps} steps)")
            else:
                _rs()
                print(f"  z={zt:.2f}: FAILED at step {steps}")
                # Try even smaller dt
                _sz(zt)
                ok2, steps2 = _ss(500, dt_start=0.001)
                if ok2 or steps2 > 10:
                    achieved = zt
                    _ck()
                    print(f"  z={zt:.2f}: recovered with tiny dt ({steps2} steps)")
                else:
                    _rs()
                    print(f"  z={zt:.2f}: STUCK - cannot proceed")
                    break

    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

    cd_val = float(fd.assemble(ocd)) if achieved > 0 else float('nan')
    pc_val = float(fd.assemble(opc)) if achieved > 0 else float('nan')

    print(f"\n{'='*70}")
    print(f"  RESULT: achieved z={achieved:.3f} at V=+0.15V")
    print(f"  cd={cd_val:.6f}, pc={pc_val:.6f}")
    if achieved >= 0.999:
        print(f"  SUCCESS: Full z=1.0 reached!")
    else:
        print(f"  Previous best was z=0.75 (standard solver)")
        if achieved > 0.75:
            print(f"  IMPROVEMENT: z increased from 0.75 to {achieved:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
