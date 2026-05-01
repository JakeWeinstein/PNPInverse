"""Test the log-concentration transform solver.

Compares convergence of standard forms vs log-transformed forms
across the full voltage range including the onset region.
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
    from Forward.bv_solver.forms_log import (
        build_context_log, build_forms_log, set_initial_conditions_log,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.sweep_order import _build_sweep_order

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78
    observable_scale = -I_SCALE

    # Extended voltage grid through onset region
    V_RHE = np.sort(np.array([
        -0.50, -0.30, -0.10, 0.00, 0.05, 0.10,
        0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.50, 0.60, 0.70,
    ]))[::-1]
    PHI_HAT = V_RHE / V_T

    SNES = {
        "snes_type": "newtonls", "snes_max_it": 500,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-14,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 100,
    }

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.1, t_end=100.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    z_nominal = [float(z_vals[i]) for i in range(n_species)]

    print(f"{'='*70}")
    print(f"  LOG-TRANSFORM SOLVER TEST")
    print(f"  {len(V_RHE)} points, V_RHE: [{V_RHE.min():.2f}, {V_RHE.max():.2f}]")
    print(f"  E_eq: ({E_EQ_R1}, {E_EQ_R2})")
    print(f"{'='*70}\n")

    # --- HYBRID: standard solver for Phase 1, log-transform for Phase 2 ---
    from Forward.bv_solver.forms import (
        build_context as build_context_std,
        build_forms as build_forms_std,
        set_initial_conditions as set_initial_conditions_std,
    )
    from Forward.bv_solver.robust_forward import _phase1_neutral_sweep

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # Phase 1: Use STANDARD solver (linear at z=0, converges everywhere)
    print("Phase 1: Neutral sweep using STANDARD (c-based) solver...")
    neutrals_c, _, p1t = _phase1_neutral_sweep(sp_list, PHI_HAT, max_eta_gap=2.0)
    print(f"  Phase 1 done ({p1t:.1f}s)\n")

    # Now build log-transform context for Phase 2
    with adj.stop_annotating():
        ctx = build_context_log(sp_list, mesh=mesh)
        ctx = build_forms_log(ctx, sp_list)
        set_initial_conditions_log(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]; W = ctx["W"]
    n = ctx["n_species"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func")
    dtc = ctx.get("dt_const"); dti = float(dt)

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)
    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

    dt_max = dti * 30.0

    def _ss(ms):
        dc = dti; dtc.assign(dti); pf = pd = None; sc = 0
        for s in range(1, ms+1):
            try:
                sol.solve()
            except Exception as e:
                return False, -1, str(e)[:80]
            Up.assign(U)
            fv = float(fd.assemble(of))
            if pf is not None:
                d = abs(fv-pf); sv = max(abs(fv),abs(pf),1e-8)
                sc = sc+1 if (d/sv<=1e-4 or d<=1e-8) else 0
                if pd and d>0:
                    r = pd/d
                    dc = min(dc*min(r,4.0),dt_max) if r>1 else max(dc*0.5,dti)
                    dtc.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s, "converged"
        return False, ms, "budget"

    def _sz(z):
        if zc:
            for i in range(n): zc[i].assign(z_nominal[i]*z)

    Uc = fd.Function(W); Upc = fd.Function(W)
    def _ck(): Uc.assign(U); Upc.assign(Up)
    def _rs(): U.assign(Uc); Up.assign(Upc)

    # --- Convert standard c-based ICs to log-space ---
    # neutrals_c[i] = (c_0_data, c_1_data, ..., c_{n-1}_data, phi_data)
    # For log-transform: u_i = ln(max(c_i, floor))
    _CONV_FLOOR = 1e-10
    neutrals_log = [None] * len(PHI_HAT)
    for idx in range(len(PHI_HAT)):
        if neutrals_c[idx] is not None:
            arrs = list(neutrals_c[idx])
            log_arrs = []
            for i in range(n):
                c_arr = arrs[i].copy()
                c_arr = np.maximum(c_arr, _CONV_FLOOR)
                log_arrs.append(np.log(c_arr))
            log_arrs.append(arrs[n].copy())  # phi unchanged
            neutrals_log[idx] = tuple(log_arrs)

    print(f"  Converted {sum(1 for x in neutrals_log if x is not None)} neutral ICs to log-space\n")

    # --- Phase 2: z=1 voltage continuation using log-transform ---
    print("Phase 2: z=1 voltage continuation (log-transform, hybrid)\n")
    asc = np.argsort(PHI_HAT)
    cd = np.full(len(V_RHE), np.nan)
    pc = np.full(len(V_RHE), np.nan)
    z_final = np.full(len(V_RHE), 0.0)
    last_z1 = None
    t2 = time.time()

    with adj.stop_annotating():
        for pos, idx in enumerate(asc):
            ph = float(PHI_HAT[idx]); v = V_RHE[idx]
            t0 = time.time(); method = ""; success = False

            # Strategy A: warm-start from previous z=1
            if last_z1 is not None:
                for s, d in zip(last_z1, U.dat): d.data[:] = s
                Up.assign(U); _sz(1.0); paf.assign(ph); _ck()
                ok, steps, reason = _ss(80)
                if ok or (steps > 3 and steps != -1):
                    success = True; z_final[idx] = 1.0
                    method = f"z1-warm({steps}s,{reason})"
                else:
                    _rs()

            # Strategy B: z-ramp from neutral
            if not success and neutrals_log[idx] is not None:
                for s, d in zip(neutrals_log[idx], U.dat): d.data[:] = s
                Up.assign(U); _sz(0.0); paf.assign(ph)

                # Direct z=1
                _ck(); _sz(1.0)
                ok, steps, reason = _ss(150)
                if ok or (steps > 5 and steps != -1):
                    success = True; z_final[idx] = 1.0
                    method = f"zramp-direct({steps}s,{reason})"
                else:
                    _rs()
                    # Fine z-ramp
                    az = 0.0
                    for zt in np.linspace(0, 1, 21)[1:]:
                        _ck(); _sz(zt)
                        ok, steps, reason = _ss(100)
                        if ok or (steps > 3 and steps != -1):
                            az = zt; _ck()
                        else:
                            _rs(); break
                    # Try z=1 from best
                    if az > 0.5:
                        _ck(); _sz(1.0)
                        ok, steps, reason = _ss(150)
                        if ok or (steps > 5 and steps != -1):
                            az = 1.0
                        else:
                            _rs()
                    z_final[idx] = az
                    success = az >= 0.999
                    method = f"zramp-fine(z={az:.3f})"

            if success or z_final[idx] > 0:
                cd[idx] = float(fd.assemble(ocd))
                pc[idx] = float(fd.assemble(opc))
                if z_final[idx] >= 0.999:
                    last_z1 = tuple(d.data_ro.copy() for d in U.dat)

            elapsed = time.time()-t0
            zs = "OK" if z_final[idx] >= 0.999 else f"z={z_final[idx]:.3f}"
            print(f"  V={v:+.3f}V (phi={ph:+7.2f}): {zs:>10}  "
                  f"cd={cd[idx]:+.6f}  pc={pc[idx]:+.6f}  [{method}] ({elapsed:.1f}s)")

    p2t = time.time()-t2
    nf = int(np.sum(z_final >= 0.999))

    print(f"\n{'='*70}")
    print(f"  LOG-TRANSFORM RESULT")
    print(f"  z=1.0: {nf}/{len(V_RHE)} converged")
    print(f"  Phase 1: {p1t:.1f}s, Phase 2: {p2t:.1f}s")
    print(f"{'='*70}")
    fm = z_final >= 0.999
    if fm.any():
        print(f"  Max anodic V at z=1.0: {V_RHE[fm].max():.3f}V vs RHE")
        print(f"  (Standard solver boundary was V=+0.100V)")
        improvement = V_RHE[fm].max() - 0.10
        if improvement > 0:
            print(f"  IMPROVEMENT: +{improvement*1000:.0f}mV beyond standard solver!")
        else:
            print(f"  No improvement over standard solver")


if __name__ == "__main__":
    main()
