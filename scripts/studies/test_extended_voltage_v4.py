"""Extended voltage v4: graduated z strategy.

z=0.9 gained +70mV. Try a smooth z(V) schedule that reduces charge
coupling as we move anodic, then see how far we can push.

Also test: can we do z-continuation AT each voltage point?
i.e., solve at z=0.9 first (easy), then ramp to z=1.0 from there.
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
    observable_scale = -I_SCALE

    # Fine voltage grid through the onset region
    V_RHE = np.sort(np.array([
        -0.30, -0.10, 0.00, 0.05, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.20,
        0.22, 0.24, 0.26, 0.28, 0.30,
        0.35, 0.40, 0.50, 0.60, 0.70,
    ]))[::-1]
    PHI_HAT = V_RHE / V_T
    n_pts = len(V_RHE)

    SNES = {
        "snes_type": "newtonls", "snes_max_it": 500,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-14,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.15,
        "snes_divergence_tolerance": 1e8,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 100,
    }

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.1, t_end=150.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    z_nominal = [float(z_vals[i]) for i in range(n_species)]

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # ---- Build neutral solutions (Phase 1) ----
    print("Phase 1: neutral sweep...")
    with adj.stop_annotating():
        ctx1 = build_context(sp_list, mesh=mesh)
        ctx1 = build_forms(ctx1, sp_list)
        set_initial_conditions(ctx1, sp_list)
    U1 = ctx1["U"]; U1p = ctx1["U_prev"]; W = ctx1["W"]
    z1c = ctx1.get("z_consts")
    if z1c:
        for zc in z1c: zc.assign(0.0)
    paf1 = ctx1.get("phi_applied_func"); dtc1 = ctx1.get("dt_const")
    dti = float(dt)

    with adj.stop_annotating():
        prob1 = fd.NonlinearVariationalProblem(
            ctx1["F_res"], U1, bcs=ctx1["bcs"], J=fd.derivative(ctx1["F_res"], U1))
        sol1 = fd.NonlinearVariationalSolver(prob1, solver_parameters=sp_dict)
    of1 = _build_bv_observable_form(ctx1, mode="current_density", reaction_index=None, scale=1.0)

    def _ss1(ms):
        dc = dti; dtc1.assign(dti); pf = pd = None; sc = 0
        for s in range(1, ms+1):
            try: sol1.solve()
            except: return False, -1
            U1p.assign(U1)
            fv = float(fd.assemble(of1))
            if pf is not None:
                d = abs(fv-pf); sv = max(abs(fv),abs(pf),1e-8)
                sc = sc+1 if (d/sv<=1e-4 or d<=1e-8) else 0
                if pd and d>0:
                    r = pd/d
                    dc = min(dc*min(r,4.0),dti*30) if r>1 else max(dc*0.5,dti)
                    dtc1.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, ms

    sweep = _build_sweep_order(PHI_HAT)
    neutral = [None]*n_pts; hub = None; pe = 0.0
    t_p1 = time.time()
    with adj.stop_annotating():
        for p, oi in enumerate(sweep):
            ei = float(PHI_HAT[oi])
            if p > 0 and np.sign(ei) != np.sign(pe) and np.sign(ei) != 0:
                if hub:
                    for s, d in zip(hub, U1.dat): d.data[:] = s
                    U1p.assign(U1)
            if p > 0 and abs(ei-pe) > 2.0:
                for br in np.linspace(pe, ei, max(2, int(abs(ei-pe)/2.0))+1)[1:-1]:
                    paf1.assign(br); _ss1(20)
            paf1.assign(ei); _ss1(100 if p==0 else 20)
            neutral[oi] = tuple(d.data_ro.copy() for d in U1.dat)
            if not hub: hub = neutral[oi]
            pe = ei
    print(f"  Phase 1: {time.time()-t_p1:.1f}s\n")

    # ---- Build Phase 2 context (separate from Phase 1) ----
    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)
    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func"); dtc = ctx.get("dt_const")
    with adj.stop_annotating():
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)
    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

    def _sz(z):
        if zc:
            for i in range(n_species): zc[i].assign(z_nominal[i]*z)

    def _ss(ms):
        dc = dti; dtc.assign(dti); pf = pd = None; sc = 0
        for s in range(1, ms+1):
            try: sol.solve()
            except: return False, -1
            Up.assign(U)
            fv = float(fd.assemble(of))
            if pf is not None:
                d = abs(fv-pf); sv = max(abs(fv),abs(pf),1e-8)
                sc = sc+1 if (d/sv<=1e-4 or d<=1e-8) else 0
                if pd and d>0:
                    r = pd/d
                    dc = min(dc*min(r,4.0),dti*30) if r>1 else max(dc*0.5,dti)
                    dtc.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, ms

    Uc = fd.Function(W); Upc = fd.Function(W)
    def _ck(): Uc.assign(U); Upc.assign(Up)
    def _rs(): U.assign(Uc); Up.assign(Upc)

    # ---- Phase 2: graduated z + per-point z refinement ----
    cd = np.full(n_pts, np.nan); pc = np.full(n_pts, np.nan)
    z_final = np.full(n_pts, 0.0)
    asc = np.argsort(PHI_HAT)

    last_data = None

    print("Phase 2: Graduated z with per-point z-refinement\n")
    print("Strategy: warm-start at z=0.85 → ramp z to 1.0 at each point\n")
    t_p2 = time.time()

    with adj.stop_annotating():
        for pos, idx in enumerate(asc):
            ph = float(PHI_HAT[idx]); v = V_RHE[idx]
            t0 = time.time()
            achieved = 0.0; method = ""

            # --- Step A: Get to z=0.85 (via warm-start or z-ramp) ---
            z_easy = 0.85
            got_easy = False

            if last_data is not None:
                for s, d in zip(last_data, U.dat): d.data[:] = s
                Up.assign(U)
                _sz(z_easy); paf.assign(ph); _ck()
                c, st = _ss(80)
                if c or (st > 3 and st != -1):
                    got_easy = True; achieved = z_easy
                    method = f"warm@{z_easy}"
                else:
                    _rs()

            if not got_easy and neutral[idx] is not None:
                for s, d in zip(neutral[idx], U.dat): d.data[:] = s
                Up.assign(U)
                _sz(0.0); paf.assign(ph); _ck()
                _sz(z_easy)
                c, st = _ss(150)
                if c or (st > 5 and st != -1):
                    got_easy = True; achieved = z_easy
                    method = f"zramp@{z_easy}"
                else:
                    _rs()
                    # Try lower z
                    for zt in [0.7, 0.5, 0.3]:
                        for s, d in zip(neutral[idx], U.dat): d.data[:] = s
                        Up.assign(U); _sz(0.0); paf.assign(ph); _ck()
                        _sz(zt)
                        c, st = _ss(150)
                        if c or (st > 5 and st != -1):
                            got_easy = True; achieved = zt; z_easy = zt
                            method = f"zramp@{zt}"
                            break
                        _rs()

            # --- Step B: Ramp z from z_easy → 1.0 in small increments ---
            if got_easy:
                _ck()  # checkpoint at z_easy
                # Try direct jump to z=1.0
                _sz(1.0)
                c, st = _ss(100)
                if c or (st > 3 and st != -1):
                    achieved = 1.0
                    method += "→1.0direct"
                else:
                    _rs()
                    # Fine z-ramp from z_easy to 1.0
                    z_steps = np.linspace(z_easy, 1.0, 11)[1:]  # 10 sub-steps
                    for zt in z_steps:
                        _ck()
                        _sz(zt)
                        c, st = _ss(80)
                        if c or (st > 3 and st != -1):
                            achieved = zt; _ck()
                        else:
                            _rs()
                            # Bisect
                            lo, hi = achieved, zt
                            for _ in range(4):
                                mid = (lo+hi)/2
                                if hi-lo < 0.005: break
                                _ck(); _sz(mid)
                                c, st = _ss(80)
                                if c or (st>3 and st!=-1):
                                    achieved = mid; lo = mid; _ck()
                                else:
                                    _rs(); hi = mid
                            break
                    method += f"→z={achieved:.3f}"

            z_final[idx] = achieved
            if achieved > 0:
                cd[idx] = float(fd.assemble(ocd))
                pc[idx] = float(fd.assemble(opc))
                last_data = tuple(d.data_ro.copy() for d in U.dat)

            elapsed = time.time() - t0
            zs = "OK" if achieved >= 0.999 else f"z={achieved:.3f}"
            print(f"  V={v:+.3f}V (phi={ph:+7.2f}): {zs:>10}  "
                  f"cd={cd[idx]:+.4f}  pc={pc[idx]:+.4f}  [{method}] ({elapsed:.1f}s)")

    p2t = time.time() - t_p2
    nf = int(np.sum(z_final >= 0.999))
    np9 = int(np.sum(z_final >= 0.85))

    print(f"\n{'='*70}")
    print(f"  z=1.0: {nf}/{n_pts},  z>=0.85: {np9}/{n_pts}")
    print(f"  Phase 2: {p2t:.1f}s")
    print(f"{'='*70}")
    fm = z_final >= 0.999
    if fm.any():
        print(f"  Max anodic V at z=1.0: {V_RHE[fm].max():.3f}V")
    fm9 = z_final >= 0.85
    if fm9.any():
        print(f"  Max anodic V at z>=0.85: {V_RHE[fm9].max():.3f}V")
        print(f"  (z>=0.85 captures ~97% of charge physics)")


if __name__ == "__main__":
    main()
