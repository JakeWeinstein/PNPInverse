"""Extended voltage v3: mesh refinement + hybrid z-eta continuation.

Hypothesis: the z=1 convergence failure at phi_hat>4 is due to insufficient
mesh resolution of the EDL (Debye length ~43nm vs smallest cell ~100nm).

Tests:
A) Double mesh density (Ny=400) to better resolve EDL
B) Hybrid z-eta: reduce z slightly (e.g., 0.95) when stepping anodic
C) Combined: fine mesh + hybrid
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


def run_test(label, Nx, Ny, beta, dt_val, z_override=None):
    """Run one mesh/parameter configuration."""
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
    observable_scale = -I_SCALE

    V_RHE = np.sort(np.array([
        -0.30, -0.10, 0.00, 0.05, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.20,
        0.25, 0.30, 0.40, 0.50,
    ]))[::-1]
    PHI_HAT = V_RHE / V_T

    SNES = {
        "snes_type": "newtonls", "snes_max_it": 500,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-14,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.15,
        "snes_divergence_tolerance": 1e8,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 120,
    }

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt_val, t_end=150.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)

    print(f"\n{'='*60}")
    print(f"  {label}: Nx={Nx}, Ny={Ny}, beta={beta}, dt={dt_val}")
    if z_override: print(f"  z_override={z_override}")
    print(f"{'='*60}")

    # Build mesh FIRST, then use it for both Phase 1 and Phase 2
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    mesh = make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta)
    n_pts = len(PHI_HAT)

    # Phase 1: neutral sweep (inline, using THIS mesh)
    print("  Phase 1: neutral sweep...")
    t_p1 = time.time()
    with adj.stop_annotating():
        ctx1 = build_context(sp_list, mesh=mesh)
        ctx1 = build_forms(ctx1, sp_list)
        set_initial_conditions(ctx1, sp_list)

    U1 = ctx1["U"]; U1_prev = ctx1["U_prev"]
    z1c = ctx1.get("z_consts")
    if z1c:
        for zc in z1c:
            zc.assign(0.0)
    paf1 = ctx1.get("phi_applied_func")
    dtc1 = ctx1.get("dt_const")
    dti1 = float(dt)

    with adj.stop_annotating():
        F1 = ctx1["F_res"]; bcs1 = ctx1["bcs"]
        J1 = fd.derivative(F1, U1)
        prob1 = fd.NonlinearVariationalProblem(F1, U1, bcs=bcs1, J=J1)
        sol1 = fd.NonlinearVariationalSolver(prob1, solver_parameters=sp_dict if isinstance(params, dict) else {})

    of1 = _build_bv_observable_form(ctx1, mode="current_density", reaction_index=None, scale=1.0)
    dtm1 = dti1 * 20.0

    def _ss1(ms):
        dc = dti1; dtc1.assign(dti1); pf = pd = None; sc = 0
        for s in range(1, ms+1):
            try: sol1.solve()
            except: return False, -1
            U1_prev.assign(U1)
            fv = float(fd.assemble(of1))
            if pf is not None:
                d = abs(fv-pf); sv = max(abs(fv),abs(pf),1e-8)
                isd = (d/sv<=1e-4) or (d<=1e-8); sc = sc+1 if isd else 0
                if pd is not None and d>0:
                    r = pd/d
                    dc = min(dc*min(r,4.0),dtm1) if r>1 else max(dc*0.5,dti1)
                    dtc1.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, ms

    from Forward.bv_solver.sweep_order import _build_sweep_order
    sweep_idx = _build_sweep_order(PHI_HAT)
    neutral_solutions = [None] * n_pts
    hub_data = None; prev_eta = 0.0

    with adj.stop_annotating():
        for p, oi in enumerate(sweep_idx):
            ei = float(PHI_HAT[oi])
            if p > 0 and np.sign(ei) != np.sign(prev_eta) and np.sign(ei) != 0:
                if hub_data:
                    for s, d in zip(hub_data, U1.dat): d.data[:] = s
                    U1_prev.assign(U1)
            if p > 0:
                gap = abs(ei - prev_eta)
                if gap > 2.0:
                    nb = max(1, int(np.ceil(gap/2.0)))
                    for br in np.linspace(prev_eta, ei, nb+1)[1:-1]:
                        paf1.assign(br); _ss1(20)
            paf1.assign(ei)
            _ss1(100 if p==0 else 20)
            neutral_solutions[oi] = tuple(d.data_ro.copy() for d in U1.dat)
            if hub_data is None: hub_data = neutral_solutions[oi]
            prev_eta = ei

    p1_time = time.time() - t_p1
    print(f"  Phase 1 done ({p1_time:.1f}s)")

    # Phase 2: sequential z=1 voltage continuation (reuse same mesh)
    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)

    U = ctx["U"]; U_prev = ctx["U_prev"]; W = ctx["W"]
    n = ctx["n_species"]
    z_consts = ctx.get("z_consts")
    phi_applied_func = ctx.get("phi_applied_func")
    dt_const = ctx.get("dt_const")
    dt_initial = float(dt)
    z_nominal = [float(z_vals[i]) for i in range(n)]

    F_res = ctx["F_res"]; bcs = ctx["bcs"]
    J_form = fd.derivative(F_res, U)
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=sp_dict)

    obs_form = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)
    obs_cd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)

    dt_max = dt_initial * 30.0

    def _run_ss(max_steps):
        dt_curr = dt_initial
        dt_const.assign(dt_initial)
        pf = pd = None; sc = 0
        for step in range(1, max_steps + 1):
            try:
                solver.solve()
            except Exception:
                return False, -1
            U_prev.assign(U)
            fv = float(fd.assemble(obs_form))
            if pf is not None:
                delta = abs(fv - pf)
                s = max(abs(fv), abs(pf), 1e-8)
                is_s = (delta / s <= 1e-4) or (delta <= 1e-8)
                sc = sc + 1 if is_s else 0
                if pd is not None and delta > 0:
                    r = pd / delta
                    dt_curr = min(dt_curr * min(r, 4.0), dt_max) if r > 1 else max(dt_curr * 0.5, dt_initial)
                    dt_const.assign(dt_curr)
                pd = delta
            pf = fv
            if sc >= 4:
                return True, step
        return False, max_steps

    def _set_z(z_val):
        if z_consts:
            for i in range(n):
                z_consts[i].assign(z_nominal[i] * z_val)

    U_ckpt = fd.Function(W); U_prev_ckpt = fd.Function(W)
    def _ckpt(): U_ckpt.assign(U); U_prev_ckpt.assign(U_prev)
    def _rest(): U.assign(U_ckpt); U_prev.assign(U_prev_ckpt)

    last_z1_data = None
    asc_idx = np.argsort(PHI_HAT)
    n_pts = len(PHI_HAT)
    results = []

    z_target = z_override if z_override else 1.0

    t2 = time.time()
    with adj.stop_annotating():
        for pos, idx in enumerate(asc_idx):
            ph = float(PHI_HAT[idx])
            v = V_RHE[idx]
            t0 = time.time()

            # Try z=1 warm-start from previous
            success = False
            if last_z1_data is not None:
                for src, dst in zip(last_z1_data, U.dat):
                    dst.data[:] = src
                U_prev.assign(U)
                _set_z(z_target)
                phi_applied_func.assign(ph)
                _ckpt()
                conv, steps = _run_ss(100)
                if conv or (steps > 3 and steps != -1):
                    success = True
                else:
                    _rest()

            # Fallback: z-ramp from neutral
            if not success and neutral_solutions[idx] is not None:
                for src, dst in zip(neutral_solutions[idx], U.dat):
                    dst.data[:] = src
                U_prev.assign(U)
                _set_z(0.0)
                phi_applied_func.assign(ph)
                _ckpt()
                _set_z(z_target)
                conv, steps = _run_ss(200)
                if conv or (steps > 5 and steps != -1):
                    success = True
                else:
                    _rest()

            if success:
                cd_val = float(fd.assemble(obs_cd))
                last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                status = "OK"
            else:
                cd_val = float('nan')
                status = "FAIL"

            elapsed = time.time() - t0
            results.append((v, ph, cd_val, status, elapsed))
            print(f"  V={v:+.3f}V: {status:>4}  cd={cd_val:+.4f}  ({elapsed:.1f}s)")

    p2_time = time.time() - t2
    n_ok = sum(1 for _, _, _, s, _ in results if s == "OK")
    max_v = max(v for v, _, _, s, _ in results if s == "OK") if n_ok > 0 else float('nan')
    print(f"  → {n_ok}/{n_pts} converged, max V = {max_v:.3f}V, time = {p1_time+p2_time:.1f}s")
    return n_ok, max_v


def main():
    print("Testing multiple strategies to extend convergence boundary\n")

    # A) Baseline: standard mesh
    run_test("A: Baseline (Ny=200)", Nx=8, Ny=200, beta=3.0, dt_val=0.1)

    # B) Finer mesh
    run_test("B: Fine mesh (Ny=400)", Nx=8, Ny=400, beta=3.0, dt_val=0.1)

    # C) More grading (concentrate cells near electrode)
    run_test("C: High grading (beta=5)", Nx=8, Ny=200, beta=5.0, dt_val=0.1)

    # D) Fine mesh + high grading
    run_test("D: Fine+graded (Ny=400,beta=5)", Nx=8, Ny=400, beta=5.0, dt_val=0.1)

    # E) z=0.9 instead of z=1 (partial charge)
    run_test("E: z=0.9 (partial charge)", Nx=8, Ny=200, beta=3.0, dt_val=0.1, z_override=0.9)

    # F) Very small dt
    run_test("F: Tiny dt=0.01", Nx=8, Ny=200, beta=3.0, dt_val=0.01)


if __name__ == "__main__":
    main()
