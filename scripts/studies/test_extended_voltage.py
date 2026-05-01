"""Test: extend voltage range using z=1 voltage continuation.

Strategy:
1. Use robust_forward Phase 1 to get neutral ICs (correct sweep order)
2. Z-ramp the most cathodic point first (easy)
3. March anodic: warm-start each point from z=1 solution of previous point
4. This should extend convergence far into the onset region
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
    observable_scale = -I_SCALE

    # Extended grid: deep cathodic through onset region
    V_RHE_unsorted = np.array([
        -0.50, -0.40, -0.30, -0.20, -0.10,
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    ])
    # Sort descending by phi_hat (= V_RHE / V_T) for Phase 1
    desc_idx = np.argsort(V_RHE_unsorted)[::-1]
    V_RHE = V_RHE_unsorted[desc_idx]
    PHI_HAT = V_RHE / V_T  # Now V_RHE[i] ↔ PHI_HAT[i] always

    SNES_OPTS = {
        "snes_type": "newtonls", "snes_max_it": 400,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
        "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
    }

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)

    print(f"{'='*70}")
    print(f"  EXTENDED VOLTAGE: z=1 Voltage Continuation")
    print(f"  {len(V_RHE)} points, V_RHE: [{V_RHE.min():.2f}, {V_RHE.max():.2f}]")
    print(f"{'='*70}\n")

    # --- Phase 1: Neutral sweep (reuse robust_forward) ---
    print("Phase 1: Neutral sweep...")
    neutral_solutions, mesh_dof, p1_time = _phase1_neutral_sweep(
        sp_list, PHI_HAT, max_eta_gap=2.0)
    print()

    # --- Phase 2: z=1 voltage continuation ---
    # Sort ascending by phi_hat (most cathodic first → march anodic)
    asc_idx = np.argsort(PHI_HAT)  # ascending phi_hat

    n_pts = len(PHI_HAT)
    cd = np.full(n_pts, np.nan)
    pc = np.full(n_pts, np.nan)
    z_achieved = np.full(n_pts, 0.0)

    # Build a SINGLE context for sequential z-ramp
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    W = ctx["W"]
    n = ctx["n_species"]
    z_consts = ctx.get("z_consts")
    phi_applied_func = ctx.get("phi_applied_func")
    dt_const = ctx.get("dt_const")
    dt_initial = float(dt)
    z_nominal = [float(z_vals[i]) for i in range(n)]

    F_res = ctx["F_res"]
    bcs = ctx["bcs"]
    J_form = fd.derivative(F_res, U)
    sp_dict = {k: v for k, v in params.items() if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=sp_dict)

    obs_form = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)
    obs_cd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)
    obs_pc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

    dt_max = dt_initial * 20.0

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

    U_ckpt = fd.Function(W)
    U_prev_ckpt = fd.Function(W)
    def _ckpt(): U_ckpt.assign(U); U_prev_ckpt.assign(U_prev)
    def _rest(): U.assign(U_ckpt); U_prev.assign(U_prev_ckpt)

    last_z1_data = None  # Most recent z=1 solution for warm-starting

    print("Phase 2: z=1 voltage continuation (cathodic → anodic)\n")
    t2 = time.time()

    with adj.stop_annotating():
        for pos, idx in enumerate(asc_idx):
            ph = float(PHI_HAT[idx])
            v = V_RHE[idx]
            t0 = time.time()
            method = ""

            # --- Strategy A: Warm-start from previous z=1 solution ---
            if last_z1_data is not None:
                for src, dst in zip(last_z1_data, U.dat):
                    dst.data[:] = src
                U_prev.assign(U)
                _set_z(1.0)
                phi_applied_func.assign(ph)
                _ckpt()

                conv, steps = _run_ss(60)
                if conv or (steps > 0 and steps != -1):
                    cd[idx] = float(fd.assemble(obs_cd))
                    pc[idx] = float(fd.assemble(obs_pc))
                    z_achieved[idx] = 1.0
                    last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                    method = f"z1-warm({steps}s)"
                else:
                    _rest()
                    method = "z1-warm-FAIL→"

            # --- Strategy B: Z-ramp from neutral IC ---
            if z_achieved[idx] < 0.999:
                if neutral_solutions[idx] is not None:
                    for src, dst in zip(neutral_solutions[idx], U.dat):
                        dst.data[:] = src
                    U_prev.assign(U)
                else:
                    set_initial_conditions(ctx, sp_list)

                _set_z(0.0)
                phi_applied_func.assign(ph)

                # Quick z=1 attempt
                _ckpt()
                _set_z(1.0)
                conv, steps = _run_ss(100)
                if conv or (steps > 0 and steps != -1):
                    z_achieved[idx] = 1.0
                    cd[idx] = float(fd.assemble(obs_cd))
                    pc[idx] = float(fd.assemble(obs_pc))
                    last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                    method += f"z-ramp-direct({steps}s)"
                else:
                    _rest()
                    # Binary z search
                    az = 0.0
                    for _ in range(8):
                        z_try = (az + 1.0) / 2.0
                        _ckpt()
                        _set_z(z_try)
                        conv, steps = _run_ss(100)
                        if conv or (steps > 0 and steps != -1):
                            az = z_try
                            _ckpt()
                        else:
                            _rest()
                            break

                    # Try z=1 from best
                    if az > 0.01:
                        _ckpt()
                        _set_z(1.0)
                        conv, steps = _run_ss(100)
                        if conv or (steps > 0 and steps != -1):
                            az = 1.0
                        else:
                            _rest()

                    z_achieved[idx] = az
                    if az > 0:
                        cd[idx] = float(fd.assemble(obs_cd))
                        pc[idx] = float(fd.assemble(obs_pc))
                        if az >= 0.999:
                            last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                    method += f"z-ramp(z={az:.3f})"

            elapsed = time.time() - t0
            z_str = "OK" if z_achieved[idx] >= 0.999 else f"z={z_achieved[idx]:.3f}"
            print(f"  V={v:+.3f}V (phi={ph:+6.2f}): {z_str:>10}  "
                  f"cd={cd[idx]:+.4f}  pc={pc[idx]:+.4f}  "
                  f"[{method}] ({elapsed:.1f}s)")

    p2_time = time.time() - t2
    n_full = int(np.sum(z_achieved >= 0.999))

    print(f"\n{'='*70}")
    print(f"  RESULT: {n_full}/{len(V_RHE)} fully converged (z=1.0)")
    print(f"  Phase 1: {p1_time:.1f}s, Phase 2: {p2_time:.1f}s")
    print(f"{'='*70}")

    if n_full > 0:
        full_mask = z_achieved >= 0.999
        max_v = V_RHE[full_mask].max()
        print(f"  Max anodic V with z=1.0: {max_v:.3f}V vs RHE")
        print(f"  η_r1 at boundary = {(max_v - E_EQ_R1)*1000:.0f} mV")


if __name__ == "__main__":
    main()
