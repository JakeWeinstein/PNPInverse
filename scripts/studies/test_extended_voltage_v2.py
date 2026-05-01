"""Extended voltage v2: fine-grained z=1 continuation with adaptive stepping.

Key ideas:
1. Use VERY fine voltage steps near the convergence boundary (1-5 mV)
2. Adaptive step size: grow when converging easily, shrink when struggling
3. More aggressive SNES: smaller dt, longer t_end, tighter line search
4. Bridge-point style: insert sub-steps between user-requested points
5. Bisection on voltage: if a step fails, halve the voltage jump
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

    # Target grid: we want these specific points
    V_RHE_TARGET = np.sort(np.array([
        -0.50, -0.40, -0.30, -0.20, -0.10,
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.50, 0.60, 0.70,
    ]))[::-1]  # descending

    PHI_HAT_TARGET = V_RHE_TARGET / V_T

    # Tighter SNES for the stiff regime
    SNES_TIGHT = {
        "snes_type": "newtonls", "snes_max_it": 500,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-14,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.2,  # Tighter line search
        "snes_divergence_tolerance": 1e8,   # Detect divergence sooner
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 100,  # More MUMPS memory
    }

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.1, t_end=100.0,   # Smaller dt, longer t_end
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_TIGHT,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)

    print(f"{'='*70}")
    print(f"  EXTENDED VOLTAGE v2: Adaptive Fine-Grained Continuation")
    print(f"  Target: {len(V_RHE_TARGET)} points, V_RHE: [{V_RHE_TARGET.min():.2f}, {V_RHE_TARGET.max():.2f}]")
    print(f"  SNES: max_it=500, maxlambda=0.2, dt=0.1, t_end=100")
    print(f"{'='*70}\n")

    # --- Phase 1: Neutral sweep ---
    print("Phase 1: Neutral sweep...")
    neutral_solutions, mesh_dof, p1_time = _phase1_neutral_sweep(
        sp_list, PHI_HAT_TARGET, max_eta_gap=2.0)
    print()

    # --- Phase 2: Adaptive z=1 voltage continuation ---
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

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
    obs_pc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

    dt_max = dt_initial * 30.0  # Allow more growth

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

    # Results storage
    n_pts = len(PHI_HAT_TARGET)
    cd = np.full(n_pts, np.nan)
    pc = np.full(n_pts, np.nan)
    z_achieved = np.full(n_pts, 0.0)

    # Sort ascending (cathodic first → march anodic)
    asc_idx = np.argsort(PHI_HAT_TARGET)

    last_z1_data = None
    last_z1_phi = None

    # Minimum voltage step for bridge points (in V_T units)
    MIN_PHI_STEP = 0.5  # ~12.8 mV — very fine
    MAX_PHI_STEP = 4.0  # ~103 mV — coarse
    MAX_BISECT = 6       # Maximum bisection depth

    print("Phase 2: Adaptive z=1 voltage continuation\n")
    t2 = time.time()

    with adj.stop_annotating():
        for pos, idx in enumerate(asc_idx):
            ph = float(PHI_HAT_TARGET[idx])
            v = V_RHE_TARGET[idx]
            t0 = time.time()
            method = ""
            success = False

            # --- Z-ramp from neutral (first point or no z=1 history) ---
            if last_z1_data is None:
                if neutral_solutions[idx] is not None:
                    for src, dst in zip(neutral_solutions[idx], U.dat):
                        dst.data[:] = src
                    U_prev.assign(U)
                _set_z(0.0)
                phi_applied_func.assign(ph)
                _ckpt()
                _set_z(1.0)
                conv, steps = _run_ss(200)
                if conv or (steps > 5 and steps != -1):
                    z_achieved[idx] = 1.0
                    cd[idx] = float(fd.assemble(obs_cd))
                    pc[idx] = float(fd.assemble(obs_pc))
                    last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                    last_z1_phi = ph
                    method = f"z-ramp({steps}s)"
                    success = True
                else:
                    _rest()
                    method = "z-ramp-FAIL"

            # --- Adaptive z=1 voltage continuation with bridge points ---
            if not success and last_z1_data is not None:
                phi_gap = ph - last_z1_phi
                # Determine step strategy
                if abs(phi_gap) <= MAX_PHI_STEP:
                    # Small enough gap — try direct
                    for src, dst in zip(last_z1_data, U.dat):
                        dst.data[:] = src
                    U_prev.assign(U)
                    _set_z(1.0)
                    phi_applied_func.assign(ph)
                    _ckpt()
                    conv, steps = _run_ss(80)
                    if conv or (steps > 3 and steps != -1):
                        z_achieved[idx] = 1.0
                        cd[idx] = float(fd.assemble(obs_cd))
                        pc[idx] = float(fd.assemble(obs_pc))
                        last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                        last_z1_phi = ph
                        method = f"z1-direct({steps}s)"
                        success = True
                    else:
                        _rest()

                if not success:
                    # Bridge: subdivide gap into fine steps, with bisection on failure
                    n_bridges = max(2, int(np.ceil(abs(phi_gap) / MIN_PHI_STEP)))
                    bridge_phis = np.linspace(last_z1_phi, ph, n_bridges + 1)[1:]

                    # Start from last z=1 solution
                    for src, dst in zip(last_z1_data, U.dat):
                        dst.data[:] = src
                    U_prev.assign(U)
                    _set_z(1.0)

                    current_phi = last_z1_phi
                    bridge_ok = True

                    for b_phi in bridge_phis:
                        phi_applied_func.assign(b_phi)
                        _ckpt()
                        conv, steps = _run_ss(80)
                        if conv or (steps > 3 and steps != -1):
                            current_phi = b_phi
                            _ckpt()
                        else:
                            # Bisect this bridge segment
                            _rest()
                            seg_lo = current_phi
                            seg_hi = b_phi
                            bisect_ok = False
                            for depth in range(MAX_BISECT):
                                seg_mid = (seg_lo + seg_hi) / 2.0
                                if abs(seg_mid - current_phi) < MIN_PHI_STEP * 0.5:
                                    break
                                phi_applied_func.assign(seg_mid)
                                _ckpt()
                                conv, steps = _run_ss(120)
                                if conv or (steps > 3 and steps != -1):
                                    current_phi = seg_mid
                                    seg_lo = seg_mid
                                    _ckpt()
                                    bisect_ok = True
                                else:
                                    _rest()
                                    seg_hi = seg_mid

                            if bisect_ok and current_phi > seg_lo - 0.01:
                                # Try to reach the original target
                                phi_applied_func.assign(b_phi)
                                _ckpt()
                                conv, steps = _run_ss(120)
                                if conv or (steps > 3 and steps != -1):
                                    current_phi = b_phi
                                    _ckpt()
                                else:
                                    _rest()
                                    bridge_ok = False
                                    break
                            else:
                                bridge_ok = False
                                break

                    if bridge_ok and abs(current_phi - ph) < 1e-6:
                        z_achieved[idx] = 1.0
                        cd[idx] = float(fd.assemble(obs_cd))
                        pc[idx] = float(fd.assemble(obs_pc))
                        last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                        last_z1_phi = ph
                        method = f"bridge({n_bridges}pts)"
                        success = True
                    elif current_phi > last_z1_phi + 0.1:
                        # Partial progress: update last_z1 to furthest bridge
                        last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                        last_z1_phi = current_phi
                        method = f"bridge-partial(→phi={current_phi:.2f})"
                    else:
                        method = f"bridge-FAIL"

            # --- Fallback: z-ramp from neutral with aggressive settings ---
            if not success and neutral_solutions[idx] is not None:
                for src, dst in zip(neutral_solutions[idx], U.dat):
                    dst.data[:] = src
                U_prev.assign(U)
                _set_z(0.0)
                phi_applied_func.assign(ph)

                az = 0.0
                _ckpt()
                _set_z(1.0)
                conv, steps = _run_ss(200)
                if conv or (steps > 5 and steps != -1):
                    az = 1.0
                else:
                    _rest()
                    # Fine z-ramp
                    z_steps = np.linspace(0, 1, 21)[1:]
                    for zt in z_steps:
                        _ckpt()
                        _set_z(zt)
                        conv, steps = _run_ss(100)
                        if conv or (steps > 3 and steps != -1):
                            az = zt
                            _ckpt()
                        else:
                            _rest()
                            break

                z_achieved[idx] = az
                if az > 0:
                    cd[idx] = float(fd.assemble(obs_cd))
                    pc[idx] = float(fd.assemble(obs_pc))
                    if az >= 0.999:
                        last_z1_data = tuple(d.data_ro.copy() for d in U.dat)
                        last_z1_phi = ph
                        success = True
                method += f"→z-ramp(z={az:.3f})"

            elapsed = time.time() - t0
            z_str = "OK" if z_achieved[idx] >= 0.999 else f"z={z_achieved[idx]:.3f}"
            print(f"  V={v:+.3f}V (phi={ph:+7.2f}): {z_str:>10}  "
                  f"cd={cd[idx]:+.4f}  pc={pc[idx]:+.4f}  "
                  f"[{method}] ({elapsed:.1f}s)")

    p2_time = time.time() - t2
    n_full = int(np.sum(z_achieved >= 0.999))

    print(f"\n{'='*70}")
    print(f"  RESULT: {n_full}/{n_pts} fully converged (z=1.0)")
    print(f"  Phase 1: {p1_time:.1f}s, Phase 2: {p2_time:.1f}s, Total: {p1_time+p2_time:.1f}s")
    print(f"{'='*70}")

    full_mask = z_achieved >= 0.999
    if full_mask.any():
        max_v = V_RHE_TARGET[full_mask].max()
        print(f"  Max anodic V with z=1.0: {max_v:.3f}V vs RHE")
        print(f"  η_r1 at boundary = {(max_v - E_EQ_R1)*1000:.0f} mV")
    else:
        print(f"  NO points converged!")


if __name__ == "__main__":
    main()
