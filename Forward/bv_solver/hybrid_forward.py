"""Hybrid forward solver: z=0 for onset + z=1 for cathodic.

Combines the strengths of both regimes:
- z=0 (neutral): Converges everywhere, captures BV kinetics for onset shape
- z=1 (charged): Accurate electromigration for cathodic transport-limited regime

The solver automatically selects the best z for each voltage point:
- V > V_TRANSITION: z=0 (neutral) — Poisson coupling fails here
- V <= V_TRANSITION: z=1 (charged) via charge continuation

Usage::

    from Forward.bv_solver.hybrid_forward import solve_curve_hybrid

    result = solve_curve_hybrid(
        solver_params=sp,
        phi_applied_values=phi_hat,
        observable_scale=-I_SCALE,
        v_transition=0.10,  # V vs RHE; z=0 above this, z=1 below
    )
"""
from __future__ import annotations

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .robust_forward import RobustCurveResult


def solve_curve_hybrid(
    solver_params: Any,
    phi_applied_values: np.ndarray,
    observable_scale: float,
    *,
    v_transition: float = 0.10,
    n_workers: Optional[int] = None,
    charge_steps: int = 15,
    max_eta_gap: float = 2.0,
) -> RobustCurveResult:
    """Solve forward I-V curve using hybrid z=0/z=1 strategy.

    Parameters
    ----------
    solver_params : SolverParams or list
        Fully configured params with E_eq, k0, alpha.
    phi_applied_values : array
        Dimensionless voltages (phi_hat = V_RHE / V_T), descending.
    observable_scale : float
        Scale for observables (typically -I_SCALE).
    v_transition : float
        V vs RHE threshold: z=0 above, z=1 below. Default 0.10V.
    n_workers : int, optional
        Workers for parallel z=1 charge continuation.
    charge_steps, max_eta_gap : int, float
        Charge continuation parameters for z=1 points.

    Returns
    -------
    RobustCurveResult with cd, pc, z_achieved for all points.
    """
    from scripts._bv_common import V_T

    n_pts = len(phi_applied_values)
    phi_transition = v_transition / V_T

    # Split points into z=0 (onset) and z=1 (cathodic)
    z0_mask = phi_applied_values > phi_transition
    z1_mask = ~z0_mask

    z0_indices = np.where(z0_mask)[0]
    z1_indices = np.where(z1_mask)[0]
    z0_phis = phi_applied_values[z0_mask]
    z1_phis = phi_applied_values[z1_mask]

    print(f"\n[hybrid] {n_pts} points: {len(z0_indices)} z=0 (V>{v_transition}V), "
          f"{len(z1_indices)} z=1 (V<={v_transition}V)")

    cd = np.full(n_pts, np.nan)
    pc = np.full(n_pts, np.nan)
    z_achieved = np.full(n_pts, 0.0)
    U_data_list: List[Optional[tuple]] = [None] * n_pts

    t_start = time.time()

    # --- Z=0 points: neutral solve (fast, sequential) ---
    if len(z0_indices) > 0:
        t_z0 = time.time()
        _solve_z0_points(
            solver_params, phi_applied_values, z0_indices, observable_scale,
            cd, pc, z_achieved, U_data_list)
        z0_time = time.time() - t_z0
        n_z0_ok = sum(1 for i in z0_indices if not np.isnan(cd[i]))
        print(f"[hybrid] z=0 done: {n_z0_ok}/{len(z0_indices)} ({z0_time:.1f}s)")

    # --- Z=1 points: charge continuation (parallel) ---
    if len(z1_indices) > 0:
        from .robust_forward import solve_curve_robust
        t_z1 = time.time()

        # Build a sub-array for the z=1 points
        z1_result = solve_curve_robust(
            solver_params, z1_phis, observable_scale,
            n_workers=n_workers, charge_steps=charge_steps,
            max_eta_gap=max_eta_gap,
        )

        # Map back to original indices
        for local_i, global_i in enumerate(z1_indices):
            cd[global_i] = z1_result.cd[local_i]
            pc[global_i] = z1_result.pc[local_i]
            z_achieved[global_i] = z1_result.z_achieved[local_i]
            U_data_list[global_i] = z1_result.U_data_list[local_i]

        z1_time = time.time() - t_z1
        print(f"[hybrid] z=1 done: {z1_result.n_converged}/{len(z1_indices)} ({z1_time:.1f}s)")

    total_time = time.time() - t_start
    n_converged = sum(1 for i in range(n_pts) if z_achieved[i] >= 0.999 or
                      (z0_mask[i] and not np.isnan(cd[i])))

    # For z=0 points, mark as "converged" (z=0 is the correct z for those points)
    for i in z0_indices:
        if not np.isnan(cd[i]):
            z_achieved[i] = 0.0  # z=0 is correct, not a failure

    # Physics validation summary across both regimes
    n_nan_total = int(np.sum(np.isnan(cd)))
    if n_nan_total > 0:
        print(f"[hybrid] {n_nan_total}/{n_pts} points have NaN observables "
              f"(includes physics validation failures)")

    print(f"[hybrid] Total: {n_converged}/{n_pts} usable ({total_time:.1f}s)")

    return RobustCurveResult(
        phi_applied=phi_applied_values,
        cd=cd, pc=pc, z_achieved=z_achieved,
        U_data_list=U_data_list,
        n_converged=n_converged, n_total=n_pts,
        phase1_time=0, phase2_time=total_time,
    )


def _solve_z0_points(solver_params, all_phis, z0_indices, observable_scale,
                     cd_out, pc_out, z_out, U_data_out):
    """Solve z=0 (neutral) points sequentially with warm-starting."""
    import warnings as _warn
    import firedrake as fd
    import pyadjoint as adj
    from .forms import build_context, build_forms, set_initial_conditions
    from .observables import _build_bv_observable_form
    from .sweep_order import _build_sweep_order
    from .validation import validate_solution_state, validate_observables
    from . import make_graded_rectangle_mesh

    sp_list = list(solver_params)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, _, c0, phi0, params = sp_list
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
    n = n_species

    eps_c = params.get("bv_convergence", {}).get("conc_floor", 1e-8) if isinstance(params, dict) else 1e-8
    exponent_clip = params.get("bv_convergence", {}).get("exponent_clip", 50.0) if isinstance(params, dict) else 50.0
    species_names = params.get("species_names") if isinstance(params, dict) else None

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)
        set_initial_conditions(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx.get("z_consts"); paf = ctx.get("phi_applied_func")
    dtc = ctx.get("dt_const"); dti = float(dt)

    # Force z=0
    for zci in zc:
        zci.assign(0.0)

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)
    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)
    opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

    dt_max = dti * 20.0

    def _ss(max_steps):
        dc = dti; dtc.assign(dti); pf = pd = None; sc = 0
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
                    dc = min(dc*min(r,4.0),dt_max) if r>1 else max(dc*0.5,dti)
                    dtc.assign(dc)
                pd = d
            pf = fv
            if sc >= 4: return True, s
        return False, max_steps

    # Get phis for z=0 points, sorted for sweep
    z0_phis = all_phis[z0_indices]
    sweep = _build_sweep_order(z0_phis)
    hub = None; pe = 0.0
    I_lim = 2.0 * max(c0)
    n_phys_fail = 0

    with adj.stop_annotating():
        for p, local_oi in enumerate(sweep):
            global_i = z0_indices[local_oi]
            ei = float(all_phis[global_i])

            if p > 0 and np.sign(ei) != np.sign(pe) and np.sign(ei) != 0 and hub:
                for s, d in zip(hub, U.dat): d.data[:] = s
                Up.assign(U)
            if p > 0 and abs(ei-pe) > 2.0:
                for br in np.linspace(pe, ei, max(2,int(abs(ei-pe)/2.0))+1)[1:-1]:
                    paf.assign(br); _ss(20)

            paf.assign(ei)
            ok, steps = _ss(100 if p==0 else 20)
            if ok:
                cd_val = float(fd.assemble(ocd))
                pc_val = float(fd.assemble(opc))

                # Physics validation on observables
                obs_vr = validate_observables(
                    cd_val, pc_val, I_lim=I_lim, phi_applied=ei, V_T=1.0)

                # Physics validation on solution state
                state_vr = validate_solution_state(
                    U, n_species=n, c_bulk=list(c0), phi_applied=ei,
                    z_vals=list(z_vals), eps_c=eps_c,
                    exponent_clip=exponent_clip, species_names=species_names)

                if not state_vr.valid or not obs_vr.valid:
                    # Physics failure: mark as NaN, don't store U_data
                    n_phys_fail += 1
                    all_msgs = state_vr.failures + obs_vr.failures
                    for msg in all_msgs:
                        _warn.warn(f"[z0 pt {global_i}] {msg}")
                    cd_out[global_i] = float('nan')
                    pc_out[global_i] = float('nan')
                    # Don't store U_data for physics failures
                else:
                    cd_out[global_i] = cd_val
                    pc_out[global_i] = pc_val
                    z_out[global_i] = 0.0  # z=0 is correct for these points
                    U_data_out[global_i] = tuple(d.data_ro.copy() for d in U.dat)

            if not hub:
                hub = tuple(d.data_ro.copy() for d in U.dat)
            pe = ei

    if n_phys_fail > 0:
        print(f"[hybrid z=0] {n_phys_fail}/{len(z0_indices)} points failed physics validation")
