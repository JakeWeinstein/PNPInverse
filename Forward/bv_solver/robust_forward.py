"""Robust parallel forward solver with charge continuation.

Parallelizes Phase 2 (per-point z-ramp) of the charge continuation algorithm
across multiple worker processes. Each worker builds its own mesh/context and
independently ramps z from 0→1 for a single voltage point.

Architecture:
    Phase 1 (sequential, fast):  Neutral sweep at z=0 across voltage grid
    Phase 2 (PARALLEL):          Per-point z-ramp 0→1 via multiprocessing
    Phase 3:                     Extract observables + optionally populate IC cache

Usage::

    from Forward.bv_solver.robust_forward import solve_curve_robust

    result = solve_curve_robust(
        solver_params=sp,
        phi_applied_values=phi_hat,
        observable_scale=-I_SCALE,
        n_workers=8,
        charge_steps=15,
        max_eta_gap=2.0,
    )
    # result.cd, result.pc, result.n_converged, result.U_data_list

For integration with the FluxCurve inference IC cache::

    from Forward.bv_solver.robust_forward import populate_ic_cache_robust

    n_cached = populate_ic_cache_robust(
        solver_params=sp,
        phi_applied_values=phi_hat,
        n_workers=8,
    )
    # FluxCurve fast-path now has ICs for all points
"""
from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustCurveResult:
    """Result from solve_curve_robust."""
    phi_applied: np.ndarray          # Voltage grid (phi_hat)
    cd: np.ndarray                   # Current density (already scaled)
    pc: np.ndarray                   # Peroxide current (already scaled)
    z_achieved: np.ndarray           # Per-point achieved z-factor
    U_data_list: List[Optional[tuple]]  # Per-point converged solution data
    n_converged: int                 # Number of fully converged (z>=1) points
    n_total: int                     # Total voltage points
    phase1_time: float               # Phase 1 wall-clock time
    phase2_time: float               # Phase 2 wall-clock time
    validation_failures: np.ndarray = None  # bool array, True if physics failed for that point


# ---------------------------------------------------------------------------
# Phase 1: Sequential neutral sweep (z=0)
# ---------------------------------------------------------------------------

def _phase1_neutral_sweep(
    solver_params: Any,
    phi_applied_values: np.ndarray,
    *,
    max_eta_gap: float = 2.0,
) -> Tuple[List[tuple], float]:
    """Run Phase 1: neutral sweep at z=0 to get initial conditions.

    Returns (neutral_solutions, mesh_dof_count) where neutral_solutions[i]
    is a tuple of numpy arrays for point i.
    """
    import time
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions

    t0 = time.time()

    n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(solver_params, mesh=mesh)
        ctx = build_forms(ctx, solver_params)
        set_initial_conditions(ctx, solver_params)

    W = ctx["W"]
    n = ctx["n_species"]
    U = ctx["U"]
    U_prev = ctx["U_prev"]

    # Zero out charge coupling for neutral solve
    z_consts = ctx.get("z_consts")
    if z_consts is not None:
        for zc in z_consts:
            zc.assign(0.0)

    phi_applied_func = ctx.get("phi_applied_func")
    dt_const = ctx.get("dt_const")
    dt_initial = float(dt)

    # Build solver
    with adj.stop_annotating():
        F_res = ctx["F_res"]
        bcs = ctx["bcs"]
        J_form = fd.derivative(F_res, U)
        sp_dict = {}
        if isinstance(params, dict):
            for k, v in params.items():
                if k.startswith(("snes_", "ksp_", "pc_", "mat_")):
                    sp_dict[k] = v
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=sp_dict)

    # Observable for convergence detection
    from Forward.bv_solver.observables import _build_bv_observable_form
    observable_form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0)

    # SER constants
    _SER_GROWTH_CAP = 4.0
    _SER_SHRINK = 0.5
    _SER_DT_MAX_RATIO = 20.0
    _STEADY_REL_TOL = 1e-4
    _STEADY_ABS_TOL = 1e-8
    _STEADY_CONSEC = 4
    _COLD_MAX_STEPS = 100
    _WARM_MAX_STEPS = 20

    dt_max = dt_initial * _SER_DT_MAX_RATIO

    def _run_to_ss(max_steps):
        dt_curr = dt_initial
        dt_const.assign(dt_initial)
        prev_flux = None
        prev_delta = None
        steady = 0
        for step in range(1, max_steps + 1):
            try:
                solver.solve()
            except Exception:
                return False, -1
            U_prev.assign(U)
            fv = float(fd.assemble(observable_form))
            if prev_flux is not None:
                delta = abs(fv - prev_flux)
                sc = max(abs(fv), abs(prev_flux), _STEADY_ABS_TOL)
                is_s = (delta / sc <= _STEADY_REL_TOL) or (delta <= _STEADY_ABS_TOL)
                steady = steady + 1 if is_s else 0
                if prev_delta is not None and delta > 0:
                    r = prev_delta / delta
                    if r > 1.0:
                        dt_curr = min(dt_curr * min(r, _SER_GROWTH_CAP), dt_max)
                    else:
                        dt_curr = max(dt_curr * _SER_SHRINK, dt_initial)
                    dt_const.assign(dt_curr)
                prev_delta = delta
            prev_flux = fv
            if steady >= _STEADY_CONSEC:
                return True, step
        return False, max_steps

    # Build sweep order: start from eta=0, branch outward
    from Forward.bv_solver.sweep_order import _build_sweep_order
    sweep_idx = _build_sweep_order(phi_applied_values)

    n_pts = len(phi_applied_values)
    neutral_solutions = [None] * n_pts

    # Hub state (near equilibrium) for branch transitions
    hub_U_data = None
    prev_eta = 0.0
    predictor_prev2 = predictor_prev = predictor_curr = None

    with adj.stop_annotating():
        for pos, orig_idx in enumerate(sweep_idx):
            eta_i = float(phi_applied_values[orig_idx])

            # Branch transition detection
            if pos > 0 and np.sign(eta_i) != np.sign(prev_eta) and np.sign(eta_i) != 0:
                if hub_U_data is not None:
                    for src, dst in zip(hub_U_data, U.dat):
                        dst.data[:] = src
                    U_prev.assign(U)
                predictor_prev2 = predictor_prev = predictor_curr = None

            # Bridge points for large gaps
            if pos > 0:
                gap = abs(eta_i - prev_eta)
                if gap > max_eta_gap:
                    n_bridge = max(1, int(np.ceil(gap / max_eta_gap)))
                    bridges = np.linspace(prev_eta, eta_i, n_bridge + 1)[1:-1]
                    for br_eta in bridges:
                        phi_applied_func.assign(br_eta)
                        _run_to_ss(_WARM_MAX_STEPS)

            # Solve at target voltage
            phi_applied_func.assign(eta_i)
            max_s = _COLD_MAX_STEPS if pos == 0 else _WARM_MAX_STEPS
            conv, steps = _run_to_ss(max_s)

            # Snapshot solution
            neutral_solutions[orig_idx] = tuple(d.data_ro.copy() for d in U.dat)

            # Save hub state (first solved point, near equilibrium)
            if hub_U_data is None:
                hub_U_data = neutral_solutions[orig_idx]

            prev_eta = eta_i
            if pos % 10 == 0 or pos == n_pts - 1:
                print(f"[Phase 1] {pos+1}/{n_pts}  eta={eta_i:+.4f} (neutral, z=0)")

    elapsed = time.time() - t0
    mesh_dof_count = W.dim()
    print(f"[Phase 1] Complete: {n_pts} neutral solutions cached ({elapsed:.1f}s)")
    return neutral_solutions, mesh_dof_count, elapsed


# ---------------------------------------------------------------------------
# Phase 2 worker: z-ramp for a single voltage point
# ---------------------------------------------------------------------------

def _z_ramp_worker(args: dict) -> dict:
    """Worker process: z-ramp 0→1 for a single voltage point.

    Each worker builds its own mesh+context to avoid Firedrake
    shared-state issues across processes.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    import time
    import firedrake as fd
    import pyadjoint as adj

    orig_idx = args["orig_idx"]
    phi_hat = args["phi_hat"]
    neutral_U_data = args["neutral_U_data"]
    sp_list = args["solver_params"]
    observable_scale = args["observable_scale"]
    charge_steps = args.get("charge_steps", 15)
    min_delta_z = args.get("min_delta_z", 0.005)

    t0 = time.time()

    n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = sp_list

    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)
        set_initial_conditions(ctx, sp_list)

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    z_consts = ctx.get("z_consts")
    phi_applied_func = ctx.get("phi_applied_func")
    dt_const = ctx.get("dt_const")
    dt_initial = float(dt)

    # Load neutral IC
    for src, dst in zip(neutral_U_data, U.dat):
        dst.data[:] = src
    U_prev.assign(U)

    # Set voltage
    phi_applied_func.assign(phi_hat)

    # Start with z=0
    z_nominal = [float(z_vals[i]) for i in range(n)]
    if z_consts is not None:
        for i, zc in enumerate(z_consts):
            zc.assign(0.0)

    # Build solver
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]
    J_form = fd.derivative(F_res, U)
    sp_dict = {}
    if isinstance(params, dict):
        for k, v in params.items():
            if k.startswith(("snes_", "ksp_", "pc_", "mat_")):
                sp_dict[k] = v
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=sp_dict)

    observable_form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0)

    # SER time-stepping
    _SER_GROWTH_CAP = 4.0
    _SER_SHRINK = 0.5
    _STEADY_REL_TOL = 1e-4
    _STEADY_ABS_TOL = 1e-8
    _STEADY_CONSEC = 4
    _MAX_STEPS = 100
    dt_max = dt_initial * 20.0

    def _run_to_ss(max_steps):
        dt_curr = dt_initial
        dt_const.assign(dt_initial)
        prev_flux = None
        prev_delta = None
        steady = 0
        for step in range(1, max_steps + 1):
            try:
                solver.solve()
            except Exception:
                return False, -1
            U_prev.assign(U)
            fv = float(fd.assemble(observable_form))
            if prev_flux is not None:
                delta = abs(fv - prev_flux)
                sc = max(abs(fv), abs(prev_flux), _STEADY_ABS_TOL)
                is_s = (delta / sc <= _STEADY_REL_TOL) or (delta <= _STEADY_ABS_TOL)
                steady = steady + 1 if is_s else 0
                if prev_delta is not None and delta > 0:
                    r = prev_delta / delta
                    if r > 1.0:
                        dt_curr = min(dt_curr * min(r, _SER_GROWTH_CAP), dt_max)
                    else:
                        dt_curr = max(dt_curr * _SER_SHRINK, dt_initial)
                    dt_const.assign(dt_curr)
                prev_delta = delta
            prev_flux = fv
            if steady >= _STEADY_CONSEC:
                return True, step
        return False, max_steps

    def _try_z(z_val):
        if z_consts is not None:
            for i in range(n):
                z_consts[i].assign(z_nominal[i] * z_val)
        converged, steps = _run_to_ss(_MAX_STEPS)
        if converged:
            return True, True
        if steps == _MAX_STEPS:
            return False, True  # Budget exhausted but usable
        return False, False     # SNES failure

    # Checkpoint/restore for backtracking
    U_ckpt = fd.Function(ctx["W"])
    U_prev_ckpt = fd.Function(ctx["W"])

    def _ckpt():
        U_ckpt.assign(U)
        U_prev_ckpt.assign(U_prev)

    def _rest():
        U.assign(U_ckpt)
        U_prev.assign(U_prev_ckpt)

    # ---- Adaptive z-ramp (same algorithm as grid_charge_continuation) ----
    achieved_z = 0.0

    # Stage 1: Try z=1.0 directly
    _ckpt()
    conv, usable = _try_z(1.0)
    if usable:
        achieved_z = 1.0
    else:
        _rest()

        # Stage 2: Binary search for foothold
        search_z = 0.5
        search_hi = 1.0
        for _ in range(4):
            _ckpt()
            _, usable = _try_z(search_z)
            if usable:
                achieved_z = search_z
                _ckpt()
                break
            else:
                _rest()
                search_hi = search_z
                search_z = search_z / 2.0
                if search_z < min_delta_z:
                    break

        # Stage 3: Geometric acceleration
        if achieved_z >= min_delta_z:
            for _ in range(20):
                if achieved_z >= 1.0 - 1e-6:
                    break
                remaining = 1.0 - achieved_z
                if remaining < min_delta_z:
                    break

                _ckpt()
                _, usable = _try_z(1.0)
                if usable:
                    achieved_z = 1.0
                    break
                _rest()

                mid_z = achieved_z + remaining / 2.0
                _ckpt()
                _, usable = _try_z(mid_z)
                if usable:
                    achieved_z = mid_z
                    _ckpt()
                    continue
                _rest()

                n_fine = max(2, charge_steps // 2)
                fine = np.linspace(achieved_z, mid_z, n_fine + 1)[1:]
                for zt in fine:
                    _ckpt()
                    _, usable = _try_z(zt)
                    if usable:
                        achieved_z = zt
                        _ckpt()
                    else:
                        _rest()
                        if zt - achieved_z < min_delta_z:
                            break

    # Extract observables from final converged state
    cd_val = float('nan')
    pc_val = float('nan')
    U_data = None
    phys_valid = True
    phys_messages: list[str] = []

    if achieved_z > 0:
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=observable_scale)
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)
        cd_val = float(fd.assemble(form_cd))
        pc_val = float(fd.assemble(form_pc))
        U_data = tuple(d.data_ro.copy() for d in U.dat)

        # --- Physics validation ---
        import warnings as _warn
        from .validation import validate_solution_state, validate_observables

        eps_c = params.get("bv_convergence", {}).get("conc_floor", 1e-8) if isinstance(params, dict) else 1e-8
        exponent_clip = params.get("bv_convergence", {}).get("exponent_clip", 50.0) if isinstance(params, dict) else 50.0
        species_names = params.get("species_names") if isinstance(params, dict) else None

        state_vr = validate_solution_state(
            U,
            n_species=n,
            c_bulk=list(c0),
            phi_applied=phi_hat,
            z_vals=list(z_vals),
            eps_c=eps_c,
            exponent_clip=exponent_clip,
            species_names=species_names,
        )

        # Estimate nondimensional diffusion-limited current
        I_lim = 2.0 * max(c0)
        obs_vr = validate_observables(
            cd_val, pc_val,
            I_lim=I_lim,
            phi_applied=phi_hat,
            V_T=1.0,  # already nondimensional
        )

        if not state_vr.valid or not obs_vr.valid:
            phys_valid = False
            phys_messages = state_vr.failures + obs_vr.failures
            for msg in phys_messages:
                _warn.warn(f"[z_ramp pt {orig_idx}] {msg}")

        phys_messages += state_vr.warnings + obs_vr.warnings

    elapsed = time.time() - t0
    converged = achieved_z >= 1.0 - 1e-6

    # Physics failure overrides convergence status
    if not phys_valid:
        converged = False

    return {
        "orig_idx": orig_idx,
        "phi_hat": phi_hat,
        "cd": cd_val,
        "pc": pc_val,
        "z_achieved": achieved_z,
        "converged": converged,
        "U_data": U_data,
        "elapsed": elapsed,
        "_phys_valid": phys_valid,
        "_phys_messages": phys_messages,
    }


# ---------------------------------------------------------------------------
# Main entry point: robust parallel curve solve
# ---------------------------------------------------------------------------

def solve_curve_robust(
    solver_params: Any,
    phi_applied_values: np.ndarray,
    observable_scale: float,
    *,
    n_workers: Optional[int] = None,
    charge_steps: int = 15,
    max_eta_gap: float = 2.0,
    min_delta_z: float = 0.005,
) -> RobustCurveResult:
    """Solve forward I-V curve using parallel charge continuation.

    Phase 1 (sequential):  Neutral sweep at z=0 for all voltage points.
    Phase 2 (parallel):    Per-point z-ramp 0→1 via multiprocessing.

    Parameters
    ----------
    solver_params : SolverParams
        Fully configured solver parameters (with E_eq, k0, alpha, etc.)
    phi_applied_values : array
        Dimensionless voltage grid (phi_hat = V_RHE / V_T), descending.
    observable_scale : float
        Scale factor for observables (typically -I_SCALE for mA/cm²).
    n_workers : int, optional
        Number of parallel workers for Phase 2. Default: cpu_count - 1.
    charge_steps : int
        Fine-stepping count for z-ramp.
    max_eta_gap : float
        Bridge point threshold for Phase 1.
    min_delta_z : float
        Minimum z-increment before giving up.

    Returns
    -------
    RobustCurveResult
    """
    import time

    n_pts = len(phi_applied_values)
    if n_workers is None:
        n_workers = min(n_pts, max(1, (os.cpu_count() or 4) - 1))

    # Convert solver_params to list for pickling
    sp_list = list(solver_params)

    print(f"\n[robust_forward] Solving {n_pts} points with {n_workers} workers")
    print(f"  phi_hat range: [{phi_applied_values.min():.1f}, {phi_applied_values.max():.1f}]")

    # --- Phase 1: Sequential neutral sweep ---
    neutral_solutions, mesh_dof, p1_time = _phase1_neutral_sweep(
        sp_list, phi_applied_values, max_eta_gap=max_eta_gap)

    # --- Phase 2: Parallel z-ramp ---
    t2_start = time.time()

    tasks = []
    for i in range(n_pts):
        if neutral_solutions[i] is None:
            continue
        tasks.append({
            "orig_idx": i,
            "phi_hat": float(phi_applied_values[i]),
            "neutral_U_data": neutral_solutions[i],
            "solver_params": sp_list,
            "observable_scale": observable_scale,
            "charge_steps": charge_steps,
            "min_delta_z": min_delta_z,
        })

    cd = np.full(n_pts, np.nan)
    pc = np.full(n_pts, np.nan)
    z_achieved = np.full(n_pts, 0.0)
    U_data_list: List[Optional[tuple]] = [None] * n_pts
    validation_failures = np.zeros(n_pts, dtype=bool)

    def _store_result(res: dict) -> None:
        i = res["orig_idx"]
        cd[i] = res["cd"]
        pc[i] = res["pc"]
        z_achieved[i] = res["z_achieved"]
        U_data_list[i] = res["U_data"]
        if not res.get("_phys_valid", True):
            validation_failures[i] = True

    if n_workers <= 1:
        # Sequential fallback (useful for debugging)
        for task in tasks:
            res = _z_ramp_worker(task)
            _store_result(res)
    else:
        ctx_mp = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx_mp) as pool:
            futures = {pool.submit(_z_ramp_worker, t): t["orig_idx"] for t in tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                    _store_result(res)
                    status = "OK" if res["converged"] else f"z={res['z_achieved']:.3f}"
                    if not res.get("_phys_valid", True):
                        status += " PHYS-FAIL"
                    if res["orig_idx"] % 5 == 0 or not res["converged"]:
                        print(f"  [Phase 2] pt {res['orig_idx']}: "
                              f"phi={res['phi_hat']:+.2f} {status} ({res['elapsed']:.1f}s)")
                except Exception as exc:
                    print(f"  [Phase 2] pt {idx}: EXCEPTION {exc}")

    p2_time = time.time() - t2_start
    n_conv = int(np.sum(z_achieved >= 1.0 - 1e-6))

    print(f"[Phase 2] Complete: {n_conv}/{n_pts} fully converged ({p2_time:.1f}s)")

    # Physics validation summary
    n_phys_fail = int(np.sum(validation_failures))
    if n_phys_fail > 0:
        print(f"[Phase 2] {n_phys_fail}/{n_pts} points failed physics validation")

    print(f"[robust_forward] Total: {p1_time + p2_time:.1f}s "
          f"(Phase 1: {p1_time:.1f}s, Phase 2: {p2_time:.1f}s)")

    return RobustCurveResult(
        phi_applied=phi_applied_values,
        cd=cd, pc=pc, z_achieved=z_achieved,
        U_data_list=U_data_list,
        n_converged=n_conv, n_total=n_pts,
        phase1_time=p1_time, phase2_time=p2_time,
        validation_failures=validation_failures,
    )


# ---------------------------------------------------------------------------
# IC cache integration: populate FluxCurve cache from robust solve
# ---------------------------------------------------------------------------

def populate_ic_cache_robust(
    solver_params: Any,
    phi_applied_values: np.ndarray,
    observable_scale: float = -1.0,
    *,
    n_workers: Optional[int] = None,
    charge_steps: int = 15,
    max_eta_gap: float = 2.0,
) -> Tuple[int, RobustCurveResult]:
    """Run robust solve and populate the FluxCurve IC cache.

    Returns (n_cached, result) where n_cached is the number of fully
    converged points whose ICs are now in the FluxCurve cache.
    """
    from FluxCurve.bv_point_solve import (
        populate_cache_entry,
        mark_cache_populated_if_complete,
        _clear_caches,
    )

    _clear_caches()

    result = solve_curve_robust(
        solver_params, phi_applied_values, observable_scale,
        n_workers=n_workers, charge_steps=charge_steps,
        max_eta_gap=max_eta_gap,
    )

    # Populate IC cache with converged solutions (skip physics failures)
    n_cached = 0
    n_phys_skipped = 0
    vf = result.validation_failures
    for i in range(result.n_total):
        if result.U_data_list[i] is not None and result.z_achieved[i] >= 1.0 - 1e-6:
            if vf is not None and vf[i]:
                n_phys_skipped += 1
                continue  # bad solutions in the cache poison subsequent warm-starts
            populate_cache_entry(i, result.U_data_list[i], _infer_mesh_dof_count(result.U_data_list[i]))
            n_cached += 1

    if n_cached == result.n_total:
        mark_cache_populated_if_complete(result.n_total)

    msg = f"[robust_forward] IC cache populated: {n_cached}/{result.n_total}"
    if n_phys_skipped > 0:
        msg += f" ({n_phys_skipped} skipped due to physics validation failure)"
    print(msg)
    return n_cached, result


def _infer_mesh_dof_count(U_data: tuple) -> int:
    """Infer total DOF count from U_data tuple."""
    return sum(arr.size for arr in U_data)
