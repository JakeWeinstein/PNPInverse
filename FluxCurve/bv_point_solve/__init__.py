"""Per-point BV adjoint solve with warm-start sequential sweep.

Unlike the Robin pipeline (independent per-point solves), the BV point
solver uses voltage continuation: points are solved sequentially from
smallest to largest |eta|, with each point's IC being the previous
point's converged solution.  This is essential for BV convergence at
large overpotentials.

When the phi_applied array contains both positive and negative values
(symmetric voltage range), a two-branch sweep is used: the negative
branch is swept outward from equilibrium first, then the positive branch.
This avoids interleaving cathodic and anodic solutions, which causes
SNES divergence at moderate |eta| due to inverted concentration profiles.
The only sign-change transition occurs near eta = 0, where solutions
are similar.  Bridge points fill any gap between branches.

Performance optimizations (2026-02-28):
- Mesh and function-space creation hoisted outside the per-point loop.
- Warm-started points use a reduced max_steps cap (default 20).
- Per-point timing instrumentation for profiling (setup/forward/adjoint).
- P1: Skip rf(control_state) tape replay; call rf.derivative() directly.
- P2: Cross-evaluation warm-start for the first point.
- P4: Quadratic/linear hybrid predictor step between consecutive voltage points.
- P5: PETSc Jacobian lagging (snes_lag_jacobian).
- P6: Adaptive pseudo-timestep (SER) via mutable fd.Constant dt.
- Bridge points: auto-insert forward-only points to span large eta gaps.
"""

from __future__ import annotations

import copy
import time as _time
import warnings as _warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Forward.bv_solver.validation import validate_solution_state

from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from FluxCurve.config import ForwardRecoveryConfig
from FluxCurve.results import PointAdjointResult
from FluxCurve.recovery import _attempt_phase_state, _relax_solver_options_for_attempt
from FluxCurve.bv_observables import (
    _build_bv_observable_form,
    _build_bv_scalar_target_in_control_space,
    _bv_gradient_controls_to_array,
)
from Forward.steady_state import SteadyStateConfig, configure_bv_solver_params

# Re-export sub-module public API
from .cache import (
    _clear_caches,
    set_parallel_pool,
    close_parallel_pool,
    populate_cache_entry,
    mark_cache_populated_if_complete,
    _cross_eval_cache,
    _all_points_cache,
    _cache_populated,
    _cache_mesh_dof_count,
    _validate_cache_mesh,
    _parallel_pool,
    _SER_GROWTH_CAP,
    _SER_SHRINK,
    _SER_DT_MAX_RATIO,
    _BRIDGE_MAX_STEPS,
    _BRIDGE_SER_DT_MAX_RATIO,
    _WARMSTART_MAX_STEPS,
)
from .predictor import (
    _apply_predictor,
    _solve_bridge_points,
    _build_sweep_order,
)
from .parallel import (
    _solve_cached_fast_path_parallel,
    _solve_cached_fast_path_parallel_multi_obs,
)
from .forward import (
    _solve_cached_fast_path,
    solve_point_objective_and_gradient,
)

__all__ = [
    # Main orchestrator
    "solve_bv_curve_points_with_warmstart",
    # Cache management (public)
    "_clear_caches",
    "set_parallel_pool",
    "close_parallel_pool",
    "populate_cache_entry",
    "mark_cache_populated_if_complete",
    # Cache state (module-level globals)
    "_cross_eval_cache",
    "_all_points_cache",
    "_cache_populated",
    "_parallel_pool",
    # Constants
    "_SER_GROWTH_CAP",
    "_SER_SHRINK",
    "_SER_DT_MAX_RATIO",
    "_BRIDGE_MAX_STEPS",
    "_BRIDGE_SER_DT_MAX_RATIO",
    "_WARMSTART_MAX_STEPS",
    # Predictor / sweep helpers
    "_apply_predictor",
    "_solve_bridge_points",
    "_build_sweep_order",
    # Parallel helpers
    "_solve_cached_fast_path_parallel",
    "_solve_cached_fast_path_parallel_multi_obs",
    # Forward / sequential helpers
    "_solve_cached_fast_path",
    "solve_point_objective_and_gradient",
]


def solve_bv_curve_points_with_warmstart(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    k0_values: Sequence[float],
    blob_initial_condition: bool,
    fail_penalty: float,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    mesh: Any = None,
    alpha_values: Optional[Sequence[float]] = None,
    control_mode: str = "k0",
    a_values: Optional[Sequence[float]] = None,
    max_eta_gap: float = 0.0,
    parallel_pool: Any = None,
) -> List[PointAdjointResult]:
    """Solve all phi_applied points sequentially with warm-start continuation.

    Points are solved outward from equilibrium, with each point's converged
    solution carried forward as the initial condition for the next.  When
    all points share the same sign, the order is ascending |phi_applied|.
    When both positive and negative values are present, a two-branch sweep
    is used: negative eta ascending in |eta|, then positive eta ascending.
    This prevents sign-change warm-start failures at moderate |eta|.
    Each point gets its own adjoint tape for dJ/d(control) computation.

    Parameters
    ----------
    base_solver_params :
        11-element list / SolverParams.
    steady :
        Steady-state convergence settings.
    phi_applied_values :
        Array of dimensionless overpotentials to sweep.
    target_flux :
        Array of target current-density values (same length as phi_applied_values).
    k0_values :
        Current k0 values to evaluate.
    mesh :
        Optional pre-built mesh (e.g. graded rectangle).
    alpha_values :
        Current alpha values (required when control_mode is "alpha" or "joint").
    control_mode : str
        ``"k0"``, ``"alpha"``, ``"joint"``, ``"steric"``, or ``"full"``.
    a_values :
        Current steric a values (required when control_mode is "steric" or "full").
    max_eta_gap : float
        Maximum allowed gap in eta between consecutive solved points.
        When > 0, auto-insert forward-only bridge points to span larger gaps.
        Default 0.0 disables bridge point insertion (backward compatible).

    Returns
    -------
    List[PointAdjointResult]
        One result per phi_applied point, in the ORIGINAL order of phi_applied_values.
    """
    # Import cache module to mutate its globals directly
    from FluxCurve.bv_point_solve import cache as _cache_mod

    import firedrake as fd
    import firedrake.adjoint as adj

    k0_list = [float(v) for v in k0_values]
    alpha_list = [float(v) for v in alpha_values] if alpha_values is not None else None
    a_list = [float(v) for v in a_values] if a_values is not None else None

    if control_mode == "k0":
        n_controls = len(k0_list)
    elif control_mode == "alpha":
        if alpha_list is None:
            raise ValueError("alpha_values required when control_mode='alpha'")
        n_controls = len(alpha_list)
    elif control_mode == "joint":
        if alpha_list is None:
            raise ValueError("alpha_values required when control_mode='joint'")
        n_controls = len(k0_list) + len(alpha_list)
    elif control_mode == "steric":
        # Steric a_vals: n_species controls (one per species)
        n_species = int(base_solver_params.n_species) if hasattr(base_solver_params, 'n_species') else int(base_solver_params[0])
        n_controls = n_species
    elif control_mode == "full":
        # Full: k0 + alpha + steric a
        if alpha_list is None:
            raise ValueError("alpha_values required when control_mode='full'")
        n_species = int(base_solver_params.n_species) if hasattr(base_solver_params, 'n_species') else int(base_solver_params[0])
        n_controls = len(k0_list) + len(alpha_list) + n_species
    else:
        raise ValueError(f"Unknown control_mode '{control_mode}'")
    n_points = len(phi_applied_values)

    abs_tol = float(max(steady.absolute_tolerance, 1e-16))
    rel_tol = float(steady.relative_tolerance)
    max_steps = int(max(1, steady.max_steps))
    required_steady = int(max(1, steady.consecutive_steps))

    max_attempts = max(1, int(forward_recovery.max_attempts))

    # Sort points for warm-start continuation.
    #
    # When all phi_applied have the same sign (or are zero), sorting by |eta|
    # is optimal: we sweep outward from equilibrium monotonically.
    #
    # When phi_applied includes BOTH positive and negative values, naive
    # |eta| sorting interleaves them (e.g., -2, +2, +3, -3, ...).
    # Warm-starting from a cathodic solution to an anodic target (or vice
    # versa) causes SNES divergence because the concentration profiles are
    # inverted (e.g., O2 depleted vs accumulated at the electrode).
    #
    # Fix: "two-branch sweep".  Process negative-or-zero eta ascending in
    # |eta|, then positive eta ascending.  The only sign-change transition
    # happens between the smallest-|eta| negative point and the smallest
    # positive point, which are both near equilibrium and have similar
    # concentration profiles.
    sorted_indices = _build_sweep_order(phi_applied_values)
    results: List[Optional[PointAdjointResult]] = [None] * n_points

    # The first point's params set phi_applied for context construction.
    first_phi = float(phi_applied_values[sorted_indices[0]])

    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )

    # ---------------------------------------------------------------
    # ONE-TIME SETUP: build mesh and function spaces ONCE (Item 5).
    # These are not adjoint-tracked and do not need to be on the tape.
    # Only the Forms and control-Function assignments need taping.
    # ---------------------------------------------------------------
    first_params = configure_bv_solver_params(
        base_solver_params,
        phi_applied=first_phi,
        k0_values=k0_list,
        alpha_values=alpha_list,
        a_values=a_list,
    )
    with adj.stop_annotating():
        base_ctx = bv_build_context(first_params, mesh=mesh)
    # base_ctx now has: mesh, V_scalar, W, U (template), U_prev (template),
    # n_species.  These are reused for every point.
    shared_mesh = base_ctx["mesh"]
    shared_W = base_ctx["W"]

    # Invalidate caches if the mesh has changed (e.g. coarse -> fine).
    _cache_mod._validate_cache_mesh(shared_W.dim())

    # P7: Checkpoint-restart fast path.
    # If we have cached solutions for all points from a previous full sweep,
    # solve each point independently using its cached IC (no sequential sweep).
    # This is much faster because points can start from a nearby solution
    # rather than sweeping from equilibrium each time.
    if _cache_mod._cache_populated and len(_cache_mod._all_points_cache) >= n_points:
        _fast_path_ok = True
        for _i in range(n_points):
            if _i not in _cache_mod._all_points_cache:
                _fast_path_ok = False
                break
        if _fast_path_ok:
            print("[checkpoint] Using cached ICs for all points (fast path)")
            fast_results = _solve_cached_fast_path(
                n_points=n_points,
                n_controls=n_controls,
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                base_solver_params=base_solver_params,
                k0_list=k0_list,
                alpha_list=alpha_list,
                a_list=a_list,
                control_mode=control_mode,
                shared_mesh=shared_mesh,
                blob_initial_condition=blob_initial_condition,
                fail_penalty=fail_penalty,
                forward_recovery=forward_recovery,
                observable_mode=observable_mode,
                observable_reaction_index=observable_reaction_index,
                observable_scale=observable_scale,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                max_steps=max_steps,
                required_steady=required_steady,
                max_attempts=max_attempts,
                mesh=mesh,
                parallel_pool=parallel_pool,
            )
            if fast_results is not None:
                return fast_results
            # Fast path failed on some point; fall through to full sequential sweep
            print("[checkpoint] Fast path failed; falling back to sequential sweep")

    # P2: Cross-evaluation warm-start.  If we have a cached solution
    # for the first point from a previous evaluation, use it instead of
    # cold-starting.  This is safe because adjacent optimizer iterations
    # have similar control values, so the old solution is a good IC.
    first_orig_idx = int(sorted_indices[0])
    carry_U_data = _cache_mod._cross_eval_cache.get(first_orig_idx, None)

    # Predictor step state: 3-point history for quadratic predictor.
    predictor_prev2: Optional[tuple] = None  # (eta, U_data) three points ago
    predictor_prev: Optional[tuple] = None   # (eta, U_data) two points ago
    predictor_curr: Optional[tuple] = None   # (eta, U_data) one point ago

    # Track previous solved eta for bridge point insertion
    prev_solved_eta: Optional[float] = None

    # Hub state: saved near-equilibrium warm-start for second branch.
    # When phi_applied contains both signs, the first branch sweeps outward
    # and saves the state at the smallest |eta| point.  When the second
    # branch starts, carry_U_data is restored to this hub state so the
    # warm-start transition is a small jump near eta = 0.
    hub_U_data: Optional[tuple] = None
    hub_eta: Optional[float] = None
    _first_branch_done = False  # set True after first point of second branch

    # Determine if we have a mixed-sign sweep (need hub logic)
    _has_mixed_signs = (
        (phi_applied_values <= 0).any() and (phi_applied_values > 0).any()
    )

    for sweep_idx, orig_idx in enumerate(sorted_indices):
        phi_applied_i = float(phi_applied_values[orig_idx])
        target_i = float(target_flux[orig_idx])

        # Skip points where the target is NaN (unconverged target data).
        # These contribute zero objective and zero gradient.  We still
        # solve them as forward-only bridge-style points to maintain
        # warm-start continuity, but do NOT record them as inference
        # points (no adjoint, no tape).
        if np.isnan(target_i):
            print(
                f"  [skip] phi={phi_applied_i:+8.4f} target=NaN "
                f"(target generation failed; skipping adjoint)"
            )
            # Still do a forward-only solve to maintain warm-start chain
            if carry_U_data is not None and max_eta_gap > 0 and prev_solved_eta is not None:
                (carry_U_data, predictor_prev, predictor_curr,
                 predictor_prev2, prev_solved_eta) = _solve_bridge_points(
                    prev_solved_eta=prev_solved_eta,
                    next_eta=phi_applied_i,
                    max_eta_gap=max_eta_gap,
                    carry_U_data=carry_U_data,
                    base_solver_params=base_solver_params,
                    k0_list=k0_list,
                    alpha_list=alpha_list,
                    a_list=a_list,
                    shared_mesh=shared_mesh,
                    observable_mode=observable_mode,
                    observable_reaction_index=observable_reaction_index,
                    observable_scale=observable_scale,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    predictor_prev=predictor_prev,
                    predictor_curr=predictor_curr,
                    predictor_prev2=predictor_prev2,
                    blob_initial_condition=blob_initial_condition,
                )
            results[orig_idx] = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i,
                simulated_flux=float("nan"),
                objective=0.0,  # zero contribution to loss
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=0,
                reason="target NaN (skipped)",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            continue

        # ---- Branch transition: restore hub state ----
        # When switching from negative to positive eta (or vice versa),
        # restore the near-equilibrium hub state instead of warm-starting
        # from the last point of the first branch (which may be far from
        # equilibrium, e.g. eta = -28).  Also clear predictor history.
        if (prev_solved_eta is not None
                and carry_U_data is not None
                and not _first_branch_done
                and _has_mixed_signs):
            prev_sign = prev_solved_eta > 0
            curr_sign = phi_applied_i > 0
            if prev_sign != curr_sign:
                _first_branch_done = True
                predictor_prev2 = None
                predictor_prev = None
                predictor_curr = None
                if hub_U_data is not None:
                    carry_U_data = hub_U_data
                    prev_solved_eta = hub_eta

        # ---- Save hub state at first converged point ----
        # The first point of the first branch is the closest to eta = 0
        # (by construction of _build_sweep_order).  Save it for the
        # branch transition.
        # (This is set after convergence below, at sweep_idx == 0.)

        # ---- Bridge point insertion (Phase 1) ----
        # If the gap from the previous solved eta exceeds max_eta_gap,
        # insert forward-only bridge points to carry the warm-start.
        if (max_eta_gap > 0
                and carry_U_data is not None
                and prev_solved_eta is not None):
            (carry_U_data, predictor_prev, predictor_curr,
             predictor_prev2, prev_solved_eta) = _solve_bridge_points(
                prev_solved_eta=prev_solved_eta,
                next_eta=phi_applied_i,
                max_eta_gap=max_eta_gap,
                carry_U_data=carry_U_data,
                base_solver_params=base_solver_params,
                k0_list=k0_list,
                alpha_list=alpha_list,
                a_list=a_list,
                shared_mesh=shared_mesh,
                observable_mode=observable_mode,
                observable_reaction_index=observable_reaction_index,
                observable_scale=observable_scale,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                predictor_prev=predictor_prev,
                predictor_curr=predictor_curr,
                predictor_prev2=predictor_prev2,
                blob_initial_condition=blob_initial_condition,
            )

        # Use reduced max_steps for warm-started points (Item 2).
        is_warmstart = carry_U_data is not None
        effective_max_steps = min(max_steps, _WARMSTART_MAX_STEPS) if is_warmstart else max_steps

        point_result = None

        # Build the attempt sequence: 0=predictor warm-start, 1=simple warm-start,
        # 2..max_attempts+1 = recovery attempts (cold start with relaxed params).
        # This ensures we try simple warm-start before cold start.
        total_attempts = max_attempts + (1 if is_warmstart else 0)
        last_reason = "all attempts failed"

        for attempt_raw in range(total_attempts):
            # Map attempt_raw to the recovery attempt index:
            # attempt_raw 0 = predictor warm-start (attempt=0)
            # attempt_raw 1 = simple warm-start if available (attempt=0, no predictor)
            # attempt_raw 2+ = recovery attempts (attempt=attempt_raw-1)
            if is_warmstart:
                if attempt_raw == 0:
                    attempt = 0
                    use_predictor = True
                elif attempt_raw == 1:
                    attempt = 0
                    use_predictor = False
                else:
                    attempt = attempt_raw - 1
                    use_predictor = False
            else:
                attempt = attempt_raw
                use_predictor = False

            # On recovery attempts (attempt > 0), revert to full max_steps.
            if attempt > 0:
                effective_max_steps = max_steps

            params = configure_bv_solver_params(
                base_solver_params,
                phi_applied=phi_applied_i,
                k0_values=k0_list,
                alpha_values=alpha_list,
                a_values=a_list,
            )
            _params_opts = params.solver_options if hasattr(params, 'solver_options') else params[10]
            if isinstance(_params_opts, dict):
                baseline_options = copy.deepcopy(_params_opts)
                phase, phase_step, _cycle = _attempt_phase_state(attempt, forward_recovery)
                _relax_solver_options_for_attempt(
                    _params_opts,
                    phase=phase,
                    phase_step=phase_step,
                    recovery=forward_recovery,
                    baseline_options=baseline_options,
                )

            # ---- Per-point setup (timed) ----
            t_setup_start = _time.perf_counter()

            # Clear tape and enable annotation for this point
            tape = adj.get_working_tape()
            tape.clear_tape()
            adj.continue_annotation()

            # Reuse the shared mesh and function space (Item 5).
            # build_context creates fresh U/U_prev Functions in shared_W.
            # build_forms creates the UFL residual (must be on the tape).
            ctx = bv_build_context(params, mesh=shared_mesh)
            ctx = bv_build_forms(ctx, params)
            bv_set_initial_conditions(ctx, params, blob=blob_initial_condition)

            # Warm-start: restore last converged state if available.
            if carry_U_data is not None and attempt == 0:
                if use_predictor:
                    _apply_predictor(
                        phi_applied_i, ctx["U"], carry_U_data,
                        predictor_prev, predictor_curr, predictor_prev2,
                        n_species=int(ctx["n_species"]),
                    )
                else:
                    # Simple warm-start: direct copy without predictor
                    for src, dst in zip(carry_U_data, ctx["U"].dat):
                        dst.data[:] = src
                ctx["U_prev"].assign(ctx["U"])

            # Assign k0 values to the adjoint-tracked Functions
            k0_funcs = list(ctx["bv_k0_funcs"])
            for j, k0_f in enumerate(k0_funcs):
                if j < len(k0_list):
                    k0_f.assign(float(k0_list[j]))

            # Assign alpha values to the adjoint-tracked Functions
            alpha_funcs = list(ctx.get("bv_alpha_funcs", []))
            if alpha_list is not None:
                for j, alpha_f in enumerate(alpha_funcs):
                    if j < len(alpha_list):
                        alpha_f.assign(float(alpha_list[j]))

            # Assign steric a values to the adjoint-tracked Functions
            steric_a_funcs = list(ctx.get("steric_a_funcs", []))
            if a_list is not None:
                for j, a_f in enumerate(steric_a_funcs):
                    if j < len(a_list):
                        a_f.assign(float(a_list[j]))

            # Select control functions based on control_mode
            if control_mode == "k0":
                control_funcs = list(k0_funcs)
            elif control_mode == "alpha":
                control_funcs = list(alpha_funcs)
            elif control_mode == "joint":
                control_funcs = list(k0_funcs) + list(alpha_funcs)
            elif control_mode == "steric":
                control_funcs = list(steric_a_funcs)
            elif control_mode == "full":
                control_funcs = list(k0_funcs) + list(alpha_funcs) + list(steric_a_funcs)
            else:
                control_funcs = list(k0_funcs)

            U = ctx["U"]
            U_prev = ctx["U_prev"]
            F_res = ctx["F_res"]
            bcs = ctx["bcs"]

            jac = fd.derivative(F_res, U)
            problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)

            # P5: Jacobian lagging -- reuse Jacobian/preconditioner for
            # 2 consecutive SNES solves before recomputing.  Cuts MUMPS
            # LU factorization frequency in half.  The "persists" flag
            # keeps the lag across PTC time steps (not just Newton iters).
            _sp_opts = params.solver_options if hasattr(params, 'solver_options') else params[10]
            solve_params = dict(_sp_opts) if isinstance(_sp_opts, dict) else {}
            solve_params.setdefault("snes_lag_jacobian", 2)
            solve_params.setdefault("snes_lag_jacobian_persists", True)
            solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_params)

            observable_form = _build_bv_observable_form(
                ctx,
                mode=observable_mode,
                reaction_index=observable_reaction_index,
                scale=float(observable_scale),
            )

            t_setup_elapsed = _time.perf_counter() - t_setup_start

            # ---- Forward solve (timed) ----
            t_fwd_start = _time.perf_counter()

            # P6: Adaptive pseudo-timestep (SER).  Read initial dt from
            # the form's fd.Constant and grow/shrink based on convergence.
            dt_const = ctx.get("dt_const")
            dt_initial = float(dt_const) if dt_const is not None else 1.0
            dt_current = dt_initial
            dt_max = dt_initial * _SER_DT_MAX_RATIO

            prev_flux_val: Optional[float] = None
            steady_count = 0
            rel_metric: Optional[float] = None
            abs_metric: Optional[float] = None
            simulated_flux = float("nan")
            steps_taken = 0
            failed_by_exception = False
            prev_delta: Optional[float] = None

            for step in range(1, effective_max_steps + 1):
                steps_taken = step
                try:
                    solver.solve()
                except Exception as exc:
                    failed_by_exception = True
                    last_reason = f"{type(exc).__name__}: {exc}"
                    break

                with adj.stop_annotating():
                    U_prev.assign(U)

                # Non-annotated assembly for convergence check
                with adj.stop_annotating():
                    simulated_flux = float(fd.assemble(observable_form))

                if prev_flux_val is not None:
                    delta = abs(simulated_flux - prev_flux_val)
                    scale_val = max(abs(simulated_flux), abs(prev_flux_val), abs_tol)
                    rel_metric = delta / scale_val
                    abs_metric = delta
                    is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                    steady_count = steady_count + 1 if is_steady else 0

                    # P6: SER timestep adaptation.
                    if dt_const is not None and prev_delta is not None and delta > 0:
                        ratio = prev_delta / delta
                        if ratio > 1.0:
                            # Residual decreasing: grow dt
                            grow = min(ratio, _SER_GROWTH_CAP)
                            dt_current = min(dt_current * grow, dt_max)
                        else:
                            # Residual increasing: shrink dt
                            dt_current = max(dt_current * _SER_SHRINK, dt_initial)
                        dt_const.assign(dt_current)

                    prev_delta = delta
                else:
                    steady_count = 0

                prev_flux_val = simulated_flux

                if steady_count >= required_steady:
                    break

            # P6: Reset dt to initial value for the next point (each
            # point should start with the base timestep).
            if dt_const is not None:
                dt_const.assign(dt_initial)

            t_fwd_elapsed = _time.perf_counter() - t_fwd_start

            # ---- Adjoint gradient (timed) ----
            t_adj_start = _time.perf_counter()

            if not failed_by_exception and steady_count >= required_steady:
                # Converged -- compute adjoint gradient.
                # Build target as an R-space Function so that fd.assemble
                # returns an adjoint-tracked AdjFloat (required by
                # ReducedFunctional).  Using fd.Constant directly causes
                # fd.assemble to return numpy.float64 which is not an
                # OverloadedType.
                target_ctrl = _build_bv_scalar_target_in_control_space(
                    ctx, target_i, name="target_flux_value",
                    control_mode=control_mode,
                )
                target_scalar = fd.assemble(
                    target_ctrl * fd.dx(domain=ctx["mesh"])
                )
                sim_scalar = fd.assemble(observable_form)
                point_objective = 0.5 * (sim_scalar - target_scalar) ** 2

                controls = [adj.Control(cf) for cf in control_funcs]
                rf = adj.ReducedFunctional(point_objective, controls)
                try:
                    # P1: Skip rf(control_state) -- avoid full tape replay.
                    # The tape already holds valid forward state from the
                    # solve loop above.  rf.derivative() only runs the
                    # backward adjoint sweep (tape.evaluate_adj), which
                    # does NOT re-execute the forward SNES solves.
                    point_objective_value = float(point_objective)
                    point_gradient = _bv_gradient_controls_to_array(
                        rf.derivative(), n_controls
                    )
                except Exception as rf_exc:
                    # Adjoint derivative failed (e.g. PETSc adjoint linear
                    # solve issue).  Treat as a failed point so the
                    # optimizer can continue.
                    print(
                        f"  [adjoint] derivative failed at "
                        f"phi={phi_applied_i:.4f}: {rf_exc}"
                    )
                    failed_by_exception = True
                    last_reason = f"adjoint derivative: {type(rf_exc).__name__}"

                if not failed_by_exception:
                    # Physics validation on converged solution state
                    _n_sp_v = int(ctx["n_species"])
                    _scaling_v = ctx.get("nondim", {})
                    _c_bulk_v = _scaling_v.get("c0_model_vals", [1.0] * _n_sp_v)
                    _z_vals_v = [float(zc) for zc in ctx["z_consts"]]
                    _conv_cfg_v = ctx.get("bv_convergence", {})
                    _eps_c_v = float(_conv_cfg_v.get("conc_floor", 1e-8))
                    _exp_clip_v = float(_conv_cfg_v.get("exponent_clip", 50.0))
                    _sol_vr = validate_solution_state(
                        U,
                        n_species=_n_sp_v,
                        c_bulk=_c_bulk_v,
                        phi_applied=phi_applied_i,
                        z_vals=_z_vals_v,
                        eps_c=_eps_c_v,
                        exponent_clip=_exp_clip_v,
                        is_logc=bool(ctx.get("logc_transform", False)),
                    )
                    for _w in _sol_vr.warnings:
                        _warnings.warn(
                            f"solve_bv_curve_points phi={phi_applied_i:.4f}: {_w}",
                            stacklevel=1,
                        )
                    if not _sol_vr.valid:
                        # Downgrade to failed -- physics violations
                        failed_by_exception = True
                        last_reason = (
                            f"physics validation (solution state): "
                            + "; ".join(_sol_vr.failures)
                        )

                if not failed_by_exception:
                    point_result = PointAdjointResult(
                        phi_applied=phi_applied_i,
                        target_flux=target_i,
                        simulated_flux=float(sim_scalar),
                        objective=point_objective_value,
                        gradient=point_gradient,
                        converged=True,
                        steps_taken=steps_taken,
                        reason="",
                        final_relative_change=rel_metric,
                        final_absolute_change=abs_metric,
                        diagnostics_valid=True,
                    )

                    # Snapshot U for warm-start of next point
                    carry_U_data = tuple(d.data_ro.copy() for d in U.dat)

                    # Update predictor state (3-point history)
                    predictor_prev2 = predictor_prev
                    predictor_prev = predictor_curr
                    predictor_curr = (phi_applied_i, carry_U_data)

                    # Track solved eta for bridge point insertion
                    prev_solved_eta = phi_applied_i

                    # P2: Cache the first point's solution for cross-eval
                    if sweep_idx == 0:
                        _cache_mod._cross_eval_cache[first_orig_idx] = carry_U_data

                    # P7: Cache this point's solution for checkpoint-restart
                    _cache_mod._all_points_cache[orig_idx] = carry_U_data

                    # Save hub state at the first converged point for
                    # branch-transition warm-start (mixed-sign sweep only).
                    if hub_U_data is None and _has_mixed_signs:
                        hub_U_data = carry_U_data
                        hub_eta = phi_applied_i

            t_adj_elapsed = _time.perf_counter() - t_adj_start

            # ---- Profiling output (Item 1) ----
            if is_warmstart and attempt == 0:
                ws_tag = "ws+pred" if use_predictor else "ws"
            else:
                ws_tag = "cold"
            print(
                f"  [timing] phi={phi_applied_i:+8.4f} "
                f"setup={t_setup_elapsed:.2f}s "
                f"fwd={t_fwd_elapsed:.2f}s "
                f"adj={t_adj_elapsed:.2f}s "
                f"steps={steps_taken}/{effective_max_steps} "
                f"({ws_tag})"
            )

            if point_result is not None:
                break  # success on this attempt

            # If this attempt failed, try next recovery phase
            if not failed_by_exception:
                last_reason = "steady-state criterion not satisfied before max_steps"

        if point_result is None:
            # Build a nonzero gradient pointing toward reducing control
            # magnitude so the optimizer has a direction to escape the
            # failure region instead of getting a zero gradient.
            _ctrl_parts = []
            if control_mode in ("k0", "joint", "full"):
                _ctrl_parts.extend(k0_list)
            if control_mode in ("alpha", "joint", "full"):
                _ctrl_parts.extend(alpha_list)
            if control_mode in ("steric", "full"):
                _ctrl_parts.extend(a_list if a_list is not None else [])
            _ctrl_arr = np.asarray(
                _ctrl_parts if _ctrl_parts else [0.0] * n_controls,
                dtype=float,
            )
            _fail_grad = np.sign(_ctrl_arr) * 1.0

            point_result = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i,
                simulated_flux=simulated_flux,
                objective=float(fail_penalty),
                gradient=_fail_grad,
                converged=False,
                steps_taken=steps_taken,
                reason=last_reason,
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            # Even if this point failed, track its eta for bridge logic
            if prev_solved_eta is None:
                prev_solved_eta = phi_applied_i

        results[orig_idx] = point_result

    # P7: Mark cache as populated after a full sequential sweep.
    # All converged points have been cached in _all_points_cache above.
    n_cached = sum(1 for i in range(n_points) if i in _cache_mod._all_points_cache)
    if n_cached == n_points:
        _cache_mod._cache_populated = True

    return [r for r in results]  # type: ignore[misc]
