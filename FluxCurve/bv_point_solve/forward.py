"""Core single-point forward solve with cached IC (sequential fast path)."""

from __future__ import annotations

import time as _time
import warnings as _warnings
from typing import Any, List, Optional, Sequence

import numpy as np

from Forward.bv_solver.validation import validate_solution_state

from .cache import (
    _SER_DT_MAX_RATIO,
    _SER_GROWTH_CAP,
    _SER_SHRINK,
    _WARMSTART_MAX_STEPS,
    _all_points_cache,
    _parallel_pool,
)
from .parallel import _solve_cached_fast_path_parallel
from FluxCurve.config import ForwardRecoveryConfig
from FluxCurve.results import PointAdjointResult
from FluxCurve.bv_observables import (
    _build_bv_observable_form,
    _build_bv_scalar_target_in_control_space,
    _bv_gradient_controls_to_array,
)


def _solve_cached_fast_path(
    *,
    n_points: int,
    n_controls: int,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    base_solver_params: Sequence[object],
    k0_list: list,
    alpha_list: Optional[list],
    a_list: Optional[list],
    control_mode: str,
    shared_mesh: Any,
    blob_initial_condition: bool,
    fail_penalty: float,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    abs_tol: float,
    rel_tol: float,
    max_steps: int,
    required_steady: int,
    max_attempts: int,
    mesh: Any,
    parallel_pool: Any = None,
) -> Optional[List[PointAdjointResult]]:
    """Solve all points independently using cached ICs (no sequential sweep).

    When *parallel_pool* is provided (a ``BVPointSolvePool`` instance), tasks
    are dispatched to worker processes for wall-clock parallelism.  If the
    parallel path fails or is unavailable, falls back to sequential.

    Returns None if any point fails SNES (caller should fall back to
    full sequential sweep).
    """
    # ---- Parallel fast path ----
    # Use explicit parameter if given, else fall back to module-level pool
    effective_pool = parallel_pool if parallel_pool is not None else _parallel_pool
    if effective_pool is not None and getattr(effective_pool, 'enabled', False):
        # Check config compatibility: the pool's worker config has a fixed
        # observable_mode/scale/control_mode/n_controls. If the current call
        # uses different settings (e.g. secondary observable in multi-obs
        # inference), skip parallel and use sequential fast path.
        pool_cfg = getattr(effective_pool, '_config', None)
        config_compatible = True
        if pool_cfg is not None:
            if str(pool_cfg.observable_mode) != str(observable_mode):
                config_compatible = False
            if abs(float(pool_cfg.observable_scale) - float(observable_scale)) > 1e-12:
                config_compatible = False
            if str(pool_cfg.control_mode) != str(control_mode):
                config_compatible = False
            if int(pool_cfg.n_controls) != int(n_controls):
                config_compatible = False

        if config_compatible:
            parallel_result = _solve_cached_fast_path_parallel(
                n_points=n_points,
                n_controls=n_controls,
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                k0_list=k0_list,
                alpha_list=alpha_list,
                a_list=a_list,
                control_mode=control_mode,
                fail_penalty=fail_penalty,
                observable_mode=observable_mode,
                observable_reaction_index=observable_reaction_index,
                observable_scale=observable_scale,
                parallel_pool=effective_pool,
            )
            if parallel_result is not None:
                return parallel_result
            print("[bv-parallel] Parallel fast path failed; falling back to sequential")
        else:
            print(
                f"[bv-parallel] Config mismatch (pool: {pool_cfg.observable_mode}, "
                f"requested: {observable_mode}); using sequential fast path"
            )

    # ---- Sequential fast path (original) ----
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )
    from Forward.steady_state import configure_bv_solver_params

    results: List[Optional[PointAdjointResult]] = [None] * n_points

    for orig_idx in range(n_points):
        phi_applied_i = float(phi_applied_values[orig_idx])
        target_i = float(target_flux[orig_idx])

        if np.isnan(target_i):
            results[orig_idx] = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i,
                simulated_flux=float("nan"),
                objective=0.0,
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=0,
                reason="target NaN (skipped)",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            continue

        cached_U_data = _all_points_cache.get(orig_idx)
        if cached_U_data is None:
            return None  # missing cache entry, fall back

        params = configure_bv_solver_params(
            base_solver_params,
            phi_applied=phi_applied_i,
            k0_values=k0_list,
            alpha_values=alpha_list,
            a_values=a_list,
        )

        t_start = _time.perf_counter()

        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()

        ctx = bv_build_context(params, mesh=shared_mesh)
        ctx = bv_build_forms(ctx, params)
        bv_set_initial_conditions(ctx, params, blob=blob_initial_condition)

        # Warm-start from cached IC
        for src, dst in zip(cached_U_data, ctx["U"].dat):
            dst.data[:] = src
        ctx["U_prev"].assign(ctx["U"])

        # Assign control values
        k0_funcs = list(ctx["bv_k0_funcs"])
        for j, k0_f in enumerate(k0_funcs):
            if j < len(k0_list):
                k0_f.assign(float(k0_list[j]))
        alpha_funcs = list(ctx.get("bv_alpha_funcs", []))
        if alpha_list is not None:
            for j, alpha_f in enumerate(alpha_funcs):
                if j < len(alpha_list):
                    alpha_f.assign(float(alpha_list[j]))
        steric_a_funcs = list(ctx.get("steric_a_funcs", []))
        if a_list is not None:
            for j, a_f in enumerate(steric_a_funcs):
                if j < len(a_list):
                    a_f.assign(float(a_list[j]))

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
        _fwd_opts = params.solver_options if hasattr(params, 'solver_options') else params[10]
        solve_params = dict(_fwd_opts) if isinstance(_fwd_opts, dict) else {}
        solve_params.setdefault("snes_lag_jacobian", 2)
        solve_params.setdefault("snes_lag_jacobian_persists", True)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_params)

        observable_form = _build_bv_observable_form(
            ctx, mode=observable_mode,
            reaction_index=observable_reaction_index,
            scale=float(observable_scale),
        )

        # SER adaptive dt
        dt_const = ctx.get("dt_const")
        dt_initial = float(dt_const) if dt_const is not None else 1.0
        dt_current = dt_initial
        dt_max = dt_initial * _SER_DT_MAX_RATIO

        prev_flux_val = None
        steady_count = 0
        rel_metric = None
        abs_metric = None
        simulated_flux = float("nan")
        steps_taken = 0
        failed = False
        prev_delta = None

        effective_max_steps = min(max_steps, _WARMSTART_MAX_STEPS)

        for step in range(1, effective_max_steps + 1):
            steps_taken = step
            try:
                solver.solve()
            except fd.ConvergenceError:
                failed = True
                break
            U_prev.assign(U)

            with adj.stop_annotating():
                simulated_flux = float(fd.assemble(observable_form))

            if prev_flux_val is not None:
                delta = abs(simulated_flux - prev_flux_val)
                scale_val = max(abs(simulated_flux), abs(prev_flux_val), abs_tol)
                rel_metric = delta / scale_val
                abs_metric = delta
                is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                steady_count = steady_count + 1 if is_steady else 0

                if dt_const is not None and prev_delta is not None and delta > 0:
                    ratio = prev_delta / delta
                    if ratio > 1.0:
                        grow = min(ratio, _SER_GROWTH_CAP)
                        dt_current = min(dt_current * grow, dt_max)
                    else:
                        dt_current = max(dt_current * _SER_SHRINK, dt_initial)
                    dt_const.assign(dt_current)
                prev_delta = delta
            else:
                steady_count = 0
            prev_flux_val = simulated_flux

            if steady_count >= required_steady:
                break

        if dt_const is not None:
            dt_const.assign(dt_initial)

        if failed or steady_count < required_steady:
            # This point failed with cached IC; fall back to full sweep
            return None

        # Compute adjoint gradient.
        # Build target via R-space Function so fd.assemble returns an
        # adjoint-tracked AdjFloat (ReducedFunctional requires OverloadedType).
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
            point_gradient = _bv_gradient_controls_to_array(
                rf.derivative(), n_controls
            )
        except Exception as exc:
            import warnings
            warnings.warn(f"Adjoint gradient computation failed: {exc}")
            return None

        # Physics validation before cache update
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
        )
        for _w in _sol_vr.warnings:
            _warnings.warn(
                f"[cached-fast-path] phi={phi_applied_i:.4f}: {_w}",
                stacklevel=1,
            )
        if _sol_vr.valid:
            # Update cache with new solution (only if physics checks pass)
            carry_U_data = tuple(d.data_ro.copy() for d in U.dat)
            _all_points_cache[orig_idx] = carry_U_data
        else:
            # Bad solution -- skip cache update to avoid propagating bad ICs
            _warnings.warn(
                f"[cached-fast-path] phi={phi_applied_i:.4f} solution state "
                f"validation failed, skipping cache update: {_sol_vr.failures}",
                stacklevel=1,
            )

        t_elapsed = _time.perf_counter() - t_start
        print(
            f"  [timing] phi={phi_applied_i:+8.4f} "
            f"total={t_elapsed:.2f}s "
            f"steps={steps_taken}/{effective_max_steps} "
            f"(cached-IC)"
        )

        results[orig_idx] = PointAdjointResult(
            phi_applied=phi_applied_i,
            target_flux=target_i,
            simulated_flux=float(sim_scalar),
            objective=float(point_objective),
            gradient=point_gradient,
            converged=True,
            steps_taken=steps_taken,
            reason="",
            final_relative_change=rel_metric,
            final_absolute_change=abs_metric,
            diagnostics_valid=True,
        )

    return [r for r in results]  # type: ignore[misc]


def solve_point_objective_and_gradient():
    """Placeholder -- not yet extracted from the main orchestrator."""
    raise NotImplementedError(
        "solve_point_objective_and_gradient is computed inline within "
        "solve_bv_curve_points_with_warmstart"
    )
