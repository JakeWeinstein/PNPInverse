"""Curve-level BV objective and gradient evaluation across the phi_applied sweep."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np

from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from FluxCurve.results import CurveAdjointResult, PointAdjointResult
from FluxCurve.recovery import _reduce_kappa_anisotropy
from FluxCurve.bv_point_solve import solve_bv_curve_points_with_warmstart


def evaluate_bv_curve_objective_and_gradient(
    *,
    request: BVFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    k0_values: np.ndarray,
    mesh=None,
    alpha_values: Optional[np.ndarray] = None,
    a_values: Optional[np.ndarray] = None,
    control_mode: str = "k0",
) -> CurveAdjointResult:
    """Evaluate BV curve objective + gradient, with anisotropy recovery fallback.

    Calls :func:`solve_bv_curve_points_with_warmstart` to get all point
    results (already solved sequentially with warm-start), then aggregates.
    """
    if len(phi_applied_values) == 0:
        raise ValueError("phi_applied_values must be non-empty")
    if control_mode == "k0":
        n_controls = int(k0_values.size)
    elif control_mode == "alpha":
        n_controls = int(alpha_values.size) if alpha_values is not None else 0
    elif control_mode == "joint":
        n_controls = int(k0_values.size) + (int(alpha_values.size) if alpha_values is not None else 0)
    elif control_mode == "steric":
        n_species = int(request.base_solver_params.n_species) if hasattr(request.base_solver_params, 'n_species') else int(request.base_solver_params[0])
        n_controls = n_species
    elif control_mode == "full":
        n_species = int(request.base_solver_params.n_species) if hasattr(request.base_solver_params, 'n_species') else int(request.base_solver_params[0])
        n_controls = int(k0_values.size) + (int(alpha_values.size) if alpha_values is not None else 0) + n_species
    else:
        n_controls = int(k0_values.size)

    def _evaluate_once(k0_eval: np.ndarray, alpha_eval: Optional[np.ndarray] = None, a_eval: Optional[np.ndarray] = None) -> CurveAdjointResult:
        points = solve_bv_curve_points_with_warmstart(
            base_solver_params=request.base_solver_params,
            steady=request.steady,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            k0_values=k0_eval.tolist(),
            blob_initial_condition=bool(request.blob_initial_condition),
            fail_penalty=float(request.fail_penalty),
            forward_recovery=request.forward_recovery,
            observable_mode=str(request.observable_mode),
            observable_reaction_index=request.observable_reaction_index,
            observable_scale=float(request.current_density_scale),
            mesh=mesh,
            alpha_values=alpha_eval.tolist() if alpha_eval is not None else None,
            a_values=a_eval.tolist() if a_eval is not None else None,
            control_mode=control_mode,
            max_eta_gap=float(getattr(request, 'max_eta_gap', 0.0)),
        )

        simulated_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)
        total_objective = 0.0
        total_gradient = np.zeros(n_controls, dtype=float)
        n_failed = 0

        for i, point in enumerate(points):
            simulated_flux[i] = point.simulated_flux
            total_objective += float(point.objective)
            if point.converged:
                total_gradient += point.gradient
            else:
                n_failed += 1
                total_gradient += point.gradient  # Include fail-penalty gradient

        return CurveAdjointResult(
            objective=float(total_objective),
            gradient=total_gradient,
            simulated_flux=simulated_flux,
            points=points,
            n_failed=n_failed,
            effective_kappa=np.asarray(k0_eval, dtype=float).copy(),
            used_anisotropy_recovery=False,
        )

    alpha_arr = np.asarray(alpha_values, dtype=float) if alpha_values is not None else None
    a_arr = np.asarray(a_values, dtype=float) if a_values is not None else None
    primary = _evaluate_once(np.asarray(k0_values, dtype=float), alpha_arr, a_arr)

    n_points = int(len(phi_applied_values))
    fail_threshold_by_points = max(1, int(request.anisotropy_trigger_failed_points))
    fail_threshold_by_fraction = int(
        np.ceil(
            max(0.0, float(request.anisotropy_trigger_failed_fraction)) * max(1, n_points)
        )
    )
    fail_trigger = max(fail_threshold_by_points, fail_threshold_by_fraction)

    if primary.n_failed < fail_trigger:
        return primary

    # Anisotropy recovery only applies to k0 control mode
    if control_mode != "k0":
        return primary

    k0_aniso = _reduce_kappa_anisotropy(
        np.asarray(k0_values, dtype=float),
        target_ratio=float(request.forward_recovery.anisotropy_target_ratio),
        blend=float(request.forward_recovery.anisotropy_blend),
    )
    if np.allclose(k0_aniso, np.asarray(k0_values, dtype=float), rtol=1e-12, atol=1e-12):
        return primary

    print(
        "[recovery] many point failures detected "
        f"({primary.n_failed}/{n_points}); retrying with anisotropy-reduced k0 "
        f"{k0_aniso.tolist()}"
    )
    secondary = _evaluate_once(k0_aniso, alpha_arr, a_arr)
    secondary.used_anisotropy_recovery = True

    if secondary.n_failed < primary.n_failed:
        return secondary
    if secondary.n_failed == primary.n_failed and secondary.objective < primary.objective:
        return secondary
    return primary


def build_residual_jacobian(
    points: List[PointAdjointResult],
    n_controls: int,
) -> tuple:
    """Build residual vector and Jacobian from per-point adjoint results.

    For each converged point i with objective J_i = 0.5*(sim_i - target_i)^2
    and gradient dJ_i/d(ctrl), we recover the Jacobian row:

        r_i = sim_i - target_i
        J_r[i,:] = dJ_i/d(ctrl) / r_i   (since dJ_i/d(ctrl) = r_i * d(sim_i)/d(ctrl))

    For failed/NaN points, r_i = 0 and J_r[i,:] = 0 (zero contribution).

    Returns (residuals, jacobian) as numpy arrays.
    """
    n_points = len(points)
    residuals = np.zeros(n_points, dtype=float)
    jacobian = np.zeros((n_points, n_controls), dtype=float)

    for i, point in enumerate(points):
        if not point.converged or np.isnan(point.simulated_flux):
            continue
        r_i = point.simulated_flux - point.target_flux
        residuals[i] = r_i
        if abs(r_i) > 1e-10:
            jacobian[i, :] = point.gradient / r_i
        # else: leave as zeros (tiny residual → negligible contribution)

    return residuals, jacobian


def evaluate_bv_multi_observable_objective_and_gradient(
    *,
    request: BVFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux_primary: np.ndarray,
    target_flux_secondary: np.ndarray,
    k0_values: np.ndarray,
    mesh=None,
    alpha_values: Optional[np.ndarray] = None,
    a_values: Optional[np.ndarray] = None,
    control_mode: str = "joint",
) -> CurveAdjointResult:
    """Evaluate combined objective from two observables (e.g. total + peroxide current).

    **v6 multi-observable parallel path (Strategy A):**
    When a multi-obs parallel pool is available and the checkpoint-restart
    cache is populated, dispatches each voltage point to a worker that
    computes BOTH primary and secondary adjoint gradients after a single
    forward solve.  This eliminates the sequential fallback for the
    secondary observable (~36% of v5 time in multi-obs phases).

    **v5 fallback path:**
    Calls ``evaluate_bv_curve_objective_and_gradient`` twice -- once with the
    primary observable_mode and once with the secondary -- then sums the
    objectives and gradients (with weighting on the secondary term).

    Returns a ``CurveAdjointResult`` with the combined objective/gradient and
    the primary observable's simulated flux for plotting.  The secondary
    result is stored in the ``_secondary_result`` attribute.
    """
    import FluxCurve.bv_point_solve as _bv_ps
    from FluxCurve.bv_point_solve import _solve_cached_fast_path_parallel_multi_obs

    secondary_mode = str(request.secondary_observable_mode or "peroxide_current")
    secondary_weight = float(request.secondary_observable_weight)
    secondary_scale = request.secondary_current_density_scale
    if secondary_scale is None:
        secondary_scale = float(request.current_density_scale)
    else:
        secondary_scale = float(secondary_scale)

    # Determine n_controls
    if control_mode == "k0":
        n_controls = int(k0_values.size)
    elif control_mode == "alpha":
        n_controls = int(alpha_values.size) if alpha_values is not None else 0
    elif control_mode == "joint":
        n_controls = int(k0_values.size) + (int(alpha_values.size) if alpha_values is not None else 0)
    elif control_mode == "steric":
        n_controls = int(request.base_solver_params.n_species) if hasattr(request.base_solver_params, 'n_species') else int(request.base_solver_params[0])
    elif control_mode == "full":
        n_species = int(request.base_solver_params.n_species) if hasattr(request.base_solver_params, 'n_species') else int(request.base_solver_params[0])
        n_controls = int(k0_values.size) + (int(alpha_values.size) if alpha_values is not None else 0) + n_species
    else:
        n_controls = int(k0_values.size)

    n_points = int(len(phi_applied_values))
    k0_list = [float(v) for v in k0_values]
    alpha_list = [float(v) for v in alpha_values] if alpha_values is not None else None
    a_list = [float(v) for v in a_values] if a_values is not None else None

    # ---- Strategy A: Multi-obs parallel single-pass ----
    # Check if we can use the multi-obs parallel path:
    # 1. Cache must be populated (at least one full sequential sweep done)
    # 2. A multi-obs pool must be active
    # 3. Pool config must match the requested observables
    # NOTE: Access module-level variables through the module object to get
    # current values (not stale import-time copies of scalars/references).
    effective_pool = _bv_ps._parallel_pool
    multi_obs_parallel_ok = False
    if (effective_pool is not None
            and getattr(effective_pool, 'enabled', False)
            and getattr(effective_pool, 'is_multi_obs', False)
            and _bv_ps._cache_populated
            and len(_bv_ps._all_points_cache) >= n_points):
        # Verify all points have cached ICs
        all_cached = all(i in _bv_ps._all_points_cache for i in range(n_points))
        if all_cached:
            pool_cfg = getattr(effective_pool, '_config', None)
            if pool_cfg is not None:
                cfg_ok = (
                    str(pool_cfg.observable_mode) == str(request.observable_mode)
                    and str(pool_cfg.secondary_observable_mode) == str(secondary_mode)
                    and str(pool_cfg.control_mode) == str(control_mode)
                    and int(pool_cfg.n_controls) == int(n_controls)
                )
                if cfg_ok:
                    multi_obs_parallel_ok = True

    if multi_obs_parallel_ok:
        print("[multi-obs] Using parallel single-pass multi-observable path")
        multi_result = _solve_cached_fast_path_parallel_multi_obs(
            n_points=n_points,
            n_controls=n_controls,
            phi_applied_values=phi_applied_values,
            target_flux_primary=target_flux_primary,
            target_flux_secondary=target_flux_secondary,
            k0_list=k0_list,
            alpha_list=alpha_list,
            a_list=a_list,
            control_mode=control_mode,
            fail_penalty=float(request.fail_penalty),
            observable_mode=str(request.observable_mode),
            observable_reaction_index=request.observable_reaction_index,
            observable_scale=float(request.current_density_scale),
            secondary_observable_mode=secondary_mode,
            secondary_observable_scale=secondary_scale,
            secondary_weight=secondary_weight,
            parallel_pool=effective_pool,
        )
        if multi_result is not None:
            primary_points = multi_result["primary_results"]
            secondary_points = multi_result["secondary_results"]

            # Aggregate primary
            sim_flux_primary = np.full(n_points, np.nan, dtype=float)
            total_obj_primary = 0.0
            total_grad_primary = np.zeros(n_controls, dtype=float)
            n_failed_primary = 0
            for i, pt in enumerate(primary_points):
                if pt is not None:
                    sim_flux_primary[i] = pt.simulated_flux
                    total_obj_primary += float(pt.objective)
                    if pt.converged:
                        total_grad_primary += pt.gradient
                    else:
                        n_failed_primary += 1
                        total_grad_primary += pt.gradient  # Include fail-penalty gradient
                else:
                    n_failed_primary += 1

            primary_result = CurveAdjointResult(
                objective=float(total_obj_primary),
                gradient=total_grad_primary,
                simulated_flux=sim_flux_primary,
                points=primary_points,
                n_failed=n_failed_primary,
                effective_kappa=np.asarray(k0_values, dtype=float).copy(),
                used_anisotropy_recovery=False,
            )

            # Aggregate secondary
            sim_flux_secondary = np.full(n_points, np.nan, dtype=float)
            total_obj_secondary = 0.0
            total_grad_secondary = np.zeros(n_controls, dtype=float)
            n_failed_secondary = 0
            for i, pt in enumerate(secondary_points):
                if pt is not None:
                    sim_flux_secondary[i] = pt.simulated_flux
                    total_obj_secondary += float(pt.objective)
                    if pt.converged:
                        total_grad_secondary += pt.gradient
                    else:
                        n_failed_secondary += 1
                        total_grad_secondary += pt.gradient  # Include fail-penalty gradient
                else:
                    n_failed_secondary += 1

            secondary_result = CurveAdjointResult(
                objective=float(total_obj_secondary),
                gradient=total_grad_secondary,
                simulated_flux=sim_flux_secondary,
                points=secondary_points,
                n_failed=n_failed_secondary,
                effective_kappa=np.asarray(k0_values, dtype=float).copy(),
                used_anisotropy_recovery=False,
            )

            combined_objective = float(primary_result.objective) + secondary_weight * float(secondary_result.objective)
            combined_gradient = np.asarray(primary_result.gradient, dtype=float) + secondary_weight * np.asarray(secondary_result.gradient, dtype=float)

            combined = CurveAdjointResult(
                objective=combined_objective,
                gradient=combined_gradient,
                simulated_flux=np.asarray(primary_result.simulated_flux, dtype=float),
                points=primary_result.points,
                n_failed=int(primary_result.n_failed) + int(secondary_result.n_failed),
                effective_kappa=np.asarray(k0_values, dtype=float).copy(),
                used_anisotropy_recovery=False,
            )
            combined._secondary_result = secondary_result  # type: ignore[attr-defined]
            return combined
        else:
            print("[multi-obs] Parallel multi-obs path failed; falling back to dual-eval")

    # ---- v5 fallback: dual evaluation (primary then secondary) ----
    # Primary evaluation (uses request.observable_mode as-is).
    # This populates _all_points_cache with converged forward states.
    primary_result = evaluate_bv_curve_objective_and_gradient(
        request=request,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux_primary,
        k0_values=k0_values,
        mesh=mesh,
        alpha_values=alpha_values,
        a_values=a_values,
        control_mode=control_mode,
    )

    # DO NOT clear caches between observables.  The forward solution cached
    # in _all_points_cache is independent of the observable form -- only the
    # adjoint differs.  Each point in the fast path builds a fresh adjoint
    # tape with the secondary observable, so gradients are correct.  Keeping
    # the cache lets the secondary evaluation skip the full sequential sweep.

    # Build a modified request for the secondary observable
    secondary_request = copy.deepcopy(request)
    secondary_request.observable_mode = secondary_mode
    secondary_request.current_density_scale = secondary_scale

    secondary_result = evaluate_bv_curve_objective_and_gradient(
        request=secondary_request,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux_secondary,
        k0_values=k0_values,
        mesh=mesh,
        alpha_values=alpha_values,
        a_values=a_values,
        control_mode=control_mode,
    )

    # NOTE (v5 parallel optimization): Do NOT clear caches here.
    # The cached forward solutions remain valid ICs for the next optimizer
    # evaluation (same mesh, similar control values).  Keeping the cache
    # lets ALL subsequent evaluations -- both primary and secondary -- use
    # the fast path (independent per-point solves from cached ICs).
    # Phase-transition cache clearing is handled by explicit _clear_caches()
    # calls in the orchestrating script.

    combined_objective = float(primary_result.objective) + secondary_weight * float(secondary_result.objective)
    combined_gradient = np.asarray(primary_result.gradient, dtype=float) + secondary_weight * np.asarray(secondary_result.gradient, dtype=float)

    combined = CurveAdjointResult(
        objective=combined_objective,
        gradient=combined_gradient,
        simulated_flux=np.asarray(primary_result.simulated_flux, dtype=float),
        points=primary_result.points,
        n_failed=int(primary_result.n_failed) + int(secondary_result.n_failed),
        effective_kappa=np.asarray(k0_values, dtype=float).copy(),
        used_anisotropy_recovery=False,
    )
    # Attach secondary result for downstream reporting
    combined._secondary_result = secondary_result  # type: ignore[attr-defined]
    return combined


def evaluate_bv_multi_ph_objective_and_gradient(
    *,
    request: BVFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    ph_conditions: List[Dict],
    k0_values: np.ndarray,
    mesh=None,
    alpha_values: Optional[np.ndarray] = None,
    a_values: Optional[np.ndarray] = None,
    control_mode: str = "joint",
) -> Dict[str, object]:
    """Evaluate combined objective across multiple pH conditions.

    Each pH condition provides a different c_H+ bulk concentration, producing
    a different I-V curve. The objective sums across all conditions (weighted).
    The k0 and alpha controls are shared across conditions.

    Parameters
    ----------
    ph_conditions : list of dict
        Each dict must have: ``target_flux`` (np.ndarray), ``c_hp_hat`` (float),
        ``c_hp_species_index`` (int), ``weight`` (float, default 1.0).
    """
    from FluxCurve.bv_point_solve import _clear_caches

    n_conditions = len(ph_conditions)
    combined_objective = 0.0
    combined_gradient = None
    condition_results = []

    for ci, cond in enumerate(ph_conditions):
        weight = float(cond.get("weight", 1.0))
        target_flux_cond = np.asarray(cond["target_flux"], dtype=float)
        c_hp_hat = float(cond["c_hp_hat"])
        c_hp_idx = int(cond.get("c_hp_species_index", 2))  # H+ is species 2 by default

        # Create modified request with this condition's c_H+ bulk
        cond_request = copy.deepcopy(request)
        # Modify bulk concentration for H+ in solver params
        _bsp = cond_request.base_solver_params
        bulk_concs = list(_bsp.c0_vals if hasattr(_bsp, 'c0_vals') else _bsp[8])
        bulk_concs[c_hp_idx] = c_hp_hat
        # Also update counterion for electroneutrality (e.g. ClO4- matches H+)
        counterion_idx = cond.get("counterion_species_index", None)
        if counterion_idx is not None:
            bulk_concs[int(counterion_idx)] = c_hp_hat
        _bsp = cond_request.base_solver_params
        if hasattr(_bsp, 'with_c0_vals'):
            cond_request.base_solver_params = _bsp.with_c0_vals(bulk_concs)
            _bsp = cond_request.base_solver_params
        else:
            _bsp[8] = bulk_concs
        # Also update the bv_bc cathodic_conc_factors c_ref_nondim for H+
        _opts = _bsp.solver_options if hasattr(_bsp, 'solver_options') else _bsp[10]
        bv_cfg = _opts.get("bv_bc", {})
        reactions = bv_cfg.get("reactions", [])
        for rxn in reactions:
            for ccf in rxn.get("cathodic_conc_factors", []):
                if int(ccf.get("species", -1)) == c_hp_idx:
                    ccf["c_ref_nondim"] = c_hp_hat

        _clear_caches()

        cond_result = evaluate_bv_curve_objective_and_gradient(
            request=cond_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux_cond,
            k0_values=k0_values,
            mesh=mesh,
            alpha_values=alpha_values,
            a_values=a_values,
            control_mode=control_mode,
        )

        combined_objective += weight * float(cond_result.objective)
        grad_cond = weight * np.asarray(cond_result.gradient, dtype=float)
        if combined_gradient is None:
            combined_gradient = grad_cond
        else:
            combined_gradient += grad_cond

        condition_results.append({
            "condition_index": ci,
            "c_hp_hat": c_hp_hat,
            "weight": weight,
            "objective": float(cond_result.objective),
            "n_failed": int(cond_result.n_failed),
            "simulated_flux": np.asarray(cond_result.simulated_flux, dtype=float),
            "result": cond_result,
        })

    _clear_caches()

    if combined_gradient is None:
        combined_gradient = np.zeros(1)

    return {
        "objective": combined_objective,
        "gradient": combined_gradient,
        "condition_results": condition_results,
    }
