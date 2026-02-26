"""Replay mode: persistent reduced-functional point models for fast re-evaluation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from FluxCurve.config import ForwardRecoveryConfig, RobinFluxCurveInferenceRequest
from FluxCurve.results import (
    CurveAdjointResult,
    PointAdjointResult,
    _ReplayBundle,
    _ReplayPointFunctional,
)
from FluxCurve.recovery import _attempt_phase_state, _relax_solver_options_for_attempt
from FluxCurve.observables import _build_observable_form, _gradient_controls_to_array
from FluxCurve.curve_eval import evaluate_curve_objective_and_gradient
from FluxCurve.point_solve import _PointSolveExecutor
from Forward.steady_state import SteadyStateConfig, configure_robin_solver_params
from Forward.robin_solver import build_context, build_forms, set_initial_conditions


def _build_replay_point_flux_functional(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied: float,
    kappa_values: Sequence[float],
    blob_initial_condition: bool,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_species_index: Optional[int],
    observable_scale: float,
    replay_extra_steady_steps: int,
) -> Tuple[Optional[_ReplayPointFunctional], str]:
    """Build one persistent replay-ready reduced functional for a sweep point.

    Replay point tapes are built dynamically by:
    1) solving until the same steady-state criterion is satisfied, and
    2) taking a few additional steady steps as a buffer for nearby replayed
       kappa values.

    Diagnostics functionals (previous-step observable + state deltas) are also
    taped so each replay evaluation can validate whether replay still appears to
    be in steady state.
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    kappa_list = [float(v) for v in kappa_values]
    baseline_params = configure_robin_solver_params(
        base_solver_params,
        phi_applied=float(phi_applied),
        kappa_values=kappa_list,
    )
    abs_tol = float(max(steady.absolute_tolerance, 1e-16))
    rel_tol = float(steady.relative_tolerance)
    max_steps = int(max(1, steady.max_steps))
    required_steady = int(max(1, steady.consecutive_steps))
    extra_steady = int(max(0, replay_extra_steady_steps))
    target_steady = int(required_steady + extra_steady)

    baseline_options: Mapping[str, Any] = {}
    if isinstance(baseline_params[10], dict):
        baseline_options = copy.deepcopy(baseline_params[10])

    last_reason = (
        "dynamic replay build failed before reaching steady state "
        f"(need {target_steady} steady steps)"
    )
    max_attempts = max(1, int(forward_recovery.max_attempts))
    for attempt in range(max_attempts):
        params = configure_robin_solver_params(
            base_solver_params,
            phi_applied=float(phi_applied),
            kappa_values=kappa_list,
        )
        if isinstance(params[10], dict):
            phase, phase_step, _cycle_index = _attempt_phase_state(attempt, forward_recovery)
            _relax_solver_options_for_attempt(
                params[10],
                phase=phase,
                phase_step=phase_step,
                recovery=forward_recovery,
                baseline_options=baseline_options if baseline_options else params[10],
            )

        tape = adj.Tape()
        failed_by_exception = False
        try:
            with adj.set_working_tape(tape):
                adj.continue_annotation()

                ctx = build_context(params)
                ctx = build_forms(ctx, params)
                set_initial_conditions(ctx, params, blob=blob_initial_condition)

                U = ctx["U"]
                U_prev = ctx["U_prev"]
                F_res = ctx["F_res"]
                bcs = ctx["bcs"]

                jac = fd.derivative(F_res, U)
                problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
                solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params[10])

                observable_form = _build_observable_form(
                    ctx,
                    mode=observable_mode,
                    species_index=observable_species_index,
                    scale=float(observable_scale),
                    state=U,
                )
                observable_prev_form = _build_observable_form(
                    ctx,
                    mode=observable_mode,
                    species_index=observable_species_index,
                    scale=float(observable_scale),
                    state=U_prev,
                )
                state_delta_sq_form = fd.inner(U - U_prev, U - U_prev) * fd.dx(domain=ctx["mesh"])
                state_norm_sq_form = fd.inner(U, U) * fd.dx(domain=ctx["mesh"])

                prev_flux: Optional[float] = None
                steady_count = 0

                for step in range(1, max_steps + 1):
                    try:
                        solver.solve()
                    except Exception as exc:
                        failed_by_exception = True
                        last_reason = f"{type(exc).__name__}: {exc}"
                        break

                    flux_now = float(fd.assemble(observable_form))
                    if prev_flux is not None:
                        delta = abs(flux_now - prev_flux)
                        scale = max(abs(flux_now), abs(prev_flux), abs_tol)
                        rel_metric = delta / scale
                        abs_metric = delta
                        is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                        steady_count = steady_count + 1 if is_steady else 0
                    else:
                        steady_count = 0

                    if steady_count >= target_steady:
                        observable_scalar = fd.assemble(observable_form)
                        observable_prev_scalar = fd.assemble(observable_prev_form)
                        state_delta_sq_scalar = fd.assemble(state_delta_sq_form)
                        state_norm_sq_scalar = fd.assemble(state_norm_sq_form)
                        controls = [adj.Control(ctrl) for ctrl in list(ctx["kappa_funcs"])]
                        reduced_flux = adj.ReducedFunctional(observable_scalar, controls)
                        reduced_flux_prev = adj.ReducedFunctional(
                            observable_prev_scalar, controls
                        )
                        reduced_state_delta_l2 = adj.ReducedFunctional(
                            state_delta_sq_scalar, controls
                        )
                        reduced_state_norm_l2 = adj.ReducedFunctional(
                            state_norm_sq_scalar, controls
                        )
                        control_state = [ctrl for ctrl in list(ctx["kappa_funcs"])]
                        return (
                            _ReplayPointFunctional(
                                phi_applied=float(phi_applied),
                                tape=tape,
                                control_state=control_state,
                                reduced_flux=reduced_flux,
                                reduced_flux_prev=reduced_flux_prev,
                                reduced_state_delta_l2=reduced_state_delta_l2,
                                reduced_state_norm_l2=reduced_state_norm_l2,
                                steady_rel_tol=rel_tol,
                                steady_abs_tol=abs_tol,
                                steps_taken=int(step),
                            ),
                            "",
                        )

                    prev_flux = flux_now
                    # Advance to next timestep with the newly solved state.
                    U_prev.assign(U)

                if not failed_by_exception:
                    observable_scalar = fd.assemble(observable_form)
                    last_reason = (
                        "steady-state criterion not satisfied before max_steps "
                        f"(max_steps={max_steps}, final_flux={float(observable_scalar):.6g})"
                    )
        except Exception as exc:
            last_reason = f"{type(exc).__name__}: {exc}"

    return None, str(last_reason)


def _build_replay_bundle(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    kappa_values: np.ndarray,
) -> Tuple[Optional[_ReplayBundle], str]:
    """Build persistent replay point models for all phi_applied values."""
    points: List[_ReplayPointFunctional] = []
    for point_idx, phi_i in enumerate(phi_applied_values.tolist()):
        point_model, reason = _build_replay_point_flux_functional(
            base_solver_params=request.base_solver_params,
            steady=request.steady,
            phi_applied=float(phi_i),
            kappa_values=np.asarray(kappa_values, dtype=float).tolist(),
            blob_initial_condition=bool(request.blob_initial_condition),
            forward_recovery=request.forward_recovery,
            observable_mode=str(request.observable_mode),
            observable_species_index=request.observable_species_index,
            observable_scale=float(request.observable_scale),
            replay_extra_steady_steps=int(request.replay_extra_steady_steps),
        )
        if point_model is None:
            return (
                None,
                f"phi_applied={float(phi_i):.6f} (point {point_idx}) build failed: {reason}",
            )
        points.append(point_model)

    return (
        _ReplayBundle(
            points=points,
            anchor_kappa=np.asarray(kappa_values, dtype=float).copy(),
        ),
        "",
    )


def _evaluate_curve_with_replay_bundle(
    *,
    bundle: _ReplayBundle,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
) -> CurveAdjointResult:
    """Evaluate full curve objective/gradient via persistent replay models.

    Each point also runs replay diagnostics:
    - absolute/relative change between current and previous-step observable
    - state change norm between current and previous timestep states

    Points failing diagnostics are marked non-converged so the caller can
    fallback to fresh forward solves and potentially rebuild replay.
    """
    import firedrake.adjoint as adj

    n_species = int(np.asarray(kappa_values, dtype=float).size)
    simulated_flux = np.full(target_flux.shape, np.nan, dtype=float)
    total_gradient = np.zeros(n_species, dtype=float)
    total_objective = 0.0
    point_rows: List[PointAdjointResult] = []
    n_failed = 0

    if int(len(bundle.points)) != int(target_flux.size):
        raise RuntimeError(
            "Replay bundle size does not match target curve size: "
            f"{len(bundle.points)} vs {target_flux.size}."
        )

    for i, point in enumerate(bundle.points):
        try:
            with adj.set_working_tape(point.tape):
                adj.continue_annotation()
                for j in range(min(n_species, len(point.control_state))):
                    point.control_state[j].assign(float(kappa_values[j]))
                flux_val = float(point.reduced_flux(point.control_state))
                flux_prev_val = float(point.reduced_flux_prev(point.control_state))
                state_delta_sq_val = float(
                    point.reduced_state_delta_l2(point.control_state)
                )
                state_norm_sq_val = float(point.reduced_state_norm_l2(point.control_state))
                dflux_dk = _gradient_controls_to_array(point.reduced_flux.derivative(), n_species)
        except Exception as exc:
            raise RuntimeError(
                f"Replay point evaluation failed at phi_applied={point.phi_applied:.6f}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        diagnostics_valid = bool(
            np.isfinite(flux_val)
            and np.isfinite(flux_prev_val)
            and np.isfinite(state_delta_sq_val)
            and np.isfinite(state_norm_sq_val)
            and np.all(np.isfinite(dflux_dk))
            and (state_delta_sq_val >= -1e-14)
            and (state_norm_sq_val >= -1e-14)
        )

        target_i = float(target_flux[i])
        residual_i = flux_val - target_i if np.isfinite(flux_val) else float("nan")
        point_objective = (
            0.5 * (residual_i**2) if np.isfinite(residual_i) else float("inf")
        )

        flux_abs_change = abs(flux_val - flux_prev_val) if diagnostics_valid else float("inf")
        flux_rel_change = (
            flux_abs_change
            / max(abs(flux_val), abs(flux_prev_val), float(point.steady_abs_tol), 1e-16)
            if diagnostics_valid
            else float("inf")
        )
        state_delta_l2 = (
            float(np.sqrt(max(0.0, state_delta_sq_val))) if diagnostics_valid else float("inf")
        )
        state_norm_l2 = (
            float(np.sqrt(max(0.0, state_norm_sq_val))) if diagnostics_valid else float("inf")
        )
        state_rel_change = (
            state_delta_l2 / max(state_norm_l2, 1e-16) if diagnostics_valid else float("inf")
        )

        steady_ok = bool(
            diagnostics_valid
            and (
                flux_rel_change <= float(point.steady_rel_tol)
                or flux_abs_change <= float(point.steady_abs_tol)
            )
        )
        point_converged = bool(diagnostics_valid and steady_ok)

        point_gradient = residual_i * dflux_dk if point_converged else np.zeros(n_species, dtype=float)

        simulated_flux[i] = flux_val
        if point_converged and np.isfinite(point_objective):
            total_objective += float(point_objective)
        else:
            # Force fallback path to run fresh forward solves for this candidate.
            total_objective += 1e12
        total_gradient += point_gradient
        if not point_converged:
            n_failed = n_failed + 1

        reason = "replay"
        if not diagnostics_valid:
            reason = "replay_diag_invalid"
        elif not steady_ok:
            reason = (
                "replay_not_steady"
                f"(rel={flux_rel_change:.3e},abs={flux_abs_change:.3e},"
                f"state_rel={state_rel_change:.3e})"
            )
        point_rows.append(
            PointAdjointResult(
                phi_applied=float(point.phi_applied),
                target_flux=target_i,
                simulated_flux=float(flux_val),
                objective=float(point_objective),
                gradient=np.asarray(point_gradient, dtype=float),
                converged=point_converged,
                steps_taken=int(point.steps_taken),
                reason=reason,
                final_relative_change=float(flux_rel_change)
                if np.isfinite(flux_rel_change)
                else None,
                final_absolute_change=float(flux_abs_change)
                if np.isfinite(flux_abs_change)
                else None,
                diagnostics_valid=diagnostics_valid,
            )
        )

    return CurveAdjointResult(
        objective=float(total_objective),
        gradient=np.asarray(total_gradient, dtype=float),
        simulated_flux=simulated_flux,
        points=point_rows,
        n_failed=int(n_failed),
        effective_kappa=np.asarray(kappa_values, dtype=float).copy(),
        used_anisotropy_recovery=False,
        used_replay_mode=True,
    )


class _DynamicReplayCurveEvaluator:
    """Hybrid curve evaluator with automatic replay enable/disable/re-enable."""

    def __init__(
        self,
        *,
        request: RobinFluxCurveInferenceRequest,
        phi_applied_values: np.ndarray,
        target_flux: np.ndarray,
        point_executor: Optional[_PointSolveExecutor] = None,
    ) -> None:
        self.request = request
        self.phi_applied_values = np.asarray(phi_applied_values, dtype=float)
        self.target_flux = np.asarray(target_flux, dtype=float)
        self.point_executor = point_executor
        self.replay_requested = bool(request.replay_mode_enabled)
        self.replay_bundle: Optional[_ReplayBundle] = None
        self.replay_enabled = False
        self.fallback_success_streak = 0
        self.reenable_after_successes = max(1, int(request.replay_reenable_after_successes))
        self.replay_rebuild_count = 0
        self.replay_diag_rebuild_count = 0
        self.replay_exception_rebuild_count = 0

    def initialize(self, *, kappa_anchor: np.ndarray) -> None:
        """Build replay bundle once at startup when replay mode is enabled."""
        if not self.replay_requested:
            return
        print(
            "[replay] building persistent point models at "
            f"kappa=[{float(kappa_anchor[0]):.6f}, {float(kappa_anchor[1]):.6f}] "
            "using dynamic steady-state replay "
            f"(extra_steady_steps={int(self.request.replay_extra_steady_steps)})"
        )
        self._enable_replay(np.asarray(kappa_anchor, dtype=float))

    def evaluate(self, *, kappa_values: np.ndarray) -> CurveAdjointResult:
        """Evaluate objective/gradient, preferring replay when available."""
        kappa_eval = np.asarray(kappa_values, dtype=float)

        if self.replay_enabled and self.replay_bundle is not None:
            try:
                replay_curve = _evaluate_curve_with_replay_bundle(
                    bundle=self.replay_bundle,
                    target_flux=self.target_flux,
                    kappa_values=kappa_eval,
                )
                if int(replay_curve.n_failed) == 0:
                    return replay_curve
                self.replay_diag_rebuild_count += 1
                self._disable_replay(
                    "replay diagnostics failed "
                    f"({int(replay_curve.n_failed)}/{len(replay_curve.points)} points); "
                    "refreshing via full steady-state solves."
                )
            except Exception as exc:
                self.replay_exception_rebuild_count += 1
                self._disable_replay(
                    "replay evaluation failure; "
                    f"falling back to resilient solves ({type(exc).__name__}: {exc})"
                )

        fallback = evaluate_curve_objective_and_gradient(
            request=self.request,
            phi_applied_values=self.phi_applied_values,
            target_flux=self.target_flux,
            kappa_values=kappa_eval,
            point_executor=self.point_executor,
        )
        fallback.used_replay_mode = False

        if not self.replay_requested:
            return fallback

        if int(fallback.n_failed) == 0:
            self.fallback_success_streak += 1
            if self.fallback_success_streak >= self.reenable_after_successes:
                print(
                    "[replay] fallback reconverged; "
                    "attempting replay re-enable at current kappa."
                )
                enabled = self._enable_replay(np.asarray(fallback.effective_kappa, dtype=float))
                if enabled:
                    self.fallback_success_streak = 0
                else:
                    # Keep retry cadence optimistic without rebuilding every call.
                    self.fallback_success_streak = self.reenable_after_successes - 1
        else:
            self.fallback_success_streak = 0

        return fallback

    def _enable_replay(self, kappa_anchor: np.ndarray) -> bool:
        if not self.replay_requested:
            return False
        bundle, reason = _build_replay_bundle(
            request=self.request,
            phi_applied_values=self.phi_applied_values,
            kappa_values=np.asarray(kappa_anchor, dtype=float),
        )
        if bundle is None:
            self.replay_bundle = None
            self.replay_enabled = False
            print(f"[replay] build failed: {reason}")
            return False

        self.replay_bundle = bundle
        self.replay_enabled = True
        self.replay_rebuild_count += 1
        print(
            "[replay] enabled with "
            f"{len(bundle.points)} persistent point models."
        )
        return True

    def _disable_replay(self, reason: str) -> None:
        if self.replay_enabled or self.replay_bundle is not None:
            print(f"[replay] disabled: {reason}")
        self.replay_enabled = False
        self.replay_bundle = None
        self.fallback_success_streak = 0

    def stats(self) -> Dict[str, int]:
        """Return replay lifecycle counters for runtime diagnostics."""
        return {
            "replay_rebuild_count": int(self.replay_rebuild_count),
            "replay_diag_rebuild_count": int(self.replay_diag_rebuild_count),
            "replay_exception_rebuild_count": int(self.replay_exception_rebuild_count),
        }
