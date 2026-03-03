"""Resilient minimization with recovery retries and guess manipulation."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import InferenceRequest, RecoveryAttempt, RecoveryConfig
from .objective import build_reduced_functional, _AttemptMonitor
from .formatting import (
    _format_float_for_log,
    _format_int_for_log,
    _format_guess_for_log,
    _summarize_exception,
    _format_recovery_summary,
)


def resilient_minimize(
    *,
    request: InferenceRequest,
    base_solver_params: Sequence[Any],
    concentration_targets: Sequence[Sequence[float]],
    phi_target: Sequence[float],
) -> Tuple[Any, List[Any], Any, List[RecoveryAttempt]]:
    """Run always-on resilient optimization with recovery retries.

    Returns
    -------
    tuple
        ``(optimized_controls, inverse_solver_params, rf, attempt_logs)``
    """
    import firedrake.adjoint as adj

    n_species = int(base_solver_params.n_species) if hasattr(base_solver_params, 'n_species') else int(base_solver_params[0])
    bounds = request.bounds
    if bounds is None:
        bounds = request.target.default_bounds(n_species)

    max_attempts_requested = (
        request.recovery.max_attempts
        if request.recovery_attempts is None
        else request.recovery_attempts
    )
    max_attempts = max(1, int(max_attempts_requested))
    current_guess = copy.deepcopy(request.initial_guess)
    last_safe_guess: Optional[Any] = None
    attempt_logs: List[RecoveryAttempt] = []
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        phase, phase_step, cycle_index = _attempt_phase_state(attempt, request.recovery)
        inverse_solver_params = request.target.apply_value(base_solver_params, current_guess)
        _isp_opts = inverse_solver_params.solver_options if hasattr(inverse_solver_params, 'solver_options') else inverse_solver_params[10]
        baseline_solver_options = (
            copy.deepcopy(_isp_opts)
            if isinstance(_isp_opts, dict)
            else None
        )
        _relax_solver_options_for_attempt(
            inverse_solver_params,
            attempt=attempt,
            phase=phase,
            phase_step=phase_step,
            cycle_index=cycle_index,
            recovery=request.recovery,
            baseline_options=baseline_solver_options,
        )

        monitor = _AttemptMonitor(target=request.target)
        rf = build_reduced_functional(
            adapter=request.adapter,
            target=request.target,
            solver_params=inverse_solver_params,
            concentration_targets=concentration_targets,
            phi_target=phi_target,
            blob_initial_condition=request.blob_initial_condition,
            print_interval=int(request.print_interval_inverse),
            extra_eval_cb_post=monitor.eval_cb_post,
        )

        minimize_kwargs = _build_minimize_kwargs(request, bounds=bounds)

        if request.recovery.verbose:
            _isp_opts_v = inverse_solver_params.solver_options if hasattr(inverse_solver_params, 'solver_options') else inverse_solver_params[10]
            opts = _isp_opts_v if isinstance(_isp_opts_v, dict) else {}
            print(
                "[resilient] "
                f"attempt={attempt + 1:03d}/{max_attempts:03d} "
                f"phase={phase:<15} "
                f"\nstep={phase_step:02d} "
                f"cycle={cycle_index:02d} "
                f"snes_max_it={_format_int_for_log(opts.get('snes_max_it')):>4} "
                f"snes_atol={_format_float_for_log(opts.get('snes_atol')):>11} "
                f"snes_rtol={_format_float_for_log(opts.get('snes_rtol')):>11} "
                f"start={_format_guess_for_log(current_guess)}"
            )

        try:
            optimized_controls = adj.minimize(
                rf,
                request.optimizer_method,
                **minimize_kwargs,
            )
            attempt_logs.append(
                RecoveryAttempt(
                    attempt_index=attempt,
                    phase=phase,
                    solver_options=copy.deepcopy(inverse_solver_params.solver_options if hasattr(inverse_solver_params, 'solver_options') else inverse_solver_params[10]),
                    start_guess=copy.deepcopy(current_guess),
                    status="success",
                    reason="Optimization completed.",
                    best_objective_seen=monitor.best_objective,
                    best_estimate_seen=copy.deepcopy(
                        monitor.best_estimate if monitor.best_estimate is not None else monitor.last_estimate
                    ),
                )
            )
            return optimized_controls, inverse_solver_params, rf, attempt_logs
        except Exception as exc:
            last_error = exc
            failure_reason = _summarize_exception(exc)
            attempt_safe_guess = (
                monitor.best_estimate
                if monitor.best_estimate is not None
                else monitor.last_estimate
            )
            if attempt_safe_guess is not None:
                last_safe_guess = copy.deepcopy(attempt_safe_guess)
            attempt_logs.append(
                RecoveryAttempt(
                    attempt_index=attempt,
                    phase=phase,
                    solver_options=copy.deepcopy(inverse_solver_params.solver_options if hasattr(inverse_solver_params, 'solver_options') else inverse_solver_params[10]),
                    start_guess=copy.deepcopy(current_guess),
                    status="failed",
                    reason=failure_reason,
                    best_objective_seen=monitor.best_objective,
                    best_estimate_seen=copy.deepcopy(
                        monitor.best_estimate if monitor.best_estimate is not None else monitor.last_estimate
                    ),
                )
            )

            if attempt >= max_attempts - 1:
                break

            next_phase, next_phase_step, _next_cycle_index = _attempt_phase_state(
                attempt + 1, request.recovery
            )
            next_guess = _next_guess_after_failure(
                current_guess=current_guess,
                monitor=monitor,
                last_safe_guess=last_safe_guess,
                next_phase=next_phase,
                next_phase_step=next_phase_step,
                recovery=request.recovery,
            )
            if request.recovery.verbose:
                print(
                    "[resilient] "
                    f"failure={attempt + 1:03d} "
                    f"reason={failure_reason} "
                    f"next_phase={next_phase:<15} "
                    f"\nstep={next_phase_step:02d} "
                    f"next_start={_format_guess_for_log(next_guess)}"
                )
            current_guess = next_guess

    summary = _format_recovery_summary(attempt_logs)
    last_error_summary = _summarize_exception(last_error) if last_error is not None else "Unknown"
    raise RuntimeError(
        "Resilient minimization failed after all attempts. "
        f"Last error: {last_error_summary}. "
        f"Attempts: {summary}"
    )


def _build_minimize_kwargs(request: InferenceRequest, *, bounds: Optional[Any]) -> Dict[str, Any]:
    """Build argument dictionary forwarded to ``firedrake.adjoint.minimize``."""
    kwargs: Dict[str, Any] = {}
    if request.tolerance is not None:
        kwargs["tol"] = request.tolerance
    if request.optimizer_options is not None:
        kwargs["options"] = dict(request.optimizer_options)
    if bounds is not None:
        kwargs["bounds"] = bounds
    return kwargs


def _attempt_phase_state(
    attempt: int, recovery: RecoveryConfig
) -> Tuple[str, int, int]:
    """Return ``(phase, phase_step, cycle_index)`` for a retry attempt index."""
    if attempt <= 0:
        return "baseline", 1, 0

    max_it_only = max(1, int(recovery.max_it_only_attempts))
    anis_only = max(1, int(recovery.anisotropy_only_attempts))
    tol_only = max(1, int(recovery.tolerance_relax_attempts))
    cycle_len = max_it_only + anis_only + tol_only

    cycle_offset = int(attempt - 1)
    cycle_index = cycle_offset // cycle_len
    idx = cycle_offset % cycle_len

    if idx < max_it_only:
        return "max_it", idx + 1, cycle_index
    idx -= max_it_only
    if idx < anis_only:
        return "anisotropy", idx + 1, cycle_index
    idx -= anis_only
    return "tolerance_relax", idx + 1, cycle_index


def _relax_solver_options_for_attempt(
    solver_params: List[Any],
    *,
    attempt: int,
    phase: str,
    phase_step: int,
    cycle_index: int,
    recovery: RecoveryConfig,
    baseline_options: Optional[Mapping[str, Any]] = None,
) -> None:
    """Relax nonlinear/linear solver settings in-place for retry attempts."""
    params = solver_params.solver_options if hasattr(solver_params, 'solver_options') else solver_params[10]
    if not isinstance(params, dict):
        return

    base = baseline_options if isinstance(baseline_options, Mapping) else params
    base_max_it = int(base.get("snes_max_it", params.get("snes_max_it", 80)))
    base_atol = float(base.get("snes_atol", params.get("snes_atol", 1e-8)))
    base_rtol = float(base.get("snes_rtol", params.get("snes_rtol", 1e-8)))
    base_ksp_rtol = float(base.get("ksp_rtol", params.get("ksp_rtol", 1e-8)))
    base_linesearch = base.get("snes_linesearch_type", params.get("snes_linesearch_type"))

    params.setdefault("snes_error_if_not_converged", True)
    params.setdefault("ksp_error_if_not_converged", True)

    def _reset_relaxation_knobs() -> None:
        params["snes_atol"] = base_atol
        params["snes_rtol"] = base_rtol
        params["ksp_rtol"] = base_ksp_rtol
        params["snes_max_it"] = base_max_it
        if base_linesearch is not None:
            params["snes_linesearch_type"] = base_linesearch

    if phase == "baseline":
        _reset_relaxation_knobs()
        return

    if phase == "max_it":
        _reset_relaxation_knobs()
        params["snes_max_it"] = int(
            min(
                float(recovery.max_it_cap),
                base_max_it * (float(recovery.max_it_growth) ** int(max(1, phase_step))),
            )
        )
        return

    if phase == "anisotropy":
        _reset_relaxation_knobs()
        return

    if phase != "tolerance_relax":
        return

    _reset_relaxation_knobs()
    local_step = int(max(1, phase_step))
    params["snes_atol"] = base_atol * (float(recovery.atol_relax_factor) ** local_step)
    params["snes_rtol"] = base_rtol * (float(recovery.rtol_relax_factor) ** local_step)
    params["ksp_rtol"] = base_ksp_rtol * (float(recovery.ksp_rtol_relax_factor) ** local_step)

    if recovery.line_search_schedule:
        idx = min(local_step - 1, len(recovery.line_search_schedule) - 1)
        params["snes_linesearch_type"] = recovery.line_search_schedule[idx]


def _next_guess_after_failure(
    *,
    current_guess: Any,
    monitor: _AttemptMonitor,
    last_safe_guess: Optional[Any],
    next_phase: str,
    next_phase_step: int,
    recovery: RecoveryConfig,
) -> Any:
    """Compute a safer initial guess after an attempt fails."""
    safe_guess = monitor.best_estimate
    if safe_guess is None:
        safe_guess = monitor.last_estimate
    if safe_guess is None:
        safe_guess = last_safe_guess

    if safe_guess is None:
        restart_guess = _scale_guess(
            current_guess, factor=max(0.0, 1.0 - recovery.fallback_shrink_if_stuck)
        )
    else:
        restart_guess = copy.deepcopy(safe_guess)

    if next_phase == "anisotropy":
        effective_blend = 1.0 - (1.0 - float(recovery.anisotropy_blend)) ** max(
            1, int(next_phase_step)
        )
        flattened = _reduce_guess_anisotropy(
            restart_guess,
            target_ratio=float(recovery.anisotropy_target_ratio),
            blend=effective_blend,
        )
        if not _guesses_close(flattened, restart_guess):
            restart_guess = flattened
    return restart_guess


def _blend_guess(*, current_guess: Any, safe_guess: Any, contraction_factor: float) -> Any:
    """Blend current guess toward a known-safe guess."""
    cur, cur_is_vector = _guess_to_array(current_guess)
    safe, safe_is_vector = _guess_to_array(safe_guess)
    cur, safe, is_vector = _align_guess_arrays(cur, safe, cur_is_vector, safe_is_vector)
    alpha = min(max(float(contraction_factor), 0.0), 1.0)
    out = safe + alpha * (cur - safe)
    return _array_to_guess(out, is_vector)


def _scale_guess(value: Any, *, factor: float) -> Any:
    """Scale scalar or vector guesses by a constant factor."""
    arr, is_vector = _guess_to_array(value)
    out = arr * float(factor)
    return _array_to_guess(out, is_vector)


def _reduce_guess_anisotropy(value: Any, *, target_ratio: float, blend: float) -> Any:
    """Reduce anisotropy (max/min magnitude ratio) in vector guesses."""
    arr, is_vector = _guess_to_array(value)
    if arr.size < 2:
        return value

    mags = np.maximum(np.abs(arr), 1e-14)
    current_ratio = float(np.max(mags) / np.min(mags))
    if current_ratio <= max(1.0, float(target_ratio)):
        return value

    geo = float(np.exp(np.mean(np.log(mags))))
    isotropic = np.sign(arr) * geo
    isotropic[np.sign(arr) == 0.0] = geo

    beta = min(max(float(blend), 0.0), 1.0)
    out = (1.0 - beta) * arr + beta * isotropic
    return _array_to_guess(out, is_vector)


def _guesses_close(a: Any, b: Any) -> bool:
    """Check approximate equality for scalar or vector guesses."""
    aa, aa_vec = _guess_to_array(a)
    bb, bb_vec = _guess_to_array(b)
    aa, bb, _ = _align_guess_arrays(aa, bb, aa_vec, bb_vec)
    return bool(np.allclose(aa, bb, rtol=1e-12, atol=1e-12))


def _guess_to_array(value: Any) -> Tuple[np.ndarray, bool]:
    """Convert a scalar/list guess to ``(array, is_vector)``."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        return arr, True
    return np.asarray([float(value)], dtype=float), False


def _align_guess_arrays(
    a: np.ndarray,
    b: np.ndarray,
    a_is_vector: bool,
    b_is_vector: bool,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Align scalar/vector arrays so elementwise operations are valid."""
    if a.size == b.size:
        return a, b, bool(a_is_vector or b_is_vector or a.size > 1)
    if a.size == 1:
        return np.full_like(b, a[0], dtype=float), b, True
    if b.size == 1:
        return a, np.full_like(a, b[0], dtype=float), True
    raise ValueError(f"Guess shape mismatch: {a.size} vs {b.size}.")


def _array_to_guess(arr: np.ndarray, is_vector: bool) -> Any:
    """Convert internal array form back to scalar/list guess form."""
    flat = np.asarray(arr, dtype=float).ravel()
    if is_vector or flat.size > 1:
        return [float(v) for v in flat.tolist()]
    return float(flat[0])
