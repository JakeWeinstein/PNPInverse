"""End-to-end runner for unified inverse parameter inference.

This module provides:
- synthetic observation generation from any compatible forward-solver adapter
- reduced-functional construction for configurable objective fields
- one orchestration function that performs data generation + optimization

The API is intentionally explicit to keep future target/solver additions simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .parameter_targets import ParameterTarget
from .solver_interface import (
    ForwardSolverAdapter,
    as_species_list,
    deep_copy_solver_params,
    extract_solution_vectors,
)


@dataclass
class SyntheticData:
    """Clean/noisy terminal fields produced by one forward solve."""

    clean_concentration_vectors: List[np.ndarray]
    clean_phi_vector: np.ndarray
    noisy_concentration_vectors: List[np.ndarray]
    noisy_phi_vector: np.ndarray

    def select_targets(self, *, use_noisy_data: bool) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return objective targets from clean or noisy fields."""
        # One toggle controls whether the inverse solve fits clean or noisy data.
        if use_noisy_data:
            return self.noisy_concentration_vectors, self.noisy_phi_vector
        return self.clean_concentration_vectors, self.clean_phi_vector


@dataclass
class InferenceRequest:
    """Configuration for one inverse run.

    Parameters
    ----------
    adapter:
        Forward solver adapter created from a compatible ``forsolve`` module.
    target:
        Parameter target definition from the target registry.
    base_solver_params:
        Baseline 11-entry solver parameter list. ``target`` will inject true and
        guess values into copies of this list.
    true_value:
        Ground-truth value used to generate synthetic observations.
    initial_guess:
        Starting value used to initialize optimization controls.
    noise_percent:
        Gaussian noise level in percent of RMS(field).
    seed:
        RNG seed for reproducible noise.
    optimizer_method:
        Method name forwarded to ``firedrake.adjoint.minimize``.
    optimizer_options:
        Optional method-specific options.
    tolerance:
        Optional optimizer tolerance.
    bounds:
        Optional bounds. If ``None``, target defaults are used.
    fit_to_noisy_data:
        If True, infer against noisy observations; else clean observations.
    blob_initial_condition:
        Passed through to forward solver initialization.
    print_interval_data:
        Forward solve print interval for data generation.
    print_interval_inverse:
        Forward solve print interval inside taped objective evaluations.
    recovery_attempts:
        Optional override for total resilient-minimization attempts. If set,
        this value takes priority over ``recovery.max_attempts``.
    recovery:
        Configuration for the always-on resilient minimizer.
    """

    adapter: ForwardSolverAdapter
    target: ParameterTarget
    base_solver_params: Sequence[Any]
    true_value: Any
    initial_guess: Any
    noise_percent: float = 10.0
    seed: Optional[int] = None
    optimizer_method: str = "L-BFGS-B"
    optimizer_options: Optional[Mapping[str, Any]] = None
    tolerance: Optional[float] = 1e-8
    bounds: Optional[Any] = None
    fit_to_noisy_data: bool = True
    blob_initial_condition: bool = True
    print_interval_data: int = 100
    print_interval_inverse: int = 100
    recovery_attempts: Optional[int] = None
    recovery: "RecoveryConfig" = field(default_factory=lambda: RecoveryConfig())


@dataclass
class RecoveryConfig:
    """Configuration for resilient minimization retries.

    The first attempt uses user-provided solver options. Subsequent attempts
    cycle through staged recovery:
    1) ``max_it`` attempts (increase ``snes_max_it`` only),
    2) ``anisotropy`` attempts (reduce parameter anisotropy),
    3) ``tolerance_relax`` attempts (loosen atol/rtol).

    The cycle then repeats. Entering the anisotropy stage resets max-iteration
    and tolerance relaxations to their baseline values for a fresh restart.
    """

    max_attempts: int = 15
    contraction_factor: float = 0.5
    fallback_shrink_if_stuck: float = 0.15
    max_it_only_attempts: int = 2
    anisotropy_only_attempts: int = 3
    tolerance_relax_attempts: int = 1
    anisotropy_target_ratio: float = 3.0
    anisotropy_blend: float = 0.5
    atol_relax_factor: float = 10.0
    rtol_relax_factor: float = 10.0
    ksp_rtol_relax_factor: float = 10.0
    max_it_growth: float = 1.5
    max_it_cap: int = 500
    line_search_schedule: Tuple[str, ...] = ("bt", "l2", "cp", "basic")
    verbose: bool = True


@dataclass
class RecoveryAttempt:
    """Per-attempt log entry produced by resilient minimization."""

    attempt_index: int
    phase: str
    solver_options: Dict[str, Any]
    start_guess: Any
    status: str
    reason: str
    best_objective_seen: Optional[float] = None
    best_estimate_seen: Optional[Any] = None


@dataclass
class InferenceResult:
    """Outputs returned by :func:`run_inverse_inference`."""

    target_key: str
    estimate: Any
    objective_value: float
    true_solver_params: List[Any]
    inverse_solver_params: List[Any]
    synthetic_data: SyntheticData
    optimized_controls: Any
    recovery_attempts: List[RecoveryAttempt]


def build_default_solver_params(
    *,
    n_species: int,
    order: int = 1,
    dt: float = 1e-2,
    t_end: float = 0.1,
    z_vals: Optional[Sequence[float]] = None,
    d_vals: Optional[Sequence[float]] = None,
    a_vals: Optional[Sequence[float]] = None,
    phi_applied: float = 0.05,
    c0_vals: Optional[Sequence[float]] = None,
    phi0: float = 1.0,
    solver_options: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Build a standard 11-entry solver parameter list.

    This utility centralizes the common list structure used across scripts and
    helps keep the unified interface consistent.
    """
    n = int(n_species)
    if n <= 0:
        raise ValueError(f"n_species must be positive; got {n}.")

    z_default = [1 if i % 2 == 0 else -1 for i in range(n)]
    d_default = [1.0 for _ in range(n)]
    a_default = [0.0 for _ in range(n)]
    c0_default = [0.1 for _ in range(n)]

    z_values = as_species_list(z_vals if z_vals is not None else z_default, n, "z_vals")
    d_values = as_species_list(d_vals if d_vals is not None else d_default, n, "d_vals")
    a_values = as_species_list(a_vals if a_vals is not None else a_default, n, "a_vals")
    c0_values = as_species_list(c0_vals if c0_vals is not None else c0_default, n, "c0_vals")

    options = dict(solver_options or {})

    # Canonical 11-entry structure used by existing forsolve modules:
    # [n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params]
    return [
        n,
        int(order),
        float(dt),
        float(t_end),
        z_values,
        d_values,
        a_values,
        float(phi_applied),
        c0_values,
        float(phi0),
        options,
    ]


def generate_synthetic_data(
    adapter: ForwardSolverAdapter,
    solver_params: Sequence[Any],
    *,
    noise_percent: float,
    seed: Optional[int] = None,
    blob_initial_condition: bool = True,
    print_interval: int = 100,
) -> SyntheticData:
    """Generate clean/noisy final-time observations from a forward solve.

    Notes
    -----
    This function is solver-agnostic as long as the adapter contract is met.
    """
    # Copy inputs so synthetic data generation never mutates caller-owned config.
    params_copy = deep_copy_solver_params(solver_params)
    n_species = int(params_copy[0])

    # Reuse adapter pipeline; this works for Dirichlet, Robin, or new solvers.
    _, U_final = adapter.run_forward(
        params_copy,
        blob_initial_condition=blob_initial_condition,
        print_interval=print_interval,
    )

    clean_c, clean_phi = extract_solution_vectors(U_final, n_species)

    # Add independent Gaussian perturbations to each observed field.
    rng = np.random.default_rng(seed)
    noisy_c = [_add_percent_noise(vec, noise_percent, rng) for vec in clean_c]
    noisy_phi = _add_percent_noise(clean_phi, noise_percent, rng)

    return SyntheticData(
        clean_concentration_vectors=clean_c,
        clean_phi_vector=clean_phi,
        noisy_concentration_vectors=noisy_c,
        noisy_phi_vector=noisy_phi,
    )


def build_reduced_functional(
    *,
    adapter: ForwardSolverAdapter,
    target: ParameterTarget,
    solver_params: Sequence[Any],
    concentration_targets: Sequence[Sequence[float]],
    phi_target: Sequence[float],
    blob_initial_condition: bool = True,
    print_interval: int = 100,
    extra_eval_cb_pre: Optional[Any] = None,
    extra_eval_cb_post: Optional[Any] = None,
):
    """Build a Firedrake-adjoint reduced functional for a configured target.

    Parameters
    ----------
    adapter:
        Forward-solver adapter that builds and solves the PDE system.
    target:
        Parameter target that provides controls and objective field selection.
    solver_params:
        Inverse-run solver parameters with initial-guess parameter values.
    concentration_targets:
        One coefficient vector per species.
    phi_target:
        Electric-potential coefficient vector.
    blob_initial_condition:
        Forward initialization mode.
    print_interval:
        Solve progress print frequency.
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    params_copy = deep_copy_solver_params(solver_params)
    n_species = int(params_copy[0])

    if len(concentration_targets) != n_species:
        raise ValueError(
            f"Expected {n_species} concentration targets, got {len(concentration_targets)}."
        )

    # Build the PDE objects/state that define the taped forward model.
    ctx = adapter.build_context(params_copy)
    ctx = adapter.build_forms(ctx, params_copy)

    # Convert numpy vectors to Firedrake Functions in the scalar space.
    c_target_fs = [
        _vector_to_function(ctx, c_vec, space_key="V_scalar")
        for c_vec in concentration_targets
    ]
    phi_target_f = _vector_to_function(ctx, phi_target, space_key="V_scalar")

    # Ensure objective construction starts from a clean tape.
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    # Everything below is annotated and contributes to the optimization graph.
    adapter.set_initial_conditions(ctx, params_copy, blob=blob_initial_condition)
    U_final = adapter.solve(ctx, params_copy, print_interval=print_interval)

    objective_terms = []
    if "concentration" in target.objective_fields:
        # Add L2 mismatch for each concentration species.
        for i in range(n_species):
            diff_i = U_final.sub(i) - c_target_fs[i]
            objective_terms.append(fd.inner(diff_i, diff_i))

    if "phi" in target.objective_fields:
        # Add L2 mismatch for electric potential.
        phi_diff = U_final.sub(n_species) - phi_target_f
        objective_terms.append(fd.inner(phi_diff, phi_diff))

    if not objective_terms:
        raise ValueError(
            f"Target '{target.key}' must request at least one objective field."
        )

    Jobj = 0.5 * fd.assemble(sum(objective_terms) * fd.dx)

    # Ask the target definition which state variables are optimization controls.
    control_functions = list(target.controls_from_context(ctx))
    if not control_functions:
        raise ValueError(f"Target '{target.key}' returned no optimization controls.")

    controls = [adj.Control(ctrl) for ctrl in control_functions]
    control_arg = controls[0] if len(controls) == 1 else controls

    # Targets can optionally inject callbacks for logging or control sync.
    # Extra callbacks are used by the resilient minimizer to monitor feasibility.
    rf_kwargs: Dict[str, Any] = {}
    target_pre = target.eval_cb_pre_factory(ctx) if target.eval_cb_pre_factory is not None else None
    target_post = (
        target.eval_cb_post_factory(ctx) if target.eval_cb_post_factory is not None else None
    )

    if target_pre is not None or extra_eval_cb_pre is not None:
        def chained_pre(m: Any) -> None:
            if target_pre is not None:
                target_pre(m)
            if extra_eval_cb_pre is not None:
                extra_eval_cb_pre(m)

        rf_kwargs["eval_cb_pre"] = chained_pre

    if target_post is not None or extra_eval_cb_post is not None:
        def chained_post(j: float, m: Any) -> None:
            if target_post is not None:
                target_post(j, m)
            if extra_eval_cb_post is not None:
                extra_eval_cb_post(j, m)

        rf_kwargs["eval_cb_post"] = chained_post

    rf = adj.ReducedFunctional(Jobj, control_arg, **rf_kwargs)
    return rf


@dataclass
class _AttemptMonitor:
    """Collect feasible points seen during one optimization attempt."""

    target: ParameterTarget
    best_objective: Optional[float] = None
    best_estimate: Optional[Any] = None
    last_estimate: Optional[Any] = None
    n_successful_evals: int = 0

    def eval_cb_post(self, j: float, m: Any) -> None:
        """Observe successful objective evaluations from the reduced functional."""
        try:
            estimate = copy.deepcopy(self.target.estimate_from_controls(m))
        except Exception:
            return

        self.last_estimate = estimate
        j_float = float(j)
        if np.isfinite(j_float) and (
            self.best_objective is None or j_float < self.best_objective
        ):
            self.best_objective = j_float
            self.best_estimate = copy.deepcopy(estimate)
        self.n_successful_evals += 1


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

    n_species = int(base_solver_params[0])
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
        baseline_solver_options = (
            copy.deepcopy(inverse_solver_params[10])
            if isinstance(inverse_solver_params[10], dict)
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
            opts = inverse_solver_params[10] if isinstance(inverse_solver_params[10], dict) else {}
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
                    solver_options=copy.deepcopy(inverse_solver_params[10]),
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
                # Persist the newest feasible point across attempts so a later
                # immediate divergence can still restart from a known-safe state.
                last_safe_guess = copy.deepcopy(attempt_safe_guess)
            attempt_logs.append(
                RecoveryAttempt(
                    attempt_index=attempt,
                    phase=phase,
                    solver_options=copy.deepcopy(inverse_solver_params[10]),
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
    """Relax nonlinear/linear solver settings in-place for retry attempts.

    ``baseline_options`` is a per-attempt snapshot of user/default solver
    options. We always relax from this baseline so tolerance changes reset at
    the start of each cycle and never compound across cycles.
    """
    params = solver_params[10]
    if not isinstance(params, dict):
        return

    base = baseline_options if isinstance(baseline_options, Mapping) else params
    base_max_it = int(base.get("snes_max_it", params.get("snes_max_it", 80)))
    base_atol = float(base.get("snes_atol", params.get("snes_atol", 1e-8)))
    base_rtol = float(base.get("snes_rtol", params.get("snes_rtol", 1e-8)))
    base_ksp_rtol = float(base.get("ksp_rtol", params.get("ksp_rtol", 1e-8)))
    base_linesearch = base.get("snes_linesearch_type", params.get("snes_linesearch_type"))

    # Make divergence explicit so failed solves surface as exceptions.
    params.setdefault("snes_error_if_not_converged", True)
    params.setdefault("ksp_error_if_not_converged", True)

    # Keep a single reset path so non-tolerance stages always return to baseline.
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
        # Stage 1: increase max nonlinear iterations only.
        params["snes_max_it"] = int(
            min(
                float(recovery.max_it_cap),
                base_max_it * (float(recovery.max_it_growth) ** int(max(1, phase_step))),
            )
        )
        return

    if phase == "anisotropy":
        # Stage 2: reset solver relaxation knobs to baseline while parameters are
        # anisotropy-reduced. This explicitly clears any tolerance/max-it relax.
        _reset_relaxation_knobs()
        return

    if phase != "tolerance_relax":
        return

    # Stage 3: loosen tolerances. Use phase-local step so relaxation resets
    # automatically every cycle.
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
    # Always prefer the most recent known-safe point from this attempt; if this
    # attempt had no successful evaluations, fall back to the best safe point
    # remembered from prior attempts.
    safe_guess = monitor.best_estimate
    if safe_guess is None:
        safe_guess = monitor.last_estimate
    if safe_guess is None:
        safe_guess = last_safe_guess

    if safe_guess is None:
        # If no feasible point is known yet, shrink the current guess magnitude.
        restart_guess = _scale_guess(
            current_guess, factor=max(0.0, 1.0 - recovery.fallback_shrink_if_stuck)
        )
    else:
        # Start from the latest known-safe parameters, not the original guess.
        restart_guess = copy.deepcopy(safe_guess)

    # In anisotropy phase, flatten component ratios before the next attempt.
    if next_phase == "anisotropy":
        # Apply stronger flattening over repeated anisotropy attempts.
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
    alpha = float(contraction_factor)
    alpha = min(max(alpha, 0.0), 1.0)
    out = safe + alpha * (cur - safe)
    return _array_to_guess(out, is_vector)


def _scale_guess(value: Any, *, factor: float) -> Any:
    """Scale scalar or vector guesses by a constant factor."""
    arr, is_vector = _guess_to_array(value)
    out = arr * float(factor)
    return _array_to_guess(out, is_vector)


def _reduce_guess_anisotropy(value: Any, *, target_ratio: float, blend: float) -> Any:
    """Reduce anisotropy (max/min magnitude ratio) in vector guesses.

    For scalar guesses or already-near-isotropic vectors, this is a no-op.
    """
    arr, is_vector = _guess_to_array(value)
    if arr.size < 2:
        return value

    mags = np.maximum(np.abs(arr), 1e-14)
    current_ratio = float(np.max(mags) / np.min(mags))
    if current_ratio <= max(1.0, float(target_ratio)):
        return value

    # Build an isotropic reference using geometric-mean magnitude to preserve scale.
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


def _format_float_for_log(value: Any) -> str:
    """Return fixed-width scientific notation for log readability."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.3e}"
    except Exception:
        return str(value)


def _format_int_for_log(value: Any) -> str:
    """Return integer-like value as a compact log string."""
    if value is None:
        return "-"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def _format_plain_float_for_log(value: Any) -> str:
    """Return fixed-width decimal float text without scientific notation."""
    if value is None:
        return f"{'-':>12}"
    try:
        v = float(value)
    except Exception:
        return f"{str(value):>12}"
    if not np.isfinite(v):
        return f"{str(v):>12}"
    # Keep a fixed column width so fast iteration logs do not visually jitter.
    if abs(v) < 5e-7:
        v = 0.0
    return f"{v:>12.6f}"


def _format_guess_for_log(value: Any) -> str:
    """Format scalar/vector guesses with consistent numeric spacing."""
    arr, _ = _guess_to_array(value)
    if arr.size == 1:
        return _format_plain_float_for_log(arr[0])
    vals = ", ".join(_format_plain_float_for_log(v) for v in arr.tolist())
    return f"[{vals}]"


def _summarize_exception(exc: Optional[Exception]) -> str:
    """Build a compact exception summary without long file paths/trace text."""
    if exc is None:
        return "Unknown"

    msg = str(exc).strip()
    if not msg:
        return type(exc).__name__

    lines = [line.strip() for line in msg.splitlines() if line.strip()]
    if not lines:
        return type(exc).__name__

    # Prefer lines that include nonlinear-convergence reason tokens.
    preferred = None
    for line in lines:
        upper = line.upper()
        if "DIVERGED_" in upper or "FAILED TO CONVERGE" in upper or "CONVERGE" in upper:
            preferred = line
            break
    if preferred is None:
        preferred = lines[-1]

    # Replace absolute paths to keep logs compact.
    preferred = re.sub(r"(?:[A-Za-z]:)?/(?:[^/\s:]+/)*[^/\s:]+", "<path>", preferred)
    preferred = re.sub(r"\s+", " ", preferred).strip()
    return f"{type(exc).__name__}: {preferred}"


def _format_recovery_summary(attempts: Sequence[RecoveryAttempt]) -> str:
    """Compact human-readable summary for failure messages."""
    if not attempts:
        return "no attempts logged"
    parts = []
    for a in attempts:
        short_reason = a.reason
        if len(short_reason) > 180:
            short_reason = short_reason[:177] + "..."
        parts.append(
            f"(attempt={a.attempt_index}, phase={a.phase}, status={a.status}, reason={short_reason})"
        )
    return "; ".join(parts)


def run_inverse_inference(request: InferenceRequest) -> InferenceResult:
    """Run one complete inverse problem from synthetic data to estimate.

    Workflow
    --------
    1. Inject the target true value into solver parameters.
    2. Generate synthetic clean/noisy data with annotation disabled.
    3. Inject the target initial guess into solver parameters.
    4. Build a reduced functional for the selected objective fields.
    5. Optimize with Firedrake-adjoint and return a structured result.
    """
    import firedrake.adjoint as adj

    # Start from a stable base configuration and derive true/guess variants from it.
    base_params = deep_copy_solver_params(request.base_solver_params)

    true_solver_params = request.target.apply_value(base_params, request.true_value)

    # Data generation should not contribute to the taped optimization graph.
    with adj.stop_annotating():
        synthetic_data = generate_synthetic_data(
            request.adapter,
            true_solver_params,
            noise_percent=float(request.noise_percent),
            seed=request.seed,
            blob_initial_condition=request.blob_initial_condition,
            print_interval=int(request.print_interval_data),
        )

    concentration_targets, phi_target = synthetic_data.select_targets(
        use_noisy_data=bool(request.fit_to_noisy_data)
    )

    # Always use resilient minimization, which retries with relaxed solver
    # settings and safer restart guesses if a forward solve diverges.
    optimized_controls, inverse_solver_params, rf, recovery_attempts = resilient_minimize(
        request=request,
        base_solver_params=base_params,
        concentration_targets=concentration_targets,
        phi_target=phi_target,
    )

    estimate = request.target.estimate_from_controls(optimized_controls)
    objective_value = float(rf(optimized_controls))

    return InferenceResult(
        target_key=request.target.key,
        estimate=estimate,
        objective_value=objective_value,
        true_solver_params=true_solver_params,
        inverse_solver_params=inverse_solver_params,
        synthetic_data=synthetic_data,
        optimized_controls=optimized_controls,
        recovery_attempts=recovery_attempts,
    )


def _vector_to_function(ctx: Dict[str, Any], vec: Sequence[float], *, space_key: str = "V_scalar"):
    """Create a Firedrake Function by directly setting coefficient vector values."""
    import firedrake as fd

    # Pull the target function space from context built by the forward solver.
    V = ctx[space_key]
    out = fd.Function(V)
    flat_vec = np.asarray(vec, dtype=float).ravel()

    # Catch shape mismatches early so objective setup fails fast and clearly.
    if flat_vec.size != out.dat.data.size:
        raise ValueError(
            f"Target vector length {flat_vec.size} != DOFs {out.dat.data.size} for {space_key}."
        )

    out.dat.data[:] = flat_vec
    return out


def _add_percent_noise(vec: Sequence[float], noise_percent: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise with sigma = ``noise_percent / 100 * RMS(vec)``."""
    v = np.asarray(vec, dtype=float)
    pct = float(noise_percent)
    if pct < 0:
        raise ValueError(f"noise_percent must be non-negative; got {pct}.")
    if pct == 0:
        return v.copy()

    # Scale sigma by field magnitude (RMS) so noise is relative, not absolute.
    rms = float(np.sqrt(np.mean(v * v)))
    sigma = (pct / 100.0) * max(rms, 1e-12)
    return v + rng.normal(0.0, sigma, size=v.shape)
