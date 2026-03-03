"""End-to-end runner for unified inverse parameter inference.

This package provides:
- synthetic observation generation from any compatible forward-solver adapter
- reduced-functional construction for configurable objective fields
- one orchestration function that performs data generation + optimization

The API is intentionally explicit to keep future target/solver additions simple.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .config import (
    InferenceRequest,
    InferenceResult,
    RecoveryAttempt,
    RecoveryConfig,
    SyntheticData,
)
from .data import (
    build_default_solver_params,
    generate_synthetic_data,
    _add_percent_noise,
    _vector_to_function,
)
from .objective import (
    build_reduced_functional,
    _AttemptMonitor,
)
from .recovery import (
    resilient_minimize,
    _build_minimize_kwargs,
    _attempt_phase_state,
    _relax_solver_options_for_attempt,
    _next_guess_after_failure,
    _blend_guess,
    _scale_guess,
    _reduce_guess_anisotropy,
    _guesses_close,
    _guess_to_array,
    _align_guess_arrays,
    _array_to_guess,
)
from .formatting import (
    _format_float_for_log,
    _format_int_for_log,
    _format_plain_float_for_log,
    _format_guess_for_log,
    _summarize_exception,
    _format_recovery_summary,
)

from ..solver_interface import deep_copy_solver_params


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

    base_params = deep_copy_solver_params(request.base_solver_params)
    true_solver_params = request.target.apply_value(base_params, request.true_value)

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


__all__ = [
    # config.py
    "SyntheticData",
    "InferenceRequest",
    "RecoveryConfig",
    "RecoveryAttempt",
    "InferenceResult",
    # data.py
    "build_default_solver_params",
    "generate_synthetic_data",
    "_add_percent_noise",
    "_vector_to_function",
    # objective.py
    "build_reduced_functional",
    "_AttemptMonitor",
    # recovery.py
    "resilient_minimize",
    "_build_minimize_kwargs",
    "_attempt_phase_state",
    "_relax_solver_options_for_attempt",
    "_next_guess_after_failure",
    "_blend_guess",
    "_scale_guess",
    "_reduce_guess_anisotropy",
    "_guesses_close",
    "_guess_to_array",
    "_align_guess_arrays",
    "_array_to_guess",
    # formatting.py
    "_format_float_for_log",
    "_format_int_for_log",
    "_format_plain_float_for_log",
    "_format_guess_for_log",
    "_summarize_exception",
    "_format_recovery_summary",
    # __init__.py (orchestrator)
    "run_inverse_inference",
]
