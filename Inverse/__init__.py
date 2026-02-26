"""Inverse inference package for PDE-constrained parameter identification.

This package is the canonical home of the unified inverse framework.
``UnifiedInverse/`` is a backward-compatible re-export shim pointing here.

Public API::

    from Inverse import (
        build_default_solver_params,
        run_inverse_inference,
        InferenceRequest,
        InferenceResult,
        RecoveryConfig,
        SyntheticData,
        ForwardSolverAdapter,
        ParameterTarget,
        build_default_target_registry,
        as_species_list,
        deep_copy_solver_params,
        extract_solution_vectors,
    )
"""

from Inverse.solver_interface import (
    ForwardSolverAdapter,
    as_species_list,
    deep_copy_solver_params,
    extract_solution_vectors,
)

from Inverse.parameter_targets import (
    ParameterTarget,
    build_default_target_registry,
    ensure_sequence,
)

from Inverse.inference_runner import (
    SyntheticData,
    InferenceRequest,
    InferenceResult,
    RecoveryConfig,
    RecoveryAttempt,
    build_default_solver_params,
    generate_synthetic_data,
    build_reduced_functional,
    resilient_minimize,
    run_inverse_inference,
)

__all__ = [
    # solver_interface
    "ForwardSolverAdapter",
    "as_species_list",
    "deep_copy_solver_params",
    "extract_solution_vectors",
    # parameter_targets
    "ParameterTarget",
    "build_default_target_registry",
    "ensure_sequence",
    # inference_runner
    "SyntheticData",
    "InferenceRequest",
    "InferenceResult",
    "RecoveryConfig",
    "RecoveryAttempt",
    "build_default_solver_params",
    "generate_synthetic_data",
    "build_reduced_functional",
    "resilient_minimize",
    "run_inverse_inference",
]
