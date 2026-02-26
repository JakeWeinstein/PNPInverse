# Backward-compatibility shim. New code should import from Inverse directly.
from Inverse import *  # noqa: F401, F403
from Inverse import (  # noqa: F401
    ForwardSolverAdapter,
    ParameterTarget,
    build_default_target_registry,
    InferenceRequest,
    InferenceResult,
    RecoveryConfig,
    RecoveryAttempt,
    SyntheticData,
    build_default_solver_params,
    build_reduced_functional,
    generate_synthetic_data,
    resilient_minimize,
    run_inverse_inference,
)

__all__ = [
    "ForwardSolverAdapter",
    "ParameterTarget",
    "InferenceRequest",
    "InferenceResult",
    "RecoveryConfig",
    "RecoveryAttempt",
    "SyntheticData",
    "build_default_solver_params",
    "build_reduced_functional",
    "generate_synthetic_data",
    "resilient_minimize",
    "run_inverse_inference",
    "build_default_target_registry",
]
