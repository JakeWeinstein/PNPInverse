"""Public API for the unified inverse interface."""

# Core orchestration/data-structure exports.
from .inference_runner import (
    InferenceRequest,
    InferenceResult,
    RecoveryAttempt,
    RecoveryConfig,
    SyntheticData,
    build_default_solver_params,
    build_reduced_functional,
    generate_synthetic_data,
    resilient_minimize,
    run_inverse_inference,
)
# Target registry exports.
from .parameter_targets import ParameterTarget, build_default_target_registry
# Adapter export used to plug in different forsolve modules.
from .solver_interface import ForwardSolverAdapter

# Explicit export list keeps wildcard imports predictable for users.
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
