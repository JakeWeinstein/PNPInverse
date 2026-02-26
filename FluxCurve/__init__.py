"""FluxCurve — adjoint-gradient Robin-kappa inference from phi_applied-flux curves.

Public API:

    from FluxCurve import (
        ForwardRecoveryConfig,
        RobinFluxCurveInferenceRequest,
        RobinFluxCurveInferenceResult,
        run_robin_kappa_flux_curve_inference,
    )
"""

from FluxCurve.config import (
    ForwardRecoveryConfig,
    RobinFluxCurveInferenceRequest,
    _ParallelPointConfig,
)
from FluxCurve.results import (
    RobinFluxCurveInferenceResult,
    PointAdjointResult,
    CurveAdjointResult,
    _ReplayPointFunctional,
    _ReplayBundle,
    _point_result_to_payload,
    _point_result_from_payload,
)
from FluxCurve.recovery import (
    clip_kappa,
    _attempt_phase_state,
    _relax_solver_options_for_attempt,
    _reduce_kappa_anisotropy,
)
from FluxCurve.observables import (
    _build_species_boundary_flux_forms,
    _build_observable_form,
    _build_scalar_target_in_control_space,
    _gradient_controls_to_array,
)
from FluxCurve.point_solve import (
    solve_point_objective_and_gradient,
    _PointSolveExecutor,
    _parallel_worker_init,
    _parallel_worker_solve_point,
)
from FluxCurve.curve_eval import (
    evaluate_curve_objective_and_gradient,
    evaluate_curve_loss_forward,
)
from FluxCurve.replay import (
    _build_replay_point_flux_functional,
    _build_replay_bundle,
    _evaluate_curve_with_replay_bundle,
    _DynamicReplayCurveEvaluator,
)
from FluxCurve.plot import (
    _LiveFitPlot,
    _as_int,
    _as_float,
    export_live_fit_gif,
)
from FluxCurve.run import (
    _normalize_kappa,
    ensure_target_curve,
    write_history_csv,
    write_point_gradient_csv,
    run_scipy_adjoint_optimization,
    run_robin_kappa_flux_curve_inference,
)

__all__ = [
    # config
    "ForwardRecoveryConfig",
    "RobinFluxCurveInferenceRequest",
    "_ParallelPointConfig",
    # results
    "RobinFluxCurveInferenceResult",
    "PointAdjointResult",
    "CurveAdjointResult",
    "_ReplayPointFunctional",
    "_ReplayBundle",
    "_point_result_to_payload",
    "_point_result_from_payload",
    # recovery
    "clip_kappa",
    "_attempt_phase_state",
    "_relax_solver_options_for_attempt",
    "_reduce_kappa_anisotropy",
    # observables
    "_build_species_boundary_flux_forms",
    "_build_observable_form",
    "_build_scalar_target_in_control_space",
    "_gradient_controls_to_array",
    # point_solve
    "solve_point_objective_and_gradient",
    "_PointSolveExecutor",
    "_parallel_worker_init",
    "_parallel_worker_solve_point",
    # curve_eval
    "evaluate_curve_objective_and_gradient",
    "evaluate_curve_loss_forward",
    # replay
    "_build_replay_point_flux_functional",
    "_build_replay_bundle",
    "_evaluate_curve_with_replay_bundle",
    "_DynamicReplayCurveEvaluator",
    # plot
    "_LiveFitPlot",
    "_as_int",
    "_as_float",
    "export_live_fit_gif",
    # run
    "_normalize_kappa",
    "ensure_target_curve",
    "write_history_csv",
    "write_point_gradient_csv",
    "run_scipy_adjoint_optimization",
    "run_robin_kappa_flux_curve_inference",
]
