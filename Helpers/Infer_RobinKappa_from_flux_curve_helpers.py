"""Backward-compatible re-export shim.

The 2597-line contents of this module have been split into the ``FluxCurve``
package.  New code should import from ``FluxCurve`` directly.
"""
# ruff: noqa: F401, F403
from FluxCurve import *
from FluxCurve.config import ForwardRecoveryConfig, RobinFluxCurveInferenceRequest, _ParallelPointConfig
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
from FluxCurve.plot import _LiveFitPlot, _as_int, _as_float, export_live_fit_gif
from FluxCurve.run import (
    _normalize_kappa,
    ensure_target_curve,
    write_history_csv,
    write_point_gradient_csv,
    run_scipy_adjoint_optimization,
    run_robin_kappa_flux_curve_inference,
)
