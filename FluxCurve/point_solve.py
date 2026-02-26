"""Per-point adjoint solve and parallel executor for phi_applied sweep."""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from FluxCurve.config import ForwardRecoveryConfig, _ParallelPointConfig, RobinFluxCurveInferenceRequest
from FluxCurve.results import PointAdjointResult, _point_result_to_payload, _point_result_from_payload
from FluxCurve.recovery import _attempt_phase_state, _relax_solver_options_for_attempt
from FluxCurve.observables import (
    _build_observable_form,
    _build_scalar_target_in_control_space,
    _gradient_controls_to_array,
)
from Forward.steady_state import SteadyStateConfig, configure_robin_solver_params
from Forward.robin_solver import build_context, build_forms, set_initial_conditions


_PARALLEL_POINT_CONFIG: Optional[_ParallelPointConfig] = None


def _parallel_worker_init(config: _ParallelPointConfig) -> None:
    """Initialize one worker process with static point-solve configuration."""
    global _PARALLEL_POINT_CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _PARALLEL_POINT_CONFIG = config


def _parallel_worker_solve_point(
    task: Tuple[int, float, float, Sequence[float]],
) -> Tuple[int, Dict[str, object]]:
    """Worker entrypoint: solve one point and return serializable payload."""
    global _PARALLEL_POINT_CONFIG
    if _PARALLEL_POINT_CONFIG is None:
        raise RuntimeError("Parallel worker config is not initialized.")

    idx, phi_applied, target_flux, kappa_values = task
    cfg = _PARALLEL_POINT_CONFIG
    point = solve_point_objective_and_gradient(
        base_solver_params=cfg.base_solver_params,
        steady=cfg.steady,
        phi_applied=float(phi_applied),
        target_flux=float(target_flux),
        kappa_values=[float(v) for v in kappa_values],
        blob_initial_condition=bool(cfg.blob_initial_condition),
        fail_penalty=float(cfg.fail_penalty),
        forward_recovery=cfg.forward_recovery,
        observable_mode=str(cfg.observable_mode),
        observable_species_index=cfg.observable_species_index,
        observable_scale=float(cfg.observable_scale),
    )
    return int(idx), _point_result_to_payload(point)


class _PointSolveExecutor:
    """Optional process-based executor for parallel phi_applied point solves."""

    def __init__(
        self,
        *,
        request: RobinFluxCurveInferenceRequest,
        n_points: int,
    ) -> None:
        self.enabled = False
        self.max_workers = 1
        self._executor: Optional[ProcessPoolExecutor] = None

        if not bool(request.parallel_point_solves_enabled):
            return
        n_points_i = int(max(0, n_points))
        if n_points_i < int(max(1, request.parallel_point_min_points)):
            return

        requested_workers = int(max(1, request.parallel_point_workers))
        workers = min(requested_workers, n_points_i)
        if workers <= 1:
            return

        method = str(request.parallel_start_method or "spawn").strip().lower()
        if method not in ("spawn", "forkserver"):
            method = "spawn"

        config = _ParallelPointConfig(
            base_solver_params=copy.deepcopy(request.base_solver_params),
            steady=copy.deepcopy(request.steady),
            blob_initial_condition=bool(request.blob_initial_condition),
            fail_penalty=float(request.fail_penalty),
            forward_recovery=copy.deepcopy(request.forward_recovery),
            observable_mode=str(request.observable_mode),
            observable_species_index=request.observable_species_index,
            observable_scale=float(request.observable_scale),
        )
        try:
            ctx = mp.get_context(method)
            self._executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_parallel_worker_init,
                initargs=(config,),
            )
            self.enabled = True
            self.max_workers = int(workers)
            print(
                "[parallel] enabled point-solve workers: "
                f"workers={self.max_workers} start_method={method}"
            )
        except Exception as exc:
            self.enabled = False
            self.max_workers = 1
            self._executor = None
            print(
                "[parallel] worker pool initialization failed; "
                f"using serial point solves ({type(exc).__name__}: {exc})"
            )

    def close(self) -> None:
        """Shutdown worker pool if active."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self.enabled = False

    def map_points(
        self,
        *,
        phi_applied_values: np.ndarray,
        target_flux: np.ndarray,
        kappa_values: np.ndarray,
    ) -> Optional[List[PointAdjointResult]]:
        """Solve all points in parallel. Returns None when executor unavailable."""
        if not self.enabled or self._executor is None:
            return None

        kappa_list = np.asarray(kappa_values, dtype=float).tolist()
        tasks: List[Tuple[int, float, float, Sequence[float]]] = []
        for i, (phi_i, target_i) in enumerate(
            zip(phi_applied_values.tolist(), target_flux.tolist())
        ):
            tasks.append((int(i), float(phi_i), float(target_i), kappa_list))

        results: List[Optional[PointAdjointResult]] = [None] * len(tasks)
        try:
            future_map = {
                self._executor.submit(_parallel_worker_solve_point, task): int(task[0])
                for task in tasks
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                payload_idx, payload = future.result()
                if int(payload_idx) != int(idx):
                    raise RuntimeError(
                        "Parallel worker returned mismatched point index "
                        f"(expected {idx}, got {payload_idx})."
                    )
                results[idx] = _point_result_from_payload(payload)
        except Exception as exc:
            print(
                "[parallel] worker execution failed; "
                f"falling back to serial point solves ({type(exc).__name__}: {exc})"
            )
            return None

        out: List[PointAdjointResult] = []
        for idx, point in enumerate(results):
            if point is None:
                raise RuntimeError(f"Missing parallel result for point index {idx}.")
            out.append(point)
        return out


def solve_point_objective_and_gradient(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied: float,
    target_flux: float,
    kappa_values: Sequence[float],
    blob_initial_condition: bool,
    fail_penalty: float,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_species_index: Optional[int],
    observable_scale: float,
) -> PointAdjointResult:
    """Solve one phi_applied point and extract dJ_i/dkappa with firedrake-adjoint."""
    import firedrake as fd
    import firedrake.adjoint as adj

    kappa_list = [float(v) for v in kappa_values]
    baseline_params = configure_robin_solver_params(
        base_solver_params,
        phi_applied=float(phi_applied),
        kappa_values=kappa_list,
    )
    n_species = int(baseline_params[0])
    abs_tol = float(max(steady.absolute_tolerance, 1e-16))
    rel_tol = float(steady.relative_tolerance)
    max_steps = int(max(1, steady.max_steps))
    required_steady = int(max(1, steady.consecutive_steps))

    baseline_options: Mapping[str, Any] = {}
    if isinstance(baseline_params[10], dict):
        baseline_options = copy.deepcopy(baseline_params[10])

    last_reason = "forward solve did not converge"
    last_flux = float("nan")
    last_steps = 0

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

        tape = adj.get_working_tape()
        tape.clear_tape()
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
        )
        prev_flux: Optional[float] = None
        steady_count = 0
        rel_metric: Optional[float] = None
        abs_metric: Optional[float] = None
        simulated_flux = float("nan")
        steps_taken = 0
        failed_by_exception = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            try:
                solver.solve()
            except Exception as exc:
                failed_by_exception = True
                last_reason = f"{type(exc).__name__}: {exc}"
                break

            U_prev.assign(U)

            # Non-annotated assembly for convergence check itself.
            with adj.stop_annotating():
                simulated_flux = float(fd.assemble(observable_form))

            if prev_flux is not None:
                delta = abs(simulated_flux - prev_flux)
                scale = max(abs(simulated_flux), abs(prev_flux), abs_tol)
                rel_metric = delta / scale
                abs_metric = delta
                is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                steady_count = steady_count + 1 if is_steady else 0
            else:
                steady_count = 0

            prev_flux = simulated_flux
            if steady_count >= required_steady:
                # Point objective in annotated mode: 0.5*(flux-target)^2.
                target_flux_control = _build_scalar_target_in_control_space(
                    ctx, target_flux, name="target_flux_value"
                )
                target_flux_scalar = fd.assemble(
                    target_flux_control * fd.dx(domain=ctx["mesh"])
                )
                simulated_flux_scalar = fd.assemble(observable_form)
                point_objective = 0.5 * (simulated_flux_scalar - target_flux_scalar) ** 2

                controls = [adj.Control(ctrl) for ctrl in list(ctx["kappa_funcs"])]
                rf = adj.ReducedFunctional(point_objective, controls)
                control_state = [ctrl for ctrl in list(ctx["kappa_funcs"])]
                point_objective_value = float(rf(control_state))
                point_gradient = _gradient_controls_to_array(rf.derivative(), n_species)

                return PointAdjointResult(
                    phi_applied=float(phi_applied),
                    target_flux=float(target_flux),
                    simulated_flux=float(simulated_flux_scalar),
                    objective=point_objective_value,
                    gradient=point_gradient,
                    converged=True,
                    steps_taken=int(steps_taken),
                    reason="",
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    diagnostics_valid=True,
                )

        last_flux = simulated_flux
        last_steps = int(steps_taken)
        if not failed_by_exception:
            last_reason = "steady-state criterion not satisfied before max_steps"

    return PointAdjointResult(
        phi_applied=float(phi_applied),
        target_flux=float(target_flux),
        simulated_flux=last_flux,
        objective=float(fail_penalty),
        gradient=np.zeros(n_species, dtype=float),
        converged=False,
        steps_taken=int(last_steps),
        reason=last_reason,
        final_relative_change=None,
        final_absolute_change=None,
        diagnostics_valid=False,
    )
