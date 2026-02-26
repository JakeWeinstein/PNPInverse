"""Top-level optimization loop and I/O for Robin-kappa flux-curve inference."""

from __future__ import annotations

import copy
import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from FluxCurve.config import RobinFluxCurveInferenceRequest
from FluxCurve.results import RobinFluxCurveInferenceResult, PointAdjointResult
from FluxCurve.recovery import clip_kappa
from FluxCurve.curve_eval import evaluate_curve_objective_and_gradient, evaluate_curve_loss_forward
from FluxCurve.replay import _DynamicReplayCurveEvaluator
from FluxCurve.point_solve import _PointSolveExecutor
from FluxCurve.plot import _LiveFitPlot, _as_int, _as_float, export_live_fit_gif
from Forward.steady_state import (
    SteadyStateConfig,
    configure_robin_solver_params,
    read_phi_applied_flux_csv,
    sweep_phi_applied_steady_flux,
    all_results_converged,
    results_to_flux_array,
    add_percent_noise,
    write_phi_applied_flux_csv,
)


def _normalize_kappa(value: Optional[Sequence[float]], *, name: str) -> Optional[List[float]]:
    """Convert 2-species kappa-like input into a validated float list."""
    if value is None:
        return None
    vals = [float(v) for v in list(value)]
    if len(vals) != 2:
        raise ValueError(f"{name} must have length 2; got {len(vals)}.")
    if any(v <= 0.0 for v in vals):
        raise ValueError(f"{name} must be strictly positive.")
    return vals


def ensure_target_curve(
    *,
    target_csv_path: str,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    true_kappa: Optional[Sequence[float]],
    noise_percent: float,
    seed: int,
    force_regenerate: bool,
    blob_initial_condition: bool,
) -> Dict[str, np.ndarray]:
    """Load target data from CSV or generate synthetic target if missing."""
    if os.path.exists(target_csv_path) and not force_regenerate:
        print(f"Loading target curve from: {target_csv_path}")
        return read_phi_applied_flux_csv(target_csv_path, flux_column="flux_noisy")

    kappa_true = _normalize_kappa(true_kappa, name="true_value")
    if kappa_true is None:
        kappa_true = [0.8, 0.8]

    if os.path.exists(target_csv_path):
        print(
            "Regenerating target curve with requested kappa_true; "
            f"overwriting existing CSV: {target_csv_path}"
        )
    else:
        print("Target CSV not found; generating synthetic target data first.")
    print(f"Synthetic target settings: kappa_true={kappa_true}, noise_percent={noise_percent}")

    target_results = sweep_phi_applied_steady_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        kappa_values=kappa_true,
        blob_initial_condition=bool(blob_initial_condition),
    )
    if not all_results_converged(target_results):
        failed = [f"{r.phi_applied:.6f}" for r in target_results if not r.converged]
        raise RuntimeError(
            "Synthetic target generation failed for some phi_applied values. "
            f"Failed values: {failed}"
        )

    target_flux_clean = results_to_flux_array(target_results)
    target_flux_noisy = add_percent_noise(target_flux_clean, noise_percent, seed=seed)

    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
    write_phi_applied_flux_csv(target_csv_path, target_results, noisy_flux=target_flux_noisy)
    print(f"Synthetic target saved to: {target_csv_path}")

    return {"phi_applied": phi_applied_values.copy(), "flux": target_flux_noisy}


def _build_scipy_options(request: RobinFluxCurveInferenceRequest) -> Dict[str, Any]:
    """Construct SciPy minimize options with sensible defaults."""
    options: Dict[str, Any] = {}
    if request.optimizer_options is not None:
        options.update(dict(request.optimizer_options))
    options.setdefault("maxiter", int(request.max_iters))
    options.setdefault("disp", True)
    if "gtol" not in options and request.optimizer_method in (
        "L-BFGS-B",
        "BFGS",
        "CG",
        "Newton-CG",
        "TNC",
    ):
        options["gtol"] = float(request.gtol)
    return options


def write_history_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write optimizer-iteration history to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = [
        "evaluation",
        "iteration",
        "kappa0",
        "kappa1",
        "objective",
        "grad0",
        "grad1",
        "grad_norm",
        "n_failed_points",
        "used_anisotropy_recovery",
        "used_replay_mode",
        "effective_kappa0",
        "effective_kappa1",
        "from_cache",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_point_gradient_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write per-point adjoint gradient diagnostics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = [
        "evaluation",
        "iteration",
        "point_index",
        "phi_applied",
        "target_observable",
        "simulated_observable",
        "target_flux",
        "simulated_flux",
        "point_objective",
        "dJ_dkappa0",
        "dJ_dkappa1",
        "converged",
        "steps_taken",
        "final_relative_change",
        "final_absolute_change",
        "diagnostics_valid",
        "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_scipy_adjoint_optimization(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    initial_kappa: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Any,
    _LiveFitPlot,
    Dict[str, int],
]:
    """Optimize kappa using SciPy minimize with analytic adjoint Jacobian."""
    from scipy.optimize import minimize

    n_species = int(initial_kappa.size)
    x0 = clip_kappa(np.asarray(initial_kappa, dtype=float), lower_bounds, upper_bounds)
    bounds = [(float(lower_bounds[i]), float(upper_bounds[i])) for i in range(n_species)]
    options = _build_scipy_options(request)

    history_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []

    # Cache expensive PDE evaluations so separate fun/jac calls at same x reuse work.
    cache: Dict[Tuple[float, ...], Dict[str, object]] = {}
    eval_counter = {"n": 0}
    iteration_counter = {"n": 0}

    best_kappa = x0.copy()
    best_loss = float("inf")
    best_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)
    point_executor = _PointSolveExecutor(
        request=request, n_points=int(phi_applied_values.size)
    )
    curve_evaluator = _DynamicReplayCurveEvaluator(
        request=request,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        point_executor=point_executor,
    )
    curve_evaluator.initialize(kappa_anchor=x0.copy())
    live_plot = _LiveFitPlot(
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        y_label=str(request.observable_label),
        title=str(request.observable_title),
        enabled=bool(request.live_plot),
        pause_seconds=float(request.live_plot_pause_seconds),
        show_eval_lines=bool(request.live_plot_eval_lines),
        eval_line_alpha=float(request.live_plot_eval_line_alpha),
        eval_max_lines=int(request.live_plot_eval_max_lines),
        capture_frames_dir=request.live_plot_capture_frames_dir,
        capture_every_n_updates=int(request.live_plot_capture_every_n_updates),
        capture_max_frames=int(request.live_plot_capture_max_frames),
    )

    def _key_from_x(x: np.ndarray) -> Tuple[float, ...]:
        x_clip = clip_kappa(np.asarray(x, dtype=float), lower_bounds, upper_bounds)
        return tuple(float(f"{v:.12g}") for v in x_clip.tolist())

    def _record_point_rows(
        *,
        eval_id: int,
        iter_id: int,
        points: Sequence[PointAdjointResult],
    ) -> None:
        for idx, point in enumerate(points):
            point_rows.append(
                {
                    "evaluation": eval_id,
                    "iteration": iter_id,
                    "point_index": idx,
                    "phi_applied": point.phi_applied,
                    "target_observable": point.target_flux,
                    "simulated_observable": point.simulated_flux,
                    "target_flux": point.target_flux,
                    "simulated_flux": point.simulated_flux,
                    "point_objective": point.objective,
                    "dJ_dkappa0": float(point.gradient[0]) if n_species >= 1 else float("nan"),
                    "dJ_dkappa1": float(point.gradient[1]) if n_species >= 2 else float("nan"),
                    "converged": int(bool(point.converged)),
                    "steps_taken": int(point.steps_taken),
                    "final_relative_change": (
                        float(point.final_relative_change)
                        if point.final_relative_change is not None
                        else float("nan")
                    ),
                    "final_absolute_change": (
                        float(point.final_absolute_change)
                        if point.final_absolute_change is not None
                        else float("nan")
                    ),
                    "diagnostics_valid": int(bool(point.diagnostics_valid)),
                    "reason": point.reason,
                }
            )
            if bool(request.print_point_gradients):
                print(
                    f"  phi={point.phi_applied:8.5f} "
                    f"target={point.target_flux:12.6f} "
                    f"sim={point.simulated_flux:12.6f} "
                    f"dJ/dk=[{point.gradient[0]:11.6f}, {point.gradient[1]:11.6f}] "
                    f"conv={int(point.converged)} "
                    f"diag={int(bool(point.diagnostics_valid))} "
                    f"rel={(float(point.final_relative_change) if point.final_relative_change is not None else float('nan')):9.3e} "
                    f"abs={(float(point.final_absolute_change) if point.final_absolute_change is not None else float('nan')):9.3e}"
                )

    def _evaluate(x: np.ndarray) -> Dict[str, object]:
        nonlocal best_kappa, best_loss, best_flux

        x_clip = clip_kappa(np.asarray(x, dtype=float), lower_bounds, upper_bounds)
        key = _key_from_x(x_clip)
        if key in cache:
            return cache[key]

        curve = curve_evaluator.evaluate(kappa_values=x_clip)

        eval_counter["n"] += 1
        eval_id = int(eval_counter["n"])
        iter_id = int(iteration_counter["n"])
        objective = float(curve.objective)
        gradient = np.asarray(curve.gradient, dtype=float)
        grad_norm = float(np.linalg.norm(gradient))
        n_failed = int(curve.n_failed)
        simulated_flux = np.asarray(curve.simulated_flux, dtype=float)

        recovery_tag = " aniso_recovery=1" if curve.used_anisotropy_recovery else ""
        replay_tag = " replay=1" if curve.used_replay_mode else ""
        print(
            f"[eval={eval_id:03d}] "
            f"kappa=[{x_clip[0]:10.6f}, {x_clip[1]:10.6f}] "
            f"loss={objective:14.6e} "
            f"grad=[{gradient[0]:12.6f}, {gradient[1]:12.6f}] "
            f"|grad|={grad_norm:12.6f} "
            f"fails={n_failed:02d}"
            f"{recovery_tag}"
            f"{replay_tag}"
        )
        if curve.used_anisotropy_recovery:
            ek = curve.effective_kappa
            print(
                f"  effective_kappa=[{float(ek[0]):10.6f}, {float(ek[1]):10.6f}]"
            )

        _record_point_rows(eval_id=eval_id, iter_id=iter_id, points=curve.points)

        live_plot.add_eval_curve(flux=simulated_flux, eval_id=eval_id)
        live_plot.update(
            current_flux=simulated_flux,
            best_flux=best_flux.copy(),
            iteration=int(iter_id),
            objective=objective,
            n_failed=n_failed,
            kappa=x_clip.copy(),
            eval_id=eval_id,
        )

        history_rows.append(
            {
                "evaluation": eval_id,
                "iteration": iter_id,
                "kappa0": float(x_clip[0]),
                "kappa1": float(x_clip[1]),
                "objective": objective,
                "grad0": float(gradient[0]),
                "grad1": float(gradient[1]),
                "grad_norm": grad_norm,
                "n_failed_points": n_failed,
                "used_anisotropy_recovery": int(bool(curve.used_anisotropy_recovery)),
                "used_replay_mode": int(bool(curve.used_replay_mode)),
                "effective_kappa0": float(curve.effective_kappa[0]),
                "effective_kappa1": float(curve.effective_kappa[1]),
                "from_cache": 0,
            }
        )

        if n_failed == 0 and np.isfinite(objective) and objective < best_loss:
            best_loss = objective
            best_kappa = x_clip.copy()
            best_flux = simulated_flux.copy()

        payload = {
            "x": x_clip.copy(),
            "objective": objective,
            "gradient": gradient.copy(),
            "simulated_flux": simulated_flux.copy(),
            "n_failed": n_failed,
            "used_anisotropy_recovery": bool(curve.used_anisotropy_recovery),
            "used_replay_mode": bool(curve.used_replay_mode),
            "effective_kappa": np.asarray(curve.effective_kappa, dtype=float).copy(),
            "points": list(curve.points),
            "eval_id": eval_id,
        }
        cache[key] = payload
        return payload

    def _fun(x: np.ndarray) -> float:
        return float(_evaluate(x)["objective"])

    def _jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(_evaluate(x)["gradient"], dtype=float)

    def _callback(xk: np.ndarray) -> None:
        iteration_counter["n"] += 1
        payload = _evaluate(xk)
        grad = np.asarray(payload["gradient"], dtype=float)
        recovery_tag = " aniso_recovery=1" if bool(payload.get("used_anisotropy_recovery", False)) else ""
        replay_tag = " replay=1" if bool(payload.get("used_replay_mode", False)) else ""
        print(
            f"[iter={iteration_counter['n']:02d}] "
            f"kappa=[{payload['x'][0]:10.6f}, {payload['x'][1]:10.6f}] "
            f"loss={float(payload['objective']):14.6e} "
            f"|grad|={float(np.linalg.norm(grad)):12.6f} "
            f"fails={int(payload['n_failed']):02d}"
            f"{recovery_tag}"
            f"{replay_tag}"
        )
        live_plot.update(
            current_flux=np.asarray(payload["simulated_flux"], dtype=float),
            best_flux=best_flux.copy(),
            iteration=int(iteration_counter["n"]),
            objective=float(payload["objective"]),
            n_failed=int(payload["n_failed"]),
            kappa=np.asarray(payload["x"], dtype=float),
            eval_id=int(payload.get("eval_id", -1)),
        )

    # Draw an initial curve before the first optimizer step.
    initial_payload = _evaluate(x0)
    live_plot.update(
        current_flux=np.asarray(initial_payload["simulated_flux"], dtype=float),
        best_flux=best_flux.copy(),
        iteration=0,
        objective=float(initial_payload["objective"]),
        n_failed=int(initial_payload["n_failed"]),
        kappa=np.asarray(initial_payload["x"], dtype=float),
        eval_id=int(initial_payload.get("eval_id", -1)),
    )

    try:
        result = minimize(
            _fun,
            x0=x0,
            jac=_jac,
            method=str(request.optimizer_method),
            bounds=bounds,
            tol=request.optimizer_tolerance,
            callback=_callback,
            options=options,
        )

        # Ensure final point is evaluated and available in outputs.
        final_x = clip_kappa(np.asarray(result.x, dtype=float), lower_bounds, upper_bounds)
        final_payload = _evaluate(final_x)

        if (
            int(final_payload["n_failed"]) == 0
            and np.isfinite(float(final_payload["objective"]))
            and float(final_payload["objective"]) < best_loss
        ):
            best_loss = float(final_payload["objective"])
            best_kappa = np.asarray(final_payload["x"], dtype=float).copy()
            best_flux = np.asarray(final_payload["simulated_flux"], dtype=float).copy()

        # If no fully converged point was seen, fallback to final-forward loss.
        if not np.isfinite(best_loss):
            best_loss, best_flux, _ = evaluate_curve_loss_forward(
                base_solver_params=request.base_solver_params,
                steady=request.steady,
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                kappa_values=best_kappa,
                blob_initial_condition=bool(request.blob_initial_condition),
                fail_penalty=float(request.fail_penalty),
                observable_scale=float(request.observable_scale),
            )

        # Ensure the final state is drawn even if callback was not called on last point.
        live_plot.update(
            current_flux=np.asarray(final_payload["simulated_flux"], dtype=float),
            best_flux=best_flux.copy(),
            iteration=int(iteration_counter["n"]),
            objective=float(final_payload["objective"]),
            n_failed=int(final_payload["n_failed"]),
            kappa=np.asarray(final_payload["x"], dtype=float),
            eval_id=int(final_payload.get("eval_id", -1)),
        )
    finally:
        point_executor.close()

    replay_stats = curve_evaluator.stats()
    print(
        "[replay] summary: "
        f"rebuilds={int(replay_stats.get('replay_rebuild_count', 0))} "
        f"diag_rebuilds={int(replay_stats.get('replay_diag_rebuild_count', 0))} "
        f"exception_rebuilds={int(replay_stats.get('replay_exception_rebuild_count', 0))}"
    )
    return (
        best_kappa,
        float(best_loss),
        best_flux,
        history_rows,
        point_rows,
        result,
        live_plot,
        replay_stats,
    )


def run_robin_kappa_flux_curve_inference(
    request: RobinFluxCurveInferenceRequest,
) -> RobinFluxCurveInferenceResult:
    """Run end-to-end Robin-kappa curve inference with adjoint gradients."""
    request_runtime = copy.deepcopy(request)
    if bool(request_runtime.replay_mode_enabled):
        print(
            "[replay] requested but currently disabled globally; "
            "running with replay_mode_enabled=False."
        )
    request_runtime.replay_mode_enabled = False
    request_runtime.steady.flux_observable = str(request_runtime.observable_mode)
    request_runtime.steady.species_index = request_runtime.observable_species_index

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)
    target_data = ensure_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_kappa=request_runtime.true_value,
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = float(request_runtime.observable_scale) * np.asarray(
        target_data["flux"], dtype=float
    )
    if target_phi_applied.size != target_flux.size:
        raise ValueError("Target phi_applied and flux lengths do not match.")
    phi_applied_values = target_phi_applied.copy()

    initial_kappa_list = _normalize_kappa(request_runtime.initial_guess, name="initial_guess")
    if initial_kappa_list is None:
        raise ValueError("initial_guess must be set.")
    initial_kappa = np.asarray(initial_kappa_list, dtype=float)

    n_species = int(request_runtime.base_solver_params[0])
    lower = np.full(n_species, float(request_runtime.kappa_lower), dtype=float)
    upper = np.full(n_species, float(request_runtime.kappa_upper), dtype=float)

    print("=== Adjoint-Gradient Robin Kappa Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Initial kappa: {initial_kappa.tolist()}")
    print(f"Bounds: lower={lower.tolist()} upper={upper.tolist()}")
    print(
        "Objective: 0.5 * sum_i (observable(phi_i, kappa) - target_i)^2, "
        f"mode={request_runtime.observable_mode}, "
        f"scale={float(request_runtime.observable_scale):.16g}, "
        "with dJ/dkappa from firedrake-adjoint per point."
    )

    (
        best_kappa,
        best_loss,
        best_sim_flux,
        history_rows,
        point_rows,
        opt_result,
        live_plot,
        replay_stats,
    ) = run_scipy_adjoint_optimization(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_kappa=initial_kappa,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    print(
        "SciPy minimize summary: "
        f"success={bool(getattr(opt_result, 'success', False))} "
        f"status={getattr(opt_result, 'status', 'n/a')} "
        f"message={str(getattr(opt_result, 'message', '')).strip()}"
    )

    if bool(request_runtime.replay_mode_enabled) and bool(request_runtime.replay_post_refine_enabled):
        refine_iters = int(max(1, request_runtime.replay_post_refine_max_iters))
        print(
            "[refine] starting replay-off refinement/verification phase "
            f"from kappa=[{float(best_kappa[0]):.6f}, {float(best_kappa[1]):.6f}] "
            f"for up to {refine_iters} iterations."
        )
        refine_request = copy.deepcopy(request_runtime)
        refine_request.replay_mode_enabled = False
        refine_request.live_plot = False
        refine_request.live_plot_eval_lines = False
        refine_request.live_plot_capture_frames_dir = None
        refine_request.live_plot_export_gif_path = None
        refine_request.max_iters = refine_iters
        refine_opts = dict(refine_request.optimizer_options or {})
        refine_opts["maxiter"] = refine_iters
        refine_request.optimizer_options = refine_opts

        (
            refine_best_kappa,
            refine_best_loss,
            refine_best_sim_flux,
            refine_history_rows,
            refine_point_rows,
            refine_opt_result,
            _,
            refine_replay_stats,
        ) = run_scipy_adjoint_optimization(
            request=refine_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_kappa=np.asarray(best_kappa, dtype=float).copy(),
            lower_bounds=lower,
            upper_bounds=upper,
        )
        replay_stats = {
            "replay_rebuild_count": int(replay_stats.get("replay_rebuild_count", 0))
            + int(refine_replay_stats.get("replay_rebuild_count", 0)),
            "replay_diag_rebuild_count": int(
                replay_stats.get("replay_diag_rebuild_count", 0)
            )
            + int(refine_replay_stats.get("replay_diag_rebuild_count", 0)),
            "replay_exception_rebuild_count": int(
                replay_stats.get("replay_exception_rebuild_count", 0)
            )
            + int(refine_replay_stats.get("replay_exception_rebuild_count", 0)),
        }
        print(
            "[refine] SciPy summary: "
            f"success={bool(getattr(refine_opt_result, 'success', False))} "
            f"status={getattr(refine_opt_result, 'status', 'n/a')} "
            f"message={str(getattr(refine_opt_result, 'message', '')).strip()}"
        )

        eval_offset = max((_as_int(r.get("evaluation"), 0) for r in history_rows), default=0)
        iter_offset = max((_as_int(r.get("iteration"), 0) for r in history_rows), default=0)
        for row in refine_history_rows:
            row_out = dict(row)
            row_out["evaluation"] = _as_int(row_out.get("evaluation"), 0) + eval_offset
            row_out["iteration"] = _as_int(row_out.get("iteration"), 0) + iter_offset
            history_rows.append(row_out)
        for row in refine_point_rows:
            row_out = dict(row)
            row_out["evaluation"] = _as_int(row_out.get("evaluation"), 0) + eval_offset
            row_out["iteration"] = _as_int(row_out.get("iteration"), 0) + iter_offset
            point_rows.append(row_out)

        take_refined = False
        if np.isfinite(refine_best_loss) and (
            (not np.isfinite(best_loss)) or (float(refine_best_loss) <= float(best_loss))
        ):
            take_refined = True
        elif bool(getattr(refine_opt_result, "success", False)):
            # If both are finite but replay-off succeeds, prefer the refined endpoint
            # as the final reported full-objective solution.
            take_refined = True

        if take_refined:
            best_kappa = np.asarray(refine_best_kappa, dtype=float).copy()
            best_loss = float(refine_best_loss)
            best_sim_flux = np.asarray(refine_best_sim_flux, dtype=float).copy()
            opt_result = refine_opt_result
            print("[refine] accepted replay-off refined solution.")
        else:
            print("[refine] retained replay-phase solution (refinement not better).")

    final_loss, final_sim_flux, forward_failures_at_best = evaluate_curve_loss_forward(
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        kappa_values=best_kappa,
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        fail_penalty=float(request_runtime.fail_penalty),
        observable_scale=float(request_runtime.observable_scale),
    )
    if np.isfinite(final_loss) and forward_failures_at_best == 0:
        best_loss = float(final_loss)
        best_sim_flux = final_sim_flux

    os.makedirs(request_runtime.output_dir, exist_ok=True)
    fit_csv_path = os.path.join(
        request_runtime.output_dir, "phi_applied_vs_steady_observable_fit.csv"
    )
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_observable,simulated_observable\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(
        request_runtime.output_dir, "robin_kappa_gradient_optimization_history.csv"
    )
    write_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(request_runtime.output_dir, "robin_kappa_point_gradients.csv")
    write_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    replay_diag_csv_path = os.path.join(
        request_runtime.output_dir, "replay_diagnostics_summary.csv"
    )
    with open(replay_diag_csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(
            f"replay_rebuild_count,{int(replay_stats.get('replay_rebuild_count', 0))}\n"
        )
        f.write(
            "replay_diag_rebuild_count,"
            f"{int(replay_stats.get('replay_diag_rebuild_count', 0))}\n"
        )
        f.write(
            "replay_exception_rebuild_count,"
            f"{int(replay_stats.get('replay_exception_rebuild_count', 0))}\n"
        )
    print(f"Saved replay diagnostics summary CSV: {replay_diag_csv_path}")

    fit_plot_path: Optional[str] = None
    if plt is None:
        print("matplotlib not available; skipping fit plot generation.")
    elif bool(request_runtime.live_plot) and getattr(live_plot, "enabled", False):
        fit_plot_path = os.path.join(
            request_runtime.output_dir, "phi_applied_vs_steady_observable_fit.png"
        )
        live_plot.update(
            current_flux=best_sim_flux.copy(),
            best_flux=best_sim_flux.copy(),
            iteration=-1,
            objective=float(best_loss),
            n_failed=int(forward_failures_at_best),
            kappa=best_kappa.copy(),
        )
        live_plot.save(fit_plot_path)
        print(f"Saved fit plot: {fit_plot_path}")
    else:
        fit_plot_path = os.path.join(
            request_runtime.output_dir, "phi_applied_vs_steady_observable_fit.png"
        )
        plt.figure(figsize=(7, 4))
        plt.plot(
            phi_applied_values,
            target_flux,
            marker="o",
            linewidth=2,
            label="target observable",
        )
        plt.plot(
            phi_applied_values,
            best_sim_flux,
            marker="s",
            label="best-fit simulated observable",
            linewidth=2,
        )
        plt.xlabel("applied voltage phi_applied")
        plt.ylabel(str(request_runtime.observable_label))
        plt.title(f"{request_runtime.observable_title} (adjoint-gradient curve fitting)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fit_plot_path, dpi=160)
        plt.close()
        print(f"Saved fit plot: {fit_plot_path}")

    live_gif_path: Optional[str] = None
    if request_runtime.live_plot_export_gif_path:
        live_gif_path = export_live_fit_gif(
            path=str(request_runtime.live_plot_export_gif_path),
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            history_rows=history_rows,
            point_rows=point_rows,
            seconds=float(request_runtime.live_plot_export_gif_seconds),
            n_frames=int(request_runtime.live_plot_export_gif_frames),
            dpi=int(request_runtime.live_plot_export_gif_dpi),
            y_label=str(request_runtime.observable_label),
            title=f"{str(request_runtime.observable_title)} progress",
        )
        if live_gif_path:
            print(f"Saved convergence GIF: {live_gif_path}")
        else:
            print("GIF export requested but could not be generated.")

    print(
        "[replay] run totals: "
        f"rebuilds={int(replay_stats.get('replay_rebuild_count', 0))} "
        f"diag_rebuilds={int(replay_stats.get('replay_diag_rebuild_count', 0))} "
        f"exception_rebuilds={int(replay_stats.get('replay_exception_rebuild_count', 0))}"
    )

    return RobinFluxCurveInferenceResult(
        best_kappa=best_kappa.copy(),
        best_loss=float(best_loss),
        phi_applied_values=phi_applied_values.copy(),
        target_flux=target_flux.copy(),
        best_simulated_flux=best_sim_flux.copy(),
        forward_failures_at_best=int(forward_failures_at_best),
        fit_csv_path=fit_csv_path,
        fit_plot_path=fit_plot_path,
        history_csv_path=history_csv_path,
        point_gradient_csv_path=point_csv_path,
        live_gif_path=live_gif_path,
        optimization_success=bool(getattr(opt_result, "success", False)),
        optimization_message=str(getattr(opt_result, "message", "")),
        replay_rebuild_count=int(replay_stats.get("replay_rebuild_count", 0)),
        replay_diag_rebuild_count=int(replay_stats.get("replay_diag_rebuild_count", 0)),
        replay_exception_rebuild_count=int(
            replay_stats.get("replay_exception_rebuild_count", 0)
        ),
    )
