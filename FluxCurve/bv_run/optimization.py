"""Optimizer wrappers for BV flux-curve inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from FluxCurve.results import PointAdjointResult, CurveAdjointResult
from FluxCurve.recovery import clip_kappa
from FluxCurve.bv_curve_eval import (
    evaluate_bv_curve_objective_and_gradient,
)
from FluxCurve.plot import _LiveFitPlot, _as_int, _as_float, export_live_fit_gif

from .io import write_bv_history_csv, write_bv_point_gradient_csv


def _build_scipy_options(request: BVFluxCurveInferenceRequest) -> Dict[str, Any]:
    """Construct SciPy minimize options with sensible defaults."""
    options: Dict[str, Any] = {}
    if request.optimizer_options is not None:
        options.update(dict(request.optimizer_options))
    options.setdefault("maxiter", int(request.max_iters))
    options.setdefault("disp", True)
    if "gtol" not in options and request.optimizer_method in (
        "L-BFGS-B", "BFGS", "CG", "Newton-CG", "TNC",
    ):
        options["gtol"] = float(request.gtol)
    return options


def run_scipy_bv_adjoint_optimization(
    *,
    request: BVFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    initial_k0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    mesh: Any = None,
    control_mode: str = "k0",
    fixed_k0: Optional[np.ndarray] = None,
    initial_alpha: Optional[np.ndarray] = None,
    alpha_lower_bounds: Optional[np.ndarray] = None,
    alpha_upper_bounds: Optional[np.ndarray] = None,
    initial_steric_a: Optional[np.ndarray] = None,
    steric_a_lower_bounds: Optional[np.ndarray] = None,
    steric_a_upper_bounds: Optional[np.ndarray] = None,
    fixed_k0_for_steric: Optional[np.ndarray] = None,
    fixed_alpha_for_steric: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Any,
    _LiveFitPlot,
]:
    """Optimize BV parameters using SciPy minimize with analytic adjoint Jacobian.

    Supports control_mode "k0" (default), "alpha", "joint", "steric", or "full".
    """
    from scipy.optimize import minimize

    n_species = int(request.base_solver_params.n_species) if hasattr(request.base_solver_params, 'n_species') else int(request.base_solver_params[0])
    n_steric = int(initial_steric_a.size) if initial_steric_a is not None else n_species

    # Determine control vector layout
    if control_mode == "k0":
        n_controls = int(initial_k0.size)
        use_log_space = bool(getattr(request, "log_space", True))
    elif control_mode == "alpha":
        if initial_alpha is None:
            raise ValueError("initial_alpha required for control_mode='alpha'")
        n_controls = int(initial_alpha.size)
        use_log_space = bool(getattr(request, "alpha_log_space", False))
    elif control_mode == "joint":
        if initial_alpha is None:
            raise ValueError("initial_alpha required for control_mode='joint'")
        n_controls = int(initial_k0.size) + int(initial_alpha.size)
        use_log_space = bool(getattr(request, "log_space", True))
    elif control_mode == "steric":
        if initial_steric_a is None:
            raise ValueError("initial_steric_a required for control_mode='steric'")
        n_controls = n_steric
        use_log_space = False
    elif control_mode == "full":
        if initial_alpha is None:
            raise ValueError("initial_alpha required for control_mode='full'")
        if initial_steric_a is None:
            raise ValueError("initial_steric_a required for control_mode='full'")
        n_controls = int(initial_k0.size) + int(initial_alpha.size) + n_steric
        use_log_space = bool(getattr(request, "log_space", True))
    else:
        raise ValueError(f"Unknown control_mode '{control_mode}'")

    n_k0 = int(initial_k0.size)
    n_alpha = int(initial_alpha.size) if initial_alpha is not None else 0

    if control_mode == "k0":
        if use_log_space:
            x0_lin = clip_kappa(np.asarray(initial_k0, dtype=float), lower_bounds, upper_bounds)
            x0 = np.log10(x0_lin)
            bounds = [
                (float(np.log10(max(lower_bounds[i], 1e-30))),
                 float(np.log10(upper_bounds[i])))
                for i in range(n_controls)
            ]
            print(f"[log-space] x0 = {x0.tolist()}, bounds = {bounds}")
        else:
            x0 = clip_kappa(np.asarray(initial_k0, dtype=float), lower_bounds, upper_bounds)
            bounds = [(float(lower_bounds[i]), float(upper_bounds[i])) for i in range(n_controls)]
    elif control_mode == "alpha":
        a_lo = alpha_lower_bounds if alpha_lower_bounds is not None else np.full(n_controls, 0.05)
        a_hi = alpha_upper_bounds if alpha_upper_bounds is not None else np.full(n_controls, 0.95)
        x0 = np.clip(np.asarray(initial_alpha, dtype=float), a_lo, a_hi)
        bounds = [(float(a_lo[i]), float(a_hi[i])) for i in range(n_controls)]
    elif control_mode == "joint":
        # k0 part (log space) + alpha part (linear)
        k0_log = bool(getattr(request, "log_space", True))
        x0_k0 = clip_kappa(np.asarray(initial_k0, dtype=float), lower_bounds, upper_bounds)
        if k0_log:
            x0_k0_x = np.log10(x0_k0)
            k0_bounds = [
                (float(np.log10(max(lower_bounds[i], 1e-30))),
                 float(np.log10(upper_bounds[i])))
                for i in range(n_k0)
            ]
        else:
            x0_k0_x = x0_k0
            k0_bounds = [(float(lower_bounds[i]), float(upper_bounds[i])) for i in range(n_k0)]
        a_lo = alpha_lower_bounds if alpha_lower_bounds is not None else np.full(n_alpha, 0.05)
        a_hi = alpha_upper_bounds if alpha_upper_bounds is not None else np.full(n_alpha, 0.95)
        x0_alpha = np.clip(np.asarray(initial_alpha, dtype=float), a_lo, a_hi)
        alpha_bounds = [(float(a_lo[i]), float(a_hi[i])) for i in range(n_alpha)]
        x0 = np.concatenate([x0_k0_x, x0_alpha])
        bounds = k0_bounds + alpha_bounds
    elif control_mode == "steric":
        s_lo = steric_a_lower_bounds if steric_a_lower_bounds is not None else np.full(n_steric, 0.001)
        s_hi = steric_a_upper_bounds if steric_a_upper_bounds is not None else np.full(n_steric, 0.5)
        x0 = np.clip(np.asarray(initial_steric_a, dtype=float), s_lo, s_hi)
        bounds = [(float(s_lo[i]), float(s_hi[i])) for i in range(n_steric)]
    elif control_mode == "full":
        # k0 (log space) + alpha (linear) + steric_a (linear)
        k0_log = bool(getattr(request, "log_space", True))
        x0_k0 = clip_kappa(np.asarray(initial_k0, dtype=float), lower_bounds, upper_bounds)
        if k0_log:
            x0_k0_x = np.log10(x0_k0)
            k0_bounds = [
                (float(np.log10(max(lower_bounds[i], 1e-30))),
                 float(np.log10(upper_bounds[i])))
                for i in range(n_k0)
            ]
        else:
            x0_k0_x = x0_k0
            k0_bounds = [(float(lower_bounds[i]), float(upper_bounds[i])) for i in range(n_k0)]
        a_lo = alpha_lower_bounds if alpha_lower_bounds is not None else np.full(n_alpha, 0.05)
        a_hi = alpha_upper_bounds if alpha_upper_bounds is not None else np.full(n_alpha, 0.95)
        x0_alpha = np.clip(np.asarray(initial_alpha, dtype=float), a_lo, a_hi)
        alpha_bounds = [(float(a_lo[i]), float(a_hi[i])) for i in range(n_alpha)]
        s_lo = steric_a_lower_bounds if steric_a_lower_bounds is not None else np.full(n_steric, 0.001)
        s_hi = steric_a_upper_bounds if steric_a_upper_bounds is not None else np.full(n_steric, 0.5)
        x0_steric = np.clip(np.asarray(initial_steric_a, dtype=float), s_lo, s_hi)
        steric_bounds = [(float(s_lo[i]), float(s_hi[i])) for i in range(n_steric)]
        x0 = np.concatenate([x0_k0_x, x0_alpha, x0_steric])
        bounds = k0_bounds + alpha_bounds + steric_bounds

    options = _build_scipy_options(request)

    history_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []

    cache: Dict[Tuple[float, ...], Dict[str, object]] = {}
    eval_counter = {"n": 0}
    iteration_counter = {"n": 0}

    # Initialize best_k0 with k0-only values (not the full x vector).
    # For joint/full modes, x0 is a concatenation of [log10(k0), alpha, ...].
    # We must extract only the k0 part.
    if control_mode in ("joint", "full"):
        _init_k0_x = x0[:n_k0]
        k0_log = bool(getattr(request, "log_space", True))
        if k0_log:
            best_k0 = np.power(10.0, _init_k0_x)
        else:
            best_k0 = _init_k0_x.copy()
        best_k0 = clip_kappa(best_k0, lower_bounds, upper_bounds)
    elif control_mode == "k0":
        if use_log_space:
            best_k0 = clip_kappa(np.power(10.0, x0), lower_bounds, upper_bounds)
        else:
            best_k0 = clip_kappa(x0.copy(), lower_bounds, upper_bounds)
    elif control_mode == "alpha":
        best_k0 = np.asarray(fixed_k0, dtype=float) if fixed_k0 is not None else initial_k0.copy()
    elif control_mode == "steric":
        best_k0 = np.asarray(fixed_k0_for_steric, dtype=float) if fixed_k0_for_steric is not None else initial_k0.copy()
    else:
        best_k0 = initial_k0.copy()
    best_loss = float("inf")
    best_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)

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

    def _x_to_params(x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert optimizer variable x to (k0, alpha, a_vals) in linear space."""
        x = np.asarray(x, dtype=float)
        if control_mode == "k0":
            if use_log_space:
                k0 = np.power(10.0, x)
            else:
                k0 = x.copy()
            k0 = clip_kappa(k0, lower_bounds, upper_bounds)
            return k0, None, None
        elif control_mode == "alpha":
            alpha = x.copy()
            k0 = np.asarray(fixed_k0, dtype=float) if fixed_k0 is not None else initial_k0.copy()
            return k0, alpha, None
        elif control_mode == "joint":
            k0_x = x[:n_k0]
            alpha_x = x[n_k0:]
            k0_log = bool(getattr(request, "log_space", True))
            if k0_log:
                k0 = np.power(10.0, k0_x)
            else:
                k0 = k0_x.copy()
            k0 = clip_kappa(k0, lower_bounds, upper_bounds)
            return k0, alpha_x.copy(), None
        elif control_mode == "steric":
            a_vals = x.copy()
            k0 = np.asarray(fixed_k0_for_steric, dtype=float) if fixed_k0_for_steric is not None else initial_k0.copy()
            alpha = np.asarray(fixed_alpha_for_steric, dtype=float) if fixed_alpha_for_steric is not None else (initial_alpha.copy() if initial_alpha is not None else None)
            return k0, alpha, a_vals
        elif control_mode == "full":
            k0_x = x[:n_k0]
            alpha_x = x[n_k0:n_k0 + n_alpha]
            steric_x = x[n_k0 + n_alpha:]
            k0_log = bool(getattr(request, "log_space", True))
            if k0_log:
                k0 = np.power(10.0, k0_x)
            else:
                k0 = k0_x.copy()
            k0 = clip_kappa(k0, lower_bounds, upper_bounds)
            return k0, alpha_x.copy(), steric_x.copy()
        else:
            return x.copy(), None, None

    def _x_to_k0(x: np.ndarray) -> np.ndarray:
        """Convert optimizer variable x to k0 (linear) space (backward compat)."""
        k0, _, _ = _x_to_params(x)
        return k0

    def _grad_to_x_space(grad_ctrl: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Chain rule transform for log-space parameters."""
        grad_x = grad_ctrl.copy()
        if control_mode == "k0":
            if use_log_space:
                k0 = _x_to_k0(x)
                grad_x = grad_ctrl * k0 * np.log(10.0)
        elif control_mode == "alpha":
            pass  # alpha is linear, no transform
        elif control_mode == "joint":
            k0_log = bool(getattr(request, "log_space", True))
            if k0_log:
                k0, _, _ = _x_to_params(x)
                grad_x[:n_k0] = grad_ctrl[:n_k0] * k0 * np.log(10.0)
            # alpha part stays linear
        elif control_mode == "steric":
            pass  # steric a is linear, no transform
        elif control_mode == "full":
            k0_log = bool(getattr(request, "log_space", True))
            if k0_log:
                k0, _, _ = _x_to_params(x)
                grad_x[:n_k0] = grad_ctrl[:n_k0] * k0 * np.log(10.0)
            # alpha and steric parts stay linear
        return grad_x

    def _key_from_x(x: np.ndarray) -> Tuple[float, ...]:
        k0, alpha, a_vals = _x_to_params(x)
        parts = list(k0.tolist())
        if alpha is not None:
            parts.extend(alpha.tolist())
        if a_vals is not None:
            parts.extend(a_vals.tolist())
        return tuple(float(f"{v:.12g}") for v in parts)

    def _record_point_rows(
        *,
        eval_id: int,
        iter_id: int,
        points: Sequence[PointAdjointResult],
    ) -> None:
        for idx, point in enumerate(points):
            row: Dict[str, object] = {
                "evaluation": eval_id,
                "iteration": iter_id,
                "point_index": idx,
                "phi_applied": point.phi_applied,
                "target_flux": point.target_flux,
                "simulated_flux": point.simulated_flux,
                "point_objective": point.objective,
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
            for j in range(n_controls):
                row[f"dJ_dk0_{j}"] = (
                    float(point.gradient[j]) if j < len(point.gradient) else float("nan")
                )
            point_rows.append(row)
            if bool(request.print_point_gradients):
                grad_str = ", ".join(f"{point.gradient[j]:11.6f}" for j in range(n_controls))
                print(
                    f"  phi={point.phi_applied:8.4f} "
                    f"target={point.target_flux:12.6f} "
                    f"sim={point.simulated_flux:12.6f} "
                    f"dJ/dk0=[{grad_str}] "
                    f"conv={int(point.converged)} "
                    f"rel={(float(point.final_relative_change) if point.final_relative_change is not None else float('nan')):9.3e}"
                )

    # Regularization config: extract once for use in _evaluate
    reg_lambda = float(getattr(request, "regularization_lambda", 0.0))
    reg_k0_prior = getattr(request, "regularization_k0_prior", None)
    reg_alpha_prior = getattr(request, "regularization_alpha_prior", None)
    if reg_k0_prior is not None:
        reg_k0_prior = np.asarray([float(v) for v in reg_k0_prior], dtype=float)
    if reg_alpha_prior is not None:
        reg_alpha_prior = np.asarray([float(v) for v in reg_alpha_prior], dtype=float)

    def _evaluate(x: np.ndarray) -> Dict[str, object]:
        nonlocal best_k0, best_loss, best_flux

        k0_eval, alpha_eval, a_eval = _x_to_params(x)
        key = _key_from_x(x)
        if key in cache:
            return cache[key]

        curve = evaluate_bv_curve_objective_and_gradient(
            request=request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            k0_values=k0_eval,
            mesh=mesh,
            alpha_values=alpha_eval,
            a_values=a_eval,
            control_mode=control_mode,
        )

        eval_counter["n"] += 1
        eval_id = int(eval_counter["n"])
        iter_id = int(iteration_counter["n"])
        objective = float(curve.objective)
        grad_ctrl = np.asarray(curve.gradient, dtype=float)
        grad_x = _grad_to_x_space(grad_ctrl, x)

        # --- Tikhonov regularization ---
        # Adds J_reg to objective and analytical grad_reg to gradient (in x-space).
        # k0: penalize in log10 space: J_reg += lambda * (log10(k0) - log10(prior))^2
        # alpha: penalize in linear space: J_reg += lambda * (alpha - prior)^2
        if reg_lambda > 0.0:
            J_reg = 0.0
            grad_reg_x = np.zeros_like(grad_x)

            if control_mode in ("k0", "joint", "full"):
                # k0 regularization in log10 space
                k0_prior = reg_k0_prior if reg_k0_prior is not None else k0_eval.copy()
                log_k0 = np.log10(np.maximum(k0_eval, 1e-30))
                log_prior = np.log10(np.maximum(k0_prior, 1e-30))
                diff_log = log_k0 - log_prior
                J_reg += reg_lambda * float(np.sum(diff_log ** 2))
                k0_log = bool(getattr(request, "log_space", True))
                if control_mode == "k0":
                    if k0_log:
                        grad_reg_x[:n_k0] = 2.0 * reg_lambda * diff_log
                    else:
                        grad_reg_x[:n_k0] = 2.0 * reg_lambda * diff_log / (k0_eval * np.log(10.0))
                elif control_mode in ("joint", "full"):
                    if k0_log:
                        grad_reg_x[:n_k0] = 2.0 * reg_lambda * diff_log
                    else:
                        grad_reg_x[:n_k0] = 2.0 * reg_lambda * diff_log / (k0_eval * np.log(10.0))

            if control_mode in ("alpha", "joint", "full") and alpha_eval is not None:
                # alpha regularization in linear space
                alpha_prior = reg_alpha_prior if reg_alpha_prior is not None else alpha_eval.copy()
                diff_alpha = alpha_eval - alpha_prior
                J_reg += reg_lambda * float(np.sum(diff_alpha ** 2))
                if control_mode == "alpha":
                    grad_reg_x[:n_alpha] += 2.0 * reg_lambda * diff_alpha
                elif control_mode == "joint":
                    grad_reg_x[n_k0:] += 2.0 * reg_lambda * diff_alpha
                elif control_mode == "full":
                    grad_reg_x[n_k0:n_k0 + n_alpha] += 2.0 * reg_lambda * diff_alpha

            objective += J_reg
            grad_x = grad_x + grad_reg_x

        # For backward compat, keep grad_k0 name for logging
        grad_k0 = grad_ctrl
        grad_norm = float(np.linalg.norm(grad_x))
        n_failed = int(curve.n_failed)
        simulated_flux = np.asarray(curve.simulated_flux, dtype=float)

        k0_str = ", ".join(f"{v:12.6e}" for v in k0_eval.tolist())
        grad_str_short = ", ".join(f"{v:12.6f}" for v in grad_k0.tolist())
        recovery_tag = " aniso_recovery=1" if curve.used_anisotropy_recovery else ""
        print(
            f"[eval={eval_id:03d}] "
            f"k0=[{k0_str}] "
            f"loss={objective:14.6e} "
            f"grad=[{grad_str_short}] "
            f"|grad|={grad_norm:12.6f} "
            f"fails={n_failed:02d}"
            f"{recovery_tag}"
        )

        _record_point_rows(eval_id=eval_id, iter_id=iter_id, points=curve.points)

        live_plot.add_eval_curve(flux=simulated_flux, eval_id=eval_id)
        live_plot.update(
            current_flux=simulated_flux,
            best_flux=best_flux.copy(),
            iteration=int(iter_id),
            objective=objective,
            n_failed=n_failed,
            kappa=k0_eval.copy(),
            eval_id=eval_id,
        )

        hist_row: Dict[str, object] = {
            "evaluation": eval_id,
            "iteration": iter_id,
            "objective": objective,
            "grad_norm": grad_norm,
            "n_failed_points": n_failed,
            "used_anisotropy_recovery": int(bool(curve.used_anisotropy_recovery)),
        }
        for j in range(n_k0):
            hist_row[f"k0_{j}"] = float(k0_eval[j])
        if alpha_eval is not None:
            for j in range(len(alpha_eval)):
                hist_row[f"alpha_{j}"] = float(alpha_eval[j])
        if a_eval is not None:
            for j in range(len(a_eval)):
                hist_row[f"steric_a_{j}"] = float(a_eval[j])
        for j in range(n_controls):
            hist_row[f"grad_{j}"] = float(grad_k0[j])
        for j in range(min(n_k0, len(curve.effective_kappa))):
            hist_row[f"effective_k0_{j}"] = float(curve.effective_kappa[j])
        history_rows.append(hist_row)

        if n_failed == 0 and np.isfinite(objective) and objective < best_loss:
            best_loss = objective
            best_k0 = k0_eval.copy()
            best_flux = simulated_flux.copy()

        payload = {
            "x": k0_eval.copy(),
            "alpha": alpha_eval.copy() if alpha_eval is not None else None,
            "a_vals": a_eval.copy() if a_eval is not None else None,
            "objective": objective,
            "gradient": grad_k0.copy(),
            "gradient_x": grad_x.copy(),
            "simulated_flux": simulated_flux.copy(),
            "n_failed": n_failed,
            "points": list(curve.points),
            "eval_id": eval_id,
        }
        cache[key] = payload
        return payload

    def _fun(x: np.ndarray) -> float:
        return float(_evaluate(x)["objective"])

    def _jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(_evaluate(x)["gradient_x"], dtype=float)

    def _callback(xk: np.ndarray) -> None:
        iteration_counter["n"] += 1
        payload = _evaluate(xk)
        k0_cb = np.asarray(payload["x"], dtype=float)
        grad_x_cb = np.asarray(payload["gradient_x"], dtype=float)
        k0_str = ", ".join(f"{v:12.6e}" for v in k0_cb.tolist())
        print(
            f"[iter={iteration_counter['n']:02d}] "
            f"k0=[{k0_str}] "
            f"loss={float(payload['objective']):14.6e} "
            f"|grad_x|={float(np.linalg.norm(grad_x_cb)):12.6f} "
            f"fails={int(payload['n_failed']):02d}"
        )
        live_plot.update(
            current_flux=np.asarray(payload["simulated_flux"], dtype=float),
            best_flux=best_flux.copy(),
            iteration=int(iteration_counter["n"]),
            objective=float(payload["objective"]),
            n_failed=int(payload["n_failed"]),
            kappa=k0_cb,
            eval_id=int(payload.get("eval_id", -1)),
        )

    # Draw initial curve before first optimizer step
    initial_payload = _evaluate(x0)
    live_plot.update(
        current_flux=np.asarray(initial_payload["simulated_flux"], dtype=float),
        best_flux=best_flux.copy(),
        iteration=0,
        objective=float(initial_payload["objective"]),
        n_failed=int(initial_payload["n_failed"]),
        kappa=_x_to_k0(x0),
        eval_id=int(initial_payload.get("eval_id", -1)),
    )

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

    final_payload = _evaluate(result.x)

    if (
        int(final_payload["n_failed"]) == 0
        and np.isfinite(float(final_payload["objective"]))
        and float(final_payload["objective"]) < best_loss
    ):
        best_loss = float(final_payload["objective"])
        best_k0 = np.asarray(final_payload["x"], dtype=float).copy()
        best_flux = np.asarray(final_payload["simulated_flux"], dtype=float).copy()

    live_plot.update(
        current_flux=np.asarray(final_payload["simulated_flux"], dtype=float),
        best_flux=best_flux.copy(),
        iteration=int(iteration_counter["n"]),
        objective=float(final_payload["objective"]),
        n_failed=int(final_payload["n_failed"]),
        kappa=np.asarray(final_payload["x"], dtype=float),
        eval_id=int(final_payload.get("eval_id", -1)),
    )

    return (
        best_k0,
        float(best_loss),
        best_flux,
        history_rows,
        point_rows,
        result,
        live_plot,
    )


def run_scipy_bv_least_squares_optimization(
    *,
    request: BVFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    initial_k0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    mesh: Any = None,
    control_mode: str = "k0",
    fixed_k0: Optional[np.ndarray] = None,
    initial_alpha: Optional[np.ndarray] = None,
    alpha_lower_bounds: Optional[np.ndarray] = None,
    alpha_upper_bounds: Optional[np.ndarray] = None,
    initial_steric_a: Optional[np.ndarray] = None,
    steric_a_lower_bounds: Optional[np.ndarray] = None,
    steric_a_upper_bounds: Optional[np.ndarray] = None,
    fixed_k0_for_steric: Optional[np.ndarray] = None,
    fixed_alpha_for_steric: Optional[np.ndarray] = None,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Any,
    _LiveFitPlot,
]:
    """Gauss-Newton optimization via scipy.optimize.least_squares.

    Uses per-point adjoint gradients to build a Jacobian for the residual
    vector r_i = sim_flux_i - target_flux_i, enabling quadratic convergence.
    """
    from scipy.optimize import least_squares
    from FluxCurve.bv_curve_eval import (
        evaluate_bv_curve_objective_and_gradient,
        build_residual_jacobian,
    )

    n_species = int(request.base_solver_params.n_species) if hasattr(request.base_solver_params, 'n_species') else int(request.base_solver_params[0])
    n_steric = int(initial_steric_a.size) if initial_steric_a is not None else n_species
    n_k0 = int(initial_k0.size)
    n_alpha = int(initial_alpha.size) if initial_alpha is not None else 0

    # Determine control vector layout (same as run_scipy_bv_adjoint_optimization)
    if control_mode == "k0":
        n_controls = n_k0
        use_log_space = bool(getattr(request, "log_space", True))
    elif control_mode == "alpha":
        n_controls = n_alpha
        use_log_space = False
    elif control_mode == "joint":
        n_controls = n_k0 + n_alpha
        use_log_space = bool(getattr(request, "log_space", True))
    elif control_mode == "steric":
        n_controls = n_steric
        use_log_space = False
    elif control_mode == "full":
        n_controls = n_k0 + n_alpha + n_steric
        use_log_space = bool(getattr(request, "log_space", True))
    else:
        raise ValueError(f"Unknown control_mode '{control_mode}'")

    # Build x0 and bounds
    if control_mode == "k0":
        if use_log_space:
            x0 = np.log10(clip_kappa(initial_k0.copy(), lower_bounds, upper_bounds))
            lb = np.array([np.log10(max(lower_bounds[i], 1e-30)) for i in range(n_controls)])
            ub = np.array([np.log10(upper_bounds[i]) for i in range(n_controls)])
        else:
            x0 = clip_kappa(initial_k0.copy(), lower_bounds, upper_bounds)
            lb = lower_bounds.copy()
            ub = upper_bounds.copy()
    elif control_mode == "alpha":
        a_lo = alpha_lower_bounds if alpha_lower_bounds is not None else np.full(n_controls, 0.05)
        a_hi = alpha_upper_bounds if alpha_upper_bounds is not None else np.full(n_controls, 0.95)
        x0 = np.clip(initial_alpha.copy(), a_lo, a_hi)
        lb = a_lo.copy()
        ub = a_hi.copy()
    elif control_mode == "joint":
        k0_log = bool(getattr(request, "log_space", True))
        x0_k0 = clip_kappa(initial_k0.copy(), lower_bounds, upper_bounds)
        if k0_log:
            x0_k0_x = np.log10(x0_k0)
            k0_lb = np.array([np.log10(max(lower_bounds[i], 1e-30)) for i in range(n_k0)])
            k0_ub = np.array([np.log10(upper_bounds[i]) for i in range(n_k0)])
        else:
            x0_k0_x = x0_k0
            k0_lb = lower_bounds[:n_k0].copy()
            k0_ub = upper_bounds[:n_k0].copy()
        a_lo = alpha_lower_bounds if alpha_lower_bounds is not None else np.full(n_alpha, 0.05)
        a_hi = alpha_upper_bounds if alpha_upper_bounds is not None else np.full(n_alpha, 0.95)
        x0_alpha = np.clip(initial_alpha.copy(), a_lo, a_hi)
        x0 = np.concatenate([x0_k0_x, x0_alpha])
        lb = np.concatenate([k0_lb, a_lo])
        ub = np.concatenate([k0_ub, a_hi])
    elif control_mode == "steric":
        s_lo = steric_a_lower_bounds if steric_a_lower_bounds is not None else np.full(n_steric, 0.001)
        s_hi = steric_a_upper_bounds if steric_a_upper_bounds is not None else np.full(n_steric, 0.5)
        x0 = np.clip(initial_steric_a.copy(), s_lo, s_hi)
        lb = s_lo.copy()
        ub = s_hi.copy()
    elif control_mode == "full":
        k0_log = bool(getattr(request, "log_space", True))
        x0_k0 = clip_kappa(initial_k0.copy(), lower_bounds, upper_bounds)
        if k0_log:
            x0_k0_x = np.log10(x0_k0)
            k0_lb = np.array([np.log10(max(lower_bounds[i], 1e-30)) for i in range(n_k0)])
            k0_ub = np.array([np.log10(upper_bounds[i]) for i in range(n_k0)])
        else:
            x0_k0_x = x0_k0
            k0_lb = lower_bounds[:n_k0].copy()
            k0_ub = upper_bounds[:n_k0].copy()
        a_lo = alpha_lower_bounds if alpha_lower_bounds is not None else np.full(n_alpha, 0.05)
        a_hi = alpha_upper_bounds if alpha_upper_bounds is not None else np.full(n_alpha, 0.95)
        x0_alpha = np.clip(initial_alpha.copy(), a_lo, a_hi)
        s_lo = steric_a_lower_bounds if steric_a_lower_bounds is not None else np.full(n_steric, 0.001)
        s_hi = steric_a_upper_bounds if steric_a_upper_bounds is not None else np.full(n_steric, 0.5)
        x0_steric = np.clip(initial_steric_a.copy(), s_lo, s_hi)
        x0 = np.concatenate([x0_k0_x, x0_alpha, x0_steric])
        lb = np.concatenate([k0_lb, a_lo, s_lo])
        ub = np.concatenate([k0_ub, a_hi, s_hi])

    history_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []
    eval_counter = {"n": 0}
    best_k0 = initial_k0.copy()
    best_loss = float("inf")
    best_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)

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

    def _x_to_params(x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        x = np.asarray(x, dtype=float)
        if control_mode == "k0":
            k0 = np.power(10.0, x) if use_log_space else x.copy()
            k0 = clip_kappa(k0, lower_bounds, upper_bounds)
            return k0, None, None
        elif control_mode == "alpha":
            k0 = np.asarray(fixed_k0, dtype=float) if fixed_k0 is not None else initial_k0.copy()
            return k0, x.copy(), None
        elif control_mode == "joint":
            k0_x = x[:n_k0]
            alpha_x = x[n_k0:]
            k0_log = bool(getattr(request, "log_space", True))
            k0 = np.power(10.0, k0_x) if k0_log else k0_x.copy()
            k0 = clip_kappa(k0, lower_bounds, upper_bounds)
            return k0, alpha_x.copy(), None
        elif control_mode == "steric":
            k0 = np.asarray(fixed_k0_for_steric, dtype=float) if fixed_k0_for_steric is not None else initial_k0.copy()
            alpha = np.asarray(fixed_alpha_for_steric, dtype=float) if fixed_alpha_for_steric is not None else (initial_alpha.copy() if initial_alpha is not None else None)
            return k0, alpha, x.copy()
        elif control_mode == "full":
            k0_x = x[:n_k0]
            alpha_x = x[n_k0:n_k0 + n_alpha]
            steric_x = x[n_k0 + n_alpha:]
            k0_log = bool(getattr(request, "log_space", True))
            k0 = np.power(10.0, k0_x) if k0_log else k0_x.copy()
            k0 = clip_kappa(k0, lower_bounds, upper_bounds)
            return k0, alpha_x.copy(), steric_x.copy()
        return x.copy(), None, None

    # Cache for avoiding redundant evals
    _cache: Dict[tuple, Any] = {}

    def _evaluate_curve(x: np.ndarray):
        nonlocal best_k0, best_loss, best_flux
        k0_eval, alpha_eval, a_eval = _x_to_params(x)
        key = tuple(float(f"{v:.12g}") for v in np.concatenate([
            k0_eval,
            alpha_eval if alpha_eval is not None else [],
            a_eval if a_eval is not None else [],
        ]))
        if key in _cache:
            return _cache[key]

        curve = evaluate_bv_curve_objective_and_gradient(
            request=request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            k0_values=k0_eval,
            mesh=mesh,
            alpha_values=alpha_eval,
            a_values=a_eval,
            control_mode=control_mode,
        )

        eval_counter["n"] += 1
        residuals, jacobian = build_residual_jacobian(curve.points, n_controls)

        # Chain rule for log-space k0 columns in the Jacobian
        if control_mode in ("k0", "joint", "full") and bool(getattr(request, "log_space", True)):
            for j in range(n_k0):
                jacobian[:, j] *= k0_eval[j] * np.log(10.0)

        simulated_flux = np.asarray(curve.simulated_flux, dtype=float)

        if curve.n_failed == 0 and np.isfinite(curve.objective) and curve.objective < best_loss:
            best_loss = float(curve.objective)
            best_k0 = k0_eval.copy()
            best_flux = simulated_flux.copy()

        k0_str = ", ".join(f"{v:12.6e}" for v in k0_eval.tolist())
        print(
            f"[GN eval={eval_counter['n']:03d}] "
            f"k0=[{k0_str}] "
            f"|r|={float(np.linalg.norm(residuals)):14.6e} "
            f"fails={int(curve.n_failed):02d}"
        )

        hist_row: Dict[str, object] = {
            "evaluation": eval_counter["n"],
            "iteration": eval_counter["n"],
            "objective": float(curve.objective),
            "grad_norm": float(np.linalg.norm(curve.gradient)),
            "n_failed_points": int(curve.n_failed),
        }
        for j in range(n_k0):
            hist_row[f"k0_{j}"] = float(k0_eval[j])
        if alpha_eval is not None:
            for j in range(len(alpha_eval)):
                hist_row[f"alpha_{j}"] = float(alpha_eval[j])
        if a_eval is not None:
            for j in range(len(a_eval)):
                hist_row[f"steric_a_{j}"] = float(a_eval[j])
        history_rows.append(hist_row)

        payload = {
            "residuals": residuals,
            "jacobian": jacobian,
            "curve": curve,
            "simulated_flux": simulated_flux,
        }
        _cache[key] = payload
        return payload

    def _residuals(x: np.ndarray) -> np.ndarray:
        return _evaluate_curve(x)["residuals"]

    def _jacobian(x: np.ndarray) -> np.ndarray:
        return _evaluate_curve(x)["jacobian"]

    options = _build_scipy_options(request)
    max_nfev = int(options.get("maxiter", request.max_iters)) * 2 + 1

    result = least_squares(
        _residuals,
        x0=x0,
        jac=_jacobian,
        bounds=(lb, ub),
        method="trf",
        max_nfev=max_nfev,
        verbose=2,
    )

    # Final eval
    final = _evaluate_curve(result.x)
    final_k0, final_alpha, final_a = _x_to_params(result.x)
    final_flux = final["simulated_flux"]
    final_loss = float(0.5 * np.sum(final["residuals"] ** 2))

    if np.isfinite(final_loss) and final_loss < best_loss:
        best_loss = final_loss
        best_k0 = final_k0.copy()
        best_flux = final_flux.copy()

    return (
        best_k0,
        float(best_loss),
        best_flux,
        history_rows,
        point_rows,
        result,
        live_plot,
    )


def _dispatch_bv_optimizer(**kwargs):
    """Route to L-BFGS-B or Gauss-Newton based on request.optimizer_method."""
    request = kwargs["request"]
    method = str(getattr(request, "optimizer_method", "L-BFGS-B"))
    if method.lower() in ("gauss_newton", "gauss-newton"):
        return run_scipy_bv_least_squares_optimization(**kwargs)
    return run_scipy_bv_adjoint_optimization(**kwargs)
