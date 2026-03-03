"""Top-level optimization loop and I/O for BV k0 flux-curve inference."""

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

from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from FluxCurve.results import PointAdjointResult, CurveAdjointResult
from FluxCurve.recovery import clip_kappa
from FluxCurve.bv_curve_eval import (
    evaluate_bv_curve_objective_and_gradient,
    evaluate_bv_multi_observable_objective_and_gradient,
    evaluate_bv_multi_ph_objective_and_gradient,
)
from FluxCurve.plot import _LiveFitPlot, _as_int, _as_float, export_live_fit_gif
from Forward.steady_state import (
    SteadyStateConfig,
    configure_bv_solver_params,
    read_phi_applied_flux_csv,
    sweep_phi_applied_steady_bv_flux,
    all_results_converged,
    results_to_flux_array,
    add_percent_noise,
    write_phi_applied_flux_csv,
)


def _normalize_k0(value: Optional[Sequence[float]], *, name: str) -> Optional[List[float]]:
    """Validate k0-like input into a positive float list."""
    if value is None:
        return None
    vals = [float(v) for v in list(value)]
    if any(v <= 0.0 for v in vals):
        raise ValueError(f"{name} must be strictly positive.")
    return vals


def ensure_bv_target_curve(
    *,
    target_csv_path: str,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    true_k0: Optional[Sequence[float]],
    current_density_scale: float,
    noise_percent: float,
    seed: int,
    force_regenerate: bool,
    blob_initial_condition: bool,
    mesh: Any = None,
) -> Dict[str, np.ndarray]:
    """Load target BV data from CSV or generate synthetic target if missing."""
    if os.path.exists(target_csv_path) and not force_regenerate:
        print(f"Loading BV target curve from: {target_csv_path}")
        return read_phi_applied_flux_csv(target_csv_path, flux_column="flux_noisy")

    k0_true = _normalize_k0(true_k0, name="true_k0")
    if k0_true is None:
        raise ValueError("true_k0 must be set for synthetic target generation.")

    if os.path.exists(target_csv_path):
        print(
            "Regenerating BV target curve with true k0; "
            f"overwriting existing CSV: {target_csv_path}"
        )
    else:
        print("BV target CSV not found; generating synthetic target data first.")
    print(f"Synthetic target settings: true_k0={k0_true}, noise_percent={noise_percent}")

    # compute_bv_current_density computes  I = -(sum R_j) * i_scale
    # The observable form computes           obs = scale * sum(R_j)
    # For consistency: i_scale = -current_density_scale
    # (current_density_scale already includes the sign, e.g. -I_SCALE)
    target_results = sweep_phi_applied_steady_bv_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        k0_values=k0_true,
        i_scale=-current_density_scale,
        mesh=mesh,
        blob_initial_condition=bool(blob_initial_condition),
    )
    if not all_results_converged(target_results):
        failed = [f"{r.phi_applied:.6f}" for r in target_results if not r.converged]
        print(f"WARNING: BV target generation failed for {len(failed)} points: {failed}")
        # Continue with partial data rather than raising

    target_flux_clean = results_to_flux_array(target_results)
    target_flux_noisy = add_percent_noise(target_flux_clean, noise_percent, seed=seed)

    os.makedirs(os.path.dirname(target_csv_path) or ".", exist_ok=True)
    write_phi_applied_flux_csv(target_csv_path, target_results, noisy_flux=target_flux_noisy)
    print(f"BV synthetic target saved to: {target_csv_path}")

    return {"phi_applied": phi_applied_values.copy(), "flux": target_flux_noisy}


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


def write_bv_history_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write optimizer-iteration history to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_bv_point_gradient_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write per-point adjoint gradient diagnostics to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    n_species = int(request.base_solver_params[0])
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

    n_species = int(request.base_solver_params[0])
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


def run_bv_k0_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run end-to-end BV k0 curve inference with adjoint gradients.

    Returns a dict with inference results and artifact paths.
    """
    from Forward.bv_solver import make_graded_rectangle_mesh

    request_runtime = copy.deepcopy(request)

    # Build graded mesh
    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)

    target_data = ensure_bv_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=request_runtime.true_k0,
        current_density_scale=float(request_runtime.current_density_scale),
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        mesh=mesh,
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = np.asarray(target_data["flux"], dtype=float)
    if target_phi_applied.size != target_flux.size:
        raise ValueError("Target phi_applied and flux lengths do not match.")
    phi_applied_values = target_phi_applied.copy()

    initial_k0_list = _normalize_k0(request_runtime.initial_guess, name="initial_guess")
    if initial_k0_list is None:
        raise ValueError("initial_guess must be set.")
    initial_k0 = np.asarray(initial_k0_list, dtype=float)

    n_controls = int(initial_k0.size)
    lower = np.full(n_controls, float(request_runtime.k0_lower), dtype=float)
    upper = np.full(n_controls, float(request_runtime.k0_upper), dtype=float)

    print("=== Adjoint-Gradient BV k0 Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Initial k0: {initial_k0.tolist()}")
    if request_runtime.true_k0 is not None:
        print(f"True k0: {list(request_runtime.true_k0)}")
    print(f"Bounds: lower={lower.tolist()} upper={upper.tolist()}")
    print(
        f"Observable mode: {request_runtime.observable_mode}, "
        f"scale={float(request_runtime.current_density_scale):.6g}"
    )
    print(f"Mesh: {request_runtime.mesh_Nx}x{request_runtime.mesh_Ny}, beta={request_runtime.mesh_beta}")

    # Multi-fidelity coarse phase
    if bool(getattr(request_runtime, 'multifidelity_enabled', False)):
        from FluxCurve.bv_point_solve import _clear_caches
        coarse_mesh = make_graded_rectangle_mesh(
            Nx=int(getattr(request_runtime, 'coarse_mesh_Nx', 4)),
            Ny=int(getattr(request_runtime, 'coarse_mesh_Ny', 100)),
            beta=float(request_runtime.mesh_beta),
        )
        coarse_request = copy.deepcopy(request_runtime)
        coarse_request.max_iters = int(getattr(request_runtime, 'coarse_max_iters', 5))
        coarse_request.live_plot = False
        coarse_request.live_plot_export_gif_path = None
        coarse_request.optimizer_method = "L-BFGS-B"  # always L-BFGS-B for coarse
        if coarse_request.optimizer_options is not None:
            coarse_opts = dict(coarse_request.optimizer_options)
            coarse_opts["maxiter"] = coarse_request.max_iters
            coarse_request.optimizer_options = coarse_opts
        print(f"\n--- Multi-fidelity Phase 1: coarse mesh "
              f"{coarse_request.coarse_mesh_Nx}x{coarse_request.coarse_mesh_Ny}, "
              f"max_iters={coarse_request.max_iters} ---")
        (coarse_k0, _, _, _, _, _, _) = run_scipy_bv_adjoint_optimization(
            request=coarse_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_k0=initial_k0,
            lower_bounds=lower,
            upper_bounds=upper,
            mesh=coarse_mesh,
        )
        _clear_caches()
        initial_k0 = coarse_k0.copy()
        print(f"--- Multi-fidelity Phase 2: fine mesh "
              f"{request_runtime.mesh_Nx}x{request_runtime.mesh_Ny}, "
              f"initial k0 from coarse: {initial_k0.tolist()} ---\n")

    (
        best_k0,
        best_loss,
        best_sim_flux,
        history_rows,
        point_rows,
        opt_result,
        live_plot,
    ) = _dispatch_bv_optimizer(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_k0=initial_k0,
        lower_bounds=lower,
        upper_bounds=upper,
        mesh=mesh,
    )

    print(
        "SciPy minimize summary: "
        f"success={bool(getattr(opt_result, 'success', False))} "
        f"status={getattr(opt_result, 'status', 'n/a')} "
        f"message={str(getattr(opt_result, 'message', '')).strip()}"
    )

    # Save outputs
    os.makedirs(request_runtime.output_dir, exist_ok=True)

    fit_csv_path = os.path.join(
        request_runtime.output_dir, "phi_applied_vs_current_density_fit.csv"
    )
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_current_density,simulated_current_density\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(
        request_runtime.output_dir, "bv_k0_optimization_history.csv"
    )
    write_bv_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(
        request_runtime.output_dir, "bv_k0_point_gradients.csv"
    )
    write_bv_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    fit_plot_path: Optional[str] = None
    if plt is not None:
        fit_plot_path = os.path.join(
            request_runtime.output_dir, "phi_applied_vs_current_density_fit.png"
        )
        if bool(request_runtime.live_plot) and getattr(live_plot, "enabled", False):
            live_plot.update(
                current_flux=best_sim_flux.copy(),
                best_flux=best_sim_flux.copy(),
                iteration=-1,
                objective=float(best_loss),
                n_failed=0,
                kappa=best_k0.copy(),
            )
            live_plot.save(fit_plot_path)
        else:
            plt.figure(figsize=(7, 4))
            plt.plot(phi_applied_values, target_flux, marker="o", linewidth=2,
                     label="target")
            plt.plot(phi_applied_values, best_sim_flux, marker="s", linewidth=2,
                     label="best-fit simulated")
            plt.xlabel("phi_applied (dimensionless)")
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

    print(f"\n=== BV k0 Inference Results ===")
    print(f"Best k0: {best_k0.tolist()}")
    if request_runtime.true_k0 is not None:
        true_k0_arr = np.asarray(request_runtime.true_k0, dtype=float)
        rel_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        print(f"True k0: {true_k0_arr.tolist()}")
        print(f"Relative error: {rel_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")
    print(f"SciPy success: {bool(getattr(opt_result, 'success', False))}")
    print(f"SciPy message: {getattr(opt_result, 'message', '')}")
    print(f"Output: {request_runtime.output_dir}/")

    return {
        "best_k0": best_k0.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "target_flux": target_flux.copy(),
        "best_simulated_flux": best_sim_flux.copy(),
        "fit_csv_path": fit_csv_path,
        "fit_plot_path": fit_plot_path,
        "history_csv_path": history_csv_path,
        "point_gradient_csv_path": point_csv_path,
        "live_gif_path": live_gif_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }


def run_bv_alpha_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run end-to-end BV alpha (transfer coefficient) inference with adjoint gradients.

    Uses fixed k0 values and infers alpha via control_mode="alpha".
    Returns a dict with inference results and artifact paths.
    """
    from Forward.bv_solver import make_graded_rectangle_mesh

    request_runtime = copy.deepcopy(request)

    # Build graded mesh
    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)

    # Fixed k0 values (known)
    fixed_k0 = request_runtime.fixed_k0
    if fixed_k0 is None:
        # Fall back to true_k0 if fixed_k0 not given
        fixed_k0 = request_runtime.true_k0
    if fixed_k0 is None:
        raise ValueError("fixed_k0 (or true_k0) must be set for alpha inference.")
    fixed_k0_arr = np.asarray([float(v) for v in fixed_k0], dtype=float)

    # Generate target curve using true alpha (and fixed k0)
    target_data = ensure_bv_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=list(fixed_k0),
        current_density_scale=float(request_runtime.current_density_scale),
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        mesh=mesh,
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = np.asarray(target_data["flux"], dtype=float)
    if target_phi_applied.size != target_flux.size:
        raise ValueError("Target phi_applied and flux lengths do not match.")
    phi_applied_values = target_phi_applied.copy()

    # Alpha initial guess
    initial_alpha_list = request_runtime.initial_alpha_guess
    if initial_alpha_list is None:
        raise ValueError("initial_alpha_guess must be set for alpha inference.")
    initial_alpha = np.asarray([float(v) for v in initial_alpha_list], dtype=float)

    n_alpha = int(initial_alpha.size)
    alpha_lower = np.full(n_alpha, float(request_runtime.alpha_lower), dtype=float)
    alpha_upper = np.full(n_alpha, float(request_runtime.alpha_upper), dtype=float)

    # k0 bounds (not optimized, but needed by the optimizer signature)
    n_k0 = int(fixed_k0_arr.size)
    k0_lower = np.full(n_k0, float(request_runtime.k0_lower), dtype=float)
    k0_upper = np.full(n_k0, float(request_runtime.k0_upper), dtype=float)

    print("=== Adjoint-Gradient BV Alpha Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Fixed k0: {fixed_k0_arr.tolist()}")
    print(f"Initial alpha: {initial_alpha.tolist()}")
    if request_runtime.true_alpha is not None:
        print(f"True alpha: {list(request_runtime.true_alpha)}")
    print(f"Alpha bounds: lower={alpha_lower.tolist()} upper={alpha_upper.tolist()}")
    print(
        f"Observable mode: {request_runtime.observable_mode}, "
        f"scale={float(request_runtime.current_density_scale):.6g}"
    )
    print(f"Mesh: {request_runtime.mesh_Nx}x{request_runtime.mesh_Ny}, beta={request_runtime.mesh_beta}")

    # Multi-fidelity coarse phase
    if bool(getattr(request_runtime, 'multifidelity_enabled', False)):
        from FluxCurve.bv_point_solve import _clear_caches
        coarse_mesh = make_graded_rectangle_mesh(
            Nx=int(getattr(request_runtime, 'coarse_mesh_Nx', 4)),
            Ny=int(getattr(request_runtime, 'coarse_mesh_Ny', 100)),
            beta=float(request_runtime.mesh_beta),
        )
        coarse_request = copy.deepcopy(request_runtime)
        coarse_request.max_iters = int(getattr(request_runtime, 'coarse_max_iters', 5))
        coarse_request.live_plot = False
        coarse_request.live_plot_export_gif_path = None
        coarse_request.optimizer_method = "L-BFGS-B"  # always L-BFGS-B for coarse
        if coarse_request.optimizer_options is not None:
            coarse_opts = dict(coarse_request.optimizer_options)
            coarse_opts["maxiter"] = coarse_request.max_iters
            coarse_request.optimizer_options = coarse_opts
        print(f"\n--- Multi-fidelity Phase 1: coarse mesh "
              f"{coarse_request.coarse_mesh_Nx}x{coarse_request.coarse_mesh_Ny}, "
              f"max_iters={coarse_request.max_iters} ---")
        (_, _, _, coarse_hist, _, _, _) = run_scipy_bv_adjoint_optimization(
            request=coarse_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_k0=fixed_k0_arr,
            lower_bounds=k0_lower,
            upper_bounds=k0_upper,
            mesh=coarse_mesh,
            control_mode="alpha",
            fixed_k0=fixed_k0_arr,
            initial_alpha=initial_alpha,
            alpha_lower_bounds=alpha_lower,
            alpha_upper_bounds=alpha_upper,
        )
        _clear_caches()
        if coarse_hist:
            for j in range(n_alpha):
                key = f"alpha_{j}"
                if key in coarse_hist[-1]:
                    initial_alpha[j] = float(coarse_hist[-1][key])
        print(f"--- Multi-fidelity Phase 2: fine mesh, "
              f"initial alpha from coarse: {initial_alpha.tolist()} ---\n")

    (
        best_k0,
        best_loss,
        best_sim_flux,
        history_rows,
        point_rows,
        opt_result,
        live_plot,
    ) = _dispatch_bv_optimizer(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_k0=fixed_k0_arr,
        lower_bounds=k0_lower,
        upper_bounds=k0_upper,
        mesh=mesh,
        control_mode="alpha",
        fixed_k0=fixed_k0_arr,
        initial_alpha=initial_alpha,
        alpha_lower_bounds=alpha_lower,
        alpha_upper_bounds=alpha_upper,
    )

    print(
        "SciPy minimize summary: "
        f"success={bool(getattr(opt_result, 'success', False))} "
        f"status={getattr(opt_result, 'status', 'n/a')} "
        f"message={str(getattr(opt_result, 'message', '')).strip()}"
    )

    # Save outputs
    os.makedirs(request_runtime.output_dir, exist_ok=True)

    fit_csv_path = os.path.join(
        request_runtime.output_dir, "phi_applied_vs_current_density_fit.csv"
    )
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_current_density,simulated_current_density\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(
        request_runtime.output_dir, "bv_alpha_optimization_history.csv"
    )
    write_bv_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(
        request_runtime.output_dir, "bv_alpha_point_gradients.csv"
    )
    write_bv_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    fit_plot_path: Optional[str] = None
    if plt is not None:
        fit_plot_path = os.path.join(
            request_runtime.output_dir, "phi_applied_vs_current_density_fit.png"
        )
        if bool(request_runtime.live_plot) and getattr(live_plot, "enabled", False):
            live_plot.update(
                current_flux=best_sim_flux.copy(),
                best_flux=best_sim_flux.copy(),
                iteration=-1,
                objective=float(best_loss),
                n_failed=0,
                kappa=best_k0.copy(),
            )
            live_plot.save(fit_plot_path)
        else:
            plt.figure(figsize=(7, 4))
            plt.plot(phi_applied_values, target_flux, marker="o", linewidth=2,
                     label="target")
            plt.plot(phi_applied_values, best_sim_flux, marker="s", linewidth=2,
                     label="best-fit simulated")
            plt.xlabel("phi_applied (dimensionless)")
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

    # Extract best alpha from history
    best_alpha = initial_alpha.copy()
    if history_rows:
        last_row = history_rows[-1]
        for j in range(n_alpha):
            key = f"alpha_{j}"
            if key in last_row:
                best_alpha[j] = float(last_row[key])

    print(f"\n=== BV Alpha Inference Results ===")
    print(f"Fixed k0: {fixed_k0_arr.tolist()}")
    print(f"Best alpha: {best_alpha.tolist()}")
    if request_runtime.true_alpha is not None:
        true_alpha_arr = np.asarray(request_runtime.true_alpha, dtype=float)
        rel_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"True alpha: {true_alpha_arr.tolist()}")
        print(f"Relative error: {rel_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")
    print(f"SciPy success: {bool(getattr(opt_result, 'success', False))}")
    print(f"SciPy message: {getattr(opt_result, 'message', '')}")
    print(f"Output: {request_runtime.output_dir}/")

    return {
        "best_alpha": best_alpha.copy(),
        "best_k0": best_k0.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "target_flux": target_flux.copy(),
        "best_simulated_flux": best_sim_flux.copy(),
        "fit_csv_path": fit_csv_path,
        "fit_plot_path": fit_plot_path,
        "history_csv_path": history_csv_path,
        "point_gradient_csv_path": point_csv_path,
        "live_gif_path": live_gif_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }


def run_bv_joint_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run end-to-end joint (k0, alpha) inference with adjoint gradients.

    Simultaneously infers both k0 and alpha via control_mode="joint".
    Returns a dict with inference results and artifact paths.
    """
    from Forward.bv_solver import make_graded_rectangle_mesh

    request_runtime = copy.deepcopy(request)

    # Build graded mesh
    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    # Ensure target data
    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)
    target_data = ensure_bv_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=request_runtime.true_k0,
        current_density_scale=float(request_runtime.current_density_scale),
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        mesh=mesh,
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = np.asarray(target_data["flux"], dtype=float)
    phi_applied_values = target_phi_applied.copy()

    initial_k0 = np.asarray(request_runtime.initial_guess, dtype=float)
    initial_alpha = np.asarray(request_runtime.initial_alpha_guess, dtype=float)
    n_k0 = int(initial_k0.size)
    n_alpha = int(initial_alpha.size)

    k0_lo = np.full(n_k0, float(request_runtime.k0_lower))
    k0_hi = np.full(n_k0, float(request_runtime.k0_upper))
    alpha_lo = np.full(n_alpha, float(request_runtime.alpha_lower))
    alpha_hi = np.full(n_alpha, float(request_runtime.alpha_upper))
    # Per-component bound overrides
    if getattr(request_runtime, 'k0_lower_per_component', None) is not None:
        k0_lo = np.asarray(request_runtime.k0_lower_per_component, dtype=float)
    if getattr(request_runtime, 'k0_upper_per_component', None) is not None:
        k0_hi = np.asarray(request_runtime.k0_upper_per_component, dtype=float)
    if getattr(request_runtime, 'alpha_lower_per_component', None) is not None:
        alpha_lo = np.asarray(request_runtime.alpha_lower_per_component, dtype=float)
    if getattr(request_runtime, 'alpha_upper_per_component', None) is not None:
        alpha_hi = np.asarray(request_runtime.alpha_upper_per_component, dtype=float)

    os.makedirs(request_runtime.output_dir, exist_ok=True)

    print("=== Adjoint-Gradient Joint (k0, alpha) Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Initial k0: {initial_k0.tolist()}")
    print(f"Initial alpha: {initial_alpha.tolist()}")
    if request_runtime.true_k0 is not None:
        print(f"True k0: {list(request_runtime.true_k0)}")
    if request_runtime.true_alpha is not None:
        print(f"True alpha: {list(request_runtime.true_alpha)}")
    print(f"k0 bounds: lower={k0_lo.tolist()} upper={k0_hi.tolist()}")
    print(f"alpha bounds: lower={alpha_lo.tolist()} upper={alpha_hi.tolist()}")
    print(f"Observable mode: {request_runtime.observable_mode}, scale={request_runtime.current_density_scale}")
    print(f"Mesh: {request_runtime.mesh_Nx}x{request_runtime.mesh_Ny}, beta={request_runtime.mesh_beta}")

    # Multi-fidelity coarse phase
    if bool(getattr(request_runtime, 'multifidelity_enabled', False)):
        from FluxCurve.bv_point_solve import _clear_caches
        coarse_mesh = make_graded_rectangle_mesh(
            Nx=int(getattr(request_runtime, 'coarse_mesh_Nx', 4)),
            Ny=int(getattr(request_runtime, 'coarse_mesh_Ny', 100)),
            beta=float(request_runtime.mesh_beta),
        )
        coarse_request = copy.deepcopy(request_runtime)
        coarse_request.max_iters = int(getattr(request_runtime, 'coarse_max_iters', 5))
        coarse_request.live_plot = False
        coarse_request.live_plot_export_gif_path = None
        coarse_request.optimizer_method = "L-BFGS-B"  # always L-BFGS-B for coarse
        if coarse_request.optimizer_options is not None:
            coarse_opts = dict(coarse_request.optimizer_options)
            coarse_opts["maxiter"] = coarse_request.max_iters
            coarse_request.optimizer_options = coarse_opts
        print(f"\n--- Multi-fidelity Phase 1: coarse mesh "
              f"{coarse_request.coarse_mesh_Nx}x{coarse_request.coarse_mesh_Ny}, "
              f"max_iters={coarse_request.max_iters} ---")
        (coarse_k0, _, _, coarse_hist, _, _, _) = run_scipy_bv_adjoint_optimization(
            request=coarse_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_k0=initial_k0,
            lower_bounds=k0_lo,
            upper_bounds=k0_hi,
            mesh=coarse_mesh,
            control_mode="joint",
            initial_alpha=initial_alpha,
            alpha_lower_bounds=alpha_lo,
            alpha_upper_bounds=alpha_hi,
        )
        _clear_caches()
        initial_k0 = coarse_k0.copy()
        if coarse_hist:
            for j in range(n_alpha):
                key = f"alpha_{j}"
                if key in coarse_hist[-1]:
                    initial_alpha[j] = float(coarse_hist[-1][key])
        print(f"--- Multi-fidelity Phase 2: fine mesh, "
              f"initial k0 from coarse: {initial_k0.tolist()}, "
              f"initial alpha from coarse: {initial_alpha.tolist()} ---\n")

    (
        best_k0,
        best_loss,
        best_sim_flux,
        history_rows,
        point_rows,
        opt_result,
        live_plot,
    ) = _dispatch_bv_optimizer(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_k0=initial_k0,
        lower_bounds=k0_lo,
        upper_bounds=k0_hi,
        mesh=mesh,
        control_mode="joint",
        initial_alpha=initial_alpha,
        alpha_lower_bounds=alpha_lo,
        alpha_upper_bounds=alpha_hi,
    )

    print(f"\nSciPy minimize summary: success={getattr(opt_result, 'success', 'n/a')} status={getattr(opt_result, 'status', 'n/a')} message={getattr(opt_result, 'message', '')}")

    # Save CSVs
    fit_csv_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_current_density_fit.csv")
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_current_density,simulated_current_density\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(request_runtime.output_dir, "bv_joint_optimization_history.csv")
    write_bv_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(request_runtime.output_dir, "bv_joint_point_gradients.csv")
    write_bv_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    # Save plot
    fit_plot_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_current_density_fit.png")
    if plt is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(phi_applied_values, target_flux, marker="o", linewidth=2,
                 label="target (noisy)")
        plt.plot(phi_applied_values, best_sim_flux, marker="s", linewidth=2,
                 label="best-fit simulated")
        plt.xlabel("phi_applied (dimensionless)")
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

    # Extract best alpha from history
    best_alpha = initial_alpha.copy()
    if history_rows:
        last_row = history_rows[-1]
        for j in range(n_alpha):
            key = f"alpha_{j}"
            if key in last_row:
                best_alpha[j] = float(last_row[key])

    print(f"\n=== Joint (k0, alpha) Inference Results ===")
    print(f"Best k0: {best_k0.tolist()}")
    print(f"Best alpha: {best_alpha.tolist()}")
    if request_runtime.true_k0 is not None:
        true_k0_arr = np.asarray(request_runtime.true_k0, dtype=float)
        k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        print(f"True k0: {true_k0_arr.tolist()}")
        print(f"k0 relative error: {k0_err.tolist()}")
    if request_runtime.true_alpha is not None:
        true_alpha_arr = np.asarray(request_runtime.true_alpha, dtype=float)
        alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"True alpha: {true_alpha_arr.tolist()}")
        print(f"alpha relative error: {alpha_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")
    print(f"SciPy success: {bool(getattr(opt_result, 'success', False))}")
    print(f"SciPy message: {getattr(opt_result, 'message', '')}")
    print(f"Output: {request_runtime.output_dir}/")

    return {
        "best_k0": best_k0.copy(),
        "best_alpha": best_alpha.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "target_flux": target_flux.copy(),
        "best_simulated_flux": best_sim_flux.copy(),
        "fit_csv_path": fit_csv_path,
        "fit_plot_path": fit_plot_path,
        "history_csv_path": history_csv_path,
        "point_gradient_csv_path": point_csv_path,
        "live_gif_path": live_gif_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }


def run_bv_steric_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run end-to-end BV steric 'a' inference with adjoint gradients.

    Uses fixed k0 and alpha values, infers steric a_vals via control_mode="steric".
    Returns a dict with inference results and artifact paths.
    """
    from Forward.bv_solver import make_graded_rectangle_mesh

    request_runtime = copy.deepcopy(request)

    # Build graded mesh
    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)

    # Fixed k0 and alpha (known)
    fixed_k0 = request_runtime.fixed_k0
    if fixed_k0 is None:
        fixed_k0 = request_runtime.true_k0
    if fixed_k0 is None:
        raise ValueError("fixed_k0 (or true_k0) must be set for steric inference.")
    fixed_k0_arr = np.asarray([float(v) for v in fixed_k0], dtype=float)

    fixed_alpha = request_runtime.fixed_alpha
    if fixed_alpha is None:
        fixed_alpha = request_runtime.true_alpha
    if fixed_alpha is None:
        raise ValueError("fixed_alpha (or true_alpha) must be set for steric inference.")
    fixed_alpha_arr = np.asarray([float(v) for v in fixed_alpha], dtype=float)

    # Generate target curve using true parameters (k0, alpha, steric_a all at true values)
    target_data = ensure_bv_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=list(fixed_k0),
        current_density_scale=float(request_runtime.current_density_scale),
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        mesh=mesh,
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = np.asarray(target_data["flux"], dtype=float)
    if target_phi_applied.size != target_flux.size:
        raise ValueError("Target phi_applied and flux lengths do not match.")
    phi_applied_values = target_phi_applied.copy()

    # Steric a initial guess
    initial_steric_a_list = request_runtime.initial_steric_a_guess
    if initial_steric_a_list is None:
        raise ValueError("initial_steric_a_guess must be set for steric inference.")
    initial_steric_a = np.asarray([float(v) for v in initial_steric_a_list], dtype=float)

    n_steric = int(initial_steric_a.size)
    steric_a_lower = np.full(n_steric, float(request_runtime.steric_a_lower), dtype=float)
    steric_a_upper = np.full(n_steric, float(request_runtime.steric_a_upper), dtype=float)

    # k0 bounds (not optimized, but needed by the optimizer signature)
    n_k0 = int(fixed_k0_arr.size)
    k0_lower = np.full(n_k0, float(request_runtime.k0_lower), dtype=float)
    k0_upper = np.full(n_k0, float(request_runtime.k0_upper), dtype=float)

    os.makedirs(request_runtime.output_dir, exist_ok=True)

    print("=== Adjoint-Gradient BV Steric 'a' Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Fixed k0: {fixed_k0_arr.tolist()}")
    print(f"Fixed alpha: {fixed_alpha_arr.tolist()}")
    print(f"Initial steric a: {initial_steric_a.tolist()}")
    if request_runtime.true_steric_a is not None:
        print(f"True steric a: {list(request_runtime.true_steric_a)}")
    print(f"Steric a bounds: lower={steric_a_lower.tolist()} upper={steric_a_upper.tolist()}")
    print(
        f"Observable mode: {request_runtime.observable_mode}, "
        f"scale={float(request_runtime.current_density_scale):.6g}"
    )
    print(f"Mesh: {request_runtime.mesh_Nx}x{request_runtime.mesh_Ny}, beta={request_runtime.mesh_beta}")

    # Multi-fidelity coarse phase
    if bool(getattr(request_runtime, 'multifidelity_enabled', False)):
        from FluxCurve.bv_point_solve import _clear_caches
        coarse_mesh = make_graded_rectangle_mesh(
            Nx=int(getattr(request_runtime, 'coarse_mesh_Nx', 4)),
            Ny=int(getattr(request_runtime, 'coarse_mesh_Ny', 100)),
            beta=float(request_runtime.mesh_beta),
        )
        coarse_request = copy.deepcopy(request_runtime)
        coarse_request.max_iters = int(getattr(request_runtime, 'coarse_max_iters', 5))
        coarse_request.live_plot = False
        coarse_request.live_plot_export_gif_path = None
        coarse_request.optimizer_method = "L-BFGS-B"  # always L-BFGS-B for coarse
        if coarse_request.optimizer_options is not None:
            coarse_opts = dict(coarse_request.optimizer_options)
            coarse_opts["maxiter"] = coarse_request.max_iters
            coarse_request.optimizer_options = coarse_opts
        print(f"\n--- Multi-fidelity Phase 1: coarse mesh "
              f"{coarse_request.coarse_mesh_Nx}x{coarse_request.coarse_mesh_Ny}, "
              f"max_iters={coarse_request.max_iters} ---")
        (_, _, _, coarse_hist, _, _, _) = run_scipy_bv_adjoint_optimization(
            request=coarse_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_k0=fixed_k0_arr,
            lower_bounds=k0_lower,
            upper_bounds=k0_upper,
            mesh=coarse_mesh,
            control_mode="steric",
            fixed_k0=fixed_k0_arr,
            initial_steric_a=initial_steric_a,
            steric_a_lower_bounds=steric_a_lower,
            steric_a_upper_bounds=steric_a_upper,
            fixed_k0_for_steric=fixed_k0_arr,
            fixed_alpha_for_steric=fixed_alpha_arr,
        )
        _clear_caches()
        if coarse_hist:
            for j in range(n_steric):
                key = f"steric_a_{j}"
                if key in coarse_hist[-1]:
                    initial_steric_a[j] = float(coarse_hist[-1][key])
        print(f"--- Multi-fidelity Phase 2: fine mesh, "
              f"initial steric_a from coarse: {initial_steric_a.tolist()} ---\n")

    (
        best_k0,
        best_loss,
        best_sim_flux,
        history_rows,
        point_rows,
        opt_result,
        live_plot,
    ) = _dispatch_bv_optimizer(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_k0=fixed_k0_arr,
        lower_bounds=k0_lower,
        upper_bounds=k0_upper,
        mesh=mesh,
        control_mode="steric",
        fixed_k0=fixed_k0_arr,
        initial_steric_a=initial_steric_a,
        steric_a_lower_bounds=steric_a_lower,
        steric_a_upper_bounds=steric_a_upper,
        fixed_k0_for_steric=fixed_k0_arr,
        fixed_alpha_for_steric=fixed_alpha_arr,
    )

    print(
        "SciPy minimize summary: "
        f"success={bool(getattr(opt_result, 'success', False))} "
        f"status={getattr(opt_result, 'status', 'n/a')} "
        f"message={str(getattr(opt_result, 'message', '')).strip()}"
    )

    # Save outputs
    fit_csv_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_current_density_fit.csv")
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_current_density,simulated_current_density\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(request_runtime.output_dir, "bv_steric_optimization_history.csv")
    write_bv_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(request_runtime.output_dir, "bv_steric_point_gradients.csv")
    write_bv_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    # Save plot
    fit_plot_path: Optional[str] = None
    if plt is not None:
        fit_plot_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_current_density_fit.png")
        plt.figure(figsize=(7, 4))
        plt.plot(phi_applied_values, target_flux, marker="o", linewidth=2,
                 label="target (noisy)")
        plt.plot(phi_applied_values, best_sim_flux, marker="s", linewidth=2,
                 label="best-fit simulated")
        plt.xlabel("phi_applied (dimensionless)")
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

    # Extract best steric a from history
    best_steric_a = initial_steric_a.copy()
    if history_rows:
        last_row = history_rows[-1]
        for j in range(n_steric):
            key = f"steric_a_{j}"
            if key in last_row:
                best_steric_a[j] = float(last_row[key])

    print(f"\n=== BV Steric 'a' Inference Results ===")
    print(f"Fixed k0: {fixed_k0_arr.tolist()}")
    print(f"Fixed alpha: {fixed_alpha_arr.tolist()}")
    print(f"Best steric a: {best_steric_a.tolist()}")
    if request_runtime.true_steric_a is not None:
        true_steric_a_arr = np.asarray(request_runtime.true_steric_a, dtype=float)
        rel_err = np.abs(best_steric_a - true_steric_a_arr) / np.maximum(np.abs(true_steric_a_arr), 1e-16)
        print(f"True steric a: {true_steric_a_arr.tolist()}")
        print(f"Per-species relative error: {rel_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")
    print(f"SciPy success: {bool(getattr(opt_result, 'success', False))}")
    print(f"SciPy message: {getattr(opt_result, 'message', '')}")
    print(f"Output: {request_runtime.output_dir}/")

    return {
        "best_steric_a": best_steric_a.copy(),
        "best_k0": best_k0.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "target_flux": target_flux.copy(),
        "best_simulated_flux": best_sim_flux.copy(),
        "fit_csv_path": fit_csv_path,
        "fit_plot_path": fit_plot_path,
        "history_csv_path": history_csv_path,
        "point_gradient_csv_path": point_csv_path,
        "live_gif_path": live_gif_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }


def run_bv_full_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run end-to-end joint (k0, alpha, steric_a) inference with adjoint gradients.

    Simultaneously infers k0, alpha, and steric a_vals via control_mode="full".
    Returns a dict with inference results and artifact paths.
    """
    from Forward.bv_solver import make_graded_rectangle_mesh

    request_runtime = copy.deepcopy(request)

    # Build graded mesh
    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    # Ensure target data
    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)
    target_data = ensure_bv_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=request_runtime.true_k0,
        current_density_scale=float(request_runtime.current_density_scale),
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        mesh=mesh,
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = np.asarray(target_data["flux"], dtype=float)
    phi_applied_values = target_phi_applied.copy()

    initial_k0 = np.asarray(request_runtime.initial_guess, dtype=float)
    initial_alpha = np.asarray(request_runtime.initial_alpha_guess, dtype=float)
    initial_steric_a = np.asarray(request_runtime.initial_steric_a_guess, dtype=float)
    n_k0 = int(initial_k0.size)
    n_alpha = int(initial_alpha.size)
    n_steric = int(initial_steric_a.size)

    k0_lo = np.full(n_k0, float(request_runtime.k0_lower))
    k0_hi = np.full(n_k0, float(request_runtime.k0_upper))
    alpha_lo = np.full(n_alpha, float(request_runtime.alpha_lower))
    alpha_hi = np.full(n_alpha, float(request_runtime.alpha_upper))
    steric_lo = np.full(n_steric, float(request_runtime.steric_a_lower))
    steric_hi = np.full(n_steric, float(request_runtime.steric_a_upper))

    os.makedirs(request_runtime.output_dir, exist_ok=True)

    print("=== Adjoint-Gradient Full (k0, alpha, steric_a) Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Initial k0: {initial_k0.tolist()}")
    print(f"Initial alpha: {initial_alpha.tolist()}")
    print(f"Initial steric a: {initial_steric_a.tolist()}")
    if request_runtime.true_k0 is not None:
        print(f"True k0: {list(request_runtime.true_k0)}")
    if request_runtime.true_alpha is not None:
        print(f"True alpha: {list(request_runtime.true_alpha)}")
    if request_runtime.true_steric_a is not None:
        print(f"True steric a: {list(request_runtime.true_steric_a)}")
    print(f"k0 bounds: lower={k0_lo.tolist()} upper={k0_hi.tolist()}")
    print(f"alpha bounds: lower={alpha_lo.tolist()} upper={alpha_hi.tolist()}")
    print(f"steric a bounds: lower={steric_lo.tolist()} upper={steric_hi.tolist()}")
    print(f"Observable mode: {request_runtime.observable_mode}, scale={request_runtime.current_density_scale}")
    print(f"Mesh: {request_runtime.mesh_Nx}x{request_runtime.mesh_Ny}, beta={request_runtime.mesh_beta}")

    # Multi-fidelity coarse phase
    if bool(getattr(request_runtime, 'multifidelity_enabled', False)):
        from FluxCurve.bv_point_solve import _clear_caches
        coarse_mesh = make_graded_rectangle_mesh(
            Nx=int(getattr(request_runtime, 'coarse_mesh_Nx', 4)),
            Ny=int(getattr(request_runtime, 'coarse_mesh_Ny', 100)),
            beta=float(request_runtime.mesh_beta),
        )
        coarse_request = copy.deepcopy(request_runtime)
        coarse_request.max_iters = int(getattr(request_runtime, 'coarse_max_iters', 5))
        coarse_request.live_plot = False
        coarse_request.live_plot_export_gif_path = None
        coarse_request.optimizer_method = "L-BFGS-B"  # always L-BFGS-B for coarse
        if coarse_request.optimizer_options is not None:
            coarse_opts = dict(coarse_request.optimizer_options)
            coarse_opts["maxiter"] = coarse_request.max_iters
            coarse_request.optimizer_options = coarse_opts
        print(f"\n--- Multi-fidelity Phase 1: coarse mesh "
              f"{coarse_request.coarse_mesh_Nx}x{coarse_request.coarse_mesh_Ny}, "
              f"max_iters={coarse_request.max_iters} ---")
        (coarse_k0, _, _, coarse_hist, _, _, _) = run_scipy_bv_adjoint_optimization(
            request=coarse_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_k0=initial_k0,
            lower_bounds=k0_lo,
            upper_bounds=k0_hi,
            mesh=coarse_mesh,
            control_mode="full",
            initial_alpha=initial_alpha,
            alpha_lower_bounds=alpha_lo,
            alpha_upper_bounds=alpha_hi,
            initial_steric_a=initial_steric_a,
            steric_a_lower_bounds=steric_lo,
            steric_a_upper_bounds=steric_hi,
        )
        _clear_caches()
        initial_k0 = coarse_k0.copy()
        if coarse_hist:
            for j in range(n_alpha):
                key = f"alpha_{j}"
                if key in coarse_hist[-1]:
                    initial_alpha[j] = float(coarse_hist[-1][key])
            for j in range(n_steric):
                key = f"steric_a_{j}"
                if key in coarse_hist[-1]:
                    initial_steric_a[j] = float(coarse_hist[-1][key])
        print(f"--- Multi-fidelity Phase 2: fine mesh, "
              f"initial k0 from coarse: {initial_k0.tolist()}, "
              f"initial alpha from coarse: {initial_alpha.tolist()}, "
              f"initial steric_a from coarse: {initial_steric_a.tolist()} ---\n")

    (
        best_k0,
        best_loss,
        best_sim_flux,
        history_rows,
        point_rows,
        opt_result,
        live_plot,
    ) = _dispatch_bv_optimizer(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_k0=initial_k0,
        lower_bounds=k0_lo,
        upper_bounds=k0_hi,
        mesh=mesh,
        control_mode="full",
        initial_alpha=initial_alpha,
        alpha_lower_bounds=alpha_lo,
        alpha_upper_bounds=alpha_hi,
        initial_steric_a=initial_steric_a,
        steric_a_lower_bounds=steric_lo,
        steric_a_upper_bounds=steric_hi,
    )

    print(f"\nSciPy minimize summary: success={getattr(opt_result, 'success', 'n/a')} status={getattr(opt_result, 'status', 'n/a')} message={getattr(opt_result, 'message', '')}")

    # Save CSVs
    fit_csv_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_current_density_fit.csv")
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_current_density,simulated_current_density\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(request_runtime.output_dir, "bv_full_optimization_history.csv")
    write_bv_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(request_runtime.output_dir, "bv_full_point_gradients.csv")
    write_bv_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    # Save plot
    fit_plot_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_current_density_fit.png")
    if plt is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(phi_applied_values, target_flux, marker="o", linewidth=2,
                 label="target (noisy)")
        plt.plot(phi_applied_values, best_sim_flux, marker="s", linewidth=2,
                 label="best-fit simulated")
        plt.xlabel("phi_applied (dimensionless)")
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

    # Extract best alpha and steric a from history
    best_alpha = initial_alpha.copy()
    best_steric_a = initial_steric_a.copy()
    if history_rows:
        last_row = history_rows[-1]
        for j in range(n_alpha):
            key = f"alpha_{j}"
            if key in last_row:
                best_alpha[j] = float(last_row[key])
        for j in range(n_steric):
            key = f"steric_a_{j}"
            if key in last_row:
                best_steric_a[j] = float(last_row[key])

    print(f"\n=== Full (k0, alpha, steric_a) Inference Results ===")
    print(f"Best k0: {best_k0.tolist()}")
    print(f"Best alpha: {best_alpha.tolist()}")
    print(f"Best steric a: {best_steric_a.tolist()}")
    if request_runtime.true_k0 is not None:
        true_k0_arr = np.asarray(request_runtime.true_k0, dtype=float)
        k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        print(f"True k0: {true_k0_arr.tolist()}")
        print(f"k0 relative error: {k0_err.tolist()}")
    if request_runtime.true_alpha is not None:
        true_alpha_arr = np.asarray(request_runtime.true_alpha, dtype=float)
        alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"True alpha: {true_alpha_arr.tolist()}")
        print(f"alpha relative error: {alpha_err.tolist()}")
    if request_runtime.true_steric_a is not None:
        true_steric_a_arr = np.asarray(request_runtime.true_steric_a, dtype=float)
        steric_err = np.abs(best_steric_a - true_steric_a_arr) / np.maximum(np.abs(true_steric_a_arr), 1e-16)
        print(f"True steric a: {true_steric_a_arr.tolist()}")
        print(f"steric a relative error: {steric_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")
    print(f"SciPy success: {bool(getattr(opt_result, 'success', False))}")
    print(f"SciPy message: {getattr(opt_result, 'message', '')}")
    print(f"Output: {request_runtime.output_dir}/")

    return {
        "best_k0": best_k0.copy(),
        "best_alpha": best_alpha.copy(),
        "best_steric_a": best_steric_a.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "target_flux": target_flux.copy(),
        "best_simulated_flux": best_sim_flux.copy(),
        "fit_csv_path": fit_csv_path,
        "fit_plot_path": fit_plot_path,
        "history_csv_path": history_csv_path,
        "point_gradient_csv_path": point_csv_path,
        "live_gif_path": live_gif_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }


def _generate_observable_target(
    *,
    base_solver_params: Sequence[object],
    steady: Any,
    phi_applied_values: np.ndarray,
    true_k0: Sequence[float],
    observable_mode: str,
    observable_scale: float,
    noise_percent: float,
    seed: int,
    mesh: Any,
    target_csv_path: str,
    force_regenerate: bool,
) -> np.ndarray:
    """Generate synthetic target data for a specific observable mode.

    Runs the BV point solver at the true k0 values with the specified
    observable mode, then adds noise. Returns the noisy target flux array.
    """
    from Forward.steady_state import add_percent_noise
    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )
    from FluxCurve.config import ForwardRecoveryConfig as _FRC

    if os.path.exists(target_csv_path) and not force_regenerate:
        data = read_phi_applied_flux_csv(target_csv_path, flux_column="flux_noisy")
        return np.asarray(data["flux"], dtype=float)

    _clear_caches()

    # Use a dummy target (zeros) -- we only need the simulated flux at true params
    dummy_target = np.zeros_like(phi_applied_values, dtype=float)
    k0_list = [float(v) for v in true_k0]

    points = solve_bv_curve_points_with_warmstart(
        base_solver_params=base_solver_params,
        steady=steady,
        phi_applied_values=phi_applied_values,
        target_flux=dummy_target,
        k0_values=k0_list,
        blob_initial_condition=False,
        fail_penalty=1e9,
        forward_recovery=_FRC(),
        observable_mode=observable_mode,
        observable_reaction_index=None,
        observable_scale=observable_scale,
        mesh=mesh,
        alpha_values=None,
        control_mode="k0",
        max_eta_gap=0.0,
    )

    _clear_caches()

    clean_flux = np.array([float(p.simulated_flux) for p in points], dtype=float)
    noisy_flux = add_percent_noise(clean_flux, noise_percent, seed=seed)

    # Save to CSV
    os.makedirs(os.path.dirname(target_csv_path) or ".", exist_ok=True)
    with open(target_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phi_applied", "flux_clean", "flux_noisy"])
        for phi, fc, fn in zip(phi_applied_values, clean_flux, noisy_flux):
            writer.writerow([f"{phi:.16g}", f"{fc:.16g}", f"{fn:.16g}"])
    print(f"Saved {observable_mode} target CSV: {target_csv_path}")

    return noisy_flux


def run_bv_multi_observable_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run joint (k0, alpha) inference fitting TWO observables simultaneously.

    Uses the primary ``observable_mode`` (typically ``"current_density"``) and
    a secondary ``secondary_observable_mode`` (typically ``"peroxide_current"``).
    The combined objective is::

        J = J_primary + weight * J_secondary

    where ``weight = request.secondary_observable_weight`` (default 1.0).

    Target data is generated separately for each observable at the true
    parameter values, ensuring the target for each observable is physically
    consistent.
    """
    from scipy.optimize import minimize
    from Forward.bv_solver import make_graded_rectangle_mesh
    from FluxCurve.bv_point_solve import _clear_caches

    request_runtime = copy.deepcopy(request)

    secondary_mode = str(request_runtime.secondary_observable_mode or "peroxide_current")
    secondary_weight = float(request_runtime.secondary_observable_weight)
    secondary_scale = request_runtime.secondary_current_density_scale
    if secondary_scale is None:
        secondary_scale = float(request_runtime.current_density_scale)
    else:
        secondary_scale = float(secondary_scale)

    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)

    if request_runtime.true_k0 is None:
        raise ValueError("true_k0 must be set for multi-observable synthetic target generation.")

    os.makedirs(request_runtime.output_dir, exist_ok=True)

    # Generate primary target (e.g. total current density)
    primary_target_csv = request_runtime.target_csv_path
    target_data_primary = ensure_bv_target_curve(
        target_csv_path=primary_target_csv,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=request_runtime.true_k0,
        current_density_scale=float(request_runtime.current_density_scale),
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=False,
        mesh=mesh,
    )
    target_flux_primary = np.asarray(target_data_primary["flux"], dtype=float)
    phi_applied_values = np.asarray(target_data_primary["phi_applied"], dtype=float)

    # Generate secondary target (e.g. peroxide current)
    secondary_target_csv = request_runtime.secondary_target_csv_path
    if secondary_target_csv is None:
        secondary_target_csv = os.path.join(
            request_runtime.output_dir,
            f"target_{secondary_mode}.csv",
        )

    target_flux_secondary = _generate_observable_target(
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_k0=request_runtime.true_k0,
        observable_mode=secondary_mode,
        observable_scale=secondary_scale,
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed) + 1,  # different seed for independence
        mesh=mesh,
        target_csv_path=secondary_target_csv,
        force_regenerate=bool(request_runtime.regenerate_target),
    )

    _clear_caches()

    # Setup optimizer
    initial_k0 = np.asarray(request_runtime.initial_guess, dtype=float)
    initial_alpha = np.asarray(request_runtime.initial_alpha_guess, dtype=float)
    n_k0 = int(initial_k0.size)
    n_alpha = int(initial_alpha.size)

    k0_lo = np.full(n_k0, float(request_runtime.k0_lower))
    k0_hi = np.full(n_k0, float(request_runtime.k0_upper))
    alpha_lo = np.full(n_alpha, float(request_runtime.alpha_lower))
    alpha_hi = np.full(n_alpha, float(request_runtime.alpha_upper))

    k0_log = bool(getattr(request_runtime, "log_space", True))
    x0_k0 = clip_kappa(initial_k0, k0_lo, k0_hi)
    if k0_log:
        x0_k0_x = np.log10(x0_k0)
        k0_bounds = [
            (float(np.log10(max(k0_lo[i], 1e-30))), float(np.log10(k0_hi[i])))
            for i in range(n_k0)
        ]
    else:
        x0_k0_x = x0_k0.copy()
        k0_bounds = [(float(k0_lo[i]), float(k0_hi[i])) for i in range(n_k0)]

    x0_alpha = np.clip(initial_alpha, alpha_lo, alpha_hi)
    alpha_bounds = [(float(alpha_lo[i]), float(alpha_hi[i])) for i in range(n_alpha)]

    x0 = np.concatenate([x0_k0_x, x0_alpha])
    bounds = k0_bounds + alpha_bounds

    print("=== Multi-Observable Joint (k0, alpha) Inference ===")
    print(f"Primary observable: {request_runtime.observable_mode}")
    print(f"Secondary observable: {secondary_mode} (weight={secondary_weight})")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Initial k0: {initial_k0.tolist()}")
    print(f"Initial alpha: {initial_alpha.tolist()}")
    if request_runtime.true_k0 is not None:
        print(f"True k0: {list(request_runtime.true_k0)}")
    if request_runtime.true_alpha is not None:
        print(f"True alpha: {list(request_runtime.true_alpha)}")

    options = _build_scipy_options(request_runtime)
    cache = {}
    eval_counter = {"n": 0}
    best_k0 = x0_k0.copy()
    best_alpha = x0_alpha.copy()
    best_loss = float("inf")
    best_flux_primary = np.full(phi_applied_values.shape, np.nan, dtype=float)
    best_flux_secondary = np.full(phi_applied_values.shape, np.nan, dtype=float)
    history_rows = []

    def _x_to_params(x):
        x = np.asarray(x, dtype=float)
        k0_x = x[:n_k0]
        alpha_x = x[n_k0:]
        if k0_log:
            k0 = np.power(10.0, k0_x)
        else:
            k0 = k0_x.copy()
        k0 = clip_kappa(k0, k0_lo, k0_hi)
        return k0, alpha_x.copy()

    def _evaluate(x):
        nonlocal best_k0, best_alpha, best_loss, best_flux_primary, best_flux_secondary

        k0_eval, alpha_eval = _x_to_params(x)
        key = tuple(float(f"{v:.12g}") for v in list(k0_eval) + list(alpha_eval))
        if key in cache:
            return cache[key]

        curve = evaluate_bv_multi_observable_objective_and_gradient(
            request=request_runtime,
            phi_applied_values=phi_applied_values,
            target_flux_primary=target_flux_primary,
            target_flux_secondary=target_flux_secondary,
            k0_values=k0_eval,
            mesh=mesh,
            alpha_values=alpha_eval,
            control_mode="joint",
        )

        eval_counter["n"] += 1
        objective = float(curve.objective)
        grad_ctrl = np.asarray(curve.gradient, dtype=float)

        # Chain rule for log-space k0
        grad_x = grad_ctrl.copy()
        if k0_log:
            grad_x[:n_k0] = grad_ctrl[:n_k0] * k0_eval * np.log(10.0)

        sim_flux_primary = np.asarray(curve.simulated_flux, dtype=float)
        sec_result = getattr(curve, "_secondary_result", None)
        sim_flux_secondary = np.asarray(sec_result.simulated_flux, dtype=float) if sec_result else np.full_like(sim_flux_primary, np.nan)

        k0_str = ", ".join(f"{v:.6e}" for v in k0_eval)
        alpha_str = ", ".join(f"{v:.4f}" for v in alpha_eval)
        print(f"  [eval {eval_counter['n']:>3d}] J={objective:12.6e} "
              f"k0=[{k0_str}] alpha=[{alpha_str}] |grad|={np.linalg.norm(grad_x):.4e}")

        if objective < best_loss:
            best_loss = objective
            best_k0 = k0_eval.copy()
            best_alpha = alpha_eval.copy()
            best_flux_primary = sim_flux_primary.copy()
            best_flux_secondary = sim_flux_secondary.copy()

        row = {
            "evaluation": eval_counter["n"],
            "objective": objective,
            "grad_norm": float(np.linalg.norm(grad_x)),
        }
        for j in range(n_k0):
            row[f"k0_{j}"] = float(k0_eval[j])
        for j in range(n_alpha):
            row[f"alpha_{j}"] = float(alpha_eval[j])
        history_rows.append(row)

        result = {"objective": objective, "grad_x": grad_x}
        cache[key] = result
        return result

    def _fun(x):
        return float(_evaluate(x)["objective"])

    def _jac(x):
        return np.asarray(_evaluate(x)["grad_x"], dtype=float)

    opt_result = minimize(
        _fun, x0, jac=_jac, method="L-BFGS-B",
        bounds=bounds, options=options,
    )

    # Save outputs
    fit_csv_path = os.path.join(request_runtime.output_dir, "multi_obs_fit.csv")
    with open(fit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phi_applied",
            "target_primary", "simulated_primary",
            "target_secondary", "simulated_secondary",
        ])
        for i in range(len(phi_applied_values)):
            writer.writerow([
                f"{phi_applied_values[i]:.16g}",
                f"{target_flux_primary[i]:.16g}",
                f"{best_flux_primary[i]:.16g}",
                f"{target_flux_secondary[i]:.16g}",
                f"{best_flux_secondary[i]:.16g}",
            ])
    print(f"Saved multi-observable fit CSV: {fit_csv_path}")

    history_csv_path = os.path.join(request_runtime.output_dir, "multi_obs_history.csv")
    write_bv_history_csv(history_csv_path, history_rows)

    # Plot
    fit_plot_path = os.path.join(request_runtime.output_dir, "multi_obs_fit.png")
    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(phi_applied_values, target_flux_primary, "o-", label="target (primary)")
        ax1.plot(phi_applied_values, best_flux_primary, "s-", label="fit (primary)")
        ax1.set_xlabel("phi_applied")
        ax1.set_ylabel(str(request_runtime.observable_label))
        ax1.set_title(f"Primary: {request_runtime.observable_mode}")
        ax1.legend()
        ax1.grid(True, alpha=0.25)

        ax2.plot(phi_applied_values, target_flux_secondary, "o-", label="target (secondary)")
        ax2.plot(phi_applied_values, best_flux_secondary, "s-", label="fit (secondary)")
        ax2.set_xlabel("phi_applied")
        ax2.set_ylabel(f"{secondary_mode}")
        ax2.set_title(f"Secondary: {secondary_mode}")
        ax2.legend()
        ax2.grid(True, alpha=0.25)

        fig.suptitle("Multi-Observable Joint Fit")
        fig.tight_layout()
        fig.savefig(fit_plot_path, dpi=160)
        plt.close(fig)
    print(f"Saved fit plot: {fit_plot_path}")

    # Print results
    print(f"\n=== Multi-Observable Inference Results ===")
    print(f"Best k0: {best_k0.tolist()}")
    print(f"Best alpha: {best_alpha.tolist()}")
    if request_runtime.true_k0 is not None:
        true_k0_arr = np.asarray(request_runtime.true_k0, dtype=float)
        k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        print(f"True k0: {true_k0_arr.tolist()}")
        print(f"k0 relative error: {k0_err.tolist()}")
    if request_runtime.true_alpha is not None:
        true_alpha_arr = np.asarray(request_runtime.true_alpha, dtype=float)
        alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"True alpha: {true_alpha_arr.tolist()}")
        print(f"alpha relative error: {alpha_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")

    return {
        "best_k0": best_k0.copy(),
        "best_alpha": best_alpha.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "target_flux_primary": target_flux_primary.copy(),
        "target_flux_secondary": target_flux_secondary.copy(),
        "best_simulated_flux_primary": best_flux_primary.copy(),
        "best_simulated_flux_secondary": best_flux_secondary.copy(),
        "fit_csv_path": fit_csv_path,
        "fit_plot_path": fit_plot_path,
        "history_csv_path": history_csv_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }


def run_bv_multi_ph_flux_curve_inference(
    request: BVFluxCurveInferenceRequest,
) -> Dict[str, Any]:
    """Run joint (k0, alpha) inference across multiple pH conditions.

    Each pH condition has a different c_H+ bulk concentration, producing a
    different I-V curve. The k0 and alpha controls are SHARED across all
    conditions. The combined objective sums (weighted) across conditions.

    pH conditions are specified via ``request.multi_ph_conditions``, a list
    of dicts each containing:
      - ``c_hp_hat``: nondimensional H+ bulk concentration
      - ``weight``: weighting factor (default 1.0)
      - ``target_csv_path``: path for this condition's target CSV
      - ``c_hp_species_index``: index of H+ in species list (default 2)
    """
    from scipy.optimize import minimize
    from Forward.bv_solver import make_graded_rectangle_mesh
    from FluxCurve.bv_point_solve import _clear_caches

    request_runtime = copy.deepcopy(request)

    conditions = request_runtime.multi_ph_conditions
    if not conditions:
        raise ValueError("multi_ph_conditions must be set for multi-pH inference.")

    mesh = make_graded_rectangle_mesh(
        Nx=int(request_runtime.mesh_Nx),
        Ny=int(request_runtime.mesh_Ny),
        beta=float(request_runtime.mesh_beta),
    )

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)

    if request_runtime.true_k0 is None:
        raise ValueError("true_k0 must be set for multi-pH synthetic target generation.")

    os.makedirs(request_runtime.output_dir, exist_ok=True)

    # Generate target data for each pH condition
    condition_targets = []
    for ci, cond in enumerate(conditions):
        c_hp_hat = float(cond["c_hp_hat"])
        c_hp_idx = int(cond.get("c_hp_species_index", 2))
        weight = float(cond.get("weight", 1.0))
        target_csv = cond.get("target_csv_path", os.path.join(
            request_runtime.output_dir, f"target_ph_cond_{ci}.csv",
        ))

        # Modify solver params for this pH condition
        cond_params = copy.deepcopy(request_runtime.base_solver_params)
        bulk_concs = list(cond_params[8])
        bulk_concs[c_hp_idx] = c_hp_hat
        # Also update counterion for electroneutrality (e.g. ClO4- matches H+)
        counterion_idx = cond.get("counterion_species_index", None)
        if counterion_idx is not None:
            bulk_concs[int(counterion_idx)] = c_hp_hat
        cond_params[8] = bulk_concs
        # Update cathodic_conc_factors c_ref_nondim for H+
        bv_cfg = cond_params[10].get("bv_bc", {})
        for rxn in bv_cfg.get("reactions", []):
            for ccf in rxn.get("cathodic_conc_factors", []):
                if int(ccf.get("species", -1)) == c_hp_idx:
                    ccf["c_ref_nondim"] = c_hp_hat

        # Generate target
        target_data = ensure_bv_target_curve(
            target_csv_path=target_csv,
            base_solver_params=cond_params,
            steady=request_runtime.steady,
            phi_applied_values=phi_applied_values,
            true_k0=request_runtime.true_k0,
            current_density_scale=float(request_runtime.current_density_scale),
            noise_percent=float(request_runtime.target_noise_percent),
            seed=int(request_runtime.target_seed) + ci,
            force_regenerate=bool(request_runtime.regenerate_target),
            blob_initial_condition=False,
            mesh=mesh,
        )
        target_flux_cond = np.asarray(target_data["flux"], dtype=float)

        condition_targets.append({
            "target_flux": target_flux_cond,
            "c_hp_hat": c_hp_hat,
            "c_hp_species_index": c_hp_idx,
            "counterion_species_index": cond.get("counterion_species_index", None),
            "weight": weight,
        })
        print(f"  pH condition {ci}: c_hp_hat={c_hp_hat:.4f}, weight={weight}")

    _clear_caches()

    # Setup optimizer (same as joint mode)
    initial_k0 = np.asarray(request_runtime.initial_guess, dtype=float)
    initial_alpha = np.asarray(request_runtime.initial_alpha_guess, dtype=float)
    n_k0 = int(initial_k0.size)
    n_alpha = int(initial_alpha.size)

    k0_lo = np.full(n_k0, float(request_runtime.k0_lower))
    k0_hi = np.full(n_k0, float(request_runtime.k0_upper))
    alpha_lo = np.full(n_alpha, float(request_runtime.alpha_lower))
    alpha_hi = np.full(n_alpha, float(request_runtime.alpha_upper))

    k0_log = bool(getattr(request_runtime, "log_space", True))
    x0_k0 = clip_kappa(initial_k0, k0_lo, k0_hi)
    if k0_log:
        x0_k0_x = np.log10(x0_k0)
        k0_bounds = [
            (float(np.log10(max(k0_lo[i], 1e-30))), float(np.log10(k0_hi[i])))
            for i in range(n_k0)
        ]
    else:
        x0_k0_x = x0_k0.copy()
        k0_bounds = [(float(k0_lo[i]), float(k0_hi[i])) for i in range(n_k0)]

    x0_alpha = np.clip(initial_alpha, alpha_lo, alpha_hi)
    alpha_bounds = [(float(alpha_lo[i]), float(alpha_hi[i])) for i in range(n_alpha)]

    x0 = np.concatenate([x0_k0_x, x0_alpha])
    bounds = k0_bounds + alpha_bounds

    print(f"\n=== Multi-pH Joint (k0, alpha) Inference ===")
    print(f"Number of pH conditions: {len(conditions)}")
    print(f"Target points per condition: {len(phi_applied_values)}")
    print(f"Initial k0: {initial_k0.tolist()}")
    print(f"Initial alpha: {initial_alpha.tolist()}")

    options = _build_scipy_options(request_runtime)
    cache = {}
    eval_counter = {"n": 0}
    best_k0 = x0_k0.copy()
    best_alpha = x0_alpha.copy()
    best_loss = float("inf")
    history_rows = []

    def _x_to_params(x):
        x = np.asarray(x, dtype=float)
        k0_x = x[:n_k0]
        alpha_x = x[n_k0:]
        if k0_log:
            k0 = np.power(10.0, k0_x)
        else:
            k0 = k0_x.copy()
        k0 = clip_kappa(k0, k0_lo, k0_hi)
        return k0, alpha_x.copy()

    def _evaluate(x):
        nonlocal best_k0, best_alpha, best_loss

        k0_eval, alpha_eval = _x_to_params(x)
        key = tuple(float(f"{v:.12g}") for v in list(k0_eval) + list(alpha_eval))
        if key in cache:
            return cache[key]

        combined = evaluate_bv_multi_ph_objective_and_gradient(
            request=request_runtime,
            phi_applied_values=phi_applied_values,
            ph_conditions=condition_targets,
            k0_values=k0_eval,
            mesh=mesh,
            alpha_values=alpha_eval,
            control_mode="joint",
        )

        eval_counter["n"] += 1
        objective = float(combined["objective"])
        grad_ctrl = np.asarray(combined["gradient"], dtype=float)

        grad_x = grad_ctrl.copy()
        if k0_log:
            grad_x[:n_k0] = grad_ctrl[:n_k0] * k0_eval * np.log(10.0)

        k0_str = ", ".join(f"{v:.6e}" for v in k0_eval)
        alpha_str = ", ".join(f"{v:.4f}" for v in alpha_eval)
        print(f"  [eval {eval_counter['n']:>3d}] J={objective:12.6e} "
              f"k0=[{k0_str}] alpha=[{alpha_str}] |grad|={np.linalg.norm(grad_x):.4e}")

        if objective < best_loss:
            best_loss = objective
            best_k0 = k0_eval.copy()
            best_alpha = alpha_eval.copy()

        row = {
            "evaluation": eval_counter["n"],
            "objective": objective,
            "grad_norm": float(np.linalg.norm(grad_x)),
        }
        for j in range(n_k0):
            row[f"k0_{j}"] = float(k0_eval[j])
        for j in range(n_alpha):
            row[f"alpha_{j}"] = float(alpha_eval[j])
        # Per-condition objectives
        for cr in combined.get("condition_results", []):
            row[f"J_cond_{cr['condition_index']}"] = float(cr["objective"])
        history_rows.append(row)

        result = {"objective": objective, "grad_x": grad_x}
        cache[key] = result
        return result

    def _fun(x):
        return float(_evaluate(x)["objective"])

    def _jac(x):
        return np.asarray(_evaluate(x)["grad_x"], dtype=float)

    opt_result = minimize(
        _fun, x0, jac=_jac, method="L-BFGS-B",
        bounds=bounds, options=options,
    )

    # Save outputs
    history_csv_path = os.path.join(request_runtime.output_dir, "multi_ph_history.csv")
    write_bv_history_csv(history_csv_path, history_rows)

    # Print results
    print(f"\n=== Multi-pH Inference Results ===")
    print(f"Best k0: {best_k0.tolist()}")
    print(f"Best alpha: {best_alpha.tolist()}")
    if request_runtime.true_k0 is not None:
        true_k0_arr = np.asarray(request_runtime.true_k0, dtype=float)
        k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        print(f"True k0: {true_k0_arr.tolist()}")
        print(f"k0 relative error: {k0_err.tolist()}")
    if request_runtime.true_alpha is not None:
        true_alpha_arr = np.asarray(request_runtime.true_alpha, dtype=float)
        alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"True alpha: {true_alpha_arr.tolist()}")
        print(f"alpha relative error: {alpha_err.tolist()}")
    print(f"Final objective: {best_loss:.12e}")

    return {
        "best_k0": best_k0.copy(),
        "best_alpha": best_alpha.copy(),
        "best_loss": float(best_loss),
        "phi_applied_values": phi_applied_values.copy(),
        "condition_targets": condition_targets,
        "history_csv_path": history_csv_path,
        "optimization_success": bool(getattr(opt_result, "success", False)),
        "optimization_message": str(getattr(opt_result, "message", "")),
    }
