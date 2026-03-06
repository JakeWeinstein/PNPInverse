"""Public pipeline functions for BV flux-curve inference."""

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
from FluxCurve.recovery import clip_kappa
from FluxCurve.bv_curve_eval import (
    evaluate_bv_multi_observable_objective_and_gradient,
    evaluate_bv_multi_ph_objective_and_gradient,
)
from FluxCurve.plot import _LiveFitPlot, export_live_fit_gif
from Forward.steady_state import read_phi_applied_flux_csv

from .io import (
    _normalize_k0,
    ensure_bv_target_curve,
    write_bv_history_csv,
    write_bv_point_gradient_csv,
    _generate_observable_target,
)
from .optimization import (
    _build_scipy_options,
    run_scipy_bv_adjoint_optimization,
    _dispatch_bv_optimizer,
)


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

    # Extract true_alpha for target generation (C2 fix: targets must use true alpha)
    _true_alpha = (
        list(request_runtime.true_alpha)
        if request_runtime.true_alpha is not None
        else None
    )

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
        alpha_values=_true_alpha,
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
        alpha_values=_true_alpha,
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
        _cp_c0 = cond_params.c0_vals if hasattr(cond_params, 'c0_vals') else cond_params[8]
        bulk_concs = list(_cp_c0)
        bulk_concs[c_hp_idx] = c_hp_hat
        # Also update counterion for electroneutrality (e.g. ClO4- matches H+)
        counterion_idx = cond.get("counterion_species_index", None)
        if counterion_idx is not None:
            bulk_concs[int(counterion_idx)] = c_hp_hat
        cond_params[8] = bulk_concs
        # Update cathodic_conc_factors c_ref_nondim for H+
        _cp_opts = cond_params.solver_options if hasattr(cond_params, 'solver_options') else cond_params[10]
        bv_cfg = _cp_opts.get("bv_bc", {})
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
