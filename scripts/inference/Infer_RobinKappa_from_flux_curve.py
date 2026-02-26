"""Adjoint-gradient inference of Robin kappa from a phi_applied-flux curve.

This entry script is intentionally short and user-config driven.
Edit the values in ``RobinFluxCurveInferenceRequest`` to set:
- true kappa used for synthetic data generation,
- initial guess for inversion,
- noise level and optimizer settings.
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

# Keep Firedrake cache paths writable in sandboxed/restricted environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from FluxCurve import (
    ForwardRecoveryConfig,
    RobinFluxCurveInferenceRequest,
    run_robin_kappa_flux_curve_inference,
)
from Inverse import build_default_solver_params
from Forward.steady_state import SteadyStateConfig


def build_solver_options() -> dict:
    """Return PETSc/SNES options used for Robin flux studies."""
    return {
        "snes_type": "newtonls",
        "snes_max_it": 100,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_linesearch_type": "bt",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "robin_bc": {
            "kappa": [0.8, 0.8],
            "c_inf": [0.01, 0.01],
            "electrode_marker": 1,
            "concentration_marker": 3,
            "ground_marker": 3,
        },
    }


def main() -> None:
    # Base solver used for all point solves in the phi_applied sweep.
    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=1e-1,
        t_end=20.0,
        z_vals=[1, -1],
        d_vals=[1.0, 1.0],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[0.1, 0.1],
        phi0=0.05,
        solver_options=build_solver_options(),
    )

    # Steady-state criterion for each sweep point.
    steady = SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        max_steps=120,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    # User-editable inference configuration.
    request = RobinFluxCurveInferenceRequest(
        base_solver_params=base_solver_params,
        steady=steady,
        true_value=[1, 2],  # Synthetic truth for target curve generation.
        initial_guess=[5.0, 5.0],  # Starting point for optimization.
        phi_applied_values=np.linspace(0.0, 0.04, 15),
        target_csv_path=os.path.join(
            "StudyResults", "robin_flux_experiment", "phi_applied_vs_steady_flux_synthetic.csv"
        ),
        output_dir=os.path.join("StudyResults", "robin_flux_experiment"),
        regenerate_target=True,
        target_noise_percent=5.0,
        target_seed=20260222,
        kappa_lower=1e-6,
        kappa_upper=20.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 80,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": True,
        },
        max_iters=8,
        gtol=1e-4,
        fail_penalty=1e9,
        print_point_gradients=True,
        # Use simple IC (uniform concentrations + linear potential), not blob.
        blob_initial_condition=False,
        # Live plot: true curve + best/current simulated curves, updated each SciPy iteration.
        live_plot=True,
        live_plot_pause_seconds=0.001,
        # Draw each fresh objective/gradient evaluation as a distinct colored line.
        live_plot_eval_lines=True,
        live_plot_eval_line_alpha=0.30,
        live_plot_eval_max_lines=120,
        # Export a standalone convergence GIF that does not depend on PDF animation support.
        live_plot_export_gif_path=os.path.join(
            "StudyResults", "robin_flux_experiment", "robin_kappa_fit_convergence.gif"
        ),
        live_plot_export_gif_seconds=5.0,
        live_plot_export_gif_frames=50,
        live_plot_export_gif_dpi=140,
        # Optional process-parallel execution across phi_applied points.
        # Safe for adjoints because each worker has its own process + tape.
        parallel_point_solves_enabled=False,
        parallel_point_workers=4,
        parallel_point_min_points=4,
        parallel_start_method="spawn",
        # If enough sweep points fail at a candidate kappa, try an anisotropy-reduced
        # kappa surrogate for that evaluation.
        anisotropy_trigger_failed_points=4,
        anisotropy_trigger_failed_fraction=0.25,
        # Per-point forward-solve retry policy (mirrors resilient_minimize staging).
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=8,
            max_it_only_attempts=2,
            anisotropy_only_attempts=1,
            tolerance_relax_attempts=2,
            max_it_growth=1.5,
            max_it_cap=500,
            atol_relax_factor=10.0,
            rtol_relax_factor=10.0,
            ksp_rtol_relax_factor=10.0,
            line_search_schedule=("bt", "l2", "cp", "basic"),
            anisotropy_target_ratio=3.0,
            anisotropy_blend=0.5,
        ),
    )

    result = run_robin_kappa_flux_curve_inference(request)

    print("\n=== Robin Kappa Inference (Adjoint Gradient) ===")
    print(f"True kappa: {request.true_value}")
    print(f"Initial guess: {request.initial_guess}")
    print(f"Estimated kappa: {result.best_kappa.tolist()}")
    print(f"Final objective value: {result.best_loss:.12e}")
    print(f"SciPy success: {result.optimization_success}")
    print(f"SciPy message: {result.optimization_message}")


if __name__ == "__main__":
    main()
