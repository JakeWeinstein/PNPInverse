"""Unified-interface example: infer Dirichlet ``phi0`` from synthetic data.

This script demonstrates the new modular inverse workflow by configuring:
1. a forward solver adapter (``Utils.forsolve``)
2. a parameter target (``dirichlet_phi0``)
3. true value, noise level, and initial guess

Run:
    python InferenceScripts/Infer_DirichletBC_from_data.py
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(_THIS_DIR)
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

from UnifiedInverse import (
    ForwardSolverAdapter,
    InferenceRequest,
    build_default_solver_params,
    build_default_target_registry,
    run_inverse_inference,
)

# Keep Firedrake cache paths writable in sandboxed/restricted environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def build_solver_options():
    """Solver options shared across data generation and inverse solve."""
    return {
        "snes_type": "newtonls",
        "snes_max_it": 100,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_linesearch_type": "bt",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


def main() -> None:
    # Base solver setup. The target mechanism will overwrite phi0 with
    # (a) true value for data generation and (b) initial guess for inversion.
    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=1e-2,
        t_end=0.1,
        z_vals=[1, -1],
        d_vals=[1.0, 1.0],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[0.1, 0.1],
        phi0=1.0,
        solver_options=build_solver_options(),
    )

    adapter = ForwardSolverAdapter.from_module_path("Utils.forsolve")
    target = build_default_target_registry()["dirichlet_phi0"]

    request = InferenceRequest(
        adapter=adapter,
        target=target,
        base_solver_params=base_solver_params,
        true_value=1.0,
        initial_guess=10.0,
        noise_percent=10.0,
        seed=20260218,
        optimizer_method="L-BFGS-B",
        optimizer_options={"disp": True, "maxiter": 80},
        tolerance=1e-8,
        # For realistic inversion, fit against noisy synthetic data.
        fit_to_noisy_data=True,
        blob_initial_condition=True,
        print_interval_data=100,
        print_interval_inverse=100,
    )

    result = run_inverse_inference(request)

    print("=== Dirichlet phi0 Inference (Unified Interface) ===")
    print(f"True phi0: {request.true_value}")
    print(f"Initial guess: {request.initial_guess}")
    print(f"Estimated phi0: {result.estimate}")
    print(f"Final objective value: {result.objective_value:.12e}")


if __name__ == "__main__":
    main()
