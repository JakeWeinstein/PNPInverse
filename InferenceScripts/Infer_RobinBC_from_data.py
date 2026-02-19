"""Unified-interface example: infer Robin boundary transfer coefficient ``kappa``."""

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
    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=1e-1,
        t_end=1,
        z_vals=[1, -1],
        d_vals=[1.0, 1.0],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[0.1, 0.1],
        phi0=5.0,
        solver_options=build_solver_options(),
    )

    request = InferenceRequest(
        adapter=ForwardSolverAdapter.from_module_path(
            "Utils.robin_forsolve", solve_function_name="forsolve_robin"
        ),
        target=build_default_target_registry()["robin_kappa"],
        base_solver_params=base_solver_params,
        true_value=[1, 5],
        initial_guess=[10.0, 10.0],
        noise_percent=10.0,
        seed=20260219,
        optimizer_method="L-BFGS-B",
        optimizer_options={
            "disp": True,
            "maxiter": 300,
            # Tighten L-BFGS-B stopping so it does not exit early in flat regions.
            "ftol": 1e-15,
            "gtol": 1e-10,
            "maxls": 50,
        },
        tolerance=1e-12,
        fit_to_noisy_data=True,
        recovery_attempts=100,
    )

    result = run_inverse_inference(request)

    print("=== Robin kappa Inference (Unified Interface) ===")
    print(f"True kappa: {request.true_value}")
    print(f"Initial guess: {request.initial_guess}")
    print(f"Estimated kappa: {result.estimate}")
    print(f"Final objective value: {result.objective_value:.12e}")


if __name__ == "__main__":
    main()
