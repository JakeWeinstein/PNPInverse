#!/usr/bin/env python3
"""Unified interface for inverse inference across compatible PNP forward solvers.

Examples
--------
Infer Dirichlet ``phi0`` with default Dirichlet forward solver:

    python InferenceScripts/Infer_parameter_from_data.py \
        --target dirichlet_phi0 \
        --true-value 1.0 \
        --initial-guess 10.0

Infer diffusion coefficients using a Robin forward solver module:

    python InferenceScripts/Infer_parameter_from_data.py \
        --solver-module Utils.robin_forsolve \
        --solve-function forsolve_robin \
        --target diffusion \
        --true-value 1.0,3.0 \
        --initial-guess 10.0,10.0

Infer Robin ``kappa`` values:

    python InferenceScripts/Infer_parameter_from_data.py \
        --solver-module Utils.robin_forsolve \
        --solve-function forsolve_robin \
        --target robin_kappa \
        --params-json '{"robin_bc": {"c_inf": [0.01, 0.01], "electrode_marker": 1, "concentration_marker": 3, "ground_marker": 3}}' \
        --true-value 0.8,0.8 \
        --initial-guess 1.0,1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

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


def _build_common_solver_options() -> Dict[str, Any]:
    """Return the shared nonlinear/PETSc options used in examples and studies."""
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


def _parse_scalar_or_vector(text: str):
    """Parse CLI value as float or float list.

    Accepted formats:
    - ``1.0``
    - ``1.0,2.0``
    - ``[1.0, 2.0]``
    """
    raw = text.strip()
    if raw.startswith("["):
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [float(v) for v in loaded]
        return float(loaded)

    if "," in raw:
        return [float(v.strip()) for v in raw.split(",") if v.strip()]

    return float(raw)


def _parse_vector(text: str, n_species: int, *, name: str) -> List[float]:
    """Parse list-like CLI argument and validate species length."""
    value = _parse_scalar_or_vector(text)
    if isinstance(value, list):
        if len(value) != n_species:
            raise ValueError(
                f"{name} must have length n_species ({n_species}); got {len(value)}"
            )
        return value

    # Scalar broadcast for convenience.
    return [float(value) for _ in range(n_species)]


def _default_solver_module_for_target(target_key: str) -> str:
    """Pick default forward solver module by target type."""
    if target_key == "robin_kappa":
        return "Utils.robin_forsolve"
    return "Utils.forsolve"


def _default_solve_function_for_module(module_path: str) -> Optional[str]:
    """Return an explicit solve function for known non-default modules."""
    if module_path == "Utils.robin_forsolve":
        return "forsolve_robin"
    return None


def _build_parser() -> argparse.ArgumentParser:
    registry = build_default_target_registry()
    target_choices = sorted(registry.keys())

    parser = argparse.ArgumentParser(
        description=(
            "Unified inverse interface: plug in a forsolve module, choose a "
            "parameter target, set true value/noise/initial guess, and run inference."
        )
    )

    parser.add_argument(
        "--target",
        choices=target_choices,
        required=True,
        help="Parameter target to infer.",
    )
    parser.add_argument(
        "--solver-module",
        default=None,
        help=(
            "Import path for forward solver module implementing the adapter contract "
            "(build_context/build_forms/set_initial_conditions/forsolve)."
        ),
    )
    parser.add_argument(
        "--solve-function",
        default=None,
        help="Optional explicit solve function name (e.g. forsolve_robin).",
    )

    parser.add_argument("--true-value", required=True, help="True value (scalar or vector).")
    parser.add_argument(
        "--initial-guess", required=True, help="Initial guess value (scalar or vector)."
    )
    parser.add_argument("--noise-percent", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument(
        "--fit-to",
        choices=["noisy", "clean"],
        default="noisy",
        help="Choose whether optimization fits noisy or clean generated data.",
    )

    parser.add_argument("--optimizer", default="L-BFGS-B")
    parser.add_argument("--optimizer-maxiter", type=int, default=100)
    parser.add_argument(
        "--optimizer-ftol",
        type=float,
        default=None,
        help="Optional optimizer ftol (useful for tighter L-BFGS-B stopping).",
    )
    parser.add_argument(
        "--optimizer-gtol",
        type=float,
        default=None,
        help="Optional optimizer gtol (useful for tighter L-BFGS-B stopping).",
    )
    parser.add_argument(
        "--optimizer-maxls",
        type=int,
        default=None,
        help="Optional optimizer max line-search steps (L-BFGS-B).",
    )
    parser.add_argument("--tol", type=float, default=1e-8)

    parser.add_argument("--n-species", type=int, default=2)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--t-end", type=float, default=0.1)
    parser.add_argument("--z-values", default="1,-1")
    parser.add_argument("--d-values", default="1.0,1.0")
    parser.add_argument("--a-values", default="0.0,0.0")
    parser.add_argument("--c0-values", default="0.1,0.1")
    parser.add_argument("--phi-applied", type=float, default=0.05)
    parser.add_argument("--phi0", type=float, default=1.0)

    parser.add_argument(
        "--params-json",
        default=None,
        help=(
            "Optional JSON object merged into solver options dict. Use this for solver-"
            "specific settings like robin_bc."
        ),
    )

    parser.add_argument(
        "--print-interval-data",
        type=int,
        default=100,
        help="Forward-step print interval during synthetic data generation.",
    )
    parser.add_argument(
        "--print-interval-inverse",
        type=int,
        default=100,
        help="Forward-step print interval during taped objective evaluations.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    # Firedrake/PETSc parse process arguments too. After argparse consumes this
    # script's flags, keep only argv[0] so PETSc does not warn about unknown args.
    sys.argv = [sys.argv[0]]

    registry = build_default_target_registry()
    target = registry[args.target]

    solver_module = args.solver_module or _default_solver_module_for_target(args.target)
    solve_function = args.solve_function
    if solve_function is None:
        solve_function = _default_solve_function_for_module(solver_module)

    adapter = ForwardSolverAdapter.from_module_path(
        solver_module,
        solve_function_name=solve_function,
    )

    solver_options = _build_common_solver_options()
    if args.params_json:
        loaded = json.loads(args.params_json)
        if not isinstance(loaded, dict):
            raise ValueError("--params-json must decode to a JSON object/dict.")
        # Shallow merge keeps defaults while allowing target-specific overrides.
        solver_options.update(loaded)

    n_species = int(args.n_species)
    z_vals = _parse_vector(args.z_values, n_species, name="z-values")
    d_vals = _parse_vector(args.d_values, n_species, name="d-values")
    a_vals = _parse_vector(args.a_values, n_species, name="a-values")
    c0_vals = _parse_vector(args.c0_values, n_species, name="c0-values")

    base_solver_params = build_default_solver_params(
        n_species=n_species,
        order=int(args.order),
        dt=float(args.dt),
        t_end=float(args.t_end),
        z_vals=z_vals,
        d_vals=d_vals,
        a_vals=a_vals,
        phi_applied=float(args.phi_applied),
        c0_vals=c0_vals,
        phi0=float(args.phi0),
        solver_options=solver_options,
    )

    true_value = _parse_scalar_or_vector(args.true_value)
    initial_guess = _parse_scalar_or_vector(args.initial_guess)

    optimizer_options: Dict[str, Any] = {
        "disp": True,
        "maxiter": int(args.optimizer_maxiter),
    }
    if args.optimizer_ftol is not None:
        optimizer_options["ftol"] = float(args.optimizer_ftol)
    if args.optimizer_gtol is not None:
        optimizer_options["gtol"] = float(args.optimizer_gtol)
    if args.optimizer_maxls is not None:
        optimizer_options["maxls"] = int(args.optimizer_maxls)

    request = InferenceRequest(
        adapter=adapter,
        target=target,
        base_solver_params=base_solver_params,
        true_value=true_value,
        initial_guess=initial_guess,
        noise_percent=float(args.noise_percent),
        seed=int(args.seed),
        optimizer_method=args.optimizer,
        optimizer_options=optimizer_options,
        tolerance=float(args.tol),
        fit_to_noisy_data=(args.fit_to == "noisy"),
        blob_initial_condition=True,
        print_interval_data=int(args.print_interval_data),
        print_interval_inverse=int(args.print_interval_inverse),
    )

    result = run_inverse_inference(request)

    print("=== Unified Inference Summary ===")
    print(f"solver module: {solver_module}")
    print(f"solve function: {adapter.solve_function_name}")
    print(f"target: {target.key}")
    print(f"true value: {true_value}")
    print(f"initial guess: {initial_guess}")
    print(f"estimate: {result.estimate}")
    print(f"objective value: {result.objective_value:.12e}")


if __name__ == "__main__":
    main()
