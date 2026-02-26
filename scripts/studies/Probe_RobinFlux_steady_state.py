"""Probe how long Robin simulations need to reach flux steady state.

This script is the first step in the experimental-style pipeline:
1) pick a coarse ``dt`` (default 1e-1),
2) define a flux-based steady-state metric,
3) measure required step/time horizon across a phi_applied range.

Run:
    python scripts/studies/Probe_RobinFlux_steady_state.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import List

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

from Inverse import build_default_solver_params
from Forward.steady_state import (
    SteadyStateConfig,
    configure_robin_solver_params,
    solve_to_steady_state_for_phi_applied,
)


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


def summarize_probe(steps: List[int], dt: float) -> None:
    """Print conservative horizon suggestions from successful probe points."""
    if not steps:
        print("No successful steady-state points; cannot suggest horizon.")
        return

    max_steps = int(max(steps))
    median_steps = int(np.median(np.asarray(steps, dtype=float)))
    # Use a modest safety factor so production sweeps have headroom.
    recommended_steps = int(np.ceil(1.5 * max_steps))
    print("\nSuggested steady-state horizon for production sweeps:")
    print(f"  max successful steps: {max_steps}")
    print(f"  median successful steps: {median_steps}")
    print(f"  recommended max_steps: {recommended_steps}")
    print(f"  recommended t_end ~= {recommended_steps * dt:.6f}")


def main() -> None:
    # Requested probe time step.
    dt = 1e-2
    max_it_schedule = [100, 200, 400, 800]

    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=dt,
        t_end=20.0,  # Not used by steady-state stepping loop, kept for completeness.
        z_vals=[1, -1],
        d_vals=[1.0, 1.0],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[0.1, 0.1],
        phi0=0.05,
        solver_options=build_solver_options(),
    )

    # Probe a moderate voltage range.
    phi_applied_values = np.linspace(0.0, 0.2, 6)

    # Steady-state metric: boundary-flux change between consecutive steps.
    steady = SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        # Smaller dt requires more steps to reach the same physical time horizon.
        max_steps=1200,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    print("=== Robin Steady-State Probe (dt = 1e-2) ===")
    print(f"snes_max_it retry schedule: {max_it_schedule}")
    print("Steady-state criterion:")
    print(
        "  max_i |J_i(n)-J_i(n-1)| / max(|J_i(n)|, |J_i(n-1)|, abs_tol) <= rel_tol "
        "or max_i |J_i(n)-J_i(n-1)| <= abs_tol"
    )
    print(f"  rel_tol={steady.relative_tolerance}, abs_tol={steady.absolute_tolerance}")
    print(f"  consecutive_steps={steady.consecutive_steps}, max_steps={steady.max_steps}")
    print()

    results = []
    used_max_its: List[int] = []
    attempts_per_phi_applied: List[int] = []
    for phi_applied in phi_applied_values.tolist():
        best_result = None
        chosen_max_it = int(max_it_schedule[-1])
        attempt_count = 0
        for max_it in max_it_schedule:
            attempt_count += 1
            trial_params = configure_robin_solver_params(
                base_solver_params,
                phi_applied=float(phi_applied),
                kappa_values=[0.8, 0.8],
            )
            solver_opts = trial_params[10]
            if not isinstance(solver_opts, dict):
                raise ValueError("Expected solver_params[10] to be a dict.")
            solver_opts["snes_max_it"] = int(max_it)

            trial = solve_to_steady_state_for_phi_applied(
                trial_params,
                steady=steady,
                blob_initial_condition=False,
            )
            best_result = trial
            chosen_max_it = int(max_it)
            if trial.converged:
                break
        if best_result is None:
            raise RuntimeError("Unexpected empty probe result.")
        results.append(best_result)
        used_max_its.append(chosen_max_it)
        attempts_per_phi_applied.append(attempt_count)

    print(
        "phi_applied converged  steps  final_time  used_max_it  attempts  "
        "steady_rel_change   steady_abs_change   observed_flux"
    )
    successful_steps: List[int] = []
    for idx, r in enumerate(results):
        if r.converged:
            successful_steps.append(int(r.steps_taken))
        print(
            f"{r.phi_applied:>9.4f}  "
            f"{str(r.converged):>9}  "
            f"{r.steps_taken:>5d}  "
            f"{r.final_time:>10.4f}  "
            f"{used_max_its[idx]:>11d}  "
            f"{attempts_per_phi_applied[idx]:>8d}  "
            f"{(r.final_relative_change if r.final_relative_change is not None else float('nan')):>17.6e}  "
            f"{(r.final_absolute_change if r.final_absolute_change is not None else float('nan')):>17.6e}  "
            f"{r.observed_flux:>13.6f}"
        )
        if not r.converged:
            print(f"   reason: {r.failure_reason}")

    summarize_probe(successful_steps, dt)

    out_dir = os.path.join("StudyResults", "robin_flux_experiment")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "steady_state_probe_dt_1e-2_maxit_retry.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phi_applied",
                "converged",
                "steps_taken",
                "final_time",
                "used_snes_max_it",
                "attempts",
                "final_relative_change",
                "final_absolute_change",
                "observed_flux",
                "species_flux",
                "failure_reason",
            ]
        )
        for idx, r in enumerate(results):
            writer.writerow(
                [
                    f"{float(r.phi_applied):.16g}",
                    int(bool(r.converged)),
                    int(r.steps_taken),
                    f"{float(r.final_time):.16g}",
                    int(used_max_its[idx]),
                    int(attempts_per_phi_applied[idx]),
                    ""
                    if r.final_relative_change is None
                    else f"{float(r.final_relative_change):.16g}",
                    ""
                    if r.final_absolute_change is None
                    else f"{float(r.final_absolute_change):.16g}",
                    f"{float(r.observed_flux):.16g}",
                    ";".join(f"{float(v):.16g}" for v in r.species_flux),
                    str(r.failure_reason),
                ]
            )
    print(f"\nSaved probe table to: {csv_path}")


if __name__ == "__main__":
    main()
