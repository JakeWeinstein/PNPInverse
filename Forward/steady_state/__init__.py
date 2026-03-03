"""Utilities for Robin-boundary flux experiments and flux-curve inversion.

This module is designed for the experimental workflow where:
1) the controlled input is applied voltage (``phi_applied``), and
2) the measured output is steady-state Robin boundary flux.

The helpers here provide:
- robust stepping to a flux-defined steady state
- voltage sweep generation (phi_applied vs steady-state flux)
- CSV serialization helpers for experimental/synthetic datasets

The code intentionally avoids firedrake-adjoint optimization machinery because
the objective is a multi-solve, steady-state curve mismatch rather than a
single terminal-state functional.
"""

from __future__ import annotations

# --- common types, I/O, noise, helpers ---
from .common import (
    SteadyStateConfig,
    SteadyStateResult,
    observed_flux_from_species_flux,
    add_percent_noise,
    write_phi_applied_flux_csv,
    read_phi_applied_flux_csv,
    results_to_flux_array,
    all_results_converged,
    _maybe_stop_annotating,
)

# --- Robin-specific steady-state solve + sweep ---
from .robin import (
    configure_robin_solver_params,
    compute_species_flux_on_robin_boundary,
    solve_to_steady_state_for_phi_applied,
    sweep_phi_applied_steady_flux,
)

# --- BV-specific steady-state solve + sweep ---
from .bv import (
    configure_bv_solver_params,
    compute_bv_reaction_rates,
    compute_bv_current_density,
    solve_bv_to_steady_state_for_phi_applied,
    sweep_phi_applied_steady_bv_flux,
)


__all__ = [
    # common
    "SteadyStateConfig",
    "SteadyStateResult",
    "observed_flux_from_species_flux",
    "add_percent_noise",
    "write_phi_applied_flux_csv",
    "read_phi_applied_flux_csv",
    "results_to_flux_array",
    "all_results_converged",
    "_maybe_stop_annotating",
    # robin
    "configure_robin_solver_params",
    "compute_species_flux_on_robin_boundary",
    "solve_to_steady_state_for_phi_applied",
    "sweep_phi_applied_steady_flux",
    # bv
    "configure_bv_solver_params",
    "compute_bv_reaction_rates",
    "compute_bv_current_density",
    "solve_bv_to_steady_state_for_phi_applied",
    "sweep_phi_applied_steady_bv_flux",
]
