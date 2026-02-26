"""Forward PNP solver package.

This package contains all forward-solver modules for the PNP system:

- :mod:`Forward.params` — ``SolverParams`` named-attribute list
- :mod:`Forward.dirichlet_solver` — Dirichlet-BC forward solver
- :mod:`Forward.robin_solver` — Robin-BC forward solver
- :mod:`Forward.bv_solver` — Butler-Volmer-BC forward solver
- :mod:`Forward.steady_state` — steady-state sweep utilities
- :mod:`Forward.noise` — noise injection for synthetic data
- :mod:`Forward.plotter` — visualization utilities

Public API (re-exported here for convenience)::

    from Forward import SolverParams
    from Forward import build_context, build_forms, set_initial_conditions, forsolve
    from Forward import SteadyStateConfig, SteadyStateResult
    from Forward import solve_to_steady_state_for_phi_applied
    from Forward import sweep_phi_applied_steady_flux
    from Forward import generate_noisy_data, generate_noisy_data_robin
    from Forward import plot_solutions, create_animations
"""

from Forward.params import SolverParams

# Dirichlet solver
from Forward.dirichlet_solver import (
    build_context,
    build_forms,
    set_initial_conditions,
    forsolve,
)

# Robin solver (also exports forsolve_robin alias)
from Forward.robin_solver import (
    build_context as build_context_robin,
    build_forms as build_forms_robin,
    set_initial_conditions as set_initial_conditions_robin,
    forsolve as forsolve_robin_fn,
    forsolve_robin,
)

# Butler-Volmer solver
from Forward.bv_solver import (
    build_context as build_context_bv,
    build_forms as build_forms_bv,
    set_initial_conditions as set_initial_conditions_bv,
    forsolve_bv,
    solve_bv_with_continuation,
    make_graded_interval_mesh,
    make_graded_rectangle_mesh,
)

# Steady-state helpers
from Forward.steady_state import (
    SteadyStateConfig,
    SteadyStateResult,
    configure_robin_solver_params,
    compute_species_flux_on_robin_boundary,
    observed_flux_from_species_flux,
    solve_to_steady_state_for_phi_applied,
    sweep_phi_applied_steady_flux,
    add_percent_noise,
    write_phi_applied_flux_csv,
    read_phi_applied_flux_csv,
    results_to_flux_array,
    all_results_converged,
    # backward-compat aliases
    solve_to_steady_state_for_phi0,
    sweep_phi0_steady_flux,
    write_phi0_flux_csv,
    read_phi0_flux_csv,
)

# Noise injection
from Forward.noise import generate_noisy_data, generate_noisy_data_robin

# Visualization
from Forward.plotter import plot_solutions, create_animations, RENDERS_DIR

__all__ = [
    # params
    "SolverParams",
    # dirichlet solver
    "build_context",
    "build_forms",
    "set_initial_conditions",
    "forsolve",
    # robin solver
    "build_context_robin",
    "build_forms_robin",
    "set_initial_conditions_robin",
    "forsolve_robin_fn",
    "forsolve_robin",
    # BV solver
    "build_context_bv",
    "build_forms_bv",
    "set_initial_conditions_bv",
    "forsolve_bv",
    "solve_bv_with_continuation",
    "make_graded_interval_mesh",
    "make_graded_rectangle_mesh",
    # steady state
    "SteadyStateConfig",
    "SteadyStateResult",
    "configure_robin_solver_params",
    "compute_species_flux_on_robin_boundary",
    "observed_flux_from_species_flux",
    "solve_to_steady_state_for_phi_applied",
    "sweep_phi_applied_steady_flux",
    "add_percent_noise",
    "write_phi_applied_flux_csv",
    "read_phi_applied_flux_csv",
    "results_to_flux_array",
    "all_results_converged",
    "solve_to_steady_state_for_phi0",
    "sweep_phi0_steady_flux",
    "write_phi0_flux_csv",
    "read_phi0_flux_csv",
    # noise
    "generate_noisy_data",
    "generate_noisy_data_robin",
    # plotter
    "plot_solutions",
    "create_animations",
    "RENDERS_DIR",
]
