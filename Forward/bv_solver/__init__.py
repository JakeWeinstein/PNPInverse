"""Butler-Volmer BC PNP forward solver.

The production stack is:

  * **3 dynamic species** (O2, H2O2, H+) with **log-concentration primary
    variables** ``u_i = ln(c_i)``: enforces positivity, removes the
    surface-concentration clip in BV evaluation, extends the converged
    voltage window.
  * **Analytic Boltzmann counterion** for ClO4- in the Poisson residual
    (Poisson-Boltzmann-Nernst-Planck reduction).
  * **Log-rate Butler-Volmer**: eliminates phantom R2 sinks at high
    anodic eta.

The stack is selected through three config flags::

    params['bv_convergence']['formulation'] = 'logc'
    params['bv_convergence']['bv_log_rate'] = True
    params['bv_bc']['boltzmann_counterions'] = [{...}]

See ``writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`` for the
formulation rationale and ``docs/bv_solver_unified_api.md`` for usage.

The canonical I-V driver is
``solve_grid_per_voltage_cold_with_warm_fallback`` (C+D continuation
orchestrator).

BV reaction config example (multi-reaction mode)::

    "bv_bc": {
        "reactions": [
            {
                "k0": 2.4e-8,
                "alpha": 0.627,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": 1.0,
                "stoichiometry": [-1, +1],
                "n_electrons": 2,
                "reversible": True,
            },
            ...
        ],
        "boltzmann_counterions": [
            {"z": -1, "c_bulk_nondim": 0.1, "phi_clamp": 50.0},
        ],
        "electrode_marker": 1,
        "concentration_marker": 3,
        "ground_marker": 3,
    }
"""

from __future__ import annotations

from Forward.bv_solver.mesh import (
    make_graded_interval_mesh,
    make_graded_rectangle_mesh,
)
from Forward.bv_solver.config import (
    _get_bv_cfg,
    _get_bv_convergence_cfg,
    _get_bv_reactions_cfg,
    _get_bv_boltzmann_counterions_cfg,
)
from Forward.bv_solver.nondim import (
    _add_bv_scaling_to_transform,
    _add_bv_reactions_scaling_to_transform,
)
from Forward.bv_solver.boltzmann import (
    add_boltzmann_counterion_residual,
    build_steric_boltzmann_expressions,
    StericBoltzmannBundle,
)
from Forward.bv_solver.dispatch import (
    build_context,
    build_forms,
    set_initial_conditions,
    build_context_logc,
    build_forms_logc,
    set_initial_conditions_logc,
    set_initial_conditions_debye_boltzmann_logc,
    # Experimental muh backend (formulation="logc_muh"); see forms_logc_muh.py.
    build_context_logc_muh,
    build_forms_logc_muh,
    set_initial_conditions_logc_muh,
    set_initial_conditions_debye_boltzmann_logc_muh,
)
from Forward.bv_solver.grid_per_voltage import (
    solve_grid_per_voltage_cold_with_warm_fallback,
    PerVoltageContinuationResult,
    PerVoltagePointResult,
)


__all__ = [
    "make_graded_interval_mesh",
    "make_graded_rectangle_mesh",
    "build_context",
    "build_forms",
    "set_initial_conditions",
    "build_context_logc",
    "build_forms_logc",
    "set_initial_conditions_logc",
    "set_initial_conditions_debye_boltzmann_logc",
    "build_context_logc_muh",
    "build_forms_logc_muh",
    "set_initial_conditions_logc_muh",
    "set_initial_conditions_debye_boltzmann_logc_muh",
    "add_boltzmann_counterion_residual",
    "build_steric_boltzmann_expressions",
    "StericBoltzmannBundle",
    "solve_grid_per_voltage_cold_with_warm_fallback",
    "PerVoltageContinuationResult",
    "PerVoltagePointResult",
]
