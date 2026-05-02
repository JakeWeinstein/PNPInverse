"""Butler-Volmer BC PNP forward solver with full nondimensionalization.

This module ships the unified BV forward solver covering the production
configurations exercised in ``writeups/WeekOfApr27/PNP Inverse Solver
Revised.tex``:

  * **Concentration formulation** (legacy): primary unknown is ``c_i``.
    Selected by ``params['bv_convergence']['formulation'] = 'concentration'``
    (default for backward compatibility with v13 inverse scripts).

  * **Log-concentration formulation** (production): primary unknown is
    ``u_i = ln(c_i)``.  Enforces positivity, removes the surface
    concentration clip in BV evaluation, and substantially extends the
    converged voltage window.  Selected by
    ``params['bv_convergence']['formulation'] = 'logc'``.

  * **Log-rate Butler-Volmer**: ``params['bv_convergence']['bv_log_rate']
    = True``.  Combines additively with the log-c primary variable to
    eliminate phantom R2 sinks at high anodic eta.

  * **Analytic Boltzmann counterion** (Poisson--Boltzmann--Nernst--Planck
    reduction): ``params['bv_bc']['boltzmann_counterions'] = [{...}]``.
    See :mod:`Forward.bv_solver.boltzmann` for the residual contribution.

The same call sequence -- :func:`build_context`, :func:`build_forms`,
:func:`set_initial_conditions`, then the standard solver helpers --
works for every combination above.  The dispatcher (in
:mod:`Forward.bv_solver.dispatch`) reads
``params['bv_convergence']['formulation']`` once at context construction
to route to the right backend (``forms.py`` or ``forms_logc.py``).

BV configuration modes
----------------------

**Per-species mode** (legacy, backward-compatible)::

    "bv_bc": {
        "k0":               [2.4e-8, 2.4e-8],   # one per species
        "alpha":            [0.627, 0.373],
        "stoichiometry":    [-1, +1],
        "c_ref":            [0.5, 0.5],
        "E_eq_v":           0.0,
        "electrode_marker":      1,
        "concentration_marker":  3,
        "ground_marker":         3,
    }

**Multi-reaction mode** (preferred — supports coupled R1+R2)::

    "bv_bc": {
        "reactions": [
            {
                "k0": 2.4e-8,
                "alpha": 0.627,
                "cathodic_species": 0,    # species consumed
                "anodic_species": 1,      # species produced (None if irreversible)
                "c_ref": 1.0,             # reference conc for anodic term
                "stoichiometry": [-1, +1],
                "n_electrons": 2,
                "reversible": True,
            },
            ...
        ],
        # Optional: analytic Boltzmann counterions in Poisson (PBNP).
        "boltzmann_counterions": [
            {"z": -1, "c_bulk_nondim": 0.1, "phi_clamp": 50.0},
        ],
        "electrode_marker": 1,
        "concentration_marker": 3,
        "ground_marker": 3,
    }

Public API
----------
build_context(solver_params, *, mesh=None) -> dict
    Dispatcher: builds a context for the formulation in
    ``solver_params[10]['bv_convergence']['formulation']``.
build_forms(ctx, solver_params) -> dict
    Dispatcher: assembles the weak forms (and Boltzmann residual when
    configured).
set_initial_conditions(ctx, solver_params, *, blob=False) -> None
    Dispatcher: sets initial conditions in the right primary variable.

The legacy direct-import names (``build_context_logc`` etc.) remain
available for scripts that explicitly want one backend; they bypass the
dispatcher.
"""

from __future__ import annotations

from Forward.bv_solver.mesh import make_graded_interval_mesh, make_graded_rectangle_mesh
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
from Forward.bv_solver.boltzmann import add_boltzmann_counterion_residual
from Forward.bv_solver.dispatch import (
    build_context,
    build_forms,
    set_initial_conditions,
    build_context_logc,
    build_forms_logc,
    set_initial_conditions_logc,
)
from Forward.bv_solver.solvers import (
    forsolve_bv,
    solve_bv_with_continuation,
    solve_bv_with_ptc,
    solve_bv_with_charge_continuation,
)
from Forward.bv_solver.grid_charge_continuation import (
    solve_grid_with_charge_continuation,
    GridChargeContinuationResult,
    GridPointResult,
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
    "add_boltzmann_counterion_residual",
    "forsolve_bv",
    "solve_bv_with_continuation",
    "solve_bv_with_ptc",
    "solve_bv_with_charge_continuation",
    "solve_grid_with_charge_continuation",
    "GridChargeContinuationResult",
    "GridPointResult",
    "solve_grid_per_voltage_cold_with_warm_fallback",
    "PerVoltageContinuationResult",
    "PerVoltagePointResult",
]
