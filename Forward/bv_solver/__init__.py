"""Butler-Volmer BC PNP forward solver with full nondimensionalization.

Supports two BV configuration modes:

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

**Multi-reaction mode** (new — for coupled reactions)::

    "bv_bc": {
        "reactions": [
            {
                "k0": 2.4e-8,
                "alpha": 0.627,
                "cathodic_species": 0,    # species consumed
                "anodic_species": 1,      # species produced (None if irreversible)
                "c_ref": 1.0,            # reference conc for anodic term
                "stoichiometry": [-1, +1],
                "n_electrons": 2,
                "reversible": True,
            },
            ...
        ],
        "electrode_marker": 1,
        "concentration_marker": 3,
        "ground_marker": 3,
    }

Multi-reaction mode enables coupled chemistry — e.g. O₂→H₂O₂ (R₁)
and H₂O₂→H₂O (R₂) share the H₂O₂ species.  Each reaction j contributes::

    R_j = k0_j · [c_cat · exp(−α_j · η̂)  −  c_ref_j · exp((1−α_j) · η̂)]

to the weak form via the stoichiometry matrix: for species i,
``F_res -= s_ij · R_j · v_i · ds(electrode)``.

The solved ``ctx["bv_rate_exprs"]`` list stores UFL expressions for each R_j,
allowing post-solve current-density computation.

Convergence strategies
----------------------
``clip_exponent``, ``regularize_concentration``, ``use_eta_in_bv`` —
see ``bv_convergence`` config block.

Public API
----------
build_context(solver_params) → dict
build_forms(ctx, solver_params) → dict
set_initial_conditions(ctx, solver_params, blob=False) → None
forsolve_bv(ctx, solver_params, print_interval=100) → Function
"""

from __future__ import annotations

from Forward.bv_solver.mesh import make_graded_interval_mesh, make_graded_rectangle_mesh
from Forward.bv_solver.config import _get_bv_cfg, _get_bv_convergence_cfg, _get_bv_reactions_cfg
from Forward.bv_solver.nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform
from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
from Forward.bv_solver.solvers import forsolve_bv, solve_bv_with_continuation, solve_bv_with_ptc, solve_bv_with_charge_continuation
from Forward.bv_solver.grid_charge_continuation import (
    solve_grid_with_charge_continuation,
    GridChargeContinuationResult,
    GridPointResult,
)
