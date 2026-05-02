"""Forward-solve helpers for the Butler-Volmer PNP solver.

After the May 2026 cleanup, the only helper that survives here is the
formulation-agnostic ``_clone_params_with_phi``.  The continuation-style
solvers (``forsolve_bv``, ``solve_bv_with_continuation``,
``solve_bv_with_ptc``, ``solve_bv_with_charge_continuation``) were
removed alongside the legacy concentration backend; the production
stack uses ``Forward.bv_solver.grid_per_voltage`` (cold-ramp + warm-walk
fallback) directly.
"""

from __future__ import annotations


def _clone_params_with_phi(solver_params, *, phi_applied: float):
    """Return a new SolverParams-like object with ``phi_applied`` replaced."""
    if hasattr(solver_params, "with_phi_applied"):
        return solver_params.with_phi_applied(float(phi_applied))
    lst = list(solver_params)
    lst[7] = float(phi_applied)
    return lst
