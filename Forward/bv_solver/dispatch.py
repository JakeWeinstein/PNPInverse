"""Formulation dispatcher for the BV-PNP forward solver.

The BV-PNP solver supports two backends:

  * ``forms.py``         — primary unknowns are concentrations ``c_i``.
  * ``forms_logc.py``    — primary unknowns are ``u_i = ln(c_i)``.

Both backends accept the same 11-tuple ``solver_params`` and produce the
same context dict shape, so they are fully interchangeable behind a
single function dispatcher.  The choice is encoded in
``solver_params[10]['bv_convergence']['formulation']`` (default
``"concentration"`` for backward compatibility with v13/v15/v16 inverse
scripts).

Optional features (independent of formulation):

  * ``bv_convergence.bv_log_rate=True`` — log-rate BV evaluation
    (Change 3 of ``writeups/WeekOfApr27/PNP Inverse Solver Revised.tex``).
  * ``bv_bc.boltzmann_counterions=[...]`` — analytic Boltzmann
    counterions in Poisson (Change 1 of the writeup).  See
    :mod:`Forward.bv_solver.boltzmann`.

This module is the implementation that the package ``__init__`` re-exports
under the canonical names ``build_context`` / ``build_forms`` /
``set_initial_conditions``.  It exists as its own module so that other
modules inside the package (``solvers.py``, ``grid_charge_continuation.py``)
can import the dispatcher without triggering a circular import via the
package init.
"""

from __future__ import annotations

from typing import Any

from .forms import (
    build_context as _build_context_concentration,
    build_forms as _build_forms_concentration,
    set_initial_conditions as _set_initial_conditions_concentration,
)
from .forms_logc import (
    build_context_logc,
    build_forms_logc,
    set_initial_conditions_logc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_dict(solver_params: Any) -> dict:
    """Return the params dict (``solver_params[10]``) regardless of input shape."""
    if isinstance(solver_params, dict):
        return solver_params
    if hasattr(solver_params, "solver_options"):
        opts = solver_params.solver_options
        return opts if isinstance(opts, dict) else {}
    try:
        opts = solver_params[10]
    except (IndexError, TypeError, KeyError):
        return {}
    return opts if isinstance(opts, dict) else {}


def _get_formulation(solver_params: Any) -> str:
    """Read ``params['bv_convergence']['formulation']`` (default 'concentration')."""
    p = _params_dict(solver_params)
    bv_conv = p.get("bv_convergence", {}) if isinstance(p, dict) else {}
    if not isinstance(bv_conv, dict):
        return "concentration"
    return str(bv_conv.get("formulation", "concentration")).strip().lower()


# ---------------------------------------------------------------------------
# Dispatcher API
# ---------------------------------------------------------------------------

def build_context(solver_params: Any, *, mesh: Any = None) -> dict:
    """Build mesh + function spaces using the formulation selected in params.

    Default is ``"concentration"`` so existing v13/v15/v16 scripts keep
    their behavior.
    """
    if _get_formulation(solver_params) == "logc":
        return build_context_logc(solver_params, mesh=mesh)
    return _build_context_concentration(solver_params, mesh=mesh)


def build_forms(ctx: dict, solver_params: Any) -> dict:
    """Assemble weak forms using the formulation selected by ``build_context``.

    The formulation is re-read from ``solver_params`` each call, but a
    mismatch with the context's primary variables raises a clear error.
    """
    formulation = _get_formulation(solver_params)
    is_logc_ctx = bool(ctx.get("logc_transform", False))
    if formulation == "logc":
        return build_forms_logc(ctx, solver_params)
    if is_logc_ctx:
        raise ValueError(
            "build_forms: params requests 'concentration' formulation but "
            "ctx was produced by build_context_logc.  Use a single formulation "
            "throughout the call chain."
        )
    return _build_forms_concentration(ctx, solver_params)


def set_initial_conditions(
    ctx: dict, solver_params: Any, *, blob: bool = False
) -> None:
    """Initialize the primary unknowns in the formulation chosen for ``ctx``."""
    if _get_formulation(solver_params) == "logc":
        # blob ICs are concentration-formulation only; no-op the flag.
        return set_initial_conditions_logc(ctx, solver_params)
    return _set_initial_conditions_concentration(ctx, solver_params, blob=blob)


__all__ = [
    "build_context",
    "build_forms",
    "set_initial_conditions",
    "build_context_logc",
    "build_forms_logc",
    "set_initial_conditions_logc",
]
