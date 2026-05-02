"""Formulation dispatcher for the BV-PNP forward solver.

After the May 2026 cleanup, the production stack supports only the
log-concentration backend (``forms_logc``).  This module remains as the
canonical import point so callers continue to use ``build_context`` /
``build_forms`` / ``set_initial_conditions`` regardless of any future
backend additions.

The ``*_logc`` names are also re-exported for scripts that want to be
explicit about the backend they target.
"""

from __future__ import annotations

from typing import Any

from .forms_logc import (
    build_context_logc,
    build_forms_logc,
    set_initial_conditions_logc,
)


def build_context(solver_params: Any, *, mesh: Any = None) -> dict:
    return build_context_logc(solver_params, mesh=mesh)


def build_forms(ctx: dict, solver_params: Any) -> dict:
    return build_forms_logc(ctx, solver_params)


def set_initial_conditions(
    ctx: dict, solver_params: Any, *, blob: bool = False
) -> None:
    """Set initial conditions in log-c space.

    ``blob`` is accepted for backward-compatible kwargs and silently
    ignored — blob ICs were a concentration-formulation feature removed
    with that backend.
    """
    return set_initial_conditions_logc(ctx, solver_params)


__all__ = [
    "build_context",
    "build_forms",
    "set_initial_conditions",
    "build_context_logc",
    "build_forms_logc",
    "set_initial_conditions_logc",
]
