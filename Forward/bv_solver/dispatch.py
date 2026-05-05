"""Formulation dispatcher for the BV-PNP forward solver.

The production stack uses the log-concentration backend (``forms_logc``).
An experimental proton-electrochemical-potential variant
(``forms_logc_muh``) is available for ``formulation="logc_muh"``; the muh
backend stores the proton primary variable as ``mu_H = u_H + em*z_H*phi``
to give Newton a smoother variable in Debye-layer regions where ``u_H``
and ``phi`` separately vary by tens of log-units.  See
``Forward/bv_solver/forms_logc_muh.py`` for the math and
``docs/electrochemical_potential_solver_plan.md`` for the landing plan.

``build_context`` / ``build_forms`` / ``set_initial_conditions`` dispatch on:

  - ``params['bv_convergence']['formulation']``  -- selects backend module
    (``"logc"`` or ``"logc_muh"``).
  - ``params['bv_convergence']['initializer']``  -- selects IC routine
    within the chosen backend (``"linear_phi"`` or ``"debye_boltzmann"``).

The ``*_logc`` and ``*_logc_muh`` names are re-exported for scripts that
want to be explicit about the backend they target.
"""

from __future__ import annotations

from typing import Any

from .forms_logc import (
    build_context_logc,
    build_forms_logc,
    set_initial_conditions_logc,
    set_initial_conditions_debye_boltzmann_logc,
)
from .forms_logc_muh import (
    build_context_logc_muh,
    build_forms_logc_muh,
    set_initial_conditions_logc_muh,
    set_initial_conditions_debye_boltzmann_logc_muh,
)


def _read_bv_convergence_field(solver_params: Any, key: str, default: str) -> str:
    """Read ``bv_convergence[key]`` with a graceful default.

    Robust to ``SolverParams`` (via ``__getitem__``), legacy 11-tuples,
    and malformed test inputs.  Falls back to ``default`` whenever the
    config block is missing or non-dict.
    """
    try:
        params_dict = solver_params[10] if hasattr(solver_params, "__getitem__") else None
    except Exception:
        params_dict = None
    if not isinstance(params_dict, dict):
        return default
    bv_conv = params_dict.get("bv_convergence", {})
    if not isinstance(bv_conv, dict):
        return default
    return str(bv_conv.get(key, default)).strip().lower()


def _read_formulation(solver_params: Any) -> str:
    return _read_bv_convergence_field(solver_params, "formulation", "logc")


def _read_initializer(solver_params: Any) -> str:
    return _read_bv_convergence_field(solver_params, "initializer", "linear_phi")


def _resolve_backend(solver_params: Any) -> str:
    """Pick a backend module name based on ``formulation``.

    Returns one of ``"logc"`` or ``"logc_muh"``.  Unknown formulations
    fall through to ``"logc"`` (the production default) -- the config
    layer (``Forward/bv_solver/config.py:_validate_formulation``) already
    rejects unknown names at parse time, so this is defensive only.
    """
    formulation = _read_formulation(solver_params)
    if formulation == "logc_muh":
        return "logc_muh"
    return "logc"


def build_context(solver_params: Any, *, mesh: Any = None) -> dict:
    backend = _resolve_backend(solver_params)
    if backend == "logc_muh":
        return build_context_logc_muh(solver_params, mesh=mesh)
    return build_context_logc(solver_params, mesh=mesh)


def build_forms(ctx: dict, solver_params: Any) -> dict:
    backend = _resolve_backend(solver_params)
    if backend == "logc_muh":
        return build_forms_logc_muh(ctx, solver_params)
    return build_forms_logc(ctx, solver_params)


def set_initial_conditions(
    ctx: dict, solver_params: Any, *, blob: bool = False
) -> None:
    """Set initial conditions for the dispatched backend.

    Dispatches on ``params['bv_convergence']['formulation']`` to pick the
    backend, then on ``params['bv_convergence']['initializer']`` to pick
    the IC routine within that backend:

      - ``"linear_phi"``     -> ``set_initial_conditions_*``
      - ``"debye_boltzmann"`` -> ``set_initial_conditions_debye_boltzmann_*``

    ``blob`` is accepted for backward-compatible kwargs and silently
    ignored -- blob ICs were a concentration-formulation feature removed
    with that backend.
    """
    backend = _resolve_backend(solver_params)
    initializer = _read_initializer(solver_params)
    if backend == "logc_muh":
        if initializer == "debye_boltzmann":
            return set_initial_conditions_debye_boltzmann_logc_muh(ctx, solver_params)
        return set_initial_conditions_logc_muh(ctx, solver_params)
    if initializer == "debye_boltzmann":
        return set_initial_conditions_debye_boltzmann_logc(ctx, solver_params)
    return set_initial_conditions_logc(ctx, solver_params)


__all__ = [
    "build_context",
    "build_forms",
    "set_initial_conditions",
    # logc backend (production)
    "build_context_logc",
    "build_forms_logc",
    "set_initial_conditions_logc",
    "set_initial_conditions_debye_boltzmann_logc",
    # logc_muh backend (experimental)
    "build_context_logc_muh",
    "build_forms_logc_muh",
    "set_initial_conditions_logc_muh",
    "set_initial_conditions_debye_boltzmann_logc_muh",
]
