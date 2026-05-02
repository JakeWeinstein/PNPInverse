"""Analytic Boltzmann counterion contribution to Poisson's equation.

When a supporting electrolyte ion is inert (does not participate in any
electrode reaction) it can be removed from the dynamic Nernst--Planck
system and replaced by an analytic Boltzmann profile in equilibrium with
the electrostatic potential:

    c(x) = c_bulk * exp(-z * phi(x) / V_T)

(In the dimensionless form used by the solver, V_T is folded into the
potential scale, so the exponent becomes ``-z * phi``.)  The ion still
contributes to the local charge density that drives Poisson's equation.

This module provides a single helper that adds the corresponding residual
term to the mixed weak form, so both the standard concentration solver
(``forms.py``) and the log-concentration solver (``forms_logc.py``) can
use the same code path.

The sign convention matches the Poisson residual built by the forms
modules:

    F_poisson  =  eps * grad(phi) . grad(w) * dx
                  - charge_rhs * sum_i z_i * c_i * w * dx
                  - charge_rhs * sum_k z_k * c_bulk_k * exp(-z_k * phi) * w * dx
                                                        (Boltzmann ions)
"""

from __future__ import annotations

from typing import Any

import firedrake as fd

from .config import _get_bv_boltzmann_counterions_cfg


def add_boltzmann_counterion_residual(ctx: dict[str, Any], params: Any) -> int:
    """Append analytic-Boltzmann counterion residuals to ``ctx['F_res']``.

    Looks up ``params['bv_bc']['boltzmann_counterions']`` (parsed via
    :func:`_get_bv_boltzmann_counterions_cfg`).  When non-empty, mutates
    ``ctx`` in place by:

        - extending ``F_res`` with one Poisson source term per counterion
          (sign and prefactor matching the existing Poisson residual);
        - re-deriving ``J_form`` from the updated residual;
        - storing the parsed config under ``ctx['boltzmann_counterions']``;
        - storing a mutable scaling Constant under
          ``ctx['boltzmann_z_scale']``.

    The ``boltzmann_z_scale`` is an R-space Function (default ``1.0`` so
    standalone usage is unchanged) that orchestrators use to *ramp the
    counterion's contribution alongside* the dynamic-species z-ramp.
    Without this scale, when ``solve_grid_with_charge_continuation``
    Phase 1 sets the dynamic species' ``z_consts`` to zero to compute a
    voltage-only neutral sweep, the analytic counterion would still
    contribute its full ``z = -1`` charge to Poisson, leaving the bulk
    non-electroneutral and breaking the V-sweep bisection.  Setting
    ``boltzmann_z_scale = 0`` during Phase 1 (then ramping to 1 in
    Phase 2 alongside ``z_consts``) keeps Phase 1 truly neutral.

    Parameters
    ----------
    ctx : dict
        Context produced by ``build_forms`` / ``build_forms_logc``.  Must
        contain ``F_res``, ``U``, ``W``, ``mesh``, and ``nondim``.
    params : dict
        The 11-tuple's params dict (``solver_params[10]``).

    Returns
    -------
    int
        Number of Boltzmann counterion entries added.  ``0`` is a no-op.
    """
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    if not counterions:
        return 0

    if "F_res" not in ctx or "U" not in ctx or "W" not in ctx:
        raise ValueError(
            "add_boltzmann_counterion_residual requires a built ctx "
            "with 'F_res', 'U', 'W', and 'nondim'.  Call build_forms "
            "or build_forms_logc first."
        )

    mesh = ctx["mesh"]
    W = ctx["W"]
    U = ctx["U"]
    scaling = ctx.get("nondim", {})

    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh)

    charge_rhs_val = float(scaling.get("charge_rhs_prefactor", 1.0))
    charge_rhs = fd.Constant(charge_rhs_val)

    # Mutable scaling Function so orchestrators can ramp the Boltzmann
    # contribution alongside the dynamic-species z-ramp.  Default 1.0
    # preserves existing standalone behavior (full counterion charge).
    R_space = fd.FunctionSpace(mesh, "R", 0)
    z_scale = fd.Function(R_space, name="boltzmann_z_scale")
    z_scale.assign(1.0)

    F_res = ctx["F_res"]
    for entry in counterions:
        z_val = int(entry["z"])
        c_bulk_val = float(entry["c_bulk_nondim"])
        phi_clamp_val = float(entry["phi_clamp"])
        if c_bulk_val == 0.0:
            continue
        z_const = fd.Constant(float(z_val))
        c_bulk_const = fd.Constant(c_bulk_val)
        phi_clamped = fd.min_value(
            fd.max_value(phi, fd.Constant(-phi_clamp_val)),
            fd.Constant(phi_clamp_val),
        )
        # Poisson source: -charge_rhs * z * c_bulk * exp(-z*phi) * w,
        # multiplied by z_scale so orchestrators can ramp it.
        F_res -= z_scale * charge_rhs * z_const * c_bulk_const * fd.exp(
            -z_const * phi_clamped
        ) * w * dx

    # Re-derive Jacobian from the updated residual.
    ctx["F_res"] = F_res
    ctx["J_form"] = fd.derivative(F_res, U)
    ctx["boltzmann_counterions"] = list(counterions)
    ctx["boltzmann_z_scale"] = z_scale
    return len(counterions)
