"""Analytic Boltzmann counterion contribution to Poisson's equation.

When a supporting electrolyte ion is inert (does not participate in any
electrode reaction) it can be removed from the dynamic Nernst--Planck
system and replaced by an analytic Boltzmann profile in equilibrium with
the electrostatic potential.

Two variants are supported per-counterion via ``steric_mode``:

    "ideal" (default, legacy):
        c(x) = c_bulk * exp(-z * phi(x))   — unbounded.

    "bikerman":
        c(x) = c_b * exp(-z * phi)
               * (1 - A_dyn(x)) / (theta_b + a_b * c_b * exp(-z * phi))

        where A_dyn(x) = sum_j a_j * c_j(x) over dynamic species, and
        theta_b = 1 - sum_j a_j * c_b_j - a_b * c_b is the bulk packing
        fraction.  This is the steady-state algebraic reduction of the
        4sp dynamic Bikerman problem for an inert counterion under the
        sign-corrected Bikerman chemical potential
        ``mu_i = ln(c_i) + z_i*phi - ln(theta)``.  See
        ``docs/steric_analytic_clo4_reduction_handoff.md`` for the full
        derivation.

This module provides:
    - ``add_boltzmann_counterion_residual`` for the legacy ideal path
      (mutates Poisson residual after ``build_forms_logc`` returns).
    - ``build_steric_boltzmann_expressions`` for the bikerman path
      (returns UFL expressions called from inside ``build_forms_logc``
      so the closure enters BOTH the dynamic-species packing and the
      Poisson source).

The sign convention matches the Poisson residual built by the forms
modules:

    F_poisson  =  eps * grad(phi) . grad(w) * dx
                  - charge_rhs * sum_i z_i * c_i * w * dx
                  - charge_rhs * sum_k z_k * c_steric_k * w * dx
                                                  (analytic counterions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import firedrake as fd

from .config import _get_bv_boltzmann_counterions_cfg


@dataclass(frozen=True)
class StericBoltzmannBundle:
    """UFL expressions and metadata for one steric-aware Boltzmann counterion.

    Attributes
    ----------
    c_steric_expr
        UFL expression for the steric-aware analytic concentration
        ``c_b * q * (1 - A_dyn) / (theta_b + a_b * c_b * q)`` where
        ``q = exp(-z * phi_clamped)``.  Caller can extract surface
        values for diagnostics.
    packing_contribution
        UFL expression for ``a_b * c_steric``.  Caller adds this to
        ``theta = 1 - A_dyn - packing_contribution`` so the dynamic
        species' steric chemical potential includes the counterion.
    charge_density
        UFL expression for ``z * c_steric`` (NOT yet multiplied by
        ``charge_rhs`` or ``z_scale``).  Caller adds
        ``-charge_rhs * z_scale * charge_density * w * dx`` to the
        Poisson residual.
    z_scale
        R-space ``Function`` shared with any ideal-mode counterions on
        the same ``ctx`` (key ``'boltzmann_z_scale'``).  Default 1.0;
        Strategy-B orchestrators ramp this alongside the dynamic
        species' z-ramp (see ``add_boltzmann_counterion_residual``
        docstring).
    metadata
        Diagnostic info (z, c_bulk_nondim, a_nondim, phi_clamp,
        theta_b, config_index) used by the diagnostics layer.
    """
    c_steric_expr: Any
    packing_contribution: Any
    charge_density: Any
    z_scale: Any
    metadata: dict = field(default_factory=dict)


def build_steric_boltzmann_expressions(
    *,
    ctx: dict[str, Any],
    params: Any,
    ci: list,
    a_dyn_funcs: list,
    a_dyn_floats: list[float],
    c0_dyn: list[float],
    z_dyn: list[int],
    phi: Any,
    R_space: Any,
) -> StericBoltzmannBundle | None:
    """Build the UFL closure expressions for steric-aware Boltzmann counterions.

    Returns ``None`` when no entries have ``steric_mode='bikerman'``.

    Validates:
        - Exactly one bikerman entry (multi-counterion case raises
          ``NotImplementedError``).
        - ``theta_b > 0`` for the production setup; raises ``ValueError``
          with a descriptive message otherwise.
        - No double-counting: the bikerman entry's (z, c_bulk) cannot
          duplicate any dynamic species' (z, c_bulk) within ``1e-9``
          relative tolerance.

    The (re)used ``ctx['boltzmann_z_scale']`` Function is the same
    R-space scaling used by the legacy ideal path; both ideal and
    bikerman counterions share it on the same ctx, so a single
    ``_set_z_factor`` call ramps both contributions.

    Parameters
    ----------
    ctx
        Build context produced by ``build_forms_logc``.  Mutated to
        publish ``boltzmann_z_scale`` if it isn't already there.
    params
        Solver params dict (``solver_params[10]``).  Read via
        ``_get_bv_boltzmann_counterions_cfg``.
    ci
        UFL expressions for dynamic species concentrations
        (``[exp(u_i) for u_i in ui]`` in the log-c forms).
    a_dyn_funcs
        R-space Bikerman size Functions for the dynamic species
        (forms_logc.py builds these as ``steric_a_funcs``).
    a_dyn_floats
        Plain ``float`` values mirroring ``a_dyn_funcs`` (used for
        bulk-side validation; UFL Functions can't be evaluated as
        floats during form build).
    c0_dyn
        Bulk concentrations of dynamic species (``c0_model_vals`` in
        the log-c forms).  Plain floats.
    z_dyn
        Charge numbers of dynamic species (``z_vals``).  Plain ints.
    phi
        UFL ``Function`` for the electrostatic potential.
    R_space
        FunctionSpace used to (re)create ``boltzmann_z_scale``.

    Returns
    -------
    StericBoltzmannBundle | None
        Bundle with the UFL closure expressions and the shared z_scale
        Function, or ``None`` if no bikerman entries.
    """
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    bikerman = [(j, e) for j, e in enumerate(counterions)
                if e.get("steric_mode", "ideal") == "bikerman"]
    if not bikerman:
        return None
    if len(bikerman) > 1:
        raise NotImplementedError(
            "multi-counterion bikerman closure not supported: when more than "
            "one counterion is steric-aware the closure algebra couples "
            "(each appears in the others' denominator).  See "
            "docs/steric_analytic_clo4_reduction_handoff.md caveats."
        )

    j, entry = bikerman[0]
    z_b = int(entry["z"])
    c_b = float(entry["c_bulk_nondim"])
    a_b = float(entry["a_nondim"])
    phi_clamp_val = float(entry["phi_clamp"])

    if not (len(a_dyn_floats) == len(c0_dyn) == len(z_dyn) == len(ci) == len(a_dyn_funcs)):
        raise ValueError(
            "build_steric_boltzmann_expressions: dynamic-species inputs misaligned "
            f"(a_dyn_floats={len(a_dyn_floats)}, c0_dyn={len(c0_dyn)}, "
            f"z_dyn={len(z_dyn)}, ci={len(ci)}, a_dyn_funcs={len(a_dyn_funcs)})"
        )

    # Validate theta_b > 0 (parser doesn't see dynamic species)
    A_dyn_bulk = sum(a * c for a, c in zip(a_dyn_floats, c0_dyn))
    theta_b = 1.0 - A_dyn_bulk - a_b * c_b
    if theta_b <= 0.0:
        raise ValueError(
            f"boltzmann_counterions[{j}] bikerman closure requires theta_b > 0, "
            f"but got theta_b = 1 - A_dyn_bulk - a_b*c_b "
            f"= 1 - {A_dyn_bulk:.6g} - {a_b:.6g}*{c_b:.6g} = {theta_b:.6g}; "
            f"reduce dynamic species' bulk fractions, a_b, or c_b."
        )

    # Double-counting guard: bikerman entry must not duplicate a dynamic species
    rel_tol = 1e-9
    for i in range(len(c0_dyn)):
        same_z = z_dyn[i] == z_b
        denom = max(abs(c0_dyn[i]), abs(c_b), 1e-300)
        same_c = abs(c0_dyn[i] - c_b) <= rel_tol * denom
        if same_z and same_c:
            raise ValueError(
                f"boltzmann_counterions[{j}] (z={z_b}, c_bulk={c_b}) "
                f"duplicates dynamic species[{i}] (z={z_dyn[i]}, c_bulk={c0_dyn[i]}); "
                f"remove from one or the other to avoid double-counting in "
                f"Poisson and steric packing."
            )

    # UFL expressions
    phi_clamped = fd.min_value(
        fd.max_value(phi, fd.Constant(-phi_clamp_val)),
        fd.Constant(phi_clamp_val),
    )
    z_b_const = fd.Constant(float(z_b))
    a_b_const = fd.Constant(a_b)
    c_b_const = fd.Constant(c_b)
    theta_b_const = fd.Constant(theta_b)

    q = fd.exp(-z_b_const * phi_clamped)
    A_dyn_local = sum(a_dyn_funcs[i] * ci[i] for i in range(len(ci)))

    # Numerator (1 - A_dyn) is mathematically positive in the physical
    # regime; floor it for FE-interpolation safety on coarse meshes
    # where the nodal interpolant of (1 - A_dyn) can dip below 0 even
    # when the symbolic expression doesn't.
    free_dyn_floor = fd.Constant(1e-10)
    free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)

    c_steric = c_b_const * q * free_dyn / (theta_b_const + a_b_const * c_b_const * q)

    packing_contribution = a_b_const * c_steric
    charge_density = z_b_const * c_steric

    # (Re)use the shared z_scale Function.  Match the legacy key name
    # so grid_per_voltage._set_z_factor ramps both contributions.
    if "boltzmann_z_scale" in ctx:
        z_scale = ctx["boltzmann_z_scale"]
    else:
        z_scale = fd.Function(R_space, name="boltzmann_z_scale")
        z_scale.assign(1.0)
        ctx["boltzmann_z_scale"] = z_scale

    return StericBoltzmannBundle(
        c_steric_expr=c_steric,
        packing_contribution=packing_contribution,
        charge_density=charge_density,
        z_scale=z_scale,
        metadata={
            "config_index": j,
            "z": z_b,
            "c_bulk_nondim": c_b,
            "a_nondim": a_b,
            "phi_clamp": phi_clamp_val,
            "theta_b": theta_b,
        },
    )


def add_boltzmann_counterion_residual(
    ctx: dict[str, Any],
    params: Any,
    *,
    skip_bikerman: bool = False,
) -> int:
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
    # Reuse the Function already on ``ctx`` if present — this keeps the
    # legacy ideal-path and the new bikerman-path expressions referring
    # to the SAME Function so a single ``_set_z_factor(ctx, z)`` ramps
    # both contributions consistently.
    if "boltzmann_z_scale" in ctx:
        z_scale = ctx["boltzmann_z_scale"]
    else:
        R_space = fd.FunctionSpace(mesh, "R", 0)
        z_scale = fd.Function(R_space, name="boltzmann_z_scale")
        z_scale.assign(1.0)

    F_res = ctx["F_res"]
    for entry in counterions:
        # Bikerman entries are wired separately by
        # build_steric_boltzmann_expressions (called from build_forms_logc
        # BEFORE the steric residual is constructed, so the closure can
        # enter both Poisson and the dynamic species' total packing).
        if skip_bikerman and entry.get("steric_mode", "ideal") == "bikerman":
            continue
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
