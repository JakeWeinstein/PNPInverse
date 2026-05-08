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
) -> list[StericBoltzmannBundle]:
    """Build UFL closure expressions for steric-aware Boltzmann counterions.

    Returns an empty list when no entries have ``steric_mode='bikerman'``.
    The legacy single-counterion case (len == 1) reduces to the same
    UFL algebra it always built; the bundle list contains exactly one
    element. The multi-counterion shared-theta closure (plan §2.1)
    extends this:

    .. code-block:: text

        For each analytic ion k (steric):
          c_k(φ) = c_b_k · exp(-z_k·φ) · (1 - A_dyn(φ))
                       / (θ_b + Σ_k' a_k' · c_b_k' · exp(-z_k'·φ))

        with A_dyn(φ) = Σ_dyn a_i · c_i_dyn(φ)
             θ_b      = 1 - A_dyn_bulk - Σ_k a_k · c_b_k

    The denominator is the same for every steric ion (shared theta);
    no coupled local NL solve needed.

    Validates:
        - ``θ_b > 0`` for the bulk (sum over ALL bikerman entries);
          raises ``ValueError`` with a descriptive message otherwise.
        - No double-counting: no bikerman entry's (z, c_bulk) may
          duplicate any dynamic species' (z, c_bulk) within ``1e-9``
          relative tolerance.

    The (re)used ``ctx['boltzmann_z_scale']`` Function is the same
    R-space scaling used by the legacy ideal path; both ideal and
    bikerman counterions share it on the same ctx, so a single
    ``_set_z_factor`` call ramps every contribution.

    Returns
    -------
    list[StericBoltzmannBundle]
        Empty list when no bikerman entries.  Otherwise one bundle per
        bikerman counterion.  All bundles share the same z_scale.
    """
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    bikerman = [(j, e) for j, e in enumerate(counterions)
                if e.get("steric_mode", "ideal") == "bikerman"]
    if not bikerman:
        return []

    if not (len(a_dyn_floats) == len(c0_dyn) == len(z_dyn) == len(ci) == len(a_dyn_funcs)):
        raise ValueError(
            "build_steric_boltzmann_expressions: dynamic-species inputs misaligned "
            f"(a_dyn_floats={len(a_dyn_floats)}, c0_dyn={len(c0_dyn)}, "
            f"z_dyn={len(z_dyn)}, ci={len(ci)}, a_dyn_funcs={len(a_dyn_funcs)})"
        )

    # ----- bulk-side validation across ALL bikerman entries -----
    A_dyn_bulk = sum(a * c for a, c in zip(a_dyn_floats, c0_dyn))
    A_an_bulk = sum(
        float(e["a_nondim"]) * float(e["c_bulk_nondim"]) for _, e in bikerman
    )
    theta_b = 1.0 - A_dyn_bulk - A_an_bulk
    if theta_b <= 0.0:
        bm_str = ", ".join(
            f"a={float(e['a_nondim']):.4g}*c_b={float(e['c_bulk_nondim']):.4g}"
            for _, e in bikerman
        )
        raise ValueError(
            f"bikerman multi-counterion closure requires theta_b > 0, but got "
            f"theta_b = 1 - A_dyn_bulk - sum(a_k*c_b_k) "
            f"= 1 - {A_dyn_bulk:.6g} - {A_an_bulk:.6g} = {theta_b:.6g}; "
            f"bikerman entries: [{bm_str}]; reduce dynamic-species packing, "
            f"per-ion a_nondim, or per-ion c_bulk_nondim."
        )

    # Double-counting guard against EVERY dynamic species AND
    # within-bikerman duplication (e.g. user lists Cs+ twice).
    rel_tol = 1e-9
    for k_idx, (j, entry) in enumerate(bikerman):
        z_k = int(entry["z"])
        c_k = float(entry["c_bulk_nondim"])
        for i in range(len(c0_dyn)):
            same_z = z_dyn[i] == z_k
            denom = max(abs(c0_dyn[i]), abs(c_k), 1e-300)
            same_c = abs(c0_dyn[i] - c_k) <= rel_tol * denom
            if same_z and same_c:
                raise ValueError(
                    f"boltzmann_counterions[{j}] (z={z_k}, c_bulk={c_k}) "
                    f"duplicates dynamic species[{i}] (z={z_dyn[i]}, "
                    f"c_bulk={c0_dyn[i]}); remove from one or the other to "
                    f"avoid double-counting in Poisson and steric packing."
                )
        for k2_idx, (j2, entry2) in enumerate(bikerman):
            if k2_idx <= k_idx:
                continue
            z_k2 = int(entry2["z"])
            c_k2 = float(entry2["c_bulk_nondim"])
            denom = max(abs(c_k), abs(c_k2), 1e-300)
            if z_k == z_k2 and abs(c_k - c_k2) <= rel_tol * denom:
                raise ValueError(
                    f"boltzmann_counterions[{j}] and [{j2}] duplicate "
                    f"(z={z_k}, c_bulk={c_k}); remove one to avoid "
                    f"double-counting."
                )

    # ----- shared UFL pieces (denominator + dynamic packing factor) -----
    A_dyn_local = sum(a_dyn_funcs[i] * ci[i] for i in range(len(ci)))
    free_dyn_floor = fd.Constant(1e-10)   # FE-interpolation safety on coarse meshes
    free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)
    theta_b_const = fd.Constant(theta_b)

    # Per-bikerman exponent uses each ion's OWN phi_clamp (each entry
    # may set its own conservative clamp).  The SHARED denominator sums
    # a_k * c_b_k * exp(-z_k * phi_clamped_k) — using each ion's clamp
    # for its own term inside the sum is consistent with the closure
    # derivation (the clamp only ever appears multiplied by that ion's
    # z_k anyway, so the per-ion clamp is the right cap).
    per_ion_q = []
    for k_idx, (j, entry) in enumerate(bikerman):
        z_k = int(entry["z"])
        c_k = float(entry["c_bulk_nondim"])
        a_k = float(entry["a_nondim"])
        phi_clamp_k = float(entry["phi_clamp"])
        z_k_const = fd.Constant(float(z_k))
        c_k_const = fd.Constant(c_k)
        a_k_const = fd.Constant(a_k)
        phi_clamped_k = fd.min_value(
            fd.max_value(phi, fd.Constant(-phi_clamp_k)),
            fd.Constant(phi_clamp_k),
        )
        q_k = fd.exp(-z_k_const * phi_clamped_k)
        per_ion_q.append({
            "z_const": z_k_const,
            "c_const": c_k_const,
            "a_const": a_k_const,
            "q": q_k,
            "z": z_k,
            "c_bulk": c_k,
            "a_nondim": a_k,
            "phi_clamp": phi_clamp_k,
            "config_index": j,
        })

    denom = theta_b_const + sum(p["a_const"] * p["c_const"] * p["q"]
                                for p in per_ion_q)

    # Shared z_scale Function — single producer / single consumer.
    if "boltzmann_z_scale" in ctx:
        z_scale = ctx["boltzmann_z_scale"]
    else:
        z_scale = fd.Function(R_space, name="boltzmann_z_scale")
        z_scale.assign(1.0)
        ctx["boltzmann_z_scale"] = z_scale

    bundles: list[StericBoltzmannBundle] = []
    for p in per_ion_q:
        c_steric_k = p["c_const"] * p["q"] * free_dyn / denom
        bundles.append(StericBoltzmannBundle(
            c_steric_expr=c_steric_k,
            packing_contribution=p["a_const"] * c_steric_k,
            charge_density=p["z_const"] * c_steric_k,
            z_scale=z_scale,
            metadata={
                "config_index": p["config_index"],
                "z": p["z"],
                "c_bulk_nondim": p["c_bulk"],
                "a_nondim": p["a_nondim"],
                "phi_clamp": p["phi_clamp"],
                "theta_b": theta_b,
            },
        ))
    return bundles


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
