"""Water self-ionization closure (Phase 6α — proton-condition variable).

Replaces the H⁺ Nernst--Planck residual with the proton-condition
residual on ``E = c_H − c_OH``, with the fast-equilibrium closure
``c_OH(y) = Kw_eff · exp(−u_H(y))`` (logc) or
``c_OH = Kw_eff · exp(em·z_H·φ − μ_H)`` (muh).  The conservative flux
in either backend reduces to

    J_E = −(D_H · c_H + D_OH · c_OH) · ∇μ_H_full
        + (D_H · c_H − D_OH · c_OH) · ∇μ_steric

(after using ``z_OH = −1`` and ``∇μ_OH = −∇μ_H + ∇μ_steric``), where
``∇μ_H_full = ∇u_H + em·z_H·∇φ + ∇μ_steric``.

The H⁺ primary variable (``u_H`` or ``μ_H``) is unchanged — only the
H⁺ residual EQUATION is replaced; the BVs on ``u_H``/``μ_H`` at the
bulk Dirichlet stay the same.  The BV electrode reaction in acid-form
ORR consumes only H⁺ (not OH⁻), so the existing
``F_res -= stoi[h_idx] * R_j * v_list[h_idx] * ds`` term puts
``J_E · n = J_H · n`` at the electrode automatically — no per-reaction
rewiring needed.

OH⁻ also enters:

* Poisson source as ``(−1) · c_OH``.
* Bikerman packing as ``a_OH · c_OH`` inside ``A_dyn``.

Default-off via ``solver_options['bv_convergence']['enable_water_ionization']``;
the feature is gated at form-build time so the disabled path is byte-
equivalent to the pre-Phase-6α stack.

See ``docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import firedrake as fd


@dataclass(frozen=True)
class WaterIonizationBundle:
    """UFL pieces produced by :func:`build_water_ionization_terms`.

    Attributes
    ----------
    h_idx
        Species index of H⁺ (z = +1) in the dynamic species list.
    kw_eff_func
        R-space Function holding the current K_w_eff value.  Updated
        in-place by ``set_reaction_kw_eff_model`` during continuation.
    d_oh_const, a_oh_const
        UFL Constants for the OH⁻ diffusivity and Bikerman size.
    c_oh_expr
        Symbolic ``Kw_eff · exp(−u_H_clamped)`` UFL expression at the
        current time level.  Caller threads this into the Poisson
        source and (when steric) the Bikerman packing.
    c_oh_prev_expr
        Same expression but built from ``u_H_clamped_old`` for the
        backward-Euler time term.
    e_var_expr
        ``c_H − c_OH`` at the current time level (the proton-condition
        variable).
    e_var_prev_expr
        ``c_H_old − c_OH_old`` at the previous time level.
    """

    h_idx: int
    kw_eff_func: Any
    d_oh_const: Any
    a_oh_const: Any
    c_oh_expr: Any
    c_oh_prev_expr: Any
    e_var_expr: Any
    e_var_prev_expr: Any


def resolve_h_index(z_vals, *, roles=None) -> int:
    """Return the index of H⁺ in a dynamic species list.

    Two paths:

    * ``roles=None`` (legacy) — infer from ``z_vals``: exactly one
      species with ``z = +1`` is required.  Raises with a descriptive
      message otherwise so config errors fail loudly at form-build time
      rather than silently mis-wiring c_OH.
    * ``roles is not None`` (Phase 6β v9 Gate 1) — pick the unique index
      whose role label is ``"proton"``.  Required when multiple
      dynamic species share ``z = +1`` (the K2SO4 stack: H⁺ + K⁺).
    """
    if roles is not None:
        if len(roles) != len(z_vals):
            raise ValueError(
                f"resolve_h_index: roles length {len(roles)} != z_vals "
                f"length {len(z_vals)}; pass one role per dynamic species."
            )
        candidates = [
            i for i, r in enumerate(roles)
            if str(r).strip().lower() == "proton"
        ]
        if not candidates:
            raise ValueError(
                "resolve_h_index: no species with role='proton' in "
                f"roles={list(roles)}."
            )
        if len(candidates) > 1:
            raise ValueError(
                "resolve_h_index: multiple species with role='proton' in "
                f"roles={list(roles)} (indices {candidates}); roles must "
                "be unique."
            )
        return candidates[0]

    candidates = [i for i, zv in enumerate(z_vals) if abs(float(zv) - 1.0) < 1e-12]
    if not candidates:
        raise ValueError(
            "enable_water_ionization=True requires a species with z=+1 (H+); "
            f"none found in z_vals={list(z_vals)}."
        )
    if len(candidates) > 1:
        raise ValueError(
            "enable_water_ionization=True requires exactly one species with "
            f"z=+1 (H+); found {len(candidates)} (indices {candidates})."
        )
    return candidates[0]


def build_water_ionization_terms(
    *,
    ctx: dict,
    conv_cfg: dict,
    z_vals,
    u_h_unclamped,
    u_h_unclamped_prev,
    ci_h_clamped,
    ci_h_prev_clamped,
    R_space,
    u_clamp: float,
    roles=None,
) -> WaterIonizationBundle:
    """Build the OH⁻ closure expressions used by the proton-condition residual.

    The H⁺ log-concentration expression (``u_h_unclamped``) is the
    pre-clamp value; the same symmetric clamp ``[-u_clamp, u_clamp]``
    applied to ``c_H = exp(u_h_unclamped)`` is applied to ``-u_H`` to
    bound ``c_OH = Kw_eff · exp(-u_H)``.  Because the clamp is symmetric
    about zero, ``clamp(-x) = -clamp(x)``, so passing the already-built
    ``ci_h_clamped`` (which uses the same symmetric clamp) and dividing
    is byte-equivalent — but the explicit ``exp(-u_clamped)`` form keeps
    the algebra parallel to the existing ``ci`` pattern and avoids
    a divide-by-near-zero at quadrature points where ``ci_h_clamped``
    underflows to ``exp(-100)`` ≈ 3.7e-44.

    Parameters
    ----------
    ctx
        Firedrake context.  Mutated in place: ``ctx['water_ionization']``
        is set to the returned bundle for downstream Q3 / diagnostics.
    conv_cfg
        ``solver_options['bv_convergence']`` sub-dict.  Reads
        ``enable_water_ionization`` (must be True; caller gates),
        ``kw_eff_hat``, ``d_oh_hat``, ``a_oh_hat``.
    z_vals
        Per-species charge list.  Used to locate the H⁺ index.
    u_h_unclamped, u_h_unclamped_prev
        UFL expressions for ``ln(c_H)`` at the current and previous
        time levels.  In ``forms_logc.py`` these are ``ui[h_idx]`` and
        ``ui_prev[h_idx]``; in ``forms_logc_muh.py`` they are the
        reconstructed ``μ_H − em·z_H·φ`` expressions.
    ci_h_clamped, ci_h_prev_clamped
        UFL expressions for ``c_H = exp(clamp(u_H))`` at the two time
        levels.  Caller has already built these for the dynamic-species
        residual; we reuse them in ``E = c_H − c_OH`` to keep the time
        derivative consistent with the rest of the species.
    R_space
        ``firedrake.FunctionSpace(mesh, "R", 0)`` for the Kw_eff
        Function.
    u_clamp
        Symmetric clamp magnitude (typically 100.0 in production).

    Returns
    -------
    WaterIonizationBundle
    """
    h_idx = resolve_h_index(list(z_vals), roles=roles)

    kw_eff_initial = float(conv_cfg.get("kw_eff_hat", 0.0))
    d_oh = float(conv_cfg.get("d_oh_hat", 0.0))
    a_oh = float(conv_cfg.get("a_oh_hat", 0.0))

    if d_oh <= 0.0:
        raise ValueError(
            f"enable_water_ionization=True but d_oh_hat={d_oh!r} is non-positive; "
            "set bv_convergence['d_oh_hat'] to the OH⁻ nondim diffusivity."
        )
    if kw_eff_initial < 0.0:
        raise ValueError(
            f"enable_water_ionization=True but kw_eff_hat={kw_eff_initial!r} "
            "is negative; the continuation ladder may start at 0 but cannot "
            "go below."
        )

    kw_eff_func = fd.Function(R_space, name="kw_eff_water_ionization")
    kw_eff_func.assign(kw_eff_initial)

    d_oh_const = fd.Constant(d_oh)
    a_oh_const = fd.Constant(a_oh)

    u_clamp_c = fd.Constant(float(u_clamp))
    neg_u_clamp_c = fd.Constant(-float(u_clamp))

    # c_OH = Kw_eff · exp(-u_H), with the same symmetric clamp on -u_H
    # as on u_H.  Because the clamp is symmetric, clamp(-u_H) = -clamp(u_H)
    # and we could equivalently write Kw_eff / ci_h_clamped — but the
    # explicit exp form preserves the existing ``ci`` algebra structure.
    neg_u_h_clamped = fd.min_value(
        fd.max_value(-u_h_unclamped, neg_u_clamp_c), u_clamp_c
    )
    neg_u_h_prev_clamped = fd.min_value(
        fd.max_value(-u_h_unclamped_prev, neg_u_clamp_c), u_clamp_c
    )

    c_oh_expr = kw_eff_func * fd.exp(neg_u_h_clamped)
    c_oh_prev_expr = kw_eff_func * fd.exp(neg_u_h_prev_clamped)

    e_var_expr = ci_h_clamped - c_oh_expr
    e_var_prev_expr = ci_h_prev_clamped - c_oh_prev_expr

    bundle = WaterIonizationBundle(
        h_idx=int(h_idx),
        kw_eff_func=kw_eff_func,
        d_oh_const=d_oh_const,
        a_oh_const=a_oh_const,
        c_oh_expr=c_oh_expr,
        c_oh_prev_expr=c_oh_prev_expr,
        e_var_expr=e_var_expr,
        e_var_prev_expr=e_var_prev_expr,
    )
    ctx["water_ionization"] = bundle
    return bundle


def build_proton_condition_flux(
    *,
    bundle: WaterIonizationBundle,
    D_h,
    c_h,
    ideal_grad_h,
    mu_steric_grad,
    steric_active: bool,
):
    """Return ``J_E = J_H − J_OH`` for the proton-condition residual.

    Uses the identities (z_H = +1, z_OH = −1, em·z_H = +1):

    * ``∇ln(c_OH) + em·z_OH·∇φ = −∇u_H − em·∇φ = −(∇u_H + em·∇φ)``
      — so the OH⁻ "ideal gradient" is exactly the negation of the
      H⁺ ideal gradient.
    * Steric chemical potential ``μ_steric = −ln(θ)`` is the same
      activity correction for every dynamic species, so OH⁻ picks up
      the SAME ``+∇μ_steric`` term as H⁺.

    The conservative NP flux is therefore

        J_E = −(D_H·c_H + D_OH·c_OH)·∇μ_ideal
              − (D_H·c_H − D_OH·c_OH)·∇μ_steric

    (both flux contributions linear in ``∇μ_ideal`` and ``∇μ_steric``
    separately; the steric correction cancels except where the
    diffusivity-weighted balance ``D_H·c_H ≠ D_OH·c_OH`` is broken).

    **Sign convention.**  Both forms files build the dynamic-species
    residual block as

        F_res += ((c_i − c_old)/dt) · v · dx + dot(Jflux, ∇v) · dx

    which assembles the weak form ``∫(∂c/∂t)·v − ∫J_NP·∇v + ∫J_NP·n·v``
    (integration by parts of ``∇·J_NP``) ONLY when the symbol ``Jflux``
    holds the *negation* of the true Nernst-Planck flux, i.e.
    ``Jflux = −J_NP = +D·c·∇μ``.  See ``forms_logc.py`` line 334 (or
    ``forms_logc_muh.py`` line 440): ``D · c · ∇μ`` is written with a
    POSITIVE sign — that is the convention this helper must match.
    Returning ``-J_E`` here keeps the same convention so callers can
    write ``F_res += dot(<this expression>, ∇v) dx`` unchanged.

    Parameters
    ----------
    bundle
        Output of :func:`build_water_ionization_terms`.  Provides the
        OH⁻ diffusivity Constant and the c_OH expression.
    D_h
        H⁺ diffusivity expression (typically ``fd.exp(logD_funcs[h_idx])``).
    c_h
        H⁺ concentration expression (``ci[h_idx]``).
    ideal_grad_h
        ``∇μ_H_ideal`` for the H⁺ species — i.e. ``∇u_H + em·z_H·∇φ``
        in logc, ``∇μ_H`` in muh.
    mu_steric_grad
        ``∇μ_steric`` UFL expression, or ``None`` when ``steric_active``
        is False.
    steric_active
        Caller-provided flag; matches the existing ``steric_active``
        gate in both forms files.

    Returns
    -------
    UFL expression for ``-J_E`` (= the residual-side ``Jflux`` symbol
    for the proton-condition variable).  At ``Kw_eff = 0`` (so c_OH = 0)
    this reduces algebraically to the H⁺ NP residual the form would
    have built without water-ionization, restoring byte-equivalence at
    the continuation floor.
    """
    d_oh = bundle.d_oh_const
    c_oh = bundle.c_oh_expr

    flux = (D_h * c_h + d_oh * c_oh) * ideal_grad_h
    if steric_active and mu_steric_grad is not None:
        flux = flux + (D_h * c_h - d_oh * c_oh) * mu_steric_grad
    return flux


def is_water_ionization_enabled(conv_cfg: dict) -> bool:
    """Centralized gate check; reads the ``enable_water_ionization`` flag.

    Returns False both when the key is missing (legacy params dict) and
    when set to a falsy value, so callers can rely on a uniform default.
    """
    return bool(conv_cfg.get("enable_water_ionization", False))


__all__ = [
    "WaterIonizationBundle",
    "build_proton_condition_flux",
    "build_water_ionization_terms",
    "is_water_ionization_enabled",
    "resolve_h_index",
]
