"""Cation hydrolysis closure (Phase 6β v9 Gates 3 + 4).

Adds a global surface coverage ``Γ_MOH`` representing the cation-
hydrolysis reservoir at the OHP::

    M⁺(H₂O)ₙ + H₂O  ⇌  MOH⁰(H₂O)ₙ₋₁ + H₃O⁺                     (Singh 2016 Eq. 2)

with field-dependent rate balance::

    R_net  =  k_hyd · c_M⁺(0) · 10^(−ΔpKa(σ_S))
              −  k_prot · c_H(0) · Γ / δ_OHP                       (forward − reverse)
    R_des  =  k_des · Γ                                            (desorption)

The proton boundary residual gains a source ``+λ · R_net`` (one H⁺
produced per hydrolysis event); the cation boundary residual gains a
sink ``−λ · R_net``.  Steady state for Γ is::

    Γ_ss(λ)  =  λ · ⟨R_net_forward⟩
                /  (λ · k_des + (1 − λ) + λ · k_prot · ⟨c_H⟩ / δ_OHP)

where ``⟨·⟩`` denotes the boundary-area-averaged value of the
forward branch
``R_net_forward = k_hyd · c_M · 10^(−ΔpKa)`` and ``⟨c_H⟩``.

Architecture mirrors :mod:`Forward.bv_solver.water_ionization`'s
``kw_eff_func`` *coefficient* pattern:

* Γ is an R-space ``Function`` that the Newton residual reads as a
  coefficient.  The mixed function space stays at the legacy
  ``species + phi`` layout (no extra Newton unknown for Γ).  This
  side-steps the Firedrake R-space-in-mixed-space matnest format
  limitation that breaks monolithic LU assembly.
* Between continuation rungs the orchestrator calls
  :func:`update_gamma_from_solution` to recompute Γ from the
  converged ``c_M(0)`` and ``c_H(0)`` boundary integrals via the
  closed-form ``Γ_ss(λ)`` formula.  At ``λ=0`` this returns ``Γ=0``
  exactly — the Dirichlet pin invariant.
* :func:`build_cation_hydrolysis_terms` is called from inside
  ``build_forms_logc[_muh]`` when ``enable_cation_hydrolysis=True``;
  it constructs the R-space Functions and stashes the bundle on
  ``ctx``.
* :func:`build_proton_boundary_source` returns the UFL expression
  for ``R_net`` so the form code can plumb it into both the proton
  boundary residual and the cation boundary residual with the
  correct signs.
* :func:`build_pka_shift` returns ``ΔpKa(σ)`` UFL expression.  At
  Gate 3B this is a placeholder returning ``Constant(0.0)``; Gate 4A
  swaps in Singh 2016 SI Eq. (4).

Default-off contract: when
``solver_options['bv_convergence']['enable_cation_hydrolysis']`` is
False the bundle is never constructed and every UFL plumbing site
short-circuits, leaving the residual byte-equivalent to the pre-Gate-3
stack.

References
----------

* ``docs/singh_2016_pka_formula.md`` — Singh 2016 SI Eq. (3)/(4)
  extraction with sign convention recovered from Tables S1/S3.
* ``.claude/plans/write-up-the-formal-joyful-papert.md`` §"Phase 1 —
  Gate 3" / §"Phase 2 — Gate 4" for the gating + tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import firedrake as fd


# ---------------------------------------------------------------------------
# Data + gating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CationHydrolysisBundle:
    """Frozen handle to the cation-hydrolysis UFL pieces and continuation knobs.

    Attributes
    ----------
    counterion_idx
        Role-resolved index of the cation species in the dynamic NP
        list (Gate 1).  At Gate 4 (K2SO4 stack) this is the K⁺ index.
    h_idx
        Role-resolved index of the proton species (the H⁺ NP residual).
    delta_ohp_func
        R-space ``Function`` holding the OHP thickness ``δ_OHP``
        (nondim length).  Production prior 0.40 nm (Bohra 2019).
    k_hyd_func, k_prot_func, k_des_func
        R-space ``Function`` objects holding the forward / reverse /
        desorption rates.  All mutable by the continuation
        orchestrator via the ``set_reaction_k_*_model`` accessors.
    lambda_hydrolysis_func
        R-space ``Function`` holding the smooth activation knob
        ``λ ∈ [0, 1]`` walked by ``lambda_hydrolysis_ladder``.  At
        ``λ=0`` the cation-hydrolysis source contribution to the
        proton/cation residuals is byte-zeroed (matches the
        disabled-feature path observables to within Newton tolerance).
    r_H_El_pm_func
        R-space ``Function`` holding the cation-specific Singh r_H_El
        distance (pm).  Made live so the Gate 4B sensitivity sweep
        can swap in different values without rebuilding the form.
        Mutable via ``set_reaction_r_H_El_pm_model``.
    gamma_func
        R-space ``Function`` holding the surface coverage Γ_MOH.
        Treated as an external coefficient — the Newton solver reads
        it but does not include it in the monolithic system.  The
        orchestrator updates it via :func:`update_gamma_from_solution`
        between continuation rungs (outer Picard).
    cation_params
        Per-cation Singh 2016 Table S1 row + r_H_El plus solver-side
        switches (anode_clamp).  Held verbatim as a Python dict.
        ``cation_params['r_H_El_pm']`` is the *initial* value mirrored
        into ``r_H_El_pm_func`` at build time; ``r_H_El_pm_func`` is
        the residual-side authoritative value during continuation.
    """

    counterion_idx: int
    h_idx: int
    delta_ohp_func: Any
    k_hyd_func: Any
    k_prot_func: Any
    k_des_func: Any
    lambda_hydrolysis_func: Any
    r_H_El_pm_func: Any
    gamma_func: Any
    cation_params: dict


def is_cation_hydrolysis_enabled(conv_cfg: dict) -> bool:
    """Centralised gate check on ``enable_cation_hydrolysis``.

    Returns False both when the key is missing (legacy params dict)
    and when set to a falsy value, so callers can rely on a uniform
    default-off behaviour.
    """
    return bool(conv_cfg.get("enable_cation_hydrolysis", False))


def resolve_counterion_index(
    roles: Optional[Sequence[str]], *, role_label: str = "counterion",
) -> int:
    """Return the index of the cation species in the dynamic NP list.

    Phase 6β v9 Gate 3B requires explicit ``roles=`` because the
    K2SO4 stack carries multiple ``z=+1`` species (H⁺ + K⁺) and
    inferring "the cation" from charge alone is ambiguous.  Mirrors
    the role-aware fork in
    :func:`Forward.bv_solver.water_ionization.resolve_h_index`.

    Parameters
    ----------
    roles
        Per-species role labels; usually
        ``["neutral", "neutral", "proton", "counterion"]`` for the
        K2SO4 stack.  Required (non-None).
    role_label
        Role string identifying the dynamic counterion.  Default
        ``"counterion"``; case-insensitive.

    Raises
    ------
    ValueError
        ``roles`` is None, no species has the requested role, or
        multiple species share it.
    """
    if roles is None:
        raise ValueError(
            "resolve_counterion_index requires explicit roles= "
            "(no z-inference fallback because the K2SO4 stack has H⁺ "
            "and K⁺ both at z=+1)."
        )
    candidates = [
        i for i, r in enumerate(roles)
        if str(r).strip().lower() == role_label.lower()
    ]
    if not candidates:
        raise ValueError(
            f"resolve_counterion_index: no species with role={role_label!r} "
            f"in roles={list(roles)}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"resolve_counterion_index: multiple species with "
            f"role={role_label!r} in roles={list(roles)} (indices "
            f"{candidates}); roles must be unique."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------


_DEFAULT_CATION_PARAMS = {
    # K⁺ Singh 2016 Table S1 + Cu r_H_El back-fit prior (Gate 4A
    # makes these per-cation; Gate 3B uses K⁺ defaults).
    "z_eff": 0.919,
    "r_M_pm": 138.0,
    "r_H_El_pm": 200.98,
    "A_pm": 620.32,
    "B": 17.154,
    "r_O_pm": 63.0,
    "anode_clamp": True,
    "pKa_bulk": 14.5,
    "pka_shift_form": "placeholder",   # Gate 3B placeholder; Gate 4A → "singh_2016_eq_4"
}


def build_cation_hydrolysis_terms(
    *,
    ctx: dict,
    conv_cfg: dict,
    z_vals: Sequence[float],
    roles: Optional[Sequence[str]],
    h_idx: int,
    R_space: Any,
) -> CationHydrolysisBundle:
    """Build R-space ``Function`` objects + resolve the cation index.

    Reads continuation defaults from
    ``conv_cfg['cation_hydrolysis_config']`` (a sub-dict).

    Parameters
    ----------
    ctx
        Firedrake context.  Mutated in place: sets
        ``ctx['cation_hydrolysis']`` to the returned bundle.
    conv_cfg
        ``solver_options['bv_convergence']`` sub-dict.
    z_vals
        Per-species charge list (used only by guard checks; the
        actual counterion index comes from ``roles``).
    roles
        Per-species role labels (Phase 6β v9 Gate 1).
    h_idx
        Pre-resolved proton index.  Caller passes the same index it
        would use for the proton boundary residual.
    R_space
        ``firedrake.FunctionSpace(mesh, "R", 0)``; reused for every
        mutable scalar.

    Returns
    -------
    CationHydrolysisBundle
    """
    counterion_idx = resolve_counterion_index(roles)

    raw_cfg = conv_cfg.get("cation_hydrolysis_config") or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(
            "cation_hydrolysis_config must be a dict; got "
            f"{type(raw_cfg).__name__}"
        )

    k_hyd_init = float(raw_cfg.get("k_hyd", 0.0))
    k_prot_init = float(raw_cfg.get("k_prot", 0.0))
    k_des_init = float(raw_cfg.get("k_des", 1.0))
    delta_ohp_init = float(raw_cfg.get("delta_ohp_hat", 1.0))
    if delta_ohp_init <= 0.0:
        raise ValueError(
            f"delta_ohp_hat must be positive (got {delta_ohp_init!r}); "
            "δ_OHP is the OHP thickness — non-positive values would "
            "give an undefined Γ/δ surface concentration."
        )
    if k_des_init <= 0.0:
        raise ValueError(
            f"k_des must be positive (got {k_des_init!r}); k_des = 0 "
            "leaves Γ unbounded at λ=1."
        )

    cation_params = dict(_DEFAULT_CATION_PARAMS)
    if "pka_shift_params" in raw_cfg and isinstance(
        raw_cfg["pka_shift_params"], dict
    ):
        cation_params.update(raw_cfg["pka_shift_params"])
    if "pka_shift_form" in raw_cfg:
        cation_params["pka_shift_form"] = str(raw_cfg["pka_shift_form"])

    delta_ohp_func = fd.Function(R_space, name="cation_hydrolysis_delta_ohp")
    delta_ohp_func.assign(delta_ohp_init)

    k_hyd_func = fd.Function(R_space, name="cation_hydrolysis_k_hyd")
    k_hyd_func.assign(k_hyd_init)

    k_prot_func = fd.Function(R_space, name="cation_hydrolysis_k_prot")
    k_prot_func.assign(k_prot_init)

    k_des_func = fd.Function(R_space, name="cation_hydrolysis_k_des")
    k_des_func.assign(k_des_init)

    lambda_init = float(conv_cfg.get("lambda_hydrolysis", 0.0))
    if not (0.0 <= lambda_init <= 1.0):
        raise ValueError(
            f"lambda_hydrolysis must lie in [0, 1] (got {lambda_init!r}); "
            "it is the smooth activation knob ramped 0 → 1 by the "
            "continuation orchestrator."
        )
    lambda_hydrolysis_func = fd.Function(
        R_space, name="cation_hydrolysis_lambda"
    )
    lambda_hydrolysis_func.assign(lambda_init)

    # Γ_MOH external coefficient.  Newton reads this; orchestrator
    # updates it via update_gamma_from_solution after each rung.
    # Initial value 0 — the disabled-feature / λ=0 baseline.
    gamma_func = fd.Function(R_space, name="cation_hydrolysis_gamma")
    gamma_func.assign(0.0)

    # r_H_El_pm — Singh's hydration-shell H to electrode distance.
    # Promoted to a live R-space Function so the Gate 4B sensitivity
    # sweep can swap values without rebuilding the form.
    r_H_El_pm_init = float(cation_params.get("r_H_El_pm", 200.98))
    if r_H_El_pm_init <= 0.0:
        raise ValueError(
            f"r_H_El_pm must be positive (got {r_H_El_pm_init!r})"
        )
    r_H_El_pm_func = fd.Function(
        R_space, name="cation_hydrolysis_r_H_El_pm",
    )
    r_H_El_pm_func.assign(r_H_El_pm_init)

    bundle = CationHydrolysisBundle(
        counterion_idx=int(counterion_idx),
        h_idx=int(h_idx),
        delta_ohp_func=delta_ohp_func,
        k_hyd_func=k_hyd_func,
        k_prot_func=k_prot_func,
        k_des_func=k_des_func,
        lambda_hydrolysis_func=lambda_hydrolysis_func,
        r_H_El_pm_func=r_H_El_pm_func,
        gamma_func=gamma_func,
        cation_params=cation_params,
    )
    ctx["cation_hydrolysis"] = bundle
    return bundle


# ---------------------------------------------------------------------------
# UFL builders — boundary source, ΔpKa
# ---------------------------------------------------------------------------


def build_pka_shift(
    *,
    cation_params: dict,
    sigma_S: Any,
    r_H_El_func: Any = None,
):
    """Return ``ΔpKa(σ_S)`` UFL expression.

    Two paths:

    * ``pka_shift_form='placeholder'`` (Gate 3B) — returns
      ``Constant(0.0)`` (no field-dependent shift).  Used by the
      Gate 3D manufactured-source unit tests where ``R_net`` is
      overridden directly.
    * ``pka_shift_form='singh_2016_eq_4'`` (Gate 4A) — Singh 2016 SI
      Eq. (4'):

      .. code-block:: text

          ΔpKa(σ)  =  +2 · A · z · σ_singh · r_H_El · (1 − r_M-O² / r_H_El²)

      with ``σ_singh = max(0, −σ_S) · (N_A / F) · 1e-24`` (the
      anode-clamped cathode-side counts/pm² value; Singh defines σ as
      a positive scalar magnitude so the anode-clamp keeps ΔpKa = 0
      at anodic bias) and ``r_M-O = r_M + r_O``.  Cathodic case
      (``σ_S < 0`` and ``r_H_El < r_M-O``) gives ``ΔpKa < 0``,
      lowering the hydrolysis pKa (more proton produced) — Singh
      Tables S2/S3 verified per
      ``docs/singh_2016_pka_formula.md`` §3.4 + §7.

    Parameters
    ----------
    cation_params
        Per-cation parameter dict (Singh Table S1 + r_H_El + solver
        switches ``anode_clamp``).
    sigma_S
        Signed Stern surface charge UFL expression in **nondim**
        units.  Caller passes ``stern_coeff * (phi_applied - phi)``
        from the form-build code — that nondim coefficient times the
        nondim potential drop has units of nondim charge density.
        We convert to physical C/m² via the scaling chain on ctx
        before applying Singh's pm-based formula.

    Returns
    -------
    UFL expression for ``ΔpKa`` (dimensionless).
    """
    form = str(cation_params.get("pka_shift_form", "placeholder")).lower()
    if form == "placeholder":
        return fd.Constant(0.0)
    if form == "singh_2016_eq_4":
        return _build_singh_2016_eq_4_pka_shift(
            cation_params=cation_params,
            sigma_S=sigma_S,
            r_H_El_func=r_H_El_func,
        )
    raise NotImplementedError(
        f"pka_shift_form={form!r} not implemented; supported forms are "
        "'placeholder' (Gate 3B) and 'singh_2016_eq_4' (Gate 4A)."
    )


# Avogadro / Faraday for the σ unit conversion in Singh Eq. (4).
# ``N_A · e = F`` (definitional), so ``N_A / F = 1/e``.  We carry the
# canonical value of the elementary charge so this helper stays
# consistent with Nondim.constants' Faraday constant
# (``F = 96485.3329 C/mol`` from CODATA).
_INVERSE_ELEMENTARY_CHARGE = 1.0 / 1.602176634e-19  # 1/e in 1/C ≈ 6.2415e18


def _avogadro_per_faraday() -> float:
    """Return ``N_A / F`` in mol⁻¹ · C⁻¹ = 1/e."""
    return _INVERSE_ELEMENTARY_CHARGE


def _build_singh_2016_eq_4_pka_shift(
    *,
    cation_params: dict,
    sigma_S: Any,
    r_H_El_func: Any = None,
):
    """Build the Singh 2016 SI Eq. (4) ΔpKa UFL expression.

    Singh writes σ in counts/pm² (see ``docs/singh_2016_pka_formula.md``
    §5.2).  ``sigma_S`` is expected in **physical C/m²** (caller in
    ``forms_logc[_muh].py`` multiplies by ``F · C_SCALE · L_SCALE``
    before invoking).

    Anode-clamp: at anodic bias ``σ_S > 0``, no cation accumulation
    at OHP, no hydrolysis driving force; clamp ``ΔpKa → 0`` via
    ``max(0, −σ_S)``.

    ``r_H_El_func`` (optional): when provided (a UFL ``Function`` /
    ``Constant`` from the bundle), its current value is read at
    every Newton step — letting the Gate 4B sweep swap r_H_El at
    run time without rebuilding the form.  When ``None`` we fall
    back to ``cation_params['r_H_El_pm']`` baked as a Constant
    (used by unit tests that don't have a bundle).
    """
    z = float(cation_params["z_eff"])
    r_M = float(cation_params["r_M_pm"])
    r_O = float(cation_params.get("r_O_pm", 63.0))
    r_M_O = r_M + r_O                     # pm (constant from r_M, r_O)
    A_pm = float(cation_params.get("A_pm", 620.32))   # Singh slope (pm)
    anode_clamp = bool(cation_params.get("anode_clamp", True))

    # Live r_H_El (or baked Constant) — picked up by the residual at
    # every Newton step.
    if r_H_El_func is not None:
        r_H_El_expr = r_H_El_func    # UFL Function reference
    else:
        r_H_El_expr = fd.Constant(float(cation_params["r_H_El_pm"]))

    # Singh's σ is in counts/pm² (positive scalar = cathode magnitude).
    counts_per_m2_per_C_per_m2 = _avogadro_per_faraday()
    pm2_per_m2 = 1.0e-24
    sigma_count_per_pm2 = (
        sigma_S * fd.Constant(counts_per_m2_per_C_per_m2 * pm2_per_m2)
    )

    if anode_clamp:
        sigma_singh = fd.max_value(fd.Constant(0.0), -sigma_count_per_pm2)
    else:
        sigma_singh = -sigma_count_per_pm2

    # Geometric factor:  G = 1 − (r_M-O / r_H_El)²
    # When r_H_El < r_M-O (Singh's standard cathode geometry),
    # G < 0 ⇒ ΔpKa < 0 (hydrolysis pKa drops at cathode).
    r_M_O_const = fd.Constant(r_M_O)
    G_expr = fd.Constant(1.0) - (r_M_O_const * r_M_O_const) / (
        r_H_El_expr * r_H_El_expr
    )
    delta_pKa = (
        fd.Constant(2.0 * A_pm * z) * r_H_El_expr * G_expr * sigma_singh
    )
    return delta_pKa


def build_proton_boundary_source(
    *,
    bundle: CationHydrolysisBundle,
    c_M_bdy_expr: Any,
    c_H_bdy_expr: Any,
    pka_shift_expr: Any,
):
    """Return UFL expression for ``R_net`` (the proton boundary source).

    Layout::

        R_net  =  k_hyd · c_M⁺ · 10^(−ΔpKa)
                  −  k_prot · c_H · Γ / δ_OHP

    Γ is read from ``bundle.gamma_func`` (an R-space *coefficient*,
    not a Newton unknown).  The orchestrator updates Γ between
    continuation rungs via :func:`update_gamma_from_solution`.

    Parameters
    ----------
    bundle
        Output of :func:`build_cation_hydrolysis_terms`.
    c_M_bdy_expr
        UFL expression for ``c_M⁺`` evaluated at the electrode
        boundary.  Caller passes ``ci[counterion_idx]`` from the
        forms module.
    c_H_bdy_expr
        UFL expression for ``c_H`` at the electrode boundary.
    pka_shift_expr
        Output of :func:`build_pka_shift`.  Constant(0.0) at Gate 3B.

    Returns
    -------
    UFL expression for ``R_net``.
    """
    pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
    R_forward = bundle.k_hyd_func * c_M_bdy_expr * pka_factor
    R_backward = (
        bundle.k_prot_func
        * c_H_bdy_expr
        * bundle.gamma_func
        / bundle.delta_ohp_func
    )
    return R_forward - R_backward


def build_forward_branch(
    *,
    bundle: CationHydrolysisBundle,
    c_M_bdy_expr: Any,
    pka_shift_expr: Any,
):
    """Return UFL expression for the forward (Γ-independent) branch only.

    ``R_net_forward = k_hyd · c_M⁺ · 10^(−ΔpKa)``.  Used by
    :func:`update_gamma_from_solution` to compute the boundary
    integral that drives the closed-form Γ_ss formula.
    """
    pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
    return bundle.k_hyd_func * c_M_bdy_expr * pka_factor


# ---------------------------------------------------------------------------
# Outer Picard update for Γ
# ---------------------------------------------------------------------------


def update_gamma_from_solution(
    ctx: dict, *, electrode_marker: Optional[int] = None,
) -> float:
    """Update ``bundle.gamma_func`` from the current ``ctx['U']``.

    Closed-form ``Γ_ss(λ)`` solving ``F_Γ = 0`` with the smooth
    λ-blended residual::

        F_Γ  =  λ · (R_net − k_des · Γ) · v_R · ds_e
                − (1 − λ) · Γ · v_R · ds_e

    Substituting ``R_net = k_hyd · c_M · 10^(−ΔpKa) − k_prot · c_H · Γ / δ``
    and integrating over the electrode boundary::

        Γ  =  λ · k_hyd · ⟨c_M · 10^(−ΔpKa)⟩
              /  (λ · k_des + (1 − λ) + λ · k_prot · ⟨c_H⟩ / δ_OHP)

    where ``⟨·⟩`` is the boundary-area-averaged value of the
    integrand on the electrode marker.  The denominator stays
    strictly positive for any ``λ ∈ [0, 1]`` and ``k_des, k_prot ≥ 0``,
    so the formula is unambiguously evaluable.

    At ``λ = 0`` the formula returns ``Γ = 0`` exactly (matches the
    Dirichlet pin invariant from Gate 3D).  At ``λ = 1`` the formula
    returns ``Γ = ⟨R_net_forward⟩ / (k_des + k_prot · ⟨c_H⟩ / δ_OHP)``
    — the closed-form steady state.

    Parameters
    ----------
    ctx
        Firedrake context built with ``enable_cation_hydrolysis=True``.
    electrode_marker
        Boundary marker integer.  Defaults to ``ctx['bv_settings']['electrode_marker']``.

    Returns
    -------
    The new Γ value (also assigned to ``bundle.gamma_func``).
    """
    bundle = ctx.get("cation_hydrolysis")
    if bundle is None:
        raise ValueError(
            "update_gamma_from_solution: ctx has no 'cation_hydrolysis' "
            "bundle.  Was the form built with enable_cation_hydrolysis=True?"
        )

    if electrode_marker is None:
        bv_cfg = ctx.get("bv_settings", {})
        electrode_marker = bv_cfg.get("electrode_marker")
        if electrode_marker is None:
            raise ValueError(
                "update_gamma_from_solution: electrode_marker not on ctx; "
                "pass explicitly or build the form with bv_settings populated."
            )

    mesh = ctx["mesh"]
    ds = fd.Measure("ds", domain=mesh)
    ci = ctx["ci_exprs"]
    c_M_expr = ci[bundle.counterion_idx]
    c_H_expr = ci[bundle.h_idx]

    lam_val = float(bundle.lambda_hydrolysis_func)
    if lam_val == 0.0:
        # Hard zero — avoids a needless boundary integral when the
        # disabled-feature baseline is being driven.
        bundle.gamma_func.assign(0.0)
        return 0.0

    k_hyd = float(bundle.k_hyd_func)
    k_prot = float(bundle.k_prot_func)
    k_des = float(bundle.k_des_func)
    delta_ohp = float(bundle.delta_ohp_func)

    area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
    if area <= 0.0:
        raise RuntimeError(
            "update_gamma_from_solution: electrode boundary has zero "
            f"area (marker={electrode_marker})."
        )

    # Gate 3D — manufactured-source override: when ``manufactured_R_inj``
    # is set in conv_cfg, R_net in the residual is replaced by a fixed
    # constant.  The Picard formula must use the same R_inj so the Γ
    # steady state matches what Newton sees.
    bv_conv = ctx.get("bv_convergence", {})
    manufactured_R_inj = bv_conv.get("manufactured_R_inj") if isinstance(
        bv_conv, dict
    ) else None

    if manufactured_R_inj is not None:
        # R_net = const (Γ-independent).  Steady state at λ-blended
        # residual ``λ·(R_inj − k_des·Γ) − (1−λ)·Γ = 0`` gives
        # ``Γ = λ·R_inj / (λ·k_des + (1−λ))``.
        R_inj_const = float(manufactured_R_inj)
        numerator = lam_val * R_inj_const
        denominator = lam_val * k_des + (1.0 - lam_val)
    else:
        # Physical R_net: integrate the forward branch and ⟨c_H⟩
        # against the boundary measure for the closed-form formula.
        sigma_S_expr = ctx.get("_cation_hydrolysis_sigma_S_expr")
        if sigma_S_expr is None:
            sigma_S_expr = fd.Constant(0.0)
        pka_shift_expr = build_pka_shift(
            cation_params=bundle.cation_params,
            sigma_S=sigma_S_expr,
            r_H_El_func=bundle.r_H_El_pm_func,
        )
        pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
        forward_avg = float(
            fd.assemble(c_M_expr * pka_factor * ds(electrode_marker))
        ) / area
        c_H_avg = float(fd.assemble(c_H_expr * ds(electrode_marker))) / area

        numerator = lam_val * k_hyd * forward_avg
        denominator = (
            lam_val * k_des
            + (1.0 - lam_val)
            + lam_val * k_prot * c_H_avg / delta_ohp
        )
    if denominator <= 0.0:
        raise RuntimeError(
            "update_gamma_from_solution: denominator non-positive "
            f"(λ={lam_val}, k_des={k_des}, k_prot={k_prot}).  Check "
            "kinetic rates and λ are configured consistently."
        )
    gamma_new = numerator / denominator
    bundle.gamma_func.assign(gamma_new)
    return float(gamma_new)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def extract_gamma_value(ctx: dict) -> float:
    """Read the current Γ_MOH scalar value from the bundle's R-space Function.

    Convenience accessor for Gate 4 diagnostics.  Returns ``Γ`` as a
    plain Python ``float``.  Raises ``ValueError`` if the bundle is
    not present (feature disabled).
    """
    bundle = ctx.get("cation_hydrolysis")
    if bundle is None:
        raise ValueError(
            "extract_gamma_value: ctx has no 'cation_hydrolysis' bundle; "
            "was the form built with enable_cation_hydrolysis=True?"
        )
    return float(bundle.gamma_func)


__all__ = [
    "CationHydrolysisBundle",
    "build_cation_hydrolysis_terms",
    "build_forward_branch",
    "build_pka_shift",
    "build_proton_boundary_source",
    "extract_gamma_value",
    "is_cation_hydrolysis_enabled",
    "resolve_counterion_index",
    "update_gamma_from_solution",
]
