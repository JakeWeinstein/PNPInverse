"""Cation hydrolysis closure (Phase 6β v9 Gates 3 + 4, v10a Langmuir cap).

Adds a global surface coverage ``Γ_MOH`` representing the cation-
hydrolysis reservoir at the OHP::

    M⁺(H₂O)ₙ + H₂O  ⇌  MOH⁰(H₂O)ₙ₋₁ + H₃O⁺                     (Singh 2016 Eq. 2)

with field-dependent rate balance::

    R_net  =  k_hyd · c_M⁺(0) · 10^(−ΔpKa(σ_S)) · (1 − Γ/Γ_max)
              −  k_prot · c_H(0) · Γ / δ_OHP                       (forward − reverse)
    R_des  =  k_des · Γ                                            (desorption)

The proton boundary residual gains a source ``+λ · R_net`` (one H⁺
produced per hydrolysis event); the cation boundary residual gains a
sink ``−λ · R_net``.  The ``(1 − Γ/Γ_max)`` vacancy factor (Phase 6β
v10a, 2026-05-10) caps surface coverage at one monolayer of MOH at
the OHP: without it, every v9 result at converged ``k_hyd ≥ 1e-3`` sat
at Γ ≈ 6+ monolayers (physically invalid; ~64 monolayers at
``k_hyd=1e-2``).  Steady state for Γ becomes::

    Γ_ss(λ)  =  λ · F₀
                /  (λ · k_des + (1 − λ) + λ · k_prot · ⟨c_H⟩ / δ_OHP
                    + λ · F₀ / Γ_max)

where ``F₀ = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩`` is the boundary-area-averaged
forward forcing and ``⟨c_H⟩`` is the boundary-averaged proton
concentration.  In the limit ``Γ_max → ∞`` the cap term ``λ · F₀ / Γ_max``
vanishes and the formula reduces to the v9 Γ_ss expression.

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

import math
import warnings
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
    gamma_max_func
        R-space ``Function`` holding the Langmuir saturation cap
        ``Γ_max`` (Phase 6β v10a, 2026-05-10).  The residual reads
        ``(1 − Γ / Γ_max)`` as the vacancy factor on the forward
        branch so coverage saturates at one monolayer.  Mutable via
        the ``set_reaction_gamma_max_model`` accessor.  Default at
        bundle build time is ``cation_hydrolysis_config['gamma_max_nondim']``
        or the smoke baseline ``0.047`` (1 monolayer of MOH at the
        OHP; ``5.6e-6 mol/m² / (C_SCALE · L_REF)``).  The v10b
        literature-calibration step will replace the default.
    cation_params
        Per-cation Singh 2016 Table S1 row + r_H_El plus solver-side
        switches (anode_clamp).  Held verbatim as a Python dict.
        ``cation_params['r_H_El_pm']`` is the *initial* value mirrored
        into ``r_H_El_pm_func`` at build time; ``r_H_El_pm_func`` is
        the residual-side authoritative value during continuation.
        ``cation_params['beta_offset_pm2']`` mirrors the live β offset
        for diagnostics consumers; the residual reads
        ``beta_offset_pm2_func`` directly.
    beta_offset_pm2_func
        Phase 6β step 10 Phase D — carbon-vs-Cu β offset in pm².  Sums
        with the live β_per_cation (computed from r_H_El_pm_func) inside
        ``_build_singh_2016_eq_4_pka_shift`` to give ``β_carbon =
        β_per_cation_Cu(r_H_El_live) + Δ_β``.  Default 0.0 at build
        time → byte-equivalent to the pre-D1 v10a/v10b residual.
        Mutable via :func:`set_reaction_beta_offset_pm2_model`.
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
    gamma_max_func: Any
    cation_params: dict
    beta_offset_pm2_func: Any


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


# Phase 6β v10a Langmuir cap smoke baseline (frozen historical).
# One monolayer of hydrated MOH at the OHP, hard-sphere areal coverage
# with r ≈ 2.3 Å (matches the K⁺ hydrated-radius monolayer):
#     Γ_max_phys  ≈ 1 / (π · (2.3e-10 m)² · N_A) ≈ 5.6e-6 mol/m²
# Nondim conversion uses the same scaling chain as ``c_HAT · L_HAT``:
#     Γ_max_hat = Γ_max_phys / (C_SCALE · L_REF)
#                = 5.6e-6 / (1.2 · 1e-4)
#                ≈ 0.047
# v10b (2026-05-10) tightens this V10A derivation chain rather than
# replacing the value -- the 4-test compatibility check finds no peer-
# reviewed source that reports MOH adsorbate coverage at the OHP in
# K2SO4 / sp2-carbon (Singh 2016 reports K_eq; Iamprasertkun 2019
# reports HOPG specific capacitance; Bohra 2019 uses variable Booth
# permittivity).  GAMMA_MAX_HAT_V10B = GAMMA_MAX_HAT_V10A_SMOKE = 0.047.
GAMMA_MAX_HAT_V10A_SMOKE: float = 0.047

# v10b literature-calibrated constants imported from the top-level
# Firedrake-free ``calibration`` package.  See
# ``calibration/v10b.py`` and
# ``docs/phase6/v10b_calibration_summary.md`` for the per-parameter
# decision rules + provenance.
from calibration.v10b import (
    GAMMA_MAX_HAT_V10B,
    K_DES_NONDIM_V10B,
    V10B_CALIBRATION_METADATA,
)

# DEPRECATED 2026-05-10: GAMMA_MAX_HAT_SMOKE is preserved as an alias
# to GAMMA_MAX_HAT_V10A_SMOKE (frozen historical 0.047) for one-cycle
# backward compatibility with v9/v10a callers.  NEW callers MUST use
# GAMMA_MAX_HAT_V10B (production) or GAMMA_MAX_HAT_V10A_SMOKE
# (explicit historical reproduction).  NEVER alias SMOKE to a V10B
# value -- that is silent provenance theft.  Removal scheduled post-
# step-9 (B.2).
GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10A_SMOKE


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
    k_des_init = float(raw_cfg.get("k_des", K_DES_NONDIM_V10B))
    delta_ohp_init = float(raw_cfg.get("delta_ohp_hat", 1.0))
    gamma_max_init = float(
        raw_cfg.get("gamma_max_nondim", GAMMA_MAX_HAT_V10B)
    )
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
    if gamma_max_init <= 0.0:
        raise ValueError(
            f"gamma_max_nondim must be positive (got {gamma_max_init!r}); "
            "Γ_max = 0 forces the vacancy factor (1 − Γ/Γ_max) to "
            "diverge."
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

    # Phase 6β v10a Langmuir saturation cap.  Mutable R-space Function
    # so the v10b literature-calibration sweep + diagnostics drivers can
    # swap values without rebuilding the form.
    gamma_max_func = fd.Function(
        R_space, name="cation_hydrolysis_gamma_max",
    )
    gamma_max_func.assign(gamma_max_init)

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

    # Phase 6β step 10 Phase D — Δ_β offset (carbon-vs-Cu pKa shift
    # coefficient).  Default 0.0 keeps the residual byte-equivalent to
    # the pre-D1 v10b stack.  Live coefficient: re-read at every Newton
    # resolve so the Phase D fit's outer loop can sweep Δ_β without
    # rebuilding the form.  Sign convention: positive Δ_β raises
    # ``β_carbon`` (less negative under Singh's cathodic geometry,
    # smaller |ΔpKa|).
    beta_offset_init = float(raw_cfg.get("beta_offset_pm2", 0.0))
    beta_offset_pm2_func = fd.Function(
        R_space, name="cation_hydrolysis_beta_offset_pm2",
    )
    beta_offset_pm2_func.assign(beta_offset_init)
    # Mirror initial offset into cation_params so diagnostics see it.
    cation_params["beta_offset_pm2"] = beta_offset_init

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
        gamma_max_func=gamma_max_func,
        cation_params=cation_params,
        beta_offset_pm2_func=beta_offset_pm2_func,
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
    beta_offset_pm2_func: Any = None,
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

          ΔpKa(σ)  =  +(β_per_cation_Cu + Δ_β) · σ_singh

      where
      ``β_per_cation_Cu = 2 · A · z · r_H_El · (1 − r_M-O² / r_H_El²)``
      and ``σ_singh = max(0, −σ_S) · (N_A / F) · 1e-24`` (the
      anode-clamped cathode-side counts/pm² value; Singh defines σ as
      a positive scalar magnitude so the anode-clamp keeps ΔpKa = 0
      at anodic bias) and ``r_M-O = r_M + r_O``.  Cathodic case
      (``σ_S < 0`` and ``r_H_El < r_M-O``) gives ``β_per_cation_Cu <
      0`` ⇒ ``ΔpKa < 0`` (Singh Tables S2/S3 verified per
      ``docs/singh_2016_pka_formula.md`` §3.4 + §7).  The Phase D
      ``Δ_β`` offset (read live from ``beta_offset_pm2_func``)
      defaults to 0 and the residual is byte-equivalent to the v10b
      stack; the Phase D fit's outer loop sweeps Δ_β.

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
    beta_offset_pm2_func
        Optional R-space ``Function`` holding the Phase 6β step 10
        Phase D Δ_β offset in pm² (carbon-vs-Cu pKa-shift coefficient
        offset).  When ``None`` (default), the residual collapses to
        ``β_per_cation_Cu · σ_singh`` byte-equivalent to the pre-D1
        v10b path.  Re-read at each Newton resolve.

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
            beta_offset_pm2_func=beta_offset_pm2_func,
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
    beta_offset_pm2_func: Any = None,
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

    ``beta_offset_pm2_func`` (Phase 6β step 10 Phase D): optional
    R-space ``Function`` holding the Δ_β carbon-vs-Cu offset (pm²).
    When provided, the residual builds
    ``β_carbon = β_per_cation_Cu(r_H_El_live) + Δ_β`` and emits
    ``ΔpKa = β_carbon · σ_singh``.  When ``None``, the residual
    collapses to the pre-D1 expression ``β_per_cation_Cu · σ_singh``
    — byte-equivalent to the v10b path.
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
    # β_per_cation_Cu in pm² — live UFL expression that tracks the
    # current r_H_El_pm value.  At r_H_El_pm = 200.98 (K+ Cu default)
    # this evaluates numerically to -45.608196 pm², matching
    # ``calibration.singh2016.compute_beta_per_cation("K+")``.
    beta_per_cation_live = (
        fd.Constant(2.0 * A_pm * z) * r_H_El_expr * G_expr
    )
    # Phase 6β step 10 Phase D — add the carbon-vs-Cu offset when
    # provided.  At Δ_β=0 the UFL `x + 0` is numerically equivalent
    # to `x`, so the residual at offset=0 matches the pre-D1 path
    # to machine precision.
    if beta_offset_pm2_func is not None:
        beta_carbon_expr = beta_per_cation_live + beta_offset_pm2_func
    else:
        beta_carbon_expr = beta_per_cation_live
    delta_pKa = beta_carbon_expr * sigma_singh
    return delta_pKa


def build_proton_boundary_source(
    *,
    bundle: CationHydrolysisBundle,
    c_M_bdy_expr: Any,
    c_H_bdy_expr: Any,
    pka_shift_expr: Any,
):
    """Return UFL expression for ``R_net`` (the proton boundary source).

    Layout (Phase 6β v10a)::

        R_net  =  k_hyd · c_M⁺ · 10^(−ΔpKa) · (1 − Γ/Γ_max)
                  −  k_prot · c_H · Γ / δ_OHP

    Γ is read from ``bundle.gamma_func`` (an R-space *coefficient*,
    not a Newton unknown).  The orchestrator updates Γ between
    continuation rungs via :func:`update_gamma_from_solution`.  The
    Langmuir vacancy factor ``(1 − Γ/Γ_max)`` caps the forward rate
    once the OHP is saturated with adsorbed MOH (v10a — 2026-05-10).
    In the limit ``Γ_max → ∞`` the factor goes to 1 and the residual
    is byte-equivalent to the v9 formulation.

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
    vacancy_factor = (
        fd.Constant(1.0) - bundle.gamma_func / bundle.gamma_max_func
    )
    R_forward = (
        bundle.k_hyd_func * c_M_bdy_expr * pka_factor * vacancy_factor
    )
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
    """Return UFL expression for the forward branch including the Langmuir cap.

    Phase 6β v10a::

        R_net_forward = k_hyd · c_M⁺ · 10^(−ΔpKa) · (1 − Γ/Γ_max)

    Used by :func:`update_gamma_from_solution` to compute the
    boundary integral that drives the closed-form Γ_ss formula.
    Returning the capped branch (rather than the uncapped ``F₀``) is
    deliberate — the rung_callback diagnostics expose both the
    average of this expression (``R_forward_capped``) and the
    average of the uncapped ``F₀`` separately so observers can see
    how close the OHP is to saturation.
    """
    pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
    vacancy_factor = (
        fd.Constant(1.0) - bundle.gamma_func / bundle.gamma_max_func
    )
    return (
        bundle.k_hyd_func * c_M_bdy_expr * pka_factor * vacancy_factor
    )


def build_forward_branch_uncapped(
    *,
    bundle: CationHydrolysisBundle,
    c_M_bdy_expr: Any,
    pka_shift_expr: Any,
):
    """Return UFL expression for the Γ-independent forward forcing ``F₀``.

    ``F₀ = k_hyd · c_M⁺ · 10^(−ΔpKa)`` — the v9 forward branch, used
    by the Langmuir closed-form Γ_ss formula and by diagnostics that
    want to see the saturated-coverage limit.  Independent of Γ.
    """
    pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
    return bundle.k_hyd_func * c_M_bdy_expr * pka_factor


# ---------------------------------------------------------------------------
# Outer Picard update for Γ
# ---------------------------------------------------------------------------


def gamma_ss_langmuir(
    *,
    lambda_val: float,
    k_hyd: float,
    k_prot: float,
    k_des: float,
    delta_ohp: float,
    forward_avg: float,
    c_H_avg: float,
    gamma_max: float,
) -> tuple:
    """Pure-Python closed-form Γ_ss for the Langmuir-capped residual.

    Phase 6β v10a — the formula that
    :func:`update_gamma_from_solution` evaluates after assembling the
    boundary integrals on the converged FE state.  Lifted to a
    standalone helper so unit tests can exercise the arithmetic
    without standing up a Firedrake mesh.

    With ``F₀ = k_hyd · forward_avg`` the closed form is::

        Γ_ss(λ) = λ · F₀ / (
            λ · k_des
            + (1 − λ)
            + λ · k_prot · ⟨c_H⟩ / δ_OHP
            + λ · F₀ / Γ_max
        )

    Returns
    -------
    tuple
        ``(gamma_clamped, gamma_unclamped, denominator_terms)`` where
        ``denominator_terms`` is a dict with keys
        ``{'constant', 'kdes', 'kprot', 'cap', 'total'}`` so tests
        can verify the v9-equivalence and saturation limits term by
        term.

    Raises
    ------
    ValueError
        If ``gamma_max <= 0`` (the Langmuir cap is required) or any
        kinetic rate is negative.  The closed form is undefined
        outside those bounds.
    RuntimeError
        Denominator collapses to a non-positive value — should never
        happen for non-negative kinetic rates and ``λ ∈ [0, 1]``, so
        signals a calling-site bug.
    """
    if gamma_max <= 0.0:
        raise ValueError(
            f"gamma_max must be positive (got {gamma_max!r})"
        )
    if not (0.0 <= lambda_val <= 1.0):
        raise ValueError(
            f"lambda_val must lie in [0, 1] (got {lambda_val!r})"
        )
    for name, val in (
        ("k_hyd", k_hyd), ("k_prot", k_prot), ("k_des", k_des),
    ):
        if val < 0.0:
            raise ValueError(f"{name} must be non-negative (got {val!r})")
    if delta_ohp <= 0.0:
        raise ValueError(
            f"delta_ohp must be positive (got {delta_ohp!r})"
        )

    F0 = k_hyd * forward_avg
    numerator = lambda_val * F0
    denom_constant = 1.0 - lambda_val
    denom_kdes = lambda_val * k_des
    denom_kprot = lambda_val * k_prot * c_H_avg / delta_ohp
    denom_cap = lambda_val * F0 / gamma_max
    denominator = denom_constant + denom_kdes + denom_kprot + denom_cap
    if denominator <= 0.0:
        raise RuntimeError(
            "gamma_ss_langmuir: denominator non-positive "
            f"(λ={lambda_val}, k_des={k_des}, k_prot={k_prot}, "
            f"Γ_max={gamma_max})"
        )
    gamma_unclamped = numerator / denominator
    gamma_clamped = max(0.0, min(gamma_max, gamma_unclamped))
    return gamma_clamped, gamma_unclamped, {
        "constant": denom_constant,
        "kdes": denom_kdes,
        "kprot": denom_kprot,
        "cap": denom_cap,
        "total": denominator,
    }


def update_gamma_from_solution(
    ctx: dict, *, electrode_marker: Optional[int] = None,
) -> float:
    """Update ``bundle.gamma_func`` from the current ``ctx['U']``.

    Phase 6β v10a — Langmuir-capped closed form.  Solving the smooth
    λ-blended residual at steady state::

        F_Γ  =  λ · (R_net − k_des · Γ) · v_R · ds_e
                − (1 − λ) · Γ · v_R · ds_e

    with the capped forward branch
    ``R_forward = F₀ · (1 − Γ/Γ_max)`` (Phase 6β v10a vacancy factor)
    and the unchanged Γ-linear reverse branch
    ``R_back = k_prot · c_H · Γ / δ`` gives::

        Γ_ss(λ)  =  λ · F₀
                    /  (λ · k_des + (1 − λ)
                        + λ · k_prot · ⟨c_H⟩ / δ_OHP
                        + λ · F₀ / Γ_max)

    where ``F₀ = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩`` is the boundary-area
    average of the uncapped forward forcing.  In the limit
    ``Γ_max → ∞`` the cap term vanishes and the v9 formula is
    recovered; in the limit ``F₀ → ∞`` (very fast hydrolysis)
    Γ_ss → Γ_max so coverage saturates at one monolayer instead of
    diverging as in v9.

    Sanity:

    * λ → 0 ⇒ Γ_ss → 0 (Dirichlet pin invariant from Gate 3D
      survives the Langmuir cap).
    * Γ_max → ∞ ⇒ ``λ · F₀ / Γ_max → 0`` (byte-recovers v9 formula).
    * F₀ → ∞ at fixed λ > 0 ⇒ Γ_ss → Γ_max (saturation).

    After the closed-form evaluation the result is clamped into
    ``[0, Γ_max]``.  The denominator's structure already keeps Γ in
    that interval for non-negative kinetic rates, so any clamp event
    signals either a numerical issue (e.g. a stale boundary average
    from a non-converged Newton state) or a formula bug.  A clamp
    emits ``RuntimeWarning`` so observers can see it during the
    Picard loop; warm-restart clamps (called by the orchestrator
    *before* the Picard loop, with the Function already populated by
    a previous sp) are silent and handled separately.

    Parameters
    ----------
    ctx
        Firedrake context built with ``enable_cation_hydrolysis=True``.
    electrode_marker
        Boundary marker integer.  Defaults to
        ``ctx['bv_settings']['electrode_marker']``.

    Returns
    -------
    The new (clamped) Γ value (also assigned to ``bundle.gamma_func``).
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

    gamma_max = float(bundle.gamma_max_func)
    if gamma_max <= 0.0:
        raise RuntimeError(
            "update_gamma_from_solution: bundle.gamma_max_func ≤ 0 "
            f"(got {gamma_max}); Γ_max must be strictly positive."
        )

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
    # steady state matches what Newton sees.  Because the manufactured
    # override replaces *the whole* R_net (forward AND backward) with a
    # Γ-independent constant in the residual, the closed-form Γ_ss has
    # NO Langmuir cap term: the cap only enters through the forward
    # branch, which the override bypasses.  Keeping the Picard formula
    # in lockstep with what Newton sees is the load-bearing invariant.
    bv_conv = ctx.get("bv_convergence", {})
    manufactured_R_inj = bv_conv.get("manufactured_R_inj") if isinstance(
        bv_conv, dict
    ) else None

    if manufactured_R_inj is not None:
        # R_net = const (Γ-independent).  Steady state at λ-blended
        # residual ``λ·(R_inj − k_des·Γ) − (1−λ)·Γ = 0`` gives
        # ``Γ = λ·R_inj / (λ·k_des + (1−λ))``.  No Langmuir cap term —
        # the override bypasses the forward branch that carries the
        # cap, so the Picard formula must match.  Manufactured R_inj < 0
        # is a *legitimate* test fixture (it drives Γ negative on
        # purpose to check the proton-sink sign convention); the clamp
        # silently pulls Γ into ``[0, Γ_max]`` and we do NOT emit a
        # RuntimeWarning here.
        R_inj_const = float(manufactured_R_inj)
        numerator = lam_val * R_inj_const
        denominator = lam_val * k_des + (1.0 - lam_val)
        if denominator <= 0.0:
            raise RuntimeError(
                "update_gamma_from_solution: denominator non-positive "
                f"(λ={lam_val}, k_des={k_des}, manufactured branch)."
            )
        gamma_unclamped = numerator / denominator
        gamma_clamped = max(0.0, min(gamma_max, gamma_unclamped))
    else:
        # Physical R_net: integrate the uncapped forward branch (F₀)
        # and ⟨c_H⟩ against the boundary measure, then delegate to the
        # pure-Python closed-form helper.
        #
        # Phase 6β step 6 — consume ctx-stored ``_cation_hydrolysis_
        # pka_shift_expr`` so Picard sees the same pKa expression as
        # the residual (R3 #8 single source of truth).  Backward-
        # compat fallback for legacy callers that built ctx without
        # the step 6 artifacts.
        pka_shift_expr = ctx.get("_cation_hydrolysis_pka_shift_expr")
        if pka_shift_expr is None:
            sigma_S_expr = ctx.get("_cation_hydrolysis_sigma_S_expr")
            if sigma_S_expr is None:
                sigma_S_expr = fd.Constant(0.0)
            pka_shift_expr = build_pka_shift(
                cation_params=bundle.cation_params,
                sigma_S=sigma_S_expr,
                r_H_El_func=bundle.r_H_El_pm_func,
                beta_offset_pm2_func=bundle.beta_offset_pm2_func,
            )
        pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
        forward_avg = float(
            fd.assemble(c_M_expr * pka_factor * ds(electrode_marker))
        ) / area
        c_H_avg = float(fd.assemble(c_H_expr * ds(electrode_marker))) / area

        gamma_clamped, gamma_unclamped, _ = gamma_ss_langmuir(
            lambda_val=lam_val,
            k_hyd=k_hyd,
            k_prot=k_prot,
            k_des=k_des,
            delta_ohp=delta_ohp,
            forward_avg=forward_avg,
            c_H_avg=c_H_avg,
            gamma_max=gamma_max,
        )
        # Only the physical path raises a warning on clamp — the
        # closed-form Langmuir formula should respect [0, Γ_max] for
        # non-negative kinetic rates and λ ∈ [0, 1], so a clamp event
        # here signals a numerical issue (stale boundary average from
        # a non-converged Newton state) or a formula bug.
        if gamma_clamped != gamma_unclamped:
            warnings.warn(
                "update_gamma_from_solution: clamped Γ "
                f"{gamma_unclamped:.6e} → {gamma_clamped:.6e} "
                f"(out of [0, {gamma_max}]; λ={lam_val}, "
                f"k_hyd={k_hyd}, k_prot={k_prot}, k_des={k_des}).  "
                "The closed-form formula should never leave bounds — "
                "investigate the boundary averages.",
                RuntimeWarning,
                stacklevel=2,
            )
    bundle.gamma_func.assign(gamma_clamped)
    return float(gamma_clamped)


def clamp_gamma_to_max(ctx: dict) -> float:
    """Clamp ``bundle.gamma_func`` into ``[0, Γ_max]`` silently.

    Helper for warm-restart paths in the orchestrator: when a fresh
    ctx is built with a smaller ``Γ_max`` than the snapshot, the
    restored Γ value may sit above the new cap.  The orchestrator
    calls this *before* the Picard loop runs so the first Newton
    solve sees a vacancy factor in ``[0, 1]``.  Unlike
    :func:`update_gamma_from_solution`, no warning is emitted — this
    clamp is *expected* on cross-parameter restores.

    Returns the new clamped Γ value as a float.
    """
    bundle = ctx.get("cation_hydrolysis")
    if bundle is None:
        return 0.0
    gamma = float(bundle.gamma_func)
    gamma_max = float(bundle.gamma_max_func)
    if gamma_max <= 0.0:
        return gamma
    gamma_clamped = max(0.0, min(gamma_max, gamma))
    if gamma_clamped != gamma:
        bundle.gamma_func.assign(gamma_clamped)
    return float(gamma_clamped)


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


def collect_v10a_rung_diagnostics(
    ctx: dict, *, electrode_marker: Optional[int] = None,
) -> dict:
    """Build the v10a diagnostic payload for a single rung_callback.

    Phase 6β v10a — extends the rung_diag dict that
    :func:`Forward.bv_solver.anchor_continuation.solve_anchor_with_continuation`
    and :func:`solve_lambda_ramp_from_warm_start` emit per rung.
    Captures everything an observer needs to reason about whether the
    Langmuir cap is active and how the closed-form Γ_ss denominator
    decomposes:

    * ``F0_avg`` — boundary-averaged uncapped forward forcing
      ``k_hyd · ⟨c_M · 10^(−ΔpKa)⟩``.
    * ``gamma`` — current Γ value (post-Picard).
    * ``gamma_max`` — current Γ_max cap.
    * ``theta`` — fractional coverage ``Γ / Γ_max``.
    * ``R_forward_capped`` — boundary-averaged capped forward branch
      ``F₀ · (1 − θ)``.  (Diagnostic, not residual-side.)
    * ``denominator_*`` — each term of the Langmuir denominator
      (constant, k_des, k_prot proton-flux, F₀/Γ_max).  Their sum is
      ``denominator_total``.
    * ``R_2e_current`` / ``R_4e_current`` — per-reaction nondim
      current densities (no I_SCALE).  Returns ``None`` if the
      reaction layout is not the expected 2-reaction parallel set.
    * ``sigma_S_C_per_m2`` — boundary-averaged signed Stern surface
      charge in C/m² (physical).  ``None`` when Stern is disabled.
    * ``sigma_S_counts_per_pm2`` — same converted via
      :func:`Forward.bv_solver.units.sigma_C_m2_to_counts_pm2`.

    Returns an empty dict if the cation_hydrolysis bundle is missing
    (the caller will append a "rung_callback_error"-style note rather
    than crashing).
    """
    bundle = ctx.get("cation_hydrolysis")
    if bundle is None:
        return {}

    diag: dict = {}
    try:
        from .units import sigma_C_m2_to_counts_pm2
    except Exception:    # pragma: no cover — defensive
        sigma_C_m2_to_counts_pm2 = None       # type: ignore[assignment]

    if electrode_marker is None:
        bv_cfg = ctx.get("bv_settings", {})
        electrode_marker = bv_cfg.get("electrode_marker")
    if electrode_marker is None:
        return {}

    mesh = ctx.get("mesh")
    if mesh is None:
        return {}

    ds = fd.Measure("ds", domain=mesh)
    try:
        area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
    except Exception as exc:
        diag["v10a_diag_error"] = f"{type(exc).__name__}: {exc}"
        return diag
    if area <= 0.0:
        diag["v10a_diag_error"] = "electrode boundary has zero area"
        return diag

    gamma = float(bundle.gamma_func)
    gamma_max = float(bundle.gamma_max_func)
    lam_val = float(bundle.lambda_hydrolysis_func)
    k_hyd = float(bundle.k_hyd_func)
    k_prot = float(bundle.k_prot_func)
    k_des = float(bundle.k_des_func)
    delta_ohp = float(bundle.delta_ohp_func)

    diag["gamma"] = gamma
    diag["gamma_max"] = gamma_max
    diag["theta"] = gamma / gamma_max if gamma_max > 0.0 else None
    diag["lambda_hydrolysis"] = lam_val
    diag["k_hyd"] = k_hyd
    diag["k_prot"] = k_prot
    diag["k_des"] = k_des
    diag["delta_ohp_hat"] = delta_ohp

    ci = ctx.get("ci_exprs")
    if ci is None:
        diag["v10a_diag_error"] = "ctx missing ci_exprs"
        return diag

    c_M_expr = ci[bundle.counterion_idx]
    c_H_expr = ci[bundle.h_idx]
    sigma_S_expr = ctx.get("_cation_hydrolysis_sigma_S_expr")
    if sigma_S_expr is None:
        sigma_S_expr = fd.Constant(0.0)

    # Always-emitted: top-level boundary concentrations + Stern σ.
    # These are independent of manufactured/physical mode (R5 #3).
    try:
        c_H_boundary_avg = (
            float(fd.assemble(c_H_expr * ds(electrode_marker))) / area
        )
        c_K_boundary_avg = (
            float(fd.assemble(c_M_expr * ds(electrode_marker))) / area
        )
        sigma_S_avg = (
            float(fd.assemble(sigma_S_expr * ds(electrode_marker))) / area
        )
    except Exception as exc:
        diag["v10a_diag_error"] = f"{type(exc).__name__}: {exc}"
        return diag

    diag["c_H_boundary_avg"] = c_H_boundary_avg
    diag["c_K_boundary_avg"] = c_K_boundary_avg
    diag["sigma_S_C_per_m2"] = sigma_S_avg
    if sigma_C_m2_to_counts_pm2 is not None:
        try:
            diag["sigma_S_counts_per_pm2"] = sigma_C_m2_to_counts_pm2(
                sigma_S_avg
            )
        except Exception as exc:
            diag["sigma_S_counts_per_pm2_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    # Per-reaction current contributions (always emitted; used by the
    # Phase A.2 / D / E ablation matrix to confirm the parallel-2e +
    # 4e topology behaves as expected as Γ saturates).
    bv_rate_exprs = ctx.get("bv_rate_exprs") or []
    try:
        for idx, R_expr in enumerate(bv_rate_exprs):
            key = (
                "R_2e_current_nondim"
                if idx == 0
                else "R_4e_current_nondim"
                if idx == 1
                else f"R_{idx}_current_nondim"
            )
            diag[key] = float(
                fd.assemble(R_expr * ds(electrode_marker))
            )
    except Exception as exc:
        diag["per_reaction_current_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Phase 6β step 6 — manufactured mode bypasses the physical R_net
    # path entirely (apply_h_source / apply_k_sink half-physical
    # ablations require manufactured_R_inj to keep Picard consistent;
    # see config.py cross-validation).  Set physical-path fields to
    # None so downstream JSON readers don't mistake a manufactured
    # rung for a physical one.
    bv_conv = ctx.get("bv_convergence", {})
    manufactured_R_inj = bv_conv.get("manufactured_R_inj", None)
    apply_h_source = bv_conv.get("apply_h_source", True)
    apply_k_sink = bv_conv.get("apply_k_sink", True)

    if manufactured_R_inj is not None:
        diag["manufactured_run"] = True
        diag["manufactured_R_inj"] = float(manufactured_R_inj)
        diag["apply_h_source_active"] = bool(apply_h_source)
        diag["apply_k_sink_active"] = bool(apply_k_sink)
        for key in (
            "F0_avg", "forward_avg_no_k_hyd", "c_H_avg", "pka_shift_avg",
            "R_forward_capped",
            "denominator_constant", "denominator_kdes",
            "denominator_kprot", "denominator_cap", "denominator_total",
            "denominator_cap_to_total_ratio", "numerator",
            "F0_decomposition", "R_4e_decomposition_log",
        ):
            diag[key] = None
        return diag

    diag["manufactured_run"] = False

    # Physical-path: consume ctx-stored pKa shift expression so
    # residual, Picard, and diagnostics all see the same UFL object
    # (R3 #8 single source of truth).  Backward-compat fallback for
    # legacy callers that built ctx without the step 6 artifacts.
    pka_shift_expr = ctx.get("_cation_hydrolysis_pka_shift_expr")
    if pka_shift_expr is None:
        pka_shift_expr = build_pka_shift(
            cation_params=bundle.cation_params,
            sigma_S=sigma_S_expr,
            r_H_El_func=bundle.r_H_El_pm_func,
            beta_offset_pm2_func=bundle.beta_offset_pm2_func,
        )
    pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)

    try:
        forward_avg = float(
            fd.assemble(c_M_expr * pka_factor * ds(electrode_marker))
        ) / area
        pka_shift_avg = (
            float(fd.assemble(pka_shift_expr * ds(electrode_marker))) / area
        )
    except Exception as exc:
        diag["v10a_diag_error"] = f"{type(exc).__name__}: {exc}"
        return diag

    c_H_avg = c_H_boundary_avg  # alias retained for v10a record compat.

    F0 = k_hyd * forward_avg
    diag["F0_avg"] = F0
    diag["forward_avg_no_k_hyd"] = forward_avg
    diag["c_H_avg"] = c_H_avg
    diag["pka_shift_avg"] = pka_shift_avg
    diag["R_forward_capped"] = (
        F0 * (1.0 - gamma / gamma_max) if gamma_max > 0.0 else None
    )

    # Langmuir denominator decomposition.  Mirrors the structure in
    # update_gamma_from_solution so observers can verify the formula
    # term-by-term.
    denom_constant = (1.0 - lam_val)
    denom_kdes = lam_val * k_des
    denom_kprot = lam_val * k_prot * c_H_avg / delta_ohp if delta_ohp > 0 else 0.0
    denom_cap = lam_val * F0 / gamma_max if gamma_max > 0.0 else None
    denom_total = (
        denom_constant + denom_kdes + denom_kprot
        + (denom_cap if denom_cap is not None else 0.0)
    )
    diag["denominator_constant"] = denom_constant
    diag["denominator_kdes"] = denom_kdes
    diag["denominator_kprot"] = denom_kprot
    diag["denominator_cap"] = denom_cap
    diag["denominator_total"] = denom_total
    diag["numerator"] = lam_val * F0
    # Phase 6β v10a' v10b-routing threshold support: ratio of cap term
    # to total denominator.  Critique session 33 R2 #6 / R3 pinned the
    # routing rule to >0.8 AND θ>0.9 AND |sensS|<0.10 — the ratio is
    # the load-bearing "cap dominates" indicator.
    if (
        denom_cap is not None and denom_total is not None
        and denom_total > 0.0
    ):
        diag["denominator_cap_to_total_ratio"] = denom_cap / denom_total
    else:
        diag["denominator_cap_to_total_ratio"] = None

    # ---- Phase 6β v10a' enhanced decompositions ---------------------------
    # F0 boundary-averaged decomposition (Jensen-safe per critique session
    # 33 R2 #5).  All averages are taken on factor PRODUCTS, never on
    # factors then multiplied.  Amplification ratios surface whether the
    # F0 growth in the cathodic region is dominated by K+ enrichment vs.
    # Singh ΔpKa (R3 #1: K+ enrichment is already load-bearing in v10a).
    try:
        c_K_avg = float(
            fd.assemble(c_M_expr * ds(electrode_marker))
        ) / area
        pka_factor_avg = float(
            fd.assemble(pka_factor * ds(electrode_marker))
        ) / area
        c_K_pka_product_avg = forward_avg          # = ⟨c_M · 10^(−ΔpKa)⟩
        F0_total = F0
        # Bulk c_K reference for the "K+-stayed-bulk" counterfactual.
        # Read from the form-build c0 (concentration BC at the bulk
        # ground node) when available; default to 1.0 nondim otherwise
        # (mPNP code typically nondimensionalises bulk concentrations
        # to 1).
        c_K_bulk = 1.0
        try:
            c0_vals = (ctx.get("nondim", {})
                          .get("c0_model_vals")
                       or ctx.get("nondim", {})
                             .get("c0_model"))
            if c0_vals is not None and bundle.counterion_idx < len(c0_vals):
                c_K_bulk = float(c0_vals[bundle.counterion_idx])
        except Exception:
            pass
        diag["F0_decomposition"] = {
            "c_K_avg":                   c_K_avg,
            "pka_factor_avg":            pka_factor_avg,
            "c_K_pka_product_avg":       c_K_pka_product_avg,
            "F0_total":                  F0_total,
            "c_K_bulk":                  c_K_bulk,
            # Counterfactual: K+ never depleted/enriched at the OHP.
            "F0_counterfactual_c_K_bulk": (
                k_hyd * c_K_bulk * pka_factor_avg
            ),
            # Counterfactual: Singh shift disabled (ΔpKa → 0, factor → 1).
            "F0_counterfactual_no_singh": k_hyd * c_K_avg,
            # Mechanism attribution.  >1 → c_K enrichment dominates;
            # <1 → c_K depletion suppresses.  Singh amplification
            # should be ≈1 in v10a baseline (pka_shift_avg ~ 1e-5).
            "amplification_from_c_K": (
                F0_total / (k_hyd * c_K_bulk * pka_factor_avg)
                if (k_hyd > 0.0 and c_K_bulk > 0.0 and pka_factor_avg > 0.0)
                else None
            ),
            "amplification_from_singh": (
                F0_total / (k_hyd * c_K_avg)
                if (k_hyd > 0.0 and c_K_avg > 0.0) else None
            ),
        }
    except Exception as exc:
        diag["F0_decomposition_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # R_4e log-space decomposition (solver-faithful: matches the
    # bv_log_rate path).  η_raw with Stern enabled is
    # (phi_applied − phi_boundary − E_eq), NOT (V_RHE − E_eq); the
    # exponent clip is applied to η/V_T BEFORE the α·n_e
    # multiplication (CLAUDE.md Hard Rule 2;
    # forms_logc.py:_build_eta_clipped).  This decomposition is a
    # SCALAR APPROXIMATION over boundary-averaged nonlinear terms;
    # ``log_R4e_measured`` (= ln of the assembled R_4e current) is
    # authoritative.
    try:
        nondim = ctx.get("nondim", {})
        rxns = nondim.get("bv_reactions", [])
        if len(rxns) >= 2:
            r4e = rxns[1]
            k0_R4e = float(r4e.get("k0_model", r4e.get("k0", 0.0)))
            alpha_R4e = float(r4e.get("alpha", 0.5))
            n_e_R4e = float(r4e.get("n_electrons", 4))
            E_eq_R4e = float(r4e.get("E_eq_model", 0.0))
            bv_exp_scale = float(ctx.get("_diag_bv_exp_scale", 1.0))
            exponent_clip = float(ctx.get("_diag_exponent_clip", 100.0))

            # Build η_raw expression — Stern-aware per Hard Rule 2.
            use_stern = bool(ctx.get("use_stern", False))
            phi_applied_func = ctx.get("phi_applied_func")
            indices = ctx.get("mixed_space_indices")
            U = ctx.get("U")
            if (
                phi_applied_func is not None
                and indices is not None
                and U is not None
            ):
                phi_var = fd.split(U)[indices.phi_index]
                if use_stern:
                    eta_raw = (
                        phi_applied_func - phi_var
                        - fd.Constant(E_eq_R4e)
                    )
                else:
                    eta_raw = phi_applied_func - fd.Constant(E_eq_R4e)
                eta_scaled = bv_exp_scale * eta_raw
                clip_const = fd.Constant(exponent_clip)
                eta_scaled_clipped = fd.min_value(
                    fd.max_value(eta_scaled, -clip_const), clip_const
                )

                eta_scaled_avg = float(
                    fd.assemble(eta_scaled * ds(electrode_marker))
                ) / area
                eta_scaled_clipped_avg = float(
                    fd.assemble(eta_scaled_clipped * ds(electrode_marker))
                ) / area
                # min/max via squared-difference detection: not
                # cheaply available without nodal extraction; report
                # whether the clip is active by comparing avg of
                # |η_scaled| > clip indicator.
                clip_indicator_avg = float(
                    fd.assemble(
                        fd.conditional(
                            abs(eta_scaled) > clip_const,
                            fd.Constant(1.0), fd.Constant(0.0),
                        ) * ds(electrode_marker)
                    )
                ) / area

                # n_e · ⟨ln(c_H/c_H_ref)⟩.
                # c_H_ref comes from the cathodic_conc_factors entry
                # for H+ (species 2 in the K2SO4 4sp stack).  The
                # ratio is in nondim units already (both are
                # nondimensionalised by C_SCALE).
                c_H_ref_nondim = 1.0
                ccfs = r4e.get("cathodic_conc_factors") or []
                for ccf in ccfs:
                    if ccf.get("species") == bundle.h_idx:
                        c_H_ref_nondim = float(
                            ccf.get("c_ref_nondim", 1.0)
                        )
                        break
                # ⟨ln(c_H)⟩ ≈ ⟨u_H⟩ for non-mu species; for mu_H the
                # reconstructed log is u_exprs[h_idx].  Use ci[h_idx]
                # via ln-of-average (Jensen approximation acknowledged).
                u_exprs = ctx.get("u_exprs")
                h_idx = bundle.h_idx
                if u_exprs is not None and h_idx < len(u_exprs):
                    ln_c_H_avg = float(
                        fd.assemble(u_exprs[h_idx] * ds(electrode_marker))
                    ) / area
                else:
                    # Fallback: ln of boundary-average c_H.  This is a
                    # stricter Jensen approximation than ⟨ln c_H⟩ but
                    # should be small for slowly-varying c_H.
                    ln_c_H_avg = (
                        math.log(c_H_avg) if c_H_avg > 0.0 else float("-inf")
                    )
                ln_c_H_ratio_avg = ln_c_H_avg - math.log(c_H_ref_nondim)

                log_k0 = math.log(k0_R4e) if k0_R4e > 0.0 else float("-inf")
                log_bv_clipped_avg = (
                    -alpha_R4e * n_e_R4e * eta_scaled_clipped_avg
                )
                n_e_log_c_H_factor_avg = n_e_R4e * ln_c_H_ratio_avg
                log_R4e_predicted = (
                    log_k0 + log_bv_clipped_avg + n_e_log_c_H_factor_avg
                )
                R4e_measured = diag.get("R_4e_current_nondim")
                if R4e_measured is not None and R4e_measured > 0.0:
                    log_R4e_measured = math.log(R4e_measured)
                elif R4e_measured is not None and R4e_measured < 0.0:
                    log_R4e_measured = math.log(abs(R4e_measured))
                else:
                    log_R4e_measured = None

                diag["R_4e_decomposition_log"] = {
                    "log_k0":                  log_k0,
                    "alpha_R4e":               alpha_R4e,
                    "n_electrons_R4e":         n_e_R4e,
                    "E_eq_R4e_nondim":         E_eq_R4e,
                    "bv_exp_scale":            bv_exp_scale,
                    "exponent_clip":           exponent_clip,
                    "eta_scaled_raw_avg":      eta_scaled_avg,
                    "eta_scaled_clipped_avg":  eta_scaled_clipped_avg,
                    "exponent_clip_active_fraction": clip_indicator_avg,
                    "log_bv_clipped_avg":      log_bv_clipped_avg,
                    "ln_c_H_avg":              ln_c_H_avg,
                    "ln_c_H_ratio_avg":        ln_c_H_ratio_avg,
                    "n_e_log_c_H_factor_avg":  n_e_log_c_H_factor_avg,
                    "c_H_ref_nondim":          c_H_ref_nondim,
                    "log_R4e_predicted":       log_R4e_predicted,
                    "log_R4e_measured":        log_R4e_measured,
                    "_note": (
                        "Scalar approximation: averages of nonlinear "
                        "terms over boundary; small/moderate "
                        "discrepancies vs measured can come from "
                        "Jensen/covariance, not necessarily a missing "
                        "Stern/diffuse term.  Order-of-magnitude "
                        "discrepancies indicate a missing residual term."
                    ),
                }
    except Exception as exc:
        diag["R_4e_decomposition_log_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return diag


__all__ = [
    "CationHydrolysisBundle",
    "GAMMA_MAX_HAT_SMOKE",            # deprecated alias (= V10A_SMOKE)
    "GAMMA_MAX_HAT_V10A_SMOKE",
    "GAMMA_MAX_HAT_V10B",
    "K_DES_NONDIM_V10B",
    "V10B_CALIBRATION_METADATA",
    "build_cation_hydrolysis_terms",
    "build_forward_branch",
    "build_forward_branch_uncapped",
    "build_pka_shift",
    "build_proton_boundary_source",
    "clamp_gamma_to_max",
    "collect_v10a_rung_diagnostics",
    "extract_gamma_value",
    "gamma_ss_langmuir",
    "is_cation_hydrolysis_enabled",
    "resolve_counterion_index",
    "update_gamma_from_solution",
]
