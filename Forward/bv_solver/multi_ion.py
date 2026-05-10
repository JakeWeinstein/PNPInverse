"""Multi-ion analytic-counterion machinery for the fast-realignment plan.

Single source of truth for the multi-ion shared-theta closure used by
both the Picard outer loop (electroneutrality at the outer-region edge)
and the spatial IC seed (forms_logc[_muh].py).  Mirrors the
single-counterion ``compute_surface_gamma`` / GC-Stern split helpers in
``picard_ic.py`` but generalizes to N steric counterions sharing one
denominator.

Closure (plan §2.1):

    For each analytic ion k (steric):
      c_k(φ) = c_b_k · exp(-z_k·φ) · (1 - A_dyn(φ))
                   / (θ_b + Σ_k' a_k' · c_b_k' · exp(-z_k'·φ))

    with A_dyn(φ) = Σ_dyn a_i · c_i_dyn(φ)
         θ_b      = 1 - A_dyn_bulk - Σ_k a_k · c_b_k

The denominator is the same for every steric ion (shared theta); no
coupled local NL solve needed.

Used by:
  - ``picard_outer_loop_general`` (post-loop reconstruction at the
    OHP edge): solves outer-region phi_o via bisection on the
    multi-ion electroneutrality residual; computes multispecies γ_s
    at OHP; computes local λ_eff for the linear-Debye Stern split.
  - ``forms_logc[_muh].py:_try_debye_boltzmann_ic*`` (spatial IC):
    interpolates each species onto the multi-ion ψ-vs-φ split with
    the shared-theta packing factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------

def _safe_exp(x: float, *, cap: float = 700.0) -> float:
    """``math.exp`` with a symmetric cap so multi-ion bisections can't
    overflow when ψ wanders far during root-finding.

    Mirrors ``picard_ic._safe_exp`` to keep the multi-ion module
    standalone (no import cycle).
    """
    if x > cap:
        return math.exp(cap)
    if x < -cap:
        return math.exp(-cap)
    return math.exp(x)


def _phi_safe_exp(
    phi: float, *, z: float, phi_clamp_val: float = 50.0,
) -> float:
    """Match ``boltzmann.py`` UFL ``phi_clamp_val`` semantics.

    ``boltzmann.py:223`` clamps ``phi`` to ``[-phi_clamp_val,
    +phi_clamp_val]`` BEFORE multiplying by ``-z`` and exponentiating.
    Mirror that here so the scalar helpers in this module agree with the
    UFL residual on what ``c_k(phi)`` is at extreme phi.

    At production parameters (|phi| ~ 5-10, phi_clamp_val=50) this is
    byte-equivalent to ``_safe_exp(-z*phi)`` with cap=700; it only
    differs once bisection wanders into clamped territory.
    """
    if phi > phi_clamp_val:
        phi = phi_clamp_val
    elif phi < -phi_clamp_val:
        phi = -phi_clamp_val
    return math.exp(-z * phi)


# ---------------------------------------------------------------------------
# CounterionConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CounterionConfig:
    """Per-counterion immutable record consumed by the multi-ion helpers.

    Mirrors the dict shape produced by
    ``DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`` etc., but typed and
    frozen.  The ctx builder accepts either dict entries or instances of
    this class — dicts are normalised into instances at the boundary.
    """
    z: int
    c_bulk_nondim: float
    a_nondim: float
    steric_mode: str = "ideal"      # "ideal" | "bikerman"
    phi_clamp: float = 50.0
    label: str = ""


def _normalize_entry(entry: Any) -> dict:
    """Coerce a CounterionConfig OR dict to a flat dict with all keys.

    Preserves the dict shape downstream callers (boltzmann.py) expect.
    """
    if isinstance(entry, CounterionConfig):
        return {
            "z": int(entry.z),
            "c_bulk_nondim": float(entry.c_bulk_nondim),
            "a_nondim": float(entry.a_nondim),
            "steric_mode": str(entry.steric_mode),
            "phi_clamp": float(entry.phi_clamp),
            "label": str(entry.label),
        }
    if isinstance(entry, dict):
        return {
            "z": int(entry["z"]),
            "c_bulk_nondim": float(entry["c_bulk_nondim"]),
            "a_nondim": float(entry.get("a_nondim", 0.0)),
            "steric_mode": str(entry.get("steric_mode", "ideal")),
            "phi_clamp": float(entry.get("phi_clamp", 50.0)),
            "label": str(entry.get("label", "")),
        }
    raise TypeError(
        f"_normalize_entry: expected CounterionConfig or dict, got {type(entry)!r}"
    )


# ---------------------------------------------------------------------------
# Counterion context builder (single producer of theta_b)
# ---------------------------------------------------------------------------

def build_counterion_ctx(
    counterions: list,
    a_dyn: list[float],
    c_dyn_bulk: list[float],
    z_dyn: list[int],
) -> dict:
    """Build the canonical multi-ion counterion context.

    Single producer of ``theta_b``; downstream readers MUST consume it
    from this ctx rather than recomputing from raw inputs (avoids
    sign / inclusion drift between Picard and the residual side).

    Parameters
    ----------
    counterions:
        Sequence of CounterionConfig OR raw config dicts (the latter
        as produced by ``DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC``
        etc.).  Mixed allowed.
    a_dyn, c_dyn_bulk, z_dyn:
        Per dynamic species, length n_dyn.

    Returns
    -------
    dict with keys:
        - ``ions``: list of normalised counterion dicts
        - ``z_dyn``, ``c_dyn_bulk``, ``a_dyn``: passthrough copies
        - ``theta_b``: bulk packing fraction (1 - A_dyn_bulk - Σ a_k c_b_k)
    """
    if not counterions:
        return {
            "ions": [], "z_dyn": list(z_dyn),
            "c_dyn_bulk": [float(c) for c in c_dyn_bulk],
            "a_dyn": [float(a) for a in a_dyn],
            "theta_b": 1.0 - sum(a * c for a, c in zip(a_dyn, c_dyn_bulk)),
        }
    ions = [_normalize_entry(e) for e in counterions]
    A_dyn_bulk = sum(a * c for a, c in zip(a_dyn, c_dyn_bulk))
    A_an_bulk = sum(
        ion["a_nondim"] * ion["c_bulk_nondim"]
        for ion in ions if ion["steric_mode"] == "bikerman"
    )
    theta_b = 1.0 - A_dyn_bulk - A_an_bulk
    if theta_b <= 0.0:
        bm_str = ", ".join(
            f"{ion.get('label','?')}: a={ion['a_nondim']:.4g}*c_b={ion['c_bulk_nondim']:.4g}"
            for ion in ions if ion["steric_mode"] == "bikerman"
        )
        raise ValueError(
            f"build_counterion_ctx: theta_b <= 0 (got {theta_b:.6g}) — "
            f"sum(a_k*c_b_k) for steric counterions = {A_an_bulk:.6g}; "
            f"A_dyn_bulk = {A_dyn_bulk:.6g}; bikerman entries: [{bm_str}]; "
            f"reduce dynamic-species packing or per-ion a_nondim/c_bulk_nondim."
        )
    return {
        "ions": ions,
        "z_dyn": [int(z) for z in z_dyn],
        "c_dyn_bulk": [float(c) for c in c_dyn_bulk],
        "a_dyn": [float(a) for a in a_dyn],
        "theta_b": float(theta_b),
    }


# ---------------------------------------------------------------------------
# Outer-region phi solve
# ---------------------------------------------------------------------------

def _ck_at_phi(*, ion: dict, phi: float, A_dyn_local: float,
               theta_b: float, ions: list, denom_cache: float | None = None) -> float:
    """Multi-steric closure single-ion concentration at potential ``phi``.

    Matches ``boltzmann.build_steric_boltzmann_expressions`` UFL algebra
    (ideal ions degenerate to ``c_b · exp(-z·phi)``; bikerman uses the
    shared-theta closure).  ``denom_cache`` lets the caller compute the
    shared denominator once and reuse it across all ions.
    """
    z = float(ion["z"])
    c_b = float(ion["c_bulk_nondim"])
    phi_clamp = float(ion.get("phi_clamp", 50.0))
    if ion.get("steric_mode", "ideal") != "bikerman":
        return c_b * _phi_safe_exp(phi, z=z, phi_clamp_val=phi_clamp)
    if denom_cache is None:
        denom_cache = theta_b + sum(
            ip["a_nondim"] * ip["c_bulk_nondim"] * _phi_safe_exp(
                phi, z=float(ip["z"]),
                phi_clamp_val=float(ip.get("phi_clamp", 50.0)),
            )
            for ip in ions if ip.get("steric_mode", "ideal") == "bikerman"
        )
    return (
        c_b * _phi_safe_exp(phi, z=z, phi_clamp_val=phi_clamp)
        * max(1.0 - A_dyn_local, 1e-12) / max(denom_cache, 1e-300)
    )


def _electroneutrality_residual(
    *, phi: float, ctx: dict, c_dyn: list[float],
) -> float:
    """ρ_total(phi) = Σ z_dyn·c_dyn + Σ z_an·c_an(phi).

    Both sums are evaluated at the *outer-region* edge (NOT bulk):
    the Picard yields ``c_dyn = [O_s, P_s, H_o]`` at the OHP-side edge
    and the analytic ions equilibrate against that local A_dyn rather
    than the bulk A_dyn_bulk (the Boltzmann profile is local).
    """
    z_dyn = ctx["z_dyn"]
    a_dyn = ctx["a_dyn"]
    ions = ctx["ions"]
    theta_b = ctx["theta_b"]
    A_dyn_local = sum(a * c for a, c in zip(a_dyn, c_dyn))
    denom = theta_b + sum(
        ion["a_nondim"] * ion["c_bulk_nondim"] * _phi_safe_exp(
            phi, z=float(ion["z"]),
            phi_clamp_val=float(ion.get("phi_clamp", 50.0)),
        )
        for ion in ions if ion.get("steric_mode", "ideal") == "bikerman"
    )
    rho_dyn = sum(z * c for z, c in zip(z_dyn, c_dyn))
    rho_an = sum(
        float(ion["z"]) * _ck_at_phi(
            ion=ion, phi=phi, A_dyn_local=A_dyn_local,
            theta_b=theta_b, ions=ions, denom_cache=denom,
        ) for ion in ions
    )
    return rho_dyn + rho_an


def solve_outer_phi_multiion(
    *,
    ctx: dict,
    c_dyn_outer: list[float],
    bracket: tuple[float, float] = (-50.0, +50.0),
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Solve outer-region electroneutrality for ``phi_o`` via bisection.

    The multi-ion analogue of the single-counterion line in
    ``picard_outer_loop_general``:
        ``phi_o = log(H_o / c_clo4_bulk)``  (1:1 ideal closed form)
    For a 2:1 sulfate or general multi-ion electrolyte that closed form
    no longer applies; bisection on ``ρ_total(phi) = 0`` gives the
    correct outer-region equilibrium potential.

    Parameters
    ----------
    ctx:
        Counterion context from ``build_counterion_ctx``.
    c_dyn_outer:
        Dynamic-species concentrations at the OHP-edge outer region
        (NOT bulk; the Picard's converged surface tuple).
    bracket:
        Initial bisection bracket for phi_o (nondim, in V_T units).
        ±50 covers the production phi_clamp.
    tol:
        Bisection convergence tolerance on ``|ρ_total|``.
    max_iter:
        Hard cap on bisection iterations.

    Returns
    -------
    phi_o : float
        Outer-region equilibrium potential (nondim).

    Raises
    ------
    ValueError
        If the bracket doesn't contain a root after expansion.
    """
    a_low, a_high = float(bracket[0]), float(bracket[1])
    f_low = _electroneutrality_residual(phi=a_low, ctx=ctx, c_dyn=c_dyn_outer)
    f_high = _electroneutrality_residual(phi=a_high, ctx=ctx, c_dyn=c_dyn_outer)
    # Bracket-expansion: try doubling out to ±200 if the initial bracket
    # has the same sign at both ends.
    expand_iters = 0
    while f_low * f_high > 0.0 and expand_iters < 8:
        a_low *= 2.0
        a_high *= 2.0
        f_low = _electroneutrality_residual(phi=a_low, ctx=ctx, c_dyn=c_dyn_outer)
        f_high = _electroneutrality_residual(phi=a_high, ctx=ctx, c_dyn=c_dyn_outer)
        expand_iters += 1
    if f_low * f_high > 0.0:
        raise ValueError(
            f"solve_outer_phi_multiion: bracket [{a_low}, {a_high}] does not "
            f"contain a root (residual signs match: f_low={f_low:.3g}, "
            f"f_high={f_high:.3g}); ctx ions={ctx['ions']!r}, "
            f"c_dyn_outer={c_dyn_outer}"
        )
    for _ in range(max_iter):
        mid = 0.5 * (a_low + a_high)
        f_mid = _electroneutrality_residual(phi=mid, ctx=ctx, c_dyn=c_dyn_outer)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid <= 0.0:
            a_high = mid
            f_high = f_mid
        else:
            a_low = mid
            f_low = f_mid
    return 0.5 * (a_low + a_high)


# ---------------------------------------------------------------------------
# Multispecies surface gamma at OHP
# ---------------------------------------------------------------------------

def compute_surface_gamma_multiion(
    *,
    H_o: float,
    psi_D: float,
    a_h: float,
    ions: list[dict],
) -> float:
    """Multispecies γ_s at the OHP using outer-region anchors.

    Generalizes ``picard_ic.compute_surface_gamma`` (which is hard-coded
    to one cation H⁺ + one anion ClO₄⁻ at bulk concentration) to N steric
    ions with their *outer-region* (NOT bulk) concentrations:

        γ_s = 1 / [
                 1 + a_H · H_o · (e^{-ψ_D} - 1)
                 + Σ_k a_k · c_k_outer · (e^{-z_k·ψ_D} - 1)
            ]

    where ``ions[k]`` MUST carry a ``"c_outer"`` key with the
    OHP-side outer-region concentration (the multi-ion shared-theta
    closure evaluated at the Picard's ``phi_o``).  For counterions in
    "ideal" mode ``c_outer = c_bulk · exp(-z·phi_o)`` per the standard
    Boltzmann.

    With ``a_H = 0`` and no steric ions the function returns 1.0.
    """
    if a_h == 0.0 and not ions:
        return 1.0
    denom = 1.0 + a_h * H_o * (_safe_exp(-psi_D) - 1.0)
    for ion in ions:
        denom += float(ion["a_nondim"]) * float(ion["c_outer"]) * (
            _safe_exp(-float(ion["z"]) * psi_D) - 1.0
        )
    if not math.isfinite(denom) or denom <= 0.0:
        return 1e-300
    return 1.0 / denom


# ---------------------------------------------------------------------------
# Local effective Debye length
# ---------------------------------------------------------------------------

def effective_debye_length_local(
    *,
    phi_o: float,
    ctx: dict,
    c_dyn_outer: list[float],
    poisson_coeff: float,
    dphi: float = 1e-4,
) -> float:
    """Local ``λ_eff = sqrt(eps / |dρ/dφ|_outer)`` via central FD.

    Differs from the bulk ``Σ z² c`` Debye length by up to ~20× in 2:1
    sulfate when ``|φ_o| ≳ 2`` because the steric closure depresses the
    counterion contribution at strong ψ.  At I = 0.3 M with Stern
    dominating, the correct screening coefficient is critical for the
    linear-Debye Stern split.

    Parameters
    ----------
    phi_o:
        Outer-region equilibrium potential at which to evaluate
        ``dρ/dφ`` (typically the value returned by
        ``solve_outer_phi_multiion``).
    ctx:
        Counterion context from ``build_counterion_ctx``.
    c_dyn_outer:
        Dynamic-species concentrations at the OHP-edge outer region.
    poisson_coeff:
        Nondim Poisson coefficient (``ε * V_T / (F · c_scale · L²)``).
    dphi:
        Central FD step for evaluating ``dρ/dφ``.

    Returns
    -------
    lambda_eff : float
        Local effective Debye length (nondim).  Floored at 1e-15 to
        avoid divide-by-zero when the local screening is degenerate.
    """
    rho_p = _electroneutrality_residual(phi=phi_o + dphi, ctx=ctx, c_dyn=c_dyn_outer)
    rho_m = _electroneutrality_residual(phi=phi_o - dphi, ctx=ctx, c_dyn=c_dyn_outer)
    drho_dphi = (rho_p - rho_m) / (2.0 * dphi)
    # ε ∇²φ = -ρ ⇒ for small perturbation φ' ≈ ψ exp(-y/λ) with
    # ε·(1/λ²) = -dρ/dφ ⇒ λ = sqrt(-ε / dρ/dφ).  dρ/dφ is negative
    # in the screening regime (positive perturbation φ' attracts
    # negative charge).  Take |dρ/dφ| to be safe.
    inv_lambda_sq = max(-drho_dphi, 1e-30) / max(poisson_coeff, 1e-300)
    return math.sqrt(1.0 / max(inv_lambda_sq, 1e-30))


# ---------------------------------------------------------------------------
# Convenience: per-ion outer-region concentrations
# ---------------------------------------------------------------------------

def per_ion_outer_concs(
    *, ctx: dict, c_dyn_outer: list[float], phi_o: float,
) -> list[float]:
    """Evaluate every counterion's OHP-edge concentration at ``phi_o``.

    The result mirrors the order of ``ctx['ions']`` and is the value
    each ion contributes to the outer-region charge balance and the
    multispecies γ_s denominator.
    """
    A_dyn_local = sum(a * c for a, c in zip(ctx["a_dyn"], c_dyn_outer))
    denom = ctx["theta_b"] + sum(
        ion["a_nondim"] * ion["c_bulk_nondim"] * _phi_safe_exp(
            phi_o, z=float(ion["z"]),
            phi_clamp_val=float(ion.get("phi_clamp", 50.0)),
        )
        for ion in ctx["ions"] if ion.get("steric_mode", "ideal") == "bikerman"
    )
    return [
        _ck_at_phi(
            ion=ion, phi=phi_o, A_dyn_local=A_dyn_local,
            theta_b=ctx["theta_b"], ions=ctx["ions"], denom_cache=denom,
        )
        for ion in ctx["ions"]
    ]
