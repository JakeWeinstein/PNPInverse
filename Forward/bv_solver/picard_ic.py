"""Shared scalar Picard outer-loop and helpers for the debye_boltzmann IC.

Used by both ``Forward.bv_solver.forms_logc._try_debye_boltzmann_ic`` and
``forms_logc_muh._try_debye_boltzmann_ic_muh``. Centralizes:

  - The Butler-Volmer Picard outer loop (2x2 algebraic surface-rate solve)
  - Bikerman-aware multispecies activity coefficient ``gamma_s``
  - Stern-aware potential split ``psi_S + psi_D = phi_applied - phi_o``
  - Numerically-safe scalar primitives (``_safe_exp``, ``_eta_clipped``,
    ``_factor_log_from_species_logs``)

The Picard loop is gamma-aware: when ``a_h = a_cl = 0`` (ideal-counterion
config) ``gamma_s = 1`` and the loop reduces to the legacy gamma-free
path. When ``stern_split=None`` the loop uses the legacy no-Stern
``eta = bv_exp_scale * (phi_applied - E)``; with a Stern config it
solves for ``(psi_S, psi_D)`` per iter and uses
``eta = bv_exp_scale * (phi_applied - phi_surface - E)``.

See also:
  - ``docs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md`` (γ Picard mismatch + Stern-η bugs)
  - ``docs/CODEX_REVIEW_HANDOFF_12_IC_PICARD_BUGS.md`` (review of the diagnosis)
  - ``docs/CHATGPT_HANDOFF_13_RESPONSE_TO_CODEX_REVIEW.md`` (resolution plan)
  - ``docs/CODEX_RESPONSE_TO_HANDOFF_13.md`` (verification protocol)
  - ``docs/PNP_BV_Analytical_Simplifications.md`` (matched-asymptotic IC math)
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Numerically-safe scalar primitives
# ---------------------------------------------------------------------------

def _safe_exp(x: float) -> float:
    """Clamped ``math.exp`` -- protects against +/-700 overflow."""
    if not math.isfinite(x):
        return float("inf") if x > 0 else 0.0
    if x > 700.0:
        return math.exp(700.0)
    if x < -700.0:
        return 0.0
    return math.exp(x)


def _eta_clipped(
    *,
    eta_drop: float,
    E: float,
    bv_exp_scale: float,
    exponent_clip: float,
    clip: bool,
) -> float:
    """Compute clipped overpotential ``bv_exp_scale * (eta_drop - E)``.

    ``eta_drop`` is the potential drop the Butler-Volmer reaction "sees":

      - No-Stern (legacy): ``eta_drop = phi_applied`` (residual sets
        ``eta_raw = phi_applied - E``).
      - Stern: ``eta_drop = psi_S = phi_applied - phi_surface`` (residual
        sets ``eta_raw = phi_applied - phi - E`` with ``phi`` at the OHP).

    The clip is applied to the scaled raw eta, NOT to ``alpha * n_e * eta``;
    see ``docs/clipping_conventions.md``.
    """
    eta = bv_exp_scale * (eta_drop - E)
    if clip:
        return max(min(eta, exponent_clip), -exponent_clip)
    return eta


def _factor_log_from_species_logs(
    log_by_species: list[float], factors: list
) -> float:
    """Sum ``power * (log(c_sp_rxn) - log(c_ref))`` over cathodic_conc_factors.

    Replaces the legacy ``_h_factor_log(H_val, factors)`` which only
    handled species-2 (H+) factors and recomputed ``log(H_s)`` inline.
    By taking pre-computed reaction-plane logs as input, this helper:

      (a) supports any reactant species in cathodic_conc_factors (e.g. an
          O2 stoichiometric factor in a future config); and
      (b) eliminates ``H_s = H_o * exp(-psi_D)`` as a hidden gamma-free
          shortcut. Callers are responsible for building ``log_by_species``
          with or without ``log(gamma_s)`` folded in.
    """
    total = 0.0
    for f in factors:
        sp_idx = int(f["species"])
        power = float(f["power"])
        c_ref_log = math.log(max(float(f["c_ref_nondim"]), 1e-30))
        total += power * (log_by_species[sp_idx] - c_ref_log)
    return total


# ---------------------------------------------------------------------------
# Surface activity coefficient (Bikerman multispecies gamma)
# ---------------------------------------------------------------------------

def compute_surface_gamma(
    H_o: float,
    c_clo4_bulk: float,
    psi_D: float,
    a_h: float,
    a_cl: float,
    c_cl_anchor: float,
) -> float:
    """Multispecies Bikerman activity coefficient at the reaction plane.

    Closed form (matches ``gamma_psi`` in
    ``forms_logc.py:_try_debye_boltzmann_ic`` and the muh counterpart):

        gamma_s = 1 / (1 + a_h * H_o     * (e^(-psi_D) - 1)
                       + a_cl * c_anchor * (e^(+psi_D) - 1))

    With ``a_h = a_cl = 0`` (ideal counterion) ``gamma_s = 1``.

    ``c_cl_anchor`` is ``H_o`` for synthesised-4sp ClO4- (electroneutrality
    pins the outer-region counterion to the proton concentration), or
    ``c_clo4_bulk`` for an explicit analytic 3sp+bikerman counterion (outer
    region = bulk).
    """
    if a_h == 0.0 and a_cl == 0.0:
        return 1.0
    denom = (
        1.0
        + a_h * H_o * (_safe_exp(-psi_D) - 1.0)
        + a_cl * c_cl_anchor * (_safe_exp(+psi_D) - 1.0)
    )
    if not math.isfinite(denom) or denom <= 0.0:
        # Saturation regime: gamma_s -> 0 in the cap. Floor at 1e-300 so
        # the next log(gamma_s) call does not underflow.
        return 1e-300
    return 1.0 / denom


# ---------------------------------------------------------------------------
# Stern split solver
# ---------------------------------------------------------------------------

def compute_surface_slope_signed(
    psi_D: float, lambda_D: float, nu_charged: float
) -> float:
    """Signed surface slope ``|dphi/dy(0)| * sign(psi_D)`` at the OHP.

    Two physical regimes:

    * ``nu_charged <= 0`` (ideal/GC): use the Gouy-Chapman first integral
      ``|dphi/dy(0)| = (2/lambda_D) * |sinh(psi_D / 2)|``.
    * ``nu_charged > 0`` (Bikerman): use the BKSA matched-asymptotic
      saturated-zone slope

          alpha_d = sqrt((2/(nu*lam_D^2)) * ln[1 + nu*(cosh|psi_D| - 1)])

      which agrees with the GC analogue in the unsaturated regime
      ``nu*(cosh-1) << 1`` and caps off in the saturated regime.

    The sign matches ``sign(psi_D)`` because the Stern Robin closure is
    sign-consistent (Codex §"Stern Split Notes" in handoff #13 response).
    """
    psi_d_abs = abs(psi_D)
    if psi_d_abs < 1e-12:
        return 0.0
    sign_psi_D = 1.0 if psi_D >= 0.0 else -1.0
    if nu_charged <= 0.0:
        # Gouy-Chapman first integral
        slope_mag = (2.0 / max(lambda_D, 1e-30)) * math.sinh(psi_d_abs / 2.0)
        return sign_psi_D * slope_mag
    # BKSA matched-asymptotic (cosh, log are evaluated on |psi_D|)
    arg_cosh = math.cosh(psi_d_abs)
    inner = 1.0 + nu_charged * (arg_cosh - 1.0)
    if inner <= 0.0:
        return sign_psi_D * float("inf")
    alpha_d = math.sqrt(
        (2.0 / (nu_charged * lambda_D ** 2)) * math.log(inner)
    )
    return sign_psi_D * alpha_d


def solve_stern_split(
    *,
    phi_applied_model: float,
    phi_o: float,
    lambda_D: float,
    c_clo4_bulk: float,
    a_cl: float,
    stern_coeff_nondim: float,
    eps_nondim: float,
) -> tuple[float, float, float]:
    """Bisect on ``psi_D`` to satisfy the Stern Robin closure.

    Splits the total drop ``phi_applied - phi_o`` into Stern (psi_S) +
    diffuse (psi_D) drops under the closure

        stern_coeff * psi_S = eps_nondim * surface_slope_signed(psi_D)
        psi_S + psi_D       = phi_applied - phi_o

    where ``surface_slope_signed`` is the BKSA slope for ``a_cl > 0`` /
    Gouy-Chapman analogue otherwise. See ``compute_surface_slope_signed``.

    Parameters
    ----------
    phi_applied_model : float
        Applied potential at the metal (model units; nondim).
    phi_o : float
        Outer-region potential (= ``ln(H_o / c_clo4_bulk)`` in the analytic
        IC, after the Picard outer loop converges H_o).
    lambda_D : float
        Nondim Debye length = ``sqrt(poisson_coefficient)``.
    c_clo4_bulk : float
        Nondim bulk counterion concentration.
    a_cl : float
        Nondim Bikerman size of the counterion. ``a_cl = 0`` -> ideal limit.
    stern_coeff_nondim : float
        Nondim Stern capacitance (= ``bv_stern_capacitance_model`` in the
        residual Robin BC). Pass ``0.0`` for no-Stern -> returns
        ``psi_S = 0``, ``phi_surface = phi_applied``.
    eps_nondim : float
        Poisson coefficient (= lambda_D ** 2).

    Returns
    -------
    (psi_S, psi_D, phi_surface) : tuple of float
        ``phi_surface = phi_applied - psi_S`` is the OHP-side potential the
        residual sees; ``psi_D`` is the diffuse-layer drop.

    Notes
    -----
    Robin identity at the returned ``(psi_S, psi_D)`` (within bisection
    tolerance):

        | stern_coeff * psi_S - eps * surface_slope_signed(psi_D) | < tol

    Falls through to a linear-Debye analytical solution when bisection
    cannot bracket the root (typically only at ``psi_D ~ 0``).
    """
    full_drop = phi_applied_model - phi_o
    if abs(stern_coeff_nondim) < 1e-30:
        return 0.0, full_drop, phi_applied_model
    if abs(full_drop) < 1e-12:
        return 0.0, 0.0, phi_applied_model

    nu_charged = 2.0 * a_cl * c_clo4_bulk

    def residual(psi_D_signed: float) -> float:
        psi_S = full_drop - psi_D_signed
        slope_signed = compute_surface_slope_signed(
            psi_D_signed, lambda_D, nu_charged
        )
        return stern_coeff_nondim * psi_S - eps_nondim * slope_signed

    # Bracket: psi_D in [0, full_drop] (sign-aware).
    # At psi_D=0: residual = stern * full_drop (sign of full_drop)
    # At psi_D=full_drop: residual = -eps * slope_at_full_drop (opposite sign)
    lo, hi = 0.0, full_drop
    f_lo = residual(lo)
    f_hi = residual(hi)
    if f_lo * f_hi > 0.0:
        # Linear-Debye limit fallback. In the small-|psi_D| regime,
        # surface_slope_signed ~ psi_D / lambda_D, giving
        #   stern_coeff * (full_drop - psi_D) = eps * psi_D / lambda_D
        # -> psi_D = stern_coeff * full_drop * lambda_D /
        #            (eps + stern_coeff * lambda_D)
        denom = eps_nondim + stern_coeff_nondim * lambda_D
        if abs(denom) < 1e-30:
            return 0.0, full_drop, phi_applied_model
        psi_D_lin = stern_coeff_nondim * full_drop * lambda_D / denom
        psi_S_lin = full_drop - psi_D_lin
        return psi_S_lin, psi_D_lin, phi_applied_model - psi_S_lin

    # ~80 iterations gives ~1e-24 tolerance on psi_D for full_drop ~ O(1).
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        # Tolerance in terms of the residual scale:
        scale = max(abs(stern_coeff_nondim * full_drop), abs(f_lo), abs(f_hi), 1.0)
        if abs(f_mid) < 1e-14 * scale:
            lo = hi = mid
            break
        if f_lo * f_mid < 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    psi_D = 0.5 * (lo + hi)
    psi_S = full_drop - psi_D
    return psi_S, psi_D, phi_applied_model - psi_S


# ---------------------------------------------------------------------------
# Picard outer loop
# ---------------------------------------------------------------------------

def picard_outer_loop(
    *,
    H_b: float,
    O_b: float,
    P_b: float,
    D_O: float,
    D_P: float,
    D_H: float,
    P_FLOOR: float,
    c_clo4_bulk: float,
    k1: float,
    k2: float,
    a1: float,
    a2: float,
    n_e: float,
    E1: float,
    E2: float,
    h_factor1: list,
    h_factor2: list,
    phi_applied_model: float,
    bv_exp_scale: float,
    exponent_clip: float,
    clip_exponent: bool,
    a_h: float = 0.0,
    a_cl: float = 0.0,
    c_cl_anchor_kind: str = "bulk",
    stern_split: dict | None = None,
    omega: float = 0.5,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> tuple[bool, str, int, dict[str, float]]:
    """Scalar Picard outer loop for the matched-asymptotic IC.

    Solves for the outer-region steady state:

      * Surface concentrations ``O_s, P_s`` and outer-region ``H_o`` from the
        diffusion-flux <-> BV-rate balance.
      * Diffuse-layer drop ``psi_D = phi_applied - phi_o`` with
        ``phi_o = ln(H_o / c_clo4_bulk)``.
      * (Optional) Stern split ``(psi_S, psi_D)`` per iter when
        ``stern_split`` is provided.

    With ``a_h = a_cl = 0`` and ``stern_split = None`` this reduces to the
    legacy gamma-free, no-Stern Picard loop (byte-equivalent to the
    in-tree ``_try_debye_boltzmann_ic`` body up to commit ``77ceff3``).

    Returns
    -------
    (ok, reason, n_iters, picard_state) : tuple
        - ``ok``: True on convergence
        - ``reason``: failure reason string, ``''`` on success
        - ``n_iters``: number of Picard iterations executed
        - ``picard_state``: dict with the converged scalar values
          (``R1, R2, O_s, P_s, H_o, phi_o, psi_D, psi_S, phi_surface,
          gamma_s, eta1, eta2``).

    Parameters
    ----------
    H_b, O_b, P_b : float
        Bulk concentrations (nondim).
    D_O, D_P, D_H : float
        Nondim diffusivities (already floored above 1e-30).
    P_FLOOR : float
        Lower clamp for surface H2O2 (callers pass ``max(P_b, 1e-30)``).
    c_clo4_bulk : float
        Bulk counterion concentration (nondim).
    k1, k2, a1, a2, n_e : float
        Reaction kinetics (nondim).
    E1, E2 : float
        Equilibrium potentials (nondim, from ``rxn["E_eq_model"]``).
    h_factor1, h_factor2 : list of dict
        ``cathodic_conc_factors`` lists from the bv_reactions config.
        Each entry has ``species``, ``power``, ``c_ref_nondim``.
    phi_applied_model : float
        Applied potential (nondim).
    bv_exp_scale : float
        Eta scaling factor (= 1.0 in production).
    exponent_clip, clip_exponent : float, bool
        Eta clip parameters.
    a_h, a_cl : float, optional
        Bikerman size of H+ and ClO4- (nondim). Default 0 (ideal limit).
    c_cl_anchor_kind : str, optional
        ``"bulk"`` (default): counterion anchor = ``c_clo4_bulk``
            (analytic 3sp+bikerman counterion; outer region = bulk).
        ``"synthesised_4sp"``: counterion anchor = ``H_o`` (4sp dynamic
            ClO4- with electroneutrality outer condition).
    stern_split : dict | None, optional
        ``None`` -> no-Stern (psi_S=0, phi_surface=phi_applied).
        dict with keys ``lambda_D``, ``stern_coeff``, ``eps`` ->
            per-iter Stern split using ``solve_stern_split``.
    omega : float, optional
        Picard relaxation factor.
    max_iters, tol : int, float, optional
        Convergence controls.
    """
    # --- Picard state ---
    R1 = 0.0
    R2 = 0.0
    H_o = H_b
    phi_o = 0.0
    O_s = O_b
    P_s = P_b
    gamma_s = 1.0

    # Initialize (psi_S, psi_D, phi_surface, eta_drop) for iter 1.  These
    # are then updated at the END of each iter (matching the legacy
    # convention where psi_D was updated post-state-update).  For
    # no-Stern, the eta drop is the full applied potential; for Stern,
    # it is the Stern-layer drop ``psi_S``:
    #   no-Stern: eta_raw = phi_applied - E          (residual)
    #   Stern:    eta_raw = phi_applied - phi(OHP) - E (residual)
    #             eta_drop = phi_applied - phi(OHP)  = psi_S.
    if stern_split is not None:
        psi_S, psi_D, phi_surface = solve_stern_split(
            phi_applied_model=phi_applied_model,
            phi_o=phi_o,
            lambda_D=stern_split["lambda_D"],
            c_clo4_bulk=c_clo4_bulk,
            a_cl=a_cl,
            stern_coeff_nondim=stern_split["stern_coeff"],
            eps_nondim=stern_split["eps"],
        )
        eta_drop = psi_S
    else:
        psi_S = 0.0
        psi_D = phi_applied_model - phi_o
        phi_surface = phi_applied_model
        eta_drop = phi_applied_model

    eta1 = _eta_clipped(
        eta_drop=eta_drop,
        E=E1,
        bv_exp_scale=bv_exp_scale,
        exponent_clip=exponent_clip,
        clip=clip_exponent,
    )
    eta2 = _eta_clipped(
        eta_drop=eta_drop,
        E=E2,
        bv_exp_scale=bv_exp_scale,
        exponent_clip=exponent_clip,
        clip=clip_exponent,
    )

    delta = float("inf")
    converged = False
    picard_iters = 0
    for k in range(1, max_iters + 1):
        R1_old, R2_old = R1, R2

        # Counterion anchor for the gamma denominator.
        c_cl_anchor = H_o if c_cl_anchor_kind == "synthesised_4sp" else c_clo4_bulk

        # Surface activity (Bikerman multispecies gamma).
        gamma_s = compute_surface_gamma(
            H_o, c_clo4_bulk, psi_D, a_h, a_cl, c_cl_anchor
        )
        log_gamma = math.log(max(gamma_s, 1e-300))

        # Reaction-plane log-concentrations. With a_h=a_cl=0 these reduce
        # to the legacy gamma-free path:
        #   log_O_rxn = log(O_s);  log_P_rxn = log(P_s);
        #   log_H_rxn = log(H_o) - psi_D = log(H_s).
        log_O_rxn = math.log(max(O_s, 1e-300)) + log_gamma
        log_P_rxn = math.log(max(P_s, 1e-300)) + log_gamma
        log_H_rxn = math.log(max(H_o, 1e-300)) - psi_D + log_gamma
        log_by_species = [log_O_rxn, log_P_rxn, log_H_rxn]

        log_h_factor1 = _factor_log_from_species_logs(log_by_species, h_factor1)
        log_h_factor2 = _factor_log_from_species_logs(log_by_species, h_factor2)

        # 2x2 coefficients. ``A1, B1, A2`` are coefficients of ``O_s, P_s``
        # in the rate expressions (unchanged structure). The gamma factors
        # enter via:
        #   A1 picks up ONE log_gamma for the O2 catalyst -> rate / O_s.
        #     (the catalyst's reaction-plane c is gamma-shifted, so the
        #      coefficient of O_s is gamma * factor.)
        #   B1 picks up ONE log_gamma for the H2O2 anodic species.
        #   A2 picks up ONE log_gamma for the H2O2 catalyst.
        # Plus the log_h_factor* picks up gamma per H+ stoich power
        # through log_H_rxn.
        # Net gamma powers (with H+ power=2 in production):
        #   A1 ~ gamma^(1+2) = gamma^3
        #   B1 ~ gamma^1
        #   A2 ~ gamma^(1+2) = gamma^3
        # See docs/CODEX_RESPONSE_TO_HANDOFF_13.md §3 ("R1 cathodic
        # carries gamma^(1+2) = gamma^3").
        log_A1 = math.log(k1) + log_gamma + log_h_factor1 - a1 * n_e * eta1
        log_B1 = math.log(k1) + log_gamma + (1.0 - a1) * n_e * eta1
        log_A2 = math.log(k2) + log_gamma + log_h_factor2 - a2 * n_e * eta2

        A1 = _safe_exp(log_A1)
        B1 = _safe_exp(log_B1)
        A2 = _safe_exp(log_A2)

        # 2x2 system (structurally unchanged from gamma-free):
        m11 = 1.0 + A1 / D_O + B1 / D_P
        m12 = -B1 / D_P
        m21 = -A2 / D_P
        m22 = 1.0 + A2 / D_P
        rhs1 = A1 * O_b - B1 * P_b
        rhs2 = A2 * P_b
        det = m11 * m22 - m12 * m21
        if not math.isfinite(det) or abs(det) < 1e-300:
            return False, f"singular_jacobian_iter_{k}_det={det:.3g}", k, _state_dict_failure(
                R1, R2, O_s, P_s, H_o, phi_o, psi_D, psi_S, phi_surface, gamma_s, eta1, eta2,
            )
        R1_new = (m22 * rhs1 - m12 * rhs2) / det
        R2_new = (-m21 * rhs1 + m11 * rhs2) / det
        if not (math.isfinite(R1_new) and math.isfinite(R2_new)):
            return False, f"non_finite_R_iter_{k}", k, _state_dict_failure(
                R1, R2, O_s, P_s, H_o, phi_o, psi_D, psi_S, phi_surface, gamma_s, eta1, eta2,
            )

        R1 = (1.0 - omega) * R1 + omega * R1_new
        R2 = (1.0 - omega) * R2 + omega * R2_new

        O_s = max(O_b - R1 / D_O, 1e-300)
        P_s = max(P_b + (R1 - R2) / D_P, P_FLOOR)
        # Ambipolar 2*D_H factor and -2 proton stoichiometry cancel
        # (PNP_BV_Analytical_Simplifications.md lines 240-244). The
        # denominator is bare D_H, NOT 2*D_H; do not "correct".
        H_o = max(H_b - (R1 + R2) / D_H, 1e-300)
        phi_o = math.log(H_o / c_clo4_bulk)

        # Update (psi_D, psi_S, phi_surface, eta_drop, eta1, eta2) at the
        # END of the iter so the post-loop state matches the legacy
        # convention (legacy gamma-free Picard updated ``psi_D`` here).
        # eta1, eta2 are constants for no-Stern (eta_drop = phi_applied)
        # but evolve per iter for Stern (eta_drop = psi_S, which depends
        # on phi_o through solve_stern_split).
        if stern_split is not None:
            psi_S, psi_D, phi_surface = solve_stern_split(
                phi_applied_model=phi_applied_model,
                phi_o=phi_o,
                lambda_D=stern_split["lambda_D"],
                c_clo4_bulk=c_clo4_bulk,
                a_cl=a_cl,
                stern_coeff_nondim=stern_split["stern_coeff"],
                eps_nondim=stern_split["eps"],
            )
            eta_drop = psi_S
        else:
            psi_S = 0.0
            psi_D = phi_applied_model - phi_o
            phi_surface = phi_applied_model
            eta_drop = phi_applied_model

        eta1 = _eta_clipped(
            eta_drop=eta_drop,
            E=E1,
            bv_exp_scale=bv_exp_scale,
            exponent_clip=exponent_clip,
            clip=clip_exponent,
        )
        eta2 = _eta_clipped(
            eta_drop=eta_drop,
            E=E2,
            bv_exp_scale=bv_exp_scale,
            exponent_clip=exponent_clip,
            clip=clip_exponent,
        )

        denom1 = max(abs(R1), 1e-30)
        denom2 = max(abs(R2), 1e-30)
        delta = abs(R1 - R1_old) / denom1 + abs(R2 - R2_old) / denom2
        picard_iters = k
        if delta < tol:
            converged = True
            break

    if not converged:
        return (
            False,
            f"picard_max_iters_delta={delta:.4g}",
            picard_iters,
            _state_dict_failure(
                R1, R2, O_s, P_s, H_o, phi_o, psi_D, psi_S, phi_surface, gamma_s, eta1, eta2,
            ),
        )

    # Post-converged gamma_s aligned with the post-iter (psi_D, H_o)
    # tuple.  For Codex's rate-consistency diagnostic the returned eta1,
    # eta2 already match the post-iter eta_drop (updated above).
    c_cl_anchor_post = (
        H_o if c_cl_anchor_kind == "synthesised_4sp" else c_clo4_bulk
    )
    gamma_s = compute_surface_gamma(
        H_o, c_clo4_bulk, psi_D, a_h, a_cl, c_cl_anchor_post
    )

    # Numerically-stable post-convergence P_s, O_s.  The 2x2 fixed
    # point gives the closed forms
    #   P_s = (D_P * P_b + R1) / (D_P + A2)
    #   O_s = (D_O * O_b + B1 * P_s) / (D_O + A1)
    # which are equivalent to the loop's ``P_s = P_b + (R1-R2)/D_P``
    # and ``O_s = O_b - R1/D_O`` at the fixed point but without the
    # near-cancellation that bites in the diffusion-limited regime
    # (where R1-R2 ~ -D_P*P_b causes the loop formula to lose 4-5
    # decimal digits to floating-point noise, yielding ``A2 * P_s_loop
    # >> R2`` after IC interpolation -- the rate-consistency mismatch
    # at high V_RHE with Stern + bikerman).
    log_gamma_post = math.log(max(gamma_s, 1e-300))
    log_O_rxn_post = math.log(max(O_s, 1e-300)) + log_gamma_post
    log_P_rxn_post = math.log(max(P_s, 1e-300)) + log_gamma_post
    log_H_rxn_post = math.log(max(H_o, 1e-300)) - psi_D + log_gamma_post
    log_by_species_post = [log_O_rxn_post, log_P_rxn_post, log_H_rxn_post]

    log_h_factor1_post = _factor_log_from_species_logs(log_by_species_post, h_factor1)
    log_h_factor2_post = _factor_log_from_species_logs(log_by_species_post, h_factor2)
    log_A1_post = math.log(k1) + log_gamma_post + log_h_factor1_post - a1 * n_e * eta1
    log_B1_post = math.log(k1) + log_gamma_post + (1.0 - a1) * n_e * eta1
    log_A2_post = math.log(k2) + log_gamma_post + log_h_factor2_post - a2 * n_e * eta2
    A1_post = _safe_exp(log_A1_post)
    B1_post = _safe_exp(log_B1_post)
    A2_post = _safe_exp(log_A2_post)

    P_s = max((D_P * P_b + R1) / (D_P + A2_post), P_FLOOR)
    O_s = max((D_O * O_b + B1_post * P_s) / (D_O + A1_post), 1e-300)

    state = {
        "R1": R1, "R2": R2,
        "O_s": O_s, "P_s": P_s, "H_o": H_o,
        "phi_o": phi_o,
        "psi_D": psi_D, "psi_S": psi_S,
        "phi_surface": phi_surface,
        "gamma_s": gamma_s,
        "eta1": eta1, "eta2": eta2,
    }
    return True, "", picard_iters, state


def _state_dict_failure(
    R1: float, R2: float, O_s: float, P_s: float, H_o: float,
    phi_o: float, psi_D: float, psi_S: float, phi_surface: float,
    gamma_s: float, eta1: float, eta2: float,
) -> dict[str, float]:
    """Return a partial state dict on Picard failure for diagnostics."""
    return {
        "R1": R1, "R2": R2,
        "O_s": O_s, "P_s": P_s, "H_o": H_o,
        "phi_o": phi_o,
        "psi_D": psi_D, "psi_S": psi_S,
        "phi_surface": phi_surface,
        "gamma_s": gamma_s,
        "eta1": eta1, "eta2": eta2,
    }
