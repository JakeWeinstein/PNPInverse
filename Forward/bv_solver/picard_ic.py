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


# ---------------------------------------------------------------------------
# Generalized N-reaction Picard outer loop (M3a.3, ORR-class topologies)
#
# See ``docs/picard_general_topology_derivation.md`` v3 for the full
# derivation.  This implements:
#
#   - §3 three-branch BV rate model (anodic surface-species linear,
#     anodic constant, irreversible).
#   - §4 affine matrix system M·R = b with per-species transport
#     coefficient λ_i (1/D_i for ordinary species; 1/(2 D_H) for H⁺).
#   - §7 outer Picard with explicit raw-vs-relaxed rate disambiguation
#     (R_solve / R_old / R) and δ on the relaxed state.
#   - §8 post-loop surface reconstruction with topology-hint dispatch
#     (sequential closed-form P_s for "sequential_2e_h2o2"; naive flux
#     balance otherwise).
#   - §9 contract: legacy-compatible floors before failure classification;
#     reject H⁺ as a linear substrate at adapter sites (item 11).
# ---------------------------------------------------------------------------

def _eta_list_from_drop(
    *,
    eta_drop: float,
    reactions: list,
    bv_exp_scale: float,
    exponent_clip: float,
    clip_exponent: bool,
) -> list[float]:
    """Per-reaction clipped η vector ``η_j = clip(scale·(η_drop − E_j))``."""
    return [
        _eta_clipped(
            eta_drop=eta_drop,
            E=float(rxn.get("E_eq_model", 0.0)),
            bv_exp_scale=bv_exp_scale,
            exponent_clip=exponent_clip,
            clip=clip_exponent,
        )
        for rxn in reactions
    ]


def _build_picard_prefactors(
    *,
    reactions: list,
    log_gamma: float,
    log_by_species: list[float],
    eta_list: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Per-reaction ``(α̂_j, β̂_j, Ĉ_j)`` per derivation v3 §3.

    Three-branch anodic dispatch:

      - branch 1 (surface-species linear): ``reversible AND anodic_species
        is not None`` → ``β̂_j ≠ 0``, ``Ĉ_j = 0``.
      - branch 2 (affine constant): ``reversible AND anodic_species is
        None AND c_ref_model > 1e-30`` → ``Ĉ_j ≠ 0``, ``β̂_j = 0``.
      - branch 3 (irreversible): ``reversible == False`` (or both above
        fail) → ``β̂_j = Ĉ_j = 0``.

    γ-power: ``α̂_j ∝ γ^{1 + Σ_f power_f}``; ``β̂_j ∝ γ^1``;
    ``Ĉ_j ∝ γ^0`` (residual does not multiply ``c_ref_model`` by activity).

    Computed in log-space (mirrors ``picard_outer_loop`` line 481-487).
    """
    alpha_hat = []
    beta_hat = []
    c_hat = []
    for j, rxn in enumerate(reactions):
        k_j = float(rxn["k0_model"])
        a_j = float(rxn["alpha"])
        n_e_j = float(rxn["n_electrons"])
        eta_j = eta_list[j]
        factors_j = rxn.get("cathodic_conc_factors", [])
        log_h_factor_j = _factor_log_from_species_logs(log_by_species, factors_j)
        # α̂_j: cathodic prefactor with γ^{1 + Σ_f power_f} embedded via
        # log_gamma + log_h_factor (which carries one log_gamma per H⁺
        # power because log_by_species is γ-shifted).
        log_alpha = (
            math.log(max(k_j, 1e-300))
            + log_gamma
            + log_h_factor_j
            - a_j * n_e_j * eta_j
        )
        alpha_hat.append(_safe_exp(log_alpha))

        reversible = bool(rxn.get("reversible", False))
        anod_sp = rxn.get("anodic_species", None)
        c_ref = float(rxn.get("c_ref_model", 0.0))

        if reversible and anod_sp is not None:
            # Branch 1: surface-species linear (γ¹).
            log_beta = (
                math.log(max(k_j, 1e-300))
                + log_gamma
                + (1.0 - a_j) * n_e_j * eta_j
            )
            beta_hat.append(_safe_exp(log_beta))
            c_hat.append(0.0)
        elif reversible and anod_sp is None and c_ref > 1e-30:
            # Branch 2: affine constant (γ⁰; no log_gamma).
            log_c = (
                math.log(max(k_j, 1e-300))
                + math.log(max(c_ref, 1e-300))
                + (1.0 - a_j) * n_e_j * eta_j
            )
            beta_hat.append(0.0)
            c_hat.append(_safe_exp(log_c))
        else:
            # Branch 3: irreversible.
            beta_hat.append(0.0)
            c_hat.append(0.0)
    return alpha_hat, beta_hat, c_hat


def _lambda_for_species(
    *, species_idx: int, h_idx: int, diffusivities: list[float]
) -> float:
    """``λ_i = 1/D_i`` for ordinary species; ``λ_H = 1/(2 D_H)`` for H⁺.

    Per derivation v3 §4 sub-note + §9 contract item 11.  Adapter sites
    must reject configs that put H⁺ as ``cathodic_species`` or
    ``anodic_species`` BEFORE calling this function (this helper only
    dispatches the divisor; it does not validate).
    """
    D_i = max(diffusivities[species_idx], 1e-30)
    if species_idx == h_idx:
        return 1.0 / (2.0 * D_i)
    return 1.0 / D_i


def _assemble_n_reaction_system(
    *,
    reactions: list,
    alpha_hat: list[float],
    beta_hat: list[float],
    c_hat: list[float],
    bulk_concs: list[float],
    diffusivities: list[float],
    h_idx: int,
):
    """Assemble M (N×N) and b (N) per derivation v3 §4.

    ``M[j,k] = δ_{j,k} − α̂_j · s_{cathsub_j, k} · λ_{cathsub_j}
              + β̂_j · s_{anodsub_j, k} · λ_{anodsub_j}``  (β̂_j=0 ⇒ second
              term gone)

    ``b[j] = α̂_j · c_{cathsub_j}_b − β̂_j · c_{anodsub_j}_b − Ĉ_j``

    where ``λ_i = 1/D_i`` for ordinary species and ``1/(2 D_H)`` for H⁺
    (substrate, not the H⁺ flux balance which doesn't enter M).

    Returns ``(M, b)`` as lists-of-lists / list (so the caller can
    convert to numpy arrays for the linear solve, keeping this helper
    importable without numpy at module load).
    """
    N = len(reactions)
    # Initialize M = identity, b = zeros.
    M = [[(1.0 if j == k else 0.0) for k in range(N)] for j in range(N)]
    b = [0.0 for _ in range(N)]
    for j, rxn_j in enumerate(reactions):
        # Disabled rxns get a trivial row M[j,j]=1, b[j]=0 ⇒ R_j=0.
        # Avoids accumulating numerically-tiny alpha_hat[j] (from k0=0
        # short-circuited to 1e-300 in _build_picard_prefactors) into M
        # off-diagonals where it could amplify in poorly-conditioned
        # matrices.  Topology classification is unaffected — the row
        # exists, R_j is just identically zero.
        if _is_reaction_disabled(rxn_j):
            continue
        cs = int(rxn_j["cathodic_species"])
        lambda_cs = _lambda_for_species(
            species_idx=cs, h_idx=h_idx, diffusivities=diffusivities
        )
        c_cs_b = float(bulk_concs[cs])
        for k, rxn_k in enumerate(reactions):
            stoich_k = list(rxn_k.get("stoichiometry", []))
            if cs >= len(stoich_k):
                continue
            s_cs_k = float(stoich_k[cs])
            if s_cs_k == 0.0:
                continue
            M[j][k] -= alpha_hat[j] * s_cs_k * lambda_cs
        b[j] += alpha_hat[j] * c_cs_b

        if beta_hat[j] != 0.0:
            anod = rxn_j.get("anodic_species", None)
            if anod is not None:
                anod = int(anod)
                lambda_as = _lambda_for_species(
                    species_idx=anod, h_idx=h_idx, diffusivities=diffusivities
                )
                c_as_b = float(bulk_concs[anod])
                for k, rxn_k in enumerate(reactions):
                    stoich_k = list(rxn_k.get("stoichiometry", []))
                    if anod >= len(stoich_k):
                        continue
                    s_as_k = float(stoich_k[anod])
                    if s_as_k == 0.0:
                        continue
                    M[j][k] += beta_hat[j] * s_as_k * lambda_as
                b[j] -= beta_hat[j] * c_as_b

        b[j] -= c_hat[j]
    return M, b


def _solve_2x2(M: list, b: list) -> tuple[float, float, float]:
    """Direct 2×2 solve preserving legacy ordering for byte-equivalence.

    Returns ``(R0, R1, det)``.  Caller checks ``det`` for singularity.
    Mirrors ``picard_outer_loop`` lines 496-502.
    """
    m11, m12 = M[0][0], M[0][1]
    m21, m22 = M[1][0], M[1][1]
    rhs1, rhs2 = b[0], b[1]
    det = m11 * m22 - m12 * m21
    if not math.isfinite(det) or abs(det) < 1e-300:
        return float("nan"), float("nan"), det
    r0 = (m22 * rhs1 - m12 * rhs2) / det
    r1 = (-m21 * rhs1 + m11 * rhs2) / det
    return r0, r1, det


def _solve_linear_system(M: list, b: list) -> tuple[list[float], float]:
    """Solve M·R = b with size-aware fast path.

    Returns ``(R_list, det_or_nan)`` where ``det_or_nan`` is the 2×2
    determinant for N=2 (used for the byte-equivalent singular-Jacobian
    check) and ``float('nan')`` for N != 2 (callers use np.linalg.solve
    failure modes for those).
    """
    N = len(b)
    if N == 0:
        return [], float("nan")
    if N == 1:
        m = M[0][0]
        if not math.isfinite(m) or abs(m) < 1e-300:
            return [float("nan")], m
        return [b[0] / m], m
    if N == 2:
        r0, r1, det = _solve_2x2(M, b)
        return [r0, r1], det
    # N ≥ 3: numpy.linalg.solve.  Imported lazily so that the
    # 2-reaction sequential path (which is byte-equivalence sensitive)
    # never depends on numpy.
    import numpy as _np
    M_np = _np.asarray(M, dtype=float)
    b_np = _np.asarray(b, dtype=float)
    try:
        R = _np.linalg.solve(M_np, b_np)
    except _np.linalg.LinAlgError:
        return [float("nan")] * N, 0.0
    return [float(v) for v in R], float("nan")


def _surface_concs_from_rates(
    *,
    R: list[float],
    reactions: list,
    bulk_concs: list[float],
    diffusivities: list[float],
    h_idx: int,
    species_floors: list[float],
) -> list[float]:
    """Signed flux balance per derivation v3 §2 with per-species floors.

    ``c_{i,s} = max(c_{i,b} + Σ_j s_{i,j}·R_j · λ_i, floor_i)``  for
    ordinary species (``λ_i = 1/D_i``); H⁺ uses the ambipolar
    ``λ_H = 1/(2 D_H)``.
    """
    n_species = len(bulk_concs)
    out = [float(bulk_concs[i]) for i in range(n_species)]
    for i in range(n_species):
        lam = _lambda_for_species(
            species_idx=i, h_idx=h_idx, diffusivities=diffusivities
        )
        delta = 0.0
        for j, rxn in enumerate(reactions):
            stoich_j = list(rxn.get("stoichiometry", []))
            if i >= len(stoich_j):
                continue
            s_ij = float(stoich_j[i])
            if s_ij == 0.0:
                continue
            delta += s_ij * R[j] * lam
        out[i] = max(out[i] + delta, species_floors[i])
    return out


def _validate_no_h_substrate(
    *, reactions: list, h_idx: int
) -> tuple[bool, str]:
    """v3 §9 item 11: reject H⁺ as cathodic_species or anodic_species.

    Returns ``(ok, reason)``; ``reason = ''`` on pass.  Adapter sites
    should call this before invoking ``picard_outer_loop_general`` and
    fall back to linear-phi IC on rejection.
    """
    for j, rxn in enumerate(reactions):
        cs = rxn.get("cathodic_species", None)
        anod = rxn.get("anodic_species", None)
        if cs is not None and int(cs) == h_idx:
            return False, f"h_plus_as_linear_substrate (cathodic, reaction {j})"
        if anod is not None and int(anod) == h_idx:
            return False, f"h_plus_as_linear_substrate (anodic, reaction {j})"
    return True, ""


def _is_reaction_disabled(rxn: dict) -> bool:
    """A reaction is disabled if ``enabled=False`` or ``k0 <= 0``.

    Mirrors the disabled-reaction guard added to ``forms_logc.py`` /
    ``forms_logc_muh.py``.  Used both at adapter sites for safety asserts
    AND inside ``_assemble_n_reaction_system`` to produce trivial rows
    (``M[j,j] = 1, b[j] = 0 ⇒ R_j = 0``) for disabled reactions.

    Reads the SCALED key ``k0_model`` when present (after nondim) and
    falls back to the raw ``k0`` for adapter-site classification (before
    nondim).  Never raises; missing keys are treated as enabled with
    ``k0 = 0`` (i.e. disabled).
    """
    if not bool(rxn.get("enabled", True)):
        return True
    k0 = float(rxn.get("k0_model", rxn.get("k0", 0.0)))
    return k0 <= 0.0


def _is_parallel_2e_4e(reactions: list, h_idx: int) -> bool:
    """Strict predicate for the parallel-2e/4e ORR topology.

    Classifies from NOMINAL config — disabled reactions don't change
    topology (so the pure-2e probe is "parallel + R_4e disabled" rather
    than "1-rxn sequential"; this preserves topology dispatch in the
    Picard).

    Required topology (per Ruggiero 2022 §1):
      R_2e:  O₂ + 2H⁺ + 2e⁻ → H₂O₂   stoich [-1, +1, -2]   reversible
      R_4e:  O₂ + 4H⁺ + 4e⁻ → 2H₂O   stoich [-1,  0, -4]   irreversible
    """
    if len(reactions) != 2:
        return False
    r2e, r4e = reactions
    if int(r2e.get("n_electrons", -1)) != 2:
        return False
    if int(r4e.get("n_electrons", -1)) != 4:
        return False
    s2 = r2e.get("stoichiometry", [])
    s4 = r4e.get("stoichiometry", [])
    if len(s2) < 3 or len(s4) < 3:
        return False
    if int(s2[1]) != +1:    # R_2e produces H2O2
        return False
    if int(s4[1]) != 0:     # R_4e doesn't touch H2O2
        return False
    if int(s2[0]) != -1 or int(s4[0]) != -1:    # both consume O2
        return False
    if h_idx >= len(s2) or h_idx >= len(s4):
        return False
    if int(s2[h_idx]) != -2:    # R_2e: -2 H+
        return False
    if int(s4[h_idx]) != -4:    # R_4e: -4 H+
        return False
    if not bool(r2e.get("reversible", False)):    # R_2e reversible
        return False
    if bool(r4e.get("reversible", False)):        # R_4e irreversible
        return False
    return True


def picard_outer_loop_general(
    *,
    reactions: list,
    bulk_concs: list[float],
    diffusivities: list[float],
    species_floors: list[float],
    h_idx: int,
    c_clo4_bulk: float,
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
    topology_hint: str = "general",
    verbose: bool = False,
) -> tuple[bool, str, int, dict]:
    """Generalized N-reaction Picard outer loop (v3 §7-§9).

    See ``docs/picard_general_topology_derivation.md`` v3 for the
    derivation and the implementation contract (§9).

    Parameters
    ----------
    reactions : list of dict
        Per-reaction config.  Required keys: ``k0_model``, ``alpha``,
        ``n_electrons``, ``E_eq_model``, ``cathodic_species``,
        ``stoichiometry``.  Optional: ``anodic_species``, ``reversible``,
        ``c_ref_model``, ``cathodic_conc_factors``.  See ``picard_ic``
        module docstring + v3 §1.
    bulk_concs, diffusivities, species_floors : lists of float
        Per dynamic species (length n_species).  ``species_floors[i]`` is
        the lower clamp on ``c_{i,s}`` (e.g. ``1e-300`` for O₂/H⁺;
        ``P_FLOOR`` for H₂O₂).
    h_idx : int
        Index of H⁺ in the species ordering.  Used by the ambipolar
        ``1/(2 D_H)`` flux-balance correction (v3 §2).
    c_clo4_bulk, phi_applied_model, bv_exp_scale, exponent_clip,
    clip_exponent, a_h, a_cl, c_cl_anchor_kind, stern_split, omega,
    max_iters, tol :
        Same semantics as ``picard_outer_loop``; see that function's
        docstring.
    topology_hint : str, optional
        Selects the post-loop surface reconstruction (v3 §8):

        - ``"sequential_2e_h2o2"``: legacy closed-form
          ``P_s = (D_P·P_b + R_1)/(D_P + A_2_post)`` and
          ``O_s = (D_O·O_b + B_1_post·P_s)/(D_O + A_1_post)``.  REQUIRED
          for byte-equivalence with the legacy 2x2 ``picard_outer_loop``.
        - ``"general"`` (default): naive signed flux balance per §2.

    Returns
    -------
    (ok, reason, n_iters, picard_state) : tuple
        ``picard_state`` includes ``R_list``, ``eta_list``,
        ``alpha_hat_list``, ``beta_hat_list``, ``c_hat_list`` for
        diagnostics.  When ``len(reactions) >= 2`` the dict ALSO carries
        legacy keys ``R1, R2, eta1, eta2`` aliased to the first two
        entries — preserves the existing wrapper/test/diagnostic API.
        ``c_i_s`` per dynamic species lives under both the per-species
        key (e.g. ``O_s, P_s, H_o``) for the standard ORR ordering AND
        a generic ``c_s_list`` for arbitrary species lists.
    """
    n_species = len(bulk_concs)
    if not (
        len(diffusivities) == n_species
        and len(species_floors) == n_species
    ):
        return False, "species_arity_mismatch", 0, {}
    if h_idx < 0 or h_idx >= n_species:
        return False, f"h_idx_out_of_range ({h_idx})", 0, {}

    ok_h, reason_h = _validate_no_h_substrate(reactions=reactions, h_idx=h_idx)
    if not ok_h:
        return False, reason_h, 0, {}

    N = len(reactions)
    if N == 0:
        return False, "empty_reactions", 0, {}

    # --- Picard state ---
    R = [0.0] * N
    H_o = float(bulk_concs[h_idx])
    phi_o = 0.0
    c_s = [float(c) for c in bulk_concs]   # working surface concs
    gamma_s = 1.0

    # Initialize Stern + eta_drop for iter 1 (matches legacy ordering).
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

    eta_list = _eta_list_from_drop(
        eta_drop=eta_drop,
        reactions=reactions,
        bv_exp_scale=bv_exp_scale,
        exponent_clip=exponent_clip,
        clip_exponent=clip_exponent,
    )

    delta = float("inf")
    converged = False
    picard_iters = 0
    alpha_hat: list[float] = [0.0] * N
    beta_hat: list[float] = [0.0] * N
    c_hat: list[float] = [0.0] * N

    for k in range(1, max_iters + 1):
        R_old = list(R)

        c_cl_anchor = H_o if c_cl_anchor_kind == "synthesised_4sp" else c_clo4_bulk
        gamma_s = compute_surface_gamma(
            H_o, c_clo4_bulk, psi_D, a_h, a_cl, c_cl_anchor
        )
        log_gamma = math.log(max(gamma_s, 1e-300))

        # γ-shifted log-concentrations at the OHP.
        log_by_species: list[float] = []
        for i in range(n_species):
            base = math.log(max(c_s[i], 1e-300))
            if i == h_idx:
                # H⁺ at OHP is post-Boltzmann shifted by -psi_D.
                log_by_species.append(base - psi_D + log_gamma)
            else:
                log_by_species.append(base + log_gamma)

        alpha_hat, beta_hat, c_hat = _build_picard_prefactors(
            reactions=reactions,
            log_gamma=log_gamma,
            log_by_species=log_by_species,
            eta_list=eta_list,
        )

        M_mat, b_vec = _assemble_n_reaction_system(
            reactions=reactions,
            alpha_hat=alpha_hat,
            beta_hat=beta_hat,
            c_hat=c_hat,
            bulk_concs=bulk_concs,
            diffusivities=diffusivities,
            h_idx=h_idx,
        )

        R_solve, det = _solve_linear_system(M_mat, b_vec)
        if N == 2:
            singular = (
                not math.isfinite(det) or abs(det) < 1e-300
            )
        else:
            singular = any(not math.isfinite(v) for v in R_solve)
        if singular:
            return (
                False,
                f"singular_jacobian_iter_{k}_det={det:.3g}",
                k,
                _state_dict_failure_general(
                    R, c_s, H_o, phi_o, psi_D, psi_S, phi_surface,
                    gamma_s, eta_list, h_idx,
                ),
            )
        if any(not math.isfinite(v) for v in R_solve):
            return (
                False,
                f"non_finite_R_iter_{k}",
                k,
                _state_dict_failure_general(
                    R, c_s, H_o, phi_o, psi_D, psi_S, phi_surface,
                    gamma_s, eta_list, h_idx,
                ),
            )

        # Per-reaction relaxation on raw solve.
        R = [(1.0 - omega) * R_old[j] + omega * R_solve[j] for j in range(N)]

        # Surface concentrations from signed flux balance + per-species floors.
        c_s = _surface_concs_from_rates(
            R=R,
            reactions=reactions,
            bulk_concs=bulk_concs,
            diffusivities=diffusivities,
            h_idx=h_idx,
            species_floors=species_floors,
        )
        H_o = c_s[h_idx]
        phi_o = math.log(max(H_o, 1e-300) / max(c_clo4_bulk, 1e-300))

        # Reject post-update non-finite state (v3 §9 item 8 (iii)).
        if not all(math.isfinite(v) for v in c_s) or not math.isfinite(phi_o):
            return (
                False,
                f"non_finite_state_iter_{k}",
                k,
                _state_dict_failure_general(
                    R, c_s, H_o, phi_o, psi_D, psi_S, phi_surface,
                    gamma_s, eta_list, h_idx,
                ),
            )

        # Stern split + eta_drop update (matches legacy post-update ordering).
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

        eta_list = _eta_list_from_drop(
            eta_drop=eta_drop,
            reactions=reactions,
            bv_exp_scale=bv_exp_scale,
            exponent_clip=exponent_clip,
            clip_exponent=clip_exponent,
        )

        delta = sum(
            abs(R[j] - R_old[j]) / max(abs(R[j]), 1e-30) for j in range(N)
        )
        picard_iters = k
        if verbose:
            R_str = " ".join(f"{x:+.3e}" for x in R)
            cs_str = " ".join(f"{x:+.3e}" for x in c_s)
            eta_str = " ".join(f"{x:+.3e}" for x in eta_list)
            print(
                f"[picard k={k:3d}] delta={delta:.3e}  R=[{R_str}]  "
                f"c_s=[{cs_str}]  phi_o={phi_o:+.3e}  psi_D={psi_D:+.3e}  "
                f"psi_S={psi_S:+.3e}  gamma_s={gamma_s:.3e}  eta=[{eta_str}]",
                flush=True,
            )
        if delta < tol:
            converged = True
            break

    if not converged:
        return (
            False,
            f"picard_max_iters_delta={delta:.4g}",
            picard_iters,
            _state_dict_failure_general(
                R, c_s, H_o, phi_o, psi_D, psi_S, phi_surface,
                gamma_s, eta_list, h_idx,
            ),
        )

    # Post-converged γ_s aligned with the post-iter (psi_D, H_o) tuple.
    c_cl_anchor_post = (
        H_o if c_cl_anchor_kind == "synthesised_4sp" else c_clo4_bulk
    )
    gamma_s = compute_surface_gamma(
        H_o, c_clo4_bulk, psi_D, a_h, a_cl, c_cl_anchor_post
    )
    log_gamma_post = math.log(max(gamma_s, 1e-300))
    log_by_species_post: list[float] = []
    for i in range(n_species):
        base = math.log(max(c_s[i], 1e-300))
        if i == h_idx:
            log_by_species_post.append(base - psi_D + log_gamma_post)
        else:
            log_by_species_post.append(base + log_gamma_post)
    alpha_hat_post, beta_hat_post, c_hat_post = _build_picard_prefactors(
        reactions=reactions,
        log_gamma=log_gamma_post,
        log_by_species=log_by_species_post,
        eta_list=eta_list,
    )

    # Topology-hint dispatch for post-loop surface reconstruction.
    if topology_hint == "sequential_2e_h2o2" and N == 2:
        # Legacy closed-form (v3 §8): P_s = (D_P·P_b + R_1)/(D_P + A_2_post)
        # then O_s = (D_O·O_b + B_1_post·P_s)/(D_O + A_1_post).
        # Hard-coded species order: O=0, P=1, H=2.
        D_O_loc = max(diffusivities[0], 1e-30)
        D_P_loc = max(diffusivities[1], 1e-30)
        P_FLOOR_loc = species_floors[1]
        O_FLOOR_loc = species_floors[0]
        P_b_loc = float(bulk_concs[1])
        O_b_loc = float(bulk_concs[0])
        A1_p, B1_p, A2_p = alpha_hat_post[0], beta_hat_post[0], alpha_hat_post[1]
        P_s_post = max((D_P_loc * P_b_loc + R[0]) / (D_P_loc + A2_p), P_FLOOR_loc)
        O_s_post = max(
            (D_O_loc * O_b_loc + B1_p * P_s_post) / (D_O_loc + A1_p), O_FLOOR_loc
        )
        c_s = list(c_s)
        c_s[0] = O_s_post
        c_s[1] = P_s_post
        # H_o from the loop's signed flux balance is preserved.

    state: dict = {
        "R_list": list(R),
        "c_s_list": list(c_s),
        "eta_list": list(eta_list),
        "alpha_hat_list": list(alpha_hat_post),
        "beta_hat_list": list(beta_hat_post),
        "c_hat_list": list(c_hat_post),
        "H_o": H_o,
        "phi_o": phi_o,
        "psi_D": psi_D, "psi_S": psi_S,
        "phi_surface": phi_surface,
        "gamma_s": gamma_s,
    }
    # Standard ORR per-species aliases (when species 0=O, 1=P, 2=H).
    if n_species >= 3:
        state["O_s"] = c_s[0]
        state["P_s"] = c_s[1]
        # ``H_o`` already set above.
    if N >= 1:
        state["R1"] = R[0]
        state["eta1"] = eta_list[0]
    if N >= 2:
        state["R2"] = R[1]
        state["eta2"] = eta_list[1]

    return True, "", picard_iters, state


def _state_dict_failure_general(
    R: list[float],
    c_s: list[float],
    H_o: float,
    phi_o: float,
    psi_D: float,
    psi_S: float,
    phi_surface: float,
    gamma_s: float,
    eta_list: list[float],
    h_idx: int,
) -> dict:
    """Partial state dict on Picard failure for the generalized loop.

    Mirrors ``_state_dict_failure`` but parametrized by the per-reaction
    R / eta lists.  Includes legacy R1/R2/eta1/eta2 aliases when
    applicable so diagnostic tooling that grew up against the 2x2
    interface keeps working.
    """
    state: dict = {
        "R_list": list(R),
        "c_s_list": list(c_s),
        "eta_list": list(eta_list),
        "H_o": H_o,
        "phi_o": phi_o,
        "psi_D": psi_D, "psi_S": psi_S,
        "phi_surface": phi_surface,
        "gamma_s": gamma_s,
    }
    if len(c_s) >= 3:
        state["O_s"] = c_s[0]
        state["P_s"] = c_s[1]
    if len(R) >= 1:
        state["R1"] = R[0]
        state["eta1"] = eta_list[0]
    if len(R) >= 2:
        state["R2"] = R[1]
        state["eta2"] = eta_list[1]
    return state
