"""Phase 6β v10a — Phase A.2 densified k_hyd × λ ramp at V_kin.

Per the locked acceptance-bundle sequence (step 5; see
``docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`` § "v10a →
E sequence"), this driver runs the densified k_hyd × λ ramp at the
v10a' selected ``V_kin = −0.10 V`` with the v10a Langmuir cap and full
v10a' decompositions emitted per rung.

Its purpose is twofold:

1. Confirm the v10a ``(1 − Γ/Γ_max)`` cap performs as designed at the
   new V_kin (v9 Phase B failed Picard at ``k_hyd ≥ 1e-1`` without a
   cap; v10a should make the full grid converge with ``Γ → Γ_max`` as
   ``k_hyd → ∞``).
2. Map the ``(k_hyd, λ) → (σ_S, c_H, Γ, θ, R_net, branch ratio,
   denom_cap_to_total_ratio)`` response surface at V_kin to feed
   v10b's literature calibration of ``Γ_max + k_des`` as PRIORITIES,
   not as derived calibration values.  (A k_hyd ramp alone does not
   identify ``k_des·Γ_max``; that's v10b's job.)

Hardened by GPT critique session 34 (4 rounds, APPROVED).  See
``docs/handoffs/CHATGPT_HANDOFF_34_phase6b-v10a-phase-A2-v-kin/`` and
``~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md`` for the full
provenance (41 issues across 4 rounds).  Key load-bearing properties
baked in here:

* Two-stage anchor: build at ``C_S = 0.10`` (convergence-pinned),
  runtime-bump to ``C_S = 0.20`` via ``set_stern_capacitance_model``
  plus Newton resolve.  Reuses the v10a' helper
  :func:`scripts.studies.phase6b_v10a_v_sweep_diagnostic._walk_lambda_zero_capture_snapshots`.
* ``rung_callback`` is the SOLE source of truth (R4 #4): per-(k_hyd, λ)
  output is built from ``augmented_rungs`` and ``partial_rungs`` only;
  ``result.rungs`` is intentionally not consumed.
* Callback exceptions are wrapped (R4 #3) so a per-rung augment failure
  surfaces as ``callback_augment_error`` rather than silently
  swallowing the snapshot.
* ``classify_picard_status`` covers 6 statuses including
  ``converged_at_iter_cap`` and ``single_iter`` (R3 #1, R2 #4).
* Mass-balance residual (R3 #10) is a HARD GATE for k_hyd_route
  eligibility.
* ``exception_phase`` is decoded from ``LadderExhausted`` text (R3 #5,
  R4 #5); a unit test guards against solver text drift.
* ``single_v_selectivity_gap_pp`` is a single-V proxy for the
  bundle's window-max selectivity criterion (R4 #1) — explicitly NOT
  a bundle pass/fail, only informs v10b priority.

Output (in ``StudyResults/<out-subdir>/``):

* ``phase_a2_v_kin.json`` — config block + per-(k_hyd, λ) records +
  ``lambda_zero_baseline_at_v_kin`` single shared baseline +
  ``v10b_priorities`` block.
* ``phase_a2_v_kin.png`` — 6-panel summary at λ=1.0 across the
  converged k_hyd subset.

Usage::

    source ../venv-firedrake/bin/activate
    python -u scripts/studies/phase6b_v10a_phase_A2_v_kin.py \\
        --v-kin -0.10 --k0-r4e-factor 1e-14 \\
        --out-subdir phase6b_v10a_phase_A2_v_kin
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level path setup — keep Firedrake imports lazy (inside functions)
# so the helpers below are importable from tests without the venv.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Constants — locked at the v10a' result (project_v10a_prime_outcome).
# ---------------------------------------------------------------------------

V_KIN_DEFAULT: float = -0.10
"""Kinetic-V selected by the v10a' V-sweep diagnostic.

See ``StudyResults/phase6b_v10a_prime_k0r4e_1e-14/iv_diagnostic.json``
and ``docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`` § v10a'
Result for the per-V breakdown.  HARD invariant across A.2 + v10b +
B.2 (R2 #17); changing this requires re-running v10a' + A.2.
"""

K0_R4E_FACTOR_DEFAULT: float = 1e-14
"""Dimensionless multiplier on ``K0_HAT_R4E`` — the v10a' branch-pass
probe at V=−0.10.  Same HARD-invariant status as V_KIN_DEFAULT.
"""

K_HYD_GRID_DEFAULT: Tuple[float, ...] = (
    1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1,
)
"""Densified k_hyd ramp — 10 points (R3 #5 expanded).

Targets the cap-onset transition (θ=0.5 at k_hyd≈1.6e-4 per the
closed-form Γ_ss; θ=0.9 at k_hyd≈1.45e-3) plus the saturation
plateau.  Half-decade through onset, decade in saturation.  See the
predicted-table sanity check in the plan §Expected qualitative
behavior.
"""

# Transition / saturation sub-grids — used by the pass-criterion check.
TRANSITION_GRID: Tuple[float, ...] = (
    1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3,
)
SATURATION_GRID: Tuple[float, ...] = (1e-2, 1e-1)

WARM_WALK_GRID: Tuple[float, ...] = (+0.55, +0.40, +0.20, +0.10, -0.10)
"""5-point grid for the λ=0 warm-walk from the anchor to V_kin.

Intermediates are by-products of the v10a' helper's grid mode (R3 #1);
only the V_kin snapshot is used for the k_hyd × λ ramp.
"""

V_ANCHOR_DEFAULT: float = 0.55

# Routing thresholds (plan §Routing) — all checked against λ=1.0 rung.
DECK_SELECTIVITY_BAND: Tuple[float, float] = (25.0, 50.0)
"""Deck-band H2O2% per acceptance bundle's window-max criterion."""

KHYD_ROUTE_THETA_MIN: float = 0.95
"""Stricter than the 0.9 cap-coverage line — k_hyd_route requires θ>0.95."""

KHYD_ROUTE_SLOPE_MAX_LN: float = 0.05
"""Local d ln Γ / d ln k_hyd gate — k_hyd_route requires < 0.05."""

TRANSPORT_O2_MAX: float = 0.9
"""Transport gate — k_hyd_route requires o2_flux_levich_ratio < 0.9."""

MASS_BALANCE_RESIDUAL_REL_MAX: float = 5e-3
"""HARD GATE per R3 #10 — |mass_balance_residual_rel| < 5e-3 at λ=1.

Catches residual-side bugs that the closed-form Γ_ss identity would
otherwise mask."""

# Pass criterion thresholds (plan §Convergence audit).
BASELINE_REPRODUCTION_REL_TOL: float = 1e-3
TRANSITION_THETA_LO: float = 0.10
TRANSITION_THETA_HI: float = 0.93
TRANSITION_MIN_COUNT: int = 4
SATURATION_SLOPE_MAX_LN: float = 0.10

# Selectivity priority cutoff (plan §Routing).
SELECTIVITY_PRIORITY_THRESHOLD_PP: float = 10.0

# Singh-amplification recalibration cutoff (plan §Routing).
SINGH_AMP_RECALIBRATION_THRESHOLD: float = 10.0
SINGH_AMP_SMALL_THRESHOLD: float = 2.0

# Expected exception-text prefixes from solve_lambda_ramp_from_warm_start
# — guarded by the prefix-drift unit test (R4 #5).
EXPECTED_LADDER_EXHAUSTED_PREFIXES: Tuple[str, ...] = (
    "warm-start SS re-converge failed",
    "λ=0 floor solve failed",
    "λ_hydrolysis ramp exhausted at λ=",
)

OUT_SUBDIR_DEFAULT: str = "phase6b_v10a_phase_A2_v_kin"


# ---------------------------------------------------------------------------
# Pure-Python helpers (no Firedrake) — used by both the driver and the
# unit tests.  These MUST stay importable without the venv.
# ---------------------------------------------------------------------------


def classify_picard_status(
    history: List[float], snes_converged: bool,
) -> str:
    """Classify per-rung Picard convergence per plan §Implementation notes.

    Returns one of:
      "snes_failed"               — overrides everything; Newton failed.
      "no_iters"                  — defensive; no Picard iterations ran.
      "single_iter"               — 1 iter; rel<1e-4 implied by helper.
      "converged"                 — n<8 AND last_rel<1e-4.
      "early_break"               — n<8 AND last_rel>=1e-4 (quit early).
      "converged_at_iter_cap"     — n==8 AND last_rel<1e-4 (valid).
      "iter_cap_hit_unconverged"  — n==8 AND last_rel>=1e-4 (failure).

    NOTE: ``single_iter`` success is defensible because under the
    current helper control flow at ``anchor_continuation.py:1819-1830``,
    ``snes_converged=True`` with ``len(history)==1`` implies the
    internal Picard rel test passed on the first update (the loop
    ``break``s on ``rel<rel_tol``).  Future solver edits that change
    the loop control could invalidate this assumption — keep this
    comment in sync with the helper.
    """
    if not snes_converged:
        return "snes_failed"
    n = len(history)
    if n == 0:
        return "no_iters"
    if n == 1:
        return "single_iter"
    last_rel = abs(history[-1] - history[-2]) / max(
        abs(history[-1]), abs(history[-2]), 1e-30,
    )
    if n < 8:
        return "converged" if last_rel < 1e-4 else "early_break"
    return (
        "converged_at_iter_cap" if last_rel < 1e-4
        else "iter_cap_hit_unconverged"
    )


def single_v_selectivity_gap_pp(
    observed_pct: float,
    deck_band: Tuple[float, float] = DECK_SELECTIVITY_BAND,
) -> float:
    """Signed distance (percentage points) between observed H2O2% and the deck band.

    Returns
    -------
    Positive iff observed < band low; negative iff observed > band high;
    ``0.0`` if observed is inside [lo, hi].

    Renamed from ``selectivity_gap_pp`` per R4 #1: this is a single-V
    proxy, NOT a bundle pass/fail criterion.  The acceptance bundle's
    primary criterion is "per-cation MAX H₂O₂% in V_RHE WINDOW"; A.2 at
    fixed V_kin cannot prove window-max agreement.
    """
    lo, hi = deck_band
    if observed_pct < lo:
        return lo - observed_pct
    if observed_pct > hi:
        return hi - observed_pct
    return 0.0


def lambda1_record(per_k_hyd_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the λ=1.0 augmented rung for a per-k_hyd record (R4 #2).

    Returns ``None`` if the record has no λ=1 rung — e.g. AdaptiveLadder
    exhausted before reaching λ=1, or the ladder bailed in the
    ``warm_reconverge`` / ``lambda_zero`` phases.

    All routing gates ONLY look at the λ=1 record; never at partial
    rungs inserted by AdaptiveLadder midpoints.
    """
    for rung in per_k_hyd_record.get("rungs", []):
        lam = rung.get("lambda_hydrolysis")
        if lam is None:
            continue
        if abs(float(lam) - 1.0) < 1e-12:
            return rung
    return None


def compute_mass_balance_residual_rel(
    rung_diag: Dict[str, Any],
) -> Optional[float]:
    """At λ=1 SS, ``R_forward_capped − denominator_kprot·γ − k_des·γ ≈ 0``.

    Catches residual-side bugs that the closed-form Γ_ss identity
    would otherwise mask (the closed-form makes R_net = k_des·Γ a
    tautology at SS).

    Returns ``None`` if any required field is missing.
    """
    forward_capped = rung_diag.get("R_forward_capped")
    denom_kprot = rung_diag.get("denominator_kprot")
    gamma = rung_diag.get("gamma")
    k_des = rung_diag.get("k_des")
    gamma_max = rung_diag.get("gamma_max")
    if any(x is None for x in (forward_capped, denom_kprot, gamma, k_des)):
        return None
    gamma_max_eff = float(gamma_max) if gamma_max is not None else 0.047
    R_resid = (
        float(forward_capped)
        - float(denom_kprot) * float(gamma)
        - float(k_des) * float(gamma)
    )
    denom = max(
        abs(float(forward_capped)),
        abs(float(k_des) * gamma_max_eff),
        1e-30,
    )
    return abs(R_resid) / denom


def augment_rung_diagnostics(
    rung_diag: Dict[str, Any],
    *,
    i_scale: float,
    i_lim_4e_mA_cm2: float,
    electrode_area_nondim: float,
    domain_height_hat: float,
    snes_converged: bool,
    gamma_picard_history: List[float],
    pc_mA_cm2: Optional[float],
) -> Dict[str, Any]:
    """Compute A.2-specific derived fields and return a defensive shallow copy.

    Implements the per-rung augmentations listed in plan §Implementation
    notes (cd_mA_cm2, pc_mA_cm2, x_2e, H2O2_selectivity_pct,
    o2_flux_levich_ratio, current_filter_ratio, picard_status,
    mass_balance_residual_rel).

    Does NOT mutate the input ``rung_diag``.
    """
    snapshot: Dict[str, Any] = dict(rung_diag)
    snapshot["picard_status"] = classify_picard_status(
        list(gamma_picard_history or []), bool(snes_converged),
    )

    cd_obs = snapshot.get("cd_observable")
    snapshot["cd_mA_cm2"] = (
        -float(i_scale) * float(cd_obs) if cd_obs is not None else None
    )
    snapshot["pc_mA_cm2"] = pc_mA_cm2

    r2 = snapshot.get("R_2e_current_nondim")
    r4 = snapshot.get("R_4e_current_nondim")
    if r2 is not None and r4 is not None and abs(float(r2) + float(r4)) > 1e-30:
        denom = float(r2) + float(r4)
        x2 = float(r2) / denom
        x4 = float(r4) / denom
        snapshot["x_2e"] = x2
        snapshot["x_4e"] = x4
        snapshot["H2O2_selectivity_pct"] = 100.0 * x2
    else:
        snapshot["x_2e"] = None
        snapshot["x_4e"] = None
        snapshot["H2O2_selectivity_pct"] = None

    # o2_flux_levich_ratio — lazy-imported because the helper lives in
    # the v10a' driver and we want this module importable from tests.
    if (
        r2 is not None and r4 is not None
        and electrode_area_nondim > 0.0
        and domain_height_hat > 0.0
    ):
        try:
            from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
                _compute_o2_flux_levich_ratio,
            )
            snapshot["o2_flux_levich_ratio"] = _compute_o2_flux_levich_ratio(
                R_2e_current_nondim=float(r2),
                R_4e_current_nondim=float(r4),
                electrode_area_nondim=float(electrode_area_nondim),
                domain_height_hat=float(domain_height_hat),
            )
        except Exception as exc:                              # pragma: no cover
            snapshot["o2_flux_levich_ratio"] = None
            snapshot["o2_flux_levich_ratio_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
    else:
        snapshot["o2_flux_levich_ratio"] = None

    if snapshot["cd_mA_cm2"] is not None and i_lim_4e_mA_cm2 > 0.0:
        snapshot["current_filter_ratio"] = (
            abs(snapshot["cd_mA_cm2"]) / i_lim_4e_mA_cm2
        )
    else:
        snapshot["current_filter_ratio"] = None

    snapshot["mass_balance_residual_rel"] = (
        compute_mass_balance_residual_rel(snapshot)
    )
    return snapshot


def _classify_route_eligibility(lam1: Dict[str, Any]) -> Tuple[bool, str]:
    """Helper for routing/no-route classification.

    Returns ``(eligible_for_route, reason_if_not)``.  ``reason_if_not``
    is one of:
      "no_lambda1", "snes_failed", "theta_below_min",
      "picard_failure", "mass_balance_failure", "transport_only", "ok".
    """
    if lam1 is None:
        return False, "no_lambda1"
    if not lam1.get("snes_converged", False):
        return False, "snes_failed"
    theta = lam1.get("theta")
    if theta is None or theta <= KHYD_ROUTE_THETA_MIN:
        return False, "theta_below_min"
    if lam1.get("picard_status") not in (
        "converged", "converged_at_iter_cap", "single_iter",
    ):
        return False, "picard_failure"
    mb = lam1.get("mass_balance_residual_rel")
    if mb is None or mb >= MASS_BALANCE_RESIDUAL_REL_MAX:
        return False, "mass_balance_failure"
    o2 = lam1.get("o2_flux_levich_ratio")
    if o2 is None or o2 >= TRANSPORT_O2_MAX:
        return False, "transport_only"
    return True, "ok"


def select_k_hyd_route(
    per_k_hyd_records: List[Dict[str, Any]],
) -> Optional[float]:
    """Find ``k_hyd_route`` per plan §Routing.

    Returns the highest k_hyd in the converged set whose λ=1 rung
    satisfies ALL gates:

      * ``theta > 0.95``
      * ``d ln Γ / d ln k_hyd < 0.05`` between this k_hyd and the
        next-lower converged k_hyd
      * ``o2_flux_levich_ratio < 0.9``
      * ``picard_status ∈ {converged, converged_at_iter_cap, single_iter}``
      * ``|mass_balance_residual_rel| < 5e-3`` (HARD GATE per R3 #10)

    Returns ``None`` if no candidate qualifies (caller routes via
    :func:`classify_no_route_cause`).
    """
    rows: List[Tuple[float, float, Dict[str, Any]]] = []
    for rec in per_k_hyd_records:
        lam1 = lambda1_record(rec)
        eligible, _ = _classify_route_eligibility(lam1)
        if not eligible:
            continue
        gamma = float(lam1["gamma"]) if lam1.get("gamma") is not None else None
        if gamma is None or gamma <= 0.0:
            continue
        rows.append((float(rec["k_hyd_target"]), gamma, lam1))
    rows.sort(key=lambda r: r[0])

    for i in reversed(range(len(rows))):
        kh, gamma, _lam1 = rows[i]
        if i == 0:
            continue  # can't compute slope without a lower neighbour
        kh_lo, gamma_lo, _ = rows[i - 1]
        if gamma_lo <= 0.0 or kh <= 0.0 or kh_lo <= 0.0:
            continue
        log_kh = math.log(kh)
        log_kh_lo = math.log(kh_lo)
        if abs(log_kh - log_kh_lo) < 1e-30:
            continue
        slope = abs(
            (math.log(gamma) - math.log(gamma_lo)) / (log_kh - log_kh_lo)
        )
        if slope >= KHYD_ROUTE_SLOPE_MAX_LN:
            continue
        return kh
    return None


def classify_no_route_cause(
    per_k_hyd_records: List[Dict[str, Any]],
) -> str:
    """Classify the cause of "no k_hyd_route exists" per plan §Routing.

    Order matters (R3 #6 + R4 #6):
      1. ``no_saturated_rung``     — cap never engaged anywhere.
      2. ``picard_failure``        — saturated rungs but Picard failed.
      3. ``mass_balance_failure``  — Picard OK but mass-balance residual
                                     above HARD GATE.
      4. ``transport_only``        — saturated/clean rungs all transport-limited.
      5. ``grid_gap``              — saturated/clean/transport-OK rung
                                     exists but local-slope gate fails.

    Returns one of the five strings; ``"picard_failure"`` if no λ=1
    record is even found (defensive).
    """
    lam1_records: List[Tuple[float, Dict[str, Any]]] = []
    for rec in per_k_hyd_records:
        lam1 = lambda1_record(rec)
        if lam1 is not None and lam1.get("snes_converged", False):
            lam1_records.append((float(rec["k_hyd_target"]), lam1))

    if not lam1_records:
        return "picard_failure"

    # 1. no_saturated_rung — gate at cap-coverage line, 0.9 (NOT 0.95).
    saturated = [
        (kh, r) for kh, r in lam1_records
        if (r.get("theta") is not None) and (float(r["theta"]) > 0.9)
    ]
    if not saturated:
        return "no_saturated_rung"

    # 2. picard_failure
    saturated_picard_ok = [
        (kh, r) for kh, r in saturated
        if r.get("picard_status") in (
            "converged", "converged_at_iter_cap", "single_iter",
        )
    ]
    if not saturated_picard_ok:
        return "picard_failure"

    # 3. mass_balance_failure
    saturated_mass_ok = [
        (kh, r) for kh, r in saturated_picard_ok
        if (
            r.get("mass_balance_residual_rel") is not None
            and float(r["mass_balance_residual_rel"])
            < MASS_BALANCE_RESIDUAL_REL_MAX
        )
    ]
    if not saturated_mass_ok:
        return "mass_balance_failure"

    # 4. transport_only
    transport_ok = [
        (kh, r) for kh, r in saturated_mass_ok
        if (
            r.get("o2_flux_levich_ratio") is not None
            and float(r["o2_flux_levich_ratio"]) < TRANSPORT_O2_MAX
        )
    ]
    if not transport_ok:
        return "transport_only"

    # 5. grid_gap — saturated+clean+transport-OK rung exists but the
    # slope gate kept select_k_hyd_route from returning it.
    return "grid_gap"


def build_v10b_priorities_block(
    per_k_hyd_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the ``v10b_priorities`` JSON sub-block per plan §Routing.

    Always emits every key (load-bearing for downstream readers).
    """
    block: Dict[str, Any] = {
        "k_hyd_route": None,
        "single_v_selectivity_gap_pp": None,
        "selectivity_band": list(DECK_SELECTIVITY_BAND),
        "kdes_gammamax_priority": None,
        "rHEl_recalibration_required": None,
        "max_amp_from_singh": None,
        "transport_re_entry_first_k_hyd": None,
        "no_route_cause": None,
    }

    amps: List[float] = []
    transport_re_entry: Optional[float] = None
    for rec in sorted(
        per_k_hyd_records, key=lambda r: float(r.get("k_hyd_target", 0.0)),
    ):
        lam1 = lambda1_record(rec)
        if lam1 is None or not lam1.get("snes_converged", False):
            continue
        f0_decomp = lam1.get("F0_decomposition") or {}
        amp = f0_decomp.get("amplification_from_singh")
        if amp is not None:
            try:
                amps.append(float(amp))
            except Exception:
                pass
        o2 = lam1.get("o2_flux_levich_ratio")
        if (
            o2 is not None
            and float(o2) >= TRANSPORT_O2_MAX
            and transport_re_entry is None
        ):
            transport_re_entry = float(rec["k_hyd_target"])
    if amps:
        block["max_amp_from_singh"] = max(amps)
    block["transport_re_entry_first_k_hyd"] = transport_re_entry

    if block["max_amp_from_singh"] is not None:
        block["rHEl_recalibration_required"] = bool(
            block["max_amp_from_singh"] > SINGH_AMP_RECALIBRATION_THRESHOLD
        )

    route = select_k_hyd_route(per_k_hyd_records)
    block["k_hyd_route"] = route

    if route is None:
        block["no_route_cause"] = classify_no_route_cause(per_k_hyd_records)
        return block

    for rec in per_k_hyd_records:
        if abs(float(rec["k_hyd_target"]) - float(route)) <= 1e-30:
            lam1 = lambda1_record(rec)
            if lam1 is None:
                break
            sel = lam1.get("H2O2_selectivity_pct")
            if sel is None:
                break
            gap = single_v_selectivity_gap_pp(float(sel))
            block["single_v_selectivity_gap_pp"] = gap
            block["kdes_gammamax_priority"] = (
                "high" if abs(gap) > SELECTIVITY_PRIORITY_THRESHOLD_PP
                else "low"
            )
            break

    return block


# ===========================================================================
# Below: Firedrake-using driver code.  Tests should not import below here.
# ===========================================================================


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6β v10a Phase A.2 — densified k_hyd × λ ramp at V_kin."
        ),
    )
    parser.add_argument(
        "--v-kin", type=float, default=V_KIN_DEFAULT,
        help=f"Kinetic-V for the k_hyd ramp. Default {V_KIN_DEFAULT}.",
    )
    parser.add_argument(
        "--v-anchor", type=float, default=V_ANCHOR_DEFAULT,
        help=f"Anchor voltage (V vs RHE). Default {V_ANCHOR_DEFAULT}.",
    )
    parser.add_argument(
        "--k0-r4e-factor", type=float, default=K0_R4E_FACTOR_DEFAULT,
        help=(
            "Dimensionless multiplier on K0_HAT_R4E. Default "
            f"{K0_R4E_FACTOR_DEFAULT:.3g}."
        ),
    )
    parser.add_argument(
        "--k-hyd-grid", default=None,
        help=(
            "Comma-separated k_hyd grid (nondim). Default = "
            f"{','.join(repr(x) for x in K_HYD_GRID_DEFAULT)} (10 points)."
        ),
    )
    parser.add_argument(
        "--with-perturbation", dest="with_perturbation", action="store_true",
        default=False,
        help=(
            "Add ±C_S perturbation column at k_hyd_route only (~3× wall on "
            "that rung).  Default: skip."
        ),
    )
    parser.add_argument(
        "--out-subdir", default=OUT_SUBDIR_DEFAULT,
        help=f"StudyResults/<subdir> output location. Default {OUT_SUBDIR_DEFAULT}.",
    )
    parser.add_argument(
        "--plot", dest="plot", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
    )
    return parser.parse_args(argv)


def _parse_k_hyd_grid(raw: Optional[str]) -> Tuple[float, ...]:
    """Parse ``--k-hyd-grid`` value (comma-separated) into a tuple of floats."""
    if raw is None or raw == "":
        return K_HYD_GRID_DEFAULT
    values: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        values.append(float(tok))
    if not values:
        raise ValueError(
            f"--k-hyd-grid parsed to empty list (raw={raw!r})"
        )
    return tuple(values)


def _run_k_hyd_ramp(
    *,
    sp_template,
    mesh,
    voltage: float,
    U_warmstart: tuple,
    k_hyd_target: float,
    k0_r4e_factor: float,
    i_scale: float,
    i_lim_4e_mA_cm2: float,
    electrode_area_nondim: float,
    domain_height_hat: float,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the λ ladder at V_kin for one k_hyd_target via the callback side-channel.

    Implements the per-(k_hyd, λ) record schema from plan §Per-(k_hyd, λ)
    record schema.  Returns a single dict with ``rungs`` (converged
    augmentations) + ``partial_rungs`` (non-converged augmentations) +
    ``exception_phase`` + ``ladder_converged``.
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted, solve_lambda_ramp_from_warm_start,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import K0_HAT_R2E, K0_HAT_R4E, V_T

    augmented_rungs: List[Dict[str, Any]] = []
    partial_rungs: List[Dict[str, Any]] = []

    def _rung_callback(scale, ok, ctx, rung_diag):
        # R4 #3 — wrap the entire callback body in try/except so a
        # per-rung augment failure surfaces as callback_augment_error
        # rather than silently dropping the snapshot.
        try:
            snapshot = dict(rung_diag)
            gamma_history = rung_diag.get("gamma_picard_history", []) or []
            pc_mA_cm2: Optional[float] = None
            if ok:
                try:
                    pc_form = _build_bv_observable_form(
                        ctx, mode="peroxide_current",
                        reaction_index=None, scale=-i_scale,
                    )
                    pc_mA_cm2 = float(fd.assemble(pc_form))
                except Exception as exc:                       # pragma: no cover
                    snapshot["pc_mA_cm2_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            augmented = augment_rung_diagnostics(
                snapshot,
                i_scale=i_scale,
                i_lim_4e_mA_cm2=i_lim_4e_mA_cm2,
                electrode_area_nondim=electrode_area_nondim,
                domain_height_hat=domain_height_hat,
                snes_converged=bool(ok),
                gamma_picard_history=gamma_history,
                pc_mA_cm2=pc_mA_cm2,
            )
            if ok:
                augmented_rungs.append(augmented)
            else:
                partial_rungs.append(augmented)
        except Exception as exc:
            unaug = dict(rung_diag)
            unaug["callback_augment_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            if ok:
                augmented_rungs.append(unaug)
            else:
                partial_rungs.append(unaug)

    # LAMBDA_LADDER is module-level data in the v10a' driver — pure
    # Python (no Firedrake), safe to import at function entry.
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import LAMBDA_LADDER

    sp_at_v = sp_template.with_phi_applied(voltage / V_T)
    k0_r4e_target = float(K0_HAT_R4E) * float(k0_r4e_factor)

    overrides: Dict[str, Any] = {"k_hyd": float(k_hyd_target)}
    if extra_overrides:
        for k, v in extra_overrides.items():
            overrides[k] = v

    exception_phase: Optional[str] = None
    exception_str: Optional[str] = None
    t_start = time.time()
    try:
        with adj.stop_annotating():
            result = solve_lambda_ramp_from_warm_start(
                sp_at_v, mesh=mesh, U_warmstart=U_warmstart,
                k0_targets={0: float(K0_HAT_R2E), 1: k0_r4e_target},
                lambda_hydrolysis_ladder=tuple(float(x) for x in LAMBDA_LADDER),
                parameter_overrides=overrides,
                rung_callback=_rung_callback,
                max_ss_steps_per_rung=300,
            )
        ladder_converged = bool(result.converged)
    except LadderExhausted as exc:
        ladder_converged = False
        exception_str = str(exc)
        if "warm-start SS re-converge failed" in exception_str:
            exception_phase = "warm_reconverge"
        elif "λ=0 floor solve failed" in exception_str:
            exception_phase = "lambda_zero"
        elif augmented_rungs or partial_rungs:
            exception_phase = "lambda_positive"
        else:
            exception_phase = "unknown"

    return {
        "k_hyd_target": float(k_hyd_target),
        "ladder_converged": ladder_converged,
        "exception_phase": exception_phase,
        "exception": exception_str,
        "wall_seconds": float(time.time() - t_start),
        "rungs": augmented_rungs,
        "partial_rungs": partial_rungs,
    }


def _extract_lambda_zero_baseline(
    *,
    warm_walk_records: List[Dict[str, Any]],
    v_kin: float,
) -> Optional[Dict[str, Any]]:
    """Find the V_kin entry in the warm-walk records.

    Plan: λ=0 baseline at V_kin is captured ONCE during the warm-walk
    via :func:`_walk_lambda_zero_capture_snapshots`'s per_point_callback;
    stored separately as ``lambda_zero_baseline_at_v_kin`` in the JSON.
    """
    for rec in warm_walk_records:
        v = rec.get("v_rhe")
        if v is None:
            continue
        if abs(float(v) - float(v_kin)) < 1e-9:
            return dict(rec)
    return None


def _convergence_audit(
    per_k_hyd_records: List[Dict[str, Any]],
    lambda_zero_baseline_at_v_kin: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the three-criterion convergence audit per plan §Convergence audit."""
    audit: Dict[str, Any] = {
        "baseline_reproduction": None,
        "transition_coverage": None,
        "saturation_coverage": None,
        "overall_pass": None,
    }
    # Baseline reproduction at k_hyd=1e-3 — compare λ=1 rung to v10a' record.
    baseline_target = {
        "gamma": 0.0405, "theta": 0.861,
        "sigma_S_C_per_m2": -0.01715, "cd_mA_cm2": -3.12,
    }
    base_rec = None
    for rec in per_k_hyd_records:
        if abs(float(rec["k_hyd_target"]) - 1e-3) <= 1e-30:
            base_rec = rec
            break
    if base_rec is not None:
        lam1 = lambda1_record(base_rec)
        if lam1 is not None and lam1.get("snes_converged", False):
            diffs: Dict[str, float] = {}
            for key, ref in baseline_target.items():
                obs = lam1.get(key)
                if obs is None:
                    diffs[key] = None
                else:
                    diffs[key] = (
                        abs(float(obs) - ref)
                        / max(abs(ref), 1e-30)
                    )
            picard_ok = lam1.get("picard_status") in (
                "converged", "converged_at_iter_cap", "single_iter",
            )
            mb_rel = lam1.get("mass_balance_residual_rel")
            mb_ok = (
                mb_rel is not None
                and float(mb_rel) < MASS_BALANCE_RESIDUAL_REL_MAX
            )
            within_tol = all(
                (d is not None) and (d < BASELINE_REPRODUCTION_REL_TOL)
                for d in diffs.values()
            )
            audit["baseline_reproduction"] = {
                "target": baseline_target,
                "observed": {k: lam1.get(k) for k in baseline_target},
                "relative_diffs": diffs,
                "rel_tol": BASELINE_REPRODUCTION_REL_TOL,
                "within_tol": within_tol,
                "picard_status": lam1.get("picard_status"),
                "picard_ok": picard_ok,
                "mass_balance_residual_rel": mb_rel,
                "mass_balance_ok": mb_ok,
                "pass": bool(within_tol and picard_ok and mb_ok),
            }
        else:
            audit["baseline_reproduction"] = {
                "pass": False, "reason": "k_hyd=1e-3 λ=1 did not converge",
            }
    else:
        audit["baseline_reproduction"] = {
            "pass": False, "reason": "k_hyd=1e-3 not in grid",
        }

    # Transition coverage.
    transition_thetas: List[Tuple[float, float]] = []
    for rec in per_k_hyd_records:
        kh = float(rec["k_hyd_target"])
        if not any(abs(kh - kt) <= kh * 1e-9 for kt in TRANSITION_GRID):
            continue
        lam1 = lambda1_record(rec)
        if lam1 is None or not lam1.get("snes_converged", False):
            continue
        theta = lam1.get("theta")
        if theta is None:
            continue
        transition_thetas.append((kh, float(theta)))
    thetas_vals = [t for _, t in transition_thetas]
    min_theta = min(thetas_vals) if thetas_vals else None
    max_theta = max(thetas_vals) if thetas_vals else None
    coverage_ok = (
        min_theta is not None and min_theta <= TRANSITION_THETA_LO
        and max_theta is not None and max_theta >= TRANSITION_THETA_HI
        and len(thetas_vals) >= TRANSITION_MIN_COUNT
    )
    audit["transition_coverage"] = {
        "min_theta": min_theta,
        "max_theta": max_theta,
        "converged_count": len(thetas_vals),
        "min_count_required": TRANSITION_MIN_COUNT,
        "theta_lo_threshold": TRANSITION_THETA_LO,
        "theta_hi_threshold": TRANSITION_THETA_HI,
        "pass": bool(coverage_ok),
    }

    # Saturation coverage.
    saturation_rows: List[Tuple[float, float, float]] = []
    for rec in per_k_hyd_records:
        kh = float(rec["k_hyd_target"])
        if not any(abs(kh - kt) <= kh * 1e-9 for kt in SATURATION_GRID):
            continue
        lam1 = lambda1_record(rec)
        if lam1 is None or not lam1.get("snes_converged", False):
            continue
        theta = lam1.get("theta")
        gamma = lam1.get("gamma")
        if theta is None or gamma is None:
            continue
        saturation_rows.append((kh, float(theta), float(gamma)))
    saturation_rows.sort()
    sat_theta_ok = any(
        t > KHYD_ROUTE_THETA_MIN for _, t, _ in saturation_rows
    )
    sat_slope_ok = False
    if len(saturation_rows) >= 2:
        kh_a, _, gamma_a = saturation_rows[-2]
        kh_b, _, gamma_b = saturation_rows[-1]
        if (
            gamma_a > 0.0 and gamma_b > 0.0
            and kh_a > 0.0 and kh_b > 0.0
            and abs(math.log(kh_b) - math.log(kh_a)) > 1e-30
        ):
            slope = abs(
                (math.log(gamma_b) - math.log(gamma_a))
                / (math.log(kh_b) - math.log(kh_a))
            )
            sat_slope_ok = slope < SATURATION_SLOPE_MAX_LN
    audit["saturation_coverage"] = {
        "rows": saturation_rows,
        "saturated_theta_ok": sat_theta_ok,
        "saturated_slope_ok": sat_slope_ok,
        "slope_max_ln": SATURATION_SLOPE_MAX_LN,
        "pass": bool(sat_theta_ok and sat_slope_ok),
    }

    audit["overall_pass"] = bool(
        audit["baseline_reproduction"].get("pass", False)
        and audit["transition_coverage"]["pass"]
        and audit["saturation_coverage"]["pass"]
    )
    return audit


def _make_plot(
    *,
    out_dir: str,
    per_k_hyd_records: List[Dict[str, Any]],
    gamma_max_nondim: float,
    v_kin: float,
) -> None:
    """6-panel summary at λ=1.0 across the converged k_hyd subset (plan §5)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                                       # pragma: no cover
        print("matplotlib not available; skipping plot", flush=True)
        return

    rows: List[Tuple[float, Dict[str, Any]]] = []
    for rec in per_k_hyd_records:
        lam1 = lambda1_record(rec)
        if lam1 is None or not lam1.get("snes_converged", False):
            continue
        rows.append((float(rec["k_hyd_target"]), lam1))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print("No converged λ=1 rungs; skipping plot", flush=True)
        return

    kh_vals = [r[0] for r in rows]
    gamma_vals = [r[1].get("gamma") for r in rows]
    theta_vals = [r[1].get("theta") for r in rows]
    rnet_vals = [
        (r[1].get("k_des") or 1.0) * (r[1].get("gamma") or 0.0)
        for r in rows
    ]
    sel_vals = [r[1].get("H2O2_selectivity_pct") for r in rows]
    o2_vals = [r[1].get("o2_flux_levich_ratio") for r in rows]
    amp_singh_vals = []
    for _, lam1 in rows:
        fd_decomp = lam1.get("F0_decomposition") or {}
        amp_singh_vals.append(fd_decomp.get("amplification_from_singh"))

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axes
    fig.suptitle(
        f"Phase 6β v10a Phase A.2 — k_hyd ramp at V_kin = {v_kin:+.3f} V "
        f"(λ=1.0 rungs)",
        fontsize=11,
    )

    def _safe(vals):
        return [v for v in vals if v is not None]

    def _xy(xs, ys):
        return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None])

    # A. Γ(k_hyd) with Γ_max ref line
    pairs = list(zip(*[(x, y) for x, y in zip(kh_vals, gamma_vals) if y is not None]))
    if pairs:
        ax_a.semilogx(pairs[0], pairs[1], "o-")
    ax_a.axhline(gamma_max_nondim, color="red", ls="--", lw=0.8,
                 label=f"Γ_max={gamma_max_nondim:.4f}")
    ax_a.set_ylabel("Γ (nondim)")
    ax_a.set_title("(A) Γ(k_hyd)")
    ax_a.legend(loc="best", fontsize=8)

    # B. θ(k_hyd)
    pairs = list(zip(*[(x, y) for x, y in zip(kh_vals, theta_vals) if y is not None]))
    if pairs:
        ax_b.semilogx(pairs[0], pairs[1], "s-")
    ax_b.axhline(KHYD_ROUTE_THETA_MIN, color="red", ls="--", lw=0.8,
                 label="route eligibility (0.95)")
    ax_b.axhline(0.9, color="gray", ls=":", lw=0.8,
                 label="cap coverage (0.9)")
    ax_b.set_ylabel("θ = Γ/Γ_max")
    ax_b.set_title("(B) θ(k_hyd)")
    ax_b.legend(loc="best", fontsize=8)

    # C. R_net(k_hyd) log-log with k_des*Γ_max ref
    pairs = list(zip(*[(x, y) for x, y in zip(kh_vals, rnet_vals) if y is not None]))
    if pairs:
        ax_c.loglog(pairs[0], pairs[1], "^-")
    ax_c.axhline(gamma_max_nondim, color="red", ls="--", lw=0.8,
                 label=f"k_des·Γ_max={gamma_max_nondim:.4f}")
    ax_c.set_ylabel("R_net = k_des·Γ (nondim)")
    ax_c.set_title("(C) R_net(k_hyd)")
    ax_c.legend(loc="best", fontsize=8)

    # D. H2O2 selectivity with deck band shaded
    pairs = list(zip(*[(x, y) for x, y in zip(kh_vals, sel_vals) if y is not None]))
    if pairs:
        ax_d.semilogx(pairs[0], pairs[1], "D-")
    lo, hi = DECK_SELECTIVITY_BAND
    ax_d.axhspan(lo, hi, color="green", alpha=0.15,
                 label=f"deck band [{lo}, {hi}]%")
    ax_d.set_ylabel("H2O2 selectivity (%)")
    ax_d.set_title("(D) selectivity_pct(k_hyd)")
    ax_d.legend(loc="best", fontsize=8)

    # E. O2 Levich ratio with transport gate
    pairs = list(zip(*[(x, y) for x, y in zip(kh_vals, o2_vals) if y is not None]))
    if pairs:
        ax_e.semilogx(pairs[0], pairs[1], "v-")
    ax_e.axhline(TRANSPORT_O2_MAX, color="red", ls="--", lw=0.8,
                 label="transport gate (0.9)")
    ax_e.set_ylabel("o2_flux_levich_ratio")
    ax_e.set_title("(E) O2 Levich(k_hyd)")
    ax_e.set_xlabel("k_hyd (nondim)")
    ax_e.legend(loc="best", fontsize=8)

    # F. amp_from_singh with log-y, 1.0 / 10 ref lines
    pairs = list(zip(*[(x, y) for x, y in zip(kh_vals, amp_singh_vals) if y is not None]))
    if pairs:
        ax_f.loglog(pairs[0], pairs[1], "*-")
    ax_f.axhline(1.0, color="gray", ls=":", lw=0.8, label="amp=1.0")
    ax_f.axhline(SINGH_AMP_RECALIBRATION_THRESHOLD, color="red", ls="--",
                 lw=0.8, label="recalibration threshold (10)")
    ax_f.set_ylabel("amplification_from_singh")
    ax_f.set_title("(F) amp_from_singh(k_hyd)")
    ax_f.set_xlabel("k_hyd (nondim)")
    ax_f.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path = os.path.join(out_dir, "phase_a2_v_kin.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    k_hyd_grid = _parse_k_hyd_grid(args.k_hyd_grid)
    v_kin = float(args.v_kin)
    v_anchor = float(args.v_anchor)
    k0_r4e_factor = float(args.k0_r4e_factor)

    out_dir = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    print(f"Phase A.2 — k_hyd × λ at V_kin={v_kin:+.3f} V, "
          f"K0_R4e_factor={k0_r4e_factor:.3g}", flush=True)
    print(f"  k_hyd grid ({len(k_hyd_grid)} points): {k_hyd_grid}", flush=True)
    print(f"  Output: {out_dir}", flush=True)

    # Lazy imports — keep test imports free of Firedrake.
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        L_EFF_M_BASELINE, STERN_F_M2_BASELINE, STERN_F_M2_ANCHOR,
        SMOKE_KINETICS, MESH_NX, MESH_NY, MESH_BETA, LAMBDA_LADDER,
        K0_INITIAL_SCALES, _build_sp, _make_mesh, _serialize,
        _i_lim_4e_mA_cm2, _walk_lambda_zero_capture_snapshots,
    )
    from scripts._bv_common import (
        I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T,
    )

    # Make sure V_kin is in the warm-walk grid (R3 #1: capture λ=0
    # baseline at V_kin once via the warm-walk's per-point callback).
    warm_grid = list(WARM_WALK_GRID)
    if not any(abs(v - v_kin) < 1e-9 for v in warm_grid):
        warm_grid.append(v_kin)
        warm_grid = sorted(warm_grid, reverse=True)
    warm_grid_t = tuple(warm_grid)

    t_start = time.time()

    # Build template SP (warm-walk and ramp share this) — Stern at
    # PRODUCTION target.  The anchor build inside
    # _walk_lambda_zero_capture_snapshots transparently switches to
    # STERN_F_M2_ANCHOR for the k0 + Kw_eff ladder.
    sp = _build_sp(
        lambda_hydrolysis=0.0,
        k0_r4e_factor=k0_r4e_factor,
        # All other kinetics at SMOKE_KINETICS defaults.
    )
    mesh = _make_mesh(l_eff_m=L_EFF_M_BASELINE)

    # Pass 1 — anchor + warm-walk to capture U snapshot at V_kin.
    print(f"Pass 1: anchor at V={v_anchor:+.3f} V → warm-walk {warm_grid_t}",
          flush=True)
    (warm_records, snapshots, mesh_dof_count,
     electrode_area_nondim, electrode_marker) = (
        _walk_lambda_zero_capture_snapshots(
            sp=sp, mesh=mesh,
            v_rhe_grid=warm_grid_t,
            v_anchor=v_anchor,
            k0_r4e_factor=k0_r4e_factor,
        )
    )

    # Locate V_kin index in the warm grid (now-sorted by helper order).
    v_kin_idx: Optional[int] = None
    for idx, rec in enumerate(warm_records):
        if abs(float(rec["v_rhe"]) - v_kin) < 1e-9:
            v_kin_idx = idx
            break
    if v_kin_idx is None or v_kin_idx not in snapshots:
        print(
            f"ERROR: warm-walk did not converge at V_kin={v_kin:+.3f} V",
            flush=True,
        )
        return 2
    U_at_v_kin = snapshots[v_kin_idx]
    print(f"  warm-walk produced U snapshot at V_kin={v_kin:+.3f} V "
          f"(idx={v_kin_idx} in warm grid)", flush=True)

    lambda_zero_baseline = _extract_lambda_zero_baseline(
        warm_walk_records=warm_records, v_kin=v_kin,
    )

    # Pass 2 — for each k_hyd, run the λ ladder at V_kin via callback.
    domain_height_hat = L_EFF_M_BASELINE / 1.0e-4
    i_lim_4e = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)
    print(f"  electrode_area_nondim={electrode_area_nondim:.6g}, "
          f"domain_height_hat={domain_height_hat:.6g}, "
          f"i_lim_4e={i_lim_4e:.4g} mA/cm²", flush=True)

    per_k_hyd_records: List[Dict[str, Any]] = []
    for i, k_hyd in enumerate(k_hyd_grid):
        t0 = time.time()
        print(f"Pass 2.{i+1}/{len(k_hyd_grid)}: k_hyd={k_hyd:.3e}",
              flush=True)
        rec = _run_k_hyd_ramp(
            sp_template=sp, mesh=mesh, voltage=v_kin,
            U_warmstart=U_at_v_kin, k_hyd_target=float(k_hyd),
            k0_r4e_factor=k0_r4e_factor,
            i_scale=float(I_SCALE),
            i_lim_4e_mA_cm2=i_lim_4e,
            electrode_area_nondim=electrode_area_nondim,
            domain_height_hat=domain_height_hat,
        )
        lam1 = lambda1_record(rec)
        summary = (
            f"  → λ=1: γ={lam1.get('gamma'):.4g}, "
            f"θ={lam1.get('theta'):.4g}, "
            f"picard={lam1.get('picard_status')}, "
            f"mb={lam1.get('mass_balance_residual_rel')}"
            if lam1 else
            f"  → no λ=1 rung (exception_phase={rec['exception_phase']})"
        )
        print(summary, flush=True)
        print(f"  wall: {time.time() - t0:.1f}s", flush=True)
        per_k_hyd_records.append(rec)

    # Optional: ±C_S perturbation at k_hyd_route only.
    if args.with_perturbation:
        route = select_k_hyd_route(per_k_hyd_records)
        if route is not None:
            print(f"Pass 3: ±C_S perturbation at k_hyd_route={route:.3e}",
                  flush=True)
            cs_lo = STERN_F_M2_BASELINE * (1.0 - 0.05)
            cs_hi = STERN_F_M2_BASELINE * (1.0 + 0.05)
            t0 = time.time()
            rec_lo = _run_k_hyd_ramp(
                sp_template=sp, mesh=mesh, voltage=v_kin,
                U_warmstart=U_at_v_kin, k_hyd_target=float(route),
                k0_r4e_factor=k0_r4e_factor,
                i_scale=float(I_SCALE),
                i_lim_4e_mA_cm2=i_lim_4e,
                electrode_area_nondim=electrode_area_nondim,
                domain_height_hat=domain_height_hat,
                extra_overrides={"stern_capacitance_f_m2": cs_lo},
            )
            rec_hi = _run_k_hyd_ramp(
                sp_template=sp, mesh=mesh, voltage=v_kin,
                U_warmstart=U_at_v_kin, k_hyd_target=float(route),
                k0_r4e_factor=k0_r4e_factor,
                i_scale=float(I_SCALE),
                i_lim_4e_mA_cm2=i_lim_4e,
                electrode_area_nondim=electrode_area_nondim,
                domain_height_hat=domain_height_hat,
                extra_overrides={"stern_capacitance_f_m2": cs_hi},
            )
            print(f"  perturbation wall: {time.time() - t0:.1f}s", flush=True)
            for rec in per_k_hyd_records:
                if abs(float(rec["k_hyd_target"]) - float(route)) <= 1e-30:
                    rec["perturbation_low_C_s"] = rec_lo
                    rec["perturbation_high_C_s"] = rec_hi
                    rec["perturbation_C_s_low"] = cs_lo
                    rec["perturbation_C_s_high"] = cs_hi
                    break

    # Build v10b_priorities + convergence audit.
    priorities = build_v10b_priorities_block(per_k_hyd_records)
    audit = _convergence_audit(per_k_hyd_records, lambda_zero_baseline)

    # Compose final JSON payload.
    config: Dict[str, Any] = {
        "v_kin": v_kin,
        "v_anchor": v_anchor,
        "k0_r4e_factor": k0_r4e_factor,
        "stern_capacitance_f_m2": STERN_F_M2_BASELINE,
        "stern_capacitance_f_m2_anchor_build": STERN_F_M2_ANCHOR,
        "K0_HAT_R2E_baseline": float(K0_HAT_R2E),
        "K0_HAT_R4E_baseline": float(K0_HAT_R4E),
        "K0_HAT_R4E_effective": float(K0_HAT_R4E) * k0_r4e_factor,
        "k_hyd_grid": list(k_hyd_grid),
        "transition_grid": list(TRANSITION_GRID),
        "saturation_grid": list(SATURATION_GRID),
        "warm_walk_grid": list(warm_grid_t),
        "lambda_ladder": list(LAMBDA_LADDER),
        "smoke_kinetics": SMOKE_KINETICS,
        "l_eff_m": L_EFF_M_BASELINE,
        "domain_height_hat": domain_height_hat,
        "electrode_area_nondim": electrode_area_nondim,
        "electrode_marker": electrode_marker,
        "mesh": {
            "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
            "domain_height_hat": domain_height_hat,
        },
        "mesh_dof_count": mesh_dof_count,
        "i_lim_4e_mA_cm2": i_lim_4e,
        "with_perturbation": bool(args.with_perturbation),
        "wall_seconds": float(time.time() - t_start),
    }
    payload: Dict[str, Any] = {
        "config": config,
        "lambda_zero_baseline_at_v_kin": lambda_zero_baseline,
        "warm_walk_records": warm_records,
        "per_k_hyd_records": per_k_hyd_records,
        "v10b_priorities": priorities,
        "convergence_audit": audit,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "phase_a2_v_kin.json")
    with open(json_path, "w") as f:
        json.dump(_serialize(payload), f, indent=2, sort_keys=False)
    print(f"Wrote {json_path}", flush=True)

    if args.plot:
        _make_plot(
            out_dir=out_dir,
            per_k_hyd_records=per_k_hyd_records,
            gamma_max_nondim=float(SMOKE_KINETICS["gamma_max_nondim"]),
            v_kin=v_kin,
        )

    # Print summary.
    print("\n=== Phase A.2 summary ===", flush=True)
    print(f"  k_hyd_route: {priorities['k_hyd_route']}", flush=True)
    if priorities["k_hyd_route"] is not None:
        print(f"  selectivity_gap_pp (single-V): "
              f"{priorities['single_v_selectivity_gap_pp']:+.2f} pp "
              f"→ kdes/Γ_max priority = {priorities['kdes_gammamax_priority']}",
              flush=True)
    else:
        print(f"  no_route_cause: {priorities['no_route_cause']}", flush=True)
    print(f"  max_amp_from_singh: {priorities['max_amp_from_singh']}", flush=True)
    print(f"  rHEl_recalibration_required: "
          f"{priorities['rHEl_recalibration_required']}", flush=True)
    print(f"  transport_re_entry_first_k_hyd: "
          f"{priorities['transport_re_entry_first_k_hyd']}", flush=True)
    print(f"  convergence_audit.overall_pass: {audit['overall_pass']}",
          flush=True)
    print(f"  Total wall: {time.time() - t_start:.1f}s", flush=True)

    # Exit code: 0 iff k_hyd_route exists; 1 otherwise.  Convergence-audit
    # failure DOES NOT change the exit code on its own — the caller wants
    # the JSON to inspect either way.
    return 0 if priorities["k_hyd_route"] is not None else 1


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(main())
