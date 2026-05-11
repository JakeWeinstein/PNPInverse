"""Phase 6β v10a — minimum V-sweep diagnostic + V_kin selection.

Walks V_RHE at smoke kinetics, runs two passes per V (λ=0 baseline +
λ=1 with Langmuir cap active), records the v10a rung diagnostics
plus a Stern-capacitance perturbation column, and emits a V_kin
candidate list filtered + scored per the **locked acceptance-bundle
rule** (docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md §
step 4).

This is steps 3 + 4 of the sequenced re-do plan after v10a landed.
Step 5 (Phase A.2 at V_kin) is a separate driver run after this one
returns a v_kin (or routes to v10c if the diagnostic aborts).

Hardened by GPT critique session 32 (3 rounds, APPROVED).  See
`docs/handoffs/CHATGPT_HANDOFF_32_phase6b-v10a-v-sweep-diagnostic/`
for the full provenance.  Key clarifications baked in:

* The "sensitivity" column is the **Stern-capacitance-manifold total
  derivative**, NOT a partial `∂R_net/∂σ_S`.  Renamed accordingly.
* FD-vs-perturbation disagreement is **never** used as a filter.
  It is logged as `path_mismatch_relative` informationally.
* Numerical-quality gating (Stage 1) is upstream of the locked rule
  (Stage 2); the three locked filters are applied verbatim.
* `|cd|/I_lim_4e` is asymmetric for parallel 2e/4e cathodes (pure 2e
  plateau caps at 0.5).  The locked rule is implemented literally;
  `o2_flux_levich_ratio` is an informational supplementary indicator.

Outputs (in `StudyResults/phase6b_v10a_v_sweep_diagnostic/`):

* ``iv_diagnostic.json`` — full per-V records + config + decision.
* ``iv_diagnostic.png`` — three subpanels: σ_S(V), R_net(V),
  |sensitivity|(V) with V_kin marked.  Optional (matplotlib).

Usage::

    python -u scripts/studies/phase6b_v10a_v_sweep_diagnostic.py [--quick]
    python -u scripts/studies/phase6b_v10a_v_sweep_diagnostic.py \
        --v-rhe-grid 0.55,0.10,-0.10,-0.30,-0.50 \
        --no-perturbation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level path setup — keep Firedrake imports lazy (inside functions)
# so the V_kin selection helpers are importable without the venv.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Constants — locked / smoke-baseline values
# ---------------------------------------------------------------------------

V_ANCHOR_DEFAULT: float = 0.55
"""Anodic anchor voltage (V vs RHE).  Known-converging from gate2/gate4."""

V_RHE_GRID_DEFAULT: Tuple[float, ...] = (
    +0.55, +0.40, +0.25, +0.10,
    +0.05,  0.00, -0.05, -0.10, -0.15, -0.20,
    -0.25, -0.30, -0.35, -0.40, -0.45, -0.50,
)
"""Production V_RHE grid: fine cathodic + sparse anodic."""

V_RHE_GRID_QUICK: Tuple[float, ...] = (
    +0.55, +0.10, -0.10, -0.30, -0.50,
)
"""Smoke / CLI ``--quick`` grid (~5-10 min wall)."""

LAMBDA_LADDER: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
"""λ ramp per voltage."""

K0_INITIAL_SCALES: Tuple[float, ...] = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
"""k0 continuation ladder."""

STERN_F_M2_BASELINE: float = 0.20
"""Production Stern capacitance, F/m² (= 20 µF/cm²).

Per ``.research/cmk3-stern-capacitance/SUMMARY.md`` (2026-05-10):
literature-anchored Bohra-Koper-Choi consensus.  Derived from Stern
thickness ``L_S = 5 Å`` and Booth-saturated permittivity
``ε_S = 11.3``.  Citation chain: Bohra et al. 2024 *JPC C*
(PMC11215773), Choi et al. 2024 (`10.1021/acs.jpcc.4c03469`),
Pillai et al. 2024 (`10.1021/acs.jpcc.3c05364`), CatINT default
(Stanford Bell group), Kilic-Bazant 2007 *Phys Rev E* 75:021503
(foundational mPNP-Stern).  The legacy ``0.10 F/m²`` value
(pre-2026-05-10) was a convergence-pinned engineering choice with
no CMK-3 anchor; it sits at the bottom of Pillai's "safe band"
(10–50 µF/cm²) and is defensible as carbon-conservative, but
``0.20 F/m²`` is the production target going forward.
"""

STERN_F_M2_ANCHOR: float = 0.10
"""Convergence-pinned Stern capacitance for the Pass 1 anchor build, F/m².

The anchor's ``solve_anchor_with_continuation`` k0 + Kw_eff ladders
were tuned at the legacy ``C_S = 0.10 F/m²`` and *do not converge*
at the production target ``0.20 F/m²`` (verified empirically
2026-05-10 — the k0 ladder exhausts at Kw_eff=0 floor with both
``linear_phi`` and ``debye_boltzmann`` initialisers).  The Stern
``c_s_ladder`` helper exists in the solver but raises
``NotImplementedError`` when combined with ``kw_eff_ladder``.

To stay on plan (production target ``C_S = 0.20``) without modifying
the production solver, the v10a' driver runs a TWO-STAGE anchor:

  1. Build the anchor at ``C_S = STERN_F_M2_ANCHOR = 0.10`` (this
     value matches the v9 / v10a baseline that converged).  Run the
     full Kw_eff ladder + k0 ladder.
  2. Once the anchor reaches production ``k0`` at full ``Kw_eff``,
     bump ``C_S`` to ``STERN_F_M2_BASELINE = 0.20`` via
     :func:`set_stern_capacitance_model` and re-solve Newton.  The
     warm-start from the C_S=0.10 converged state lets Newton
     absorb the Stern bump in a single solve at the same anchor V.

The post-bump U snapshot then seeds the warm-walk + Pass 2 lambda
ramp at ``C_S = 0.20``.  This preserves the plan's intent
(production diagnostic at C_S=0.20) without changing the production
solver's ladder-combination invariants.
"""

L_EFF_M_BASELINE: float = 16e-6
"""Production boundary-layer thickness, m."""

PERTURB_FRACTION: float = 0.05
"""ε in C_S → C_S · (1 ± ε) for the perturbation column.

Per critique session 32 R3 nit #1, the log denominator uses the
exact form ``log(1+ε) - log(1-ε)``, not the ``2ε`` first-order
approximation.
"""

# Numerical-quality floors (Stage 1 — estimator validity).  All in
# *physical* C/m² units so a reader can sanity-check them against
# Stern surface charge magnitudes.

SIGMA_ABS_MIN_C_PER_M2: float = 1e-4
"""Absolute floor on |σ_+ - σ_-| (physical C/m²) per critique session
32 R2 issue #1.  ~0.5% of typical Stern surface charge magnitude at
cathodic V on this stack; well above Newton residual noise at SNES
rtol=1e-10.  If no V clears it, the estimator is unidentifiable
across the whole sweep — driver emits
``no_valid_stern_capacitance_sensitivity``.
"""

SENSITIVITY_FLOOR_ABS: float = 1e-12
"""Absolute floor on slope magnitudes (derivative units) per critique
session 32 R2 issue #3.  Below this, the slope is dominated by
floating-point noise and the estimator is unreliable.
"""

SENSITIVITY_FLOOR_ADAPTIVE_FRACTION: float = 1e-3
"""Adaptive component of ``sensitivity_floor``: ``max(|S+|, |S-|) · k``.

Tracks the local scale so a V with a genuinely huge slope isn't
flagged just because FD has a coarser response.
"""

# Smoke kinetics — locked baseline per the acceptance bundle.

SMOKE_KINETICS: Dict[str, float] = {
    "k_hyd_nondim":     1e-3,
    "k_prot_nondim":    1e-3,
    "k_des_nondim":     1.0,
    "delta_ohp_hat":    4e-6,                      # 0.40 nm / 100 µm
    "gamma_max_nondim": 0.047,                     # 1 monolayer MOH
    "r_H_El_pm":        200.98,                    # Singh Cu prior
}

# Mesh / solver constants.

MESH_NX:     int = 8
MESH_NY:     int = 80
MESH_BETA:   float = 3.0
U_CLAMP:     float = 100.0
EXPONENT_CLIP: float = 100.0

OUT_SUBDIR_DEFAULT: str = "phase6b_v10a_v_sweep_diagnostic"


# ---------------------------------------------------------------------------
# Levich limit (R1 issue #4 — imports from _bv_common to fix D_O2 typo)
# ---------------------------------------------------------------------------


def _i_lim_4e_mA_cm2(l_eff_m: float) -> float:
    """4-electron Levich limit, mA/cm².

    ``I_lim_4e = 4 · F · D_O2 · c_O2 / l_eff_m`` [A/m²], times 0.1 →
    mA/cm².  Imports ``D_O2 = 1.9e-9 m²/s`` and ``C_O2 = 1.2 mol/m³``
    from :mod:`scripts._bv_common` to stay in lockstep with
    ``I_SCALE`` and the rest of the stack.

    At ``l_eff = 16 µm``: ``I_lim_4e ≈ 5.50 mA/cm²``.

    NOTE on parallel-2e/4e asymmetry (critique session 32 R1 issue #5):
    at the O₂-flux Levich limit,
    ``|cd|/I_lim_4e = (1 + x_4e)/2`` where ``x_4e = R_4e/(R_2e+R_4e)``.
    Pure 2e plateau caps at 0.5; only pure-4e plateaus reach 1.0.
    The driver implements the locked filter ``|cd|/I_lim_4e < 0.9``
    literally; ``o2_flux_levich_ratio`` is the branch-selectivity-
    independent supplementary indicator.
    """
    from scripts._bv_common import D_O2, C_O2

    F_CONST: float = 96485.3329
    return 4.0 * F_CONST * D_O2 * C_O2 / l_eff_m * 0.1


def _compute_o2_flux_levich_ratio(
    *,
    R_2e_current_nondim: float,
    R_4e_current_nondim: float,
    electrode_area_nondim: float,
    domain_height_hat: float,
) -> float:
    """Branch-selectivity-independent O₂-transport indicator (R2 issue #5).

    Numerator: per-area average boundary O₂ consumption rate
        ``(R_2e + R_4e) / electrode_area_nondim``
        (both R_j_current_nondim values are boundary integrals from
        :func:`collect_v10a_rung_diagnostics`; dividing by the
        electrode area gives the per-area mean).

    Denominator: bulk Levich O₂ flux (per area)
        ``D_O2_HAT · C_O2_HAT / domain_height_hat``.
        With ``D_O2_HAT = C_O2_HAT = 1`` by convention, this reduces
        to ``1 / domain_height_hat``.

    Returns a non-negative ratio in ``[0, 1]`` at the transport limit
    regardless of R_2e/R_4e split.  ``abs()`` keeps the sign positive
    regardless of which sign convention the diagnostics collector
    emits for the per-reaction currents.
    """
    if electrode_area_nondim <= 0.0 or domain_height_hat <= 0.0:
        return 0.0
    # Local import keeps the helper standalone for tests.
    try:
        from scripts._bv_common import D_O2_HAT, C_O2_HAT
    except ImportError:                                # pragma: no cover
        D_O2_HAT, C_O2_HAT = 1.0, 1.0

    o2_consumption_nondim = R_2e_current_nondim + R_4e_current_nondim
    o2_consumption_per_area = o2_consumption_nondim / electrode_area_nondim
    levich_flux = D_O2_HAT * C_O2_HAT / domain_height_hat
    if levich_flux <= 0.0:
        return 0.0
    return abs(o2_consumption_per_area) / levich_flux


# ---------------------------------------------------------------------------
# V_kin selection — Stage 1 (estimator validity) + Stage 2 (LOCKED rule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VKinDecision:
    """Outcome of :func:`select_v_kin`.

    Exactly one of ``v_kin`` is set OR one of the three failure flags
    is True.

    Attributes
    ----------
    v_kin
        Selected voltage (V vs RHE) or ``None`` on any failure path.
    score
        ``|dRnet_dsigma_along_stern_capacitance|`` at v_kin, or ``None``.
    abort_to_v10c
        No V has ``σ_S < 0`` — physics fail-stop per the locked rule.
        Routes to v10c (C_S bracket sweep).
    no_valid_stern_capacitance_sensitivity
        No V passed the Stage 1 estimator-validity gate at the
        fallback tier.  Retry with a different perturbation knob
        or larger ε.  Not triggered when ``abort_to_v10c`` already
        applies (precedence per critique session 32 R3 nit #3).
    no_candidate_passed_locked_rule
        Some V have σ_S < 0 and valid estimators, but none cleared
        the locked three-filter set (even after dropping the branch
        filter in fallback).
    fallback_used
        True iff the branch filter was dropped and the fallback
        quality tier (one_sided_disagreement ≤ 0.5) was used to
        find a candidate.
    per_v_decisions
        Per-V dict of every filter/quality flag the rule examined.
        Useful for the JSON record + downstream auditing.
    """

    v_kin: Optional[float]
    score: Optional[float]
    abort_to_v10c: bool = False
    no_valid_stern_capacitance_sensitivity: bool = False
    no_candidate_passed_locked_rule: bool = False
    fallback_used: bool = False
    per_v_decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return asdict(self)


def _safe_div(num: float, den: float) -> Optional[float]:
    """Float div with zero-guard, returns ``None`` on zero/NaN."""
    if den == 0.0 or not math.isfinite(num) or not math.isfinite(den):
        return None
    return num / den


def _one_sided_slope(
    R_perturbed: float, R_unperturbed: float,
    sigma_perturbed: float, sigma_unperturbed: float,
) -> Optional[float]:
    """``ΔR_net / Δσ_S`` for a single (perturbed, unperturbed) pair.

    Returns ``None`` if the σ_S difference is below numerical
    precision — caller decides what to do (typically flag as
    estimator-invalid).
    """
    d_sigma = sigma_perturbed - sigma_unperturbed
    d_R = R_perturbed - R_unperturbed
    if abs(d_sigma) < 1e-300:
        return None
    return d_R / d_sigma


def _compute_perturbation_quantities(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill in the derived perturbation quantities (slopes, gaps).

    Mutates ``record`` in-place AND returns it.  Idempotent: safe to
    call repeatedly.
    """
    sigma_0 = record.get("sigma_S_unperturbed")
    sigma_lo = record.get("sigma_S_low")
    sigma_hi = record.get("sigma_S_high")
    R_0 = record.get("R_net_unperturbed")
    R_lo = record.get("R_net_low")
    R_hi = record.get("R_net_high")

    record["S_minus"] = (
        _one_sided_slope(R_lo, R_0, sigma_lo, sigma_0)
        if None not in (sigma_0, sigma_lo, R_0, R_lo)
        else None
    )
    record["S_plus"] = (
        _one_sided_slope(R_hi, R_0, sigma_hi, sigma_0)
        if None not in (sigma_0, sigma_hi, R_0, R_hi)
        else None
    )

    # Central difference (slope and ratio).
    if (
        sigma_lo is not None and sigma_hi is not None
        and R_lo is not None and R_hi is not None
    ):
        d_sigma_central = sigma_hi - sigma_lo
        d_R_central = R_hi - R_lo
        record["sigma_S_delta_two_sided"] = d_sigma_central
        record["R_net_delta_two_sided"] = d_R_central
        record["dRnet_dsigma_along_stern_capacitance"] = (
            _safe_div(d_R_central, d_sigma_central)
        )
    else:
        record["sigma_S_delta_two_sided"] = None
        record["R_net_delta_two_sided"] = None
        record["dRnet_dsigma_along_stern_capacitance"] = None

    # Per-side denominators (R2 issue #2: need to check these
    # individually, not just the two-sided gap).
    record["sigma_S_delta_one_sided_minus"] = (
        sigma_lo - sigma_0 if (sigma_lo is not None and sigma_0 is not None)
        else None
    )
    record["sigma_S_delta_one_sided_plus"] = (
        sigma_hi - sigma_0 if (sigma_hi is not None and sigma_0 is not None)
        else None
    )

    # Log-step denominator (R3 nit #1: use exact form, not 2ε).
    cs_0 = record.get("C_s_unperturbed")
    cs_lo = record.get("C_s_low")
    cs_hi = record.get("C_s_high")
    if (
        cs_0 is not None and cs_lo is not None and cs_hi is not None
        and cs_0 > 0.0 and cs_lo > 0.0 and cs_hi > 0.0
    ):
        log_step = math.log(cs_hi / cs_0) - math.log(cs_lo / cs_0)
        record["log_step_denominator"] = log_step
        if (
            log_step > 0.0
            and record.get("R_net_delta_two_sided") is not None
            and record.get("sigma_S_delta_two_sided") is not None
        ):
            record["dRnet_dlogCs"] = record["R_net_delta_two_sided"] / log_step
            record["dsigma_dlogCs"] = record["sigma_S_delta_two_sided"] / log_step
        else:
            record["dRnet_dlogCs"] = None
            record["dsigma_dlogCs"] = None
    else:
        record["log_step_denominator"] = None
        record["dRnet_dlogCs"] = None
        record["dsigma_dlogCs"] = None

    return record


def _sensitivity_floor(record: Dict[str, Any]) -> float:
    """Derivative-units floor (R2 issue #3 + R3 nit accept).

    ``max(SENSITIVITY_FLOOR_ABS, k · max(|S_+|, |S_-|))``.
    Always positive; never reuses dimensionless ε.
    """
    s_plus = record.get("S_plus")
    s_minus = record.get("S_minus")
    candidates = [SENSITIVITY_FLOOR_ABS]
    if s_plus is not None:
        candidates.append(abs(s_plus) * SENSITIVITY_FLOOR_ADAPTIVE_FRACTION)
    if s_minus is not None:
        candidates.append(abs(s_minus) * SENSITIVITY_FLOOR_ADAPTIVE_FRACTION)
    return max(candidates)


def _compute_locked_filter_flags(
    record: Dict[str, Any], *, i_lim_4e_mA_cm2: float,
) -> Dict[str, Any]:
    """Compute the three explicit locked-filter booleans (R2 issue #6).

    Each filter named precisely so downstream readers don't confuse
    "current filter only" with "all three filters".
    """
    sigma_S = record.get("sigma_S_C_per_m2")
    cd_mA = record.get("cd_mA_cm2")
    r2e = record.get("R_2e_current_nondim")
    r4e = record.get("R_4e_current_nondim")

    sigma_neg = (sigma_S is not None) and (sigma_S < 0.0)
    if cd_mA is not None and i_lim_4e_mA_cm2 > 0.0:
        current_passed = (abs(cd_mA) / i_lim_4e_mA_cm2) < 0.9
    else:
        current_passed = False
    if r2e is not None and r4e is not None:
        denom = r2e + r4e
        if abs(denom) > 1e-30:
            x_4e = r4e / denom
            x_2e = r2e / denom
            # Both branches active condition; equivalently x_4e ∈ [0.05, 0.95]
            branch_passed = (0.05 <= x_4e <= 0.95)
            # Also require both contributions positive (signed flux check).
            branch_passed = branch_passed and (x_2e >= 0.0 and x_4e >= 0.0)
        else:
            branch_passed = False
    else:
        branch_passed = False

    record["locked_sigma_neg_filter_passed"] = bool(sigma_neg)
    record["locked_current_filter_passed"] = bool(current_passed)
    record["locked_branch_filter_passed"] = bool(branch_passed)
    record["locked_three_filters_passed"] = bool(
        sigma_neg and current_passed and branch_passed
    )

    # Levich-asymmetry warning — ties to *current* filter only.
    o2_ratio = record.get("o2_flux_levich_ratio")
    if o2_ratio is not None and current_passed:
        record["locked_current_filter_passes_but_o2_transport_limited"] = (
            o2_ratio > 0.9
        )
    else:
        record["locked_current_filter_passes_but_o2_transport_limited"] = False

    return record


def _compute_estimator_validity(
    record: Dict[str, Any],
    *,
    sigma_abs_min: float,
    median_nonzero_delta_sigma: float,
) -> Dict[str, Any]:
    """Stage 1 estimator-validity flags (NOT part of the locked rule).

    Implements the GPT-approved validity criteria:
      * perturbation_converged
      * two-sided gap (adaptive floor)
      * per-side gap (adaptive floor, half of two-sided — R3 nit #2)
      * one-sided slope agreement (sensitivity_floor in derivative
        units)
    """
    perturbation_converged = bool(record.get("perturbation_converged", False))

    # Two-sided gap on |σ_+ - σ_-|.
    two_sided_threshold = max(sigma_abs_min, 0.1 * median_nonzero_delta_sigma)
    delta_two_sided = record.get("sigma_S_delta_two_sided")
    two_sided_gap_ok = (
        delta_two_sided is not None
        and abs(delta_two_sided) >= two_sided_threshold
    )

    # Per-side gap on |σ_± − σ_0| (R3 nit #2: inherits adaptive scale).
    per_side_threshold = 0.5 * two_sided_threshold
    delta_minus = record.get("sigma_S_delta_one_sided_minus")
    delta_plus = record.get("sigma_S_delta_one_sided_plus")
    per_side_gap_ok = (
        delta_minus is not None and delta_plus is not None
        and abs(delta_minus) >= per_side_threshold
        and abs(delta_plus) >= per_side_threshold
    )

    # One-sided slope agreement.
    s_plus = record.get("S_plus")
    s_minus = record.get("S_minus")
    if s_plus is not None and s_minus is not None:
        floor = _sensitivity_floor(record)
        denom = max(abs(s_plus), abs(s_minus), floor)
        one_sided_disagreement = abs(s_plus - s_minus) / denom
    else:
        one_sided_disagreement = None

    record["two_sided_gap_threshold"] = two_sided_threshold
    record["per_side_gap_threshold"] = per_side_threshold
    record["two_sided_gap_ok"] = bool(two_sided_gap_ok)
    record["per_side_gap_ok"] = bool(per_side_gap_ok)
    record["one_sided_disagreement"] = one_sided_disagreement
    record["sensitivity_floor"] = _sensitivity_floor(record)

    record["primary_valid"] = bool(
        perturbation_converged
        and two_sided_gap_ok
        and per_side_gap_ok
        and one_sided_disagreement is not None
        and one_sided_disagreement <= 0.25
    )
    record["fallback_valid"] = bool(
        perturbation_converged
        and two_sided_gap_ok
        and per_side_gap_ok
        and one_sided_disagreement is not None
        and one_sided_disagreement <= 0.50
    )

    # path_mismatch_relative — informational only (R1 issue #3).
    fd = record.get("dRnet_dsigma_along_voltage")
    perturb = record.get("dRnet_dsigma_along_stern_capacitance")
    if fd is not None and perturb is not None:
        floor = _sensitivity_floor(record)
        denom = max(abs(fd), abs(perturb), floor)
        record["path_mismatch_relative"] = abs(fd - perturb) / denom
    else:
        record["path_mismatch_relative"] = None

    return record


def select_v_kin(
    per_v_records: List[Dict[str, Any]],
    *,
    i_lim_4e_mA_cm2: float,
    electrode_area_nondim: Optional[float] = None,
    domain_height_hat: Optional[float] = None,
    sigma_abs_min: float = SIGMA_ABS_MIN_C_PER_M2,
) -> VKinDecision:
    """Pick V_kin per the locked acceptance-bundle rule.

    Two-stage flow (critique session 32 R2 issue #4):

      **Stage 1 — Estimator validity (NOT in the locked rule):**
        Filter to V where the Stern-capacitance-manifold derivative
        is numerically identifiable (gap floors, slope agreement).
        Two tiers: primary (one_sided_disagreement ≤ 0.25) and
        fallback (≤ 0.50).

      **Stage 2 — Locked physics rule on valid-estimator V:**
        ``argmax(|dRnet_dsigma_along_stern_capacitance|)`` subject to
        the three locked filters:
          - σ_S(V) < 0
          - |cd|/I_lim_4e < 0.9
          - R_2e/(R_2e + R_4e) ∈ [0.05, 0.95]
        Fallback: drop branch filter; relax to fallback-tier validity.

    Selection precedence (R3 nit #3 — most informative status wins):

      1. ``abort_to_v10c`` if no V has σ_S < 0.
      2. ``no_valid_stern_capacitance_sensitivity`` if no V passes
         fallback validity AND step 1 didn't already abort.
      3. Locked rule (primary, then fallback) on Stage 1 survivors.
      4. ``no_candidate_passed_locked_rule`` if some V are
         estimator-valid but none clear the locked filters.

    Parameters
    ----------
    per_v_records
        List of dicts produced by the driver's record-collection
        step.  Required keys per V (see module docstring): ``v_rhe``,
        ``sigma_S_C_per_m2``, ``cd_mA_cm2``, ``R_2e_current_nondim``,
        ``R_4e_current_nondim``, ``perturbation_converged``,
        ``sigma_S_unperturbed``, ``sigma_S_low``, ``sigma_S_high``,
        ``R_net_unperturbed``, ``R_net_low``, ``R_net_high``,
        ``C_s_unperturbed``, ``C_s_low``, ``C_s_high``.  Optional:
        ``dRnet_dsigma_along_voltage`` (FD column), ``o2_flux_levich_ratio``.
    i_lim_4e_mA_cm2
        4-electron Levich limit, mA/cm² (from :func:`_i_lim_4e_mA_cm2`).
    electrode_area_nondim, domain_height_hat
        Optional — only used when this function is responsible for
        computing ``o2_flux_levich_ratio`` (when the driver hasn't
        already done so).  Pass ``None`` to skip O₂-Levich
        re-computation.
    sigma_abs_min
        Absolute floor on σ-gap, physical C/m² (R2 issue #1).
        Default 1e-4.

    Returns
    -------
    VKinDecision
    """
    # Compute the median |σ_+ - σ_-| across V for the adaptive floor.
    nonzero_deltas: List[float] = []
    enriched = [dict(r) for r in per_v_records]
    for rec in enriched:
        _compute_perturbation_quantities(rec)
        d = rec.get("sigma_S_delta_two_sided")
        if d is not None and abs(d) > 0.0 and math.isfinite(d):
            nonzero_deltas.append(abs(d))
    if nonzero_deltas:
        median_nonzero = statistics.median(nonzero_deltas)
    else:
        median_nonzero = 0.0

    # Compute o2_flux_levich_ratio if asked (and not pre-populated).
    for rec in enriched:
        if (
            rec.get("o2_flux_levich_ratio") is None
            and electrode_area_nondim is not None
            and domain_height_hat is not None
            and rec.get("R_2e_current_nondim") is not None
            and rec.get("R_4e_current_nondim") is not None
        ):
            rec["o2_flux_levich_ratio"] = _compute_o2_flux_levich_ratio(
                R_2e_current_nondim=rec["R_2e_current_nondim"],
                R_4e_current_nondim=rec["R_4e_current_nondim"],
                electrode_area_nondim=electrode_area_nondim,
                domain_height_hat=domain_height_hat,
            )

    for rec in enriched:
        _compute_estimator_validity(
            rec,
            sigma_abs_min=sigma_abs_min,
            median_nonzero_delta_sigma=median_nonzero,
        )
        _compute_locked_filter_flags(rec, i_lim_4e_mA_cm2=i_lim_4e_mA_cm2)

    # ---- Stage 0 precedence: abort_to_v10c FIRST (R3 nit #3).
    any_sigma_neg = any(r["locked_sigma_neg_filter_passed"] for r in enriched)
    if not any_sigma_neg:
        return VKinDecision(
            v_kin=None, score=None,
            abort_to_v10c=True,
            per_v_decisions=enriched,
        )

    # ---- Stage 1: estimator validity.
    primary_valid = [r for r in enriched if r["primary_valid"]]
    fallback_valid = [r for r in enriched if r["fallback_valid"]]
    if not fallback_valid:
        # No V passes even the loose tier; report estimator failure.
        return VKinDecision(
            v_kin=None, score=None,
            no_valid_stern_capacitance_sensitivity=True,
            per_v_decisions=enriched,
        )

    # ---- Stage 2: locked rule on the primary-valid set.
    def _score(r: Dict[str, Any]) -> float:
        s = r.get("dRnet_dsigma_along_stern_capacitance")
        return abs(s) if (s is not None and math.isfinite(s)) else -math.inf

    primary_candidates = [
        r for r in primary_valid if r["locked_three_filters_passed"]
    ]
    if primary_candidates:
        best = max(primary_candidates, key=_score)
        return VKinDecision(
            v_kin=float(best["v_rhe"]),
            score=_score(best),
            fallback_used=False,
            per_v_decisions=enriched,
        )

    # Fallback: drop branch filter; relax to fallback-valid tier.
    fallback_candidates = [
        r for r in fallback_valid
        if r["locked_sigma_neg_filter_passed"]
        and r["locked_current_filter_passed"]
    ]
    if fallback_candidates:
        best = max(fallback_candidates, key=_score)
        return VKinDecision(
            v_kin=float(best["v_rhe"]),
            score=_score(best),
            fallback_used=True,
            per_v_decisions=enriched,
        )

    # No candidate passed the locked rule.
    return VKinDecision(
        v_kin=None, score=None,
        no_candidate_passed_locked_rule=True,
        per_v_decisions=enriched,
    )


# ---------------------------------------------------------------------------
# FD sensitivity post-pass (R1 issue #3: informational only)
# ---------------------------------------------------------------------------


def attach_fd_sensitivity(records: List[Dict[str, Any]]) -> None:
    """Compute the central-difference ``dRnet_dsigma_along_voltage`` per V.

    Mutates each record in-place.  Endpoints use forward / backward
    differences (less accurate but populated).  ``None`` whenever the
    σ_S step is degenerate.  **Logged informationally only**; never
    used as a filter (per critique session 32 R1 issue #3 + R2 issue #4).
    """
    n = len(records)
    if n < 2:
        for r in records:
            r["dRnet_dsigma_along_voltage"] = None
        return
    # Sort by V_RHE for consistent neighbour finding.
    order = sorted(range(n), key=lambda i: float(records[i].get("v_rhe", 0.0)))
    for k, idx in enumerate(order):
        rec = records[idx]
        sigma = rec.get("sigma_S_C_per_m2")
        r_net = rec.get("R_net_unperturbed")
        if sigma is None or r_net is None:
            rec["dRnet_dsigma_along_voltage"] = None
            continue
        if 0 < k < len(order) - 1:
            prev_rec = records[order[k - 1]]
            next_rec = records[order[k + 1]]
        elif k == 0 and len(order) >= 2:
            prev_rec = rec
            next_rec = records[order[k + 1]]
        else:
            prev_rec = records[order[k - 1]]
            next_rec = rec
        sigma_a = prev_rec.get("sigma_S_C_per_m2")
        sigma_b = next_rec.get("sigma_S_C_per_m2")
        r_a = prev_rec.get("R_net_unperturbed")
        r_b = next_rec.get("R_net_unperturbed")
        if None in (sigma_a, sigma_b, r_a, r_b):
            rec["dRnet_dsigma_along_voltage"] = None
            continue
        rec["dRnet_dsigma_along_voltage"] = _safe_div(r_b - r_a, sigma_b - sigma_a)


# ===========================================================================
# Below: Firedrake-using driver code (imports lazy inside functions).
# Tests should NOT need to import from below this line.
# ===========================================================================


# ---------------------------------------------------------------------------
# SolverParams + mesh builders (mirrors the gate4 driver pattern)
# ---------------------------------------------------------------------------


def _scale_k0_r4e_in_reactions(
    base_reactions: List[Dict[str, Any]], factor: float,
) -> List[Dict[str, Any]]:
    """Deep-copy ``base_reactions`` and rescale the 4e branch's ``k0``.

    Operates on the parallel-2e/4e layout where index 0 is R_2e and
    index 1 is R_4e (matches ``PARALLEL_2E_4E_REACTIONS_4SP`` from
    ``scripts._bv_common``).  Defensive deep-copy of the ``stoichiometry``
    list and ``cathodic_conc_factors`` (each entry a dict) keeps the
    module-level constant un-aliased so subsequent calls with a different
    factor see a fresh rescaling baseline.

    Per Plan §Implementation notes (`--k0-r4e-factor` wiring): the
    ``k0`` value of ``base_reactions[1]`` is replaced by
    ``base_reactions[1]['k0'] * factor``.  The 2e branch is untouched.

    ``make_bv_solver_params`` already enforces ``k0 > 0`` so a non-positive
    factor surfaces as a downstream validation error (we don't pre-check
    here to keep the error message in one place).
    """
    rescaled: List[Dict[str, Any]] = []
    for r in base_reactions:
        copy = dict(r)
        if "stoichiometry" in copy:
            copy["stoichiometry"] = list(copy["stoichiometry"])
        if "cathodic_conc_factors" in copy and copy["cathodic_conc_factors"]:
            copy["cathodic_conc_factors"] = [
                dict(f) for f in copy["cathodic_conc_factors"]
            ]
        rescaled.append(copy)
    rescaled[1]["k0"] = float(rescaled[1]["k0"]) * float(factor)
    return rescaled


def _build_sp(
    *,
    stern_capacitance_f_m2: float = STERN_F_M2_BASELINE,
    r_H_El_pm: float = SMOKE_KINETICS["r_H_El_pm"],
    k_des_nondim: float = SMOKE_KINETICS["k_des_nondim"],
    k_hyd_nondim: float = SMOKE_KINETICS["k_hyd_nondim"],
    k_prot_nondim: float = SMOKE_KINETICS["k_prot_nondim"],
    delta_ohp_hat: float = SMOKE_KINETICS["delta_ohp_hat"],
    gamma_max_nondim: float = SMOKE_KINETICS["gamma_max_nondim"],
    lambda_hydrolysis: float = 0.0,
    l_eff_m: float = L_EFF_M_BASELINE,
    k0_r4e_factor: float = 1.0,
):
    """Build SolverParams for the v10a stack at the given parameters.

    ``k0_r4e_factor`` is a dimensionless multiplier applied to the 4e
    branch's ``k0`` (default 1.0 preserves back-compat).  Used by the
    v10a' diagnostic to walk K0_R4e through the bracket
    ``{1e-10 … 1e-24}`` per the Plan's Case B/C path.  See project
    memory ``project_k0_r4e_ratio_regimes`` for the three-regime
    characterization of this knob.
    """
    from scripts._bv_common import (
        A_OH_HAT, D_OH_HAT, KW_HAT,
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        PARALLEL_2E_4E_REACTIONS_4SP,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
        make_cation_hydrolysis_config,
        setup_firedrake_env,
    )
    setup_firedrake_env()

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol":   1e-7,
        "snes_rtol":   1e-10,
        "snes_stol":   1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    cation_cfg = make_cation_hydrolysis_config(
        k_hyd=float(k_hyd_nondim),
        k_prot=float(k_prot_nondim),
        k_des=float(k_des_nondim),
        delta_ohp_hat=float(delta_ohp_hat),
        cation="K+",
        r_H_El_pm=float(r_H_El_pm),
        pka_shift_form="singh_2016_eq_4",
        gamma_max_nondim=float(gamma_max_nondim),
    )

    bv_reactions = _scale_k0_r4e_in_reactions(
        PARALLEL_2E_4E_REACTIONS_4SP, float(k0_r4e_factor),
    )

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        snes_opts=snes_opts,
        formulation="logc_muh",
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=bv_reactions,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=float(stern_capacitance_f_m2),
        initializer="linear_phi",
        l_eff_m=float(l_eff_m),
        enable_water_ionization=True,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
        enable_cation_hydrolysis=True,
        cation_hydrolysis_config=cation_cfg,
        lambda_hydrolysis=float(lambda_hydrolysis),
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _build_sp_at_cs(*, sp_template, stern_capacitance_f_m2: float):
    """Return a fresh SolverParams identical to ``sp_template`` but with
    a different Stern capacitance.

    Used by the two-stage anchor: build the anchor at
    ``STERN_F_M2_ANCHOR`` and the warm-walk + Pass 2 at
    ``STERN_F_M2_BASELINE``.  The override walks
    ``solver_options['bv_bc']['stern_capacitance_f_m2']`` — that's where
    ``make_bv_solver_params`` plumbs it (via :func:`_make_bv_bc_cfg`),
    and where :mod:`Forward.bv_solver.nondim` reads it from at
    form-build time to compute the nondim ``bv_stern_capacitance_model``.
    """
    new_opts = dict(sp_template.solver_options)
    new_bv_bc = dict(new_opts.get("bv_bc", {}))
    new_bv_bc["stern_capacitance_f_m2"] = float(stern_capacitance_f_m2)
    new_opts["bv_bc"] = new_bv_bc
    return sp_template.with_solver_options(new_opts)


def _make_mesh(*, l_eff_m: float = L_EFF_M_BASELINE):
    from Forward.bv_solver import make_graded_rectangle_mesh
    return make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=l_eff_m / 1.0e-4,
    )


def _build_kw_ladder():
    from scripts._bv_common import KW_HAT
    return (0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT)


# ---------------------------------------------------------------------------
# Pass 1 — λ=0 anchor + warm-walk, capture U snapshots per V
# ---------------------------------------------------------------------------


def _walk_lambda_zero_capture_snapshots(
    *, sp, mesh, v_rhe_grid: Tuple[float, ...], v_anchor: float,
    k0_r4e_factor: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[int, tuple], int, float, int]:
    """Anchor + warm-walk through ``v_rhe_grid`` at λ=0.

    Returns
    -------
    (records, snapshots, mesh_dof_count, electrode_area_nondim,
     electrode_marker)

    - records: per-V dicts of baseline observables.
    - snapshots: ``{grid_idx: U_snapshot_tuple}`` for converged V.
    - mesh_dof_count: ``ctx['U'].function_space().dim()``.
    - electrode_area_nondim: assembled `Constant(1.0) * ds(electrode_marker)`.
    - electrode_marker: int.
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    import numpy as np
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.anchor_continuation import (
        extract_preconverged_anchor,
        solve_anchor_with_continuation,
        LadderExhausted,
    )
    from Forward.bv_solver.cation_hydrolysis import (
        collect_v10a_rung_diagnostics, extract_gamma_value,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T

    # Two-stage anchor (see STERN_F_M2_ANCHOR docstring): build at
    # C_S = 0.10 (convergence-pinned), bump to STERN_F_M2_BASELINE
    # = 0.20 via runtime accessor + Newton resolve.  The sp passed in
    # has C_S = STERN_F_M2_BASELINE; we build a one-off variant for
    # the anchor with C_S = STERN_F_M2_ANCHOR.
    from Forward.bv_solver.anchor_continuation import (
        set_stern_capacitance_model,
    )
    sp_anchor_cs = _build_sp_at_cs(
        sp_template=sp,
        stern_capacitance_f_m2=STERN_F_M2_ANCHOR,
    )
    sp_anchor = sp_anchor_cs.with_phi_applied(v_anchor / V_T)
    k0_r4e_target = float(K0_HAT_R4E) * float(k0_r4e_factor)

    print(f"Pass 1: anchor at V={v_anchor:+.3f} V (kw ladder, λ=0); "
          f"C_S_anchor={STERN_F_M2_ANCHOR:.3f} F/m² → bump to "
          f"{STERN_F_M2_BASELINE:.3f} F/m² post-anchor; "
          f"K0_R4e_target={k0_r4e_target:.3g}", flush=True)
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: k0_r4e_target},
                initial_scales=K0_INITIAL_SCALES,
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                kw_eff_ladder=_build_kw_ladder(),
            )
    except LadderExhausted as exc:
        raise RuntimeError(f"Anchor failed: {exc}") from exc
    print(f"  anchor done in {time.time() - t0:.1f}s; "
          f"converged={anchor_result.converged}", flush=True)
    if not anchor_result.converged:
        raise RuntimeError("Anchor did not reach k0=1.0 — re-tune ladder.")

    ctx_anchor = anchor_result.ctx
    mesh_dof_count = ctx_anchor["U"].function_space().dim()
    electrode_marker = int(ctx_anchor["bv_settings"]["electrode_marker"])
    ds = fd.Measure("ds", domain=ctx_anchor["mesh"])
    electrode_area_nondim = float(
        fd.assemble(fd.Constant(1.0) * ds(electrode_marker))
    )
    print(f"  mesh_dof_count={mesh_dof_count}, "
          f"electrode_area_nondim={electrode_area_nondim:.6g}", flush=True)

    # Stage 2 of the two-stage anchor: bump C_S 0.10 → 0.20 and
    # re-solve Newton on ctx['U'] to absorb the residual change.
    print(f"  bumping C_S from {STERN_F_M2_ANCHOR:.3f} to "
          f"{STERN_F_M2_BASELINE:.3f} F/m² and re-solving anchor...",
          flush=True)
    t_bump = time.time()
    set_stern_capacitance_model(ctx_anchor, float(STERN_F_M2_BASELINE))
    with adj.stop_annotating():
        ctx_anchor["_last_solver"].solve()
    print(f"  C_S bump re-solved in {time.time() - t_bump:.1f}s",
          flush=True)

    # Capture U at C_S = STERN_F_M2_BASELINE (post-bump).  The
    # extract_preconverged_anchor helper would copy result.U_data
    # which is the pre-bump snapshot; build the PreconvergedAnchor
    # directly so the warm-walk seeds at C_S = 0.20.
    import numpy as np
    from Forward.bv_solver.anchor_continuation import PreconvergedAnchor

    U_post_bump = snapshot_U(ctx_anchor["U"])
    anchor = PreconvergedAnchor(
        phi_applied_eta=float(v_anchor / V_T),
        U_snapshot=tuple(np.asarray(arr).copy() for arr in U_post_bump),
        k0_targets=tuple(
            (int(j), float(k))
            for j, k in sorted(
                {0: float(K0_HAT_R2E), 1: k0_r4e_target}.items()
            )
        ),
        mesh_dof_count=int(mesh_dof_count),
        ladder_history=tuple(
            (float(s), str(o)) for s, o in anchor_result.ladder_history
        ),
    )

    records: List[Dict[str, Any]] = [
        {
            "v_rhe": float(v),
            "phi_applied_eta": float(v) / V_T,
            "lambda_zero_converged": False,
        }
        for v in v_rhe_grid
    ]
    snapshots: Dict[int, tuple] = {}

    # H+ idx = 2 in the K2SO4 4sp stack.
    h_idx = 2
    # K+ idx = 3.
    k_idx = 3

    def _grab(orig_idx: int, _phi_eta: float, ctx: dict) -> None:
        records[orig_idx]["lambda_zero_converged"] = True
        try:
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density",
                reaction_index=None, scale=-I_SCALE,
            )
            f_pc = _build_bv_observable_form(
                ctx, mode="peroxide_current",
                reaction_index=None, scale=-I_SCALE,
            )
            records[orig_idx]["cd_mA_cm2"] = float(fd.assemble(f_cd))
            records[orig_idx]["pc_mA_cm2"] = float(fd.assemble(f_pc))
        except Exception as exc:
            records[orig_idx]["cd_mA_cm2"] = None
            records[orig_idx]["pc_mA_cm2"] = None
            records[orig_idx]["lambda_zero_observable_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        # surface concentrations
        try:
            ci = ctx["ci_exprs"]
            ds_local = fd.Measure("ds", domain=ctx["mesh"])
            em = ctx["bv_settings"]["electrode_marker"]
            area_local = float(
                fd.assemble(fd.Constant(1.0) * ds_local(em))
            )
            records[orig_idx]["c_H_surface_nondim"] = (
                float(fd.assemble(ci[h_idx] * ds_local(em))) / area_local
            )
            records[orig_idx]["c_K_surface_nondim"] = (
                float(fd.assemble(ci[k_idx] * ds_local(em))) / area_local
            )
        except Exception as exc:
            records[orig_idx]["c_H_surface_nondim"] = None
            records[orig_idx]["c_K_surface_nondim"] = None
            records[orig_idx]["lambda_zero_surface_conc_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        # Γ at λ=0 must be 0.
        try:
            records[orig_idx]["gamma_at_lambda_zero"] = (
                extract_gamma_value(ctx)
            )
        except Exception:
            records[orig_idx]["gamma_at_lambda_zero"] = None

        # σ_S at λ=0 baseline — from the v10a diagnostics helper.
        try:
            diag = collect_v10a_rung_diagnostics(ctx)
            records[orig_idx]["lambda_zero_sigma_S_C_per_m2"] = (
                diag.get("sigma_S_C_per_m2")
            )
            records[orig_idx]["lambda_zero_sigma_S_counts_per_pm2"] = (
                diag.get("sigma_S_counts_per_pm2")
            )
            records[orig_idx]["lambda_zero_R_2e_current_nondim"] = (
                diag.get("R_2e_current_nondim")
            )
            records[orig_idx]["lambda_zero_R_4e_current_nondim"] = (
                diag.get("R_4e_current_nondim")
            )
        except Exception as exc:
            records[orig_idx]["lambda_zero_v10a_diag_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        snapshots[orig_idx] = snapshot_U(ctx["U"])

    phi_grid = np.array([float(v) / V_T for v in v_rhe_grid])
    t1 = time.time()
    with adj.stop_annotating():
        grid_result = solve_grid_with_anchor(
            sp, mesh=mesh, anchor=anchor,
            phi_applied_values=phi_grid,
            per_point_callback=_grab,
        )
    walk_wall = time.time() - t1
    n_converged = sum(p.converged for p in grid_result.points.values())
    print(f"  λ=0 walk: {n_converged}/{len(v_rhe_grid)} converged in "
          f"{walk_wall:.1f}s", flush=True)
    return records, snapshots, mesh_dof_count, electrode_area_nondim, electrode_marker


# ---------------------------------------------------------------------------
# Pass 2 — λ=1 ramp per V (+ optional ±C_S perturbation)
# ---------------------------------------------------------------------------


def _run_lambda_ramp(
    *, sp_template, mesh, voltage: float,
    U_warmstart: tuple, parameter_overrides: Optional[Dict[str, Any]] = None,
    k0_r4e_factor: float = 1.0,
):
    """Wrap :func:`solve_lambda_ramp_from_warm_start` for a single V.

    Returns ``(result, final_diag)`` or ``(None, None)`` on failure.
    """
    from Forward.bv_solver.anchor_continuation import (
        solve_lambda_ramp_from_warm_start, LadderExhausted,
    )
    from scripts._bv_common import K0_HAT_R2E, K0_HAT_R4E, V_T

    sp_at_v = sp_template.with_phi_applied(voltage / V_T)
    k0_r4e_target = float(K0_HAT_R4E) * float(k0_r4e_factor)
    try:
        result = solve_lambda_ramp_from_warm_start(
            sp_at_v, mesh=mesh, U_warmstart=U_warmstart,
            k0_targets={0: float(K0_HAT_R2E), 1: k0_r4e_target},
            lambda_hydrolysis_ladder=LAMBDA_LADDER,
            parameter_overrides=parameter_overrides or {},
        )
    except LadderExhausted as exc:
        return None, f"LadderExhausted: {exc}"

    # Final rung is the converged λ=1.0 state.  Last rung diagnostic
    # carries the v10a fields (F0_avg, gamma, theta, sigma_S, etc.).
    final = None
    for rung in reversed(result.rungs):
        if rung.get("snes_converged") and rung.get("lambda_hydrolysis", -1) == 1.0:
            final = rung
            break
    if final is None and result.rungs:
        final = result.rungs[-1]
    return result, final


def _extract_R_net_nondim(diag: Dict[str, Any]) -> Optional[float]:
    """``R_net = k_des · Γ`` at steady state (mass balance).

    Returns nondim flux value, or None if Γ is missing.
    """
    if diag is None:
        return None
    gamma = diag.get("gamma")
    k_des = diag.get("k_des")
    if gamma is None or k_des is None:
        return None
    return float(k_des) * float(gamma)


def _lambda_one_with_perturbation(
    *, sp_template, mesh, snapshots: Dict[int, tuple], records: List[Dict[str, Any]],
    epsilon: float = PERTURB_FRACTION, run_perturbation: bool = True,
    k0_r4e_factor: float = 1.0,
) -> None:
    """Run λ=1 ramp at each converged V; optional ±C_S perturbation.

    Mutates ``records`` in-place adding the λ=1 + perturbation
    quantities.
    """
    from Forward.bv_solver.cation_hydrolysis import (
        collect_v10a_rung_diagnostics,
    )
    from scripts._bv_common import I_SCALE

    cs_base = STERN_F_M2_BASELINE

    for idx, rec in enumerate(records):
        if idx not in snapshots:
            rec["lambda_one_converged"] = False
            rec["perturbation_converged"] = False
            continue
        U_snap = snapshots[idx]
        voltage = float(rec["v_rhe"])

        # Base solve (unperturbed C_S).
        t_v = time.time()
        result_base, final_base = _run_lambda_ramp(
            sp_template=sp_template, mesh=mesh,
            voltage=voltage, U_warmstart=U_snap,
            parameter_overrides={"stern_capacitance_f_m2": cs_base},
            k0_r4e_factor=k0_r4e_factor,
        )
        if result_base is None or final_base is None or not final_base.get("snes_converged"):
            rec["lambda_one_converged"] = False
            rec["perturbation_converged"] = False
            rec["lambda_one_error"] = (
                final_base.get("lambda_one_error", "unknown") if final_base else "no rungs"
            )
            continue
        rec["lambda_one_converged"] = True
        rec["lambda_one_wall_seconds"] = float(time.time() - t_v)

        # Fold the v10a diagnostics from the final λ=1 rung into the record.
        for key in (
            "gamma", "gamma_max", "theta",
            "lambda_hydrolysis", "k_hyd", "k_prot", "k_des",
            "delta_ohp_hat",
            "F0_avg", "forward_avg_no_k_hyd",
            "c_H_avg", "pka_shift_avg",
            "R_forward_capped",
            "denominator_constant", "denominator_kdes",
            "denominator_kprot", "denominator_cap",
            "denominator_total", "denominator_cap_to_total_ratio",
            "numerator",
            "sigma_S_C_per_m2", "sigma_S_counts_per_pm2",
            "R_2e_current_nondim", "R_4e_current_nondim",
            # Phase 6β v10a' enhanced decompositions.
            "F0_decomposition", "R_4e_decomposition_log",
            "F0_decomposition_error", "R_4e_decomposition_log_error",
        ):
            rec[key] = final_base.get(key)
        # cd_observable from solve_lambda_ramp_from_warm_start is nondim
        # (scale=1.0 default in the orchestrator).  Convert to mA/cm² so
        # the locked rule's |cd|/I_lim_4e ratio is dimensionally
        # consistent with the Levich helper.  Sign matches the existing
        # mangan_full_grid driver convention (cathodic is negative).
        cd_nondim = final_base.get("cd_observable")
        rec["cd_observable_nondim"] = cd_nondim
        rec["cd_mA_cm2"] = (
            -float(I_SCALE) * float(cd_nondim) if cd_nondim is not None
            else None
        )

        # R_net at λ=1 SS = k_des · Γ (mass balance).
        rec["R_net_unperturbed"] = _extract_R_net_nondim(final_base)
        rec["sigma_S_unperturbed"] = final_base.get("sigma_S_C_per_m2")
        rec["C_s_unperturbed"] = cs_base

        if not run_perturbation:
            rec["perturbation_converged"] = False
            rec["perturbation_skipped"] = True
            continue

        cs_low = cs_base * (1.0 - epsilon)
        cs_high = cs_base * (1.0 + epsilon)

        # Perturbed solves warm-start from the same U_snap (NOT from
        # the base result), so each perturbation lives within the same
        # numerical-precision neighbourhood.
        result_lo, final_lo = _run_lambda_ramp(
            sp_template=sp_template, mesh=mesh,
            voltage=voltage, U_warmstart=U_snap,
            parameter_overrides={"stern_capacitance_f_m2": cs_low},
            k0_r4e_factor=k0_r4e_factor,
        )
        result_hi, final_hi = _run_lambda_ramp(
            sp_template=sp_template, mesh=mesh,
            voltage=voltage, U_warmstart=U_snap,
            parameter_overrides={"stern_capacitance_f_m2": cs_high},
            k0_r4e_factor=k0_r4e_factor,
        )
        ok_lo = (
            result_lo is not None and final_lo is not None
            and final_lo.get("snes_converged", False)
            and final_lo.get("lambda_hydrolysis") == 1.0
        )
        ok_hi = (
            result_hi is not None and final_hi is not None
            and final_hi.get("snes_converged", False)
            and final_hi.get("lambda_hydrolysis") == 1.0
        )
        rec["perturbation_converged"] = bool(ok_lo and ok_hi)
        rec["C_s_low"] = cs_low
        rec["C_s_high"] = cs_high
        rec["epsilon"] = epsilon
        rec["fixed_at_perturb"] = [
            "V_RHE", "k_hyd", "k_prot", "k_des",
            "gamma_max", "r_H_El", "delta_ohp_hat",
        ]
        rec["relaxed_at_perturb"] = ["phi", "U(species)", "Gamma_via_Picard"]

        rec["sigma_S_low"] = (
            final_lo.get("sigma_S_C_per_m2") if ok_lo else None
        )
        rec["sigma_S_high"] = (
            final_hi.get("sigma_S_C_per_m2") if ok_hi else None
        )
        rec["R_net_low"] = (
            _extract_R_net_nondim(final_lo) if ok_lo else None
        )
        rec["R_net_high"] = (
            _extract_R_net_nondim(final_hi) if ok_hi else None
        )


# ---------------------------------------------------------------------------
# JSON output + summary plot
# ---------------------------------------------------------------------------


def _serialize(obj: Any) -> Any:
    """Recursive JSON-safe conversion (handles numpy/None/inf)."""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (int, bool)):
        return obj if isinstance(obj, bool) else int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return float(obj)
    if obj is None or isinstance(obj, str):
        return obj
    # numpy scalar?
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj) if math.isfinite(float(obj)) else None
        if isinstance(obj, np.integer):
            return int(obj)
    except ImportError:                                    # pragma: no cover
        pass
    return str(obj)


def _write_outputs(
    *, records: List[Dict[str, Any]], decision: VKinDecision,
    config: Dict[str, Any], i_lim_4e_mA_cm2: float,
    electrode_area_nondim: float, domain_height_hat: float,
    out_dir: str, plot: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "config":                config,
        "i_lim_4e_mA_cm2":       i_lim_4e_mA_cm2,
        "electrode_area_nondim": electrode_area_nondim,
        "domain_height_hat":     domain_height_hat,
        "per_v_records":         decision.per_v_decisions or records,
        "v_kin_decision":        decision.to_json(),
    }
    json_path = os.path.join(out_dir, "iv_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(_serialize(payload), f, indent=2, sort_keys=False)
    print(f"Wrote {json_path}", flush=True)

    if not plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot", flush=True)
        return

    enriched = decision.per_v_decisions or records
    v = [r.get("v_rhe") for r in enriched]
    sigma_S = [r.get("sigma_S_C_per_m2") for r in enriched]
    R_net = [r.get("R_net_unperturbed") for r in enriched]
    sens = [r.get("dRnet_dsigma_along_stern_capacitance") for r in enriched]
    abs_sens = [abs(s) if s is not None else None for s in sens]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)

    def _x_y(xs, ys):
        return [
            (x, y) for x, y in zip(xs, ys)
            if x is not None and y is not None
        ]

    pairs = _x_y(v, sigma_S)
    axes[0].plot([p[0] for p in pairs], [p[1] for p in pairs], "o-")
    axes[0].axhline(0.0, color="gray", ls="--", lw=0.8)
    axes[0].set_ylabel("σ_S (C/m²)")
    axes[0].set_title("Phase 6β v10a — V-sweep diagnostic")

    pairs = _x_y(v, R_net)
    axes[1].plot([p[0] for p in pairs], [p[1] for p in pairs], "s-")
    axes[1].set_ylabel("R_net = k_des·Γ (nondim)")

    pairs = _x_y(v, abs_sens)
    axes[2].plot([p[0] for p in pairs], [p[1] for p in pairs], "^-")
    axes[2].set_ylabel("|dR_net/dσ_S| (Stern-cap. manifold)")
    axes[2].set_xlabel("V_RHE (V)")
    if decision.v_kin is not None:
        for ax in axes:
            ax.axvline(decision.v_kin, color="red", ls=":", lw=1.0,
                       label=f"V_kin = {decision.v_kin:+.3f} V")
        axes[0].legend(loc="best", fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "iv_diagnostic.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {plot_path}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6β v10a V-sweep diagnostic + V_kin selection.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use the 5-point quick grid (~5-10 min).",
    )
    parser.add_argument(
        "--v-rhe-grid", default=None,
        help="Override V_RHE grid as comma-separated floats, e.g. "
             "'0.55,0.10,-0.10,-0.30,-0.50'.",
    )
    parser.add_argument(
        "--v-anchor", type=float, default=V_ANCHOR_DEFAULT,
        help=f"Anchor voltage (V vs RHE). Default {V_ANCHOR_DEFAULT}.",
    )
    parser.add_argument(
        "--no-perturbation", action="store_true",
        help="Skip the ±C_S perturbation pass (FD-only sensitivity).",
    )
    parser.add_argument(
        "--k0-r4e-factor", type=float, default=1.0,
        help="Dimensionless multiplier on K0_HAT_R4E (default 1.0). "
             "Use scientific notation, e.g. '1e-14' for the "
             "v10a' V=-0.10 branch-pass probe.  See plan §Implementation "
             "notes for the V-vs-factor table.",
    )
    parser.add_argument(
        "--out-subdir", default=OUT_SUBDIR_DEFAULT,
        help=f"StudyResults/<subdir> output location. Default "
             f"{OUT_SUBDIR_DEFAULT}.",
    )
    parser.add_argument(
        "--plot", dest="plot", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-plot", dest="plot", action="store_false",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.v_rhe_grid is not None:
        v_grid = tuple(float(x) for x in args.v_rhe_grid.split(","))
    elif args.quick:
        v_grid = V_RHE_GRID_QUICK
    else:
        v_grid = V_RHE_GRID_DEFAULT

    # ``_ROOT`` already resolves to the ``PNPInverse/`` project root
    # (parent of ``scripts/``); ``StudyResults`` sits directly under it.
    out_dir = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    print(f"V_RHE grid ({len(v_grid)} points): {v_grid}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    t_start = time.time()

    k0_r4e_factor = float(args.k0_r4e_factor)
    print(f"k0_r4e_factor = {k0_r4e_factor:.3g} "
          f"(K0_HAT_R4E_eff = K0_HAT_R4E · {k0_r4e_factor:.3g})", flush=True)

    sp = _build_sp(lambda_hydrolysis=0.0, k0_r4e_factor=k0_r4e_factor)
    mesh = _make_mesh(l_eff_m=L_EFF_M_BASELINE)

    # Pass 1.
    records, snapshots, mesh_dof_count, electrode_area_nondim, electrode_marker = (
        _walk_lambda_zero_capture_snapshots(
            sp=sp, mesh=mesh,
            v_rhe_grid=v_grid, v_anchor=args.v_anchor,
            k0_r4e_factor=k0_r4e_factor,
        )
    )

    # Pass 2.
    print(f"Pass 2: λ=1 ramps (perturbation={not args.no_perturbation})",
          flush=True)
    t2 = time.time()
    _lambda_one_with_perturbation(
        sp_template=sp, mesh=mesh,
        snapshots=snapshots, records=records,
        run_perturbation=(not args.no_perturbation),
        k0_r4e_factor=k0_r4e_factor,
    )
    print(f"  λ=1 + perturbation: {time.time() - t2:.1f}s", flush=True)

    # Attach O₂ Levich ratio per record.
    domain_height_hat = L_EFF_M_BASELINE / 1.0e-4
    for rec in records:
        r2 = rec.get("R_2e_current_nondim")
        r4 = rec.get("R_4e_current_nondim")
        if r2 is not None and r4 is not None and electrode_area_nondim > 0.0:
            rec["o2_flux_levich_ratio"] = _compute_o2_flux_levich_ratio(
                R_2e_current_nondim=r2, R_4e_current_nondim=r4,
                electrode_area_nondim=electrode_area_nondim,
                domain_height_hat=domain_height_hat,
            )
        else:
            rec["o2_flux_levich_ratio"] = None

    # FD column (informational only).
    attach_fd_sensitivity(records)

    # V_kin selection.
    i_lim_4e = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)
    print(f"I_lim_4e at l_eff={L_EFF_M_BASELINE*1e6:.1f} µm = "
          f"{i_lim_4e:.4g} mA/cm²", flush=True)
    decision = select_v_kin(
        records, i_lim_4e_mA_cm2=i_lim_4e,
        electrode_area_nondim=electrode_area_nondim,
        domain_height_hat=domain_height_hat,
    )

    if decision.abort_to_v10c:
        print("=== abort_to_v10c: no V has σ_S < 0 ===", flush=True)
    elif decision.no_valid_stern_capacitance_sensitivity:
        print("=== no_valid_stern_capacitance_sensitivity ===", flush=True)
    elif decision.no_candidate_passed_locked_rule:
        print("=== no_candidate_passed_locked_rule ===", flush=True)
    else:
        print(f"=== V_kin = {decision.v_kin:+.3f} V "
              f"(score={decision.score:.4g}, "
              f"fallback_used={decision.fallback_used}) ===", flush=True)

    transport_limited_count = sum(
        1 for r in (decision.per_v_decisions or records)
        if r.get("locked_current_filter_passes_but_o2_transport_limited")
    )
    if transport_limited_count > 0:
        print(f"  (warning: {transport_limited_count} V passed locked "
              f"current filter but o2_flux_levich_ratio > 0.9 — see "
              f"locked-rule asymmetry note)", flush=True)

    from scripts._bv_common import K0_HAT_R2E, K0_HAT_R4E
    config = {
        "v_rhe_grid":            list(v_grid),
        "v_anchor":              args.v_anchor,
        "lambda_ladder":         list(LAMBDA_LADDER),
        "k0_initial_scales":     list(K0_INITIAL_SCALES),
        "smoke_kinetics":        SMOKE_KINETICS,
        "stern_capacitance_f_m2": STERN_F_M2_BASELINE,
        "l_eff_m":               L_EFF_M_BASELINE,
        "perturb_fraction":      PERTURB_FRACTION,
        "run_perturbation":      (not args.no_perturbation),
        "sigma_abs_min_C_per_m2": SIGMA_ABS_MIN_C_PER_M2,
        "k0_r4e_factor":         k0_r4e_factor,
        "K0_HAT_R2E_baseline":   float(K0_HAT_R2E),
        "K0_HAT_R4E_baseline":   float(K0_HAT_R4E),
        "K0_HAT_R4E_effective":  float(K0_HAT_R4E) * k0_r4e_factor,
        "mesh":                  {
            "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
            "domain_height_hat": domain_height_hat,
        },
        "electrode_marker":      electrode_marker,
        "mesh_dof_count":        mesh_dof_count,
        "wall_seconds":          float(time.time() - t_start),
    }
    _write_outputs(
        records=records, decision=decision, config=config,
        i_lim_4e_mA_cm2=i_lim_4e,
        electrode_area_nondim=electrode_area_nondim,
        domain_height_hat=domain_height_hat,
        out_dir=out_dir, plot=args.plot,
    )

    return 0 if decision.v_kin is not None else 1


if __name__ == "__main__":
    sys.exit(main())
