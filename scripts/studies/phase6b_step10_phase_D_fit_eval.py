"""Phase 6β step 10 — Phase D Δ_β fit evaluator (per-eval driver).

One invocation evaluates a single Δ_β candidate against the locked
24-point V grid using the V10B-calibrated stack and the Phase A.2
continuation topology (anchor at V_anchor with λ=0; warm-walk grid at
λ=0; per-V λ ramp 0→1).  Emits one JSON per eval.

Plan: ``~/.claude/plans/phase6b-step10-phase-D-deltaBeta-fit.md``
(v7-FINAL, GPT critique session 37).  Section references inline.

CLI::

    python -u scripts/studies/phase6b_step10_phase_D_fit_eval.py \\
        --delta-beta 0.0 --sigma-mapping stern \\
        --out-subdir phase6b_step10_phase_D --out-name eval_db_0p0_stern.json

``evaluate_delta_beta`` is the Python entry point; the Phase D
orchestration (pre-fit grid, Brent, identifiability gate) imports it
from this module and drives it in-process.

Lazy Firedrake imports
----------------------

Top-of-module is import-clean so the CLI argparse, V grid constant,
mask, ring-onset interp, selectivity formula, and aggregation helpers
are testable without the venv-firedrake environment.  Firedrake-backed
helpers live below ``# === Firedrake-required driver body ===``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Module-level path setup
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Locked constants — bundle invariants (DO NOT TOUCH in Phase D)
# ---------------------------------------------------------------------------

V_RHE_PRODUCTION_GRID: Tuple[float, ...] = (
    -0.10,
    -0.06, -0.01, 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34,
    0.39, 0.44, 0.49, 0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84,
    0.89, 0.94, 0.99, 1.00,
)
"""Locked 24-point V_RHE grid (plan §D2).

Index 0 (``V_KIN = -0.10``) is solved + reported per-V but EXCLUDED
from the Phase D observable aggregation; observables are masked to
``[V_KIN_OBS_MASK_LO, V_KIN_OBS_MASK_HI] = [-0.06, +1.0]`` per the
acceptance bundle line 107 (deck overlap window).
"""

V_ANCHOR: float = +0.55
"""Anchor voltage — anchor-only, NOT a grid point.  Anchor outputs
are excluded from observable aggregation (plan §D2)."""

V_KIN_OBS_MASK_LO: float = -0.06
V_KIN_OBS_MASK_HI: float = +1.00
"""Observable extraction mask bounds (deck overlap)."""

V_KIN_BYTE_EQUIV_BASELINE: float = -0.10
"""V_KIN reference — solved per-V; used for the byte-equivalence
HARD reproduction baseline (D5(a)) and the ``pka_shift_avg < 0``
sign-guard verification.  EXCLUDED from observable aggregation."""

LAMBDA_LADDER: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
"""Per-V λ ramp ladder (plan §D2 step 4).  5 rungs."""

K_HYD_BASELINE: float = 1e-3
"""``k_hyd_baseline`` per plan §2 hard invariant + step 5."""

A2_WARM_GRID: Tuple[float, ...] = (+0.55, +0.40, +0.20, +0.10, -0.10)
"""5-point grid used by the A.2-compatible HARD reproduction (D5(a)).
NOT the 24-point production grid.  Reach V_KIN via the same intermediate
sequence as the v10a/A.2 driver.
"""

K0_R4E_FACTOR_V10B: float = 1e-14
"""V10B HARD invariant — locked in ``calibration/v10b.py:64`` and
``project_v10a_prime_outcome`` memory.  Multiplier on the 4e branch's
``K0_HAT_R4E`` so the V_kin = -0.10 V solve lands in the cd ~ -3 mA/cm²
band that matches the v10a' anchor.  Phase D MUST inherit this — the
HARD reproduction baseline (D5(a)) compares to the v10b A.2 V_kin record
which used ``K0_R4e_factor = 1e-14``.
"""

N_COLLECTION: float = 0.224
"""Ruggiero §2 RRDE collection efficiency.  Used in the n_e_RRDE
formula."""

# σ-mapping enums
SIGMA_MAPPING_STERN: str = "stern"
SIGMA_MAPPING_ABLATION: str = "ablation_singh_0.141"
SIGMA_MAPPINGS: Tuple[str, ...] = (SIGMA_MAPPING_STERN, SIGMA_MAPPING_ABLATION)

ABLATION_SIGMA_SINGH_COUNTS_PM2: float = 0.141
"""Singh's K Cu cell-level σ value (V-independent), per
``docs/phase6/singh_2016_pka_formula.md`` line 241."""

# HARD per-V gate thresholds (plan §D2)
GATE_MASS_BALANCE_RESIDUAL_REL_MAX: float = 5e-3
GATE_ANALYTIC_GAMMA_REL_MAX: float = 5e-3
GATE_PKA_SHIFT_OVERFLOW: float = 15.0
"""``|pka_shift_avg(V)| ≤ 15`` -- post-solve guard; out-of-domain
candidates (``> 15``) are flagged ``pka_shift_overflow`` and excluded
from optimizer consumption (plan §D6)."""

# Adaptive ring-onset refinement
RING_ONSET_THRESHOLD_MA_CM2: float = 0.01
RING_ONSET_REFINE_N: int = 4
RING_ONSET_REFINE_SPACING_V: float = 0.01

SELECTIVITY_MIN_CATHODIC_DISK_MA_CM2: float = 1e-3
"""Selectivity-aggregation guard: at V's where the disk current has
|cd| < 1 µA/cm² (or cd > 0), the system has effectively no ORR
happening and the formula ``200·(I_ring/N) / (|I_disk| + I_ring/N)``
returns an unphysical value (>100% when cd → 0 with non-zero ring
noise; flips sign at anodic V).  These V records are EXCLUDED from
selectivity / n_e aggregation but kept in the per-V records for
diagnostic completeness.  Threshold matches the deck convention where
the ring-onset gate is at 0.01 mA/cm² ring-basis (Brianna xlsx
column 8); the disk threshold is 10x lower so we don't accidentally
exclude legitimate ORR-onset V's where ring is at 0.01 but disk is
at 0.005.
"""

OUT_SUBDIR_DEFAULT: str = "phase6b_step10_phase_D"

ANCHOR_CACHE_VERSION: str = "phase_D_anchor_cache_v1"
"""Schema version tag stamped into every pickled anchor cache.  Bump
when the on-disk dict layout changes; ``_try_load_anchor_cache``
treats version mismatches as a miss and falls back to a full solve.
"""

ANCHOR_CACHE_DISABLE_ENV: str = "PHASE_D_DISABLE_ANCHOR_CACHE"
"""Set ``PHASE_D_DISABLE_ANCHOR_CACHE=1`` in the environment to
short-circuit Optimization F (cache load AND save) — useful for
benchmarking the cold path or for forcing a re-solve after a code
change that does NOT alter the cache hash."""


# ===========================================================================
# Pure-Python helpers — observable extraction, masks, aggregation
# (no Firedrake; tested in tests/test_phase6b_step10_phase_D_fit_eval.py)
# ===========================================================================


def in_observable_mask(v_rhe: float) -> bool:
    """True if ``v_rhe`` lies in the locked observable mask
    ``[V_KIN_OBS_MASK_LO, V_KIN_OBS_MASK_HI]`` (V_KIN excluded)."""
    return V_KIN_OBS_MASK_LO <= float(v_rhe) <= V_KIN_OBS_MASK_HI


def selectivity_h2o2_pct(
    *, i_disk_mA_cm2: float, i_ring_mA_cm2: float,
    n_collection: float = N_COLLECTION,
) -> float:
    """``200 · (I_ring / N) / (|I_disk| + I_ring / N)`` — Ruggiero §2 RRDE
    selectivity formula.  Returns percent.

    Returns 0.0 when both currents are zero (well-defined limit).
    """
    n = float(n_collection)
    if n <= 0.0:
        raise ValueError(f"n_collection must be > 0 (got {n!r})")
    i_ring_corrected = float(i_ring_mA_cm2) / n
    abs_disk = abs(float(i_disk_mA_cm2))
    denom = abs_disk + i_ring_corrected
    if denom == 0.0:
        return 0.0
    return 200.0 * i_ring_corrected / denom


def n_e_rrde(
    *, i_disk_mA_cm2: float, i_ring_mA_cm2: float,
    n_collection: float = N_COLLECTION,
) -> float:
    """``4 · |I_disk| / (|I_disk| + I_ring / N)`` — Ruggiero §2 RRDE
    electron-transfer number.

    Returns 4.0 when both currents are zero (pure-4e limit).
    """
    n = float(n_collection)
    if n <= 0.0:
        raise ValueError(f"n_collection must be > 0 (got {n!r})")
    i_ring_corrected = float(i_ring_mA_cm2) / n
    abs_disk = abs(float(i_disk_mA_cm2))
    denom = abs_disk + i_ring_corrected
    if denom == 0.0:
        return 4.0
    return 4.0 * abs_disk / denom


def find_ring_onset_v(
    *,
    v_values: Sequence[float],
    ring_currents_mA_cm2: Sequence[float],
    threshold_mA_cm2: float = RING_ONSET_THRESHOLD_MA_CM2,
    mask_lo: float = V_KIN_OBS_MASK_LO,
    mask_hi: float = V_KIN_OBS_MASK_HI,
) -> Optional[float]:
    """Find the first V (sweeping anodic → cathodic in DESCENDING order
    within the mask) where ``ring_current >= threshold``.  Linear
    interpolation between the bracket endpoints.

    Returns ``None`` if no crossing in the masked window.

    Plan §D2 spec:
    > extraction sorts V **descending** (anodic → cathodic) within
    > mask; finds first V where gross_h2o2_current ≥ 0.01 mA/cm²;
    > linear interp.
    """
    if len(v_values) != len(ring_currents_mA_cm2):
        raise ValueError(
            f"v_values len {len(v_values)} != ring_currents len "
            f"{len(ring_currents_mA_cm2)}"
        )
    paired = [
        (float(v), float(i))
        for v, i in zip(v_values, ring_currents_mA_cm2)
        if i is not None and not (isinstance(i, float) and math.isnan(i))
        and mask_lo <= float(v) <= mask_hi
    ]
    if not paired:
        return None
    # Descending in V (anodic → cathodic).
    paired.sort(key=lambda t: -t[0])
    threshold = float(threshold_mA_cm2)
    prev_v, prev_i = paired[0]
    if prev_i >= threshold:
        # Already above threshold at the most-anodic V — extrapolation
        # not supported; report this V as the onset.
        return prev_v
    for v, i in paired[1:]:
        if i >= threshold:
            # Crossed between (prev_v, prev_i) and (v, i).  Linear
            # interpolation: solve for v_cross such that
            # prev_i + (v_cross - prev_v) / (v - prev_v) * (i - prev_i)
            # = threshold.
            denom = i - prev_i
            if denom == 0.0:
                return v
            frac = (threshold - prev_i) / denom
            return prev_v + frac * (v - prev_v)
        prev_v, prev_i = v, i
    return None


def aggregate_observables(
    per_v_records: Sequence[Dict[str, Any]],
    *,
    n_collection: float = N_COLLECTION,
    mask_lo: float = V_KIN_OBS_MASK_LO,
    mask_hi: float = V_KIN_OBS_MASK_HI,
) -> Dict[str, Any]:
    """Compute Phase D primary + secondary observables from per-V records.

    Plan §D2 aggregation list:
    * ``max_H2O2_selectivity_in_window`` — Ruggiero §2 RRDE formula
      ``200·(I_ring/N) / (|I_disk| + I_ring/N)``.  Equivalent to
      ``200·I_disk_2e / (|I_disk_total| + I_disk_2e)`` in disk-basis
      currents.
    * ``argmax_V_for_selectivity``
    * ``ring_onset_V_at_0.01_mA_cm2`` — onset is on the **ring basis**
      to match the deck's "Ring Onset Pot (V, @ 0.01 mA/cm²)"
      convention (Brianna xlsx column 8).
    * ``max_ring_current_in_window`` — max of the **ring-basis** ring
      current to match the deck's "Max Ring Current (mA/cm²)"
      convention (column 9).
    * ``n_e_rrde_at_argmax_V``

    Records outside the mask are EXCLUDED.  Records inside the mask
    that have None on either current are SKIPPED for that observable
    (NaN-aggregation tolerance, but a HARD per-V gate fail in
    production invalidates the entire eval -- the synthetic-NaN test
    just exercises this code path safely).
    """
    masked = [
        rec for rec in per_v_records
        if rec.get("v_rhe") is not None
        and mask_lo <= float(rec["v_rhe"]) <= mask_hi
    ]

    sel_pairs: List[Tuple[float, float]] = []
    n_e_lookup: Dict[float, float] = {}
    ring_pairs: List[Tuple[float, float]] = []
    n_excluded_low_cathodic = 0
    for rec in masked:
        v = float(rec["v_rhe"])
        cd = rec.get("cd_mA_cm2")
        # ring_current_ring_basis_mA_cm2 = pc * N (actual ring current)
        # gross_h2o2_current_mA_cm2 = pc (disk-basis 2e partial current)
        ring_basis = rec.get("ring_current_ring_basis_mA_cm2")
        if ring_basis is None:
            # Backward-compat: fall back to deriving ring-basis from
            # gross_h2o2 (= pc, disk basis) × N.
            gross = rec.get("gross_h2o2_current_mA_cm2")
            if gross is not None:
                ring_basis = float(gross) * float(n_collection)
        if (
            cd is not None and ring_basis is not None
            and not (isinstance(cd, float) and math.isnan(cd))
            and not (isinstance(ring_basis, float)
                     and math.isnan(ring_basis))
        ):
            # Selectivity guard: only meaningful when disk current is
            # cathodic with |cd| above the anodic-noise floor.  See
            # SELECTIVITY_MIN_CATHODIC_DISK_MA_CM2 docstring.
            if (
                float(cd) > -SELECTIVITY_MIN_CATHODIC_DISK_MA_CM2
            ):
                n_excluded_low_cathodic += 1
            else:
                sel = selectivity_h2o2_pct(
                    i_disk_mA_cm2=cd,
                    i_ring_mA_cm2=ring_basis,
                    n_collection=n_collection,
                )
                sel_pairs.append((v, sel))
                n_e_lookup[v] = n_e_rrde(
                    i_disk_mA_cm2=cd,
                    i_ring_mA_cm2=ring_basis,
                    n_collection=n_collection,
                )
        if (
            ring_basis is not None
            and not (isinstance(ring_basis, float)
                     and math.isnan(ring_basis))
        ):
            ring_pairs.append((v, ring_basis))

    out: Dict[str, Any] = {
        "max_H2O2_selectivity_in_window_pct": None,
        "argmax_V_for_selectivity": None,
        "ring_onset_V_at_0.01_mA_cm2": None,
        "max_ring_current_in_window_mA_cm2": None,
        "n_e_rrde_at_argmax_V": None,
        "n_records_in_mask": len(masked),
        "n_records_with_finite_currents": len(sel_pairs),
        "n_excluded_low_cathodic_disk": n_excluded_low_cathodic,
        "selectivity_min_cathodic_disk_threshold_mA_cm2": (
            SELECTIVITY_MIN_CATHODIC_DISK_MA_CM2
        ),
    }
    if sel_pairs:
        argmax_v, argmax_sel = max(sel_pairs, key=lambda t: t[1])
        out["max_H2O2_selectivity_in_window_pct"] = argmax_sel
        out["argmax_V_for_selectivity"] = argmax_v
        out["n_e_rrde_at_argmax_V"] = n_e_lookup.get(argmax_v)
    if ring_pairs:
        out["max_ring_current_in_window_mA_cm2"] = max(
            r for _, r in ring_pairs
        )
        out["ring_onset_V_at_0.01_mA_cm2"] = find_ring_onset_v(
            v_values=[v for v, _ in ring_pairs],
            ring_currents_mA_cm2=[r for _, r in ring_pairs],
            mask_lo=mask_lo, mask_hi=mask_hi,
        )
    return out


def adaptive_ring_onset_refinement_grid(
    *,
    v_values: Sequence[float],
    ring_currents_mA_cm2: Sequence[float],
    threshold_mA_cm2: float = RING_ONSET_THRESHOLD_MA_CM2,
    n_refine: int = RING_ONSET_REFINE_N,
    spacing_v: float = RING_ONSET_REFINE_SPACING_V,
    existing_v_set: Optional[Sequence[float]] = None,
) -> List[float]:
    """Identify the V bracket where ``ring_current`` crosses
    ``threshold`` (descending sort within mask) and return up to
    ``n_refine`` additional V points at ``spacing_v`` inside the
    bracket (NOT at endpoints; no duplicates).

    Returns an empty list if no crossing or bracket is too narrow.
    Existing V values (``existing_v_set``) are excluded.

    Plan §D2 adaptive ring-onset refinement spec.
    """
    paired = [
        (float(v), float(i))
        for v, i in zip(v_values, ring_currents_mA_cm2)
        if i is not None and not (isinstance(i, float) and math.isnan(i))
    ]
    if not paired:
        return []
    paired.sort(key=lambda t: -t[0])
    threshold = float(threshold_mA_cm2)
    bracket: Optional[Tuple[float, float]] = None
    prev_v, prev_i = paired[0]
    for v, i in paired[1:]:
        if i >= threshold > prev_i:
            bracket = (v, prev_v)  # (cathodic-side, anodic-side)
            break
        prev_v, prev_i = v, i
    if bracket is None:
        return []
    v_lo, v_hi = bracket
    if v_hi - v_lo <= spacing_v:
        return []
    existing = (
        set(round(float(x), 6) for x in (existing_v_set or ()))
    )
    out: List[float] = []
    # Insert points strictly INSIDE the bracket at ``spacing_v`` apart,
    # starting from v_lo + spacing.
    v = v_lo + spacing_v
    while v < v_hi - 1e-9 and len(out) < int(n_refine):
        if round(v, 6) not in existing:
            out.append(round(v, 6))
            existing.add(round(v, 6))
        v += spacing_v
    return out


def per_v_gate_status(
    rec: Dict[str, Any],
) -> Tuple[bool, str]:
    """Evaluate the HARD per-V gates from plan §D2.

    Returns (passes, status_str).  status is one of:
    * ``"ok"``
    * ``"newton_unconverged"``
    * ``"picard_unconverged"``
    * ``"mass_balance_high"``
    * ``"analytic_gamma_high"``
    * ``"pka_shift_overflow"``
    """
    if not rec.get("snes_converged", False):
        return False, "newton_unconverged"
    if rec.get("picard_status") not in (None, "converged", "single_iter"):
        return False, "picard_unconverged"
    mb = rec.get("mass_balance_residual_rel")
    if mb is None or abs(float(mb)) >= GATE_MASS_BALANCE_RESIDUAL_REL_MAX:
        return False, "mass_balance_high"
    ag = rec.get("analytic_gamma_rel")
    if ag is not None and abs(float(ag)) >= GATE_ANALYTIC_GAMMA_REL_MAX:
        return False, "analytic_gamma_high"
    pka = rec.get("pka_shift_avg")
    if pka is not None and abs(float(pka)) > GATE_PKA_SHIFT_OVERFLOW:
        return False, "pka_shift_overflow"
    return True, "ok"


def predict_pka_shift_max(
    *, beta_K_Cu: float, delta_beta: float, sigma_max: float,
) -> float:
    """Pre-solve domain check: ``|pka_shift_avg| ≤ 15`` (plan §D6).

    Returns the predicted max ``|ΔpKa|`` over the V grid for the given
    Δ_β + sigma_max.  Caller compares against
    :data:`GATE_PKA_SHIFT_OVERFLOW`.
    """
    return abs((float(beta_K_Cu) + float(delta_beta)) * float(sigma_max))


def signed_sigma_C_m2_to_counts_pm2(sigma_C_m2: float) -> float:
    """Convert signed σ in C/m² to signed σ in counts/pm² (no clamp).

    Identity-preserving inverse of the clamp+conversion in
    ``cation_hydrolysis._build_singh_2016_eq_4_pka_shift``.
    """
    counts_per_m2_per_C_per_m2 = (1.0 / 1.602176634e-19)
    pm2_per_m2 = 1.0e-24
    return float(sigma_C_m2) * counts_per_m2_per_C_per_m2 * pm2_per_m2


def cathodic_clamped_sigma_singh(sigma_C_m2_signed: float) -> float:
    """Apply Singh's anodic-clamp + conversion: returns the value the
    residual sees.  For anodic σ > 0 returns 0.
    """
    return max(
        0.0, -signed_sigma_C_m2_to_counts_pm2(sigma_C_m2_signed),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Phase D fit_eval CLI.  See module docstring for examples."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6β step 10 Phase D — single-Δ_β forward eval against "
            "the locked 24-point V grid."
        ),
    )
    parser.add_argument(
        "--delta-beta", type=float, required=True,
        help=(
            "Δ_β value to evaluate (pm²; carbon-vs-Cu pKa-shift coefficient "
            "offset).  Plan §3.1."
        ),
    )
    parser.add_argument(
        "--sigma-mapping", type=str, default=SIGMA_MAPPING_STERN,
        choices=list(SIGMA_MAPPINGS),
        help=(
            "σ-mapping convention (plan §3.3).  'stern' is production "
            "(σ from PNP/Stern solve); 'ablation_singh_0.141' is the "
            "V-independent ablation path."
        ),
    )
    parser.add_argument(
        "--out-subdir", type=str, default=OUT_SUBDIR_DEFAULT,
        help=(
            "Output directory under StudyResults/.  Default: "
            f"{OUT_SUBDIR_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--out-name", type=str, default=None,
        help=(
            "JSON filename basename (without dir).  Default derived from "
            "delta-beta + sigma-mapping: "
            "'eval_db_<delta-beta>_<sigma-mapping>.json'."
        ),
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Emit a per-eval PNG summary alongside the JSON.",
    )
    parser.add_argument(
        "--mode", type=str, default="production",
        choices=("production", "a2_reproduction"),
        help=(
            "'production' uses the locked 24-pt V grid + per-V λ ramp.  "
            "'a2_reproduction' uses the 5-pt A.2 warm grid + V_kin ramp "
            "for the D5(a) HARD reproduction baseline (plan §D5)."
        ),
    )
    return parser.parse_args(argv)


def _default_out_name(delta_beta: float, sigma_mapping: str) -> str:
    """Generate a deterministic JSON basename from Δ_β + σ-mapping."""
    db_str = (
        f"{delta_beta:.6e}".replace(".", "p").replace("+", "")
        .replace("-", "neg") if delta_beta < 0 else
        f"{delta_beta:.6e}".replace(".", "p").replace("+", "")
    )
    return f"eval_db_{db_str}_{sigma_mapping}.json"


# ===========================================================================
# Anchor-cache helpers (Optimization F) -- Firedrake-free
# ===========================================================================


def _compute_anchor_cache_hash(
    *,
    sigma_mapping: str,
    v_anchor: float,
    k0_r4e_factor: float,
    stern_capacitance_baseline: float,
    stern_capacitance_anchor: float,
    l_eff_m: float,
    mesh_nx: int,
    mesh_ny: int,
    mesh_beta: float,
    kw_eff_ladder_signature: Tuple[float, ...],
    k0_initial_scales: Tuple[float, ...],
    k0_targets_signature: Tuple[Tuple[int, float], ...],
    cache_version: str = ANCHOR_CACHE_VERSION,
) -> str:
    """Stable 8-hex-char hash of the anchor-build inputs.

    Anything that affects the converged Newton state at the V_anchor +
    λ=0 anchor (after the C_S two-stage bump) must enter this hash.
    Δ_β does NOT enter — at λ=0 the cation-hydrolysis source/sink is
    zeroed, so the anchor is independent of β_offset.

    Returns the hex digest of an MD5 of a deterministically-ordered
    JSON encoding of the inputs.  MD5 is fine here — this is a cache
    key, not a security primitive.
    """
    import hashlib
    import json

    payload = {
        "version": str(cache_version),
        "sigma_mapping": str(sigma_mapping),
        "v_anchor": float(v_anchor),
        "k0_r4e_factor": float(k0_r4e_factor),
        "stern_capacitance_baseline": float(stern_capacitance_baseline),
        "stern_capacitance_anchor": float(stern_capacitance_anchor),
        "l_eff_m": float(l_eff_m),
        "mesh_nx": int(mesh_nx),
        "mesh_ny": int(mesh_ny),
        "mesh_beta": float(mesh_beta),
        "kw_eff_ladder": [float(x) for x in kw_eff_ladder_signature],
        "k0_initial_scales": [float(x) for x in k0_initial_scales],
        "k0_targets": [
            [int(j), float(k)] for j, k in k0_targets_signature
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.md5(encoded.encode("utf-8")).hexdigest()
    return digest[:8]


def _anchor_cache_path(
    *, sigma_mapping: str, config_hash: str, out_subdir: str = OUT_SUBDIR_DEFAULT,
) -> str:
    """Absolute pickle path for the cached V_anchor + λ=0 anchor.

    Filename layout: ``anchor_cache_<sigma_mapping>_<config_hash>.pkl``.
    Lives under ``StudyResults/<out_subdir>/`` so the cache lifecycle
    follows Phase D study artifacts.
    """
    out_dir = os.path.join(_ROOT, "StudyResults", out_subdir)
    return os.path.join(
        out_dir, f"anchor_cache_{sigma_mapping}_{config_hash}.pkl"
    )


def _save_anchor_cache(
    *,
    cache_path: str,
    preconverged_anchor: Any,
    electrode_marker: int,
    electrode_area_nondim: float,
    mesh_dof_count: int,
    stern_capacitance_baseline: float,
    k0_r4e_factor: float,
    v_anchor: float,
    sigma_mapping: str,
    config_hash: str,
    cache_version: str = ANCHOR_CACHE_VERSION,
) -> None:
    """Pickle the post-bump anchor + auxiliary scalars to ``cache_path``.

    Creates the parent directory if needed.  Uses
    ``pickle.HIGHEST_PROTOCOL`` so numpy arrays in
    ``preconverged_anchor.U_snapshot`` round-trip exactly.
    """
    import pickle

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "version": str(cache_version),
        "sigma_mapping": str(sigma_mapping),
        "config_hash": str(config_hash),
        "preconverged_anchor": preconverged_anchor,
        "electrode_marker": int(electrode_marker),
        "electrode_area_nondim": float(electrode_area_nondim),
        "mesh_dof_count": int(mesh_dof_count),
        "stern_capacitance_baseline": float(stern_capacitance_baseline),
        "k0_r4e_factor": float(k0_r4e_factor),
        "v_anchor": float(v_anchor),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _try_load_anchor_cache(
    *,
    cache_path: str,
    config_hash: str,
    cache_version: str = ANCHOR_CACHE_VERSION,
) -> Optional[Dict[str, Any]]:
    """Best-effort load of a cached anchor pickle.

    Returns the unpickled dict on success, or ``None`` on any miss
    (file absent, pickle corruption, version mismatch, hash mismatch).
    A miss ALWAYS falls back to a full anchor solve — so this helper
    must never raise; logging is the caller's responsibility.
    """
    import pickle

    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != cache_version:
        return None
    if payload.get("config_hash") != config_hash:
        return None
    if payload.get("preconverged_anchor") is None:
        return None
    return payload


# ===========================================================================
# Firedrake-required driver body
# ===========================================================================


def _evaluate_delta_beta_warm_at_lambda_1(
    *,
    delta_beta_pm2: float,
    sigma_mapping: str,
    v_grid: Sequence[float],
    v_anchor: float,
    mode: str,
    progress: Callable[[str], None],
) -> Dict[str, Any]:
    """OPTIMIZED Phase D forward eval: warm-walk at λ=1 (after a single
    λ ramp at the anchor), with anchor cached across Δ_β evals.

    Topology:

    1. **Anchor at V_anchor with C_S two-stage at λ=0** (Optimization F:
       cache lookup).  Cache HIT skips the kw_eff ladder + C_S bump
       entirely and reuses a :class:`PreconvergedAnchor` snapshot
       pickled from a prior eval at the same σ-mapping + identical sp
       build hash.  Cache MISS runs the kw_eff ladder + Stern bump as
       before, then saves the post-bump U snapshot for subsequent
       evals.  Cached because the anchor at (V=+0.55, λ=0) is
       independent of Δ_β (cation-hydrolysis source/sink is zeroed at
       λ=0), so one cached anchor per σ-mapping serves every Δ_β.
    2. **λ ramp 0→1 at V_anchor (5-rung ``LAMBDA_LADDER``).**  After
       this, the anchor's U snapshot carries the converged (λ=1,
       Γ_ss) state at ``v_anchor``.
    3. For each V in ``v_grid``, sorted by ``|V - v_anchor|`` ascending,
       run a single-rung ``solve_lambda_ramp_from_warm_start`` at that
       V with ``ladder=(1.0,)``.  Each call builds a fresh ctx + form
       at the new V, restores U from the nearest converged neighbour,
       ``reconverge_at_ss`` absorbs the V change, and the Picard outer
       loop at λ=1 converges Γ_ss.

    Skipping the λ=0 warm-walk through ``v_grid`` (~3 min/V at K2SO4
    stiffness) is the main savings; per-V cost drops from ~167 s to
    ~10 s.

    NOTE: an "Optimization A" single-ctx walk
    (``warm_walk_phi`` + manual Picard, sharing one ctx across all
    V's) was prototyped but reverted (validation 2026-05-11) — for
    small V changes from a warm U, ``warm_walk_phi``'s 8-substep
    bisecting march is far more conservative than the
    ``reconverge_at_ss`` SER loop in
    :func:`solve_lambda_ramp_from_warm_start`, regressing per-V wall
    from ~10 s back to ~150 s.

    See ``_evaluate_delta_beta_legacy_per_v_ramp`` for the legacy
    "warm-walk at λ=0 + per-V λ ramp" topology used by
    ``a2_reproduction`` mode (D5(a) HARD reproduction baseline).
    """
    import os as _os
    import time

    import firedrake as fd
    import firedrake.adjoint as adj
    import numpy as np

    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted, PreconvergedAnchor,
        set_stern_capacitance_model,
        solve_anchor_with_continuation,
        solve_lambda_ramp_from_warm_start,
    )
    from Forward.bv_solver.grid_per_voltage import snapshot_U
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import (
        I_SCALE, K0_HAT_R2E, K0_HAT_R4E, L_REF, V_T,
    )
    from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
        augment_rung_diagnostics,
    )
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        L_EFF_M_BASELINE, MESH_BETA, MESH_NX, MESH_NY,
        STERN_F_M2_ANCHOR, STERN_F_M2_BASELINE,
        K0_INITIAL_SCALES,
        _build_kw_ladder, _build_sp_at_cs, _i_lim_4e_mA_cm2,
    )

    t_start = time.time()

    sp = _build_sp_for_phase_D(
        sigma_mapping=sigma_mapping,
        delta_beta_pm2=delta_beta_pm2,
    )
    mesh = _make_mesh()
    domain_height_hat = float(L_EFF_M_BASELINE) / float(L_REF)
    i_lim_4e_mA_cm2 = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)
    k0_r4e_target = float(K0_HAT_R4E) * float(K0_R4E_FACTOR_V10B)
    k0_targets = {0: float(K0_HAT_R2E), 1: k0_r4e_target}

    # ---- Step 0 — Cache key + lookup
    cache_disabled = bool(
        _os.environ.get(ANCHOR_CACHE_DISABLE_ENV, "").strip()
    )
    kw_eff_ladder_sig = tuple(float(x) for x in _build_kw_ladder())
    config_hash = _compute_anchor_cache_hash(
        sigma_mapping=sigma_mapping,
        v_anchor=float(v_anchor),
        k0_r4e_factor=float(K0_R4E_FACTOR_V10B),
        stern_capacitance_baseline=float(STERN_F_M2_BASELINE),
        stern_capacitance_anchor=float(STERN_F_M2_ANCHOR),
        l_eff_m=float(L_EFF_M_BASELINE),
        mesh_nx=int(MESH_NX),
        mesh_ny=int(MESH_NY),
        mesh_beta=float(MESH_BETA),
        kw_eff_ladder_signature=kw_eff_ladder_sig,
        k0_initial_scales=tuple(float(x) for x in K0_INITIAL_SCALES),
        k0_targets_signature=tuple(
            (int(j), float(k)) for j, k in sorted(k0_targets.items())
        ),
    )
    cache_path = _anchor_cache_path(
        sigma_mapping=sigma_mapping, config_hash=config_hash,
    )
    cached_payload: Optional[Dict[str, Any]] = (
        None if cache_disabled
        else _try_load_anchor_cache(
            cache_path=cache_path, config_hash=config_hash,
        )
    )
    cache_status: str
    if cache_disabled:
        cache_status = "disabled"
    elif cached_payload is not None:
        cache_status = "hit"
    else:
        cache_status = "miss"

    anchor_lam_rungs: List[Dict[str, Any]] = []

    if cached_payload is not None:
        # ---- Step 1 (cache HIT) — skip kw_eff ladder + bump entirely.
        progress(
            f"[Δ_β={delta_beta_pm2:+.6g}, σ={sigma_mapping}] Step 1: "
            f"anchor cache HIT at {cache_path} (config_hash={config_hash}) "
            f"-- skipping kw_eff ladder + C_S bump."
        )
        precon = cached_payload["preconverged_anchor"]
        electrode_marker = int(cached_payload["electrode_marker"])
        electrode_area_nondim = float(
            cached_payload["electrode_area_nondim"]
        )
        mesh_dof_count = int(cached_payload["mesh_dof_count"])
        U_anchor_lambda_0 = tuple(
            np.asarray(arr).copy() for arr in precon.U_snapshot
        )
        ladder_history_for_anchor = list(precon.ladder_history)
    else:
        # ---- Step 1 (cache MISS) — full two-stage anchor
        progress(
            f"[Δ_β={delta_beta_pm2:+.6g}, σ={sigma_mapping}] Step 1: "
            f"anchor cache "
            f"{'DISABLED' if cache_disabled else 'MISS'} "
            f"(config_hash={config_hash}) -- running two-stage anchor at "
            f"V={v_anchor:+.3f} V (kw_eff ladder, λ=0) ..."
        )
        sp_anchor_low_cs = _build_sp_at_cs(
            sp_template=sp, stern_capacitance_f_m2=STERN_F_M2_ANCHOR,
        )
        sp_anchor_at_v = sp_anchor_low_cs.with_phi_applied(v_anchor / V_T)
        t0 = time.time()
        try:
            with adj.stop_annotating():
                anchor_result = solve_anchor_with_continuation(
                    sp_anchor_at_v, mesh=mesh,
                    k0_targets=k0_targets,
                    initial_scales=K0_INITIAL_SCALES,
                    max_inserts_per_step=4,
                    max_ss_steps_per_rung=300,
                    ic_at_target=True,
                    kw_eff_ladder=_build_kw_ladder(),
                )
        except LadderExhausted as exc:
            raise RuntimeError(f"Anchor failed: {exc}") from exc
        if not anchor_result.converged:
            raise RuntimeError(
                "Anchor did not reach k0=1.0 — re-tune ladder."
            )
        progress(f"  anchor done in {time.time() - t0:.1f}s")

        ctx_anchor = anchor_result.ctx
        mesh_dof_count = ctx_anchor["U"].function_space().dim()
        electrode_marker = int(ctx_anchor["bv_settings"]["electrode_marker"])
        ds = fd.Measure("ds", domain=ctx_anchor["mesh"])
        electrode_area_nondim = float(
            fd.assemble(fd.Constant(1.0) * ds(electrode_marker))
        )

        # Stage 2 of two-stage anchor: bump C_S 0.10 → 0.20
        progress(
            f"  bumping C_S {STERN_F_M2_ANCHOR:.3f} → "
            f"{STERN_F_M2_BASELINE:.3f} F/m² and re-solving anchor ..."
        )
        t_bump = time.time()
        set_stern_capacitance_model(ctx_anchor, float(STERN_F_M2_BASELINE))
        with adj.stop_annotating():
            ctx_anchor["_last_solver"].solve()
        progress(f"  C_S bump done in {time.time() - t_bump:.1f}s")
        U_anchor_lambda_0 = snapshot_U(ctx_anchor["U"])

        # Save cache for subsequent evals at the same σ-mapping
        if not cache_disabled:
            try:
                precon = PreconvergedAnchor(
                    phi_applied_eta=float(v_anchor) / float(V_T),
                    U_snapshot=tuple(
                        np.asarray(arr).copy() for arr in U_anchor_lambda_0
                    ),
                    k0_targets=tuple(
                        (int(j), float(k))
                        for j, k in sorted(k0_targets.items())
                    ),
                    mesh_dof_count=int(mesh_dof_count),
                    ladder_history=tuple(
                        (float(s), str(o))
                        for s, o in anchor_result.ladder_history
                    ),
                )
                _save_anchor_cache(
                    cache_path=cache_path,
                    preconverged_anchor=precon,
                    electrode_marker=int(electrode_marker),
                    electrode_area_nondim=float(electrode_area_nondim),
                    mesh_dof_count=int(mesh_dof_count),
                    stern_capacitance_baseline=float(STERN_F_M2_BASELINE),
                    k0_r4e_factor=float(K0_R4E_FACTOR_V10B),
                    v_anchor=float(v_anchor),
                    sigma_mapping=str(sigma_mapping),
                    config_hash=str(config_hash),
                )
                progress(
                    f"  anchor cache SAVED to {cache_path}"
                )
            except Exception as exc:                       # pragma: no cover
                progress(
                    f"  anchor cache SAVE failed (non-fatal): "
                    f"{type(exc).__name__}: {exc}"
                )

    # ---- Step 2 — λ ramp 0→1 at V_anchor (full 5-rung ladder)
    progress(
        f"Step 2: λ ramp 0→1 at V={v_anchor:+.3f} V "
        f"(ladder = {LAMBDA_LADDER}) ..."
    )
    overrides = {
        "k_hyd": float(K_HYD_BASELINE),
        "beta_offset_pm2": float(delta_beta_pm2),
    }

    def _anchor_rung_callback(scale, ok, ctx, rung_diag):
        snapshot = dict(rung_diag)
        snapshot["lambda_hydrolysis"] = float(scale)
        snapshot["snes_converged"] = bool(ok)
        anchor_lam_rungs.append(snapshot)

    t_ramp = time.time()
    try:
        with adj.stop_annotating():
            ramp_result = solve_lambda_ramp_from_warm_start(
                sp.with_phi_applied(v_anchor / V_T),
                mesh=mesh,
                U_warmstart=U_anchor_lambda_0,
                k0_targets=k0_targets,
                lambda_hydrolysis_ladder=LAMBDA_LADDER,
                parameter_overrides=overrides,
                rung_callback=_anchor_rung_callback,
                max_ss_steps_per_rung=300,
            )
    except LadderExhausted as exc:
        raise RuntimeError(
            f"λ ramp at V_anchor failed: {exc}"
        ) from exc
    progress(
        f"  λ ramp at V_anchor done in {time.time() - t_ramp:.1f}s"
    )

    U_anchor_lambda_1 = snapshot_U(ramp_result.ctx["U"])

    # ---- Step 3 — Walk V grid at λ=1 (single-Picard per V)
    progress(
        f"Step 3: walk {len(v_grid)} V's at λ=1 (single-rung Picard "
        f"each, sorted nearest-first from V_anchor) ..."
    )
    # Sort grid by |V - v_anchor| ascending; build a snapshot map for
    # nearest-neighbour warm-starts (anchor is the seed).
    sorted_indices = sorted(
        range(len(v_grid)),
        key=lambda i: abs(float(v_grid[i]) - float(v_anchor)),
    )
    converged_v_to_U: Dict[float, tuple] = {
        round(float(v_anchor), 9): U_anchor_lambda_1
    }
    per_v_records: List[Dict[str, Any]] = [
        {"v_rhe": float(v), "snes_converged": False,
         "picard_status": "not_yet_solved"}
        for v in v_grid
    ]
    per_v_raw_lam1: List[Dict[str, Any]] = [
        {"v_rhe": float(v), "lambda1_rung": None}
        for v in v_grid
    ]

    t_walk = time.time()
    for orig_idx in sorted_indices:
        voltage = float(v_grid[orig_idx])
        # Nearest converged neighbour (smallest |V - V_neighbour|).
        if not converged_v_to_U:
            per_v_records[orig_idx] = {
                "v_rhe": voltage, "snes_converged": False,
                "picard_status": "no_converged_neighbour",
            }
            continue
        neighbour_v = min(
            converged_v_to_U.keys(),
            key=lambda nv: abs(nv - voltage),
        )
        U_warm = converged_v_to_U[neighbour_v]
        sp_at_v = sp.with_phi_applied(voltage / V_T)

        v_rungs: List[Dict[str, Any]] = []
        v_partial_rungs: List[Dict[str, Any]] = []
        pc_mA_cm2_callback: Optional[float] = None

        def _v_rung_callback(scale, ok, ctx, rung_diag, _voltage=voltage):
            nonlocal pc_mA_cm2_callback
            snapshot = dict(rung_diag)
            snapshot["lambda_hydrolysis"] = float(scale)
            snapshot["snes_converged"] = bool(ok)
            pc_mA_cm2: Optional[float] = None
            if ok:
                try:
                    pc_form = _build_bv_observable_form(
                        ctx, mode="peroxide_current",
                        reaction_index=None, scale=-I_SCALE,
                    )
                    pc_mA_cm2 = float(fd.assemble(pc_form))
                    pc_mA_cm2_callback = pc_mA_cm2
                except Exception as exc:                  # pragma: no cover
                    snapshot["callback_observable_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            try:
                augmented = augment_rung_diagnostics(
                    snapshot,
                    i_scale=I_SCALE,
                    i_lim_4e_mA_cm2=i_lim_4e_mA_cm2,
                    electrode_area_nondim=float(electrode_area_nondim),
                    domain_height_hat=domain_height_hat,
                    snes_converged=bool(ok),
                    gamma_picard_history=rung_diag.get(
                        "gamma_picard_history", []
                    ) or [],
                    pc_mA_cm2=pc_mA_cm2,
                )
                snapshot = augmented
            except Exception as exc:                       # pragma: no cover
                snapshot["augment_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            if ok:
                v_rungs.append(snapshot)
            else:
                v_partial_rungs.append(snapshot)

        # Try single-rung at λ=1 first (fast, ~10 s/V at "easy" V's
        # where the boundary state changes smoothly).  On
        # LadderExhausted (typical at cathodic V's where σ_S grows
        # fast and Γ_ss(V) jumps significantly), retry with the full
        # 5-rung ladder (~50 s/V) which gives Picard a controlled
        # ramp 0→1 from the warm-start.  Average per-V cost stays
        # near 10 s for ~80% of V's, ~50 s for the stiff cathodic
        # tail.
        ladder_used = "single_rung"
        v_result = None
        try:
            with adj.stop_annotating():
                v_result = solve_lambda_ramp_from_warm_start(
                    sp_at_v, mesh=mesh, U_warmstart=U_warm,
                    k0_targets=k0_targets,
                    lambda_hydrolysis_ladder=(1.0,),
                    parameter_overrides=overrides,
                    rung_callback=_v_rung_callback,
                    max_ss_steps_per_rung=300,
                )
            v_converged = bool(v_result.converged)
        except LadderExhausted as exc:
            single_rung_exc_str = str(exc)
            v_partial_rungs.append({
                "single_rung_ladder_exhausted": True,
                "exception": single_rung_exc_str,
            })
            progress(
                f"  V={voltage:+.3f}: single-rung failed "
                f"({single_rung_exc_str[:60]}) — falling back to "
                f"5-rung λ ladder."
            )
            # Reset captured rungs so the fallback's diagnostics are
            # the source of truth (avoid mixing failed single-rung
            # state into v_rungs).
            v_rungs.clear()
            ladder_used = "five_rung_fallback"
            try:
                with adj.stop_annotating():
                    v_result = solve_lambda_ramp_from_warm_start(
                        sp_at_v, mesh=mesh, U_warmstart=U_warm,
                        k0_targets=k0_targets,
                        lambda_hydrolysis_ladder=LAMBDA_LADDER,
                        parameter_overrides=overrides,
                        rung_callback=_v_rung_callback,
                        max_ss_steps_per_rung=300,
                    )
                v_converged = bool(v_result.converged)
            except LadderExhausted as exc2:
                v_converged = False
                v_partial_rungs.append({
                    "five_rung_ladder_exhausted": True,
                    "exception": str(exc2),
                })

        # Locate the converged λ=1.0 rung
        lam1_rung: Optional[Dict[str, Any]] = None
        for rung in v_rungs:
            if abs(float(rung.get("lambda_hydrolysis", -1)) - 1.0) < 1e-12:
                lam1_rung = rung
                break

        per_v_raw_lam1[orig_idx] = {
            "v_rhe": voltage,
            "neighbour_v": float(neighbour_v),
            "ladder_converged": v_converged,
            "ladder_used": ladder_used,
            "lambda1_rung": lam1_rung,
        }
        per_v_records[orig_idx] = _aug_per_v_record(
            lam1_rung, v_rhe=voltage,
        )

        # Cache U snapshot for downstream warm-starts
        if v_converged:
            converged_v_to_U[round(voltage, 9)] = snapshot_U(
                v_result.ctx["U"]
            )
        else:
            progress(
                f"  V={voltage:+.3f}: ramp at λ=1 failed (warm from "
                f"V={neighbour_v:+.3f})."
            )
    progress(
        f"  λ=1 walk done: "
        f"{sum(1 for r in per_v_records if r.get('snes_converged'))}/"
        f"{len(per_v_records)} V converged in {time.time() - t_walk:.1f}s"
    )

    # ---- Per-V HARD gate evaluation
    per_v_gate_results: List[Dict[str, Any]] = []
    n_gate_pass = 0
    n_gate_fail = 0
    for rec in per_v_records:
        passes, status = per_v_gate_status(rec)
        per_v_gate_results.append({
            "v_rhe": rec.get("v_rhe"),
            "passes": passes,
            "status": status,
        })
        if passes:
            n_gate_pass += 1
        else:
            n_gate_fail += 1

    # ---- Sign-guard at V_KIN_BYTE_EQUIV_BASELINE (-0.10 V)
    sign_guard_status: str = "not_evaluated"
    sign_guard_pka_shift: Optional[float] = None
    for rec in per_v_records:
        if (
            rec.get("v_rhe") is not None
            and abs(float(rec["v_rhe"]) - V_KIN_BYTE_EQUIV_BASELINE) < 1e-9
            and rec.get("pka_shift_avg") is not None
        ):
            sign_guard_pka_shift = float(rec["pka_shift_avg"])
            sign_guard_status = (
                "ok" if sign_guard_pka_shift < 0.0 else "violation"
            )
            break

    sigma_clamped_profile = [
        {
            "v_rhe": rec.get("v_rhe"),
            "sigma_local_clamped_counts_pm2": rec.get(
                "sigma_local_clamped_counts_pm2"
            ),
        }
        for rec in per_v_records
    ]
    sigma_clamped_max_in_window = max(
        (
            float(p["sigma_local_clamped_counts_pm2"])
            for p in sigma_clamped_profile
            if p["sigma_local_clamped_counts_pm2"] is not None
            and p.get("v_rhe") is not None
            and in_observable_mask(float(p["v_rhe"]))
        ),
        default=None,
    )

    aggregated = aggregate_observables(per_v_records)

    elapsed = time.time() - t_start
    return {
        "config": {
            "delta_beta_pm2": float(delta_beta_pm2),
            "sigma_mapping": sigma_mapping,
            "v_grid": [float(v) for v in v_grid],
            "v_anchor": float(v_anchor),
            "mode": mode,
            "topology": "warm_walk_at_lambda_1",
            "lambda_ladder_at_anchor": list(LAMBDA_LADDER),
            "k_hyd_baseline": K_HYD_BASELINE,
            "n_collection": N_COLLECTION,
            "observable_mask": [
                V_KIN_OBS_MASK_LO, V_KIN_OBS_MASK_HI,
            ],
            "v_kin_excluded_from_observables": V_KIN_BYTE_EQUIV_BASELINE,
            "k0_r4e_factor": float(K0_R4E_FACTOR_V10B),
            "anchor_cache": {
                "status": cache_status,
                "config_hash": config_hash,
                "path": cache_path,
            },
        },
        "anchor_lambda_ramp_rungs": anchor_lam_rungs,
        "per_v_records": per_v_records,
        "per_v_lam1_diagnostics": per_v_raw_lam1,
        "per_v_gate_results": per_v_gate_results,
        "n_gate_pass": n_gate_pass,
        "n_gate_fail": n_gate_fail,
        "sign_guard": {
            "status": sign_guard_status,
            "pka_shift_avg_at_V_kin": sign_guard_pka_shift,
        },
        "sigma_clamped_profile": sigma_clamped_profile,
        "sigma_clamped_max_in_window": sigma_clamped_max_in_window,
        "aggregated_observables": aggregated,
        "wall_seconds": elapsed,
        "mesh_dof_count": int(mesh_dof_count),
    }


def _build_sp_for_phase_D(
    *,
    sigma_mapping: str,
    delta_beta_pm2: float,
):
    """Construct SolverParams for the Phase D K-only V10B stack.

    Reuses the v10a' ``_build_sp`` (which already plumbs V10B kinetics +
    K2SO4 4sp + parallel-2e/4e + Stern + cation_hydrolysis with
    ``"singh_2016_eq_4"``) and overlays Phase-D-specific knobs:

    * ``beta_offset_pm2`` (initial value via cation_hydrolysis_config).
    * ``override_sigma_singh_counts_pm2`` for the ablation path.

    The resulting SolverParams runs at λ=0; the per-V λ ramp is driven
    by the orchestrator via ``solve_lambda_ramp_from_warm_start``.
    """
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import _build_sp
    sp = _build_sp(lambda_hydrolysis=0.0, k0_r4e_factor=1.0)

    # Overlay Phase-D-specific cation_hydrolysis_config knobs.
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts.get("bv_convergence", {}))
    new_cation = dict(new_bv.get("cation_hydrolysis_config", {}))
    new_cation["beta_offset_pm2"] = float(delta_beta_pm2)
    new_bv["cation_hydrolysis_config"] = new_cation

    if sigma_mapping == SIGMA_MAPPING_ABLATION:
        new_bv["override_sigma_singh_counts_pm2"] = float(
            ABLATION_SIGMA_SINGH_COUNTS_PM2
        )
    elif sigma_mapping == SIGMA_MAPPING_STERN:
        new_bv["override_sigma_singh_counts_pm2"] = None
    else:
        raise ValueError(f"unknown sigma_mapping: {sigma_mapping!r}")
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _make_mesh():
    """Phase D uses the v10a' mesh defaults (NX=8, NY=80, β=3.0,
    L_eff=16 µm).  All Phase D forward evals share a single mesh."""
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import _make_mesh
    return _make_mesh()


def _per_v_lambda_ramp(
    *,
    sp_template,
    mesh,
    voltage: float,
    U_warmstart: tuple,
    delta_beta_pm2: float,
    i_scale: float,
    i_lim_4e_mA_cm2: float,
    electrode_area_nondim: float,
    domain_height_hat: float,
) -> Dict[str, Any]:
    """Run the 5-rung λ ladder for a single V grid point.

    Returns a dict with the λ=1 rung diagnostics PLUS aggregated
    per-V observables (cd_mA_cm2, gross_h2o2_current_mA_cm2, etc.).
    Wraps ``solve_lambda_ramp_from_warm_start`` with the Phase D
    parameter-overrides bundle (k_hyd baseline + Δ_β offset) and reuses
    :func:`scripts.studies.phase6b_v10a_phase_A2_v_kin.augment_rung_diagnostics`
    so the rung dict carries ``mass_balance_residual_rel``,
    ``picard_status``, ``cd_mA_cm2``, ``x_2e``, etc. (the same fields
    the A.2 V_kin baseline emits — required for the HARD reproduction
    comparison and the per-V gate evaluation).
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted, solve_lambda_ramp_from_warm_start,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import K0_HAT_R2E, K0_HAT_R4E, V_T
    from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
        augment_rung_diagnostics,
    )

    sp_at_v = sp_template.with_phi_applied(voltage / V_T)
    overrides = {
        "k_hyd": float(K_HYD_BASELINE),
        "beta_offset_pm2": float(delta_beta_pm2),
    }
    k0_r4e_target = float(K0_HAT_R4E) * float(K0_R4E_FACTOR_V10B)

    augmented_rungs: List[Dict[str, Any]] = []
    partial_rungs: List[Dict[str, Any]] = []

    def _rung_callback(scale, ok, ctx, rung_diag):
        snapshot = dict(rung_diag)
        snapshot["lambda_hydrolysis"] = float(scale)
        snapshot["snes_converged"] = bool(ok)
        # Augment with peroxide current at every converged rung so the
        # final λ=1 record carries the gross H₂O₂ disk-side current.
        pc_mA_cm2: Optional[float] = None
        if ok:
            try:
                pc_form = _build_bv_observable_form(
                    ctx, mode="peroxide_current",
                    reaction_index=None, scale=-i_scale,
                )
                pc_mA_cm2 = float(fd.assemble(pc_form))
            except Exception as exc:                  # pragma: no cover
                snapshot["callback_observable_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        # Pull in A.2-style augmented fields (mass_balance, cd_mA_cm2,
        # x_2e, picard_status, ...) so the per-V gate has all required
        # diagnostics + the HARD reproduction comparison sees the same
        # field schema.
        try:
            augmented = augment_rung_diagnostics(
                snapshot,
                i_scale=i_scale,
                i_lim_4e_mA_cm2=i_lim_4e_mA_cm2,
                electrode_area_nondim=electrode_area_nondim,
                domain_height_hat=domain_height_hat,
                snes_converged=bool(ok),
                gamma_picard_history=rung_diag.get(
                    "gamma_picard_history", []
                ) or [],
                pc_mA_cm2=pc_mA_cm2,
            )
            snapshot = augmented
        except Exception as exc:                       # pragma: no cover
            snapshot["augment_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        if ok:
            augmented_rungs.append(snapshot)
        else:
            partial_rungs.append(snapshot)

    exception_phase: Optional[str] = None
    exception_str: Optional[str] = None
    try:
        with adj.stop_annotating():
            result = solve_lambda_ramp_from_warm_start(
                sp_at_v, mesh=mesh, U_warmstart=U_warmstart,
                k0_targets={0: float(K0_HAT_R2E), 1: k0_r4e_target},
                lambda_hydrolysis_ladder=LAMBDA_LADDER,
                parameter_overrides=overrides,
                rung_callback=_rung_callback,
                max_ss_steps_per_rung=300,
            )
        ladder_converged = bool(result.converged)
    except LadderExhausted as exc:
        ladder_converged = False
        exception_str = str(exc)
        exception_phase = "lambda_ramp_exhausted"

    # Locate the converged λ=1.0 rung.
    lam1: Optional[Dict[str, Any]] = None
    for rung in augmented_rungs:
        if abs(float(rung.get("lambda_hydrolysis", -1)) - 1.0) < 1e-12:
            lam1 = rung
            break
    return {
        "v_rhe": float(voltage),
        "ladder_converged": ladder_converged,
        "exception_phase": exception_phase,
        "exception": exception_str,
        "rungs": augmented_rungs,
        "partial_rungs": partial_rungs,
        "lambda1_rung": lam1,
    }


def _aug_per_v_record(
    lam1: Optional[Dict[str, Any]],
    *,
    v_rhe: float,
    n_collection: float = N_COLLECTION,
) -> Dict[str, Any]:
    """Build the per-V observable record from the λ=1 rung diagnostics.

    Phase D per-V emission spec from plan §D2.
    """
    from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir

    rec: Dict[str, Any] = {"v_rhe": float(v_rhe)}
    if lam1 is None:
        rec["snes_converged"] = False
        rec["picard_status"] = "missing"
        return rec

    rec["snes_converged"] = bool(lam1.get("snes_converged", False))
    rec["picard_status"] = lam1.get("picard_status")
    cd = lam1.get("cd_mA_cm2") or lam1.get("cd_mA_cm2_callback")
    pc = lam1.get("pc_mA_cm2")
    rec["cd_mA_cm2"] = cd
    rec["pc_mA_cm2"] = pc
    rec["R_2e_current_nondim"] = lam1.get("R_2e_current_nondim")
    rec["R_4e_current_nondim"] = lam1.get("R_4e_current_nondim")
    # Phase D treats peroxide current as the "gross H₂O₂" ring-basis
    # current (positive: disk-side cathodic peroxide producton converted
    # to ring current via N).
    rec["gross_h2o2_current_mA_cm2"] = pc
    rec["ring_current_ring_basis_mA_cm2"] = (
        None if pc is None else float(pc) * float(n_collection)
    )
    rec["theta"] = lam1.get("theta")
    rec["gamma_final"] = lam1.get("gamma_final")
    rec["sigma_S_C_per_m2"] = lam1.get("sigma_S_C_per_m2")
    rec["sigma_local_clamped_counts_pm2"] = (
        cathodic_clamped_sigma_singh(lam1["sigma_S_C_per_m2"])
        if lam1.get("sigma_S_C_per_m2") is not None else None
    )
    rec["pka_shift_avg"] = lam1.get("pka_shift_avg")
    rec["mass_balance_residual_rel"] = lam1.get(
        "mass_balance_residual_rel"
    )
    # Analytic Γ vs reported Γ (consistency with closed-form Γ_ss).
    f0_avg = lam1.get("forward_avg_no_k_hyd")
    c_h_avg = lam1.get("c_H_avg")
    gamma_max = lam1.get("gamma_max")
    k_hyd = lam1.get("k_hyd")
    k_prot = lam1.get("k_prot")
    k_des = lam1.get("k_des")
    delta_ohp = lam1.get("delta_ohp_hat")
    gamma_reported = lam1.get("gamma_final")
    if all(
        v is not None for v in (
            f0_avg, c_h_avg, gamma_max, k_hyd, k_prot, k_des,
            delta_ohp, gamma_reported,
        )
    ):
        try:
            gamma_predicted, _, _ = gamma_ss_langmuir(
                lambda_val=1.0,
                k_hyd=float(k_hyd),
                k_prot=float(k_prot),
                k_des=float(k_des),
                delta_ohp=float(delta_ohp),
                forward_avg=float(f0_avg),
                c_H_avg=float(c_h_avg),
                gamma_max=float(gamma_max),
            )
            denom = max(abs(float(gamma_reported)), 1e-30)
            rec["analytic_gamma_rel"] = (
                abs(gamma_predicted - float(gamma_reported)) / denom
            )
        except Exception as exc:                       # pragma: no cover
            rec["analytic_gamma_rel"] = None
            rec["analytic_gamma_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
    else:
        rec["analytic_gamma_rel"] = None
    return rec


def evaluate_delta_beta(
    *,
    delta_beta_pm2: float,
    sigma_mapping: str = SIGMA_MAPPING_STERN,
    v_grid: Sequence[float] = V_RHE_PRODUCTION_GRID,
    v_anchor: float = V_ANCHOR,
    mode: str = "production",
    progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Per-eval forward driver: returns the JSON-ready result dict.

    Parameters
    ----------
    delta_beta_pm2
        Phase D Δ_β value to plumb into the residual.
    sigma_mapping
        Plan §3.3 σ-mapping enum.
    v_grid
        Ordered V_RHE points to solve (excluding the anchor).  Default
        is the locked production 24-point grid; the A.2 reproduction
        mode passes ``A2_WARM_GRID``.
    v_anchor
        Anchor voltage (default +0.55 V).
    mode
        Continuation topology:

        * ``"production"`` (NEW default, optimized) — anchor at
          ``v_anchor`` with λ=0 + two-stage C_S, λ ramp 0→1 at
          ``v_anchor``, then warm-walk at λ=1 through ``v_grid``
          (single-Picard at λ=1 per V).  ~3-5x faster than the legacy
          per-V-ramp topology, validated against the legacy path
          byte-perfectly via 10.B.0 a2_reproduction.
        * ``"production_legacy"`` — anchor + λ=0 walk through ``v_grid``
          + per-V λ ramp at each V.  Slower; kept for sanity-check
          and continuation-path-drift diagnostics.
        * ``"a2_reproduction"`` — 5-pt A.2 warm grid + V_kin λ ramp
          ONLY (other warm-grid V's not ramped).  Used for the D5(a)
          HARD byte-equivalence check vs the v10b A.2 V_kin baseline.
    progress
        Called with status strings between phases.

    Returns
    -------
    dict
        JSON-serialisable result with config, per-V records,
        aggregated observables, gate status, σ_local_clamped profile.
    """
    # Phase D production topology (after the optimization-A regression
    # was reverted on 2026-05-11): walk V at λ=0 with RELAXED warm-walk
    # tolerances (n_substeps=4, max_ss=60, ss_rel_tol=1e-3 — vs the
    # solve_grid_with_anchor defaults of 8/150/1e-4).  At λ=0 the
    # cation-hydrolysis source is byte-zeroed so the residual is much
    # smoother than at λ=1, and small V steps (0.05 V on the production
    # grid) converge under the relaxed settings ~5x faster than the
    # defaults.  Then per-V single-rung Picard at λ=1 (~10 s/V).
    # ``a2_reproduction`` mode keeps the unrelaxed defaults (passes
    # ``walk_*=None``) so the D5(a) HARD reproduction stays
    # byte-equivalent to the v10b A.2 reference.
    import time

    t_start = time.time()

    sp = _build_sp_for_phase_D(
        sigma_mapping=sigma_mapping,
        delta_beta_pm2=delta_beta_pm2,
    )
    mesh = _make_mesh()

    # Pass 1 — anchor + λ=0 warm-walk through v_grid.
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        _walk_lambda_zero_capture_snapshots,
    )
    if mode == "production":
        walk_n_substeps: Optional[int] = 4
        walk_max_ss_steps: Optional[int] = 60
        walk_ss_rel_tol: Optional[float] = 1e-3
    else:
        walk_n_substeps = None
        walk_max_ss_steps = None
        walk_ss_rel_tol = None
    progress(
        f"[Δ_β={delta_beta_pm2:+.6g}, σ={sigma_mapping}] Pass 1: anchor + "
        f"warm-walk over {len(v_grid)} V points at λ=0 "
        f"(walk_n_substeps={walk_n_substeps}, "
        f"walk_max_ss={walk_max_ss_steps}, "
        f"walk_ss_rel_tol={walk_ss_rel_tol}) ..."
    )
    walk_records, snapshots, mesh_dof, electrode_area, electrode_marker = (
        _walk_lambda_zero_capture_snapshots(
            sp=sp, mesh=mesh,
            v_rhe_grid=tuple(float(v) for v in v_grid),
            v_anchor=float(v_anchor),
            k0_r4e_factor=float(K0_R4E_FACTOR_V10B),
            walk_n_substeps=walk_n_substeps,
            walk_max_ss_steps=walk_max_ss_steps,
            walk_ss_rel_tol=walk_ss_rel_tol,
        )
    )
    progress(
        f"  warm-walk done: {sum(1 for r in walk_records if r.get('lambda_zero_converged'))}/"
        f"{len(walk_records)} converged in {time.time() - t_start:.1f}s"
    )

    # Auxiliary scales for augment_rung_diagnostics.
    from scripts._bv_common import I_SCALE, L_REF
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        L_EFF_M_BASELINE, _i_lim_4e_mA_cm2,
    )

    domain_height_hat = float(L_EFF_M_BASELINE) / float(L_REF)
    i_lim_4e_mA_cm2 = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)

    # Pass 2 — per-V λ ramp from each warm-walk snapshot to λ=1.
    # a2_reproduction mode restricts the ramp to V_kin only (the
    # other warm-grid V's are intermediate steps for the warm-walk
    # only; the A.2 baseline run also only ramped at V_kin).
    per_v_records: List[Dict[str, Any]] = []
    per_v_raw_lam1: List[Dict[str, Any]] = []
    n_v_failed = 0
    t_pass2 = time.time()
    if mode == "a2_reproduction":
        ramp_v_set = (V_KIN_BYTE_EQUIV_BASELINE,)
    else:
        ramp_v_set = tuple(float(v) for v in v_grid)
    for grid_idx, voltage in enumerate(v_grid):
        in_ramp_set = any(
            abs(float(voltage) - float(rv)) < 1e-9 for rv in ramp_v_set
        )
        if not in_ramp_set:
            # Skip λ ramp at warm-walk-only V's (a2_reproduction mode).
            # Record a placeholder so per_v_records stays grid-aligned.
            per_v_records.append({
                "v_rhe": float(voltage),
                "snes_converged": False,
                "picard_status": "lambda_ramp_not_run",
                "lambda_ramp_skipped": True,
                "skip_reason": "warm_walk_only_in_a2_mode",
            })
            per_v_raw_lam1.append({
                "v_rhe": float(voltage), "lambda1_rung": None,
            })
            continue
        if grid_idx not in snapshots:
            progress(
                f"  V={voltage:+.3f}: λ=0 warm-walk failed — recording "
                f"per-V gate fail (newton_unconverged)."
            )
            per_v_records.append({
                "v_rhe": float(voltage),
                "snes_converged": False,
                "picard_status": "lambda_zero_unconverged",
                "lambda_ramp_skipped": True,
            })
            per_v_raw_lam1.append({
                "v_rhe": float(voltage), "lambda1_rung": None,
            })
            n_v_failed += 1
            continue
        ramp_result = _per_v_lambda_ramp(
            sp_template=sp, mesh=mesh, voltage=float(voltage),
            U_warmstart=snapshots[grid_idx],
            delta_beta_pm2=float(delta_beta_pm2),
            i_scale=I_SCALE,
            i_lim_4e_mA_cm2=i_lim_4e_mA_cm2,
            electrode_area_nondim=float(electrode_area),
            domain_height_hat=domain_height_hat,
        )
        per_v_raw_lam1.append({
            "v_rhe": float(voltage),
            "ladder_converged": ramp_result["ladder_converged"],
            "exception_phase": ramp_result["exception_phase"],
            "lambda1_rung": ramp_result["lambda1_rung"],
        })
        rec = _aug_per_v_record(
            ramp_result["lambda1_rung"], v_rhe=voltage,
        )
        per_v_records.append(rec)
        if not rec.get("snes_converged", False):
            n_v_failed += 1
    t_pass2_done = time.time()
    progress(
        f"  λ-ramp done: {len(per_v_records) - n_v_failed}/"
        f"{len(per_v_records)} V points converged in "
        f"{t_pass2_done - t_pass2:.1f}s"
    )

    # Per-V HARD gate evaluation
    per_v_gate_results: List[Dict[str, Any]] = []
    n_gate_pass = 0
    n_gate_fail = 0
    for rec in per_v_records:
        passes, status = per_v_gate_status(rec)
        per_v_gate_results.append({
            "v_rhe": rec.get("v_rhe"),
            "passes": passes,
            "status": status,
        })
        if passes:
            n_gate_pass += 1
        else:
            n_gate_fail += 1

    # Sign-guard at V_KIN_BYTE_EQUIV_BASELINE = -0.10 (V_KIN).
    sign_guard_status: str = "not_evaluated"
    sign_guard_pka_shift: Optional[float] = None
    for rec in per_v_records:
        if (
            rec.get("v_rhe") is not None
            and abs(float(rec["v_rhe"]) - V_KIN_BYTE_EQUIV_BASELINE) < 1e-9
            and rec.get("pka_shift_avg") is not None
        ):
            sign_guard_pka_shift = float(rec["pka_shift_avg"])
            sign_guard_status = (
                "ok" if sign_guard_pka_shift < 0.0 else "violation"
            )
            break

    # σ_local_clamped profile (per-V, mask-applied)
    sigma_clamped_profile = [
        {
            "v_rhe": rec.get("v_rhe"),
            "sigma_local_clamped_counts_pm2": rec.get(
                "sigma_local_clamped_counts_pm2"
            ),
        }
        for rec in per_v_records
    ]
    sigma_clamped_max_in_window = max(
        (
            float(p["sigma_local_clamped_counts_pm2"])
            for p in sigma_clamped_profile
            if p["sigma_local_clamped_counts_pm2"] is not None
            and p.get("v_rhe") is not None
            and in_observable_mask(float(p["v_rhe"]))
        ),
        default=None,
    )

    # Aggregated observables (mask-applied)
    aggregated = aggregate_observables(per_v_records)

    elapsed = time.time() - t_start
    out: Dict[str, Any] = {
        "config": {
            "delta_beta_pm2": float(delta_beta_pm2),
            "sigma_mapping": sigma_mapping,
            "v_grid": [float(v) for v in v_grid],
            "v_anchor": float(v_anchor),
            "mode": mode,
            "lambda_ladder": list(LAMBDA_LADDER),
            "k_hyd_baseline": K_HYD_BASELINE,
            "n_collection": N_COLLECTION,
            "observable_mask": [
                V_KIN_OBS_MASK_LO, V_KIN_OBS_MASK_HI,
            ],
            "v_kin_excluded_from_observables": V_KIN_BYTE_EQUIV_BASELINE,
        },
        "warm_walk_records": walk_records,
        "per_v_records": per_v_records,
        "per_v_lam1_diagnostics": per_v_raw_lam1,
        "per_v_gate_results": per_v_gate_results,
        "n_gate_pass": n_gate_pass,
        "n_gate_fail": n_gate_fail,
        "sign_guard": {
            "status": sign_guard_status,
            "pka_shift_avg_at_V_kin": sign_guard_pka_shift,
        },
        "sigma_clamped_profile": sigma_clamped_profile,
        "sigma_clamped_max_in_window": sigma_clamped_max_in_window,
        "aggregated_observables": aggregated,
        "wall_seconds": elapsed,
        "mesh_dof_count": int(mesh_dof),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: parse args, run one Δ_β evaluation, write JSON."""
    import json
    import time

    args = _parse_args(argv)
    out_dir = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_name = args.out_name or _default_out_name(
        args.delta_beta, args.sigma_mapping,
    )
    out_path = os.path.join(out_dir, out_name)
    print(f"Output: {out_path}", flush=True)
    print(
        f"Δ_β = {args.delta_beta:+.6g} pm², "
        f"σ-mapping = {args.sigma_mapping}, mode = {args.mode}",
        flush=True,
    )

    if args.mode == "a2_reproduction":
        v_grid = A2_WARM_GRID
    else:
        v_grid = V_RHE_PRODUCTION_GRID

    t0 = time.time()
    result = evaluate_delta_beta(
        delta_beta_pm2=float(args.delta_beta),
        sigma_mapping=args.sigma_mapping,
        v_grid=v_grid,
        v_anchor=V_ANCHOR,
        mode=args.mode,
    )
    print(f"Eval done in {time.time() - t0:.1f}s", flush=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_path}", flush=True)

    if args.plot:
        try:
            _make_plot(result, out_path.replace(".json", ".png"))
            print(f"Wrote {out_path.replace('.json', '.png')}", flush=True)
        except Exception as exc:
            print(
                f"plot generation failed (non-fatal): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    agg = result["aggregated_observables"]
    print(
        f"Aggregated: max_H2O2% = "
        f"{agg.get('max_H2O2_selectivity_in_window_pct')}, "
        f"argmax V = {agg.get('argmax_V_for_selectivity')}, "
        f"ring_onset = {agg.get('ring_onset_V_at_0.01_mA_cm2')}, "
        f"max_ring = {agg.get('max_ring_current_in_window_mA_cm2')}, "
        f"n_e@argmax = {agg.get('n_e_rrde_at_argmax_V')}",
        flush=True,
    )
    print(
        f"Gates: {result['n_gate_pass']} pass / "
        f"{result['n_gate_fail']} fail; "
        f"sign_guard = {result['sign_guard']['status']}",
        flush=True,
    )
    return 0


def _make_plot(result: Dict[str, Any], png_path: str) -> None:
    """Emit a 4-panel summary PNG: cd vs V, gross_H2O2 vs V,
    selectivity vs V, σ_local_clamped vs V.  Non-blocking on plot
    failures (caller catches)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = result["per_v_records"]
    v = [float(r["v_rhe"]) for r in recs if r.get("v_rhe") is not None]
    cd = [r.get("cd_mA_cm2") for r in recs if r.get("v_rhe") is not None]
    gh2o2 = [
        r.get("gross_h2o2_current_mA_cm2")
        for r in recs if r.get("v_rhe") is not None
    ]
    sigma_clamped = [
        r.get("sigma_local_clamped_counts_pm2")
        for r in recs if r.get("v_rhe") is not None
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    axes[0].plot(v, cd, "o-")
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].set_xlabel("V_RHE (V)")
    axes[0].set_ylabel("Disk current density (mA/cm²)")
    axes[0].set_title("cd vs V")
    axes[1].plot(v, gh2o2, "o-")
    axes[1].set_xlabel("V_RHE (V)")
    axes[1].set_ylabel("gross H2O2 current (mA/cm²)")
    axes[1].set_title("gross H2O2 vs V")
    axes[2].plot(v, sigma_clamped, "o-")
    axes[2].set_xlabel("V_RHE (V)")
    axes[2].set_ylabel("σ_local_clamped (counts/pm²)")
    axes[2].set_title("σ profile")
    sel = []
    for cdi, ghi in zip(cd, gh2o2):
        if cdi is None or ghi is None:
            sel.append(None)
        else:
            sel.append(selectivity_h2o2_pct(
                i_disk_mA_cm2=cdi, i_ring_mA_cm2=ghi,
            ))
    axes[3].plot(v, sel, "o-")
    axes[3].axvspan(
        V_KIN_OBS_MASK_LO, V_KIN_OBS_MASK_HI, alpha=0.1, color="green",
        label="observable mask",
    )
    axes[3].set_xlabel("V_RHE (V)")
    axes[3].set_ylabel("H2O2 selectivity (%)")
    axes[3].set_title("Selectivity vs V")
    axes[3].legend(fontsize=8)
    cfg = result["config"]
    fig.suptitle(
        f"Phase D eval: Δ_β = {cfg['delta_beta_pm2']:+.6g} pm², "
        f"σ-mapping = {cfg['sigma_mapping']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
