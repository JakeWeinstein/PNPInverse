"""Phase 6β step 10 Phase D — orchestrator (plan §10.B.0–10.B.9).

Drives the per-eval driver
:mod:`scripts.studies.phase6b_step10_phase_D_fit_eval` through:

* 10.B.0 — A.2-compatible HARD reproduction baseline.
* 10.B.1 — Stern Δ_β=0 production-grid baseline (+ 3 noise-floor dup).
* 10.B.2 — Ablation Δ_β=0 production-grid baseline.
* 10.B.3 — Stern pre-fit grid (7 additional Δ_β).
* 10.B.4 — Ablation pre-fit grid (6 additional Δ_β).
* 10.B.5 — D7 4-criterion identifiability gate.
* 10.B.6 — Stern Brent (`scipy.optimize.minimize_scalar`).
* 10.B.7 — Ablation Brent.
* 10.B.8 — σ-mapping divergence flag.
* 10.B.9 — Outcome verdict (A_LOCKED_PASS / B_FALSIFIED /
  C_NON_IDENTIFIABLE) → emit phase_E_spec / falsification_report /
  identifiability_report JSON per D9.

Resumable: each Δ_β evaluation is cached as a per-eval JSON in
``StudyResults/<out-subdir>/``.  Re-running the orchestrator skips
already-completed evals.

CLI::

    python -u scripts/studies/phase6b_step10_phase_D_orchestrate.py \\
        [--phases 10.B.0,10.B.1,...] [--out-subdir phase6b_step10_phase_D] \\
        [--skip-a2-reproduction] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Locked deck targets (from 10.A.0 data audit pin)
# ---------------------------------------------------------------------------

DECK_TARGET_AUDIT_PATH: str = (
    "StudyResults/phase6b_step10_phase_D/data_audit_K_at_pH4.json"
)


def _load_deck_target() -> Dict[str, Any]:
    """Read the locked deck target pin from 10.A.0."""
    p = os.path.join(_ROOT, DECK_TARGET_AUDIT_PATH)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"deck target pin missing: {p} — re-run 10.A.0 data audit"
        )
    return json.loads(open(p).read())


# ---------------------------------------------------------------------------
# Plan §D6 / D7 / D8 constants
# ---------------------------------------------------------------------------

PRIMARY_GATE_PP_ABSOLUTE: float = 10.0
"""Plan §3.2 / §D8: ±10 pp absolute primary gate (locked, NOT adaptive)."""

D7_LOSS_RANGE_MIN_PP2: float = 1.0
"""Plan §D7 #1: Δ_loss ≥ 1 pp² across loss_finite_valid."""

D7_NOISE_DUP_COUNT: int = 3
"""Plan §D7 #2: 3 dup Δ_β=0 evals (Stern path)."""

D7_NOISE_FLOOR_MULTIPLIER: float = 3.0
"""Plan §D7 #2: Δ_loss ≥ 3·noise_std."""

D7_SLOPE_MIN_PP2_PER_DPKA: float = 0.01
"""Plan §D7 #3: max |d(L)/d(target_ΔpKa)| ≥ 0.01 pp²/ΔpKa unit."""

SIGMA_DIVERGENCE_THRESHOLD_REL: float = 0.30
"""Plan §3.3 / §D8: bundle-locked rel divergence threshold."""

XATOL_TARGET_DPKA: float = 0.05
"""Plan §D4: optimizer tolerance in target-ΔpKa-effect space."""

BRENT_MAXITER: int = 16
"""Plan §D4."""

EPS_T_MARGIN: float = 1e-6
"""Plan §D4: ΔpKa-effect margin from sign-flip."""

T_TARGETS_STERN: Tuple[float, ...] = (
    -5.0, -3.0, -1.0, -0.1, -0.01, -0.001, -1e-4,
)
"""Plan §D4 Stern pre-fit targets in ΔpKa-effect space."""

T_TARGETS_ABLATION: Tuple[float, ...] = (
    -14.9, -10.0, -8.0, -4.0, -1.0, -0.1,
)
"""Plan §D4 Ablation pre-fit targets in ΔpKa-effect space."""

T_LOWER_STERN: float = -5.0
T_UPPER_STERN: float = -EPS_T_MARGIN
T_LOWER_ABLATION: float = -14.9
T_UPPER_ABLATION: float = -EPS_T_MARGIN

ABLATION_SIGMA: float = 0.141
"""Plan §D4 Singh K-Cu cell-level σ (V-independent)."""


# ---------------------------------------------------------------------------
# Pure-Python orchestration helpers
# ---------------------------------------------------------------------------


def stern_delta_beta_for_target(
    *, target_dpka: float, beta_K_Cu: float, sigma_max: float,
) -> float:
    """Convert ΔpKa-effect target → Δ_β (pm²) for the Stern path.

    From Δ_β = (T - β_K_Cu·σ_max) / σ_max where T is the target
    ΔpKa-effect, β_K_Cu the Cu prior, σ_max the max σ over the V grid.
    """
    if sigma_max == 0.0:
        raise ZeroDivisionError("sigma_max must be > 0 for Stern bracket")
    return (float(target_dpka) - float(beta_K_Cu) * float(sigma_max)) / float(
        sigma_max
    )


def ablation_delta_beta_for_target(
    *, target_dpka: float, beta_K_Cu: float, sigma: float = ABLATION_SIGMA,
) -> float:
    """Convert ΔpKa-effect target → Δ_β (pm²) for the Ablation path."""
    if sigma == 0.0:
        raise ZeroDivisionError("sigma must be > 0 for Ablation bracket")
    return (float(target_dpka) - float(beta_K_Cu) * float(sigma)) / float(
        sigma
    )


def loss_from_eval(
    eval_result: Dict[str, Any], deck_target_pct: float,
) -> Tuple[float, str]:
    """Compute (loss, status) from a per-eval JSON.

    status ∈ {"finite_valid", "solve_failed", "sign_guard_violation",
              "pka_shift_overflow", "no_selectivity"}.

    Plan §D6 bookkeeping: any HARD per-V gate fail (n_gate_fail > 0)
    invalidates the entire eval (loss = inf, status = solve_failed).
    """
    n_gate_fail = int(eval_result.get("n_gate_fail", 0))
    sign_guard = (
        eval_result.get("sign_guard", {}).get("status", "not_evaluated")
    )
    agg = eval_result.get("aggregated_observables", {})
    sel = agg.get("max_H2O2_selectivity_in_window_pct")

    if sign_guard == "violation":
        return float("inf"), "sign_guard_violation"
    if n_gate_fail > 0:
        # Check whether ANY V-fail was a pka_shift_overflow (Phase D-
        # specific signal): the candidate exited the safe ΔpKa domain.
        for g in eval_result.get("per_v_gate_results", []):
            if g.get("status") == "pka_shift_overflow":
                return float("inf"), "pka_shift_overflow"
        return float("inf"), "solve_failed"
    if sel is None:
        return float("inf"), "no_selectivity"
    return abs(float(sel) - float(deck_target_pct)), "finite_valid"


def evaluate_one_cached(
    *,
    delta_beta_pm2: float,
    sigma_mapping: str,
    deck_target_pct: float,
    out_dir: str,
    out_name: Optional[str] = None,
    progress: Callable[[str], None] = print,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one Δ_β evaluation, reusing an existing JSON if present.

    Returns the bookkeeping dict suitable for ``loss_all``:
    ``{"delta_beta_pm2", "sigma_mapping", "loss", "status",
       "eval_path", "from_cache", "wall_seconds"}``.
    """
    from scripts.studies.phase6b_step10_phase_D_fit_eval import (
        evaluate_delta_beta,
        _default_out_name,
    )

    if out_name is None:
        out_name = _default_out_name(delta_beta_pm2, sigma_mapping)
    out_path = os.path.join(out_dir, out_name)

    cached: Optional[Dict[str, Any]] = None
    if os.path.exists(out_path):
        try:
            cached = json.loads(open(out_path).read())
        except Exception as exc:
            progress(
                f"  cached JSON unreadable ({type(exc).__name__}: {exc}) "
                f"-- will re-run"
            )

    if cached is not None:
        loss, status = loss_from_eval(cached, deck_target_pct)
        progress(
            f"  cached eval (Δ_β={delta_beta_pm2:+.6g} σ={sigma_mapping}): "
            f"loss={loss!r}, status={status}"
        )
        return {
            "delta_beta_pm2": float(delta_beta_pm2),
            "sigma_mapping": sigma_mapping,
            "loss": float(loss),
            "status": status,
            "eval_path": out_path,
            "from_cache": True,
            "wall_seconds": float(cached.get("wall_seconds", 0.0)),
        }

    if dry_run:
        progress(
            f"  [DRY RUN] would evaluate Δ_β={delta_beta_pm2:+.6g} "
            f"σ={sigma_mapping}; out_path={out_path}"
        )
        return {
            "delta_beta_pm2": float(delta_beta_pm2),
            "sigma_mapping": sigma_mapping,
            "loss": None,
            "status": "dry_run_skipped",
            "eval_path": out_path,
            "from_cache": False,
            "wall_seconds": 0.0,
        }

    progress(
        f"  evaluating Δ_β={delta_beta_pm2:+.6g} σ={sigma_mapping} ..."
    )
    t0 = time.time()
    result = evaluate_delta_beta(
        delta_beta_pm2=float(delta_beta_pm2),
        sigma_mapping=sigma_mapping,
    )
    elapsed = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    loss, status = loss_from_eval(result, deck_target_pct)
    progress(
        f"    done in {elapsed:.1f}s; loss={loss!r}, status={status}"
    )
    return {
        "delta_beta_pm2": float(delta_beta_pm2),
        "sigma_mapping": sigma_mapping,
        "loss": float(loss),
        "status": status,
        "eval_path": out_path,
        "from_cache": False,
        "wall_seconds": elapsed,
    }


def filter_finite_valid(
    loss_all: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Plan §D6: project ``loss_all`` to entries with finite_valid status."""
    return [e for e in loss_all if e.get("status") == "finite_valid"]


def d7_identifiability_gate(
    loss_finite_valid: Sequence[Dict[str, Any]],
    *,
    noise_std: float,
    sigma_max: float,
    beta_K_Cu: float,
) -> Dict[str, Any]:
    """Plan §D7 — 4-criterion identifiability gate.

    Returns dict with per-criterion verdict + ``overall_pass``.
    """
    losses = [float(e["loss"]) for e in loss_finite_valid]
    out: Dict[str, Any] = {
        "n_finite_valid": len(loss_finite_valid),
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "delta_loss_pp2": (
            (max(losses) - min(losses)) if len(losses) >= 2 else None
        ),
        "noise_std_pp2": float(noise_std),
        "noise_floor_threshold": (
            float(noise_std) * D7_NOISE_FLOOR_MULTIPLIER
        ),
        "criteria": {},
        "overall_pass": False,
    }

    if len(losses) < 2:
        out["criteria"]["range"] = {
            "passes": False, "note": "n_finite_valid < 2",
        }
        out["criteria"]["noise_floor"] = {
            "passes": False, "note": "n_finite_valid < 2",
        }
        out["criteria"]["slope"] = {
            "passes": False, "note": "n_finite_valid < 2",
        }
        out["criteria"]["unimodality"] = {
            "passes": False, "note": "n_finite_valid < 2",
        }
        return out

    delta_loss = max(losses) - min(losses)
    range_pass = delta_loss >= D7_LOSS_RANGE_MIN_PP2
    out["criteria"]["range"] = {
        "passes": range_pass,
        "delta_loss_pp2": delta_loss,
        "threshold_pp2": D7_LOSS_RANGE_MIN_PP2,
    }

    noise_pass = delta_loss >= float(noise_std) * D7_NOISE_FLOOR_MULTIPLIER
    out["criteria"]["noise_floor"] = {
        "passes": noise_pass,
        "delta_loss_pp2": delta_loss,
        "noise_threshold": float(noise_std) * D7_NOISE_FLOOR_MULTIPLIER,
    }

    # Slope in target-ΔpKa space.  For each (db, loss) compute target_dpka
    # = (β_K_Cu + db) · σ_max.  Sort by target_dpka and finite-diff slopes.
    pairs = []
    for e in loss_finite_valid:
        db = float(e["delta_beta_pm2"])
        target_dpka = (float(beta_K_Cu) + db) * float(sigma_max)
        pairs.append((target_dpka, float(e["loss"])))
    pairs.sort(key=lambda t: t[0])
    slopes: List[float] = []
    for i in range(1, len(pairs)):
        dx = pairs[i][0] - pairs[i - 1][0]
        if abs(dx) < 1e-30:
            continue
        slopes.append((pairs[i][1] - pairs[i - 1][1]) / dx)
    if not slopes:
        out["criteria"]["slope"] = {
            "passes": False, "note": "no slope pairs",
        }
    else:
        max_abs_slope = max(abs(s) for s in slopes)
        slope_pass = max_abs_slope >= D7_SLOPE_MIN_PP2_PER_DPKA
        out["criteria"]["slope"] = {
            "passes": slope_pass,
            "max_abs_slope_pp2_per_dpka": max_abs_slope,
            "threshold": D7_SLOPE_MIN_PP2_PER_DPKA,
        }

    # Unimodality — at most one interior local minimum.
    losses_sorted = [p[1] for p in pairs]
    minima = 0
    for i in range(1, len(losses_sorted) - 1):
        if (
            losses_sorted[i] < losses_sorted[i - 1]
            and losses_sorted[i] < losses_sorted[i + 1]
        ):
            minima += 1
    uni_pass = minima <= 1
    out["criteria"]["unimodality"] = {
        "passes": uni_pass,
        "interior_minima_count": minima,
    }

    out["overall_pass"] = bool(
        out["criteria"]["range"]["passes"]
        and out["criteria"]["noise_floor"]["passes"]
        and out["criteria"]["slope"]["passes"]
        and out["criteria"]["unimodality"]["passes"]
    )
    return out


def sigma_mapping_divergence(
    *,
    delta_beta_fit_stern: float,
    delta_beta_fit_ablation: float,
    threshold_rel: float = SIGMA_DIVERGENCE_THRESHOLD_REL,
) -> Dict[str, Any]:
    """Plan §3.3 / §D8: compute the σ-mapping divergence flag."""
    db_s = float(delta_beta_fit_stern)
    db_a = float(delta_beta_fit_ablation)
    denom = max(abs(db_s), abs(db_a))
    if denom == 0.0:
        return {
            "rel_divergence": 0.0, "flag": "ok",
            "delta_beta_fit_stern": db_s,
            "delta_beta_fit_ablation": db_a,
        }
    rel_div = abs(db_s - db_a) / denom
    return {
        "rel_divergence": rel_div,
        "flag": "non_identifiable" if rel_div > threshold_rel else "ok",
        "threshold_rel": threshold_rel,
        "delta_beta_fit_stern": db_s,
        "delta_beta_fit_ablation": db_a,
    }


def determine_outcome(
    *,
    loss_finite_valid_stern: Sequence[Dict[str, Any]],
    deck_target_pct: float,
    primary_gate_pp: float = PRIMARY_GATE_PP_ABSOLUTE,
    d7_pass: bool,
    sign_guard_status_at_min: str,
) -> Dict[str, Any]:
    """Plan §D8 outcome verdict (A/B/C)."""
    if not d7_pass:
        return {
            "verdict": "OUTCOME_C_NON_IDENTIFIABLE_flagged",
            "primary_gap_pp": None,
            "primary_pass": None,
            "sign_guard_status": sign_guard_status_at_min,
        }
    if not loss_finite_valid_stern:
        return {
            "verdict": "OUTCOME_B_FALSIFIED_documented",
            "reason": "no_finite_valid_evals",
            "primary_gap_pp": None,
            "primary_pass": False,
            "sign_guard_status": sign_guard_status_at_min,
        }
    best = min(loss_finite_valid_stern, key=lambda e: float(e["loss"]))
    primary_gap = float(best["loss"])
    primary_pass = primary_gap <= float(primary_gate_pp)
    sign_guard_pass = sign_guard_status_at_min == "ok"

    if primary_pass and sign_guard_pass:
        return {
            "verdict": "OUTCOME_A_LOCKED_PASS",
            "primary_gap_pp": primary_gap,
            "primary_pass": True,
            "sign_guard_status": sign_guard_status_at_min,
            "delta_beta_fit_pm2": float(best["delta_beta_pm2"]),
        }
    return {
        "verdict": "OUTCOME_B_FALSIFIED_documented",
        "primary_gap_pp": primary_gap,
        "primary_pass": primary_pass,
        "sign_guard_status": sign_guard_status_at_min,
        "delta_beta_fit_pm2": float(best["delta_beta_pm2"]),
        "reason": (
            "sign_guard_violation" if not sign_guard_pass
            else "primary_gate_exceeded"
        ),
    }


# ---------------------------------------------------------------------------
# Phase 10.B execution
# ---------------------------------------------------------------------------


def run_phase_10B(
    *,
    out_dir: str,
    skip_a2_reproduction: bool = False,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Run the full 10.B.0–10.B.9 pipeline.  Returns the verdict bundle."""
    from calibration.singh2016 import compute_beta_per_cation

    deck = _load_deck_target()["deck_target_locked"]
    deck_target_pct = float(deck["max_h2o2_selectivity_pct_mean"])
    progress(
        f"== Phase 10.B START: deck target = "
        f"{deck_target_pct:.3f} pp (n={deck['n_rows']}, "
        f"std={deck['max_h2o2_selectivity_pct_std']:.2f} pp)"
    )

    beta_K_Cu = compute_beta_per_cation("K+")  # -45.608196
    progress(f"  β_K_Cu = {beta_K_Cu:.6f} pm²")

    bookkeeping: Dict[str, Any] = {
        "deck_target_pct": deck_target_pct,
        "beta_K_Cu": beta_K_Cu,
        "loss_all": [],
        "phases": {},
        "verdict": None,
    }

    # ---- 10.B.0 — A.2 HARD reproduction
    if not skip_a2_reproduction:
        progress("== 10.B.0 — A.2 HARD reproduction (Δ_β=0, stern, a2_repro)")
        a2_path = os.path.join(out_dir, "eval_db_0p0_stern_a2_repro.json")
        if os.path.exists(a2_path):
            progress(f"  cached A.2 reproduction at {a2_path}")
        else:
            from scripts.studies.phase6b_step10_phase_D_fit_eval import (
                evaluate_delta_beta, A2_WARM_GRID, V_ANCHOR,
            )
            if dry_run:
                progress("  [DRY RUN] would run A.2 reproduction")
            else:
                t0 = time.time()
                a2_result = evaluate_delta_beta(
                    delta_beta_pm2=0.0,
                    sigma_mapping="stern",
                    v_grid=A2_WARM_GRID,
                    v_anchor=V_ANCHOR,
                    mode="a2_reproduction",
                )
                with open(a2_path, "w") as f:
                    json.dump(a2_result, f, indent=2, default=str)
                progress(f"  done in {time.time() - t0:.1f}s")
        # HARD-stop A.2 reproduction comparison (D5(a))
        bookkeeping["phases"]["10.B.0"] = _check_a2_reproduction(
            a2_path=a2_path, progress=progress,
        )
        if not bookkeeping["phases"]["10.B.0"].get("passes", False):
            with open(
                os.path.join(out_dir, "delta_beta_zero_baseline_FAIL.md"),
                "w",
            ) as f:
                f.write(
                    "# Phase D 10.B.0 HARD A.2 reproduction FAILED\n\n"
                    f"See {a2_path} and the comparison report below.\n\n"
                    f"```json\n{json.dumps(bookkeeping['phases']['10.B.0'], indent=2)}\n```\n"
                )
            bookkeeping["verdict"] = {
                "verdict": "10.B.0_HARD_REPRODUCTION_FAIL",
                "reason": "byte-equivalence vs A.2 baseline failed",
            }
            return bookkeeping

    # ---- 10.B.1 — Stern Δ_β=0 production-grid baseline + 3 dup for noise
    progress("== 10.B.1 — Stern Δ_β=0 production-grid baseline (+ 2 dup)")
    stern_baseline_evals: List[Dict[str, Any]] = []
    for dup_idx in range(D7_NOISE_DUP_COUNT):
        suffix = "" if dup_idx == 0 else f"_dup{dup_idx}"
        out_name = f"eval_db_0p0_stern_baseline{suffix}.json"
        ev = evaluate_one_cached(
            delta_beta_pm2=0.0,
            sigma_mapping="stern",
            deck_target_pct=deck_target_pct,
            out_dir=out_dir,
            out_name=out_name,
            progress=progress,
            dry_run=dry_run,
        )
        stern_baseline_evals.append(ev)
        bookkeeping["loss_all"].append(ev)
    bookkeeping["phases"]["10.B.1"] = {"baselines": stern_baseline_evals}

    if dry_run:
        progress("[DRY RUN] stopping after 10.B.1 dry run.")
        return bookkeeping

    # Compute σ_local_clamped_max_over_grid from Stern Δ_β=0 baseline
    primary_baseline = stern_baseline_evals[0]
    sigma_max = _read_sigma_max_from_eval(primary_baseline["eval_path"])
    bookkeeping["sigma_local_clamped_max_over_grid"] = sigma_max
    progress(f"  σ_local_clamped_max_over_grid = {sigma_max!r}")

    # Compute noise floor from 3 dup baselines
    valid_dup_losses = [
        float(e["loss"]) for e in stern_baseline_evals
        if e.get("status") == "finite_valid"
    ]
    if len(valid_dup_losses) >= 2:
        noise_mean = sum(valid_dup_losses) / len(valid_dup_losses)
        noise_std = math.sqrt(
            sum((x - noise_mean) ** 2 for x in valid_dup_losses)
            / max(1, len(valid_dup_losses) - 1)
        )
    else:
        noise_std = 0.0
    bookkeeping["noise_std_from_baseline_dups"] = noise_std
    progress(f"  noise_std (from dup baselines) = {noise_std:.6g} pp")

    # ---- 10.B.2 — Ablation Δ_β=0 baseline
    progress("== 10.B.2 — Ablation Δ_β=0 production-grid baseline")
    ablation_baseline = evaluate_one_cached(
        delta_beta_pm2=0.0,
        sigma_mapping="ablation_singh_0.141",
        deck_target_pct=deck_target_pct,
        out_dir=out_dir,
        out_name="eval_db_0p0_ablation_baseline.json",
        progress=progress,
        dry_run=dry_run,
    )
    bookkeeping["loss_all"].append(ablation_baseline)
    bookkeeping["phases"]["10.B.2"] = {"baseline": ablation_baseline}

    # ---- 10.B.3 — Stern pre-fit grid (7 additional Δ_β values)
    progress("== 10.B.3 — Stern pre-fit grid (7 additional Δ_β)")
    stern_pre_fit: List[Dict[str, Any]] = []
    if sigma_max is None or sigma_max == 0.0:
        progress(
            "  WARNING: σ_max from Stern baseline is 0/None; cannot "
            "construct Stern pre-fit grid.  Skipping 10.B.3."
        )
    else:
        for t in T_TARGETS_STERN:
            db = stern_delta_beta_for_target(
                target_dpka=t, beta_K_Cu=beta_K_Cu, sigma_max=sigma_max,
            )
            ev = evaluate_one_cached(
                delta_beta_pm2=db,
                sigma_mapping="stern",
                deck_target_pct=deck_target_pct,
                out_dir=out_dir,
                progress=progress,
                dry_run=dry_run,
            )
            stern_pre_fit.append(ev)
            bookkeeping["loss_all"].append(ev)
    bookkeeping["phases"]["10.B.3"] = {"pre_fit_grid": stern_pre_fit}

    # ---- 10.B.4 — Ablation pre-fit grid (6 additional Δ_β values)
    progress("== 10.B.4 — Ablation pre-fit grid (6 additional Δ_β)")
    ablation_pre_fit: List[Dict[str, Any]] = []
    for t in T_TARGETS_ABLATION:
        db = ablation_delta_beta_for_target(
            target_dpka=t, beta_K_Cu=beta_K_Cu, sigma=ABLATION_SIGMA,
        )
        ev = evaluate_one_cached(
            delta_beta_pm2=db,
            sigma_mapping="ablation_singh_0.141",
            deck_target_pct=deck_target_pct,
            out_dir=out_dir,
            progress=progress,
            dry_run=dry_run,
        )
        ablation_pre_fit.append(ev)
        bookkeeping["loss_all"].append(ev)
    bookkeeping["phases"]["10.B.4"] = {"pre_fit_grid": ablation_pre_fit}

    # ---- 10.B.5 — D7 identifiability gate
    progress("== 10.B.5 — D7 identifiability gate")
    stern_loss_all = [
        e for e in bookkeeping["loss_all"]
        if e.get("sigma_mapping") == "stern"
    ]
    stern_finite = filter_finite_valid(stern_loss_all)
    d7_result = d7_identifiability_gate(
        stern_finite, noise_std=noise_std,
        sigma_max=sigma_max or 1.0, beta_K_Cu=beta_K_Cu,
    )
    bookkeeping["phases"]["10.B.5"] = d7_result
    progress(
        f"  D7 result: overall_pass = {d7_result['overall_pass']}; "
        f"per-criterion: { {k: v.get('passes') for k, v in d7_result['criteria'].items()} }"
    )

    if not d7_result["overall_pass"]:
        progress(
            "== OUTCOME_C_NON_IDENTIFIABLE_flagged — STOPping fit."
        )
        bookkeeping["verdict"] = determine_outcome(
            loss_finite_valid_stern=stern_finite,
            deck_target_pct=deck_target_pct,
            d7_pass=False,
            sign_guard_status_at_min="not_evaluated",
        )
        _emit_outcome_report(
            out_dir=out_dir, bookkeeping=bookkeeping, progress=progress,
        )
        return bookkeeping

    # ---- 10.B.6 + 10.B.7 — Brent refinement (Stern + Ablation)
    progress("== 10.B.6 — Stern Brent refinement")
    stern_brent_result = _run_brent(
        sigma_mapping="stern",
        deck_target_pct=deck_target_pct,
        out_dir=out_dir,
        beta_K_Cu=beta_K_Cu,
        sigma_max=sigma_max or 1.0,
        bookkeeping=bookkeeping,
        progress=progress,
        dry_run=dry_run,
    )
    bookkeeping["phases"]["10.B.6"] = stern_brent_result

    progress("== 10.B.7 — Ablation Brent refinement")
    ablation_brent_result = _run_brent(
        sigma_mapping="ablation_singh_0.141",
        deck_target_pct=deck_target_pct,
        out_dir=out_dir,
        beta_K_Cu=beta_K_Cu,
        sigma_max=ABLATION_SIGMA,
        bookkeeping=bookkeeping,
        progress=progress,
        dry_run=dry_run,
    )
    bookkeeping["phases"]["10.B.7"] = ablation_brent_result

    # ---- 10.B.8 — σ-mapping divergence
    progress("== 10.B.8 — σ-mapping divergence check")
    div_result = sigma_mapping_divergence(
        delta_beta_fit_stern=stern_brent_result["delta_beta_fit_pm2"],
        delta_beta_fit_ablation=ablation_brent_result["delta_beta_fit_pm2"],
    )
    bookkeeping["phases"]["10.B.8"] = div_result
    progress(
        f"  rel_divergence = {div_result['rel_divergence']:.4g} "
        f"({div_result['flag']})"
    )

    # ---- 10.B.9 — Outcome verdict
    progress("== 10.B.9 — Outcome verdict")
    sign_guard_at_min = _read_sign_guard_at_min(stern_brent_result)
    bookkeeping["verdict"] = determine_outcome(
        loss_finite_valid_stern=filter_finite_valid([
            e for e in bookkeeping["loss_all"]
            if e.get("sigma_mapping") == "stern"
        ]),
        deck_target_pct=deck_target_pct,
        d7_pass=True,
        sign_guard_status_at_min=sign_guard_at_min,
    )
    progress(f"== Phase D verdict: {bookkeeping['verdict']['verdict']}")
    _emit_outcome_report(
        out_dir=out_dir, bookkeeping=bookkeeping, progress=progress,
    )
    return bookkeeping


def _check_a2_reproduction(
    *, a2_path: str, progress: Callable[[str], None],
) -> Dict[str, Any]:
    """D5(a) HARD-stop comparison vs the v10b A.2 V_kin baseline.

    COMPARISON_FIELDS rel ≤ 1e-3.
    """
    audit = json.loads(
        open(os.path.join(_ROOT, DECK_TARGET_AUDIT_PATH)).read()
    )
    target = audit["a2_reproduction_target"]
    fields = list(target["values"].keys())
    tolerance = float(target["tolerance_rel"])
    eval_data = json.loads(open(a2_path).read())

    # Locate V_kin entry in per_v_records
    v_kin = float(target["V_kin"])
    v_kin_rec = None
    for rec in eval_data.get("per_v_records", []):
        if (
            rec.get("v_rhe") is not None
            and abs(float(rec["v_rhe"]) - v_kin) < 1e-9
        ):
            v_kin_rec = rec
            break
    if v_kin_rec is None or not v_kin_rec.get("snes_converged", False):
        return {
            "passes": False,
            "reason": "no_v_kin_record_or_unconverged",
            "v_kin": v_kin,
        }

    # Pull λ=1 rung diagnostic
    lam1_diag = None
    for d in eval_data.get("per_v_lam1_diagnostics", []):
        if (
            d.get("v_rhe") is not None
            and abs(float(d["v_rhe"]) - v_kin) < 1e-9
        ):
            lam1_diag = d.get("lambda1_rung")
            break
    if lam1_diag is None:
        return {
            "passes": False,
            "reason": "no_lambda1_rung_diagnostic",
            "v_kin": v_kin,
        }

    deltas: Dict[str, Any] = {}
    all_pass = True
    for f in fields:
        target_val = target["values"][f]
        observed = lam1_diag.get(f)
        if f == "cd_mA_cm2" and observed is None:
            observed = lam1_diag.get("cd_mA_cm2_callback")
        if observed is None:
            deltas[f] = {"target": target_val, "observed": None,
                         "rel": None, "passes": False}
            all_pass = False
            continue
        denom = max(abs(float(target_val)), 1e-30)
        rel = abs(float(observed) - float(target_val)) / denom
        passes = rel <= tolerance
        deltas[f] = {
            "target": float(target_val),
            "observed": float(observed),
            "rel": float(rel),
            "passes": passes,
        }
        if not passes:
            all_pass = False
            progress(
                f"  FIELD {f} FAIL: target={target_val!r}, "
                f"observed={observed!r}, rel={rel:.4g} > {tolerance}"
            )

    return {
        "passes": all_pass,
        "v_kin": v_kin,
        "tolerance_rel": tolerance,
        "fields": deltas,
    }


def _read_sigma_max_from_eval(eval_path: str) -> Optional[float]:
    if not os.path.exists(eval_path):
        return None
    data = json.loads(open(eval_path).read())
    return data.get("sigma_clamped_max_in_window")


def _read_sign_guard_at_min(brent_result: Dict[str, Any]) -> str:
    """Read the sign-guard status from the Brent best-eval JSON."""
    best_path = brent_result.get("best_eval_path")
    if best_path is None or not os.path.exists(best_path):
        return "not_evaluated"
    data = json.loads(open(best_path).read())
    return data.get("sign_guard", {}).get("status", "not_evaluated")


def _run_brent(
    *,
    sigma_mapping: str,
    deck_target_pct: float,
    out_dir: str,
    beta_K_Cu: float,
    sigma_max: float,
    bookkeeping: Dict[str, Any],
    progress: Callable[[str], None],
    dry_run: bool,
) -> Dict[str, Any]:
    """Run scipy.optimize.minimize_scalar(method='bounded') over Δ_β.

    Returns dict with {converged, delta_beta_fit_pm2, loss_at_fit,
    n_evals, best_eval_path, brent_history}.
    """
    if dry_run:
        return {
            "converged": False,
            "delta_beta_fit_pm2": 0.0,
            "loss_at_fit": float("inf"),
            "n_evals": 0,
            "best_eval_path": None,
            "brent_history": [],
            "note": "DRY RUN — Brent skipped",
        }

    import scipy.optimize as opt

    if sigma_mapping == "stern":
        bounds = (
            stern_delta_beta_for_target(
                target_dpka=T_LOWER_STERN, beta_K_Cu=beta_K_Cu,
                sigma_max=sigma_max,
            ),
            stern_delta_beta_for_target(
                target_dpka=T_UPPER_STERN, beta_K_Cu=beta_K_Cu,
                sigma_max=sigma_max,
            ),
        )
        xatol = abs(XATOL_TARGET_DPKA / sigma_max)
    elif sigma_mapping == "ablation_singh_0.141":
        bounds = (
            ablation_delta_beta_for_target(
                target_dpka=T_LOWER_ABLATION, beta_K_Cu=beta_K_Cu,
                sigma=ABLATION_SIGMA,
            ),
            ablation_delta_beta_for_target(
                target_dpka=T_UPPER_ABLATION, beta_K_Cu=beta_K_Cu,
                sigma=ABLATION_SIGMA,
            ),
        )
        xatol = abs(XATOL_TARGET_DPKA / ABLATION_SIGMA)
    else:
        raise ValueError(f"unknown sigma_mapping {sigma_mapping!r}")

    progress(
        f"  Brent bounds: {bounds!r}, xatol = {xatol:.4g}, maxiter = {BRENT_MAXITER}"
    )

    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    def _objective(db: float) -> float:
        ev = evaluate_one_cached(
            delta_beta_pm2=float(db),
            sigma_mapping=sigma_mapping,
            deck_target_pct=deck_target_pct,
            out_dir=out_dir,
            progress=progress,
            dry_run=False,
        )
        bookkeeping["loss_all"].append(ev)
        history.append(ev)
        nonlocal best
        if best is None or float(ev["loss"]) < float(best["loss"]):
            best = ev
        return float(ev["loss"])

    result = opt.minimize_scalar(
        _objective,
        method="bounded",
        bounds=bounds,
        options={"xatol": xatol, "maxiter": BRENT_MAXITER},
    )
    return {
        "converged": bool(result.success),
        "delta_beta_fit_pm2": float(result.x),
        "loss_at_fit": float(result.fun),
        "n_evals": int(result.nfev),
        "best_eval_path": best["eval_path"] if best is not None else None,
        "brent_message": str(result.message),
        "bounds": list(bounds),
        "xatol": xatol,
        "history": history,
    }


def _emit_outcome_report(
    *, out_dir: str, bookkeeping: Dict[str, Any],
    progress: Callable[[str], None],
) -> None:
    """D9: emit phase_E_spec.json | falsification_report.json |
    identifiability_report.json depending on the verdict.
    """
    verdict = bookkeeping.get("verdict", {})
    v = verdict.get("verdict", "unknown")
    if v == "OUTCOME_A_LOCKED_PASS":
        path = os.path.join(out_dir, "phase_E_spec.json")
    elif v == "OUTCOME_B_FALSIFIED_documented":
        path = os.path.join(out_dir, "falsification_report.json")
    elif v == "OUTCOME_C_NON_IDENTIFIABLE_flagged":
        path = os.path.join(out_dir, "identifiability_report.json")
    else:
        path = os.path.join(out_dir, "phase_D_report.json")
    with open(path, "w") as f:
        json.dump(bookkeeping, f, indent=2, default=str)
    progress(f"  emitted {path}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-subdir", type=str, default="phase6b_step10_phase_D",
    )
    p.add_argument(
        "--skip-a2-reproduction", action="store_true",
        help="Skip the 10.B.0 HARD reproduction (use only when previously verified).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the eval plan without calling the forward solver.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_dir = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Out dir: {out_dir}", flush=True)
    bookkeeping = run_phase_10B(
        out_dir=out_dir,
        skip_a2_reproduction=args.skip_a2_reproduction,
        dry_run=args.dry_run,
    )
    print(f"Verdict: {bookkeeping.get('verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
