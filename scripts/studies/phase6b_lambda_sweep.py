"""Phase 6β step 10 follow-up — λ_hydrolysis sweep at Δ_β=0, v10b params.

After the Phase D fit verdict
``OUTCOME_C_NON_IDENTIFIABLE_flagged`` (Δ_β alone cannot match the
deck) and the physical-a_nondim production fix landed in
``scripts/_bv_common.py`` on 2026-05-21, this driver tests the next
hypothesis: there exists a hydrolysis activation level
``λ_target ∈ (0, 1)`` (with Δ_β held at 0) where the model's max
H₂O₂% over the deck-mask V grid matches the deck K₂SO₄ at pH 4
mean (50.95 pp).

Motivation (from the Phase D + bridge data):

* No-hydrolysis bridge runs at v10b params give max H₂O₂% ≈ 38.9 pp
  (under-shoot the deck by ~12 pp).
* Phase D production fit at λ=1, Δ_β=0 gives max H₂O₂% = 66.58 pp
  (over-shoot the deck by ~16 pp).
* Linearly, this suggests λ* ≈ 0.43.  The physical-a fix may shift
  both endpoints, so re-sample the loss curve directly.

CLI::

    python -u scripts/studies/phase6b_lambda_sweep.py \\
        --lambda-grid 0.0 0.25 0.5 0.75 1.0 \\
        --out-subdir phase6b_lambda_sweep

Outputs per-λ JSON files
``eval_lambda_<X>_<sigma>.json`` plus a sweep-summary
``lambda_sweep_summary.json`` that emits the loss curve
``|max_H2O2_pct(λ) - 50.95|`` against the deck K₂SO₄ at pH 4 mean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
    SIGMA_MAPPING_STERN,
    V_ANCHOR,
    V_KIN_OBS_MASK_HI,
    V_KIN_OBS_MASK_LO,
    V_RHE_PRODUCTION_GRID,
    evaluate_delta_beta,
)


DECK_K_AT_PH4_MEAN_H2O2_PCT: float = 50.95
"""Phase D deck target — K₂SO₄ at pH ∈ [3.5, 4.5] mean of max H₂O₂%
in the deck-mask V window.  Source:
``StudyResults/phase6b_step10_phase_D/data_audit_K_at_pH4.json``."""

DEFAULT_LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
"""Default 5-point λ_target grid.  Aligned with the LAMBDA_LADDER
rungs the Phase D fit_eval helper already uses, so each per-V ramp
ladder is a clean prefix of the production ladder."""

DEFAULT_OUT_SUBDIR: str = "phase6b_lambda_sweep"


def _default_eval_name(lambda_target: float, sigma_mapping: str) -> str:
    """Deterministic JSON basename for a per-λ eval result."""
    safe = f"{lambda_target:.4f}".replace(".", "p")
    return f"eval_lambda_{safe}_{sigma_mapping}.json"


def _aggregate_max_h2o2_in_mask(
    aggregated: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Pull ``max_H2O2_selectivity_in_window_pct`` from an
    ``evaluate_delta_beta`` result's ``aggregated_observables`` dict.

    Returns None when aggregation is missing or unable to find a
    finite max (e.g. all per-V records failed gates).
    """
    if not isinstance(aggregated, dict):
        return None
    val = aggregated.get("max_H2O2_selectivity_in_window_pct")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _run_one_lambda(
    *,
    lambda_target: float,
    sigma_mapping: str,
    v_grid: Sequence[float],
    v_anchor: float,
    out_dir: str,
    progress_prefix: str,
) -> Dict[str, Any]:
    """Run a single λ_target eval and write its JSON to ``out_dir``.

    Returns a compact summary dict for inclusion in the sweep summary.
    """
    def _progress(msg: str) -> None:
        print(f"{progress_prefix}{msg}", flush=True)

    t_start = time.time()
    _progress(
        f"=== START λ_target={lambda_target:.4f}, σ={sigma_mapping} ==="
    )
    try:
        result = evaluate_delta_beta(
            delta_beta_pm2=0.0,
            sigma_mapping=sigma_mapping,
            v_grid=v_grid,
            v_anchor=v_anchor,
            mode="production",
            progress=_progress,
            lambda_target=lambda_target,
        )
        eval_error: Optional[str] = None
    except Exception as exc:                              # pragma: no cover
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "lambda_target": float(lambda_target),
            "delta_beta_pm2": 0.0,
            "sigma_mapping": str(sigma_mapping),
        }
        eval_error = f"{type(exc).__name__}: {exc}"

    wall_seconds = time.time() - t_start

    # Stamp the eval result with the λ_target + Δ_β=0 metadata in case
    # the caller wants to read individual eval JSONs without consulting
    # the sweep summary.
    result["lambda_sweep_metadata"] = {
        "lambda_target": float(lambda_target),
        "delta_beta_pm2": 0.0,
        "sigma_mapping": str(sigma_mapping),
        "v_anchor": float(v_anchor),
        "v_grid": [float(v) for v in v_grid],
        "wall_seconds": float(wall_seconds),
        "eval_error": eval_error,
    }

    out_name = _default_eval_name(lambda_target, sigma_mapping)
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    _progress(f"  eval JSON saved -> {out_path}")

    aggregated = result.get("aggregated_observables")
    max_h2o2_pct = _aggregate_max_h2o2_in_mask(aggregated)
    n_gate_pass = result.get("n_gate_pass")
    n_gate_fail = result.get("n_gate_fail")
    argmax_v = (
        aggregated.get("argmax_V_for_selectivity")
        if isinstance(aggregated, dict) else None
    )

    summary_entry = {
        "lambda_target": float(lambda_target),
        "sigma_mapping": str(sigma_mapping),
        "eval_json_path": out_path,
        "wall_seconds": float(wall_seconds),
        "max_H2O2_pct_in_mask": max_h2o2_pct,
        "argmax_V_for_selectivity": argmax_v,
        "n_gate_pass": n_gate_pass,
        "n_gate_fail": n_gate_fail,
        "eval_error": eval_error,
    }
    if max_h2o2_pct is not None:
        summary_entry["loss_vs_deck_pp"] = (
            abs(max_h2o2_pct - DECK_K_AT_PH4_MEAN_H2O2_PCT)
        )
        summary_entry["delta_vs_deck_pp"] = (
            max_h2o2_pct - DECK_K_AT_PH4_MEAN_H2O2_PCT
        )

    _progress(
        f"=== END   λ_target={lambda_target:.4f}, σ={sigma_mapping} "
        f"({wall_seconds:.1f}s, "
        f"max_H2O2_pct={max_h2o2_pct}, "
        f"gate_pass={n_gate_pass}/{n_gate_pass + n_gate_fail if n_gate_pass is not None and n_gate_fail is not None else '?'}"
        f") ==="
    )
    return summary_entry


def _json_default(obj: Any) -> Any:
    """Fallback serializer for objects not JSON-native (numpy scalars,
    sets, etc.).  Returns a string repr for anything unrecognized so
    the dump never aborts mid-write."""
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return repr(obj)


def _build_sweep_summary(
    entries: Sequence[Dict[str, Any]],
    *,
    sigma_mapping: str,
    deck_target_pct: float,
    v_anchor: float,
    v_grid: Sequence[float],
    wall_seconds_total: float,
) -> Dict[str, Any]:
    """Aggregate per-λ entries into the sweep-level summary.

    Identifies the λ that minimizes ``loss_vs_deck_pp`` (skipping
    entries with errored evals or missing observables).  Also reports
    a simple linear-interp estimate of λ* by bracketing the deck target
    between adjacent λ entries, when monotonic in the relevant range.
    """
    sorted_entries = sorted(
        entries, key=lambda e: float(e["lambda_target"])
    )

    finite_entries = [
        e for e in sorted_entries
        if e.get("eval_error") is None
        and e.get("max_H2O2_pct_in_mask") is not None
    ]

    best_entry: Optional[Dict[str, Any]] = None
    if finite_entries:
        best_entry = min(
            finite_entries, key=lambda e: float(e["loss_vs_deck_pp"])
        )

    # Linear interpolation in (λ, max_H2O2_pct) space for an estimate
    # of where the curve crosses ``deck_target_pct``.  Only meaningful
    # when the curve is monotonic in the bracket.
    lambda_star_linear: Optional[float] = None
    bracket: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    for lo, hi in zip(finite_entries[:-1], finite_entries[1:]):
        lo_y = float(lo["max_H2O2_pct_in_mask"])
        hi_y = float(hi["max_H2O2_pct_in_mask"])
        if (lo_y - deck_target_pct) * (hi_y - deck_target_pct) <= 0:
            bracket = (lo, hi)
            lo_x = float(lo["lambda_target"])
            hi_x = float(hi["lambda_target"])
            if hi_y == lo_y:
                lambda_star_linear = 0.5 * (lo_x + hi_x)
            else:
                frac = (deck_target_pct - lo_y) / (hi_y - lo_y)
                lambda_star_linear = lo_x + frac * (hi_x - lo_x)
            break

    return {
        "sigma_mapping": str(sigma_mapping),
        "deck_target_max_H2O2_pct": float(deck_target_pct),
        "v_anchor": float(v_anchor),
        "v_grid": [float(v) for v in v_grid],
        "v_mask": [
            float(V_KIN_OBS_MASK_LO), float(V_KIN_OBS_MASK_HI)
        ],
        "lambda_grid": [
            float(e["lambda_target"]) for e in sorted_entries
        ],
        "max_H2O2_pct_in_mask_by_lambda": [
            e.get("max_H2O2_pct_in_mask") for e in sorted_entries
        ],
        "loss_vs_deck_pp_by_lambda": [
            e.get("loss_vs_deck_pp") for e in sorted_entries
        ],
        "best_lambda": (
            float(best_entry["lambda_target"]) if best_entry else None
        ),
        "best_loss_vs_deck_pp": (
            float(best_entry["loss_vs_deck_pp"]) if best_entry else None
        ),
        "best_max_H2O2_pct": (
            float(best_entry["max_H2O2_pct_in_mask"])
            if best_entry else None
        ),
        "deck_target_bracket": (
            {
                "lo_lambda": float(bracket[0]["lambda_target"]),
                "hi_lambda": float(bracket[1]["lambda_target"]),
                "lo_max_H2O2_pct": float(
                    bracket[0]["max_H2O2_pct_in_mask"]
                ),
                "hi_max_H2O2_pct": float(
                    bracket[1]["max_H2O2_pct_in_mask"]
                ),
            }
            if bracket else None
        ),
        "lambda_star_linear_interp": lambda_star_linear,
        "wall_seconds_total": float(wall_seconds_total),
        "per_lambda_entries": list(sorted_entries),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6β λ_hydrolysis sweep at Δ_β=0, v10b params, "
            "physical a_nondim."
        ),
    )
    parser.add_argument(
        "--lambda-grid", type=float, nargs="+",
        default=list(DEFAULT_LAMBDA_GRID),
        help=(
            "λ_target values to evaluate.  Each must lie in [0, 1].  "
            f"Default: {DEFAULT_LAMBDA_GRID}."
        ),
    )
    parser.add_argument(
        "--sigma-mapping", type=str, default=SIGMA_MAPPING_STERN,
        choices=("stern", "ablation_singh_0.141"),
        help=(
            "σ-mapping convention (Phase D plan §3.3).  Default: "
            f"{SIGMA_MAPPING_STERN}."
        ),
    )
    parser.add_argument(
        "--out-subdir", type=str, default=DEFAULT_OUT_SUBDIR,
        help=(
            f"Output directory under StudyResults/.  Default: "
            f"{DEFAULT_OUT_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--v-anchor", type=float, default=float(V_ANCHOR),
        help=(
            f"Anchor voltage (V vs RHE).  Default: {V_ANCHOR}."
        ),
    )
    parser.add_argument(
        "--deck-target-pct", type=float,
        default=DECK_K_AT_PH4_MEAN_H2O2_PCT,
        help=(
            "Deck max H₂O₂%% target the loss is computed against.  "
            f"Default: {DECK_K_AT_PH4_MEAN_H2O2_PCT} pp (K₂SO₄ pH 4 mean)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = _parse_args(argv)

    for lam in ns.lambda_grid:
        if not (0.0 <= float(lam) <= 1.0):
            raise ValueError(
                f"--lambda-grid entries must lie in [0, 1] (got {lam!r})"
            )
    lambda_grid: Tuple[float, ...] = tuple(
        sorted(set(float(lam) for lam in ns.lambda_grid))
    )

    out_dir = os.path.join(_ROOT, "StudyResults", ns.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78, flush=True)
    print(
        f"  Phase 6β λ_hydrolysis sweep — Δ_β=0, σ={ns.sigma_mapping}",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(f"  λ grid       = {list(lambda_grid)}", flush=True)
    print(f"  V_anchor     = {ns.v_anchor:+.3f} V", flush=True)
    print(
        f"  V grid       = [{V_RHE_PRODUCTION_GRID[0]:+.3f}, "
        f"{V_RHE_PRODUCTION_GRID[-1]:+.3f}] V "
        f"({len(V_RHE_PRODUCTION_GRID)} pts)",
        flush=True,
    )
    print(
        f"  V mask       = [{V_KIN_OBS_MASK_LO:+.3f}, "
        f"{V_KIN_OBS_MASK_HI:+.3f}] V (deck overlap)",
        flush=True,
    )
    print(f"  deck target  = {ns.deck_target_pct:.2f} pp", flush=True)
    print(f"  output dir   = {out_dir}", flush=True)

    t_total = time.time()
    entries: List[Dict[str, Any]] = []
    for i, lam in enumerate(lambda_grid):
        prefix = f"  [λ {i + 1}/{len(lambda_grid)} = {lam:.4f}] "
        entry = _run_one_lambda(
            lambda_target=float(lam),
            sigma_mapping=ns.sigma_mapping,
            v_grid=V_RHE_PRODUCTION_GRID,
            v_anchor=float(ns.v_anchor),
            out_dir=out_dir,
            progress_prefix=prefix,
        )
        entries.append(entry)

    summary = _build_sweep_summary(
        entries,
        sigma_mapping=ns.sigma_mapping,
        deck_target_pct=float(ns.deck_target_pct),
        v_anchor=float(ns.v_anchor),
        v_grid=V_RHE_PRODUCTION_GRID,
        wall_seconds_total=time.time() - t_total,
    )

    summary_path = os.path.join(out_dir, "lambda_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print()
    print("=" * 78, flush=True)
    print("  Sweep summary", flush=True)
    print("=" * 78, flush=True)
    print(f"  total wall:                 {summary['wall_seconds_total']:.1f}s", flush=True)
    for e in summary["per_lambda_entries"]:
        lam = e["lambda_target"]
        mh = e.get("max_H2O2_pct_in_mask")
        loss = e.get("loss_vs_deck_pp")
        mh_str = f"{mh:7.3f}" if mh is not None else "  None "
        loss_str = f"{loss:7.3f}" if loss is not None else "  None "
        print(
            f"  λ={lam:5.3f}: max_H2O2%={mh_str} pp  loss={loss_str} pp",
            flush=True,
        )
    print(f"  best λ:                     {summary['best_lambda']}", flush=True)
    print(f"  best loss:                  {summary['best_loss_vs_deck_pp']} pp", flush=True)
    print(
        f"  λ_star (linear bracket):    {summary['lambda_star_linear_interp']}",
        flush=True,
    )
    print(f"  summary JSON:               {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
