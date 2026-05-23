"""Phase 6β step 10 follow-up — K0_R4e factor sweep at λ=1, Δ_β=0, v10b params.

After the λ-sweep (2026-05-21) confirmed *binary saturation* of model
selectivity in λ (any λ ≥ 0.1 → max H₂O₂% = 66.58 pp; λ=0 → 38.9 pp,
deck = 50.95 pp), the next single-parameter knob to probe is the
``K0_R4e / K0_R2e`` rate-constant ratio.

This factor is a known placeholder per the conjecture audit
(``docs/phase6/CONJECTURE_AUDIT_2026-05-09.md``): no measured value
in the data folder; the production value ``1e-14`` was Claude-fit to
qualitative selectivity, and the v10b chain inherited it.  The Tafel
xlsx that would constrain it is the open data-ask M1.

Quick algebra: at λ=1, model H₂O₂% = 66.58 pp implies I_4e/I_2e ≈ 1.
Deck H₂O₂% = 50.95 pp implies I_4e/I_2e ≈ 1.93.  So roughly ~2×
more 4e activity (naive linear).  Tafel exponentials may dominate
the actual response — needs a sweep.

CLI::

    python -u scripts/studies/phase6b_k0_r4e_sweep.py \\
        --k0-factor-grid 1e-18 1e-16 1e-14 1e-12 1e-10 \\
        --out-subdir phase6b_k0_r4e_sweep

Outputs per-K0 JSON files ``eval_k0r4e_<X>_<sigma>.json`` plus a
sweep-summary ``k0_r4e_sweep_summary.json`` that emits the loss
curve ``|max_H2O2_pct(K0) - 50.95|`` against the deck K₂SO₄/pH 4 mean.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from scripts.studies.phase6b_step10_phase_D_fit_eval import (
    K0_R4E_FACTOR_V10B,
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

DEFAULT_K0_FACTOR_GRID: Tuple[float, ...] = (
    1e-18, 1e-16, 1e-14, 1e-12, 1e-10,
)
"""Default 5-point K0_R4e factor grid (decade-spaced).  Centered on the
v10b production value 1e-14; brackets the historical pre-hydrolysis
finding that ratio ≈ 1e-18 gave Mangan-like 35-50% selectivity.  At
λ=1 + hydrolysis-on the response may differ."""

DEFAULT_OUT_SUBDIR: str = "phase6b_k0_r4e_sweep"


def _format_k0(k0: float) -> str:
    """Deterministic filename-safe encoding of a K0 factor in scientific
    notation, e.g. ``1.0e-14`` -> ``1p0eN14``."""
    s = f"{k0:.2e}"  # e.g. '1.00e-14'
    s = s.replace(".", "p").replace("+", "P").replace("-", "N")
    return s


def _default_eval_name(k0_factor: float, sigma_mapping: str) -> str:
    """Deterministic JSON basename for a per-K0 eval result."""
    return f"eval_k0r4e_{_format_k0(k0_factor)}_{sigma_mapping}.json"


def _aggregate_max_h2o2_in_mask(
    aggregated: Optional[Dict[str, Any]],
) -> Optional[float]:
    if not isinstance(aggregated, dict):
        return None
    val = aggregated.get("max_H2O2_selectivity_in_window_pct")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _json_default(obj: Any) -> Any:
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


def _run_one_k0(
    *,
    k0_factor: float,
    sigma_mapping: str,
    v_grid: Sequence[float],
    v_anchor: float,
    lambda_target: float,
    delta_beta_pm2: float,
    out_dir: str,
    progress_prefix: str,
) -> Dict[str, Any]:
    """Run a single K0_R4e factor eval and write its JSON to ``out_dir``."""
    def _progress(msg: str) -> None:
        print(f"{progress_prefix}{msg}", flush=True)

    t_start = time.time()
    _progress(
        f"=== START K0={k0_factor:.3e}, σ={sigma_mapping}, λ={lambda_target} ==="
    )
    try:
        result = evaluate_delta_beta(
            delta_beta_pm2=delta_beta_pm2,
            sigma_mapping=sigma_mapping,
            v_grid=v_grid,
            v_anchor=v_anchor,
            mode="production",
            progress=_progress,
            lambda_target=lambda_target,
            k0_r4e_factor=k0_factor,
        )
        eval_error: Optional[str] = None
    except Exception as exc:                              # pragma: no cover
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "k0_r4e_factor": float(k0_factor),
            "lambda_target": float(lambda_target),
            "delta_beta_pm2": float(delta_beta_pm2),
            "sigma_mapping": str(sigma_mapping),
        }
        eval_error = f"{type(exc).__name__}: {exc}"

    wall_seconds = time.time() - t_start

    result["k0_sweep_metadata"] = {
        "k0_r4e_factor": float(k0_factor),
        "lambda_target": float(lambda_target),
        "delta_beta_pm2": float(delta_beta_pm2),
        "sigma_mapping": str(sigma_mapping),
        "v_anchor": float(v_anchor),
        "v_grid": [float(v) for v in v_grid],
        "wall_seconds": float(wall_seconds),
        "eval_error": eval_error,
    }

    out_name = _default_eval_name(k0_factor, sigma_mapping)
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
        "k0_r4e_factor": float(k0_factor),
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
        f"=== END   K0={k0_factor:.3e} "
        f"({wall_seconds:.1f}s, max_H2O2%={max_h2o2_pct}, "
        f"argmax_V={argmax_v}, "
        f"gate_pass={n_gate_pass}/{n_gate_pass + n_gate_fail if (n_gate_pass is not None and n_gate_fail is not None) else '?'}) ==="
    )
    return summary_entry


def _build_sweep_summary(
    entries: Sequence[Dict[str, Any]],
    *,
    sigma_mapping: str,
    deck_target_pct: float,
    lambda_target: float,
    delta_beta_pm2: float,
    v_anchor: float,
    v_grid: Sequence[float],
    wall_seconds_total: float,
) -> Dict[str, Any]:
    """Aggregate per-K0 entries into the sweep-level summary.

    Identifies the K0 that minimizes ``loss_vs_deck_pp`` (skipping
    entries with errored evals or missing observables).  Linearly
    bracketing the deck target in log10(K0) space when monotonic.
    """
    sorted_entries = sorted(
        entries, key=lambda e: float(e["k0_r4e_factor"])
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

    k0_star_linear: Optional[float] = None
    bracket: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    # Bracket in log10(K0) space — K0 spans decades.
    for lo, hi in zip(finite_entries[:-1], finite_entries[1:]):
        lo_y = float(lo["max_H2O2_pct_in_mask"])
        hi_y = float(hi["max_H2O2_pct_in_mask"])
        if (lo_y - deck_target_pct) * (hi_y - deck_target_pct) <= 0:
            bracket = (lo, hi)
            lo_log = math.log10(float(lo["k0_r4e_factor"]))
            hi_log = math.log10(float(hi["k0_r4e_factor"]))
            if hi_y == lo_y:
                k0_log_star = 0.5 * (lo_log + hi_log)
            else:
                frac = (deck_target_pct - lo_y) / (hi_y - lo_y)
                k0_log_star = lo_log + frac * (hi_log - lo_log)
            k0_star_linear = 10.0 ** k0_log_star
            break

    return {
        "sigma_mapping": str(sigma_mapping),
        "deck_target_max_H2O2_pct": float(deck_target_pct),
        "lambda_target": float(lambda_target),
        "delta_beta_pm2": float(delta_beta_pm2),
        "v_anchor": float(v_anchor),
        "v_grid": [float(v) for v in v_grid],
        "v_mask": [
            float(V_KIN_OBS_MASK_LO), float(V_KIN_OBS_MASK_HI)
        ],
        "k0_r4e_factor_grid": [
            float(e["k0_r4e_factor"]) for e in sorted_entries
        ],
        "max_H2O2_pct_in_mask_by_k0": [
            e.get("max_H2O2_pct_in_mask") for e in sorted_entries
        ],
        "loss_vs_deck_pp_by_k0": [
            e.get("loss_vs_deck_pp") for e in sorted_entries
        ],
        "best_k0_r4e_factor": (
            float(best_entry["k0_r4e_factor"]) if best_entry else None
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
                "lo_k0": float(bracket[0]["k0_r4e_factor"]),
                "hi_k0": float(bracket[1]["k0_r4e_factor"]),
                "lo_max_H2O2_pct": float(
                    bracket[0]["max_H2O2_pct_in_mask"]
                ),
                "hi_max_H2O2_pct": float(
                    bracket[1]["max_H2O2_pct_in_mask"]
                ),
            }
            if bracket else None
        ),
        "k0_star_log10_interp": k0_star_linear,
        "wall_seconds_total": float(wall_seconds_total),
        "per_k0_entries": list(sorted_entries),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6β K0_R4e factor sweep at λ=1, Δ_β=0, v10b params, "
            "physical a_nondim."
        ),
    )
    parser.add_argument(
        "--k0-factor-grid", type=float, nargs="+",
        default=list(DEFAULT_K0_FACTOR_GRID),
        help=(
            "K0_R4e factor values to evaluate (each multiplies K0_HAT_R4E "
            "to produce the 4e branch's k0 target).  Must be positive.  "
            f"Default: {DEFAULT_K0_FACTOR_GRID}."
        ),
    )
    parser.add_argument(
        "--lambda-target", type=float, default=1.0,
        help=(
            "λ_hydrolysis target (held fixed across the K0 sweep).  "
            "Default: 1.0 (full hydrolysis)."
        ),
    )
    parser.add_argument(
        "--delta-beta-pm2", type=float, default=0.0,
        help=(
            "Δ_β (carbon-vs-Cu offset, pm²).  Default: 0.0 (Singh Cu prior)."
        ),
    )
    parser.add_argument(
        "--sigma-mapping", type=str, default=SIGMA_MAPPING_STERN,
        choices=("stern", "ablation_singh_0.141"),
        help=f"σ-mapping convention.  Default: {SIGMA_MAPPING_STERN}.",
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
        help=f"Anchor voltage (V vs RHE).  Default: {V_ANCHOR}.",
    )
    parser.add_argument(
        "--deck-target-pct", type=float,
        default=DECK_K_AT_PH4_MEAN_H2O2_PCT,
        help=(
            f"Deck max H₂O₂%% target.  Default: "
            f"{DECK_K_AT_PH4_MEAN_H2O2_PCT} pp (K₂SO₄ pH 4 mean)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    ns = _parse_args(argv)

    for k in ns.k0_factor_grid:
        if not (float(k) > 0.0):
            raise ValueError(
                f"--k0-factor-grid entries must be > 0 (got {k!r})"
            )
    k0_grid: Tuple[float, ...] = tuple(
        sorted(set(float(k) for k in ns.k0_factor_grid))
    )

    out_dir = os.path.join(_ROOT, "StudyResults", ns.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78, flush=True)
    print(
        f"  Phase 6β K0_R4e factor sweep — λ={ns.lambda_target}, "
        f"Δ_β={ns.delta_beta_pm2}, σ={ns.sigma_mapping}",
        flush=True,
    )
    print("=" * 78, flush=True)
    print(
        f"  K0 grid       = {[f'{k:.2e}' for k in k0_grid]}",
        flush=True,
    )
    print(
        f"  v10b ref K0   = {K0_R4E_FACTOR_V10B:.2e}  "
        f"(current Phase D / v10b production value)",
        flush=True,
    )
    print(f"  V_anchor      = {ns.v_anchor:+.3f} V", flush=True)
    print(
        f"  V grid        = [{V_RHE_PRODUCTION_GRID[0]:+.3f}, "
        f"{V_RHE_PRODUCTION_GRID[-1]:+.3f}] V "
        f"({len(V_RHE_PRODUCTION_GRID)} pts)",
        flush=True,
    )
    print(
        f"  V mask        = [{V_KIN_OBS_MASK_LO:+.3f}, "
        f"{V_KIN_OBS_MASK_HI:+.3f}] V (deck overlap)",
        flush=True,
    )
    print(f"  deck target   = {ns.deck_target_pct:.2f} pp", flush=True)
    print(f"  output dir    = {out_dir}", flush=True)

    t_total = time.time()
    entries: List[Dict[str, Any]] = []
    for i, k0 in enumerate(k0_grid):
        prefix = f"  [K0 {i + 1}/{len(k0_grid)} = {k0:.2e}] "
        entry = _run_one_k0(
            k0_factor=float(k0),
            sigma_mapping=ns.sigma_mapping,
            v_grid=V_RHE_PRODUCTION_GRID,
            v_anchor=float(ns.v_anchor),
            lambda_target=float(ns.lambda_target),
            delta_beta_pm2=float(ns.delta_beta_pm2),
            out_dir=out_dir,
            progress_prefix=prefix,
        )
        entries.append(entry)

    summary = _build_sweep_summary(
        entries,
        sigma_mapping=ns.sigma_mapping,
        deck_target_pct=float(ns.deck_target_pct),
        lambda_target=float(ns.lambda_target),
        delta_beta_pm2=float(ns.delta_beta_pm2),
        v_anchor=float(ns.v_anchor),
        v_grid=V_RHE_PRODUCTION_GRID,
        wall_seconds_total=time.time() - t_total,
    )

    summary_path = os.path.join(out_dir, "k0_r4e_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print()
    print("=" * 78, flush=True)
    print("  Sweep summary", flush=True)
    print("=" * 78, flush=True)
    print(f"  total wall:                 {summary['wall_seconds_total']:.1f}s", flush=True)
    for e in summary["per_k0_entries"]:
        k0 = e["k0_r4e_factor"]
        mh = e.get("max_H2O2_pct_in_mask")
        loss = e.get("loss_vs_deck_pp")
        mh_str = f"{mh:7.3f}" if mh is not None else "  None "
        loss_str = f"{loss:7.3f}" if loss is not None else "  None "
        argv_str = f"{e.get('argmax_V_for_selectivity')}"
        print(
            f"  K0={k0:.2e}: max_H2O2%={mh_str} pp  loss={loss_str} pp  argmax_V={argv_str}",
            flush=True,
        )
    print(f"  best K0:                    {summary['best_k0_r4e_factor']}", flush=True)
    print(f"  best loss:                  {summary['best_loss_vs_deck_pp']} pp", flush=True)
    print(
        f"  K0* (log10 bracket):        {summary['k0_star_log10_interp']}",
        flush=True,
    )
    print(f"  summary JSON:               {summary_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
