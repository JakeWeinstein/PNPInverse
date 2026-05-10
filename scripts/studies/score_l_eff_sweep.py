"""L_eff sweep shape-feature scorer — verdict.json with falsifiable
prediction pass/fail per ``.claude/plans/l-eff-transport-sweep.md``.

Per-combo features extracted:

  - ``plateau_cathodic_mA_cm2``      : cd at the deepest converged
                                       cathodic V_RHE (target: -0.18 mA/cm²
                                       deck baseline).
  - ``plateau_target_residual_pct``  : 100 * (model - target) / target
                                       (negative = model too small).
  - ``has_peak``                     : True if cd has a local cathodic
                                       maximum strictly inside the V_RHE
                                       grid (not just at an endpoint).
  - ``peak_v_rhe`` / ``peak_magnitude_mA_cm2`` /
    ``peak_target_residual_pct`` : the located peak (or None) compared
                                       against the deck peak (-0.40 mA/cm²
                                       at +0.10 V).
  - ``v_decay_zero``                 : V_RHE at which |cd| first crosses
                                       below ``DECAY_THRESHOLD_MA_CM2``
                                       moving anodic from the plateau.
  - ``max_surface_pH``               : maximum surface pH proxy across
                                       converged points (target: < 9 at
                                       smallest L_eff).
  - ``surface_pH_plausibility_pass`` : ``max_surface_pH < 9.0``.
  - ``anchor_converged`` / ``n_grid_converged`` : run-quality fields.

Top-level verdict per falsifiable prediction (plan §"Falsifiable
predictions"):

  1. Levich linearity: slope of ``log(|plateau|)`` vs ``log(1/L_eff)``
     in [0.85, 1.15] for each ratio.
  2. No peak: ``has_peak`` is False at every (L_eff, ratio).
  3. Surface pH < 9: ``max_surface_pH < 9.0`` at the smallest L_eff
     across both ratios.

If 1+2+3 PASS -> GPT's diagnosis confirmed; next step is local-pH/buffer
chemistry (Phase 6α).  Per-prediction failures map to investigation
branches (plan §"Decision tree").

No Firedrake imports — pure JSON / numpy.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DEFAULT_SWEEP_DIR = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08"
    / "l_eff_transport_sweep"
)
DEFAULT_VERDICT = DEFAULT_SWEEP_DIR / "verdict.json"

DECK_LEFT_PLATEAU_MA_CM2 = -0.18
DECK_PEAK_MA_CM2 = -0.40
DECK_PEAK_V_RHE = +0.10
DECAY_THRESHOLD_MA_CM2 = 0.01     # |cd| < 0.01 = effectively zero
PEAK_LOCAL_TOL_MA_CM2 = 1e-4      # ignore numerical wiggles below this
SLOPE_LO, SLOPE_HI = 0.85, 1.15   # Levich linearity tolerance band
SURFACE_PH_THRESHOLD = 9.0


def _arr(data: dict, key: str) -> np.ndarray:
    return np.array(
        [x if x is not None else np.nan for x in data.get(key, [])],
        dtype=float,
    )


def _combo_label(l_eff_m: float, ratio: float) -> str:
    return f"L{round(l_eff_m * 1e6)}um_ratio_{ratio:g}"


def _load_combo(sweep_dir: Path, l_eff_m: float, ratio: float) -> dict | None:
    path = sweep_dir / _combo_label(l_eff_m, ratio) / "iv_curve.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-combo feature extraction
# ---------------------------------------------------------------------------


def _plateau_at_min_v_rhe(data: dict) -> tuple[float | None, float | None]:
    """Return (V_RHE, cd) at the most cathodic converged point."""
    v_rhe = np.array(data["v_rhe"], dtype=float)
    cd = _arr(data, "cd_mA_cm2")
    converged = np.array(data["converged"], dtype=bool)
    mask = np.isfinite(cd) & converged
    if not mask.any():
        return None, None
    idx = np.where(mask)[0]
    j = idx[np.argmin(v_rhe[idx])]
    return float(v_rhe[j]), float(cd[j])


def _detect_interior_peak(data: dict) -> tuple[bool, float | None, float | None]:
    """Return (has_peak, peak_v_rhe, peak_magnitude_mA_cm2).

    A peak is a strict local minimum of the cd array (most-cathodic =
    most-negative) that is NOT at the endpoints — i.e. cd dips below
    both the leftmost and rightmost neighbors of the interior point by
    at least ``PEAK_LOCAL_TOL_MA_CM2``.
    """
    v_rhe = np.array(data["v_rhe"], dtype=float)
    cd = _arr(data, "cd_mA_cm2")
    converged = np.array(data["converged"], dtype=bool)
    mask = np.isfinite(cd) & converged
    if mask.sum() < 3:
        return False, None, None

    v_arr = v_rhe[mask]
    cd_arr = cd[mask]
    sort_idx = np.argsort(v_arr)
    v_sorted = v_arr[sort_idx]
    cd_sorted = cd_arr[sort_idx]

    # Interior local min in cd (most-negative is most-cathodic peak).
    best_idx = None
    best_cd = 0.0
    for i in range(1, len(cd_sorted) - 1):
        left = cd_sorted[i - 1]
        right = cd_sorted[i + 1]
        if (cd_sorted[i] < left - PEAK_LOCAL_TOL_MA_CM2
                and cd_sorted[i] < right - PEAK_LOCAL_TOL_MA_CM2):
            if cd_sorted[i] < best_cd:
                best_cd = float(cd_sorted[i])
                best_idx = i
    if best_idx is None:
        return False, None, None
    return True, float(v_sorted[best_idx]), float(cd_sorted[best_idx])


def _v_decay_zero(data: dict) -> float | None:
    """Return the most cathodic V_RHE at which |cd| first drops below
    ``DECAY_THRESHOLD_MA_CM2`` while sweeping anodically from the deepest
    cathodic converged point.  ``None`` if no such crossing exists."""
    v_rhe = np.array(data["v_rhe"], dtype=float)
    cd = _arr(data, "cd_mA_cm2")
    converged = np.array(data["converged"], dtype=bool)
    mask = np.isfinite(cd) & converged
    if mask.sum() < 2:
        return None
    v_arr = v_rhe[mask]
    cd_arr = cd[mask]
    sort_idx = np.argsort(v_arr)
    v_sorted = v_arr[sort_idx]
    abs_cd = np.abs(cd_arr[sort_idx])
    for i in range(1, len(v_sorted)):
        if abs_cd[i - 1] >= DECAY_THRESHOLD_MA_CM2 and abs_cd[i] < DECAY_THRESHOLD_MA_CM2:
            return float(v_sorted[i])
    if abs_cd[-1] < DECAY_THRESHOLD_MA_CM2:
        return float(v_sorted[-1])
    return None


def _max_surface_pH(data: dict) -> float | None:
    pH = _arr(data, "surface_pH_proxy")
    converged = np.array(data["converged"], dtype=bool)
    mask = np.isfinite(pH) & converged
    if not mask.any():
        return None
    return float(np.nanmax(pH[mask]))


def _score_combo(data: dict) -> dict[str, Any]:
    plateau_v, plateau_cd = _plateau_at_min_v_rhe(data)
    if plateau_cd is None:
        plateau_pct = None
    else:
        plateau_pct = float(
            100.0 * (plateau_cd - DECK_LEFT_PLATEAU_MA_CM2)
            / DECK_LEFT_PLATEAU_MA_CM2
        )

    has_peak, peak_v, peak_mag = _detect_interior_peak(data)
    if peak_mag is not None:
        peak_pct = float(
            100.0 * (peak_mag - DECK_PEAK_MA_CM2) / DECK_PEAK_MA_CM2
        )
    else:
        peak_pct = None

    max_pH = _max_surface_pH(data)
    pH_pass = (max_pH is not None) and (max_pH < SURFACE_PH_THRESHOLD)

    n_grid_converged = int(sum(1 for c in data.get("converged", []) if c))

    return {
        "L_eff_m": float(data.get("l_eff_m", float("nan"))),
        "L_eff_um": float(data.get("l_eff_m", float("nan"))) * 1e6,
        "ratio": float(data.get("ratio", float("nan"))),
        "plateau_v_rhe": plateau_v,
        "plateau_cathodic_mA_cm2": plateau_cd,
        "plateau_target_residual_pct": plateau_pct,
        "has_peak": bool(has_peak),
        "peak_v_rhe": peak_v,
        "peak_magnitude_mA_cm2": peak_mag,
        "peak_target_residual_pct": peak_pct,
        "v_decay_zero": _v_decay_zero(data),
        "max_surface_pH": max_pH,
        "surface_pH_plausibility_pass": bool(pH_pass),
        "anchor_converged": bool(
            data.get("anchor", {}).get("converged", False)
        ),
        "n_grid_converged": n_grid_converged,
        "n_grid_total": int(data.get("n_total", 0)),
    }


# ---------------------------------------------------------------------------
# Falsifiable verdict logic
# ---------------------------------------------------------------------------


def _levich_slope_per_ratio(
    per_combo: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fit log(|plateau|) vs log(1/L_eff) for each ratio.  Returns a
    nested dict keyed by ``str(ratio)`` with ``slope``, ``log10_intercept``,
    ``in_band``, ``L_eff_um_used``, ``plateau_used_mA_cm2``, ``n_used``.
    """
    by_ratio: dict[float, list[tuple[float, float]]] = {}
    for entry in per_combo:
        cd = entry.get("plateau_cathodic_mA_cm2")
        L = entry.get("L_eff_m")
        if cd is None or not np.isfinite(cd) or cd >= 0:
            continue
        if L is None or not np.isfinite(L) or L <= 0:
            continue
        by_ratio.setdefault(float(entry["ratio"]), []).append(
            (float(L), float(cd))
        )

    out: dict[str, dict[str, Any]] = {}
    for ratio, pts in by_ratio.items():
        if len(pts) < 2:
            out[f"{ratio:g}"] = {
                "slope": None,
                "in_band": False,
                "n_used": len(pts),
                "reason": "fewer than 2 plateau points",
            }
            continue
        L = np.array([p[0] for p in pts], dtype=float)
        plateau = np.array([p[1] for p in pts], dtype=float)
        log_invL = np.log(1.0 / L)
        log_p = np.log(np.abs(plateau))
        slope, intercept = np.polyfit(log_invL, log_p, 1)
        in_band = bool(SLOPE_LO <= slope <= SLOPE_HI)
        out[f"{ratio:g}"] = {
            "slope": float(slope),
            "log_intercept": float(intercept),
            "in_band": in_band,
            "n_used": int(len(pts)),
            "L_eff_um_used": [float(v * 1e6) for v in L],
            "plateau_used_mA_cm2": [float(v) for v in plateau],
            "slope_band": [SLOPE_LO, SLOPE_HI],
        }
    return out


def _smallest_l_eff_pH_pass(
    per_combo: list[dict[str, Any]],
) -> dict[str, Any]:
    """At the smallest L_eff in the sweep, do BOTH ratios show
    ``max_surface_pH < SURFACE_PH_THRESHOLD``?  That's prediction 3.
    """
    converged_combos = [
        e for e in per_combo
        if (e.get("anchor_converged") and e.get("max_surface_pH") is not None)
    ]
    if not converged_combos:
        return {
            "pass": False,
            "reason": "no combo with a populated max_surface_pH",
        }
    smallest = min(converged_combos, key=lambda e: e["L_eff_m"])
    smallest_L_um = smallest["L_eff_um"]
    siblings = [e for e in converged_combos if e["L_eff_um"] == smallest_L_um]
    all_pass = all(
        e["max_surface_pH"] is not None
        and e["max_surface_pH"] < SURFACE_PH_THRESHOLD
        for e in siblings
    )
    return {
        "pass": bool(all_pass),
        "smallest_L_eff_um": float(smallest_L_um),
        "siblings": [
            {
                "ratio": e["ratio"],
                "max_surface_pH": e["max_surface_pH"],
                "passes_threshold": bool(
                    e["max_surface_pH"] is not None
                    and e["max_surface_pH"] < SURFACE_PH_THRESHOLD
                ),
            }
            for e in siblings
        ],
        "threshold": SURFACE_PH_THRESHOLD,
    }


def build_verdict(sweep_dir: Path) -> dict[str, Any]:
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {sweep_dir}")
    with open(summary_path) as f:
        summary = json.load(f)

    per_combo: list[dict[str, Any]] = []
    for combo_summary in summary.get("per_combo", []):
        l_eff_m = float(combo_summary["l_eff_m"])
        ratio = float(combo_summary["ratio"])
        data = _load_combo(sweep_dir, l_eff_m, ratio)
        if data is None:
            per_combo.append({
                "L_eff_m": l_eff_m,
                "L_eff_um": l_eff_m * 1e6,
                "ratio": ratio,
                "missing": True,
            })
            continue
        per_combo.append(_score_combo(data))

    # Prediction 1: Levich linearity per ratio.
    levich = _levich_slope_per_ratio(per_combo)
    p1_pass = bool(levich) and all(v.get("in_band", False) for v in levich.values())

    # Prediction 2: no peak anywhere.
    any_peak = any(e.get("has_peak", False) for e in per_combo)
    p2_pass = not any_peak

    # Prediction 3: pH < 9 at smallest L_eff for both ratios.
    p3 = _smallest_l_eff_pH_pass(per_combo)

    overall_pass = p1_pass and p2_pass and bool(p3.get("pass", False))

    verdict: dict[str, Any] = {
        "overall_pass": bool(overall_pass),
        "predictions": {
            "1_plateau_levich_linear": {
                "pass": bool(p1_pass),
                "per_ratio": levich,
                "criterion": (
                    "log(|plateau|) vs log(1/L_eff) slope in "
                    f"[{SLOPE_LO}, {SLOPE_HI}] for each ratio"
                ),
            },
            "2_no_peak": {
                "pass": bool(p2_pass),
                "any_peak": bool(any_peak),
                "criterion": (
                    "no interior local minimum of cd at any (L_eff, ratio)"
                ),
            },
            "3_pH_under_9_at_smallest_L": {
                **p3,
                "criterion": (
                    f"max_surface_pH < {SURFACE_PH_THRESHOLD} at smallest "
                    "L_eff for both ratios"
                ),
            },
        },
        "deck_targets": {
            "left_plateau_mA_cm2": DECK_LEFT_PLATEAU_MA_CM2,
            "peak_mA_cm2": DECK_PEAK_MA_CM2,
            "peak_v_rhe": DECK_PEAK_V_RHE,
            "decay_threshold_mA_cm2": DECAY_THRESHOLD_MA_CM2,
        },
        "per_combo": per_combo,
    }
    return verdict


def main(argv: list[str]) -> int:
    sweep_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_SWEEP_DIR
    out_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_VERDICT
    try:
        verdict = build_verdict(sweep_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(verdict, f, indent=2)

    print("=" * 78)
    print("  L_eff sweep verdict")
    print("=" * 78)
    overall = verdict["overall_pass"]
    print(f"  overall_pass: {overall}")
    for pred_key, pred_val in verdict["predictions"].items():
        flag = "PASS" if pred_val.get("pass", False) else "FAIL"
        print(f"    [{flag}] {pred_key}: {pred_val.get('criterion')}")
    print(f"  output -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
