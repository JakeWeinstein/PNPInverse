"""Run D verdict — Mangan deck p.15 tolerance bands.

Loads Run C JSON output (StudyResults/mangan_p15_comparison/run_C/iv_curve.json)
and the digitised target (data/mangan_deck_p15_h2o2_current.csv), computes the
five B7 tolerance bands, classifies the verdict outcome, and writes
StudyResults/mangan_p15_comparison/run_D_verdict.md.

B7 tolerance bands (semi_quant tier):

    Peak voltage           |  exp ~ +0.10 V   |  +/- 50 mV
    Peak magnitude         |  exp ~ -0.40     |  +/- 25%
    Left plateau (V=-0.32) |  exp ~ -0.17     |  +/- 25%
    Onset to zero          |  exp ~ +0.45 V   |  +/- 50 mV
    Shoulder in [+0.18, +0.30]              |  qualitative yes/no

Verdict outcomes (Plan B§"Run D — Verdict"):

    all_bands_met        ->  M2 = option 1 (keep surrogate); unlikely
    magnitude_off_shape_correct
                         ->  M2 candidates option 2 (cheap diagnostic)
                             then option 3 (real fix); most likely
    shape_wrong          ->  Debug forward solver before declaring M2 needed
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

import numpy as np


RUN_C_JSON = os.path.join(_ROOT, "StudyResults/mangan_p15_comparison/run_C/iv_curve.json")
EXPERIMENTAL_CSV = os.path.join(_ROOT, "data/mangan_deck_p15_h2o2_current.csv")
VERDICT_MD = os.path.join(_ROOT, "StudyResults/mangan_p15_comparison/run_D_verdict.md")

# Digitised features from page 15 (per docs/m0_target_extraction.md B2).
EXP_PEAK_V = 0.10            # +/- 50 mV band
EXP_PEAK_J = -0.40           # +/- 25% band
EXP_LEFT_PLATEAU_V = -0.32
EXP_LEFT_PLATEAU_J = -0.17   # +/- 25% band
EXP_ONSET_V = 0.45           # +/- 50 mV band; defined as where j first crosses 0 from below
SHOULDER_V_RANGE = (0.18, 0.30)

PEAK_V_TOL = 0.050           # V
PEAK_J_TOL_REL = 0.25
PLATEAU_J_TOL_REL = 0.25
ONSET_V_TOL = 0.050          # V


def _load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    v_list: list[float] = []
    j_list: list[float] = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                continue
            v_list.append(float(row[0]))
            j_list.append(float(row[1]))
    return np.array(v_list, dtype=float), np.array(j_list, dtype=float)


def _interp_at(v_target: float, v: np.ndarray, j: np.ndarray) -> float:
    """Linear interpolation of j(v_target).  v must be monotonically
    increasing on the points where j is finite.
    """
    finite = np.isfinite(j)
    if finite.sum() < 2:
        return float("nan")
    return float(np.interp(v_target, v[finite], j[finite]))


def _onset_voltage(v: np.ndarray, j: np.ndarray, threshold: float = 0.0) -> float:
    """Find the V where j(V) first crosses threshold from below (j becoming
    less negative).  Returns NaN if no crossing detected.
    """
    finite = np.isfinite(j)
    vf = v[finite]
    jf = j[finite]
    if len(jf) < 2:
        return float("nan")
    # Ascending v
    order = np.argsort(vf)
    vf = vf[order]
    jf = jf[order]
    # Find the right-most index where j is below threshold and the next is above.
    for i in range(len(jf) - 1):
        if jf[i] < threshold and jf[i + 1] >= threshold:
            # Linear interp between (vf[i], jf[i]) and (vf[i+1], jf[i+1]).
            if jf[i + 1] == jf[i]:
                return float(vf[i + 1])
            frac = (threshold - jf[i]) / (jf[i + 1] - jf[i])
            return float(vf[i] + frac * (vf[i + 1] - vf[i]))
    return float("nan")


def _peak_estimate(v: np.ndarray, j: np.ndarray) -> tuple[float, float]:
    """Most-negative-j point on (v, j) with NaN handling.  Uses parabolic
    refinement around the discrete minimum when possible.
    """
    finite = np.isfinite(j)
    if not finite.any():
        return float("nan"), float("nan")
    vf = v[finite]
    jf = j[finite]
    order = np.argsort(vf)
    vf = vf[order]
    jf = jf[order]
    idx = int(np.argmin(jf))
    # Parabolic refinement around the discrete min
    if 1 <= idx <= len(vf) - 2:
        x0, x1, x2 = vf[idx - 1], vf[idx], vf[idx + 1]
        y0, y1, y2 = jf[idx - 1], jf[idx], jf[idx + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denom != 0.0:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
            if a > 0:  # downward-opening (peak in -j)
                v_peak = -b / (2 * a)
                if x0 < v_peak < x2:
                    j_peak = float(np.interp(v_peak, vf, jf))
                    return float(v_peak), j_peak
    return float(vf[idx]), float(jf[idx])


def _shoulder_present(
    v: np.ndarray, j: np.ndarray, lo: float, hi: float, peak_v: float, peak_j: float,
) -> tuple[bool, str]:
    """Heuristic shoulder detection in v in [lo, hi]: looks for a slope
    flattening (an inflection or local plateau) between the peak voltage
    and the onset voltage, where dj/dV transitions from negative-to-zero
    and stays small over a small range.

    Returns (is_present_bool, description_string).
    """
    finite = np.isfinite(j)
    vf = v[finite]
    jf = j[finite]
    order = np.argsort(vf)
    vf = vf[order]
    jf = jf[order]
    in_window = (vf >= lo) & (vf <= hi)
    if in_window.sum() < 3:
        return False, f"insufficient samples in [{lo:.2f}, {hi:.2f}] V"
    vw = vf[in_window]
    jw = jf[in_window]
    # First derivative
    djdv = np.gradient(jw, vw)
    # Look for sign change or flatness: shoulder is where the descent
    # slowed substantially before climbing (an inflection between the
    # peak and the onset).  Simple proxy: |dj/dV| has a local minimum
    # in the window and j has not yet climbed back above 50% of
    # |peak_j|.
    abs_slope = np.abs(djdv)
    if len(abs_slope) >= 3:
        # Find a local-min of |slope| not at the boundary
        for i in range(1, len(abs_slope) - 1):
            if abs_slope[i] < abs_slope[i - 1] and abs_slope[i] < abs_slope[i + 1]:
                if jw[i] < 0.5 * peak_j:  # still meaningfully cathodic
                    return True, (
                        f"local |dj/dV| minimum at V={vw[i]:+.3f} V "
                        f"with j={jw[i]:+.3f} mA/cm^2"
                    )
    return False, "no inflection / slope flattening detected"


def main() -> None:
    if not os.path.exists(RUN_C_JSON):
        print(f"ERROR: Run C JSON not found at {RUN_C_JSON}")
        sys.exit(2)
    if not os.path.exists(EXPERIMENTAL_CSV):
        print(f"ERROR: experimental CSV not found at {EXPERIMENTAL_CSV}")
        sys.exit(2)

    with open(RUN_C_JSON) as f:
        run_c = json.load(f)
    em = run_c.get("experiment_metadata", {})
    rep = run_c.get("report", {})
    v_model = np.array(rep["v_rhe"], dtype=float)
    pc_model = np.array(
        [np.nan if x is None else x for x in rep["pc_mA_cm2"]],
        dtype=float,
    )
    converged = list(rep.get("converged", []))
    n_total = int(rep.get("n_total", len(v_model)))
    n_conv = int(rep.get("n_converged", sum(bool(x) for x in converged)))

    exp_v, exp_j = _load_csv(EXPERIMENTAL_CSV)

    # 1. Peak (model)
    model_peak_v, model_peak_j = _peak_estimate(v_model, pc_model)

    # 2. Left plateau magnitude at V_RHE ~ -0.32 V (interpolate model)
    model_left_plateau_j = _interp_at(EXP_LEFT_PLATEAU_V, v_model, pc_model)

    # 3. Onset position (model crosses 0 from below)
    model_onset_v = _onset_voltage(v_model, pc_model, threshold=0.0)

    # 4. Shoulder visibility
    shoulder_ok, shoulder_desc = _shoulder_present(
        v_model, pc_model,
        SHOULDER_V_RANGE[0], SHOULDER_V_RANGE[1],
        model_peak_v, model_peak_j,
    )

    # Tolerance band evaluations
    bands: list[dict[str, Any]] = []

    def _abs_rel_err(model_val: float, exp_val: float) -> float:
        if not (np.isfinite(model_val) and np.isfinite(exp_val)) or exp_val == 0:
            return float("nan")
        return abs(model_val - exp_val) / abs(exp_val)

    def _abs_err(model_val: float, exp_val: float) -> float:
        if not (np.isfinite(model_val) and np.isfinite(exp_val)):
            return float("nan")
        return abs(model_val - exp_val)

    bands.append({
        "feature": "peak voltage",
        "exp": EXP_PEAK_V, "model": model_peak_v,
        "abs_err_V": _abs_err(model_peak_v, EXP_PEAK_V),
        "tolerance": PEAK_V_TOL,
        "tolerance_unit": "V (absolute)",
        "passes": (np.isfinite(model_peak_v) and abs(model_peak_v - EXP_PEAK_V) <= PEAK_V_TOL),
    })
    bands.append({
        "feature": "peak magnitude",
        "exp": EXP_PEAK_J, "model": model_peak_j,
        "rel_err": _abs_rel_err(model_peak_j, EXP_PEAK_J),
        "tolerance": PEAK_J_TOL_REL,
        "tolerance_unit": "relative (|err|/|exp|)",
        "passes": (
            np.isfinite(model_peak_j)
            and _abs_rel_err(model_peak_j, EXP_PEAK_J) <= PEAK_J_TOL_REL
        ),
    })
    bands.append({
        "feature": f"left plateau magnitude at V={EXP_LEFT_PLATEAU_V:+.2f} V",
        "exp": EXP_LEFT_PLATEAU_J, "model": model_left_plateau_j,
        "rel_err": _abs_rel_err(model_left_plateau_j, EXP_LEFT_PLATEAU_J),
        "tolerance": PLATEAU_J_TOL_REL,
        "tolerance_unit": "relative (|err|/|exp|)",
        "passes": (
            np.isfinite(model_left_plateau_j)
            and _abs_rel_err(model_left_plateau_j, EXP_LEFT_PLATEAU_J) <= PLATEAU_J_TOL_REL
        ),
    })
    bands.append({
        "feature": "onset to zero",
        "exp": EXP_ONSET_V, "model": model_onset_v,
        "abs_err_V": _abs_err(model_onset_v, EXP_ONSET_V),
        "tolerance": ONSET_V_TOL,
        "tolerance_unit": "V (absolute)",
        "passes": (np.isfinite(model_onset_v) and abs(model_onset_v - EXP_ONSET_V) <= ONSET_V_TOL),
    })
    bands.append({
        "feature": f"shoulder in V_RHE in {SHOULDER_V_RANGE}",
        "exp": "yes (visible on page 15)",
        "model": "yes" if shoulder_ok else "no",
        "model_description": shoulder_desc,
        "tolerance": "qualitative",
        "passes": shoulder_ok,
    })

    n_pass = sum(1 for b in bands if b["passes"])
    n_quant = 4
    quant_pass = sum(1 for b in bands[:4] if b["passes"])

    # Verdict classification
    shape_correct = (
        bands[0]["passes"]    # peak voltage
        and bands[3]["passes"] # onset
        and bands[4]["passes"] # shoulder
    )
    if quant_pass == 4 and bands[4]["passes"]:
        outcome = "all_bands_met"
        m2_recommendation = (
            "M2 = option 1 (keep `pH_countercharge_surrogate`).  Surprising "
            "result -- screening details are not load-bearing for the page 15 "
            "comparison.  Revisit on a multi-cation sweep or a target with "
            "richer EDL physics."
        )
    elif shape_correct and not bands[1]["passes"]:
        outcome = "magnitude_off_shape_correct"
        m2_recommendation = (
            "M2 = option 3 (steric multi-ion analytic closure) is the real "
            "fix.  Option 2 (ideal salt pair) is a useful intermediate "
            "diagnostic to validate that the solver tolerates 1-nm Debye "
            "conditioning before adding the multi-ion bookkeeping.  "
            "Magnitude-off / shape-correct diagnoses missing EDL screening "
            "at deck-correct ionic strength."
        )
    else:
        outcome = "shape_wrong"
        m2_recommendation = (
            "Debug forward solver before declaring M2 needed.  Diagnostic "
            "checklist: BV constants vs CLAUDE.md production values, "
            "clip=100 active in this run, IC stability post-M1.5, mesh "
            "resolution, residual-side / IC consistency on the saturated "
            "Bikerman manifold."
        )

    # Stamp the verdict markdown
    lines: list[str] = []
    lines.append("# Run D Verdict — Mangan Deck Page 15 Comparison")
    lines.append("")
    lines.append("Plan B §\"Run D — Verdict\".  Compares the model from "
                 "`StudyResults/mangan_p15_comparison/run_C/iv_curve.json` "
                 "against `data/mangan_deck_p15_h2o2_current.csv` (digitised "
                 "page 15) using the B7 tolerance bands.")
    lines.append("")
    lines.append("## Run C metadata")
    lines.append("")
    lines.append("```")
    for k in (
        "catalyst", "geometry", "pH_bulk", "cation", "anion_model",
        "rotation_rate_rpm", "L_eff_m", "N_collection",
        "electrolyte_model", "comparison_status", "source_authority",
        "target_curve", "acceptance_tier",
    ):
        if k in em:
            lines.append(f"{k:<22s} = {em[k]}")
    lines.append(f"converged              = {n_conv} / {n_total}")
    lines.append("```")
    lines.append("")
    lines.append("## Tolerance band measurements")
    lines.append("")
    lines.append("| # | Feature | Experimental | Model | Error | Tolerance | Pass |")
    lines.append("|---|---------|--------------|-------|-------|-----------|------|")
    for i, b in enumerate(bands, start=1):
        feat = b["feature"]
        exp = b["exp"]
        model = b["model"]
        if b["tolerance"] == "qualitative":
            err_s = b.get("model_description", "")
            tol_s = "qualitative"
        elif "rel_err" in b:
            err = b["rel_err"]
            err_s = f"{err*100:.1f}%" if np.isfinite(err) else "nan"
            tol_s = f"{b['tolerance']*100:.0f}% (rel)"
        elif "abs_err_V" in b:
            err = b["abs_err_V"]
            err_s = f"{err*1000:.1f} mV" if np.isfinite(err) else "nan"
            tol_s = f"{b['tolerance']*1000:.0f} mV"
        else:
            err_s = "?"
            tol_s = "?"
        passes_s = "PASS" if b["passes"] else "FAIL"
        lines.append(
            f"| {i} | {feat} | {exp} | {model} | {err_s} | {tol_s} | {passes_s} |"
        )
    lines.append("")
    lines.append(f"**Quantitative bands passing**: {quant_pass} / {n_quant}")
    lines.append(f"**Total bands passing (incl. shoulder)**: {n_pass} / 5")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"`outcome = \"{outcome}\"`")
    lines.append("")
    lines.append("## M2 decision implication")
    lines.append("")
    lines.append(m2_recommendation)
    lines.append("")
    lines.append("## Cross-references")
    lines.append("")
    lines.append("- Plan: `~/.claude/plans/swirling-crunching-wren.md`")
    lines.append("- M0 extraction: `docs/m0_target_extraction.md`")
    lines.append("- Run C output: `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`, "
                 "`comparison.png`")
    lines.append("- Digitised target: `data/mangan_deck_p15_h2o2_current.csv`")
    lines.append("- M1 deferred-parameter convention: "
                 "`memory/project_mangan_m1_deferred_parameters.md`")
    lines.append("")

    os.makedirs(os.path.dirname(VERDICT_MD), exist_ok=True)
    with open(VERDICT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"Run D verdict written to {VERDICT_MD}")
    print()
    print(f"Outcome: {outcome}")
    print(f"Quantitative bands passing: {quant_pass}/{n_quant}; total: {n_pass}/5")
    print()
    for b in bands:
        print(f"  {b['feature']:<48s}: {'PASS' if b['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
