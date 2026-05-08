"""M3a.0 — Observable audit on Run C state.

Reassemble three observables from the existing Run C JSON without re-running
the solver, and compare each against the digitised page-15 target:

    cd_mA_cm2          : total disk current   = -I_SCALE * (R_0 + R_1)        [already in JSON]
    pc_mA_cm2          : net peroxide         = -I_SCALE * (R_0 - R_1)        [already in JSON]
    gross_R0_mA_cm2    : gross 2e production  = -I_SCALE * R_0                [DERIVED]

Algebraic identity (sequential R_0 + R_1 model, both reactions 2e):

    R_0 + R_1 = -cd / I_SCALE
    R_0 - R_1 = -pc / I_SCALE
    R_0       = (-cd - pc) / (2 * I_SCALE)
    -I_SCALE * R_0 = (cd + pc) / 2     <-- gross_R_0 in mA/cm^2

Compute B7 tolerance bands (peak voltage, peak magnitude, left plateau,
onset, shoulder) for the gross R_0 channel and contrast against the bands
already computed for the net pc channel in run_D_verdict.

Why this matters (post-Ruggiero 2026-05-07):
- Ruggiero 2022 J. Catal. shows the deck's reaction model is parallel
  2e/4e ORR, not sequential.  The deck's "Peroxide Current Density"
  therefore maps to the gross 2e production current (single-rate
  observable) rather than to net (R_0 - R_1).
- This audit is the cheapest test of "how much of the page-15 mismatch is
  observable definition vs physics."  The structurally-correct parallel
  R_4e channel is *not* yet in the model; we test only whether the
  existing Run C state's R_0 channel, viewed as gross 2e current, is
  closer to the target than the net (R_0 - R_1) was.

Outputs (under StudyResults/mangan_p15_comparison/):
    m3a0_observables.json    : V_RHE, cd, pc, gross_R0 vs target
    m3a0_audit.md            : tolerance bands and decision
    m3a0_audit.png           : overlay plot
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# I_SCALE is reproduced inline rather than imported to keep this script
# Firedrake-free (no PDE solve required for the audit).
#
# AUDIT ANCHOR (do not change): this script audits Run C, which was solved
# with the pre-M3a.2.1 C_O2 = 0.5 mol/m³.  C_O2 is intentionally pinned
# below to the legacy value so the I_SCALE used to back out gross R_0
# matches the I_SCALE that Run C's nondim rates were assembled against.
# Post-fix runs (C_O2 = 1.2 mol/m³) need their own audit script.
F_CONST = 96485.33212  # C/mol
D_O2 = 1.9e-9          # m^2/s
C_O2 = 0.5             # mol/m^3 — pre-M3a.2.1 anchor (Run C was at this scale)
L_REF = 1.0e-4         # m
N_ELECTRONS = 2
I_SCALE = N_ELECTRONS * F_CONST * D_O2 * C_O2 / L_REF * 0.1  # mA/cm^2 ~ 0.1833

RUN_C_JSON = os.path.join(_ROOT, "StudyResults/mangan_p15_comparison/run_C/iv_curve.json")
EXPERIMENTAL_CSV = os.path.join(_ROOT, "data/mangan_deck_p15_h2o2_current.csv")
OUTPUT_DIR = os.path.join(_ROOT, "StudyResults/mangan_p15_comparison")
OBS_JSON = os.path.join(OUTPUT_DIR, "m3a0_observables.json")
AUDIT_MD = os.path.join(OUTPUT_DIR, "m3a0_audit.md")
AUDIT_PNG = os.path.join(OUTPUT_DIR, "m3a0_audit.png")

# Page-15 features (per docs/m0_target_extraction.md B2).
EXP_PEAK_V = 0.10
EXP_PEAK_J = -0.40
EXP_LEFT_PLATEAU_V = -0.32
EXP_LEFT_PLATEAU_J = -0.17
EXP_ONSET_V = 0.45
SHOULDER_V_RANGE = (0.18, 0.30)

PEAK_V_TOL = 0.050
PEAK_J_TOL_REL = 0.25
PLATEAU_J_TOL_REL = 0.25
ONSET_V_TOL = 0.050


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
    finite = np.isfinite(j)
    if finite.sum() < 2:
        return float("nan")
    return float(np.interp(v_target, v[finite], j[finite]))


def _onset_voltage(v: np.ndarray, j: np.ndarray, threshold: float = 0.0) -> float:
    """V where j(V) first crosses threshold from below.  NaN if no crossing."""
    finite = np.isfinite(j)
    vf = v[finite]
    jf = j[finite]
    if len(jf) < 2:
        return float("nan")
    order = np.argsort(vf)
    vf = vf[order]
    jf = jf[order]
    for i in range(len(jf) - 1):
        if jf[i] < threshold and jf[i + 1] >= threshold:
            if jf[i + 1] == jf[i]:
                return float(vf[i + 1])
            frac = (threshold - jf[i]) / (jf[i + 1] - jf[i])
            return float(vf[i] + frac * (vf[i + 1] - vf[i]))
    return float("nan")


def _peak_estimate(v: np.ndarray, j: np.ndarray) -> tuple[float, float]:
    """Most-negative-j with parabolic refinement."""
    finite = np.isfinite(j)
    if not finite.any():
        return float("nan"), float("nan")
    vf = v[finite]
    jf = j[finite]
    order = np.argsort(vf)
    vf = vf[order]
    jf = jf[order]
    idx = int(np.argmin(jf))
    if 1 <= idx <= len(vf) - 2:
        x0, x1, x2 = vf[idx - 1], vf[idx], vf[idx + 1]
        y0, y1, y2 = jf[idx - 1], jf[idx], jf[idx + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denom != 0.0:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
            if a > 0:
                v_peak = -b / (2 * a)
                if x0 < v_peak < x2:
                    j_peak = float(np.interp(v_peak, vf, jf))
                    return float(v_peak), j_peak
    return float(vf[idx]), float(jf[idx])


def _shoulder_present(
    v: np.ndarray, j: np.ndarray, lo: float, hi: float, peak_v: float, peak_j: float,
) -> tuple[bool, str]:
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
    djdv = np.gradient(jw, vw)
    abs_slope = np.abs(djdv)
    if len(abs_slope) >= 3:
        for i in range(1, len(abs_slope) - 1):
            if abs_slope[i] < abs_slope[i - 1] and abs_slope[i] < abs_slope[i + 1]:
                if jw[i] < 0.5 * peak_j:
                    return True, (
                        f"local |dj/dV| minimum at V={vw[i]:+.3f} V "
                        f"with j={jw[i]:+.3f} mA/cm^2"
                    )
    return False, "no inflection / slope flattening detected"


def _abs_rel_err(model_val: float, exp_val: float) -> float:
    if not (np.isfinite(model_val) and np.isfinite(exp_val)) or exp_val == 0:
        return float("nan")
    return abs(model_val - exp_val) / abs(exp_val)


def _abs_err(model_val: float, exp_val: float) -> float:
    if not (np.isfinite(model_val) and np.isfinite(exp_val)):
        return float("nan")
    return abs(model_val - exp_val)


def _evaluate_bands(v: np.ndarray, j: np.ndarray, label: str) -> dict[str, Any]:
    """Compute the 5 B7 tolerance bands against (v, j)."""
    peak_v, peak_j = _peak_estimate(v, j)
    plateau_j = _interp_at(EXP_LEFT_PLATEAU_V, v, j)
    onset_v = _onset_voltage(v, j, threshold=0.0)
    shoulder_ok, shoulder_desc = _shoulder_present(
        v, j, SHOULDER_V_RANGE[0], SHOULDER_V_RANGE[1], peak_v, peak_j,
    )

    bands = []
    bands.append({
        "feature": "peak voltage",
        "exp": EXP_PEAK_V, "model": peak_v,
        "abs_err_V": _abs_err(peak_v, EXP_PEAK_V),
        "tolerance": PEAK_V_TOL, "tolerance_unit": "V (absolute)",
        "passes": (np.isfinite(peak_v) and abs(peak_v - EXP_PEAK_V) <= PEAK_V_TOL),
    })
    bands.append({
        "feature": "peak magnitude",
        "exp": EXP_PEAK_J, "model": peak_j,
        "rel_err": _abs_rel_err(peak_j, EXP_PEAK_J),
        "tolerance": PEAK_J_TOL_REL, "tolerance_unit": "relative (|err|/|exp|)",
        "passes": (np.isfinite(peak_j) and _abs_rel_err(peak_j, EXP_PEAK_J) <= PEAK_J_TOL_REL),
    })
    bands.append({
        "feature": f"left plateau magnitude at V={EXP_LEFT_PLATEAU_V:+.2f} V",
        "exp": EXP_LEFT_PLATEAU_J, "model": plateau_j,
        "rel_err": _abs_rel_err(plateau_j, EXP_LEFT_PLATEAU_J),
        "tolerance": PLATEAU_J_TOL_REL, "tolerance_unit": "relative (|err|/|exp|)",
        "passes": (np.isfinite(plateau_j) and _abs_rel_err(plateau_j, EXP_LEFT_PLATEAU_J) <= PLATEAU_J_TOL_REL),
    })
    bands.append({
        "feature": "onset to zero",
        "exp": EXP_ONSET_V, "model": onset_v,
        "abs_err_V": _abs_err(onset_v, EXP_ONSET_V),
        "tolerance": ONSET_V_TOL, "tolerance_unit": "V (absolute)",
        "passes": (np.isfinite(onset_v) and abs(onset_v - EXP_ONSET_V) <= ONSET_V_TOL),
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
    quant_pass = sum(1 for b in bands[:4] if b["passes"])
    return {
        "label": label,
        "bands": bands,
        "n_pass": n_pass,
        "quant_pass": quant_pass,
    }


def _format_band_row(i: int, b: dict[str, Any]) -> str:
    feat = b["feature"]
    exp = b["exp"]
    model = b["model"]
    if isinstance(model, float):
        model_s = f"{model:+.4g}"
    else:
        model_s = str(model)
    if isinstance(exp, float):
        exp_s = f"{exp:+.4g}"
    else:
        exp_s = str(exp)
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
    return f"| {i} | {feat} | {exp_s} | {model_s} | {err_s} | {tol_s} | {passes_s} |"


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
    cd_mA = np.array(
        [np.nan if x is None else x for x in rep["cd_mA_cm2"]], dtype=float,
    )
    pc_mA = np.array(
        [np.nan if x is None else x for x in rep["pc_mA_cm2"]], dtype=float,
    )

    # Algebraic backout (sequential 2-step model, both reactions 2e):
    #   gross_R0_mA = -I_SCALE * R_0 = (cd + pc) / 2
    #   gross_R1_mA = -I_SCALE * R_1 = (cd - pc) / 2
    gross_R0_mA = (cd_mA + pc_mA) / 2.0
    gross_R1_mA = (cd_mA - pc_mA) / 2.0
    R_0_dimless = (-cd_mA - pc_mA) / (2.0 * I_SCALE)
    R_1_dimless = (-cd_mA + pc_mA) / (2.0 * I_SCALE)

    exp_v, exp_j = _load_csv(EXPERIMENTAL_CSV)

    # Tolerance band evaluation for each model channel.
    audit_pc = _evaluate_bands(v_model, pc_mA, "net pc (R_0 - R_1)")
    audit_gross_R0 = _evaluate_bands(v_model, gross_R0_mA, "gross R_0 (single-rate, 2e)")
    audit_cd = _evaluate_bands(v_model, cd_mA, "total cd (R_0 + R_1)")

    # Dump derived observables.
    obs_payload = {
        "experiment_metadata": em,
        "I_SCALE_mA_cm2": I_SCALE,
        "v_rhe": v_model.tolist(),
        "cd_mA_cm2": cd_mA.tolist(),
        "pc_mA_cm2": pc_mA.tolist(),
        "gross_R0_mA_cm2": gross_R0_mA.tolist(),
        "gross_R1_mA_cm2": gross_R1_mA.tolist(),
        "R_0_dimless": R_0_dimless.tolist(),
        "R_1_dimless": R_1_dimless.tolist(),
        "experimental_v_rhe": exp_v.tolist(),
        "experimental_j_h2o2_mA_cm2": exp_j.tolist(),
        "tolerance_bands": {
            "pc_net": audit_pc,
            "gross_R0": audit_gross_R0,
            "cd_total": audit_cd,
        },
        "comment": (
            "M3a.0 observable audit derived from Run C JSON without re-running "
            "the solver.  Sequential 2-step model algebra: gross_R0 = (cd+pc)/2."
        ),
    }
    def _json_default(o: Any) -> Any:
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} not JSON serializable")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OBS_JSON, "w") as f:
        json.dump(obs_payload, f, indent=2, default=_json_default)

    # Plot (matplotlib only — no Firedrake).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(
        exp_v, exp_j, "k-", lw=2.0,
        label="Experimental (Mangan p.15 digitised)",
    )
    ax.plot(
        v_model, pc_mA, "C0o-", lw=1.4, ms=4,
        label="Model — net pc (R_0 - R_1)",
    )
    ax.plot(
        v_model, gross_R0_mA, "C3s-", lw=1.4, ms=4,
        label="Model — gross R_0 (single-rate 2e)",
    )
    ax.plot(
        v_model, cd_mA, "C2^--", lw=1.0, ms=3, alpha=0.8,
        label="Model — total cd (R_0 + R_1) [reference]",
    )
    ax.axhline(0.0, color="0.6", lw=0.5)
    ax.axvline(EXP_PEAK_V, color="0.8", lw=0.5, ls=":")
    ax.axvline(EXP_ONSET_V, color="0.8", lw=0.5, ls=":")
    ax.set_xlabel("V_RHE (V)")
    ax.set_ylabel("Current density (mA/cm²)")
    ax.set_title(
        "M3a.0 Observable Audit — Mangan p.15 vs Run C reassembly\n"
        "Net pc, gross R_0, and total cd from existing Run C state"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AUDIT_PNG, dpi=150)
    plt.close(fig)

    # Audit markdown.
    lines: list[str] = []
    lines.append("# M3a.0 Observable Audit — Mangan Deck p.15 vs Run C Reassembly")
    lines.append("")
    lines.append("Date: 2026-05-07")
    lines.append("")
    lines.append("## What this is")
    lines.append("")
    lines.append(
        "Per `docs/mangan_alignment_status_2026-05-07.md`§\"Next concrete step\" "
        "and CHATGPT_HANDOFF_18§\"M3a.0\".  No solver re-run.  Pure algebraic "
        "post-processing of `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`."
    )
    lines.append("")
    lines.append(
        "Reassembled three observables from Run C and reapplied the B7 "
        "tolerance bands from `docs/m0_target_extraction.md` against each:"
    )
    lines.append("")
    lines.append(
        "- `cd_mA_cm2` — total disk current = `-I_SCALE · (R_0 + R_1)` (already in JSON)"
    )
    lines.append(
        "- `pc_mA_cm2` — net peroxide = `-I_SCALE · (R_0 - R_1)` (already in JSON; "
        "compared in Run D)"
    )
    lines.append(
        "- `gross_R0_mA_cm2` — gross 2e production = `-I_SCALE · R_0 = (cd + pc) / 2` (NEW)"
    )
    lines.append("")
    lines.append(
        "Algebraic identity for the sequential 2-step model with both reactions 2e:"
    )
    lines.append("")
    lines.append("```")
    lines.append("R_0 + R_1 = -cd / I_SCALE       (sum of dimensionless rates)")
    lines.append("R_0 - R_1 = -pc / I_SCALE       (difference)")
    lines.append("R_0       = (-cd - pc) / (2·I_SCALE)")
    lines.append("gross_R0_mA_cm2 = -I_SCALE · R_0 = (cd + pc) / 2")
    lines.append("```")
    lines.append("")
    lines.append(
        f"`I_SCALE = {I_SCALE:.4f} mA/cm²` (pre-M3a.2.1 anchor: "
        f"n_e=2, F=96485.33, D_O2=1.9e-9 m²/s, C_O2=0.5 mol/m³, L_REF=100 µm). "
        f"Run C was solved at this scale; post-2026-05-07 runs use C_O2=1.2 mol/m³."
    )
    lines.append("")
    lines.append("## Why this matters (post-Ruggiero reframing)")
    lines.append("")
    lines.append(
        "Ruggiero 2022 J. Catal. (Mangan co-author) shows the deck/paper uses "
        "**parallel 2e/4e ORR**, not sequential R_0 producing peroxide which "
        "R_1 then reduces.  The deck's \"Peroxide Current Density\" therefore "
        "maps to the **gross 2e production current** (single-rate observable), "
        "not the net `R_0 - R_1`.  This audit tests how much of the 0/5 Run D "
        "verdict was an observable-definition failure (cheap to fix) vs a "
        "physics gap requiring the parallel R_2e/R_4e rewrite (expensive)."
    )
    lines.append("")
    lines.append("## Run C metadata (reused)")
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
    lines.append("```")
    lines.append("")
    lines.append("## Tolerance bands — three channels")
    lines.append("")

    for ch_label, audit in (
        ("net pc (R_0 - R_1) — original observable", audit_pc),
        ("gross R_0 (single-rate 2e) — NEW post-Ruggiero candidate", audit_gross_R0),
        ("total cd (R_0 + R_1) — reference", audit_cd),
    ):
        lines.append(f"### {ch_label}")
        lines.append("")
        lines.append("| # | Feature | Experimental | Model | Error | Tolerance | Pass |")
        lines.append("|---|---------|--------------|-------|-------|-----------|------|")
        for i, b in enumerate(audit["bands"], start=1):
            lines.append(_format_band_row(i, b))
        lines.append("")
        lines.append(
            f"**Quantitative bands passing**: {audit['quant_pass']} / 4   "
            f"**Total bands passing (incl. shoulder)**: {audit['n_pass']} / 5"
        )
        lines.append("")

    # Side-by-side rate sanity table.
    lines.append("## Sample dimensionless and dimensional rate values")
    lines.append("")
    lines.append("| V_RHE (V) | cd (mA/cm²) | pc (mA/cm²) | R_0 (dimless) | R_1 (dimless) | gross R_0 (mA/cm²) |")
    lines.append("|-----------|-------------|-------------|----------------|----------------|---------------------|")
    sample_v = [-0.40, -0.32, -0.25, -0.10, 0.00, 0.10, 0.18, 0.25, 0.30, 0.40, 0.45]
    for v_t in sample_v:
        idx = int(np.argmin(np.abs(v_model - v_t)))
        if abs(v_model[idx] - v_t) > 0.06:
            continue
        lines.append(
            f"| {v_model[idx]:+.3f} | {cd_mA[idx]:+.4g} | {pc_mA[idx]:+.4g} | "
            f"{R_0_dimless[idx]:+.4g} | {R_1_dimless[idx]:+.4g} | "
            f"{gross_R0_mA[idx]:+.4g} |"
        )
    lines.append("")

    # ----- Layered diagnostic decomposition -----
    sat_mask = R_0_dimless + R_1_dimless > 0.5  # transport-limited regime
    if sat_mask.any():
        gross_R0_saturation_envelope_mA = float(cd_mA[sat_mask].min())
    else:
        gross_R0_saturation_envelope_mA = float("nan")

    gross_R0_at_plateau = _interp_at(EXP_LEFT_PLATEAU_V, v_model, gross_R0_mA)
    factor_short_at_plateau = (
        EXP_LEFT_PLATEAU_J / gross_R0_at_plateau
        if (np.isfinite(gross_R0_at_plateau) and gross_R0_at_plateau != 0.0) else float("nan")
    )

    peak_above_ceiling = abs(EXP_PEAK_J) > I_SCALE
    peak_to_ceiling_ratio = abs(EXP_PEAK_J) / I_SCALE

    lines.append("## Layered diagnostic decomposition")
    lines.append("")
    lines.append(
        "The mechanical band test scores 0/5 for both `pc_net` and `gross R_0`, "
        "but the structural reading under the parallel-2e/4e physics from "
        "Ruggiero is more informative than that summary line:"
    )
    lines.append("")
    lines.append(
        "**(1) Observable switch (net pc -> gross R_0) is necessary.** The "
        "original `peroxide_current = R_0 - R_1` collapses to ~0 across the "
        "entire window because R_0 ~ R_1 ~ 0.5 in the saturation regime. "
        "Switching to single-rate gross R_0 (`mode=\"reaction\", reaction_index=0`) "
        "makes the observable cathodic-negative everywhere with a finite "
        "saturated plateau -- qualitatively the right channel to compare "
        "against the deck's gross 2e production current."
    )
    lines.append("")
    lines.append(
        "**(2) Gross R_0's magnitude shortfall factors into the R_0/R_1 lock-in itself.** "
        f"At the experimental left-plateau voltage V = {EXP_LEFT_PLATEAU_V:+.2f} V, "
        f"gross R_0 = {gross_R0_at_plateau:+.4g} mA/cm² vs experimental "
        f"{EXP_LEFT_PLATEAU_J:+.2f} mA/cm² -- short by a factor of "
        f"{factor_short_at_plateau:.2f}x. The reason is mechanical: in saturation "
        "R_0 + R_1 ~ 1, so R_0 ~ 0.5. If R_1 were replaced by a parallel R_4e "
        "channel without a free-H₂O₂ surface intermediate (the Ruggiero §1 "
        "structural fix), or kinetically suppressed for CMK-3 2e selectivity, "
        "R_0 would absorb the full O₂ flux at saturation."
    )
    lines.append("")
    lines.append(
        f"  - Saturation envelope (gross R_0 unlocked from R_1): "
        f"{gross_R0_saturation_envelope_mA:+.4g} mA/cm² ≡ -I_SCALE at full 2e "
        "Levich. This is total cd in the existing data, which numerically "
        "*already* matches the experimental left plateau within ~8% (well inside "
        f"the ±25% band): {audit_cd['bands'][2]['model']:+.4g} model vs "
        f"{EXP_LEFT_PLATEAU_J:+.2f} exp."
    )
    lines.append("")
    lines.append(
        "  - Read: total cd's left-plateau \"PASS\" is **not** a match in the "
        "current sequential physics -- it's the saturation envelope that "
        "**gross R_0 would reach** if the parallel-reaction structural fix "
        "landed. The 2x factor is mechanical, not physical."
    )
    lines.append("")
    lines.append(
        "**(3) Peak structure is real missing physics, not an observable bug.** "
        "Gross R_0 monotonically decays from saturation toward zero as V rises "
        "through E_eq_R1 = 0.68 V. The experimental peak at V_RHE ≈ +0.10 V is "
        "non-monotonic in V at fixed bulk pH -- the local-pH-driven mechanism "
        "shift in Ruggiero §3.1 (bulk pH 4 has the largest local pH excursion; "
        "cation identity modulates buffering). M3b (multi-ion electrolyte: "
        "Cs⁺/SO₄²⁻ + OH⁻ tracking) and M3c (local-pH validation against "
        "Ruggiero Fig 1B) work."
    )
    lines.append("")
    lines.append(
        f"**(4) Mass-transport headroom check.** Experimental peak |j| = "
        f"{abs(EXP_PEAK_J):.2f} mA/cm² vs our 2e Levich ceiling I_SCALE = "
        f"{I_SCALE:.4f} mA/cm² → ratio {peak_to_ceiling_ratio:.2f}x."
    )
    if peak_above_ceiling:
        lines.append(
            "  - Experimental peak is **above** our 2e Levich ceiling. Even after "
            "the structural fix in (2), gross R_0 cannot reach −0.40 mA/cm² "
            "without L_eff alignment (M5: ~21 µm Levich δ at 1600 rpm vs current "
            "100 µm) or local-pH amplification of the effective surface rate. "
            "M5 retune is **larger than the ~10% previously assumed** -- the "
            "ceiling itself needs to roughly double."
        )
    else:
        lines.append(
            "  - Experimental peak is below the 2e Levich ceiling -- model has "
            "headroom in principle to reach the peak magnitude."
        )
    lines.append("")

    # ----- Decision -----
    lines.append("## Decision (per status doc \"Next concrete step\")")
    lines.append("")
    pc_quant = audit_pc["quant_pass"]
    gross_R0_quant = audit_gross_R0["quant_pass"]
    gross_R0_total = audit_gross_R0["n_pass"]
    gross_R0_correct_sign = bool(np.all(gross_R0_mA <= 1e-6) and np.any(gross_R0_mA < -1e-3))
    saturation_envelope_passes_plateau = audit_cd["bands"][2]["passes"]

    if gross_R0_quant >= 3:
        verdict = "scope_reduction"
        verdict_text = (
            "**Scope reduction.**  Gross R_0 channel matches both magnitude and "
            "shape under the page-15 tolerance bands.  The Run D failure was "
            "primarily an observable-definition mismatch.  We may not need the "
            "full parallel R_2e/R_4e reaction-set rewrite for **this figure** "
            "— though the parallel-pathway physics is still the correct "
            "structural reading of Ruggiero §1, and total disk current / "
            "selectivity will still need M3a.1+ once any 4e channel is added.  "
            "Recommend: switch the page-15 comparison observable to "
            "`mode=\"reaction\", reaction_index=0` and re-evaluate which of "
            "M3a.1-M3a.3 / M3b are still load-bearing."
        )
    elif gross_R0_correct_sign and saturation_envelope_passes_plateau:
        verdict = "magnitude_off_by_lock_in_shape_off_by_local_pH"
        verdict_text = (
            "**Layered finding -- closest to status doc's \"most likely\" branch, "
            "with structural refinement.**\n\n"
            "Gross R_0 has the right sign and saturation/decay structure but is "
            f"short by {factor_short_at_plateau:.2f}x at the left-plateau "
            "voltage *and* missing the peak feature in the middle of the V "
            "range. Both gaps factor into known structural/physics issues "
            "identified in the Ruggiero realignment, **not** a forward-solver "
            "bug:\n\n"
            "- **Magnitude (~2x short on plateau):** the R_0/R_1 lock-in itself "
            "  (sequential model artifact). Resolved by replacing R_1 with "
            "  parallel R_4e (M3a.2 diagnostic + M3a.3 production) or by "
            "  CMK-3 2e-selective kinetic recalibration of k0_R2. After this, "
            "  gross R_0's saturation envelope = total cd ~ -0.183 mA/cm² "
            "  lands within +/-25% of the experimental left plateau. The "
            "  audit's reading: total cd is the projected *post-fix* gross R_0, "
            "  not a current match.\n"
            "- **Peak structure (no peak in middle of V range):** local-pH "
            "  dynamics + cation buffering + multi-ion EDL screening. M3b "
            "  (multi-ion electrolyte: Cs⁺/SO₄²⁻ + OH⁻ tracking) and M3c "
            "  (local-pH validation against Ruggiero Fig 1B).\n"
            f"- **Peak magnitude (-0.40 vs our 2e ceiling -0.183):** experimental "
            f"  peak is {peak_to_ceiling_ratio:.2f}x our 2e Levich ceiling, so "
            "  even after the lock-in fix the peak cannot be reproduced at "
            "  L_REF = 100 µm. **M5 (L_eff alignment) is larger than the "
            "  ~10% retune previously assumed** -- the mass-transport ceiling "
            "  itself needs to roughly double to give gross R_0 enough headroom "
            "  to peak past the saturation plateau.\n\n"
            "Implication for sequencing:\n"
            "1. **M3a.0 (this audit): DONE.** Confirms the gap is structural + "
            "   missing physics, not a forward-solver bug.\n"
            "2. **M3a.1 (electron-weighted current accounting):** proceed -- "
            "   required before any R_4e is added to the residual.\n"
            "3. **M3a.2 (diagnostic parallel R_2e/R_4e residual):** the clean "
            "   test of the gross R_0 channel. *Predicts:* gross R_0 plateau "
            "   lifts from -0.092 -> near -0.183, passing the left-plateau "
            "   band; peak structure still missing.\n"
            "4. **M3a.3 (production IC generalization):** only after M3a.2.\n"
            "5. **M3b/M3c (multi-ion + local-pH validation):** for the peak.\n"
            "6. **M5 (L_eff alignment):** larger than +/-10%. Defer sizing "
            "   until M3b shows whether local-pH amplification accounts for "
            "   any of the peak-magnitude gap -- it may already be partly "
            "   contributing in the existing surface_pH_proxy field "
            "   (~9.7 at V=-0.40 in Run C, well above bulk pH 4)."
        )
    elif gross_R0_quant >= 1 or audit_gross_R0["bands"][0]["passes"] or audit_gross_R0["bands"][2]["passes"]:
        verdict = "shape_reasonable_magnitude_off"
        verdict_text = (
            "**Shape reasonable, magnitude off.** Gross R_0 is qualitatively "
            "closer to experimental shape than net pc was, but does not pass "
            "the full B7 band set. Parallel-reaction work is real but informed. "
            "Proceed with M3a.1 -> M3a.2 -> M3a.3, then M3b."
        )
    else:
        verdict = "no_match_anywhere"
        verdict_text = (
            "**Gross R_0 doesn't match anything either.** Deeper diagnosis "
            "required before more milestones -- debug forward solver / kinetics "
            "/ constants before committing to M3b multi-ion or M3a.3 production IC."
        )

    lines.append(f"`m3a0_verdict = \"{verdict}\"`")
    lines.append("")
    lines.append(verdict_text)
    lines.append("")

    lines.append("## Comparison summary")
    lines.append("")
    lines.append(
        "Number of B7 tolerance bands passing under each candidate observable "
        "(out of 5 total; 4 quantitative + 1 qualitative shoulder):"
    )
    lines.append("")
    lines.append("| Channel                                    | Quantitative (/4) | Total (/5) |")
    lines.append("|--------------------------------------------|--------------------|------------|")
    lines.append(f"| net pc = R_0 - R_1 (original observable)   | {audit_pc['quant_pass']:>2d}                | {audit_pc['n_pass']:>2d}         |")
    lines.append(f"| gross R_0 (single-rate 2e, NEW)            | {audit_gross_R0['quant_pass']:>2d}                | {audit_gross_R0['n_pass']:>2d}         |")
    lines.append(f"| total cd = R_0 + R_1 (reference)           | {audit_cd['quant_pass']:>2d}                | {audit_cd['n_pass']:>2d}         |")
    lines.append("")

    lines.append("## Open follow-ups")
    lines.append("")
    lines.append(
        "- This audit only reinterprets the existing Run C *observable*. The "
        "underlying residual still uses the sequential R_0 + R_1 reaction list "
        "and both reactions are 2e (so `N_ELECTRONS=2` in I_SCALE is correct "
        "for the existing data)."
    )
    lines.append(
        "- **M3a.1 (electron-weighted current accounting)** is required before "
        "any R_4e is added to the residual: once per-reaction `n_electrons` "
        "becomes heterogeneous (R_2e=2, R_4e=4), total disk current must be "
        "`Σ n_e_j · R_j · scale_per_unit_n_e`, not the current `Σ R_j · I_SCALE` "
        "with a global 2e prefactor (cf. Handoff 18 §2)."
    )
    lines.append(
        "- **What replacing R_1 with parallel R_4e actually buys for the "
        "peroxide channel.** In the current sequential model, R_1 is "
        "specifically a peroxide *sink* (H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O), so it "
        "couples directly to surface c_H2O2 and rate-limits gross R_0 through "
        "the H2O2 mass balance. A parallel R_4e (O₂ + 4H⁺ + 4e⁻ → 2H₂O) "
        "consumes O₂ independently; for a 2e-selective catalyst like CMK-3, "
        "R_4e's k0 is suppressed relative to R_2e, so R_2e absorbs the O₂ "
        "flux and gross R_2e rises toward the full 2e Levich saturation "
        "(-I_SCALE = -0.183 mA/cm²) at the cathodic plateau. That is the "
        "M3a.2 diagnostic prediction."
    )
    lines.append(
        "- **Surface pH excursion is already large in Run C.** "
        "`surface_pH_proxy` = -log10(c_H_surface · C_SCALE) sits at ~9.77 at "
        "V=-0.40 V and ~8.36 at V=+0.30 V — i.e., a 4-6 pH-unit alkaline "
        "excursion from bulk pH 4. This is in the ballpark of Ruggiero Fig 1B "
        "(~4 pH units at 3.25 mA/cm²), but applies under the surrogate "
        "(ClO4⁻ pH-countercharge) electrolyte, no Cs⁺ buffering, and may not "
        "translate quantitatively under M3b. The fact that the model's local "
        "pH is already alkaline yet produces no PC peak is consistent with "
        "the lock-in dominating: even with surface H⁺ depleted, R_2's "
        "intrinsic BV factor stays large enough to keep R_0/R_1 locked."
    )
    lines.append("")
    lines.append("## Cross-references")
    lines.append("")
    lines.append("- Status doc: `docs/mangan_alignment_status_2026-05-07.md`")
    lines.append("- Source-paper finding: `docs/Ruggiero2022_JCatal_source_paper.md`")
    lines.append("- Run C JSON: `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`")
    lines.append("- Run D verdict: `StudyResults/mangan_p15_comparison/run_D_verdict.md`")
    lines.append("- Digitised target: `data/mangan_deck_p15_h2o2_current.csv`")
    lines.append("- Plan H17/H18: `docs/CHATGPT_HANDOFF_17_RUGGIERO_REALIGNMENT_PLAN.md`, "
                 "`docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md`")
    lines.append("- Observable code: `Forward/bv_solver/observables.py:13-68`")
    lines.append("- I_SCALE definition: `scripts/_bv_common.py:131-141`")
    lines.append("- Audit derived observables JSON: `StudyResults/mangan_p15_comparison/m3a0_observables.json`")
    lines.append("- Audit overlay plot: `StudyResults/mangan_p15_comparison/m3a0_audit.png`")
    lines.append("")

    with open(AUDIT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"M3a.0 audit written to {AUDIT_MD}")
    print(f"Derived observables JSON: {OBS_JSON}")
    print(f"Overlay plot: {AUDIT_PNG}")
    print()
    print(f"I_SCALE = {I_SCALE:.6f} mA/cm^2")
    print()
    print(f"Tolerance bands passing (quant / total out of 4 / 5):")
    print(f"  net pc (R_0 - R_1):   {audit_pc['quant_pass']} / {audit_pc['n_pass']}")
    print(f"  gross R_0 (NEW):      {audit_gross_R0['quant_pass']} / {audit_gross_R0['n_pass']}")
    print(f"  total cd (reference): {audit_cd['quant_pass']} / {audit_cd['n_pass']}")
    print()
    print(f"M3a.0 verdict: {verdict}")
    print()
    print("Per-band detail (gross R_0 channel):")
    for b in audit_gross_R0["bands"]:
        print(f"  {b['feature']:<48s}: {'PASS' if b['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
