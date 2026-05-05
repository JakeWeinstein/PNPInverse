"""Peroxide-window proton-electrochemical-potential (muh) sweep.

Tests whether ``formulation="logc_muh"`` (Phase 2 hybrid: H+ stored as
``mu_H = u_H + em*z_H*phi``) extends convergence past the
``V_RHE >= +0.68 V`` peroxide-window wall, and/or reduces SNES iteration
counts at peroxide voltages, vs the production ``"logc"`` baseline.

Per the plan
``~/.claude/plans/look-at-docs-electrochemical-potential-s-misty-trinket.md``,
the muh transform targets the *Newton-stiffness* failure mode (``u_H`` and
``phi`` separately blow up in the Debye layer; ``mu_H`` is nearly flat).
It does **not** fix the *physical-validity* failure mode (analytic
Boltzmann ``c_ClO4 = c_bulk * exp(+phi)`` exceeds the Bikerman steric
cap ~100 at high anodic phi).  The summary.md emitted by this study
**reports those two axes separately** so a "Newton converged" row is not
confused with a "physically valid" row.

Sweep
-----
Outer loop:  C_S in [None, 0.05, 0.10, 0.20, 0.40, 1.00] F/m².
Inner loop:  V_RHE in [0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00] V.

At Ny=200 with the production stack, ``initializer="debye_boltzmann"``,
and ``formulation="logc_muh"``.

Validation gates
----------------
* **Gate 1 (HARD, code regression).** ``C_S=None`` muh row must reproduce
  the existing ``logc`` no-Stern baseline at every voltage where the
  baseline converges, within ``rel_tol=1e-6``.  The muh transform is
  algebraic; converged CD/PC must be identical (mu_H reconstructs to the
  same physical c_H).  FAIL on this gate aborts the physics analysis:
  the muh wiring is broken.
* **Gate 2 (NUMERICAL, peroxide-window crossing).** For at least one
  finite ``C_S`` (or no-Stern, if muh enables it):
    - ``V_RHE in {0.68, 0.70, 0.75}`` all converge.
* **Gate 3 (PHYSICAL, steric validity).** Among the rows that pass Gate 2,
  any whose ``c_clo4_surface`` stays under the Bikerman cap (~100 nondim)
  at every peroxide-window voltage is flagged "physically plausible";
  otherwise the row is "Newton-converged-but-non-physical" and should
  not be used as inverse-problem ground truth.
* **Gate 4 (NEW vs Stern baseline).** Compare the muh sweep against the
  ``StudyResults/peroxide_window_stern_test/`` companion baseline (logc
  at the same V × C_S grid).  Report SNES-iteration deltas and
  convergence-window deltas per row.  WARN-only — informational.

Outputs
-------
``StudyResults/peroxide_window_muh_test/iv_curve.json``
``StudyResults/peroxide_window_muh_test/diagnostics.json``
``StudyResults/peroxide_window_muh_test/results.csv``
``StudyResults/peroxide_window_muh_test/summary.md``
``StudyResults/peroxide_window_muh_test/comparison.png``  (optional)

Run from PNPInverse/ with ../venv-firedrake/bin/activate active::

    python scripts/studies/peroxide_window_muh_test.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Defaults (overridable via argparse)
# ---------------------------------------------------------------------------

V_TEST_DEFAULT = (0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00)
CS_DEFAULT = (None, 0.05, 0.10, 0.20, 0.40, 1.00)
MESH_NY_DEFAULT = 200
EXPONENT_CLIP_DEFAULT = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
STERIC_CAP = 100.0
INITIALIZER_DEFAULT = "debye_boltzmann"
FORMULATION_FIXED = "logc_muh"
OUT_SUBDIR = "peroxide_window_muh_test"

GATE1_REL_TOL = 1e-6
GATE2_VOLTAGES = (0.68, 0.70, 0.75)

# Path to the Stern test summary (logc baseline) for cross-comparison.
STERN_BASELINE_DIR = "peroxide_window_stern_test"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mesh-ny", type=int, default=MESH_NY_DEFAULT,
                   help=f"Graded mesh Ny (default {MESH_NY_DEFAULT}).")
    p.add_argument("--clip", type=float, default=EXPONENT_CLIP_DEFAULT,
                   help=f"BV exponent_clip (default {EXPONENT_CLIP_DEFAULT}).")
    p.add_argument("--initializer", type=str, default=INITIALIZER_DEFAULT,
                   help=f"Cold-start IC (default '{INITIALIZER_DEFAULT}').")
    p.add_argument(
        "--cs-list", type=str,
        default=",".join("None" if c is None else f"{c:g}" for c in CS_DEFAULT),
        help=("Comma-separated C_S values in F/m²; 'None' for no-Stern. "
              f"Default '{','.join('None' if c is None else f'{c:g}' for c in CS_DEFAULT)}'."),
    )
    p.add_argument(
        "--v-list", type=str,
        default=",".join(f"{v:g}" for v in V_TEST_DEFAULT),
        help=("Comma-separated V_RHE values. "
              f"Default '{','.join(f'{v:g}' for v in V_TEST_DEFAULT)}'."),
    )
    return p.parse_args()


def _parse_cs_list(s: str) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "none":
            out.append(None)
        else:
            out.append(float(tok))
    return out


def _parse_v_list(s: str) -> list[float]:
    return [float(tok.strip()) for tok in s.split(",") if tok.strip()]


def _cs_label(cs: Optional[float]) -> str:
    return "None" if cs is None else f"{cs:g}"


# ---------------------------------------------------------------------------
# Per-C_S sweep
# ---------------------------------------------------------------------------

def _run_one_cs(
    cs: Optional[float],
    *,
    v_rhe_grid: list[float],
    mesh_ny: int,
    exponent_clip: float,
    initializer: str,
) -> dict[str, Any]:
    """Cold + warm-walk solve over V_RHE grid for a single C_S branch."""
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(mesh_ny), beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION_FIXED,
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        stern_capacitance_f_m2=cs,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer=initializer,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(exponent_clip)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(v_rhe_grid)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = np.array(v_rhe_grid) / V_T
    t0 = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=N_SUBSTEPS_WARM,
            bisect_depth_warm=BISECT_DEPTH_WARM,
            per_point_callback=_grab,
        )
    elapsed = time.time() - t0

    converged_flags = [bool(result.points[i].converged) for i in range(NV)]
    methods = [result.points[i].method for i in range(NV)]
    z_achieved = [float(result.points[i].achieved_z_factor) for i in range(NV)]
    diagnostics_per_v = [result.points[i].diagnostics for i in range(NV)]
    failure_reasons = [
        getattr(result.points[i], "failure_reason", None) for i in range(NV)
    ]

    return {
        "cs_label": _cs_label(cs),
        "cs_f_m2": cs,
        "wall_seconds": float(elapsed),
        "v_rhe": list(v_rhe_grid),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc],
        "converged": converged_flags,
        "method": methods,
        "z_achieved": z_achieved,
        "diagnostics": diagnostics_per_v,
        "failure_reason": failure_reasons,
        "n_converged": int(sum(converged_flags)),
        "n_total": int(NV),
    }


# ---------------------------------------------------------------------------
# Per-row record assembly + CSV
# ---------------------------------------------------------------------------

def _per_row_records(
    reports: list[dict],
    *,
    v_t: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in reports:
        diags = r["diagnostics"]
        for i, v in enumerate(r["v_rhe"]):
            d = diags[i] or {}
            phi_app_hat = r["phi_applied_hat"][i]
            phi_surf_hat = d.get("phi_surface_mean")
            stern_drop = (phi_app_hat - phi_surf_hat) if phi_surf_hat is not None else None
            row: dict[str, Any] = {
                "cs_label":              r["cs_label"],
                "cs_f_m2":               r["cs_f_m2"],
                "v_rhe":                 v,
                "converged":             r["converged"][i],
                "method":                r["method"][i],
                "cd_mA_cm2":             r["cd_mA_cm2"][i],
                "pc_mA_cm2":             r["pc_mA_cm2"][i],
                "phi_applied_nondim":    phi_app_hat,
                "phi_surface_nondim":    phi_surf_hat,
                "phi_stern_drop_nondim": stern_drop,
                "phi_stern_drop_v":      stern_drop * v_t if stern_drop is not None else None,
                "c_o2_surface":          d.get("c0_surface_mean"),
                "c_h2o2_surface":        d.get("c1_surface_mean"),
                "c_h_surface":           d.get("c2_surface_mean"),
                "c_clo4_surface":        d.get("c_counterion0_surface_mean"),
                "surface_within_steric": d.get("surface_counterion_within_steric"),
                # MUH-specific: raw mu_H surface mean (= u_H_surf + em*z*phi_surf).
                "mu_h_surface":          d.get("mu2_surface_mean"),
                "snes_iters":            d.get("snes_iters"),
                "snes_reason":           d.get("snes_reason"),
                "picard_iters":          d.get("picard_iters"),
                "initializer_fallback":  d.get("initializer_fallback"),
                "z_achieved":            r["z_achieved"][i],
                "failure_reason":        r["failure_reason"][i],
            }
            rows.append(row)
    return rows


def _save_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Cross-formulation baseline loader (for Gate 1 + Gate 4)
# ---------------------------------------------------------------------------

def _load_logc_baseline(stern_iv_path: str) -> dict:
    """Load the logc Stern test iv_curve.json for cross-formulation Gate 1+4.

    Returns ``{"reports": <list of per-cs dicts>}`` or ``{}`` if the file
    is missing.  The structure mirrors what ``_run_one_cs`` produces for
    the logc study.
    """
    if not os.path.isfile(stern_iv_path):
        return {}
    try:
        with open(stern_iv_path) as f:
            data = json.load(f)
        return data
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _evaluate_gate1(
    reports: list[dict], baseline: dict
) -> dict[str, Any]:
    """Code-regression: muh CS=None matches logc CS=None at converged voltages."""
    findings: list[str] = []
    by_label_muh = {r["cs_label"]: r for r in reports}
    none_muh = by_label_muh.get("None")

    if none_muh is None:
        findings.append("FAIL: no C_S=None muh row in sweep.")
        return {"verdict": "FAIL", "findings": findings}

    if not baseline or "reports" not in baseline:
        findings.append(
            f"INFO: logc baseline not found at "
            f"StudyResults/{STERN_BASELINE_DIR}/iv_curve.json -- "
            f"Gate 1 cross-formulation check skipped.  Run "
            f"`python scripts/studies/peroxide_window_stern_test.py` first."
        )
        return {"verdict": "INFO", "findings": findings}

    by_label_logc = {r["cs_label"]: r for r in baseline["reports"]}
    none_logc = by_label_logc.get("None")
    if none_logc is None:
        findings.append("INFO: no C_S=None row in logc baseline; skipping.")
        return {"verdict": "INFO", "findings": findings}

    drift_violations: list[str] = []
    for i, v in enumerate(none_muh["v_rhe"]):
        if not none_muh["converged"][i]:
            continue
        # Find matching V index in logc baseline.
        logc_i = next(
            (j for j, vv in enumerate(none_logc["v_rhe"])
             if abs(vv - v) < 1e-9),
            None,
        )
        if logc_i is None:
            continue
        if not none_logc["converged"][logc_i]:
            continue
        cd_m = none_muh["cd_mA_cm2"][i]
        pc_m = none_muh["pc_mA_cm2"][i]
        cd_l = none_logc["cd_mA_cm2"][logc_i]
        pc_l = none_logc["pc_mA_cm2"][logc_i]
        if cd_m is None or pc_m is None or cd_l is None or pc_l is None:
            continue
        cd_drift = abs(cd_m - cd_l) / max(abs(cd_l), 1e-30)
        pc_drift = abs(pc_m - pc_l) / max(abs(pc_l), 1e-30)
        if cd_drift > GATE1_REL_TOL or pc_drift > GATE1_REL_TOL:
            drift_violations.append(
                f"V={v:+.3f}: CD drift={cd_drift:.2e}, PC drift={pc_drift:.2e}"
                f" (tol={GATE1_REL_TOL:.0e})"
            )

    if not drift_violations:
        findings.append(
            f"PASS: muh C_S=None reproduces logc C_S=None CD/PC at all "
            f"converged voltages within rel_tol={GATE1_REL_TOL:.0e}.  The "
            f"muh transform is algebraic; this is the load-bearing check "
            f"that mu_H reconstruction recovers the same physical c_H."
        )
        verdict = "PASS"
    else:
        findings.append(
            "FAIL: muh C_S=None drifted from logc C_S=None; the muh "
            "transform has a defect or there is a residual sign error."
        )
        findings.extend(f"  - {x}" for x in drift_violations)
        verdict = "FAIL"

    return {"verdict": verdict, "findings": findings}


def _evaluate_gate2(reports: list[dict]) -> dict[str, Any]:
    """Numerical: at least one C_S branch converges at all peroxide voltages."""
    successes: list[dict] = []
    near_misses: list[dict] = []
    for r in reports:
        idx_map = {round(v, 4): i for i, v in enumerate(r["v_rhe"])}
        converged_at = {}
        for v_target in GATE2_VOLTAGES:
            i = idx_map.get(round(v_target, 4))
            converged_at[v_target] = (
                bool(r["converged"][i]) if i is not None else False
            )
        all_conv = all(converged_at.get(v, False) for v in GATE2_VOLTAGES)
        record = {
            "cs_label": r["cs_label"],
            "cs_f_m2": r["cs_f_m2"],
            "converged_at": converged_at,
            "all_converged": all_conv,
        }
        if all_conv:
            successes.append(record)
        else:
            near_misses.append(record)

    findings: list[str] = []
    if not successes:
        findings.append(
            f"FAIL: no C_S branch converged at all of "
            f"{list(GATE2_VOLTAGES)} V."
        )
        if near_misses:
            findings.append("Near-misses:")
            for nm in near_misses:
                conv_str = ", ".join(
                    f"V={v:+.2f}: {nm['converged_at'].get(v, 'n/a')}"
                    for v in GATE2_VOLTAGES
                )
                findings.append(f"  - C_S={nm['cs_label']}: {conv_str}")
        verdict = "FAIL"
    else:
        none_in = [s for s in successes if s["cs_f_m2"] is None]
        if none_in:
            findings.append(
                f"PASS+NOTABLE: muh at C_S=None converged at all of "
                f"{list(GATE2_VOLTAGES)} V.  This is a stronger result "
                f"than the logc Stern study, which required finite C_S "
                f"to cross +0.68 V."
            )
        else:
            findings.append(
                f"PASS: muh at finite C_S converged at all of "
                f"{list(GATE2_VOLTAGES)} V."
            )
        for s in successes:
            findings.append(f"  - C_S={s['cs_label']} F/m²: all converged.")
        verdict = "PASS"
    return {
        "verdict": verdict, "findings": findings,
        "successes": successes, "near_misses": near_misses,
    }


def _evaluate_gate3(reports: list[dict]) -> dict[str, Any]:
    """Physical: among Gate 2 successes, which keep c_ClO4_surface within steric scale?"""
    findings: list[str] = []
    physical: list[dict] = []
    suspect: list[dict] = []
    for r in reports:
        idx_map = {round(v, 4): i for i, v in enumerate(r["v_rhe"])}
        steric_at = {}
        all_conv = True
        for v_target in GATE2_VOLTAGES:
            i = idx_map.get(round(v_target, 4))
            if i is None:
                all_conv = False
                continue
            if not r["converged"][i]:
                all_conv = False
                continue
            d = r["diagnostics"][i] or {}
            steric_at[v_target] = d.get("surface_counterion_within_steric")
        if not all_conv:
            continue
        all_within = all(steric_at.get(v) is True for v in GATE2_VOLTAGES)
        record = {
            "cs_label": r["cs_label"], "cs_f_m2": r["cs_f_m2"],
            "steric_at": steric_at, "all_within_steric": all_within,
        }
        if all_within:
            physical.append(record)
        else:
            suspect.append(record)

    if not physical and not suspect:
        findings.append("INFO: no Gate-2-success rows to evaluate.")
        return {"verdict": "INFO", "findings": findings,
                "physical": physical, "suspect": suspect}

    if physical:
        findings.append(
            f"PASS: {len(physical)} row(s) keep surface c_ClO4 within "
            f"the Bikerman steric scale (~{STERIC_CAP:g} nondim) at all "
            f"of {list(GATE2_VOLTAGES)} V."
        )
        for s in physical:
            findings.append(f"  - C_S={s['cs_label']} F/m²: physically plausible.")
    if suspect:
        findings.append(
            f"WARN: {len(suspect)} row(s) Newton-converged but exceeded "
            f"steric cap.  These states are non-physical; use only as "
            f"numerical diagnostics, NOT inverse-problem ground truth."
        )
        for s in suspect:
            steric_str = ", ".join(
                f"V={v:+.2f}: {s['steric_at'].get(v)}"
                for v in GATE2_VOLTAGES
            )
            findings.append(f"  - C_S={s['cs_label']}: {steric_str}")

    verdict = "PASS" if physical else "WARN"
    return {"verdict": verdict, "findings": findings,
            "physical": physical, "suspect": suspect}


def _evaluate_gate4(reports: list[dict], baseline: dict) -> dict[str, Any]:
    """Cross-formulation comparison: muh vs logc per (C_S, V).

    WARN-only.  Reports SNES iter deltas + convergence-window deltas.
    """
    findings: list[str] = []
    if not baseline or "reports" not in baseline:
        findings.append(
            f"INFO: logc baseline missing -- cross-formulation comparison "
            f"skipped.  Run the Stern test first to populate "
            f"StudyResults/{STERN_BASELINE_DIR}/."
        )
        return {"verdict": "INFO", "findings": findings}

    by_logc = {r["cs_label"]: r for r in baseline["reports"]}
    by_muh = {r["cs_label"]: r for r in reports}
    common = sorted(set(by_logc) & set(by_muh))
    if not common:
        findings.append("INFO: no overlapping C_S rows.")
        return {"verdict": "INFO", "findings": findings}

    iter_deltas: list[float] = []
    conv_diffs: list[str] = []
    for label in common:
        r_l = by_logc[label]
        r_m = by_muh[label]
        for i, v in enumerate(r_m["v_rhe"]):
            j = next(
                (k for k, vv in enumerate(r_l["v_rhe"]) if abs(vv - v) < 1e-9),
                None,
            )
            if j is None:
                continue
            ok_l = r_l["converged"][j]
            ok_m = r_m["converged"][i]
            if ok_l != ok_m:
                conv_diffs.append(
                    f"C_S={label}, V={v:+.2f}: logc={ok_l}, muh={ok_m}"
                )
                continue
            if not (ok_l and ok_m):
                continue
            d_l = r_l["diagnostics"][j] if "diagnostics" in r_l else None
            d_m = r_m["diagnostics"][i]
            if d_l and d_m:
                it_l = d_l.get("snes_iters")
                it_m = d_m.get("snes_iters")
                if it_l is not None and it_m is not None:
                    iter_deltas.append(it_m - it_l)

    if iter_deltas:
        avg = sum(iter_deltas) / len(iter_deltas)
        findings.append(
            f"INFO: SNES iter delta (muh - logc) over {len(iter_deltas)} "
            f"matched converged points: mean={avg:+.2f}, "
            f"min={min(iter_deltas):+d}, max={max(iter_deltas):+d}.  "
            f"Negative = muh is faster."
        )
    if conv_diffs:
        findings.append(
            f"INFO: {len(conv_diffs)} (C_S, V) point(s) with differing "
            f"convergence outcomes between formulations:"
        )
        findings.extend(f"  - {x}" for x in conv_diffs)

    if not iter_deltas and not conv_diffs:
        findings.append("INFO: no overlapping converged points to compare.")

    return {"verdict": "INFO", "findings": findings}


def _decision_block(
    gate1: dict[str, Any],
    gate2: dict[str, Any],
    gate3: dict[str, Any],
) -> list[str]:
    out: list[str] = []
    if gate1["verdict"] == "FAIL":
        out.append(
            "**Decision: HALT.** Gate 1 (cross-formulation regression) failed. "
            "The muh transform produced different CD/PC than logc at "
            "no-Stern.  This indicates a sign error or substitution defect "
            "in `forms_logc_muh.py`; do not trust any peroxide-window output "
            "from this run.  Run the test suite "
            "(`pytest tests/test_logc_muh_formulation.py -m slow`) and "
            "investigate the residual-equivalence test."
        )
        return out

    if gate2["verdict"] == "FAIL":
        out.append(
            "**Decision: muh + Stern still does not cross +0.68 V.**  Phase 5 "
            "shows the muh transform alone is insufficient for the peroxide "
            "window.  Next steps per the plan: Phase 7 (charged-species mu "
            "for 4sp+Bikerman), or revisit IC strategy "
            "(`docs/4sp_drop_boltzmann_investigation.md` Option 2a/b/c)."
        )
        return out

    if gate3["verdict"] == "PASS":
        physical = gate3.get("physical", [])
        labels = ", ".join(f"C_S={s['cs_label']} F/m²" for s in physical)
        out.append(
            f"**Decision: muh delivers a physically plausible peroxide-window "
            f"branch.** Convergence + steric validity at peroxide voltages: "
            f"{labels}.  This is the strongest possible result -- proceed "
            f"to inverse-problem extension on the muh stack."
        )
    else:
        out.append(
            "**Decision: muh extends Newton convergence but does not solve "
            "physical validity.** The analytic Boltzmann ClO4- residual "
            "still produces c_ClO4_surface > steric cap at peroxide "
            "voltages.  This was the predicted outcome (per "
            "`~/.claude/plans/look-at-docs-electrochemical-potential-s-misty-trinket.md` "
            "Phase 7 caveat).  Proceed to Phase 7 (charged-species mu on "
            "4sp+Bikerman) for a physically valid solver at peroxide V."
        )
    return out


# ---------------------------------------------------------------------------
# Summary.md emitter
# ---------------------------------------------------------------------------

def _make_summary(
    *,
    reports: list[dict],
    rows: list[dict],
    v_test: list[float],
    cs_list: list[Optional[float]],
    mesh_ny: int,
    exponent_clip: float,
    initializer: str,
    gate1: dict[str, Any],
    gate2: dict[str, Any],
    gate3: dict[str, Any],
    gate4: dict[str, Any],
    output_dir: str,
) -> str:
    lines: list[str] = []
    lines.append("# Peroxide-window proton-electrochemical-potential (muh) sweep")
    lines.append("")
    lines.append("Study script: `scripts/studies/peroxide_window_muh_test.py`.")
    lines.append("Plan: `~/.claude/plans/look-at-docs-electrochemical-potential-s-misty-trinket.md`.")
    lines.append("Forms: `Forward/bv_solver/forms_logc_muh.py`.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Formulation: `{FORMULATION_FIXED}` (Phase 2 hybrid: H+ as mu_H)")
    lines.append(f"- V_RHE grid: {v_test}")
    lines.append(
        f"- C_S grid (F/m²): "
        f"{[('None' if c is None else c) for c in cs_list]}")
    lines.append(f"- Mesh Ny: {mesh_ny} (graded, beta=3, Nx=8)")
    lines.append(f"- exponent_clip: {exponent_clip}")
    lines.append(f"- Initializer: {initializer}")
    lines.append(f"- Stack: 3sp + Boltzmann ClO4- + log-c + log-rate + muh(H+)")
    lines.append(
        f"- Orchestrator: `solve_grid_per_voltage_cold_with_warm_fallback` "
        f"(C+D, n_substeps_warm={N_SUBSTEPS_WARM}, "
        f"bisect_depth_warm={BISECT_DEPTH_WARM})"
    )
    lines.append("")

    # Convergence matrix
    lines.append("## Convergence matrix (rows: C_S, cols: V_RHE)")
    lines.append("")
    header = "| C_S \\ V_RHE | " + " | ".join(f"{v:+.2f}" for v in v_test) + " |"
    sep = "|" + "|".join(["---"] * (len(v_test) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in reports:
        cells = []
        for i in range(len(v_test)):
            ok = r["converged"][i]
            method = r["method"][i]
            if ok:
                cells.append("✓" if method == "cold" else "✓ (warm)")
            else:
                cells.append("✗")
        lines.append("| " + r["cs_label"] + " | " + " | ".join(cells) + " |")
    lines.append("")

    # Gate verdicts
    lines.append("## Validation gates")
    lines.append("")
    for label, gate in (
        ("Gate 1 — Code regression: muh vs logc at C_S=None (HARD)", gate1),
        ("Gate 2 — Numerical convergence at peroxide V (REQUIRED)", gate2),
        ("Gate 3 — Physical validity: c_ClO4 ≤ steric cap (PHYSICS)", gate3),
        ("Gate 4 — Cross-formulation comparison (INFO)", gate4),
    ):
        lines.append(f"### {label}: **{gate['verdict']}**")
        lines.append("")
        for f in gate["findings"]:
            lines.append(f)
        lines.append("")

    # Decision
    lines.append("## Decision")
    lines.append("")
    lines.extend(_decision_block(gate1, gate2, gate3))
    lines.append("")

    # Per-C_S CD at the peroxide-window voltages
    lines.append("## Peroxide-window observables (CD, mA/cm²)")
    lines.append("")
    pw_v = [v for v in v_test if v >= 0.66]
    header = "| C_S | " + " | ".join(f"{v:+.2f}" for v in pw_v) + " |"
    sep = "|" + "|".join(["---"] * (len(pw_v) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in reports:
        idx_map = {round(v, 4): i for i, v in enumerate(r["v_rhe"])}
        cells = []
        for v in pw_v:
            i = idx_map.get(round(v, 4))
            if i is None:
                cells.append("—")
                continue
            cd = r["cd_mA_cm2"][i]
            cells.append(f"{cd:+.3e}" if cd is not None else "—")
        lines.append("| " + r["cs_label"] + " | " + " | ".join(cells) + " |")
    lines.append("")

    # Surface c_ClO4
    lines.append(f"## Surface c_ClO4 (nondim; steric cap ~{STERIC_CAP:g})")
    lines.append("")
    by_row = {(row["cs_label"], round(row["v_rhe"], 4)): row for row in rows}
    header = "| C_S | " + " | ".join(f"{v:+.2f}" for v in v_test) + " |"
    sep = "|" + "|".join(["---"] * (len(v_test) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in reports:
        cells = []
        for v in v_test:
            row = by_row.get((r["cs_label"], round(v, 4)))
            c = row.get("c_clo4_surface") if row else None
            if c is None:
                cells.append("—")
            elif c > STERIC_CAP:
                cells.append(f"**{c:.2e}**")
            else:
                cells.append(f"{c:.2e}")
        lines.append("| " + r["cs_label"] + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Bold values exceed the Bikerman steric scale "
                 "and indicate a non-physical converged state.")
    lines.append("")

    # Mu_H surface mean (muh-specific diagnostic)
    lines.append("## mu_H surface mean (muh-specific)")
    lines.append("")
    lines.append(
        "Raw mu_H = u_H + em*z_H*phi at the electrode.  The smaller the "
        "y-range of mu_H over the domain, the closer the diffuse layer is "
        "to Boltzmann equilibrium (analytic-cancellation property).  Bulk "
        "mu_H = log(c_H_bulk) ≈ -1.6 nondim."
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in reports:
        cells = []
        for v in v_test:
            row = by_row.get((r["cs_label"], round(v, 4)))
            mu_h = row.get("mu_h_surface") if row else None
            cells.append(f"{mu_h:+.3f}" if mu_h is not None else "—")
        lines.append("| " + r["cs_label"] + " | " + " | ".join(cells) + " |")
    lines.append("")

    # Stern voltage drop
    lines.append("## Stern voltage drop, phi_m - phi_s (nondim)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for r in reports:
        if r["cs_f_m2"] is None or r["cs_f_m2"] <= 0:
            continue
        cells = []
        for v in v_test:
            row = by_row.get((r["cs_label"], round(v, 4)))
            drop = row.get("phi_stern_drop_nondim") if row else None
            cells.append(f"{drop:+.3f}" if drop is not None else "—")
        lines.append("| " + r["cs_label"] + " | " + " | ".join(cells) + " |")
    lines.append("")

    # Artifacts
    lines.append("## Artifacts")
    lines.append("")
    rel = lambda fn: os.path.relpath(os.path.join(output_dir, fn), _ROOT)
    lines.append(f"- `{rel('iv_curve.json')}` — per-C_S CD/PC and convergence.")
    lines.append(f"- `{rel('diagnostics.json')}` — full per-(C_S, V) diagnostic dump.")
    lines.append(f"- `{rel('results.csv')}` — flat per-row dataset.")
    lines.append(f"- `{rel('comparison.png')}` — CD/PC + Stern drop vs V (when matplotlib is available).")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _make_comparison_plot(
    reports: list[dict],
    rows: list[dict],
    *,
    v_test: list[float],
    png_path: str,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(4, 1, hspace=0.35)
    ax_cd = fig.add_subplot(gs[0])
    ax_pc = fig.add_subplot(gs[1], sharex=ax_cd)
    ax_drop = fig.add_subplot(gs[2], sharex=ax_cd)
    ax_clo4 = fig.add_subplot(gs[3], sharex=ax_cd)

    cmap = plt.get_cmap("plasma")
    n_finite = sum(1 for r in reports
                   if r["cs_f_m2"] is not None and r["cs_f_m2"] > 0)
    finite_idx = 0
    by_row = {(row["cs_label"], round(row["v_rhe"], 4)): row for row in rows}

    for r in reports:
        v_arr = np.asarray(r["v_rhe"])
        cd = np.array([np.nan if x is None else x for x in r["cd_mA_cm2"]],
                      dtype=float)
        pc = np.array([np.nan if x is None else x for x in r["pc_mA_cm2"]],
                      dtype=float)
        if r["cs_f_m2"] is None:
            color, marker, label = "k", "o", "C_S=None (no Stern)"
        else:
            color = cmap(0.15 + 0.7 * (finite_idx / max(n_finite - 1, 1)))
            marker = "s"
            label = f"C_S={r['cs_label']} F/m²"
            finite_idx += 1
        ax_cd.plot(v_arr, cd, marker=marker, color=color, ls="-", label=label)
        ax_pc.plot(v_arr, pc, marker=marker, color=color, ls="-", label=label)

    finite_idx = 0
    for r in reports:
        v_arr = np.asarray(r["v_rhe"])
        if r["cs_f_m2"] is not None and r["cs_f_m2"] > 0:
            color = cmap(0.15 + 0.7 * (finite_idx / max(n_finite - 1, 1)))
            label = f"C_S={r['cs_label']} F/m²"
            drop = np.array([
                (by_row.get((r["cs_label"], round(vi, 4))) or {}).get(
                    "phi_stern_drop_nondim", np.nan)
                for vi in v_arr
            ], dtype=float)
            ax_drop.plot(v_arr, drop, marker="s", color=color, ls="-",
                         label=label)
            finite_idx += 1
        # c_ClO4 surface — for all rows.
        if r["cs_f_m2"] is None:
            color, marker, label = "k", "o", "C_S=None"
        else:
            label = f"C_S={r['cs_label']}"
            marker = "s"
        c_arr = np.array([
            (by_row.get((r["cs_label"], round(vi, 4))) or {}).get(
                "c_clo4_surface", np.nan)
            for vi in v_arr
        ], dtype=float)
        ax_clo4.plot(v_arr, c_arr, marker=marker, ls="-", label=label)

    for ax in (ax_cd, ax_pc, ax_drop, ax_clo4):
        ax.axvline(0.68, color="green", ls="--", lw=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)

    ax_cd.set_ylabel("CD (mA/cm²)")
    ax_cd.set_title(
        "Peroxide-window muh sweep -- 3sp + Boltzmann + log-c + log-rate + mu_H\n"
        f"(Ny={MESH_NY_DEFAULT}, clip={EXPONENT_CLIP_DEFAULT:.0f}, "
        f"initializer={INITIALIZER_DEFAULT}, dashed line: E_eq_R1=+0.68 V)"
    )
    ax_cd.legend(fontsize=7, loc="best")

    ax_pc.set_ylabel("PC (mA/cm²) [symlog]")
    ax_pc.set_yscale("symlog", linthresh=1e-6)
    ax_pc.legend(fontsize=7, loc="best")

    ax_drop.set_ylabel("phi_m - phi_s (nondim)")
    ax_drop.legend(fontsize=7, loc="best")

    ax_clo4.set_ylabel("Surface c_ClO4 (nondim)")
    ax_clo4.set_yscale("log")
    ax_clo4.axhline(STERIC_CAP, color="red", ls=":", lw=1.0,
                    label=f"steric cap ~{STERIC_CAP:g}")
    ax_clo4.set_xlabel("V vs RHE (V)")
    ax_clo4.legend(fontsize=7, loc="best")

    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = _parse_args()
    cs_list = _parse_cs_list(cli.cs_list)
    v_test = _parse_v_list(cli.v_list)

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  Peroxide-window muh sweep (formulation=logc_muh)")
    print("=" * 78)
    print(f"  V_TEST          = {v_test}")
    print(f"  C_S list (F/m²) = {['None' if c is None else c for c in cs_list]}")
    print(f"  mesh_Ny         = {cli.mesh_ny}")
    print(f"  exponent_clip   = {cli.clip}")
    print(f"  initializer     = {cli.initializer}")
    print(f"  formulation     = {FORMULATION_FIXED}")
    print(f"  output          = {out_dir}")
    print()

    reports: list[dict] = []
    t_start = time.time()
    for cs in cs_list:
        label = _cs_label(cs)
        print(f"--- pass: C_S = {label} F/m² (muh) ---")
        report = _run_one_cs(
            cs,
            v_rhe_grid=v_test,
            mesh_ny=cli.mesh_ny,
            exponent_clip=cli.clip,
            initializer=cli.initializer,
        )
        reports.append(report)
        print(f"  converged {report['n_converged']}/{report['n_total']}  "
              f"in {report['wall_seconds']:.1f}s")
        for i, v in enumerate(v_test):
            ok = report["converged"][i]
            cd = report["cd_mA_cm2"][i]
            pc = report["pc_mA_cm2"][i]
            cd_s = f"{cd:+.3e}" if cd is not None else "(none)"
            pc_s = f"{pc:+.3e}" if pc is not None else "(none)"
            d = report["diagnostics"][i] or {}
            phi_s = d.get("phi_surface_mean")
            phi_app = report["phi_applied_hat"][i]
            drop_str = (f"{phi_app - phi_s:+.2f}"
                        if phi_s is not None else "n/a")
            mu_h = d.get("mu2_surface_mean")
            mu_h_str = f"{mu_h:+.2f}" if mu_h is not None else "n/a"
            steric = d.get("surface_counterion_within_steric")
            print(f"    V={v:+.3f}  ok={ok}  cd={cd_s}  pc={pc_s}  "
                  f"method={report['method'][i]}  "
                  f"stern_drop={drop_str}  mu_H_surf={mu_h_str}  "
                  f"steric={steric}")
        print()

    rows = _per_row_records(reports, v_t=_get_v_t())

    iv_path = os.path.join(out_dir, "iv_curve.json")
    with open(iv_path, "w") as f:
        json.dump({
            "formulation":    FORMULATION_FIXED,
            "v_rhe":          v_test,
            "cs_list":        [None if c is None else float(c) for c in cs_list],
            "mesh_Ny":        int(cli.mesh_ny),
            "exponent_clip":  float(cli.clip),
            "initializer":    cli.initializer,
            "n_substeps_warm": int(N_SUBSTEPS_WARM),
            "bisect_depth_warm": int(BISECT_DEPTH_WARM),
            "steric_cap":     float(STERIC_CAP),
            "reports": [
                {k: v for k, v in r.items() if k != "diagnostics"}
                for r in reports
            ],
        }, f, indent=2)
    print(f"  iv_curve.json    -> {iv_path}")

    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({
            "formulation": FORMULATION_FIXED,
            "v_rhe": v_test,
            "reports": [
                {
                    "cs_label": r["cs_label"],
                    "cs_f_m2": r["cs_f_m2"],
                    "diagnostics_at_v": r["diagnostics"],
                }
                for r in reports
            ],
        }, f, indent=2, default=str)
    print(f"  diagnostics.json -> {diag_path}")

    csv_path = os.path.join(out_dir, "results.csv")
    _save_csv(rows, csv_path)
    print(f"  results.csv      -> {csv_path}")

    # Load logc baseline (Stern test).
    stern_iv_path = os.path.join(
        _ROOT, "StudyResults", STERN_BASELINE_DIR, "iv_curve.json"
    )
    baseline = _load_logc_baseline(stern_iv_path)
    if baseline:
        print(f"  loaded logc baseline: {stern_iv_path}")

    gate1 = _evaluate_gate1(reports, baseline)
    gate2 = _evaluate_gate2(reports)
    gate3 = _evaluate_gate3(reports)
    gate4 = _evaluate_gate4(reports, baseline)

    summary = _make_summary(
        reports=reports, rows=rows, v_test=v_test, cs_list=cs_list,
        mesh_ny=cli.mesh_ny, exponent_clip=cli.clip,
        initializer=cli.initializer,
        gate1=gate1, gate2=gate2, gate3=gate3, gate4=gate4,
        output_dir=out_dir,
    )
    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  summary.md       -> {summary_path}")

    png_path = os.path.join(out_dir, "comparison.png")
    err = _make_comparison_plot(reports, rows, v_test=v_test, png_path=png_path)
    if err is None:
        print(f"  comparison.png   -> {png_path}")
    else:
        print(f"  plot skipped: {err}")

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Total wall time:    {elapsed:.1f}s")
    print(f"  Gate 1 (regress):   {gate1['verdict']}")
    print(f"  Gate 2 (numerical): {gate2['verdict']}")
    print(f"  Gate 3 (physical):  {gate3['verdict']}")
    print(f"  Gate 4 (vs logc):   {gate4['verdict']}")
    print("=" * 78)


def _get_v_t() -> float:
    from scripts._bv_common import V_T
    return float(V_T)


if __name__ == "__main__":
    main()
