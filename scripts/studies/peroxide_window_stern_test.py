"""Peroxide-window Stern-capacitance sweep.

Tests whether finite Stern capacitance unblocks the peroxide-window wall
at ``V_RHE >= +0.68 V`` for the production 3sp + Boltzmann + log-c +
log-rate stack.  Per ``docs/stern_layer_physics_and_next_steps.md``,
finite Stern is a *physics branch* (not just a numerical trick): it
swaps the idealised ``phi_s = phi_m`` Dirichlet BC for a Robin BC that
lets the compact-layer voltage drop ``phi_m - phi_s`` be solved for,
and the BV overpotential becomes ``eta = phi_applied - phi - E_eq``.

The companion ``StudyResults/peroxide_window_pb_init_test/`` study
established that initializer-only changes (``debye_boltzmann``) do not
unblock the wall.  This study tests the next-step Stern hypothesis.

Sweep
-----
Outer loop:  C_S in [None, 0.05, 0.10, 0.20, 0.40, 1.00] F/m².
Inner loop:  V_RHE in [0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00] V.

At Ny=200 with the production stack and ``initializer="debye_boltzmann"``
(per ``docs/ic_refinement_study.md`` §5).

Validation gates (encoded in the emitted summary.md)
---------------------------------------------------
* **Gate 1 (HARD, code regression).** ``C_S = None`` row must reproduce
  the existing no-Stern baseline at every voltage where the baseline
  converges, within ``rel_tol = 1e-6``.  ``C_S = 0.0`` row must be
  runtime-equivalent to ``None``.  FAIL on this gate aborts the
  physics analysis: the wiring is broken.
* **Gate 2 (PHYSICS, peroxide-window success).** For at least one
  finite ``C_S`` in the sweep:
    - ``V_RHE in {0.68, 0.70, 0.75}`` all converge.
    - ``c_clo4_surface`` stays below the Bikerman steric scale (~100
      nondim) at every converged voltage, OR the row is flagged as
      "physically suspect" in the summary.
* **Gate 3 (SOFT, large-C_S consistency).** The largest finite
  ``C_S`` (1.00 F/m²) should approach the no-Stern result on the
  no-Stern-converging window (V <= 0.66).  WARN-only — the Robin →
  Dirichlet limit is numerically stiff and doesn't have to match
  exactly.

Outputs
-------
``StudyResults/peroxide_window_stern_test/iv_curve.json``
``StudyResults/peroxide_window_stern_test/diagnostics.json``
``StudyResults/peroxide_window_stern_test/results.csv``
``StudyResults/peroxide_window_stern_test/summary.md``
``StudyResults/peroxide_window_stern_test/comparison.png``  (optional)

Run from PNPInverse/ with ../venv-firedrake/bin/activate active::

    python scripts/studies/peroxide_window_stern_test.py
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
# C_S values; ``None`` is the no-Stern baseline branch.
CS_DEFAULT = (None, 0.05, 0.10, 0.20, 0.40, 1.00)
MESH_NY_DEFAULT = 200
EXPONENT_CLIP_DEFAULT = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
STERIC_CAP = 100.0
INITIALIZER_DEFAULT = "debye_boltzmann"
OUT_SUBDIR = "peroxide_window_stern_test"

# Pre-Stern baseline for Gate 1 (debye_boltzmann initializer at V=+0.66 V).
# Refreshed 2026-05-04 after the steric chemical-potential sign fix at
# Forward/bv_solver/forms_logc.py:266 (mu_steric: +ln(1-Phi) -> -ln(1-Phi)).
# The 3sp+Boltzmann path uses a_vals_hat=[0.01]*3, so its residual is
# affected by the sign flip; pre-fix values shifted by ~0.36% on both
# CD and PC.  See docs/4sp_drop_boltzmann_investigation.md "Resolution"
# section for context.
BASELINE_NO_STERN = {
    0.66: {
        "cd_mA_cm2": 1.2968453558282709e-08,
        "pc_mA_cm2": 1.2969358369725412e-08,
    },
}
GATE1_REL_TOL = 1e-6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
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
    if cs is None:
        return "None"
    return f"{cs:g}"


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
        formulation="logc", log_rate=True,
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

# In nondim mode (production), the BV exponent scale is 1.0, so
# ``eta_scaled = eta_raw`` and the clip applies directly to eta_raw.
_BV_EXP_SCALE_NONDIM = 1.0
# ``E_eq`` values (V).  Reaction 1 = ORR (O2 → H2O2), Reaction 2 = HOR
# (H2O2 → H2O).  Match the values used in
# scripts/studies/peroxide_window_pb_init_test.py.
_E_EQ_R1_V = 0.68
_E_EQ_R2_V = 1.78


def _eta_clipped_active(
    *,
    phi_applied_hat: float,
    phi_surface_hat: Optional[float],
    cs: Optional[float],
    e_eq_v: float,
    v_t: float,
    exponent_clip: float,
) -> Optional[bool]:
    """Whether the BV exponent for a given reaction is at the clip floor.

    With no Stern, ``eta_raw = phi_applied - E_eq``.  With finite Stern
    active and a converged ``phi_surface`` available,
    ``eta_raw = phi_applied - phi_surface - E_eq``.  Both in nondim
    units (V_T as scale).
    """
    e_eq_hat = e_eq_v / v_t
    if cs is not None and cs > 0 and phi_surface_hat is not None:
        eta_raw = phi_applied_hat - phi_surface_hat - e_eq_hat
    else:
        eta_raw = phi_applied_hat - e_eq_hat
    eta_scaled = _BV_EXP_SCALE_NONDIM * eta_raw
    return bool(abs(eta_scaled) >= exponent_clip)


def _per_row_records(
    reports: list[dict],
    *,
    v_t: float,
    exponent_clip: float,
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
                "phi_diffuse_drop_nondim": phi_surf_hat,  # bulk = 0
                "c_o2_surface":          d.get("c0_surface_mean"),
                "c_h2o2_surface":        d.get("c1_surface_mean"),
                "c_h_surface":           d.get("c2_surface_mean"),
                "c_clo4_surface":        d.get("c_counterion0_surface_mean"),
                "surface_within_steric": d.get("surface_counterion_within_steric"),
                "eta_clipped_active_r1": _eta_clipped_active(
                    phi_applied_hat=phi_app_hat,
                    phi_surface_hat=phi_surf_hat,
                    cs=r["cs_f_m2"],
                    e_eq_v=_E_EQ_R1_V,
                    v_t=v_t,
                    exponent_clip=exponent_clip,
                ),
                "eta_clipped_active_r2": _eta_clipped_active(
                    phi_applied_hat=phi_app_hat,
                    phi_surface_hat=phi_surf_hat,
                    cs=r["cs_f_m2"],
                    e_eq_v=_E_EQ_R2_V,
                    v_t=v_t,
                    exponent_clip=exponent_clip,
                ),
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
# Validation gates
# ---------------------------------------------------------------------------

GATE2_VOLTAGES = (0.68, 0.70, 0.75)


def _evaluate_gate1(reports: list[dict]) -> dict[str, Any]:
    """Code-regression: C_S=None matches baseline; C_S=0.0 matches None."""
    findings: list[str] = []
    by_label = {r["cs_label"]: r for r in reports}
    none_r = by_label.get("None")
    zero_r = by_label.get("0")  # 0.0 prints as "0" via :g

    if none_r is None:
        findings.append("FAIL: no C_S=None row in sweep.")
        return {"verdict": "FAIL", "findings": findings}

    # 1a. C_S=None vs baseline
    none_violations = []
    for i, v in enumerate(none_r["v_rhe"]):
        if not none_r["converged"][i]:
            continue
        v_round = round(v, 4)
        base = BASELINE_NO_STERN.get(v_round)
        if base is None:
            continue
        cd = none_r["cd_mA_cm2"][i]
        pc = none_r["pc_mA_cm2"][i]
        if cd is None or pc is None:
            none_violations.append(
                f"V={v:+.3f}: converged but CD/PC is None")
            continue
        cd_drift = abs(cd - base["cd_mA_cm2"]) / abs(base["cd_mA_cm2"])
        pc_drift = abs(pc - base["pc_mA_cm2"]) / abs(base["pc_mA_cm2"])
        if cd_drift > GATE1_REL_TOL or pc_drift > GATE1_REL_TOL:
            none_violations.append(
                f"V={v:+.3f}: CD drift={cd_drift:.2e}, PC drift={pc_drift:.2e}"
                f" (tol={GATE1_REL_TOL:.0e})"
            )
    if not none_violations:
        findings.append(
            f"PASS: C_S=None reproduces no-Stern baseline at all checked "
            f"voltages within rel_tol={GATE1_REL_TOL:.0e}."
        )
    else:
        findings.append("FAIL: C_S=None drifted from baseline:")
        findings.extend(f"  - {x}" for x in none_violations)

    # 1b. C_S=0.0 vs C_S=None at runtime
    if zero_r is None:
        findings.append(
            "INFO: no C_S=0.0 row in sweep (skipping runtime-equivalence check)."
        )
    else:
        zero_violations = []
        for i, v in enumerate(none_r["v_rhe"]):
            none_ok = none_r["converged"][i]
            zero_ok = zero_r["converged"][i]
            if none_ok != zero_ok:
                zero_violations.append(
                    f"V={v:+.3f}: convergence differs (None={none_ok}, 0.0={zero_ok})"
                )
                continue
            if not none_ok:
                continue
            cd_n, pc_n = none_r["cd_mA_cm2"][i], none_r["pc_mA_cm2"][i]
            cd_z, pc_z = zero_r["cd_mA_cm2"][i], zero_r["pc_mA_cm2"][i]
            if cd_n is None or cd_z is None or pc_n is None or pc_z is None:
                continue
            cd_drift = abs(cd_z - cd_n) / max(abs(cd_n), 1e-30)
            pc_drift = abs(pc_z - pc_n) / max(abs(pc_n), 1e-30)
            if cd_drift > GATE1_REL_TOL or pc_drift > GATE1_REL_TOL:
                zero_violations.append(
                    f"V={v:+.3f}: CD drift={cd_drift:.2e}, "
                    f"PC drift={pc_drift:.2e} (tol={GATE1_REL_TOL:.0e})"
                )
        if not zero_violations:
            findings.append(
                f"PASS: C_S=0.0 is runtime-equivalent to C_S=None within "
                f"rel_tol={GATE1_REL_TOL:.0e}."
            )
        else:
            findings.append("FAIL: C_S=0.0 differs from C_S=None at runtime:")
            findings.extend(f"  - {x}" for x in zero_violations)

    verdict = "PASS" if all(f.startswith("PASS") or f.startswith("INFO")
                            for f in findings) else "FAIL"
    return {"verdict": verdict, "findings": findings}


def _evaluate_gate2(reports: list[dict]) -> dict[str, Any]:
    """Peroxide-window success: at least one finite C_S converges at
    {0.68, 0.70, 0.75} V with surface c_clo4 within steric scale."""
    successes: list[dict] = []
    near_misses: list[dict] = []
    for r in reports:
        if r["cs_f_m2"] is None or r["cs_f_m2"] <= 0:
            continue
        idx_map = {round(v, 4): i for i, v in enumerate(r["v_rhe"])}
        converged_at = {}
        steric_at = {}
        for v_target in GATE2_VOLTAGES:
            i = idx_map.get(round(v_target, 4))
            if i is None:
                continue
            converged_at[v_target] = bool(r["converged"][i])
            d = r["diagnostics"][i] or {}
            steric_at[v_target] = d.get("surface_counterion_within_steric")
        all_conv = all(converged_at.get(v, False) for v in GATE2_VOLTAGES)
        all_within_steric = all(steric_at.get(v) is True for v in GATE2_VOLTAGES)
        record = {
            "cs_label": r["cs_label"],
            "cs_f_m2": r["cs_f_m2"],
            "converged_at": converged_at,
            "steric_at": steric_at,
            "all_converged": all_conv,
            "all_within_steric": all_within_steric,
        }
        if all_conv:
            successes.append(record)
        else:
            near_misses.append(record)

    findings: list[str] = []
    if not successes:
        findings.append(
            f"FAIL: no finite C_S converged at all of "
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
        physical = [s for s in successes if s["all_within_steric"]]
        if physical:
            findings.append(
                f"PASS: {len(physical)} finite C_S value(s) converged at all "
                f"of {list(GATE2_VOLTAGES)} V with surface c_ClO4 within "
                f"steric scale (~{STERIC_CAP:g} nondim)."
            )
            for s in physical:
                findings.append(
                    f"  - C_S={s['cs_label']} F/m²: physically plausible."
                )
        else:
            findings.append(
                f"WARN: {len(successes)} finite C_S value(s) converged at all "
                f"of {list(GATE2_VOLTAGES)} V, but surface c_ClO4 exceeded "
                f"the steric scale at one or more voltages.  Converged but "
                f"physically suspect."
            )
        for s in successes:
            steric_str = ", ".join(
                f"V={v:+.2f}: within={s['steric_at'].get(v)}"
                for v in GATE2_VOLTAGES
            )
            findings.append(f"  - C_S={s['cs_label']}: {steric_str}")
        verdict = "PASS" if physical else "WARN"
    return {
        "verdict": verdict, "findings": findings,
        "successes": successes, "near_misses": near_misses,
    }


def _evaluate_gate3(reports: list[dict]) -> dict[str, Any]:
    """Large-C_S consistency: largest finite C_S approaches no-Stern on
    overlap window (V <= 0.66).  WARN-only."""
    findings: list[str] = []
    by_label = {r["cs_label"]: r for r in reports}
    none_r = by_label.get("None")
    finite = [r for r in reports if r["cs_f_m2"] is not None and r["cs_f_m2"] > 0]
    if none_r is None or not finite:
        findings.append("INFO: cannot compute (missing None or finite branch).")
        return {"verdict": "INFO", "findings": findings}

    largest = max(finite, key=lambda r: r["cs_f_m2"])
    findings.append(
        f"Comparison: C_S={largest['cs_label']} F/m² (largest) vs C_S=None "
        f"on V <= 0.66 V."
    )
    deviations: list[float] = []
    for i, v in enumerate(none_r["v_rhe"]):
        if v > 0.66:
            continue
        if not (none_r["converged"][i] and largest["converged"][i]):
            findings.append(f"  V={v:+.3f}: convergence differs.")
            continue
        cd_n = none_r["cd_mA_cm2"][i]
        cd_l = largest["cd_mA_cm2"][i]
        if cd_n is None or cd_l is None:
            continue
        rel = abs(cd_l - cd_n) / max(abs(cd_n), 1e-30)
        deviations.append(rel)
        findings.append(f"  V={v:+.3f}: CD rel deviation = {rel:.2e}.")
    if deviations:
        max_dev = max(deviations)
        if max_dev < 0.05:
            findings.append(
                f"INFO: max deviation {max_dev:.2e} < 5%, large-C_S limit "
                f"is consistent with no-Stern."
            )
            verdict = "INFO"
        else:
            findings.append(
                f"WARN: max deviation {max_dev:.2e} >= 5%.  Robin → "
                f"Dirichlet limit is numerically stiff at this C_S; this "
                f"is not necessarily a code bug."
            )
            verdict = "WARN"
    else:
        verdict = "INFO"
    return {"verdict": verdict, "findings": findings}


def _decision_block(
    gate1: dict[str, Any],
    gate2: dict[str, Any],
) -> list[str]:
    """Decision rules quoted from
    docs/stern_layer_physics_and_next_steps.md §6."""
    out: list[str] = []
    if gate1["verdict"] == "FAIL":
        out.append("**Decision: HALT.** Code regression failed (Gate 1).  Do "
                   "not proceed to physics analysis until the wiring is "
                   "fixed.  Roll back ``scripts/_bv_common.py`` and "
                   "investigate.")
        return out

    if gate2["verdict"] == "PASS":
        successes = gate2.get("successes", [])
        physical = [s for s in successes if s["all_within_steric"]]
        labels = ", ".join(f"C_S={s['cs_label']} F/m²" for s in physical)
        out.append(
            f"**Decision: Stern is the preferred peroxide-window branch.** "
            f"Plausible Stern value(s): {labels}.  Recommended next steps:"
        )
        out.append("1. Inspect the per-C_S sensitivity around the selected "
                   "value(s) — small changes in C_S should give smooth, "
                   "continuous CD/PC variation.")
        out.append("2. Update inverse/plot scripts to make the model branch "
                   "explicit.  Suggested labels:")
        out.append("   ```")
        out.append("   model = \"no_stern\"")
        for s in physical:
            cs_tag = (str(s["cs_f_m2"]).replace(".", "p")
                      .replace("-", "neg"))
            out.append(f"   model = \"stern_Cs_{cs_tag}_F_m2\"")
        out.append("   ```")
        out.append("3. Do **not** silently replace historical no-Stern "
                   "results.  Treat finite Stern as a separate branch.")
    elif gate2["verdict"] == "WARN":
        out.append(
            "**Decision: Stern unblocks the wall but is physically "
            "suspect.** Some finite C_S converged at the peroxide-window "
            "voltages, but surface c_ClO4 exceeded the steric scale at one "
            "or more voltages.  Treat the converged state as a "
            "Newton-converged-but-non-physical solution; the next step is "
            "either Bikerman/steric saturation in the residual or "
            "``formulation=\"logc_muH\"``."
        )
    else:  # FAIL
        out.append(
            "**Decision: Stern alone does not cross the peroxide-window "
            "wall.** Per ``docs/stern_layer_physics_and_next_steps.md`` §6, "
            "next steps are:"
        )
        out.append("1. Keep this study as a negative result.")
        out.append("2. Proceed to PB initializer with exponent homotopy.")
        out.append("3. Only then move to ``formulation=\"logc_muH\"`` as the "
                   "larger formulation rewrite.")
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
    output_dir: str,
) -> str:
    lines: list[str] = []
    lines.append("# Peroxide-window Stern-capacitance sweep")
    lines.append("")
    lines.append("Study script: `scripts/studies/peroxide_window_stern_test.py`.")
    lines.append("Physics rationale: "
                 "`docs/stern_layer_physics_and_next_steps.md`.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- V_RHE grid: {v_test}")
    lines.append(
        f"- C_S grid (F/m²): "
        f"{[('None' if c is None else c) for c in cs_list]}")
    lines.append(f"- Mesh Ny: {mesh_ny} (graded, beta=3, Nx=8)")
    lines.append(f"- exponent_clip: {exponent_clip}")
    lines.append(f"- Initializer: {initializer}")
    lines.append(f"- Stack: 3sp + Boltzmann ClO4- + log-c + log-rate")
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
    for label, gate in (("Gate 1 — Code regression (HARD)", gate1),
                        ("Gate 2 — Peroxide-window success (PHYSICS)", gate2),
                        ("Gate 3 — Large-C_S consistency (SOFT)", gate3)):
        lines.append(f"### {label}: **{gate['verdict']}**")
        lines.append("")
        for f in gate["findings"]:
            lines.append(f)
        lines.append("")

    # Decision
    lines.append("## Decision")
    lines.append("")
    lines.extend(_decision_block(gate1, gate2))
    lines.append("")

    # Per-C_S CD/PC at the peroxide-window voltages
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

    # Stern voltage drop (only for finite C_S)
    lines.append("## Stern voltage drop, phi_m - phi_s (nondim)")
    lines.append("")
    header = "| C_S | " + " | ".join(f"{v:+.2f}" for v in v_test) + " |"
    sep = "|" + "|".join(["---"] * (len(v_test) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    by_row = {(row["cs_label"], round(row["v_rhe"], 4)): row for row in rows}
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

    # Surface c_ClO4
    lines.append(f"## Surface c_ClO4 (nondim; steric cap ~{STERIC_CAP:g})")
    lines.append("")
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

    # Artifacts
    lines.append("## Artifacts")
    lines.append("")
    rel = lambda fn: os.path.relpath(os.path.join(output_dir, fn), _ROOT)
    lines.append(f"- `{rel('iv_curve.json')}` — per-C_S CD/PC and convergence.")
    lines.append(f"- `{rel('diagnostics.json')}` — full per-(C_S, V) "
                 "diagnostic dump from `collect_diagnostics`.")
    lines.append(f"- `{rel('results.csv')}` — flat per-row dataset for "
                 "external analysis.")
    lines.append(f"- `{rel('comparison.png')}` — CD/PC and Stern drop vs V "
                 "by C_S (when matplotlib is available).")
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

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.32)
    ax_cd = fig.add_subplot(gs[0])
    ax_pc = fig.add_subplot(gs[1], sharex=ax_cd)
    ax_drop = fig.add_subplot(gs[2], sharex=ax_cd)

    cmap = plt.get_cmap("viridis")
    n_finite = sum(1 for r in reports if r["cs_f_m2"] is not None
                   and r["cs_f_m2"] > 0)
    finite_idx = 0

    for r in reports:
        v = np.asarray(r["v_rhe"])
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
        ax_cd.plot(v, cd, marker=marker, color=color, ls="-", label=label)
        ax_pc.plot(v, pc, marker=marker, color=color, ls="-", label=label)

    # Stern drop subplot — only for finite C_S
    by_row = {(row["cs_label"], round(row["v_rhe"], 4)): row for row in rows}
    finite_idx = 0
    for r in reports:
        if r["cs_f_m2"] is None or r["cs_f_m2"] <= 0:
            continue
        v = np.asarray(r["v_rhe"])
        drop = np.array([
            (by_row.get((r["cs_label"], round(vi, 4))) or {}).get(
                "phi_stern_drop_nondim", np.nan)
            for vi in v
        ], dtype=float)
        color = cmap(0.15 + 0.7 * (finite_idx / max(n_finite - 1, 1)))
        ax_drop.plot(v, drop, marker="s", color=color, ls="-",
                     label=f"C_S={r['cs_label']} F/m²")
        finite_idx += 1

    for ax in (ax_cd, ax_pc, ax_drop):
        ax.axvline(0.68, color="green", ls="--", lw=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)

    ax_cd.set_ylabel("CD (mA/cm²)")
    ax_cd.set_title(
        "Peroxide-window Stern sweep — production stack\n"
        "(3sp + Boltzmann + log-c + log-rate, "
        f"Ny={MESH_NY_DEFAULT}, clip={EXPONENT_CLIP_DEFAULT:.0f}, "
        f"initializer={INITIALIZER_DEFAULT}, dashed line: E_eq_R1=+0.68 V)"
    )
    ax_cd.legend(fontsize=7, loc="best")

    ax_pc.set_ylabel("PC (mA/cm²) [symlog]")
    ax_pc.set_yscale("symlog", linthresh=1e-6)
    ax_pc.legend(fontsize=7, loc="best")

    ax_drop.set_ylabel("phi_m - phi_s (nondim)")
    ax_drop.set_xlabel("V vs RHE (V)")
    ax_drop.legend(fontsize=7, loc="best")

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
    print("  Peroxide-window Stern-capacitance sweep")
    print("=" * 78)
    print(f"  V_TEST          = {v_test}")
    print(f"  C_S list (F/m²) = {['None' if c is None else c for c in cs_list]}")
    print(f"  mesh_Ny         = {cli.mesh_ny}")
    print(f"  exponent_clip   = {cli.clip}")
    print(f"  initializer     = {cli.initializer}")
    print(f"  output          = {out_dir}")
    print()

    reports: list[dict] = []
    t_start = time.time()
    for cs in cs_list:
        label = _cs_label(cs)
        print(f"--- pass: C_S = {label} F/m² ---")
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
            steric = d.get("surface_counterion_within_steric")
            print(f"    V={v:+.3f}  ok={ok}  cd={cd_s}  pc={pc_s}  "
                  f"method={report['method'][i]}  "
                  f"stern_drop={drop_str}  steric={steric}")
        print()

    rows = _per_row_records(reports, v_t=_get_v_t(), exponent_clip=cli.clip)

    iv_path = os.path.join(out_dir, "iv_curve.json")
    with open(iv_path, "w") as f:
        json.dump({
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

    gate1 = _evaluate_gate1(reports)
    gate2 = _evaluate_gate2(reports)
    gate3 = _evaluate_gate3(reports)

    summary = _make_summary(
        reports=reports, rows=rows, v_test=v_test, cs_list=cs_list,
        mesh_ny=cli.mesh_ny, exponent_clip=cli.clip,
        initializer=cli.initializer,
        gate1=gate1, gate2=gate2, gate3=gate3,
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
    print(f"  Gate 2 (physics):   {gate2['verdict']}")
    print(f"  Gate 3 (large C_S): {gate3['verdict']}")
    print("=" * 78)


def _get_v_t() -> float:
    """Lazy import of V_T to avoid pulling Nondim before main()."""
    from scripts._bv_common import V_T
    return float(V_T)


if __name__ == "__main__":
    main()
