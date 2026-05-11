"""Phase 6 beta v10b C_S sensitivity bracket sweep at V_kin.

Per plan section D7-D1.  Runs the v10b stack at V_kin = -0.10 V, lambda
= 1.0, k_hyd_baseline = 1e-3 nondim across the C_S bracket
``{0.05, 0.10, 0.20, 0.30} F/m^2`` using the two-stage anchor pattern
(build anchor at STERN_F_M2_ANCHOR = 0.10, runtime-bump to target C_S
via set_stern_capacitance_model + Newton resolve).

Hard gates (failure -> escalate to v10c; no 3/4 fallback):

* 4/4 mandatory.
* ``cd_mA_cm2 < 0`` (cathodic) at V_kin for each rung.
* No R_4e sign flip: ``R_4e_current_nondim > 0`` at every rung
  where ``|R_4e_current_nondim| > 1e-6``.
* ``R_net >= 0`` at every rung (positive by construction).
* Per-rung analytic-vs-solver mass-balance: rel <= 5e-3 via
  ``gamma_ss_langmuir`` with per-rung state.

Output: ``StudyResults/phase6b_v10b_cs_bracket/cs_bracket.{json,png}``
and accompanying summary.

Lazy imports per plan section 4 Phase v10b.D: ``gamma_ss_langmuir``
plus Firedrake plus ``Forward.bv_solver.*`` plus
``scripts.studies.phase6b_v10a_v_sweep_diagnostic`` plus
``scripts.studies.phase6b_v10a_phase_A2_v_kin`` are imported INSIDE
the solver-running function so that CLI / schema tests stay
Firedrake-free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Module-level constants (stdlib + argparse + numpy + json only at module
# scope per Round 6 + Round 7 lazy-import policy).
# ---------------------------------------------------------------------------

V_KIN_DEFAULT: float = -0.10
V_ANCHOR_DEFAULT: float = 0.55
K0_R4E_FACTOR_DEFAULT: float = 1e-14
K_HYD_BASELINE: float = 1e-3
LAMBDA_FINAL: float = 1.0
OUT_SUBDIR_DEFAULT: str = "phase6b_v10b_cs_bracket"

CS_BRACKET: tuple = (0.05, 0.10, 0.20, 0.30)
"""C_S sensitivity rungs per plan D7-D1."""

MASS_BALANCE_REL_MAX: float = 5e-3
"""Per-rung HARD gate: analytic-vs-solver Gamma relative mismatch."""

R_4E_SIGN_FLOOR: float = 1e-6
"""Below this |R_4e_current_nondim| floor the sign gate is N/A."""


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 beta v10b -- C_S sensitivity bracket at V_kin "
            "with two-stage anchor pattern + per-rung analytic Gamma "
            "HARD gate."
        ),
    )
    parser.add_argument("--v-kin", type=float, default=V_KIN_DEFAULT)
    parser.add_argument("--v-anchor", type=float, default=V_ANCHOR_DEFAULT)
    parser.add_argument(
        "--k0-r4e-factor", type=float, default=K0_R4E_FACTOR_DEFAULT,
    )
    parser.add_argument("--k-hyd", type=float, default=K_HYD_BASELINE)
    parser.add_argument(
        "--cs-bracket", default=",".join(str(v) for v in CS_BRACKET),
        help=(
            "Comma-separated C_S values (F/m^2).  Default "
            f"{','.join(str(v) for v in CS_BRACKET)}."
        ),
    )
    parser.add_argument(
        "--out-subdir", default=OUT_SUBDIR_DEFAULT,
    )
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser.parse_args(argv)


def _parse_cs_bracket(s: str) -> tuple:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        val = float(p)
        if val <= 0.0:
            raise ValueError(
                f"--cs-bracket entries must be positive (got {val!r})"
            )
        out.append(val)
    if not out:
        raise ValueError("--cs-bracket must contain at least one entry")
    return tuple(out)


# ---------------------------------------------------------------------------
# Per-rung HARD-gate evaluation -- pure Python (no Firedrake).
# ---------------------------------------------------------------------------


def _evaluate_hard_gates(
    *,
    lam1: Dict[str, Any],
    analytic_gamma_clamped: float,
) -> Dict[str, Any]:
    """Return the v10b D7-D1 HARD-gate evaluation for a single rung."""
    cd = lam1.get("cd_mA_cm2")
    r4 = lam1.get("R_4e_current_nondim")
    r_net = lam1.get("R_net")
    gamma_solver = lam1.get("gamma")

    cd_ok = cd is not None and float(cd) < 0.0
    if r4 is None or abs(float(r4)) <= R_4E_SIGN_FLOOR:
        r4_sign_ok = True
        r4_note = "below_sign_floor"
    else:
        r4_sign_ok = float(r4) > 0.0
        r4_note = "ok" if r4_sign_ok else "negative"
    if r_net is None:
        r_net_ok = False
        r_net_note = "missing"
    else:
        r_net_ok = float(r_net) >= 0.0
        r_net_note = "ok" if r_net_ok else "negative"

    if gamma_solver is None or not (gamma_solver > 0.0):
        mb_rel = None
        mb_ok = False
    else:
        denom = max(
            abs(float(gamma_solver)),
            abs(float(analytic_gamma_clamped)),
            1e-12,
        )
        mb_rel = (
            abs(float(gamma_solver) - float(analytic_gamma_clamped))
            / denom
        )
        mb_ok = bool(mb_rel <= MASS_BALANCE_REL_MAX)

    return {
        "cd_ok": bool(cd_ok),
        "cd_mA_cm2": cd,
        "r4_sign_ok": bool(r4_sign_ok),
        "r4_note": r4_note,
        "R_4e_current_nondim": r4,
        "r_net_ok": bool(r_net_ok),
        "r_net_note": r_net_note,
        "R_net": r_net,
        "analytic_solver_gamma_rel": mb_rel,
        "analytic_solver_gamma_ok": bool(mb_ok),
        "gamma_max_rel_threshold": MASS_BALANCE_REL_MAX,
        "pass": bool(cd_ok and r4_sign_ok and r_net_ok and mb_ok),
    }


# ---------------------------------------------------------------------------
# Lazy-imported main driver.
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cs_bracket = _parse_cs_bracket(args.cs_bracket)
    v_kin = float(args.v_kin)
    v_anchor = float(args.v_anchor)
    k0_r4e_factor = float(args.k0_r4e_factor)
    k_hyd = float(args.k_hyd)
    out_dir = os.path.join(_ROOT, "StudyResults", args.out_subdir)

    print(
        f"=== Phase 6 beta v10b C_S sensitivity bracket ===",
        flush=True,
    )
    print(
        f"  V_kin={v_kin:+.3f} V, v_anchor={v_anchor:+.3f} V, "
        f"K0_R4e_factor={k0_r4e_factor:.3g}",
        flush=True,
    )
    print(f"  k_hyd={k_hyd:.3e}, C_S bracket={cs_bracket}", flush=True)
    print(f"  Output: {out_dir}", flush=True)

    # Lazy imports (Round 6 + Round 7 patch P38).
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        L_EFF_M_BASELINE, STERN_F_M2_ANCHOR, MESH_NX, MESH_NY, MESH_BETA,
        LAMBDA_LADDER, _build_sp, _make_mesh, _serialize,
        _i_lim_4e_mA_cm2, _walk_lambda_zero_capture_snapshots,
    )
    from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
        _run_k_hyd_ramp, lambda1_record,
    )
    from scripts._bv_common import (
        I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T,
    )
    from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
    from calibration.v10b import (
        V10B_KINETICS, V10B_CALIBRATION_METADATA,
    )

    warm_grid = (0.55, 0.40, 0.20, 0.10, -0.10)
    if not any(abs(v - v_kin) < 1e-9 for v in warm_grid):
        warm_grid = tuple(sorted(list(warm_grid) + [v_kin], reverse=True))

    t_start = time.time()

    # Build template SP at production C_S (the two-stage anchor inside
    # _walk_lambda_zero_capture_snapshots already builds at
    # STERN_F_M2_ANCHOR = 0.10 and runtime-bumps to whatever the
    # SP declares).  C_S overrides per rung happen later via
    # parameter_overrides through _run_k_hyd_ramp.
    sp = _build_sp(
        lambda_hydrolysis=0.0,
        k0_r4e_factor=k0_r4e_factor,
        k_hyd_nondim=k_hyd,
    )
    mesh = _make_mesh(l_eff_m=L_EFF_M_BASELINE)

    print(
        f"Pass 1: anchor at V={v_anchor:+.3f} V "
        f"-> warm-walk {warm_grid}",
        flush=True,
    )
    (warm_records, snapshots, mesh_dof_count,
     electrode_area_nondim, electrode_marker) = (
        _walk_lambda_zero_capture_snapshots(
            sp=sp, mesh=mesh,
            v_rhe_grid=warm_grid,
            v_anchor=v_anchor,
            k0_r4e_factor=k0_r4e_factor,
        )
    )

    v_kin_idx: Optional[int] = None
    for idx, rec in enumerate(warm_records):
        if abs(float(rec["v_rhe"]) - v_kin) < 1e-9:
            v_kin_idx = idx
            break
    if v_kin_idx is None or v_kin_idx not in snapshots:
        print(
            f"ERROR: warm-walk did not converge at V_kin={v_kin:+.3f} V",
            flush=True,
        )
        return 2
    U_at_v_kin = snapshots[v_kin_idx]
    print(
        f"  warm-walk produced U snapshot at V_kin={v_kin:+.3f} V "
        f"(idx={v_kin_idx})",
        flush=True,
    )

    domain_height_hat = L_EFF_M_BASELINE / 1.0e-4
    i_lim_4e = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)

    # Pass 2 -- C_S bracket sweep at V_kin via parameter_overrides.
    rung_records: List[Dict[str, Any]] = []
    for i, cs_val in enumerate(cs_bracket):
        t0 = time.time()
        print(
            f"Pass 2.{i+1}/{len(cs_bracket)}: "
            f"C_S={cs_val:.3f} F/m^2",
            flush=True,
        )
        rec = _run_k_hyd_ramp(
            sp_template=sp, mesh=mesh, voltage=v_kin,
            U_warmstart=U_at_v_kin, k_hyd_target=k_hyd,
            k0_r4e_factor=k0_r4e_factor,
            i_scale=float(I_SCALE),
            i_lim_4e_mA_cm2=i_lim_4e,
            electrode_area_nondim=electrode_area_nondim,
            domain_height_hat=domain_height_hat,
            extra_overrides={"stern_capacitance_f_m2": float(cs_val)},
        )
        lam1 = lambda1_record(rec)

        # Analytic Gamma via gamma_ss_langmuir using per-rung state.
        if lam1 is not None and lam1.get("snes_converged", False):
            # Derive forward_avg_no_k_hyd via F0_avg / k_hyd if F0_avg
            # is the only field emitted (per Round 5 issue #2).
            f0_avg = lam1.get("F0_avg")
            forward_avg_no_k_hyd = lam1.get("forward_avg_no_k_hyd")
            if forward_avg_no_k_hyd is None and f0_avg is not None:
                k_hyd_rung = float(lam1.get("k_hyd", k_hyd))
                if k_hyd_rung > 0.0:
                    forward_avg_no_k_hyd = float(f0_avg) / k_hyd_rung
            c_H_avg = lam1.get("c_H_boundary_avg")
            if (
                forward_avg_no_k_hyd is not None
                and c_H_avg is not None
                and lam1.get("k_prot") is not None
                and lam1.get("k_des") is not None
                and lam1.get("delta_ohp_hat") is not None
                and lam1.get("gamma_max") is not None
            ):
                analytic_clamped, analytic_unclamped, denom_decomp = (
                    gamma_ss_langmuir(
                        lambda_val=LAMBDA_FINAL,
                        k_hyd=float(lam1["k_hyd"]),
                        k_prot=float(lam1["k_prot"]),
                        k_des=float(lam1["k_des"]),
                        delta_ohp=float(lam1["delta_ohp_hat"]),
                        forward_avg=float(forward_avg_no_k_hyd),
                        c_H_avg=float(c_H_avg),
                        gamma_max=float(lam1["gamma_max"]),
                    )
                )
            else:
                analytic_clamped = float("nan")
                analytic_unclamped = float("nan")
                denom_decomp = None
            gates = _evaluate_hard_gates(
                lam1=lam1,
                analytic_gamma_clamped=analytic_clamped,
            )
        else:
            analytic_clamped = float("nan")
            analytic_unclamped = float("nan")
            denom_decomp = None
            gates = {
                "cd_ok": False, "r4_sign_ok": False, "r_net_ok": False,
                "analytic_solver_gamma_ok": False,
                "pass": False,
                "reason": "snes_failed_or_no_lambda1",
            }

        wall = time.time() - t0
        rung_record = {
            "rung_index": i,
            "C_S_F_m2": float(cs_val),
            "k_hyd": k_hyd,
            "voltage_V_RHE": v_kin,
            "lambda_final": LAMBDA_FINAL,
            "analytic_gamma_clamped": (
                float(analytic_clamped) if analytic_clamped == analytic_clamped
                else None
            ),
            "analytic_gamma_unclamped": (
                float(analytic_unclamped) if analytic_unclamped == analytic_unclamped
                else None
            ),
            "denominator_decomposition": denom_decomp,
            "hard_gates": gates,
            "ladder_converged": rec.get("ladder_converged"),
            "exception_phase": rec.get("exception_phase"),
            "exception_str": rec.get("exception"),
            "lambda1": lam1,
            "wall_seconds": wall,
        }
        rung_records.append(rung_record)
        print(
            f"  C_S={cs_val:.3f}: pass={gates.get('pass')}, "
            f"cd={lam1.get('cd_mA_cm2') if lam1 else None}, "
            f"theta={lam1.get('theta') if lam1 else None}, "
            f"R_net={lam1.get('R_net') if lam1 else None}, "
            f"analytic_rel={gates.get('analytic_solver_gamma_rel')}, "
            f"wall={wall:.1f}s",
            flush=True,
        )

    # Roll up.
    rungs_passing = sum(1 for r in rung_records if r["hard_gates"]["pass"])
    total = len(rung_records)
    all_pass = bool(rungs_passing == total)

    config: Dict[str, Any] = {
        "v_kin": v_kin,
        "v_anchor": v_anchor,
        "k0_r4e_factor": k0_r4e_factor,
        "k_hyd": k_hyd,
        "cs_bracket": list(cs_bracket),
        "lambda_final": LAMBDA_FINAL,
        "stern_f_m2_anchor": STERN_F_M2_ANCHOR,
        "K0_HAT_R2E_baseline": float(K0_HAT_R2E),
        "K0_HAT_R4E_baseline": float(K0_HAT_R4E),
        "K0_HAT_R4E_effective": float(K0_HAT_R4E) * k0_r4e_factor,
        "lambda_ladder": list(LAMBDA_LADDER),
        "v10b_kinetics": V10B_KINETICS,
        "v10b_calibration_metadata": V10B_CALIBRATION_METADATA,
        "warm_walk_grid": list(warm_grid),
        "l_eff_m": L_EFF_M_BASELINE,
        "domain_height_hat": domain_height_hat,
        "electrode_area_nondim": electrode_area_nondim,
        "electrode_marker": electrode_marker,
        "mesh": {
            "Nx": MESH_NX, "Ny": MESH_NY, "beta": MESH_BETA,
            "domain_height_hat": domain_height_hat,
        },
        "mesh_dof_count": mesh_dof_count,
        "i_lim_4e_mA_cm2": i_lim_4e,
        "wall_seconds": float(time.time() - t_start),
    }
    payload: Dict[str, Any] = {
        "config": config,
        "rungs": rung_records,
        "summary": {
            "rungs_passing": int(rungs_passing),
            "rungs_total": int(total),
            "all_pass": bool(all_pass),
            "decision": (
                "v10b_cs_bracket_PASS" if all_pass
                else "v10b_cs_bracket_FAIL_escalate_to_v10c"
            ),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "cs_bracket.json")
    with open(json_path, "w") as f:
        json.dump(_serialize(payload), f, indent=2, sort_keys=False)
    print(f"Wrote {json_path}", flush=True)

    if args.plot:
        _make_plot(out_dir=out_dir, payload=payload)

    print(
        f"\n=== v10b C_S bracket summary ===\n"
        f"  rungs_passing: {rungs_passing}/{total}\n"
        f"  decision: {payload['summary']['decision']}\n"
        f"  wall: {time.time() - t_start:.1f}s",
        flush=True,
    )
    return 0 if all_pass else 1


def _make_plot(*, out_dir: str, payload: Dict[str, Any]) -> None:
    """4-panel diagnostic plot of the C_S bracket sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                                       # pragma: no cover
        print("matplotlib not available; skipping plot", flush=True)
        return

    rungs = payload["rungs"]
    cs_vals = [float(r["C_S_F_m2"]) for r in rungs]
    cds = [
        r["lambda1"].get("cd_mA_cm2") if r["lambda1"] else None
        for r in rungs
    ]
    sigmas = [
        r["lambda1"].get("sigma_S_C_per_m2") if r["lambda1"] else None
        for r in rungs
    ]
    r_nets = [
        r["lambda1"].get("R_net") if r["lambda1"] else None
        for r in rungs
    ]
    rels = [r["hard_gates"].get("analytic_solver_gamma_rel") for r in rungs]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.plot(cs_vals, cds, "o-")
    ax.set_xlabel("C_S [F/m^2]")
    ax.set_ylabel("cd_mA_cm2")
    ax.set_title("(A) Current density")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cs_vals, sigmas, "s-")
    ax.set_xlabel("C_S [F/m^2]")
    ax.set_ylabel("sigma_S [C/m^2]")
    ax.set_title("(B) Stern surface charge")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(cs_vals, r_nets, "^-")
    ax.set_xlabel("C_S [F/m^2]")
    ax.set_ylabel("R_net [nondim]")
    ax.set_title("(C) R_net (k_des * Gamma)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(
        cs_vals,
        [r if r is not None and r > 0 else 1e-16 for r in rels],
        "d-",
    )
    ax.axhline(
        payload["rungs"][0]["hard_gates"]["gamma_max_rel_threshold"],
        color="r", linestyle="--", label="HARD gate"
    )
    ax.set_xlabel("C_S [F/m^2]")
    ax.set_ylabel("|Gamma_solver - Gamma_analytic| / max(...)")
    ax.set_title("(D) Analytic-vs-solver Gamma rel residual")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png = os.path.join(out_dir, "cs_bracket.png")
    plt.savefig(png, dpi=120)
    plt.close(fig)
    print(f"Wrote {png}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
