"""Phase 6 beta v10b coupled Gamma_max x k_des sensitivity matrix.

Per plan section D7-D4.  Runs the v10b stack at V_kin = -0.10 V across
the 30-rung matrix:

* ``Gamma_max in {V10B/2, V10B, V10B*2}`` (3 values).
* ``k_des in {0.01, 0.1, 1.0, 10.0, 100.0}`` (5 values; spans the
  engineering Eyring prior in [10^-2, 10^2] nondim ~ DeltaG_des in
  [0.69, 0.94] eV).
* ``k_hyd in {1e-3, 1e-1}`` (2 values; baseline + cap-saturated route).

Two-stage anchor at every rung: build anchor at STERN_F_M2_ANCHOR =
0.10 F/m^2, runtime-bump to STERN_F_M2_BASELINE = 0.20 F/m^2 via
set_stern_capacitance_model + Newton resolve.  Per-rung Gamma_max +
k_des overrides go through ``parameter_overrides`` on
``solve_lambda_ramp_from_warm_start`` (handled by
``set_reaction_gamma_max_model`` + ``set_reaction_k_des_model``
dispatch in anchor_continuation.py:1689-1697).

Hard gates (per plan D7-D4; same as D7-D1):

* 30/30 converge.
* ``cd_mA_cm2 < 0`` at V_kin.
* ``R_4e_current_nondim > 0`` where ``|R_4e_current_nondim| > 1e-6``.
* ``R_net >= 0``.
* Per-rung analytic-vs-solver mass-balance: rel <= 5e-3 via
  ``gamma_ss_langmuir`` with per-rung state (NOT baseline state).

Output: ``StudyResults/phase6b_v10b_gamma_kdes_matrix/matrix.{json,png}``.

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
from typing import Any, Dict, List, Optional, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


V_KIN_DEFAULT: float = -0.10
V_ANCHOR_DEFAULT: float = 0.55
K0_R4E_FACTOR_DEFAULT: float = 1e-14
LAMBDA_FINAL: float = 1.0
OUT_SUBDIR_DEFAULT: str = "phase6b_v10b_gamma_kdes_matrix"

K_DES_BRACKET: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)
K_HYD_BRACKET: tuple = (1e-3, 1e-1)
GAMMA_MAX_RATIOS: tuple = (0.5, 1.0, 2.0)
"""Gamma_max bracket is {V10B/2, V10B, V10B*2} via these multipliers."""

MASS_BALANCE_REL_MAX: float = 5e-3
R_4E_SIGN_FLOOR: float = 1e-6


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 beta v10b -- coupled Gamma_max x k_des matrix "
            "(30 rungs) at V_kin with two-stage anchor + per-rung "
            "analytic Gamma HARD gate."
        ),
    )
    parser.add_argument("--v-kin", type=float, default=V_KIN_DEFAULT)
    parser.add_argument("--v-anchor", type=float, default=V_ANCHOR_DEFAULT)
    parser.add_argument(
        "--k0-r4e-factor", type=float, default=K0_R4E_FACTOR_DEFAULT,
    )
    parser.add_argument(
        "--k-des-bracket", default=",".join(str(v) for v in K_DES_BRACKET),
    )
    parser.add_argument(
        "--k-hyd-bracket", default=",".join(str(v) for v in K_HYD_BRACKET),
    )
    parser.add_argument(
        "--gamma-max-ratios",
        default=",".join(str(v) for v in GAMMA_MAX_RATIOS),
        help=(
            "Comma-separated multipliers applied to GAMMA_MAX_HAT_V10B "
            "to form the Gamma_max bracket.  Default 0.5,1.0,2.0 "
            "(= {V10B/2, V10B, V10B*2})."
        ),
    )
    parser.add_argument(
        "--out-subdir", default=OUT_SUBDIR_DEFAULT,
    )
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser.parse_args(argv)


def _parse_positive_csv(s: str, name: str) -> tuple:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[float] = []
    for p in parts:
        val = float(p)
        if val <= 0.0:
            raise ValueError(
                f"--{name} entries must be positive (got {val!r})"
            )
        out.append(val)
    if not out:
        raise ValueError(f"--{name} must contain at least one entry")
    return tuple(out)


def _enumerate_rungs(
    *,
    k_des_bracket: tuple,
    k_hyd_bracket: tuple,
    gamma_max_values: tuple,
) -> List[Tuple[float, float, float]]:
    """Return rung list as (gamma_max, k_des, k_hyd) triples in stable order."""
    out: List[Tuple[float, float, float]] = []
    for g in gamma_max_values:
        for k_d in k_des_bracket:
            for k_h in k_hyd_bracket:
                out.append((float(g), float(k_d), float(k_h)))
    return out


def _evaluate_hard_gates(
    *,
    lam1: Dict[str, Any],
    analytic_gamma_clamped: float,
) -> Dict[str, Any]:
    """Same v10b D7-D4 HARD-gate evaluation as the C_S bracket driver."""
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


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    k_des_bracket = _parse_positive_csv(args.k_des_bracket, "k-des-bracket")
    k_hyd_bracket = _parse_positive_csv(args.k_hyd_bracket, "k-hyd-bracket")
    gamma_max_ratios = _parse_positive_csv(
        args.gamma_max_ratios, "gamma-max-ratios",
    )
    v_kin = float(args.v_kin)
    v_anchor = float(args.v_anchor)
    k0_r4e_factor = float(args.k0_r4e_factor)
    out_dir = os.path.join(_ROOT, "StudyResults", args.out_subdir)

    print(
        f"=== Phase 6 beta v10b coupled Gamma_max x k_des matrix ===",
        flush=True,
    )
    print(
        f"  V_kin={v_kin:+.3f} V, v_anchor={v_anchor:+.3f} V, "
        f"K0_R4e_factor={k0_r4e_factor:.3g}",
        flush=True,
    )
    print(f"  k_des={k_des_bracket}", flush=True)
    print(f"  k_hyd={k_hyd_bracket}", flush=True)
    print(f"  gamma_max_ratios={gamma_max_ratios} (multipliers on V10B)",
          flush=True)
    print(f"  Output: {out_dir}", flush=True)

    # Lazy imports.
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        L_EFF_M_BASELINE, STERN_F_M2_BASELINE, STERN_F_M2_ANCHOR,
        MESH_NX, MESH_NY, MESH_BETA,
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
        GAMMA_MAX_HAT_V10B, V10B_KINETICS, V10B_CALIBRATION_METADATA,
    )

    gamma_max_values = tuple(
        float(r) * float(GAMMA_MAX_HAT_V10B) for r in gamma_max_ratios
    )
    rungs_grid = _enumerate_rungs(
        k_des_bracket=k_des_bracket,
        k_hyd_bracket=k_hyd_bracket,
        gamma_max_values=gamma_max_values,
    )
    print(f"  total rungs: {len(rungs_grid)}", flush=True)

    warm_grid = (0.55, 0.40, 0.20, 0.10, -0.10)
    if not any(abs(v - v_kin) < 1e-9 for v in warm_grid):
        warm_grid = tuple(sorted(list(warm_grid) + [v_kin], reverse=True))

    t_start = time.time()

    # Build template SP at production C_S (two-stage anchor handles
    # 0.10 -> 0.20 bump inside _walk_lambda_zero_capture_snapshots).
    # Use a baseline k_hyd in template SP -- the per-rung override
    # supersedes it through parameter_overrides.
    sp = _build_sp(
        lambda_hydrolysis=0.0,
        k0_r4e_factor=k0_r4e_factor,
        k_hyd_nondim=float(k_hyd_bracket[0]),
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
        f"  warm-walk produced U snapshot at V_kin={v_kin:+.3f} V",
        flush=True,
    )

    domain_height_hat = L_EFF_M_BASELINE / 1.0e-4
    i_lim_4e = _i_lim_4e_mA_cm2(L_EFF_M_BASELINE)

    # Pass 2 -- matrix sweep.
    rung_records: List[Dict[str, Any]] = []
    for i, (gamma_max, k_des, k_hyd) in enumerate(rungs_grid):
        t0 = time.time()
        print(
            f"Pass 2.{i+1}/{len(rungs_grid)}: "
            f"Gamma_max={gamma_max:.4g} (V10B={GAMMA_MAX_HAT_V10B:.4g}), "
            f"k_des={k_des:.3g}, k_hyd={k_hyd:.3g}",
            flush=True,
        )
        # Per-rung parameter overrides for k_hyd, k_des, gamma_max.
        # set_reaction_gamma_max_model + set_reaction_k_des_model are
        # already wired into the _OVERRIDE_DISPATCH dict in
        # anchor_continuation.py:1689-1697 under the keys
        # "k_des" and "gamma_max_nondim".
        rec = _run_k_hyd_ramp(
            sp_template=sp, mesh=mesh, voltage=v_kin,
            U_warmstart=U_at_v_kin, k_hyd_target=k_hyd,
            k0_r4e_factor=k0_r4e_factor,
            i_scale=float(I_SCALE),
            i_lim_4e_mA_cm2=i_lim_4e,
            electrode_area_nondim=electrode_area_nondim,
            domain_height_hat=domain_height_hat,
            extra_overrides={
                "k_des": float(k_des),
                "gamma_max_nondim": float(gamma_max),
            },
        )
        lam1 = lambda1_record(rec)

        if lam1 is not None and lam1.get("snes_converged", False):
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
                lam1=lam1, analytic_gamma_clamped=analytic_clamped,
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
            "gamma_max_nondim": gamma_max,
            "gamma_max_ratio_v10b": gamma_max / float(GAMMA_MAX_HAT_V10B),
            "k_des": k_des,
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
            f"  pass={gates.get('pass')}, "
            f"theta={lam1.get('theta') if lam1 else None}, "
            f"R_net={lam1.get('R_net') if lam1 else None}, "
            f"analytic_rel={gates.get('analytic_solver_gamma_rel')}, "
            f"wall={wall:.1f}s",
            flush=True,
        )

    rungs_passing = sum(1 for r in rung_records if r["hard_gates"]["pass"])
    total = len(rung_records)
    all_pass = bool(rungs_passing == total)

    config: Dict[str, Any] = {
        "v_kin": v_kin,
        "v_anchor": v_anchor,
        "k0_r4e_factor": k0_r4e_factor,
        "k_des_bracket": list(k_des_bracket),
        "k_hyd_bracket": list(k_hyd_bracket),
        "gamma_max_ratios": list(gamma_max_ratios),
        "gamma_max_values": list(gamma_max_values),
        "GAMMA_MAX_HAT_V10B": float(GAMMA_MAX_HAT_V10B),
        "lambda_final": LAMBDA_FINAL,
        "stern_f_m2_anchor": STERN_F_M2_ANCHOR,
        "stern_f_m2_baseline": STERN_F_M2_BASELINE,
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
                "v10b_matrix_PASS" if all_pass
                else "v10b_matrix_FAIL_escalate_to_v10c"
            ),
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "matrix.json")
    with open(json_path, "w") as f:
        json.dump(_serialize(payload), f, indent=2, sort_keys=False)
    print(f"Wrote {json_path}", flush=True)

    if args.plot:
        _make_plot(
            out_dir=out_dir, payload=payload,
            k_hyd_bracket=k_hyd_bracket,
            gamma_max_values=gamma_max_values,
            k_des_bracket=k_des_bracket,
        )

    print(
        f"\n=== v10b Gamma_max x k_des matrix summary ===\n"
        f"  rungs_passing: {rungs_passing}/{total}\n"
        f"  decision: {payload['summary']['decision']}\n"
        f"  wall: {time.time() - t_start:.1f}s",
        flush=True,
    )
    return 0 if all_pass else 1


def _make_plot(
    *, out_dir: str, payload: Dict[str, Any],
    k_hyd_bracket: tuple, gamma_max_values: tuple, k_des_bracket: tuple,
) -> None:
    """2D heatmap of theta + R_net per k_hyd; per-rung mass-balance bar."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:                                       # pragma: no cover
        print("matplotlib/numpy not available; skipping plot", flush=True)
        return

    rungs = payload["rungs"]
    rung_by_key: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    for r in rungs:
        key = (
            float(r["gamma_max_nondim"]),
            float(r["k_des"]),
            float(r["k_hyd"]),
        )
        rung_by_key[key] = r

    fig, axes = plt.subplots(
        2, len(k_hyd_bracket), figsize=(6 * len(k_hyd_bracket), 8),
        squeeze=False,
    )
    for col, k_hyd in enumerate(k_hyd_bracket):
        # theta heatmap: rows = gamma_max, columns = k_des
        theta_arr = np.full(
            (len(gamma_max_values), len(k_des_bracket)), np.nan,
        )
        rnet_arr = np.full_like(theta_arr, np.nan)
        for i, g in enumerate(gamma_max_values):
            for j, k_d in enumerate(k_des_bracket):
                r = rung_by_key.get((g, k_d, k_hyd))
                if r is None or r["lambda1"] is None:
                    continue
                lam1 = r["lambda1"]
                if lam1.get("theta") is not None:
                    theta_arr[i, j] = float(lam1["theta"])
                if lam1.get("R_net") is not None:
                    rnet_arr[i, j] = float(lam1["R_net"])

        ax_t = axes[0, col]
        im = ax_t.imshow(
            theta_arr, aspect="auto", origin="lower",
            cmap="viridis", vmin=0.0, vmax=1.0,
        )
        ax_t.set_xticks(range(len(k_des_bracket)))
        ax_t.set_xticklabels([f"{v:.2g}" for v in k_des_bracket])
        ax_t.set_yticks(range(len(gamma_max_values)))
        ax_t.set_yticklabels([f"{v:.3g}" for v in gamma_max_values])
        ax_t.set_xlabel("k_des [nondim]")
        ax_t.set_ylabel("Gamma_max [nondim]")
        ax_t.set_title(f"(A.{col+1}) theta @ k_hyd={k_hyd:.3g}")
        plt.colorbar(im, ax=ax_t)

        ax_r = axes[1, col]
        im2 = ax_r.imshow(
            rnet_arr, aspect="auto", origin="lower", cmap="plasma",
        )
        ax_r.set_xticks(range(len(k_des_bracket)))
        ax_r.set_xticklabels([f"{v:.2g}" for v in k_des_bracket])
        ax_r.set_yticks(range(len(gamma_max_values)))
        ax_r.set_yticklabels([f"{v:.3g}" for v in gamma_max_values])
        ax_r.set_xlabel("k_des [nondim]")
        ax_r.set_ylabel("Gamma_max [nondim]")
        ax_r.set_title(f"(B.{col+1}) R_net @ k_hyd={k_hyd:.3g}")
        plt.colorbar(im2, ax=ax_r)

    plt.tight_layout()
    png = os.path.join(out_dir, "matrix.png")
    plt.savefig(png, dpi=120)
    plt.close(fig)
    print(f"Wrote {png}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
