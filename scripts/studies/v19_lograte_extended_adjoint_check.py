"""V19 — adjoint vs FD check at extended V_GRID with log-rate BV.

Per GPT's `PNP Log Rate Next Steps Handoff.md` Task 3.  Verifies that
pyadjoint correctly tapes through the log-rate BV form at the new important
voltages (V=+0.30, +0.40, +0.50, +0.60) where R1 has dropped out and R2
becomes visible / unclips.

Components checked at each voltage:
    dCD/dlog_k0_1, dCD/dlog_k0_2, dCD/dalpha_1, dCD/dalpha_2
    dPC/dlog_k0_1, dPC/dlog_k0_2, dPC/dalpha_1, dPC/dalpha_2

Adjoint gradients are taken w.r.t. the bv_k0 / bv_alpha control functions
(in raw k0 / alpha basis), then converted to log_k0 basis via dCD/dlog_k0
= k0 · dCD/dk0.  FD step is in log_k0 / alpha directly (matches the FIM
audit convention).

Pass criterion: rel_err < 1% for components with magnitude > 1e-6 of the
largest component for that observable; absolute error checked for
near-zero components.

Output:
    StudyResults/v19_lograte_extended_adjoint_check/
        adjoint_vs_fd.csv
        adjoint_vs_fd.json
        summary.md
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def main() -> None:
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.params import SolverParams

    # --- Configuration ---
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--v-test", nargs="+", type=float,
                   default=[0.30, 0.40, 0.50, 0.60])
    p.add_argument("--log-rate", type=int, default=1,
                   help="0 or 1; default 1 (log-rate ON).")
    p.add_argument("--annotate-steps", type=int, default=5,
                   help="Number of annotated SNES iterations after warm-start.")
    p.add_argument("--ss-rel-tol", type=float, default=1e-4,
                   help="Steady-state convergence relative tolerance for "
                        "warm-start and FD solves (default 1e-4).")
    p.add_argument("--ss-abs-tol", type=float, default=1e-8,
                   help="Steady-state convergence absolute tolerance "
                        "(default 1e-8).")
    p.add_argument("--annotate-dt", type=float, default=None,
                   help="If set, override dt_const to this value before the "
                        "annotated SNES iterations. Large values (e.g. 1e10) "
                        "make the solve effectively steady-state.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Max pseudo-time steps in run_ss for warm-start "
                        "and FD perturbed solves.")
    p.add_argument("--h-fd", type=float, default=None,
                   help="FD step in log_k0 (alpha h = 0.1 * this). "
                        "Default 1e-3 / 1e-4.")
    p.add_argument("--u-clamp", type=float, default=30.0,
                   help="Bulk PDE u_clamp (default 30).")
    p.add_argument("--fd-cold-ramp", action="store_true",
                   help="If set, perform a fresh cold-ramp at each FD "
                        "perturbed parameter (no warm-start from TRUE cache).")
    p.add_argument("--out-dir", type=str, default=None)
    cli_args = p.parse_args()

    H2O2_SEED = 1e-4
    THREE_SPECIES_Z = [0, 0, 1]
    THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
    V_TEST = list(cli_args.v_test)
    LOG_RATE = bool(cli_args.log_rate)
    N_ANNOTATE = int(cli_args.annotate_steps)
    SS_REL_TOL = float(cli_args.ss_rel_tol)
    SS_ABS_TOL = float(cli_args.ss_abs_tol)
    ANNOTATE_DT = (float(cli_args.annotate_dt)
                   if cli_args.annotate_dt is not None else None)
    MAX_STEPS = int(cli_args.max_steps)
    PARAMS = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]
    if cli_args.h_fd is not None:
        h_log = float(cli_args.h_fd); h_a = float(cli_args.h_fd) * 0.1
        H_FD = np.array([h_log, h_log, h_a, h_a])
    else:
        H_FD = np.array([1e-3, 1e-3, 1e-4, 1e-4])

    OUT_DIR = cli_args.out_dir or os.path.join(
        _ROOT, "StudyResults", "v19_lograte_extended_adjoint_check")
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    def make_3sp_sp(eta_hat, k0_1, k0_2, a_1, a_2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = {
            "clip_exponent": True, "exponent_clip": 50.0,
            "regularize_concentration": True, "conc_floor": 1e-12,
            "use_eta_in_bv": True, "bv_log_rate": LOG_RATE,
            "u_clamp": float(cli_args.u_clamp),
        }
        params["nondim"] = _make_nondim_cfg()
        reaction_1 = {
            "k0": k0_1, "alpha": a_1,
            "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
            "reversible": True, "E_eq_v": E_EQ_R1,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
        }
        reaction_2 = {
            "k0": k0_2, "alpha": a_2,
            "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
            "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
            "reversible": False, "E_eq_v": E_EQ_R2,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
        }
        params["bv_bc"] = {
            "reactions": [reaction_1, reaction_2],
            "k0": [k0_1] * 3, "alpha": [a_1] * 3,
            "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
            "E_eq_v": 0.0,
            "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
        }
        c0_vec = [float(C_O2_HAT), float(H2O2_SEED), float(C_HP_HAT)]
        return SolverParams.from_list([
            3, 1, 0.25, 80.0, THREE_SPECIES_Z, THREE_SPECIES_D, THREE_SPECIES_A,
            eta_hat, c0_vec, 0.0, params,
        ])

    def add_boltzmann(ctx):
        mesh = ctx["mesh"]; W = ctx["W"]; U = ctx["U"]
        scaling = ctx["nondim"]
        phi = fd.split(U)[-1]
        w = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=mesh)
        charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)),
                              fd.Constant(50.0))
        ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
        ctx["J_form"] = fd.derivative(ctx["F_res"], U)
        return ctx

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    def _snapshot(U): return tuple(d.data_ro.copy() for d in U.dat)
    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat):
            dst.data[:] = src
        Up.assign(U)

    def build_solve(V_RHE, k0_1, k0_2, a_1, a_2):
        sp = make_3sp_sp(V_RHE / V_T, k0_1, k0_2, a_1, a_2)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT)
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        z_nominal = [float(sp[4][i]) for i in range(ctx["n_species"])]
        return ctx, sol, of_cd, of_pc, z_nominal

    def make_run_ss(ctx, sol, of_cd, max_steps=200,
                    ss_rel_tol=SS_REL_TOL, ss_abs_tol=SS_ABS_TOL):
        U = ctx["U"]; Up = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_init = 0.25
        def run_ss(annotate_steps=0):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            success_step = -1
            for s in range(1, max_steps + 1):
                try:
                    sol.solve()
                except Exception:
                    return False, success_step
                Up.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    d = abs(fv - prev_flux)
                    sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
                    if d / sv <= ss_rel_tol or d <= ss_abs_tol:
                        sc += 1
                    else:
                        sc = 0
                    if prev_delta and d > 0:
                        r = prev_delta / d
                        dt_val = (min(dt_val * min(r, 4), dt_init * 20)
                                  if r > 1 else max(dt_val * 0.5, dt_init))
                        dt_const.assign(dt_val)
                    prev_delta = d
                prev_flux = fv
                if sc >= 4:
                    return True, s
            return False, success_step
        return run_ss

    # --- Cold-ramp at TRUE for each test V → IC cache ---
    print("=" * 72)
    print("V19 ADJOINT-vs-FD CHECK at extended V (log-rate ON)")
    print("=" * 72)
    print(f"V_TEST: {V_TEST}")

    true_caches: dict[float, Any] = {}
    print("\nStep 1: cold-ramp at TRUE for each V_TEST")
    for V in V_TEST:
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(
            float(V), K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            for zci in zc: zci.assign(0.0)
            paf.assign(V / V_T)
            ok, _ = run_ss()
            if not ok:
                print(f"  V={V:+.2f}  cold solve FAILED at z=0; skipping")
                continue
            achieved_z = 0.0
            for z_val in np.linspace(0, 1, 21)[1:]:
                ckpt = _snapshot(U)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                ok, _ = run_ss()
                if ok:
                    achieved_z = float(z_val)
                else:
                    _restore(ckpt, U, Up); break
            if achieved_z < 1.0 - 1e-3:
                print(f"  V={V:+.2f}  cold ramp stopped at z={achieved_z:.3f}; skipping")
                continue
        cd_t = float(fd.assemble(of_cd))
        pc_t = float(fd.assemble(of_pc))
        true_caches[float(V)] = {
            "snap": _snapshot(U),
            "cd_target": cd_t, "pc_target": pc_t,
        }
        print(f"  V={V:+.2f}  cd={cd_t:+.4e}  pc={pc_t:+.4e}")

    if not true_caches:
        print("FATAL: no V_TEST converged at TRUE; aborting.")
        return

    # --- For each V, compute adjoint and FD ---
    rows = []  # (V, observable, param, adjoint, FD, rel_err, verdict)
    print()
    print("Step 2: adjoint and FD at TRUE per V")

    for V in V_TEST:
        if float(V) not in true_caches:
            continue
        cache = true_caches[float(V)]

        # --- Adjoint at TRUE: warm-start unannotated, then annotate 5 SNES steps ---
        # (Same pattern as scripts/studies/v18_logc_lsq_inverse.py:solve_warm_annotated.)
        tape = adj.get_working_tape()
        tape.clear_tape()

        ctx, sol, of_cd, of_pc, z_nominal = build_solve(
            float(V), K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd, max_steps=MAX_STEPS)

        # Step (a): warm-start unannotated convergence
        with adj.stop_annotating():
            _restore(cache["snap"], U, Up)
            for i in range(n):
                zc[i].assign(z_nominal[i])
            paf.assign(float(V) / V_T)
            ok, _ = run_ss()
        if not ok:
            print(f"  V={V:+.2f}  warm-start did not converge; skipping")
            continue

        # Step (b): annotated SNES iterations at steady-state
        if ANNOTATE_DT is not None:
            ctx["dt_const"].assign(ANNOTATE_DT)
        adj.continue_annotation()
        try:
            for _ in range(N_ANNOTATE):
                sol.solve()
                Up.assign(U)
        except Exception as e:
            adj.pause_annotation()
            print(f"  V={V:+.2f}  annotated solve raised: {e}")
            continue

        cd_assembled = fd.assemble(of_cd)
        pc_assembled = fd.assemble(of_pc)

        k0_funcs = list(ctx["bv_k0_funcs"])[:2]
        alpha_funcs = list(ctx["bv_alpha_funcs"])[:2]
        controls = [adj.Control(f) for f in k0_funcs + alpha_funcs]

        rf_cd = adj.ReducedFunctional(cd_assembled, controls)
        rf_pc = adj.ReducedFunctional(pc_assembled, controls)

        try:
            grad_cd = rf_cd.derivative()
            grad_pc = rf_pc.derivative()
        except Exception as e:
            adj.pause_annotation()
            print(f"  V={V:+.2f}  derivative raised: {e}")
            continue

        def _val(g):
            if hasattr(g, "dat"):
                return float(g.dat[0].data_ro[0])
            if hasattr(g, "values"):
                return float(g.values()[0])
            return float(g)

        adjoint_cd_raw = [_val(g) for g in grad_cd]   # [dcd/dk0_1, dcd/dk0_2, dcd/da_1, dcd/da_2]
        adjoint_pc_raw = [_val(g) for g in grad_pc]

        # Convert dk0 → dlog_k0 via chain rule: d/dlog_k0 = k0 · d/dk0
        # alpha gradients are unchanged (FD is in alpha directly)
        adjoint_cd = [
            K0_HAT_R1 * adjoint_cd_raw[0],   # dcd/dlog_k0_1
            K0_HAT_R2 * adjoint_cd_raw[1],   # dcd/dlog_k0_2
            adjoint_cd_raw[2],                # dcd/dalpha_1
            adjoint_cd_raw[3],                # dcd/dalpha_2
        ]
        adjoint_pc = [
            K0_HAT_R1 * adjoint_pc_raw[0],
            K0_HAT_R2 * adjoint_pc_raw[1],
            adjoint_pc_raw[2],
            adjoint_pc_raw[3],
        ]

        adj.pause_annotation()
        tape.clear_tape()

        # --- Central FD in (log_k0_1, log_k0_2, alpha_1, alpha_2) basis ---
        theta_true = np.array([np.log(K0_HAT_R1), np.log(K0_HAT_R2),
                               ALPHA_R1, ALPHA_R2])
        fd_cd = np.zeros(4); fd_pc = np.zeros(4)

        def cd_pc_at(theta):
            k0_1 = float(np.exp(theta[0]))
            k0_2 = float(np.exp(theta[1]))
            a_1 = float(theta[2]); a_2 = float(theta[3])
            with adj.stop_annotating():
                ctx, sol, of_cd_local, of_pc_local, z_nom_local = build_solve(
                    float(V), k0_1, k0_2, a_1, a_2)
                U_local = ctx["U"]; Up_local = ctx["U_prev"]
                zc_local = ctx["z_consts"]
                paf_local = ctx["phi_applied_func"]
                run_ss_local = make_run_ss(ctx, sol, of_cd_local, max_steps=MAX_STEPS)
                if cli_args.fd_cold_ramp:
                    # Fresh cold-ramp from default IC (no warm-start)
                    for zci in zc_local: zci.assign(0.0)
                    paf_local.assign(float(V) / V_T)
                    ok_local, _ = run_ss_local()
                    if not ok_local:
                        return float("nan"), float("nan")
                    for z_val in np.linspace(0, 1, 21)[1:]:
                        ckpt = _snapshot(U_local)
                        for i in range(n):
                            zc_local[i].assign(z_nom_local[i] * z_val)
                        ok_local, _ = run_ss_local()
                        if not ok_local:
                            _restore(ckpt, U_local, Up_local); break
                    if not ok_local:
                        return float("nan"), float("nan")
                else:
                    _restore(cache["snap"], U_local, Up_local)
                    for i in range(n):
                        zc_local[i].assign(z_nom_local[i])
                    paf_local.assign(float(V) / V_T)
                    ok_local, _ = run_ss_local()
            if not ok_local:
                return float("nan"), float("nan")
            return (float(fd.assemble(of_cd_local)),
                    float(fd.assemble(of_pc_local)))

        for j in range(4):
            tp = theta_true.copy(); tp[j] += H_FD[j]
            tm = theta_true.copy(); tm[j] -= H_FD[j]
            cd_p, pc_p = cd_pc_at(tp)
            cd_m, pc_m = cd_pc_at(tm)
            print(f"  V={V:+.2f}  param={PARAMS[j]:>10}  "
                  f"cd_TRUE={cache['cd_target']:+.6e}  "
                  f"cd_p={cd_p:+.6e}  cd_m={cd_m:+.6e}  "
                  f"d_p={cd_p-cache['cd_target']:+.4e}  "
                  f"d_m={cd_m-cache['cd_target']:+.4e}")
            if np.isfinite(cd_p) and np.isfinite(cd_m):
                fd_cd[j] = (cd_p - cd_m) / (2 * H_FD[j])
            else:
                fd_cd[j] = float("nan")
            if np.isfinite(pc_p) and np.isfinite(pc_m):
                fd_pc[j] = (pc_p - pc_m) / (2 * H_FD[j])
            else:
                fd_pc[j] = float("nan")

        # --- Compare per (observable, param) ---
        for obs_name, adj_vals, fd_vals in [
            ("cd", adjoint_cd, fd_cd), ("pc", adjoint_pc, fd_pc)
        ]:
            # Use largest-component scaling per observable to set "near-zero" threshold
            ref = max(abs(v) for v in fd_vals if np.isfinite(v))
            for j, p in enumerate(PARAMS):
                a = adj_vals[j]; f = fd_vals[j]
                if not np.isfinite(f):
                    rel_err = float("nan")
                    verdict = "FD-NAN"
                elif abs(f) < 1e-6 * ref:
                    abs_err = abs(a - f)
                    verdict = "PASS-NEAR0" if abs_err < 1e-6 * ref else "FAIL-NEAR0"
                    rel_err = abs_err / max(ref, 1e-30)
                else:
                    rel_err = abs(a - f) / abs(f)
                    verdict = "PASS" if rel_err < 0.01 else "FAIL"
                rows.append({
                    "V_RHE": float(V),
                    "observable": obs_name,
                    "param": p,
                    "adjoint": float(a),
                    "fd": float(f),
                    "rel_err": float(rel_err),
                    "verdict": verdict,
                })

        print(f"  V={V:+.2f}  adjoint+FD done")

    # --- Print summary table ---
    print()
    print("=" * 100)
    print(f"  {'V':>5}  {'obs':>3}  {'param':>10}  {'adjoint':>14}  "
          f"{'FD':>14}  {'rel_err':>10}  {'verdict':>10}")
    print("=" * 100)
    for r in rows:
        print(f"  {r['V_RHE']:+5.2f}  {r['observable']:>3}  {r['param']:>10}  "
              f"{r['adjoint']:>+14.4e}  {r['fd']:>+14.4e}  "
              f"{r['rel_err']:>10.3e}  {r['verdict']:>10}")

    # --- Save outputs ---
    csv_path = os.path.join(OUT_DIR, "adjoint_vs_fd.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V_RHE", "observable", "param", "adjoint", "fd",
                    "rel_err", "verdict"])
        for r in rows:
            w.writerow([r["V_RHE"], r["observable"], r["param"],
                        r["adjoint"], r["fd"], r["rel_err"], r["verdict"]])

    json_path = os.path.join(OUT_DIR, "adjoint_vs_fd.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "V_TEST": V_TEST,
                "PARAMS": PARAMS,
                "FD_steps": H_FD.tolist(),
                "log_rate": LOG_RATE,
                "annotate_steps": N_ANNOTATE,
                "ss_rel_tol": SS_REL_TOL,
                "ss_abs_tol": SS_ABS_TOL,
            },
            "results": rows,
        }, f, indent=2)

    n_fail = sum(1 for r in rows if r["verdict"].startswith("FAIL"))
    n_pass = sum(1 for r in rows if r["verdict"].startswith("PASS"))
    n_total = len(rows)

    summary_path = os.path.join(OUT_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write("# V19 — adjoint vs FD at extended V_GRID with log-rate\n\n")
        f.write(f"V tested: {V_TEST}\n\n")
        f.write(f"Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;\n")
        f.write(f"absolute error < 1e-6 · max|FD| for near-zero components.\n\n")
        f.write(f"**Result: {n_pass}/{n_total} PASS, {n_fail} FAIL**\n\n")
        f.write("## Per-component results\n\n")
        f.write("| V | obs | param | adjoint | FD | rel_err | verdict |\n")
        f.write("|---:|:---:|:---|---:|---:|---:|:---|\n")
        for r in rows:
            f.write(f"| {r['V_RHE']:+.2f} | {r['observable']} | {r['param']} "
                    f"| {r['adjoint']:+.4e} | {r['fd']:+.4e} "
                    f"| {r['rel_err']:.3e} | {r['verdict']} |\n")

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {summary_path}")
    print()
    print(f"OVERALL: {n_pass}/{n_total} PASS, {n_fail} FAIL")


if __name__ == "__main__":
    main()
