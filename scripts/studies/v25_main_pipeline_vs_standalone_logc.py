"""V25 — Verify that the patched main pipeline reproduces the standalone log-c stack.

After PNP Inverse Solver Revised.tex (Apr 27 writeup), the production
forward solver is:

    3 species + analytic Boltzmann counterion  (Change 1)
    log-concentration primary variable          (Change 2)
    log-rate Butler-Volmer                      (Change 3)

Until now this stack lived only in the standalone scripts (notably
``scripts/studies/v18_logc_lsq_inverse.py`` and
``scripts/studies/v24_3sp_logc_vs_4sp_validation.py``), which build the
log-c context themselves and inject the Boltzmann residual via an inline
``add_boltzmann()`` helper.

The Forward/bv_solver package now exposes a formulation dispatcher that
routes ``build_context`` / ``build_forms`` / ``set_initial_conditions``
to ``forms.py`` or ``forms_logc.py`` based on
``params['bv_convergence']['formulation']``.  Boltzmann counterions are
configured through ``bv_bc.boltzmann_counterions`` and are added by the
forms modules automatically (no inline helper needed).

This script verifies the patched ``solve_grid_with_charge_continuation``
produces observables identical (within numerical tolerance) to the
standalone v18-style cold/warm pipeline at a shared voltage grid and
TRUE kinetic parameters.

Run::

    ../venv-firedrake/bin/python scripts/studies/v25_main_pipeline_vs_standalone_logc.py

Output::

    StudyResults/v25_main_pipeline_vs_standalone_logc/
        comparison.csv       — per-voltage CD/PC for both pipelines
        comparison.json      — same data + verdict
        summary.md           — human-readable summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v-grid", nargs="+", type=float,
        default=[-0.30, -0.20, -0.10, 0.00, 0.05, 0.10],
        help="V_RHE points (V).  Default spans a voltage window where both "
             "pipelines converge cleanly.",
    )
    p.add_argument("--mesh-ny", type=int, default=200)
    p.add_argument(
        "--rel-tol-pct", type=float, default=0.5,
        help="Relative-error pass threshold as %% of max|observable|. "
             "0.5%% is well inside the F2 diffusion-limit tolerance "
             "and tighter than the v24 5%% target since both pipelines "
             "now share the same form code.",
    )
    p.add_argument("--out-subdir", type=str,
                   default="v25_main_pipeline_vs_standalone_logc")
    return p.parse_args()


def main():
    args = _parse_args()

    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        make_bv_solver_params,
        SNES_OPTS_CHARGED,
        _make_nondim_cfg, _make_bv_convergence_cfg,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.params import SolverParams

    V_GRID = np.array(args.v_grid, dtype=float)
    NV = len(V_GRID)

    OUT_DIR = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("V25: patched main pipeline vs standalone log-c (parity check)")
    print("=" * 72)
    print(f"  V_GRID:     {V_GRID.tolist()}")
    print(f"  mesh Ny:    {args.mesh_ny}")
    print(f"  K0_HAT_R1:  {K0_HAT_R1:.6e}")
    print(f"  K0_HAT_R2:  {K0_HAT_R2:.6e}")
    print(f"  ALPHA R1/2: {ALPHA_R1} / {ALPHA_R2}")
    print(f"  E_eq R1/2:  {E_EQ_R1} / {E_EQ_R2} V")
    print(f"  Output:     {OUT_DIR}")
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(args.mesh_ny), beta=3.0)

    # -------------------------------------------------------------
    # Pipeline A: patched main pipeline via
    #             solve_grid_per_voltage_cold_with_warm_fallback (C+D)
    #             with formulation=logc + bv_log_rate=True + Boltzmann counterion
    # -------------------------------------------------------------
    print("--- Pipeline A: patched main pipeline "
          "(dispatcher → forms_logc, C+D orchestrator) ---")

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    sp_main = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    cd_main = np.full(NV, np.nan)
    pc_main = np.full(NV, np.nan)
    z_main = np.full(NV, np.nan)
    method_main = ["MISSING"] * NV

    def _extract_main(orig_idx, _phi, ctx):
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_main[orig_idx] = float(fd.assemble(form_cd))
        pc_main[orig_idx] = float(fd.assemble(form_pc))

    phi_hat_grid = V_GRID / V_T
    t_main = time.time()
    with adj.stop_annotating():
        res_main = solve_grid_per_voltage_cold_with_warm_fallback(
            sp_main,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=4,
            bisect_depth_warm=3,
            per_point_callback=_extract_main,
        )
    t_main = time.time() - t_main
    for idx, point in res_main.points.items():
        z_main[idx] = point.achieved_z_factor
        method_main[idx] = point.method
    n_main_ok = int(np.sum(~np.isnan(cd_main)))
    print(f"  patched-main converged at {n_main_ok}/{NV} points ({t_main:.1f}s)")
    for i, V in enumerate(V_GRID):
        print(f"    V={V:+.3f}: cd={cd_main[i]:+.6e}  pc={pc_main[i]:+.6e}  "
              f"z={z_main[i]:.3f}  method={method_main[i]}")

    # -------------------------------------------------------------
    # Pipeline B: standalone v18-style cold/warm log-c stack
    #             (lifted verbatim from v24 to keep it 1:1 with production).
    # -------------------------------------------------------------
    print()
    print("--- Pipeline B: standalone v18-style cold/warm log-c stack ---")

    H2O2_SEED = 1e-4
    THREE_SPECIES_Z = [0, 0, 1]
    THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
    THREE_SPECIES_C0 = [C_O2_HAT, H2O2_SEED, C_HP_HAT]

    SP_DICT_3SP = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    ss_rel_tol = 1e-4
    ss_abs_tol = 1e-8
    ss_consec = 4

    def make_3sp_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = _make_bv_convergence_cfg(log_rate=True)
        params["nondim"] = _make_nondim_cfg()
        reaction_1 = {
            "k0": k0_r1, "alpha": alpha_r1,
            "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
            "reversible": True, "E_eq_v": E_EQ_R1,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
        }
        reaction_2 = {
            "k0": k0_r2, "alpha": alpha_r2,
            "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
            "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
            "reversible": False, "E_eq_v": E_EQ_R2,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
        }
        params["bv_bc"] = {
            "reactions": [reaction_1, reaction_2],
            "k0": [k0_r1] * 3, "alpha": [alpha_r1] * 3,
            "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
            "E_eq_v": 0.0,
            "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
        }
        return SolverParams.from_list([
            3, 1, 0.25, 80.0, THREE_SPECIES_Z, THREE_SPECIES_D, THREE_SPECIES_A,
            eta_hat, THREE_SPECIES_C0, 0.0, params,
        ])

    def add_boltzmann(ctx):
        scaling = ctx["nondim"]
        W = ctx["W"]; U = ctx["U"]; mesh_ = ctx["mesh"]
        phi = fd.split(U)[-1]
        w = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=mesh_)
        charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)),
                              fd.Constant(50.0))
        ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
        ctx["J_form"] = fd.derivative(ctx["F_res"], U)
        return ctx

    def _snapshot(U): return tuple(d.data_ro.copy() for d in U.dat)
    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat):
            dst.data[:] = src
        Up.assign(U)

    def build_solve(V_RHE):
        sp = make_3sp_sp(V_RHE / V_T, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)  # standalone helper, NOT the new config-driven path
        set_initial_conditions_logc(ctx, list(sp))
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT_3SP)
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        z_nominal = [float(sp[4][i]) for i in range(ctx["n_species"])]
        return ctx, sol, of_cd, of_pc, z_nominal

    def make_run_ss(ctx, sol, of_cd):
        U = ctx["U"]; Up = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_init = 0.25

        def run_ss(max_steps):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for s in range(1, max_steps + 1):
                try:
                    sol.solve()
                except Exception:
                    return False
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
                if sc >= ss_consec:
                    return True
            return False
        return run_ss

    def solve_cold_3sp(V_RHE, max_z_steps=20):
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_RHE)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            for zci in zc:
                zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(200):
                return None, None, 0.0
            achieved_z = 0.0
            for z_val in np.linspace(0, 1, max_z_steps + 1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                if run_ss(120):
                    achieved_z = z_val
                else:
                    _restore(ckpt, U, Up)
                    break
            if achieved_z < 1.0 - 1e-3:
                return None, None, achieved_z
        return (float(fd.assemble(of_cd)),
                float(fd.assemble(of_pc)),
                achieved_z)

    cd_stand = np.full(NV, np.nan)
    pc_stand = np.full(NV, np.nan)
    z_stand = np.full(NV, np.nan)
    t_stand = time.time()
    print("  cold-start z-ramp at each voltage...")
    for i, V in enumerate(V_GRID):
        t_v = time.time()
        cd_v, pc_v, z_a = solve_cold_3sp(float(V))
        z_stand[i] = z_a
        if cd_v is not None:
            cd_stand[i] = cd_v
            pc_stand[i] = pc_v
            print(f"    V={V:+.3f}: cd={cd_v:+.6e}  pc={pc_v:+.6e}  "
                  f"z={z_a:.3f}  ({time.time()-t_v:.1f}s)")
        else:
            print(f"    V={V:+.3f}: cold FAILED (z={z_a:.3f})  "
                  f"({time.time()-t_v:.1f}s)")
    t_stand = time.time() - t_stand
    n_stand_ok = int(np.sum(~np.isnan(cd_stand)))
    print(f"  standalone converged at {n_stand_ok}/{NV} points ({t_stand:.1f}s)")

    # -------------------------------------------------------------
    # Compare
    # -------------------------------------------------------------
    cd_max_main = float(np.nanmax(np.abs(cd_main))) if n_main_ok else float("nan")
    pc_max_main = float(np.nanmax(np.abs(pc_main))) if n_main_ok else float("nan")
    rel_tol = args.rel_tol_pct / 100.0

    rows = []
    for i, V in enumerate(V_GRID):
        if np.isnan(cd_main[i]) or np.isnan(cd_stand[i]):
            rows.append({
                "V_RHE": float(V),
                "cd_main": (None if np.isnan(cd_main[i]) else float(cd_main[i])),
                "cd_stand": (None if np.isnan(cd_stand[i]) else float(cd_stand[i])),
                "pc_main": (None if np.isnan(pc_main[i]) else float(pc_main[i])),
                "pc_stand": (None if np.isnan(pc_stand[i]) else float(pc_stand[i])),
                "abs_dcd": None, "abs_dpc": None,
                "cd_err_pct_of_max": None, "pc_err_pct_of_max": None,
                "verdict": "MISSING",
            })
            continue
        d_cd = float(cd_main[i] - cd_stand[i])
        d_pc = float(pc_main[i] - pc_stand[i])
        cd_err_pct = 100.0 * abs(d_cd) / max(cd_max_main, 1e-30)
        pc_err_pct = 100.0 * abs(d_pc) / max(pc_max_main, 1e-30)
        cd_pass = cd_err_pct <= args.rel_tol_pct
        pc_pass = pc_err_pct <= args.rel_tol_pct
        verdict = "PASS" if (cd_pass and pc_pass) else "FLAG"
        rows.append({
            "V_RHE": float(V),
            "cd_main": float(cd_main[i]),
            "cd_stand": float(cd_stand[i]),
            "pc_main": float(pc_main[i]),
            "pc_stand": float(pc_stand[i]),
            "abs_dcd": abs(d_cd), "abs_dpc": abs(d_pc),
            "cd_err_pct_of_max": cd_err_pct,
            "pc_err_pct_of_max": pc_err_pct,
            "verdict": verdict,
        })

    overlap = [r for r in rows if r["verdict"] != "MISSING"]
    cd_errs = [r["cd_err_pct_of_max"] for r in overlap]
    pc_errs = [r["pc_err_pct_of_max"] for r in overlap]
    summary = {
        "n_overlap": len(overlap),
        "cd_max_err_pct": (max(cd_errs) if cd_errs else None),
        "cd_mean_err_pct": (float(np.mean(cd_errs)) if cd_errs else None),
        "pc_max_err_pct": (max(pc_errs) if pc_errs else None),
        "pc_mean_err_pct": (float(np.mean(pc_errs)) if pc_errs else None),
        "n_pass": sum(1 for r in overlap if r["verdict"] == "PASS"),
        "n_flag": sum(1 for r in overlap if r["verdict"] == "FLAG"),
        "rel_tol_pct": args.rel_tol_pct,
    }

    # -------------------------------------------------------------
    # Persist
    # -------------------------------------------------------------
    raw = {
        "config": {
            "v_grid": V_GRID.tolist(),
            "mesh_ny": args.mesh_ny,
            "rel_tol_pct": args.rel_tol_pct,
            "K0_HAT_R1": K0_HAT_R1,
            "K0_HAT_R2": K0_HAT_R2,
            "ALPHA_R1": ALPHA_R1,
            "ALPHA_R2": ALPHA_R2,
            "E_EQ_R1": E_EQ_R1,
            "E_EQ_R2": E_EQ_R2,
            "I_SCALE": I_SCALE,
            "V_T": V_T,
        },
        "rows": rows,
        "summary": summary,
        "wall_seconds": {"main": t_main, "standalone": t_stand},
    }
    with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
        json.dump(raw, f, indent=2)
    with open(os.path.join(OUT_DIR, "comparison.csv"), "w") as f:
        f.write("V_RHE,cd_main,cd_stand,pc_main,pc_stand,"
                "abs_dcd,abs_dpc,cd_err_pct_of_max,pc_err_pct_of_max,verdict\n")
        for r in rows:
            def _fmt(v):
                return "" if v is None else f"{v:.8e}"
            f.write(",".join([
                f"{r['V_RHE']:.4f}",
                _fmt(r["cd_main"]), _fmt(r["cd_stand"]),
                _fmt(r["pc_main"]), _fmt(r["pc_stand"]),
                _fmt(r["abs_dcd"]), _fmt(r["abs_dpc"]),
                _fmt(r["cd_err_pct_of_max"]),
                _fmt(r["pc_err_pct_of_max"]),
                r["verdict"],
            ]) + "\n")

    md = []
    md.append("# V25 — main-pipeline log-c parity vs standalone log-c\n")
    md.append("Verifies that the patched ``Forward.bv_solver`` dispatcher "
              "(formulation=logc + bv_log_rate=True + bv_bc.boltzmann_counterions) "
              "reproduces the production standalone path used in "
              "``v18_logc_lsq_inverse.py`` / ``v24_3sp_logc_vs_4sp_validation.py`` "
              "to numerical precision.\n")
    md.append("## Setup\n")
    md.append(f"- V_GRID = {V_GRID.tolist()}")
    md.append(f"- mesh: graded rectangle Nx=8, Ny={args.mesh_ny}, beta=3.0")
    md.append(f"- TRUE: K0_HAT_R1={K0_HAT_R1:.6e}, K0_HAT_R2={K0_HAT_R2:.6e}, "
              f"α1={ALPHA_R1}, α2={ALPHA_R2}")
    md.append(f"- E_eq R1/R2 = {E_EQ_R1} / {E_EQ_R2} V (RHE)\n")
    md.append("## Per-voltage comparison\n")
    md.append("| V_RHE (V) | cd_main | cd_stand | |Δcd|/cd_max% | "
              "pc_main | pc_stand | |Δpc|/pc_max% | verdict |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r["verdict"] == "MISSING":
            md.append(
                f"| {r['V_RHE']:+.3f} | "
                f"{r['cd_main']!r} | {r['cd_stand']!r} | — | "
                f"{r['pc_main']!r} | {r['pc_stand']!r} | — | MISSING |"
            )
        else:
            md.append(
                f"| {r['V_RHE']:+.3f} | "
                f"{r['cd_main']:+.4e} | {r['cd_stand']:+.4e} | "
                f"{r['cd_err_pct_of_max']:.4f} | "
                f"{r['pc_main']:+.4e} | {r['pc_stand']:+.4e} | "
                f"{r['pc_err_pct_of_max']:.4f} | {r['verdict']} |"
            )
    md.append("")
    md.append("## Aggregate\n")
    md.append(f"- overlap voltages: {summary['n_overlap']} / {NV}")
    if summary["n_overlap"]:
        md.append(f"- max |Δcd|/cd_max: {summary['cd_max_err_pct']:.4f}%")
        md.append(f"- mean |Δcd|/cd_max: {summary['cd_mean_err_pct']:.4f}%")
        md.append(f"- max |Δpc|/pc_max: {summary['pc_max_err_pct']:.4f}%")
        md.append(f"- mean |Δpc|/pc_max: {summary['pc_mean_err_pct']:.4f}%")
        md.append(f"- per-voltage verdicts: PASS={summary['n_pass']}, "
                  f"FLAG={summary['n_flag']}")
    md.append(f"- wall: main={t_main:.1f}s, standalone={t_stand:.1f}s")

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if summary["n_overlap"]:
        print(f"  overlap points: {summary['n_overlap']}/{NV}")
        print(f"  CD max err %: {summary['cd_max_err_pct']:.4f}")
        print(f"  PC max err %: {summary['pc_max_err_pct']:.4f}")
        print(f"  PASS: {summary['n_pass']}  FLAG: {summary['n_flag']}")
    else:
        print("  No overlap points.")
    print(f"  Saved: {OUT_DIR}/{{summary.md, comparison.csv, comparison.json}}")


if __name__ == "__main__":
    main()
