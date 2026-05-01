"""V19 Stage 3.1 — BV exponent cap continuation diagnostic.

Tests whether warm-starting cap > 50 from a converged cap=50 state can reach
higher caps that cold-start cannot.  Per the v19_bv_clip_audit result, all caps
in {60, 70, 80, 100, None} fail at z=0.000 from cold start.  The handoff's
Stage 3.1 recommends warm-start cap continuation as a cheaper alternative to
the FV/SG prototype:

    cap = 50 → 60 → 70 → 80 → 100 → None
    At each voltage, use the previous cap's solution as the initial condition.

This script:
  1. Cold-solves cap=50 at V=-0.10 (the easiest voltage; biggest c_H2O2_surf
     hence smallest residual jump under cap increase).
  2. For a sequence of cap values (50 → 51 → 52 → 55 → 60 → 70 → 80 → 100 → None),
     warm-starts from the previous cap's converged U and tries to solve.
  3. Reports the highest cap reachable per voltage, and whether the
     cap=51 step (the smallest possible probe of the cliff) succeeds at all.

Both `--log-rate` and standard formulations are supported.

Usage:
  python scripts/studies/v19_bv_cap_continuation.py
  python scripts/studies/v19_bv_cap_continuation.py --log-rate
  python scripts/studies/v19_bv_cap_continuation.py --caps 50 51 52 55 60 70 \\
      --voltage -0.10
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--caps", nargs="+", default=None,
                   help="Cap continuation sequence "
                        "(default: 50 51 52 55 60 70 80 100 none).")
    p.add_argument("--voltages", nargs="+", type=float, default=None,
                   help="Voltages to test (default: -0.10).")
    p.add_argument("--log-rate", action="store_true",
                   help="Enable bv_log_rate=True.")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.params import SolverParams

    H2O2_SEED = 1e-4
    THREE_SPECIES_Z = [0, 0, 1]
    THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
    USE_LOG_RATE = bool(args.log_rate)

    if args.caps is None:
        caps_raw = ["50", "51", "52", "55", "60", "70", "80", "100", "none"]
    else:
        caps_raw = list(args.caps)
    caps: list[Optional[float]] = []
    for c in caps_raw:
        cs = str(c).lower()
        caps.append(None if cs in ("none", "null", "off") else float(c))

    voltages = args.voltages if args.voltages is not None else [-0.10]
    OUT_DIR = args.out_dir or os.path.join(
        _ROOT, "StudyResults",
        "v19_bv_cap_continuation_lograte" if USE_LOG_RATE
        else "v19_bv_cap_continuation",
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    def make_conv_cfg(cap: Optional[float]) -> dict[str, Any]:
        return {
            "clip_exponent": cap is not None,
            "exponent_clip": float(cap) if cap is not None else 50.0,
            "regularize_concentration": True,
            "conc_floor": 1e-12,
            "use_eta_in_bv": True,
            "bv_log_rate": USE_LOG_RATE,
        }

    def make_3sp_sp(eta_hat, k0_1, k0_2, a_1, a_2, cap):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = make_conv_cfg(cap)
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

    def build_solve(V_RHE, k0_1, k0_2, a_1, a_2, cap):
        sp = make_3sp_sp(V_RHE / V_T, k0_1, k0_2, a_1, a_2, cap)
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
        of_r1 = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=1.0)
        of_r2 = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=1, scale=1.0)
        z_nominal = [float(sp[4][i]) for i in range(ctx["n_species"])]
        return ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal

    def make_run_ss(ctx, sol, of_cd, max_steps=200):
        U = ctx["U"]; Up = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_init = 0.25
        def run_ss():
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for _ in range(1, max_steps + 1):
                try:
                    sol.solve()
                except Exception:
                    return False
                Up.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    d = abs(fv - prev_flux)
                    sv = max(abs(fv), abs(prev_flux), 1e-8)
                    if d / sv <= 1e-4 or d <= 1e-8:
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
                    return True
            return False
        return run_ss

    def domain_h2o2(U):
        u = U.sub(1).dat.data_ro
        c = np.exp(np.clip(u, -30.0, 30.0))
        return float(c.min()), float(c.max())

    def cold_solve_cap50(V_RHE, k0_1, k0_2, a_1, a_2):
        """Cold-ramp at cap=50 to get an anchor IC."""
        ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2, 50.0)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        achieved_z = 0.0
        with adj.stop_annotating():
            for zci in zc: zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss():
                return None
            for z_val in np.linspace(0, 1, 21)[1:]:
                ckpt = _snapshot(U)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                if run_ss():
                    achieved_z = float(z_val)
                else:
                    _restore(ckpt, U, Up); break
            if achieved_z < 1.0 - 1e-3:
                return None
        cd = float(fd.assemble(of_cd))
        pc = float(fd.assemble(of_pc))
        r1 = float(fd.assemble(of_r1))
        r2 = float(fd.assemble(of_r2))
        c_min, c_max = domain_h2o2(U)
        return {
            "cd": cd, "pc": pc, "r1": r1, "r2": r2,
            "c_h2o2_min": c_min, "c_h2o2_max": c_max,
            "snap": _snapshot(U),
        }

    def warm_solve(V_RHE, k0_1, k0_2, a_1, a_2, cap, ic_data):
        ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2, cap)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            _restore(ic_data, U, Up)
            for i in range(n):
                zc[i].assign(z_nominal[i])
            paf.assign(V_RHE / V_T)
            if not run_ss():
                return None
        cd = float(fd.assemble(of_cd))
        pc = float(fd.assemble(of_pc))
        r1 = float(fd.assemble(of_r1))
        r2 = float(fd.assemble(of_r2))
        c_min, c_max = domain_h2o2(U)
        return {
            "cd": cd, "pc": pc, "r1": r1, "r2": r2,
            "c_h2o2_min": c_min, "c_h2o2_max": c_max,
            "snap": _snapshot(U),
        }

    print("=" * 72)
    print("V19 Stage 3.1 — BV cap continuation diagnostic")
    print("=" * 72)
    print(f"caps:      {[('none' if c is None else c) for c in caps]}")
    print(f"voltages:  {voltages}")
    print(f"log_rate:  {USE_LOG_RATE}")
    print(f"out_dir:   {OUT_DIR}")
    print()

    all_results = []
    for V in voltages:
        print(f"\n[V={V:+.2f}] Cold-solving cap=50 anchor")
        t0 = time.time()
        anchor = cold_solve_cap50(float(V), K0_HAT_R1, K0_HAT_R2,
                                   ALPHA_R1, ALPHA_R2)
        if anchor is None:
            print(f"  cap=50 cold solve FAILED at V={V:+.2f}; skipping")
            continue
        print(f"  cap=50 anchor: cd={anchor['cd']:+.4e}  "
              f"r2={anchor['r2']:.3e}  H2O2[{anchor['c_h2o2_min']:.2e},"
              f"{anchor['c_h2o2_max']:.2e}]  ({time.time()-t0:.1f}s)")

        # Warm-start each subsequent cap from the previous successful cap
        last_snap = anchor["snap"]
        last_cap = 50.0
        per_cap = []
        per_cap.append({
            "V_RHE": float(V), "cap": "50", "from_cap": "cold",
            "ok": True, "elapsed_s": round(time.time()-t0, 2),
            "cd": anchor["cd"], "pc": anchor["pc"],
            "r1": anchor["r1"], "r2": anchor["r2"],
            "c_h2o2_min": anchor["c_h2o2_min"],
            "c_h2o2_max": anchor["c_h2o2_max"],
        })

        for cap in caps:
            if cap is not None and float(cap) <= 50.0:
                continue  # already done
            cap_label = "none" if cap is None else f"{cap:g}"
            t_c = time.time()
            res = warm_solve(float(V), K0_HAT_R1, K0_HAT_R2,
                             ALPHA_R1, ALPHA_R2, cap, last_snap)
            elapsed = round(time.time() - t_c, 2)
            if res is None:
                print(f"  cap={cap_label} (warm from "
                      f"cap={last_cap if last_cap else 'cold'}): "
                      f"FAILED  ({elapsed}s)")
                per_cap.append({
                    "V_RHE": float(V), "cap": cap_label,
                    "from_cap": str(last_cap) if last_cap else "cold",
                    "ok": False, "elapsed_s": elapsed,
                })
                break
            print(f"  cap={cap_label} (warm from "
                  f"cap={last_cap}): "
                  f"cd={res['cd']:+.4e}  r2={res['r2']:.3e}  "
                  f"H2O2[{res['c_h2o2_min']:.2e},{res['c_h2o2_max']:.2e}]  "
                  f"({elapsed}s)")
            per_cap.append({
                "V_RHE": float(V), "cap": cap_label,
                "from_cap": str(last_cap),
                "ok": True, "elapsed_s": elapsed,
                "cd": res["cd"], "pc": res["pc"],
                "r1": res["r1"], "r2": res["r2"],
                "c_h2o2_min": res["c_h2o2_min"],
                "c_h2o2_max": res["c_h2o2_max"],
            })
            last_snap = res["snap"]
            last_cap = cap

        # Report highest cap reachable
        ok_caps = [pc["cap"] for pc in per_cap if pc["ok"]]
        last_ok = ok_caps[-1] if ok_caps else "none"
        print(f"  highest cap reachable at V={V:+.2f}: {last_ok}")
        all_results.extend(per_cap)

    # Save to CSV
    csv_path = os.path.join(OUT_DIR, "cap_continuation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V_RHE", "cap", "from_cap", "ok", "elapsed_s",
                    "cd", "pc", "r1", "r2", "c_h2o2_min", "c_h2o2_max"])
        for r in all_results:
            w.writerow([r["V_RHE"], r["cap"], r["from_cap"], r["ok"],
                        r["elapsed_s"],
                        r.get("cd", ""), r.get("pc", ""),
                        r.get("r1", ""), r.get("r2", ""),
                        r.get("c_h2o2_min", ""), r.get("c_h2o2_max", "")])

    json_path = os.path.join(OUT_DIR, "cap_continuation.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "caps": caps_raw,
                "voltages": voltages,
                "log_rate": USE_LOG_RATE,
            },
            "results": all_results,
        }, f, indent=2)

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  {'V_RHE':>7}  {'highest_cap_reached':>20}")
    for V in voltages:
        ok_caps = [r["cap"] for r in all_results
                   if r["V_RHE"] == float(V) and r["ok"]]
        last_ok = ok_caps[-1] if ok_caps else "(none)"
        print(f"  {V:>+7.2f}  {last_ok:>20}")
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
