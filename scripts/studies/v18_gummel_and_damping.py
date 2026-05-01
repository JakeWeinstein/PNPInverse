"""V18: Gummel iteration + aggressive damping strategies for z=1 convergence.

Strategy 1: Gummel (operator-split) iteration
  - Fix phi, solve NP for concentrations (mild nonlinearity from BV BC)
  - Fix c, solve Poisson for phi (linear)
  - Under-relax phi updates
  - Each subproblem is well-conditioned individually

Strategy 2: Very aggressive Newton damping (maxlambda=0.01)
Strategy 3: Trust-region Newton (newtontr)
Strategy 4: Smaller dt with more steps

Tests at V_RHE = 0.15V, 0.20V, 0.25V where the standard solver fails.
"""
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    FOUR_SPECIES_CHARGED, make_bv_solver_params,
    SNES_OPTS_CHARGED,
)
setup_firedrake_env()

import numpy as np
import time
import json
import firedrake as fd
import pyadjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
from Forward.bv_solver.observables import _build_bv_observable_form

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_solver_strategies")
os.makedirs(OUT_DIR, exist_ok=True)


def get_z1_anchor(V_anchor=0.10):
    """Get z=1 solution at anchor voltage using standard z-ramp."""
    phi_hat = V_anchor / V_T
    sp = make_bv_solver_params(
        eta_hat=phi_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    with adj.stop_annotating():
        ctx = build_context(list(sp), mesh=mesh)
        ctx = build_forms(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    z_nominal = [float(sp[4][i]) for i in range(n)]

    # z=0 sweep
    for zci in zc:
        zci.assign(0.0)
    paf.assign(phi_hat)

    with adj.stop_annotating():
        for _ in range(100):
            sol.solve()
            Up.assign(U)

        for z_val in np.linspace(0, 1, 21)[1:]:
            for i in range(n):
                zc[i].assign(z_nominal[i] * z_val)
            for _ in range(60):
                sol.solve()
                Up.assign(U)

    return ctx, sp, mesh, tuple(d.data_ro.copy() for d in U.dat)


def strategy_aggressive_damping(V_target, ctx, sp, anchor_data):
    """Strategy 2: Very aggressive Newton damping."""
    U = ctx["U"]; Up = ctx["U_prev"]; paf = ctx["phi_applied_func"]
    dt_const = ctx["dt_const"]
    n = ctx["n_species"]

    phi_target = V_target / V_T

    # Restore anchor
    for src, dst in zip(anchor_data, U.dat):
        dst.data[:] = src
    Up.assign(U)

    # Try progressively more aggressive damping
    for maxlam in [0.1, 0.05, 0.02, 0.01, 0.005]:
        # Restore
        for src, dst in zip(anchor_data, U.dat):
            dst.data[:] = src
        Up.assign(U)

        sp_dict = {
            "snes_type": "newtonls",
            "snes_max_it": 500,
            "snes_atol": 1e-7,
            "snes_rtol": 1e-10,
            "snes_stol": 1e-12,
            "snes_linesearch_type": "l2",
            "snes_linesearch_maxlambda": maxlam,
            "snes_divergence_tolerance": 1e12,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77,
            "mat_mumps_icntl_14": 80,
        }

        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

        paf.assign(phi_target)
        dt_const.assign(0.1)  # smaller dt

        success = True
        for step in range(500):
            try:
                sol.solve()
            except Exception as e:
                success = False
                break
            Up.assign(U)

            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            if any(c < -0.1 for c in c_min):
                success = False
                break

        if success:
            ocd = _build_bv_observable_form(ctx, mode="current_density", scale=-I_SCALE)
            cd = float(fd.assemble(ocd))
            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            print(f"    maxlam={maxlam}: SUCCESS after {step+1} steps, "
                  f"cd={cd:.6f}, c_min={[f'{c:.2e}' for c in c_min]}")
            return True, cd
        else:
            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            print(f"    maxlam={maxlam}: FAILED at step {step+1}, "
                  f"c_min={[f'{c:.2e}' for c in c_min]}")

    return False, float("nan")


def strategy_trust_region(V_target, ctx, sp, anchor_data):
    """Strategy 3: Trust-region Newton (newtontr)."""
    U = ctx["U"]; Up = ctx["U_prev"]; paf = ctx["phi_applied_func"]
    dt_const = ctx["dt_const"]
    n = ctx["n_species"]

    phi_target = V_target / V_T

    for src, dst in zip(anchor_data, U.dat):
        dst.data[:] = src
    Up.assign(U)

    sp_dict = {
        "snes_type": "newtontr",
        "snes_max_it": 500,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_trtol": 0.01,  # trust region tolerance
        "snes_divergence_tolerance": 1e12,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77,
        "mat_mumps_icntl_14": 80,
    }

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    paf.assign(phi_target)
    dt_const.assign(0.1)

    for step in range(300):
        try:
            sol.solve()
        except Exception as e:
            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            print(f"    newtontr: FAILED at step {step+1} ({e}), "
                  f"c_min={[f'{c:.2e}' for c in c_min]}")
            return False, float("nan")
        Up.assign(U)

        c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
        if any(c < -0.1 for c in c_min):
            print(f"    newtontr: negative c at step {step+1}, "
                  f"c_min={[f'{c:.2e}' for c in c_min]}")
            return False, float("nan")

    ocd = _build_bv_observable_form(ctx, mode="current_density", scale=-I_SCALE)
    cd = float(fd.assemble(ocd))
    print(f"    newtontr: SUCCESS after {step+1} steps, cd={cd:.6f}")
    return True, cd


def strategy_tiny_voltage_steps(V_target, ctx, sp, anchor_data, V_anchor=0.10):
    """Strategy 4: Very small voltage increments (1mV) from anchor."""
    U = ctx["U"]; Up = ctx["U_prev"]; paf = ctx["phi_applied_func"]
    dt_const = ctx["dt_const"]
    n = ctx["n_species"]

    for src, dst in zip(anchor_data, U.dat):
        dst.data[:] = src
    Up.assign(U)

    sp_dict = {
        "snes_type": "newtonls",
        "snes_max_it": 300,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e12,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77,
        "mat_mumps_icntl_14": 80,
    }

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    # 1mV steps
    dV = 0.001  # 1mV
    V_steps = np.arange(V_anchor + dV, V_target + dV/2, dV)
    dt_const.assign(0.1)

    last_good_V = V_anchor
    for V_step in V_steps:
        phi_step = V_step / V_T
        U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
        paf.assign(phi_step)

        success = True
        for s in range(30):
            try:
                sol.solve()
            except Exception:
                success = False
                break
            Up.assign(U)

        c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
        if success and all(c > -0.01 for c in c_min):
            last_good_V = V_step
        else:
            for src, dst in zip(U_ckpt, U.dat):
                dst.data[:] = src
            Up.assign(U)
            print(f"    1mV steps: failed at V={V_step:.4f}V "
                  f"(c_min={[f'{c:.2e}' for c in c_min]})")
            break

    if last_good_V >= V_target - 0.001:
        ocd = _build_bv_observable_form(ctx, mode="current_density", scale=-I_SCALE)
        cd = float(fd.assemble(ocd))
        print(f"    1mV steps: SUCCESS, reached V={last_good_V:.4f}V, cd={cd:.6f}")
        return True, cd
    else:
        print(f"    1mV steps: boundary at V={last_good_V:.4f}V")
        return False, float("nan")


def strategy_smaller_dt(V_target, ctx, sp, anchor_data, V_anchor=0.10):
    """Strategy 5: Much smaller dt (0.01) with more iterations."""
    U = ctx["U"]; Up = ctx["U_prev"]; paf = ctx["phi_applied_func"]
    dt_const = ctx["dt_const"]
    n = ctx["n_species"]

    for src, dst in zip(anchor_data, U.dat):
        dst.data[:] = src
    Up.assign(U)

    sp_dict = {
        "snes_type": "newtonls",
        "snes_max_it": 300,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e12,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    # Small voltage step to target
    phi_target = V_target / V_T
    paf.assign(phi_target)

    # Very small dt
    dt_val = 0.01
    dt_const.assign(dt_val)

    success = True
    for step in range(2000):
        try:
            sol.solve()
        except Exception as e:
            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            print(f"    small dt: FAILED at step {step+1}, "
                  f"c_min={[f'{c:.2e}' for c in c_min]}")
            success = False
            break
        Up.assign(U)

        if step % 200 == 0:
            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            ocd = _build_bv_observable_form(ctx, mode="current_density", scale=-I_SCALE)
            cd = float(fd.assemble(ocd))
            print(f"    dt=0.01, step {step}: cd={cd:.6f}, "
                  f"c_min={[f'{c:.2e}' for c in c_min]}")

    if success:
        ocd = _build_bv_observable_form(ctx, mode="current_density", scale=-I_SCALE)
        cd = float(fd.assemble(ocd))
        print(f"    small dt: SUCCESS, cd={cd:.6f}")
        return True, cd
    return False, float("nan")


def main():
    print("Getting z=1 anchor at V_RHE=0.10V...")
    t0 = time.time()
    ctx, sp, mesh, anchor_data = get_z1_anchor(V_anchor=0.10)
    print(f"Anchor obtained in {time.time()-t0:.1f}s\n")

    V_TESTS = [0.15, 0.20, 0.25]
    results = {}

    for V in V_TESTS:
        print(f"\n{'='*60}")
        print(f"Testing V_RHE = {V:.2f}V")
        print(f"{'='*60}")

        # Strategy 2: Aggressive damping
        print("\n  Strategy: Aggressive damping")
        t0 = time.time()
        ok, cd = strategy_aggressive_damping(V, ctx, sp, anchor_data)
        results[f"damping_{V}"] = {"ok": ok, "cd": cd, "time": time.time()-t0}

        # Strategy 3: Trust region
        print("\n  Strategy: Trust-region Newton")
        t0 = time.time()
        ok, cd = strategy_trust_region(V, ctx, sp, anchor_data)
        results[f"tr_{V}"] = {"ok": ok, "cd": cd, "time": time.time()-t0}

        # Strategy 4: 1mV steps
        print("\n  Strategy: 1mV voltage steps")
        t0 = time.time()
        ok, cd = strategy_tiny_voltage_steps(V, ctx, sp, anchor_data)
        results[f"1mV_{V}"] = {"ok": ok, "cd": cd, "time": time.time()-t0}

        # Strategy 5: Small dt
        print("\n  Strategy: Small dt (0.01)")
        t0 = time.time()
        ok, cd = strategy_smaller_dt(V, ctx, sp, anchor_data)
        results[f"smalldt_{V}"] = {"ok": ok, "cd": cd, "time": time.time()-t0}

    # Summary
    print("\n" + "=" * 70)
    print("SOLVER STRATEGY COMPARISON")
    print("=" * 70)
    for k, v in sorted(results.items()):
        print(f"  {k:30s}: ok={v['ok']}, cd={v['cd']:.6f}, time={v['time']:.1f}s")

    with open(os.path.join(OUT_DIR, "strategy_comparison.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR}/strategy_comparison.json")


if __name__ == "__main__":
    main()
