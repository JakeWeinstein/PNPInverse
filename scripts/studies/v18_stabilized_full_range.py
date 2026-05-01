"""V18: Full-range test of artificial diffusion stabilized solver.

The d_art_scale=0.01 config achieved z=1 at V_RHE up to 0.30V.
Now test the full range -0.5V to +0.7V and compare with standard solver.

Critical question: how much does the artificial diffusion change the
observable currents compared to the "true" z=1 solution?
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

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_stabilized_full")
os.makedirs(OUT_DIR, exist_ok=True)


def solve_stabilized_curve(V_points, d_art_scale=0.01, z_steps=20):
    """Solve I-V curve using stabilized solver for each voltage point."""
    results = []

    for V_RHE in V_points:
        phi_hat = V_RHE / V_T
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
        W = ctx["W"]

        dx = fd.Measure("dx", domain=mesh)
        ci = fd.split(U)[:-1]
        phi = fd.split(U)[-1]
        v_tests = fd.TestFunctions(W)

        F_res = ctx["F_res"]
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        z_vals = list(sp[4])
        D_vals = scaling["D_model_vals"]

        # Add artificial diffusion for charged species only
        h = fd.CellSize(mesh)
        for i in range(n):
            z_i = float(z_vals[i])
            if abs(z_i) > 0:
                D_i = float(D_vals[i])
                drift_speed = fd.Constant(abs(z_i) * D_i * em)
                D_art = fd.Constant(d_art_scale) * h * drift_speed * fd.sqrt(
                    fd.dot(fd.grad(phi), fd.grad(phi)) + fd.Constant(1e-10))
                F_res += D_art * fd.dot(fd.grad(ci[i]), fd.grad(v_tests[i])) * dx

        J_stab = fd.derivative(F_res, U)

        sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

        prob = fd.NonlinearVariationalProblem(F_res, U, bcs=ctx["bcs"], J=J_stab)
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

        ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

        dt_init = 0.25

        def run_ss(max_steps=60):
            dt_val = dt_init
            dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; steady_count = 0
            for s in range(1, max_steps + 1):
                try:
                    sol.solve()
                except Exception as e:
                    return False, s - 1, str(e)
                Up.assign(U)
                fv = float(fd.assemble(ocd))
                if prev_flux is not None:
                    delta = abs(fv - prev_flux)
                    scale = max(abs(fv), abs(prev_flux), 1e-8)
                    if delta / scale <= 1e-4 or delta <= 1e-8:
                        steady_count += 1
                    else:
                        steady_count = 0
                    if prev_delta and delta > 0:
                        r = prev_delta / delta
                        if r > 1:
                            dt_val = min(dt_val * min(r, 4.0), dt_init * 20)
                        else:
                            dt_val = max(dt_val * 0.5, dt_init)
                        dt_const.assign(dt_val)
                    prev_delta = delta
                prev_flux = fv
                if steady_count >= 4:
                    return True, s, "converged"
            return False, max_steps, "budget"

        # Phase 1: z=0
        for zci in zc:
            zci.assign(0.0)
        paf.assign(phi_hat)

        t0 = time.time()
        with adj.stop_annotating():
            ok_z0, _, _ = run_ss(100)
        cd_z0 = float(fd.assemble(ocd))

        # Phase 2: z-ramp
        z_nominal = [float(sp[4][i]) for i in range(n)]
        z_schedule = np.linspace(0, 1, z_steps + 1)[1:]
        achieved_z = 0.0

        with adj.stop_annotating():
            for z_val in z_schedule:
                U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                ok_z, steps_z, msg_z = run_ss(60)
                if ok_z or steps_z > 0:
                    achieved_z = z_val
                else:
                    for src, dst in zip(U_ckpt, U.dat):
                        dst.data[:] = src
                    Up.assign(U)
                    # Fine steps
                    z_fine = np.linspace(achieved_z, z_val, 6)[1:]
                    for z_f in z_fine:
                        U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                        for i in range(n):
                            zc[i].assign(z_nominal[i] * z_f)
                        ok_f, steps_f, _ = run_ss(60)
                        if ok_f or steps_f > 0:
                            achieved_z = z_f
                        else:
                            for src, dst in zip(U_ckpt2, U.dat):
                                dst.data[:] = src
                            Up.assign(U)
                            break
                    break

        cd_final = float(fd.assemble(ocd))
        pc_final = float(fd.assemble(opc))
        c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
        elapsed = time.time() - t0

        entry = {
            "V_RHE": float(V_RHE), "phi_hat": float(phi_hat),
            "cd_z0": float(cd_z0), "cd_z1": float(cd_final),
            "pc_z1": float(pc_final),
            "z_achieved": float(achieved_z),
            "c_min": c_min, "time": elapsed,
        }
        results.append(entry)
        status = "FULL" if achieved_z >= 0.999 else f"z={achieved_z:.2f}"
        print(f"  V={V_RHE:6.3f}V: cd={cd_final:10.6f}, pc={pc_final:10.6f}, "
              f"z={achieved_z:.3f} [{status}], c_min_ClO4={c_min[3]:.2e}, {elapsed:.1f}s")

    return results


def main():
    # Full voltage range
    V_cathodic = np.arange(-0.50, 0.05, 0.05)   # cathodic region
    V_onset = np.arange(0.05, 0.75, 0.025)       # onset region (finer)
    V_all = np.concatenate([V_cathodic, V_onset])
    V_all = np.sort(np.unique(np.round(V_all, 4)))

    print(f"Testing {len(V_all)} voltage points from {V_all[0]:.3f}V to {V_all[-1]:.3f}V")
    print("Using d_art_scale=0.01 artificial diffusion stabilization\n")

    results = solve_stabilized_curve(V_all, d_art_scale=0.01)

    # Summary
    print("\n" + "=" * 80)
    print("STABILIZED SOLVER I-V CURVE (d_art=0.01, physical E_eq)")
    print("=" * 80)
    print(f"{'V_RHE':>8} {'cd_z0':>10} {'cd_z1':>10} {'pc_z1':>10} {'z':>6} {'status':>8}")
    print("-" * 60)
    n_full = 0
    for r in results:
        V = r["V_RHE"]
        z = r["z_achieved"]
        status = "FULL" if z >= 0.999 else f"z={z:.2f}"
        if z >= 0.999:
            n_full += 1
        print(f"{V:8.3f} {r['cd_z0']:10.6f} {r['cd_z1']:10.6f} "
              f"{r['pc_z1']:10.6f} {z:6.3f} {status:>8}")

    print(f"\nFull z=1.0: {n_full}/{len(results)} points")

    # Comparison with standard solver (where both converge)
    # Standard solver z=1 boundary was V_RHE=0.10V
    print("\nComparison at V_RHE=0.10V (both solvers converge to z=1):")
    for r in results:
        if abs(r["V_RHE"] - 0.10) < 0.001:
            print(f"  Stabilized: cd={r['cd_z1']:.6f}")
            # Standard solver value from first diagnostic: -0.162887
            print(f"  Standard:   cd=-0.162887")
            diff = abs(r['cd_z1'] - (-0.162887)) / abs(-0.162887) * 100
            print(f"  Difference: {diff:.2f}%")

    with open(os.path.join(OUT_DIR, "stabilized_iv_curve.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR}/stabilized_iv_curve.json")


if __name__ == "__main__":
    main()
