"""V18: Test log-concentration transform convergence across voltage range.

Compares standard forms.py vs forms_logc.py z-ramp convergence.
Goal: extend z=1.0 convergence past V_RHE = +0.10V into the onset region.

Output: StudyResults/v18_logc_test/
"""
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    FOUR_SPECIES_CHARGED, make_bv_solver_params,
    SNES_OPTS_CHARGED, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
)
setup_firedrake_env()

import numpy as np
import time
import json
import firedrake as fd
import pyadjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms_logc import (
    build_context_logc, build_forms_logc, set_initial_conditions_logc,
    build_bv_observable_form_logc,
)

# Physical E_eq
E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_logc_test")
os.makedirs(OUT_DIR, exist_ok=True)


def test_logc_single_point(V_RHE: float, z_steps: int = 20):
    """Test log-c solver z-ramp at a single voltage point."""
    phi_hat = V_RHE / V_T

    sp = make_bv_solver_params(
        eta_hat=phi_hat,
        dt=0.25,
        t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
        E_eq_r1=E_EQ_R1,
        E_eq_r2=E_EQ_R2,
    )

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        set_initial_conditions_logc(ctx, list(sp))

    U = ctx["U"]
    Up = ctx["U_prev"]
    zc = ctx["z_consts"]
    n = ctx["n_species"]
    dt_const = ctx["dt_const"]
    paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=fd.derivative(ctx["F_res"], U))
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = build_bv_observable_form_logc(ctx, mode="current_density", scale=-I_SCALE)

    # Phase 1: z=0
    for zci in zc:
        zci.assign(0.0)
    paf.assign(phi_hat)

    dt_init = 0.25
    dt_val = dt_init

    def run_ss(max_steps=100):
        nonlocal dt_val
        dt_val = dt_init
        dt_const.assign(dt_val)
        prev_flux = None
        prev_delta = None
        steady_count = 0
        for s in range(1, max_steps + 1):
            try:
                sol.solve()
            except Exception as e:
                return False, s - 1, str(e)
            Up.assign(U)
            fv = float(fd.assemble(of))
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
        return False, max_steps, "budget_exhausted"

    with adj.stop_annotating():
        ok_z0, steps_z0, msg_z0 = run_ss(100)

    cd_z0 = float(fd.assemble(of))

    # Extract actual concentrations from log-space
    c_min_z0 = []
    for i in range(n):
        u_arr = U.dat[i].data_ro
        c_arr = np.exp(np.clip(u_arr, -30, 30))
        c_min_z0.append(float(c_arr.min()))

    print(f"\n--- V_RHE={V_RHE:.2f}V  phi_hat={phi_hat:.2f} ---")
    print(f"  Phase 1 (z=0, log-c): ok={ok_z0}, steps={steps_z0}, "
          f"cd={cd_z0:.6f}, c_min={[f'{c:.2e}' for c in c_min_z0]}")

    # Phase 2: z-ramp
    z_nominal = [float(sp[4][i]) for i in range(n)]
    z_schedule = np.linspace(0.0, 1.0, z_steps + 1)[1:]

    achieved_z = 0.0

    with adj.stop_annotating():
        for z_val in z_schedule:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)

            for i in range(n):
                zc[i].assign(z_nominal[i] * z_val)

            ok_z, steps_z, msg_z = run_ss(60)
            cd_z = float(fd.assemble(of))

            # Extract concentrations
            c_min = []
            for i in range(n):
                u_arr = U.dat[i].data_ro
                c_arr = np.exp(np.clip(u_arr, -30, 30))
                c_min.append(float(c_arr.min()))

            # Also check raw u_i values
            u_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
            u_max = [float(U.dat[i].data_ro.max()) for i in range(n)]

            if ok_z or steps_z > 0:
                achieved_z = z_val
                print(f"  z={z_val:.3f}: ok={ok_z}, steps={steps_z}, "
                      f"cd={cd_z:.6f}, c_min={[f'{c:.2e}' for c in c_min]}, "
                      f"u_range=[{min(u_min):.1f}, {max(u_max):.1f}]")
            else:
                # Restore checkpoint
                for src, dst in zip(U_ckpt, U.dat):
                    dst.data[:] = src
                Up.assign(U)
                print(f"  z={z_val:.3f}: FAILED ({msg_z}), "
                      f"u_range=[{min(u_min):.1f}, {max(u_max):.1f}]")

                # Try finer steps
                z_fine = np.linspace(achieved_z, z_val, 6)[1:]
                for z_f in z_fine:
                    U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n):
                        zc[i].assign(z_nominal[i] * z_f)
                    ok_f, steps_f, msg_f = run_ss(60)
                    if ok_f or steps_f > 0:
                        achieved_z = z_f
                        print(f"    z={z_f:.4f}: ok={ok_f}, steps={steps_f}")
                    else:
                        for src, dst in zip(U_ckpt2, U.dat):
                            dst.data[:] = src
                        Up.assign(U)
                        print(f"    z={z_f:.4f}: FAILED")
                        break
                break

    result = {
        "V_RHE": V_RHE, "phi_hat": phi_hat,
        "z0_cd": cd_z0, "z0_ok": ok_z0,
        "achieved_z": achieved_z,
        "solver": "logc",
    }
    print(f"  >> Achieved z={achieved_z:.4f}")
    return result


def main():
    # Focus on the onset region where standard solver fails
    v_points = [-0.3, -0.1, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]

    all_results = []
    for V in v_points:
        t0 = time.time()
        try:
            r = test_logc_single_point(V, z_steps=20)
            r["time"] = time.time() - t0
            all_results.append(r)
        except Exception as e:
            print(f"\n  V_RHE={V:.2f}: EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"V_RHE": V, "achieved_z": 0.0,
                                "error": str(e), "time": time.time() - t0})

    # Summary
    print("\n" + "=" * 70)
    print("LOG-C SOLVER Z-RAMP CONVERGENCE MAP")
    print("=" * 70)
    print(f"{'V_RHE':>8} {'phi_hat':>8} {'z_achieved':>10} {'cd_z0':>10} {'time':>8} {'status':>12}")
    print("-" * 68)
    for r in all_results:
        V = r["V_RHE"]
        phi = V / V_T
        z = r.get("achieved_z", 0)
        cd = r.get("z0_cd", float("nan"))
        t = r.get("time", 0)
        status = "FULL" if z >= 0.999 else f"PARTIAL({z:.2f})" if z > 0 else "FAILED"
        print(f"{V:8.2f} {phi:8.2f} {z:10.4f} {cd:10.4f} {t:8.1f}s {status:>12}")

    # Reference: standard solver results
    print("\nReference: Standard solver z=1.0 boundary was at V_RHE = +0.10V")

    with open(os.path.join(OUT_DIR, "logc_convergence_map.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/logc_convergence_map.json")


if __name__ == "__main__":
    main()
