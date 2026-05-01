"""V18: Test selective log-c transform (ClO4- only) across voltage range.

The mesh DOES resolve the Debye layer (0.01nm elements vs 30nm Debye).
The issue is CG1 positivity violation for exponentially-depleting ClO4-.
Fix: u_3 = ln(c_3) for ClO4- only. No artificial diffusion.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    FOUR_SPECIES_CHARGED, make_bv_solver_params, SNES_OPTS_CHARGED,
)
setup_firedrake_env()

import numpy as np
import time
import firedrake as fd
import pyadjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms_mixed_logc import (
    build_context_mixed_logc, build_forms_mixed_logc,
    set_initial_conditions_mixed_logc,
)
from Forward.bv_solver.observables import _build_bv_observable_form

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78


def test_point(V_RHE, z_steps=20):
    phi_hat = V_RHE / V_T
    sp = make_bv_solver_params(
        eta_hat=phi_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context_mixed_logc(list(sp), mesh=mesh, log_species={3})
        ctx = build_forms_mixed_logc(ctx, list(sp))
        set_initial_conditions_mixed_logc(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)

    dt_init = 0.25
    def run_ss(max_steps=60):
        dt_val = dt_init; dt_const.assign(dt_val)
        prev_flux = None; prev_delta = None; sc = 0
        for s in range(1, max_steps+1):
            try: sol.solve()
            except Exception as e: return False, s-1, str(e)
            Up.assign(U)
            fv = float(fd.assemble(of))
            if prev_flux is not None:
                d = abs(fv-prev_flux); sv = max(abs(fv),abs(prev_flux),1e-8)
                if d/sv <= 1e-4 or d <= 1e-8: sc += 1
                else: sc = 0
                if prev_delta and d > 0:
                    r = prev_delta/d
                    dt_val = min(dt_val*min(r,4),dt_init*20) if r>1 else max(dt_val*0.5,dt_init)
                    dt_const.assign(dt_val)
                prev_delta = d
            prev_flux = fv
            if sc >= 4: return True, s, "converged"
        return False, max_steps, "budget"

    # z=0
    for zci in zc: zci.assign(0.0)
    paf.assign(phi_hat)

    print(f"\n--- V_RHE={V_RHE:.2f}V (phi_hat={phi_hat:.2f}) ---")
    with adj.stop_annotating():
        ok_z0, steps_z0, msg_z0 = run_ss(100)

    cd_z0 = float(fd.assemble(of))
    # ClO4- is in log-space: c_ClO4 = exp(u_3)
    u3_data = U.dat[3].data_ro
    c3_min = float(np.exp(np.clip(u3_data.min(), -50, 50)))
    c3_max = float(np.exp(np.clip(u3_data.max(), -50, 50)))
    print(f"  z=0: ok={ok_z0}, cd={cd_z0:.6f}, ClO4 c_range=[{c3_min:.2e}, {c3_max:.2e}]")

    # z-ramp
    z_nominal = [float(sp[4][i]) for i in range(n)]
    achieved_z = 0.0

    with adj.stop_annotating():
        for z_val in np.linspace(0, 1, z_steps+1)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n): zc[i].assign(z_nominal[i]*z_val)
            ok_z, steps_z, msg_z = run_ss(60)

            u3_data = U.dat[3].data_ro
            c3_min = float(np.exp(np.clip(u3_data.min(), -50, 50)))

            if ok_z or steps_z > 0:
                achieved_z = z_val
                cd_z = float(fd.assemble(of))
                if z_val in [0.05, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] or not ok_z:
                    print(f"  z={z_val:.3f}: ok={ok_z}, cd={cd_z:.6f}, ClO4_min={c3_min:.2e}")
            else:
                for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                Up.assign(U)
                print(f"  z={z_val:.3f}: FAILED ({msg_z}), ClO4_min={c3_min:.2e}")
                # Fine steps
                for z_f in np.linspace(achieved_z, z_val, 6)[1:]:
                    U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n): zc[i].assign(z_nominal[i]*z_f)
                    ok_f, steps_f, _ = run_ss(60)
                    if ok_f or steps_f > 0:
                        achieved_z = z_f
                        print(f"    z={z_f:.4f}: ok={ok_f}")
                    else:
                        for src, dst in zip(U_ckpt2, U.dat): dst.data[:] = src
                        Up.assign(U)
                        print(f"    z={z_f:.4f}: FAILED")
                        break
                break

    cd_final = float(fd.assemble(of))
    print(f"  >> z={achieved_z:.4f}, cd={cd_final:.6f}")
    return achieved_z, cd_final


def main():
    # Standard solver reference: z=1 boundary at V=0.10V
    # Target: extend past 0.10V with correct physics
    V_points = [-0.30, -0.10, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

    results = []
    for V in V_points:
        t0 = time.time()
        try:
            z, cd = test_point(V)
            results.append({"V": V, "z": z, "cd": cd, "time": time.time()-t0})
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"V": V, "z": 0, "cd": float("nan"), "error": str(e), "time": time.time()-t0})

    print("\n" + "="*70)
    print("MIXED LOG-C (ClO4- only) CONVERGENCE MAP")
    print("="*70)
    print(f"{'V_RHE':>8} {'z_achieved':>10} {'cd':>12} {'time':>6} {'status':>10}")
    print("-"*50)

    # Standard solver reference
    std_ref = {-0.30: -0.181333, -0.10: -0.177529, 0.0: -0.173297, 0.05: -0.169359, 0.10: -0.162887}

    for r in results:
        V = r["V"]
        z = r["z"]
        cd = r["cd"]
        t = r["time"]
        status = "FULL" if z >= 0.999 else f"z={z:.2f}"
        ref = std_ref.get(V)
        if ref and z >= 0.999:
            err = abs(cd - ref) / abs(ref) * 100
            print(f"{V:8.2f} {z:10.4f} {cd:12.6f} {t:5.1f}s {status:>10} (err vs std: {err:.1f}%)")
        else:
            print(f"{V:8.2f} {z:10.4f} {cd:12.6f} {t:5.1f}s {status:>10}")


if __name__ == "__main__":
    main()
