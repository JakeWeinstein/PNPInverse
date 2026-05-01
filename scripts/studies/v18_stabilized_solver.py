"""V18: Stabilized PNP solver with positivity penalty + artificial diffusion.

The monolithic FEM solver fails because:
1. Co-ion (ClO4-) goes negative in the underresolved Debye layer
2. Once negative, Newton amplifies the error catastrophically

This script tests TWO stabilization approaches applied to the existing solver:
A) Positivity penalty: add F_penalty = -gamma * max(-c, 0)^2 * v * dx
B) Artificial diffusion: add D_art * |z*grad(phi)| * grad(c) * grad(v) * dx

Both are applied within the standard forms.py framework by modifying F_res.
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
import firedrake as fd
import pyadjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
from Forward.bv_solver.observables import _build_bv_observable_form

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78


def test_stabilized(V_RHE, gamma=1e6, d_art_scale=0.01, z_steps=20):
    """Test z-ramp at V_RHE with penalty + artificial diffusion stabilization."""
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

    # ---- ADD STABILIZATION TO F_res ----
    F_res = ctx["F_res"]
    scaling = ctx["nondim"]
    em = float(scaling["electromigration_prefactor"])
    z_vals = list(sp[4])
    D_vals = scaling["D_model_vals"]

    # A) Positivity penalty: penalize negative concentrations
    gamma_c = fd.Constant(gamma)
    for i in range(n):
        neg_part = fd.max_value(fd.Constant(0.0) - ci[i], fd.Constant(0.0))
        F_res += gamma_c * neg_part * neg_part * v_tests[i] * dx

    # B) Artificial diffusion: isotropic diffusion proportional to |drift|
    # For charged species, add D_art * h^2 * |grad(phi)|^2 so the effective
    # Peclet number stays O(1) even in the Debye layer.
    # h = local mesh size (approximate)
    h = fd.CellSize(mesh)
    for i in range(n):
        z_i = float(z_vals[i])
        if abs(z_i) > 0:
            D_i = float(D_vals[i])
            # Artificial diffusion coefficient: d_art * h * |v_drift|
            # v_drift = z * D * em * grad(phi)
            drift_speed = fd.Constant(abs(z_i) * D_i * em)
            D_artificial = fd.Constant(d_art_scale) * h * drift_speed * fd.sqrt(fd.dot(fd.grad(phi), fd.grad(phi)) + fd.Constant(1e-10))
            F_res += D_artificial * fd.dot(fd.grad(ci[i]), fd.grad(v_tests[i])) * dx

    # Re-derive Jacobian with stabilization
    J_stab = fd.derivative(F_res, U)

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    prob = fd.NonlinearVariationalProblem(F_res, U, bcs=ctx["bcs"], J=J_stab)
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)

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
        return False, max_steps, "budget"

    # Phase 1: z=0
    for zci in zc:
        zci.assign(0.0)
    paf.assign(phi_hat)

    print(f"\n--- V_RHE={V_RHE:.2f}V (gamma={gamma:.0e}, d_art={d_art_scale}) ---")

    with adj.stop_annotating():
        ok_z0, steps_z0, msg_z0 = run_ss(100)

    cd_z0 = float(fd.assemble(of))
    print(f"  z=0: ok={ok_z0}, cd={cd_z0:.6f}")

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
            cd_z = float(fd.assemble(of))
            c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]

            if ok_z or steps_z > 0:
                achieved_z = z_val
                if z_val in [0.05, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] or not ok_z:
                    print(f"  z={z_val:.3f}: ok={ok_z}, steps={steps_z}, "
                          f"cd={cd_z:.6f}, c_min={[f'{c:.2e}' for c in c_min]}")
            else:
                for src, dst in zip(U_ckpt, U.dat):
                    dst.data[:] = src
                Up.assign(U)
                print(f"  z={z_val:.3f}: FAILED, c_min={[f'{c:.2e}' for c in c_min]}")

                # Fine steps
                z_fine = np.linspace(achieved_z, z_val, 6)[1:]
                for z_f in z_fine:
                    U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n):
                        zc[i].assign(z_nominal[i] * z_f)
                    ok_f, steps_f, _ = run_ss(60)
                    if ok_f or steps_f > 0:
                        achieved_z = z_f
                        print(f"    z={z_f:.4f}: ok={ok_f}")
                    else:
                        for src, dst in zip(U_ckpt2, U.dat):
                            dst.data[:] = src
                        Up.assign(U)
                        print(f"    z={z_f:.4f}: FAILED")
                        break
                break

    print(f"  >> Achieved z={achieved_z:.4f}")
    return achieved_z, cd_z0


def main():
    # Test at the boundary voltage V=0.15V with different stabilization params
    print("="*70)
    print("STABILIZATION PARAMETER SWEEP at V_RHE=0.15V")
    print("="*70)

    configs = [
        {"gamma": 0,     "d_art_scale": 0,     "label": "baseline (no stab)"},
        {"gamma": 1e4,   "d_art_scale": 0,     "label": "penalty only (1e4)"},
        {"gamma": 1e6,   "d_art_scale": 0,     "label": "penalty only (1e6)"},
        {"gamma": 0,     "d_art_scale": 0.01,  "label": "art. diff only (0.01)"},
        {"gamma": 0,     "d_art_scale": 0.1,   "label": "art. diff only (0.1)"},
        {"gamma": 0,     "d_art_scale": 1.0,   "label": "art. diff only (1.0)"},
        {"gamma": 1e4,   "d_art_scale": 0.1,   "label": "both (1e4, 0.1)"},
        {"gamma": 1e6,   "d_art_scale": 0.1,   "label": "both (1e6, 0.1)"},
    ]

    results = []
    for cfg in configs:
        t0 = time.time()
        z, cd = test_stabilized(0.15, gamma=cfg["gamma"], d_art_scale=cfg["d_art_scale"])
        elapsed = time.time() - t0
        results.append({**cfg, "z_achieved": z, "cd_z0": cd, "time": elapsed})

    print("\n" + "="*70)
    print("STABILIZATION RESULTS at V_RHE=0.15V (standard: z_max=0.79)")
    print("="*70)
    for r in results:
        print(f"  {r['label']:30s}: z_achieved={r['z_achieved']:.4f}, time={r['time']:.1f}s")

    # If any worked, test at 0.20V and 0.30V
    best = max(results, key=lambda x: x["z_achieved"])
    if best["z_achieved"] > 0.80:
        print(f"\nBest config: {best['label']}")
        print("Testing at higher voltages...")
        for V in [0.20, 0.25, 0.30]:
            t0 = time.time()
            z, cd = test_stabilized(V, gamma=best["gamma"], d_art_scale=best["d_art_scale"])
            print(f"  V={V:.2f}: z_achieved={z:.4f}, time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
