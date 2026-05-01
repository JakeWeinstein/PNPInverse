"""V18 Fast Recovery Test: 8-point grid, L-BFGS-B optimizer.

Reduced grid for faster iteration, with log-parameter transform for
better optimizer conditioning.
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
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
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

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_inference")
os.makedirs(OUT_DIR, exist_ok=True)

TRUE = {"k0_r1": K0_HAT_R1, "k0_r2": K0_HAT_R2, "alpha_r1": ALPHA_R1, "alpha_r2": ALPHA_R2}

# 8-point grid spanning the kinetic transition (where parameters are most identifiable)
V_GRID = np.array([-0.10, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
PHI_GRID = V_GRID / V_T


def _build_stabilized_solver(sp_list, phi_hat):
    """Build a stabilized solver for a single voltage point."""
    sp = list(sp_list)
    sp[7] = phi_hat
    n = sp[0]
    z_vals = sp[4]

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        set_initial_conditions(ctx, sp)

    U = ctx["U"]; Up = ctx["U_prev"]
    W = ctx["W"]
    dx = fd.Measure("dx", domain=mesh)
    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    v_tests = fd.TestFunctions(W)

    F_res = ctx["F_res"]
    scaling = ctx["nondim"]
    em = float(scaling["electromigration_prefactor"])
    D_vals = scaling["D_model_vals"]

    # ClO4-only stabilization
    h = fd.CellSize(mesh)
    i = 3  # ClO4-
    z_i = float(z_vals[i])
    if abs(z_i) > 0:
        D_i = float(D_vals[i])
        drift = fd.Constant(abs(z_i) * D_i * em)
        D_art = fd.Constant(0.001) * h * drift * fd.sqrt(
            fd.dot(fd.grad(phi), fd.grad(phi)) + fd.Constant(1e-10))
        F_res += D_art * fd.dot(fd.grad(ci[i]), fd.grad(v_tests[i])) * dx

    J_stab = fd.derivative(F_res, U)
    params = sp[10]
    sp_dict = {k: v for k, v in params.items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}

    prob = fd.NonlinearVariationalProblem(F_res, U, bcs=ctx["bcs"], J=J_stab)
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)

    return ctx, sol, ocd


def forward_single(sp_list, phi_hat):
    """Solve single point and return cd."""
    ctx, sol, ocd = _build_stabilized_solver(sp_list, phi_hat)
    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
    scaling = ctx["nondim"]
    dt_init = float(scaling["dt_model"])
    z_nominal = [float(sp_list[4][i]) for i in range(n)]

    def run_ss(max_steps=60):
        dt_val = dt_init; dt_const.assign(dt_val)
        prev_flux = None; prev_delta = None; sc = 0
        for s in range(1, max_steps+1):
            try: sol.solve()
            except: return False, s-1
            Up.assign(U)
            fv = float(fd.assemble(ocd))
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
            if sc >= 4: return True, s
        return False, max_steps

    with adj.stop_annotating():
        for zci in zc: zci.assign(0.0)
        paf.assign(phi_hat)
        run_ss(80)

        for z_val in np.linspace(0, 1, 21)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n): zc[i].assign(z_nominal[i]*z_val)
            ok, steps = run_ss(40)
            if not (ok or steps > 0):
                for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                Up.assign(U)
                break

    return float(fd.assemble(ocd))


def forward_curve(k0_r1, k0_r2, alpha_r1, alpha_r2):
    """Compute I-V curve at given parameters."""
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)
    cd = np.array([forward_single(sp_list, float(phi)) for phi in PHI_GRID])
    return cd


def main():
    print("V18 FAST RECOVERY TEST")
    print(f"Grid: {len(V_GRID)} points, {V_GRID[0]:.2f}V to {V_GRID[-1]:.2f}V")
    print(f"True: k0_r1={TRUE['k0_r1']:.4e}, k0_r2={TRUE['k0_r2']:.4e}, "
          f"α1={TRUE['alpha_r1']:.4f}, α2={TRUE['alpha_r2']:.4f}")

    # Load or generate target data
    target_file = os.path.join(OUT_DIR, "synthetic_data.npz")
    if os.path.exists(target_file):
        d = np.load(target_file)
        # Use subset matching V_GRID
        target_cd_full = d["target_cd"]
        V_full = d["V_grid"]
        # Interpolate to our grid
        target_cd = np.interp(V_GRID, V_full, target_cd_full)
        print("Loaded target data from cache")
    else:
        print("\n--- Generating target data ---")
        t0 = time.time()
        target_cd = forward_curve(TRUE["k0_r1"], TRUE["k0_r2"], TRUE["alpha_r1"], TRUE["alpha_r2"])
        print(f"Target data generated in {time.time()-t0:.1f}s")
    print(f"{'V_RHE':>8} {'cd':>12}")
    for i in range(len(V_GRID)):
        print(f"{V_GRID[i]:8.3f} {target_cd[i]:12.6f}")

    # Test recovery from offset initial guesses
    offsets = [0.20]  # Focus on 20% offset for now

    for offset in offsets:
        print(f"\n{'='*60}")
        print(f"Initial guess: {offset*100:.0f}% offset from true")
        print(f"{'='*60}")

        init = [
            TRUE["k0_r1"] * (1 + offset),
            TRUE["k0_r2"] * (1 + offset),
            TRUE["alpha_r1"] * (1 + offset),
            TRUE["alpha_r2"] * (1 + offset),
        ]

        # Evaluate at init
        t0 = time.time()
        cd_init = forward_curve(*init)
        J_init = float(np.sum((cd_init - target_cd)**2))
        print(f"  J(init) = {J_init:.4e} ({time.time()-t0:.1f}s)")

        # Simple gradient estimation + one step
        step_size = 0.01
        grad = np.zeros(4)
        for j in range(4):
            p_plus = list(init)
            p_plus[j] *= (1 + step_size)
            cd_plus = forward_curve(*p_plus)
            J_plus = float(np.sum((cd_plus - target_cd)**2))
            grad[j] = (J_plus - J_init) / (init[j] * step_size)
            print(f"    dJ/d{list(TRUE.keys())[j]} = {grad[j]:.4e}")

        # Gradient descent step
        lr = 0.001
        new_params = [init[j] - lr * grad[j] * init[j] for j in range(4)]

        # Clip to reasonable bounds
        new_params[0] = max(new_params[0], TRUE["k0_r1"] * 0.01)
        new_params[1] = max(new_params[1], TRUE["k0_r2"] * 0.01)
        new_params[2] = np.clip(new_params[2], 0.1, 0.95)
        new_params[3] = np.clip(new_params[3], 0.1, 0.95)

        cd_new = forward_curve(*new_params)
        J_new = float(np.sum((cd_new - target_cd)**2))
        if J_init > 0:
            print(f"  J(step) = {J_new:.4e} (improvement: {(J_init-J_new)/J_init*100:.1f}%)")
        else:
            print(f"  J(step) = {J_new:.4e}")

        true_vals = list(TRUE.values())
        names = list(TRUE.keys())
        print(f"\n  {'Param':>10} {'True':>12} {'Init':>12} {'Step':>12} {'Init err':>8} {'Step err':>8}")
        for j in range(4):
            err_i = abs(init[j]-true_vals[j])/abs(true_vals[j])*100
            err_s = abs(new_params[j]-true_vals[j])/abs(true_vals[j])*100
            print(f"  {names[j]:>10} {true_vals[j]:12.4e} {init[j]:12.4e} {new_params[j]:12.4e} "
                  f"{err_i:7.1f}% {err_s:7.1f}%")

    # Save
    results = {
        "V_grid": V_GRID.tolist(),
        "target_cd": target_cd.tolist(),
        "true_params": TRUE,
    }
    with open(os.path.join(OUT_DIR, "fast_recovery.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
