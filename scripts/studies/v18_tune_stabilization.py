"""V18: Tune artificial diffusion to minimize bias while maintaining convergence.

Tests:
1. Different d_art_scale values (0.001 to 0.01)
2. ClO4-only stabilization (species 3 only, not H+)
3. Comparison with standard solver at overlap points

Goal: find the minimum stabilization that still converges at V=0.15-0.30V.
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

# Standard solver reference values (from first diagnostic)
STD_REF = {
    -0.30: -0.181333,
    -0.10: -0.177529,
    0.00: -0.173297,
    0.05: -0.169359,
    0.10: -0.162887,
}

V_TEST = [-0.30, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def solve_point_stabilized(V_RHE, d_art_scale=0.01, species_mask=None):
    """Solve single voltage point with stabilization. Returns (cd, z_achieved)."""
    phi_hat = V_RHE / V_T
    sp = make_bv_solver_params(
        eta_hat=phi_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
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

    if species_mask is None:
        species_mask = list(range(n))

    h = fd.CellSize(mesh)
    for i in species_mask:
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

    dt_init = 0.25
    def run_ss(max_steps=60):
        dt_val = dt_init; dt_const.assign(dt_val)
        prev_flux = None; prev_delta = None; steady_count = 0
        for s in range(1, max_steps+1):
            try: sol.solve()
            except: return False, s-1
            Up.assign(U)
            fv = float(fd.assemble(ocd))
            if prev_flux is not None:
                delta = abs(fv-prev_flux); scale = max(abs(fv),abs(prev_flux),1e-8)
                if delta/scale <= 1e-4 or delta <= 1e-8: steady_count += 1
                else: steady_count = 0
                if prev_delta and delta > 0:
                    r = prev_delta/delta
                    dt_val = min(dt_val*min(r,4),dt_init*20) if r>1 else max(dt_val*0.5,dt_init)
                    dt_const.assign(dt_val)
                prev_delta = delta
            prev_flux = fv
            if steady_count >= 4: return True, s
        return False, max_steps

    # z=0
    for zci in zc: zci.assign(0.0)
    paf.assign(phi_hat)
    with adj.stop_annotating():
        run_ss(100)

    # z-ramp
    z_nominal = [float(sp[4][i]) for i in range(n)]
    achieved_z = 0.0
    with adj.stop_annotating():
        for z_val in np.linspace(0, 1, 21)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n): zc[i].assign(z_nominal[i]*z_val)
            ok, steps = run_ss(60)
            if ok or steps > 0:
                achieved_z = z_val
            else:
                for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                Up.assign(U)
                # Fine steps
                for z_f in np.linspace(achieved_z, z_val, 6)[1:]:
                    U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n): zc[i].assign(z_nominal[i]*z_f)
                    ok_f, steps_f = run_ss(60)
                    if ok_f or steps_f > 0: achieved_z = z_f
                    else:
                        for src, dst in zip(U_ckpt2, U.dat): dst.data[:] = src
                        Up.assign(U); break
                break

    cd = float(fd.assemble(ocd))
    return cd, achieved_z


def main():
    configs = [
        {"d_art": 0.001, "mask": None, "label": "all species, 0.001"},
        {"d_art": 0.002, "mask": None, "label": "all species, 0.002"},
        {"d_art": 0.003, "mask": None, "label": "all species, 0.003"},
        {"d_art": 0.005, "mask": None, "label": "all species, 0.005"},
        {"d_art": 0.01,  "mask": None, "label": "all species, 0.01"},
        {"d_art": 0.001, "mask": [3],  "label": "ClO4 only, 0.001"},
        {"d_art": 0.005, "mask": [3],  "label": "ClO4 only, 0.005"},
        {"d_art": 0.01,  "mask": [3],  "label": "ClO4 only, 0.01"},
        {"d_art": 0.05,  "mask": [3],  "label": "ClO4 only, 0.05"},
    ]

    # Test at key voltages
    V_KEY = [0.10, 0.15, 0.20, 0.30]

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg['label']}")
        print(f"{'='*60}")
        for V in V_KEY:
            t0 = time.time()
            cd, z = solve_point_stabilized(V, d_art_scale=cfg["d_art"], species_mask=cfg["mask"])
            elapsed = time.time() - t0

            # Compare with standard at V=0.10
            ref = STD_REF.get(V, None)
            if ref and z >= 0.999:
                err = abs(cd - ref) / abs(ref) * 100
                print(f"  V={V:5.2f}: cd={cd:.6f} (ref={ref:.6f}, err={err:.1f}%), "
                      f"z={z:.3f}, {elapsed:.1f}s")
            elif z >= 0.999:
                print(f"  V={V:5.2f}: cd={cd:.6f}, z={z:.3f}, {elapsed:.1f}s")
            else:
                print(f"  V={V:5.2f}: z_achieved={z:.3f} (PARTIAL), {elapsed:.1f}s")


if __name__ == "__main__":
    main()
