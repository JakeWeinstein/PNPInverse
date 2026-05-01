"""V18 Simple Adjoint Test: Single-point adjoint gradient with stabilization.

Tests that the adjoint gradient works through the stabilized forms.
Uses the same pattern as FluxCurve/bv_point_solve/forward.py but
with monkey-patched build_forms for stabilization.
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
import firedrake as fd
import firedrake.adjoint as adj

from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
from Forward.bv_solver.stabilization import add_stabilization
from Forward.bv_solver.observables import _build_bv_observable_form

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_adjoint_inference")
os.makedirs(OUT_DIR, exist_ok=True)

TRUE = {"k0_r1": K0_HAT_R1, "k0_r2": K0_HAT_R2, "alpha_r1": ALPHA_R1, "alpha_r2": ALPHA_R2}


def test_adjoint_gradient(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, target_cd):
    """Compute objective + adjoint gradient at a single voltage point."""
    phi_hat = V_RHE / V_T

    sp = make_bv_solver_params(
        eta_hat=phi_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    sp_list = list(sp)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # Clear tape and enable annotation
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    # Build context + forms WITH annotation (for adjoint tape)
    ctx = build_context(sp_list, mesh=mesh)
    ctx = build_forms(ctx, sp_list)
    ctx = add_stabilization(ctx, sp_list, d_art_scale=0.001, stabilized_species=[3])
    set_initial_conditions(ctx, sp_list)

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]

    # Get k0 and alpha controls
    k0_funcs = list(ctx["bv_k0_funcs"])
    alpha_funcs = list(ctx["bv_alpha_funcs"])

    # Build solver
    F_res = ctx["F_res"]
    J_form = fd.derivative(F_res, U)
    prob = fd.NonlinearVariationalProblem(F_res, U, bcs=ctx["bcs"], J=J_form)

    sp_dict = {k: v for k, v in sp_list[10].items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(sp_list[10], dict) else {}
    solver = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    observable_form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE
    )

    # Phase 1: z=0 warm-up (WITH annotation off for speed)
    with adj.stop_annotating():
        for zci in zc:
            zci.assign(0.0)
        paf.assign(phi_hat)
        for _ in range(80):
            solver.solve()
            Up.assign(U)

        # Phase 2: z-ramp (with annotation off)
        z_nominal = [float(sp_list[4][i]) for i in range(n)]
        for z_val in np.linspace(0, 1, 21)[1:]:
            for i in range(n):
                zc[i].assign(z_nominal[i] * z_val)
            for _ in range(40):
                solver.solve()
                Up.assign(U)

    # Phase 3: Final time-steps WITH annotation (for adjoint tape)
    # Reset z to nominal
    for i in range(n):
        zc[i].assign(float(z_nominal[i]))

    # A few annotated time-steps from the converged z=1 state
    for step in range(5):
        solver.solve()
        Up.assign(U)

    # Compute objective: J = 0.5 * (cd - target)^2
    sim_cd_val = fd.assemble(observable_form)

    # Build target as an R-space function (pyadjoint-compatible)
    R_space = fd.FunctionSpace(mesh, "R", 0)
    target_f = fd.Function(R_space, name="target_cd")
    target_f.assign(target_cd)
    target_scalar = fd.assemble(target_f * fd.dx(domain=mesh))

    # The objective must be an AdjFloat (from fd.assemble)
    point_obj = 0.5 * (sim_cd_val - target_scalar) ** 2

    # Define controls and compute adjoint gradient
    controls = [adj.Control(f) for f in k0_funcs + alpha_funcs]
    rf = adj.ReducedFunctional(point_obj, controls)

    try:
        gradient = rf.derivative()
    except Exception as e:
        print(f"  Adjoint derivative failed: {e}")
        return float(sim_cd_val), [float("nan")] * (len(k0_funcs) + len(alpha_funcs))

    # Extract gradient values
    grad_vals = []
    for g in gradient:
        if hasattr(g, 'dat'):
            grad_vals.append(float(g.dat[0].data_ro[0]))
        elif hasattr(g, 'values'):
            grad_vals.append(float(g.values()[0]))
        else:
            grad_vals.append(float(g))

    return float(sim_cd_val), grad_vals


def main():
    print("V18 ADJOINT GRADIENT TEST")
    print("=" * 60)

    # Generate target data at true params (one point)
    V_test = 0.20  # onset region
    print(f"\nTest point: V_RHE = {V_test}V")

    # Target data
    print("Computing target at true params...")
    t0 = time.time()
    target_cd, _ = test_adjoint_gradient(
        V_test, TRUE["k0_r1"], TRUE["k0_r2"], TRUE["alpha_r1"], TRUE["alpha_r2"],
        target_cd=0.0  # dummy target for this call
    )
    print(f"  target cd = {target_cd:.6f} ({time.time()-t0:.1f}s)")

    # Test at true params (gradient should be ~0 when target=sim)
    print(f"\nGradient at TRUE params (target={target_cd:.6f})...")
    t0 = time.time()
    cd_true, grad_true = test_adjoint_gradient(
        V_test, TRUE["k0_r1"], TRUE["k0_r2"], TRUE["alpha_r1"], TRUE["alpha_r2"],
        target_cd=target_cd,
    )
    print(f"  cd = {cd_true:.6f}, grad = {grad_true}")
    print(f"  ({time.time()-t0:.1f}s)")

    # Test at 20% offset
    offset = 0.20
    init = [TRUE["k0_r1"]*(1+offset), TRUE["k0_r2"]*(1+offset),
            TRUE["alpha_r1"]*(1+offset), TRUE["alpha_r2"]*(1+offset)]
    print(f"\nGradient at 20% offset (target={target_cd:.6f})...")
    t0 = time.time()
    cd_off, grad_off = test_adjoint_gradient(
        V_test, init[0], init[1], init[2], init[3],
        target_cd=target_cd,
    )
    J_off = 0.5 * (cd_off - target_cd)**2
    print(f"  cd = {cd_off:.6f}, J = {J_off:.6e}")
    print(f"  gradient: {grad_off}")
    print(f"  ({time.time()-t0:.1f}s)")

    names = ["k0_r1", "k0_r2", "alpha_r1", "alpha_r2"]
    print(f"\n{'Param':>12} {'Value':>12} {'Gradient':>12} {'Direction':>10}")
    for i, name in enumerate(names):
        val = init[i]
        g = grad_off[i]
        direction = "decrease" if g > 0 else "increase"
        print(f"{name:>12} {val:12.4e} {g:12.4e} {direction:>10}")


if __name__ == "__main__":
    main()
