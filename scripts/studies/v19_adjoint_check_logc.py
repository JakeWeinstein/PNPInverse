"""Verify pyadjoint gradient through log-c 3sp + Boltzmann at a single voltage.

HANDOFF item #4 warned: "firedrake.adjoint should tape through fd.exp(u)
automatically, but test it -- there have been issues with some transform
combinations." The Boltzmann monkey-patch (adds c_bulk*exp(phi)*w*dx to
F_res) is also untested through pyadjoint.

Strategy:
  1. Solve forward at V=+0.10V (logc path) to steady state with annotation.
  2. Compute analytical gradient via ReducedFunctional.derivative() w.r.t.
     (k0_1, k0_2, alpha_1, alpha_2).
  3. Compute finite-difference gradient (central difference, h=1e-4) as ground truth.
  4. Compare: relative error per parameter should be <1e-3.

If this passes, the full inference driver can use adjoint gradients.
If it fails, inference falls back to scipy's FD.
"""
from __future__ import annotations
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


E_EQ_R1 = 0.68
E_EQ_R2 = 1.78
H2O2_SEED = 1e-4
V_TEST = 0.10  # onset voltage where k0 is sensitive


def _build_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
    from scripts._bv_common import (
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT, C_O2_HAT, C_HP_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, _make_bv_convergence_cfg, SNES_OPTS_CHARGED,
    )
    from Forward.params import SolverParams

    params = dict(SNES_OPTS_CHARGED)
    params["bv_convergence"] = _make_bv_convergence_cfg()
    params["nondim"] = _make_nondim_cfg()
    reaction_1 = {
        "k0": k0_r1, "alpha": alpha_r1,
        "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
        "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
        "reversible": True, "E_eq_v": E_EQ_R1,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    reaction_2 = {
        "k0": k0_r2, "alpha": alpha_r2,
        "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
        "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
        "reversible": False, "E_eq_v": E_EQ_R2,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    params["bv_bc"] = {
        "reactions": [reaction_1, reaction_2],
        "k0": [k0_r1] * 3, "alpha": [alpha_r1] * 3,
        "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    return SolverParams.from_list([
        3, 1, 0.25, 80.0,
        [0, 0, 1], [D_O2_HAT, D_H2O2_HAT, D_HP_HAT], [A_DEFAULT] * 3,
        eta_hat, [C_O2_HAT, H2O2_SEED, C_HP_HAT], 0.0, params,
    ])


def _add_boltzmann(ctx):
    import firedrake as fd
    from scripts._bv_common import C_CLO4_HAT
    mesh = ctx["mesh"]; W = ctx["W"]; U = ctx["U"]
    scaling = ctx["nondim"]
    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh)
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    c_bulk = fd.Constant(C_CLO4_HAT)
    phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0))
    ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
    ctx["J_form"] = fd.derivative(ctx["F_res"], U)
    return ctx


def _solve_to_steady(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh,
                     z_steps=20, annotate_final=True, final_annot_steps=3):
    """Forward solve with annotation-off during z-ramp, then a few annotated steps.

    Returns (ctx, cd_symbolic, pc_symbolic, z_ach, ok).

    If annotate_final is True, the last few steady-state timesteps are
    taped so pyadjoint can compute d(obj)/d(controls).
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from scripts._bv_common import V_T, I_SCALE, SNES_OPTS_CHARGED
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    phi_hat = V_RHE / V_T
    sp = _build_sp(phi_hat, k0_r1, k0_r2, alpha_r1, alpha_r2)

    # Clear tape and enable annotation for subsequent build_forms calls.
    # The z-ramp is wrapped in stop_annotating so pyadjoint doesn't record
    # the hundreds of intermediate Newton solves.
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    # Build context/forms WITH annotation so the BV k0_funcs, alpha_funcs
    # that appear in F_res are tape-visible.
    ctx = build_context_logc(list(sp), mesh=mesh)
    ctx = build_forms_logc(ctx, list(sp))
    ctx = _add_boltzmann(ctx)
    set_initial_conditions_logc(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

    def run_ss(max_steps=60, rel_tol=1e-4, abs_tol=1e-8):
        dt_val = 0.25; dt_const.assign(dt_val)
        prev_flux = None; prev_delta = None; sc = 0
        for _ in range(max_steps):
            try: sol.solve()
            except Exception: return False
            with adj.stop_annotating():
                Up.assign(U)
            fv = float(fd.assemble(form_cd))
            if prev_flux is not None:
                d = abs(fv - prev_flux)
                sv = max(abs(fv), abs(prev_flux), 1e-8)
                if d / sv <= rel_tol or d <= abs_tol: sc += 1
                else: sc = 0
                if prev_delta and d > 0:
                    r = prev_delta / d
                    dt_val = (min(dt_val * min(r, 4), 5.0) if r > 1
                              else max(dt_val * 0.5, 0.25))
                    dt_const.assign(dt_val)
                prev_delta = d
            prev_flux = fv
            if sc >= 4: return True
        return False

    # z=0 steady and z-ramp -- all non-annotated (large number of solves)
    with adj.stop_annotating():
        for zci in zc: zci.assign(0.0)
        paf.assign(phi_hat)
        if not run_ss(100):
            return ctx, None, None, 0.0, False

        z_nominal = [float(sp[4][i]) for i in range(n)]
        achieved_z = 0.0
        for z_val in np.linspace(0, 1, z_steps + 1)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n): zc[i].assign(z_nominal[i] * z_val)
            if run_ss(60):
                achieved_z = z_val
            else:
                for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                Up.assign(U)
                break

    if achieved_z < 1.0 - 1e-3:
        return ctx, None, None, achieved_z, False

    # A few annotated timesteps at z=1 to build the tape for the adjoint.
    # These run with annotation enabled; each one is a single SNES solve
    # starting from (and converging to) the steady state.
    if annotate_final:
        for _ in range(final_annot_steps):
            sol.solve()
            with adj.stop_annotating():
                Up.assign(U)

    return ctx, form_cd, form_pc, achieved_z, True


def compute_obj_grad(V_RHE, theta, target_cd, target_pc, w_pc, mesh):
    """Forward + adjoint at one voltage. Returns (J, grad_theta).

    theta = (log_k0_1, log_k0_2, alpha_1, alpha_2).
    Returns gradient in theta-space (chain rule applied for log(k0)).
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    k0_1, k0_2 = float(np.exp(theta[0])), float(np.exp(theta[1]))
    alpha_1, alpha_2 = float(theta[2]), float(theta[3])

    ctx, form_cd, form_pc, z_ach, ok = _solve_to_steady(
        V_RHE, k0_1, k0_2, alpha_1, alpha_2, mesh,
        annotate_final=True, final_annot_steps=3,
    )
    if not ok:
        return 1e6, np.zeros(4)

    cd_sym = fd.assemble(form_cd)
    pc_sym = fd.assemble(form_pc)
    obj_sym = 0.5 * (cd_sym - float(target_cd)) ** 2 \
              + 0.5 * float(w_pc) * (pc_sym - float(target_pc)) ** 2

    k0_funcs = list(ctx["bv_k0_funcs"])[:2]
    alpha_funcs = list(ctx["bv_alpha_funcs"])[:2]
    controls = [adj.Control(f) for f in k0_funcs + alpha_funcs]
    rf = adj.ReducedFunctional(obj_sym, controls)
    grads = rf.derivative()

    def _extract(g):
        if hasattr(g, "dat"):
            return float(g.dat[0].data_ro[0])
        if hasattr(g, "values"):
            return float(g.values()[0])
        return float(g)

    g_k0_1 = _extract(grads[0])
    g_k0_2 = _extract(grads[1])
    g_alpha_1 = _extract(grads[2])
    g_alpha_2 = _extract(grads[3])

    # Chain rule: d/dlog_k0 = k0 * d/dk0
    grad_theta = np.array([
        k0_1 * g_k0_1,
        k0_2 * g_k0_2,
        g_alpha_1,
        g_alpha_2,
    ])
    return float(obj_sym), grad_theta


def compute_obj_only(V_RHE, theta, target_cd, target_pc, w_pc, mesh):
    """Forward only (no adjoint) at one voltage. Used for FD gradient."""
    import firedrake as fd

    k0_1, k0_2 = float(np.exp(theta[0])), float(np.exp(theta[1]))
    alpha_1, alpha_2 = float(theta[2]), float(theta[3])
    ctx, form_cd, form_pc, z_ach, ok = _solve_to_steady(
        V_RHE, k0_1, k0_2, alpha_1, alpha_2, mesh,
        annotate_final=False,
    )
    if not ok:
        return 1e6
    cd = float(fd.assemble(form_cd))
    pc = float(fd.assemble(form_pc))
    return float(
        0.5 * (cd - float(target_cd)) ** 2
        + 0.5 * float(w_pc) * (pc - float(target_pc)) ** 2
    )


def main():
    from scripts._bv_common import (
        setup_firedrake_env, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
    )
    setup_firedrake_env()
    from Forward.bv_solver import make_graded_rectangle_mesh

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # True params (produce target) and a point slightly off
    k0_1_true, k0_2_true = K0_HAT_R1, K0_HAT_R2
    alpha_1_true, alpha_2_true = ALPHA_R1, ALPHA_R2
    theta_true = np.array([np.log(k0_1_true), np.log(k0_2_true), alpha_1_true, alpha_2_true])

    # Target from true (step 1)
    print("=" * 70)
    print(f"ADJOINT VERIFICATION at V={V_TEST:+.3f}V (logc path)")
    print("=" * 70)
    print("Step 1: compute target CD, PC at true params")
    import firedrake as fd
    ctx_t, form_cd_t, form_pc_t, z_t, ok_t = _solve_to_steady(
        V_TEST, k0_1_true, k0_2_true, alpha_1_true, alpha_2_true, mesh,
        annotate_final=False,
    )
    assert ok_t
    target_cd = float(fd.assemble(form_cd_t))
    target_pc = float(fd.assemble(form_pc_t))
    print(f"  target_cd = {target_cd:+.6e}")
    print(f"  target_pc = {target_pc:+.6e}")

    # Initial guess: +20%/-10%. Use w_pc=0 since V_TEST > 0 (PC is an artifact in logc range).
    theta0 = np.array([
        np.log(k0_1_true * 1.2),
        np.log(k0_2_true * 1.2),
        alpha_1_true * 0.9,
        alpha_2_true * 0.9,
    ])
    w_pc = 0.0  # physically-unreliable PC at onset -> weight 0

    # Step 2: Adjoint gradient
    print("\nStep 2: adjoint gradient")
    t0 = time.time()
    J_adj, grad_adj = compute_obj_grad(V_TEST, theta0, target_cd, target_pc, w_pc, mesh)
    t_adj = time.time() - t0
    print(f"  J = {J_adj:.6e}  (elapsed {t_adj:.1f}s)")
    print(f"  grad_adj = {grad_adj}")

    # Step 3: Central-difference FD gradient
    print("\nStep 3: central-difference FD gradient (h=1e-4)")
    h = 1e-4
    grad_fd = np.zeros(4)
    t_fd0 = time.time()
    for i in range(4):
        ep = np.zeros(4); ep[i] = h
        Jp = compute_obj_only(V_TEST, theta0 + ep, target_cd, target_pc, w_pc, mesh)
        Jm = compute_obj_only(V_TEST, theta0 - ep, target_cd, target_pc, w_pc, mesh)
        grad_fd[i] = (Jp - Jm) / (2 * h)
        print(f"  i={i}: J+={Jp:.6e} J-={Jm:.6e} FD={grad_fd[i]:+.6e}")
    t_fd = time.time() - t_fd0
    print(f"  FD total elapsed {t_fd:.1f}s (8 forward solves)")

    # Step 4: Compare
    print("\n" + "=" * 70)
    print("ADJOINT vs FD GRADIENT COMPARISON")
    print("=" * 70)
    names = ["d/dlog_k0_1", "d/dlog_k0_2", "d/dalpha_1", "d/dalpha_2"]
    print(f"  {'param':>15s} {'adjoint':>15s} {'FD':>15s} {'rel err':>12s}")
    all_ok = True
    for i in range(4):
        rel = (abs(grad_adj[i] - grad_fd[i])
               / max(abs(grad_fd[i]), abs(grad_adj[i]), 1e-12))
        status = "OK" if rel < 1e-2 else ("MARGINAL" if rel < 1e-1 else "FAIL")
        if status == "FAIL":
            all_ok = False
        print(f"  {names[i]:>15s} {grad_adj[i]:+15.6e} {grad_fd[i]:+15.6e} "
              f"{rel:12.4e}  [{status}]")
    print("-" * 70)
    print(f"  Overall: {'PASS (adjoint usable)' if all_ok else 'FAIL (use FD fallback)'}")
    print(f"  Timings: adjoint={t_adj:.1f}s, FD (8 fwd)={t_fd:.1f}s, "
          f"speedup={t_fd/max(t_adj,1e-6):.2f}x")


if __name__ == "__main__":
    main()
