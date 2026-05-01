"""V18 adjoint-vs-FD verification through forms_logc + Boltzmann.

Verifies that pyadjoint can correctly tape through:
  - log-concentration transform (u = ln c)
  - 3-species reduced system
  - Boltzmann ClO4- background (monkey-patched into Poisson source)
  - Joint observable J = 0.5*(cd-target_cd)^2 + 0.5*(pc-target_pc)^2

If this passes, the main 4-param inversion can switch from Nelder-Mead to
L-BFGS-B with adjoint gradients (~5-10x speedup).

Tests at V_RHE = +0.20 V (peak k0 sensitivity) at the +20% offset point.
Compares adjoint dJ/dtheta vs central-difference FD for theta in
{log k0_1, log k0_2, alpha_1, alpha_2}.

Pass criterion: |adjoint - FD| / |FD| < 5% for each component (with 5%
slack for FD truncation + SNES convergence noise).
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


def main():
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, _make_bv_convergence_cfg,
        SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
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
    THREE_SPECIES_C0 = [C_O2_HAT, H2O2_SEED, C_HP_HAT]

    V_TEST = 0.20  # V_RHE: strongest k0 sensitivity per logc_k0_sensitivity scan

    OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_logc_joint_observable")
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    def make_3sp_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
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
            3, 1, 0.25, 80.0, THREE_SPECIES_Z, THREE_SPECIES_D, THREE_SPECIES_A,
            eta_hat, THREE_SPECIES_C0, 0.0, params,
        ])

    def add_boltzmann(ctx):
        mesh = ctx["mesh"]
        W = ctx["W"]; U = ctx["U"]
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

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    def solve_to_steady(k0_r1, k0_r2, alpha_r1, alpha_r2,
                        annotate_final_steps=0, warm_ic=None,
                        warm_max_steps=80):
        """Solve to steady state at V_TEST.

        If warm_ic is provided (tuple of numpy arrays), load it as the IC and
        skip the z-ramp (run z=1 directly).  Otherwise do a full cold z=0 -> z=1
        ramp.

        If annotate_final_steps>0, the final N solves are annotated so that
        pyadjoint can build the gradient.

        Returns (cd, pc, U_snapshot, ctx).
        """
        sp = make_3sp_sp(V_TEST / V_T, k0_r1, k0_r2, alpha_r1, alpha_r2)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))

        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT)
        of_cd = _build_bv_observable_form(ctx, mode="current_density",
                                           reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(ctx, mode="peroxide_current",
                                           reaction_index=None, scale=-I_SCALE)

        dt_init = 0.25
        z_nominal = [float(sp[4][i]) for i in range(n)]

        def run_ss(max_steps):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for s in range(1, max_steps + 1):
                try:
                    sol.solve()
                except Exception:
                    return False
                Up.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    d = abs(fv - prev_flux); sv = max(abs(fv), abs(prev_flux), 1e-8)
                    if d / sv <= 1e-4 or d <= 1e-8: sc += 1
                    else: sc = 0
                    if prev_delta and d > 0:
                        r = prev_delta / d
                        dt_val = (min(dt_val * min(r, 4), dt_init * 20)
                                  if r > 1 else max(dt_val * 0.5, dt_init))
                        dt_const.assign(dt_val)
                    prev_delta = d
                prev_flux = fv
                if sc >= 4: return True
            return False

        if warm_ic is not None:
            # Warm-start: load IC, set z=1, run to convergence (no annotation
            # for the warm-up steps; final N annotated below).
            with adj.stop_annotating():
                for src, dst in zip(warm_ic, U.dat):
                    dst.data[:] = src
                Up.assign(U)
                for i in range(n): zc[i].assign(z_nominal[i])
                paf.assign(V_TEST / V_T)
                if annotate_final_steps == 0:
                    if not run_ss(warm_max_steps):
                        return None, None, None, ctx
                else:
                    # Run warm-up for (warm_max_steps - annotate_final_steps)
                    # steps, then break and finish with annotation.
                    warm_only = max(1, warm_max_steps - annotate_final_steps)
                    if not run_ss(warm_only):
                        # converge fully without annotation, then add a few
                        # annotated steps from the converged state.
                        if not run_ss(warm_max_steps):
                            return None, None, None, ctx
        else:
            # Cold path: z=0, then z-ramp 0->1
            with adj.stop_annotating():
                for zci in zc: zci.assign(0.0)
                paf.assign(V_TEST / V_T)
                if not run_ss(100):
                    return None, None, None, ctx

                achieved_z = 0.0
                for z_val in np.linspace(0, 1, 21)[1:]:
                    ckpt = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n): zc[i].assign(z_nominal[i] * z_val)
                    if run_ss(60):
                        achieved_z = z_val
                    else:
                        for src, dst in zip(ckpt, U.dat): dst.data[:] = src
                        Up.assign(U)
                        break
                if achieved_z < 1.0 - 1e-3:
                    return None, None, None, ctx
                for i in range(n): zc[i].assign(z_nominal[i])

        # Phase 3: a few annotated steps from converged state.
        if annotate_final_steps > 0:
            for _ in range(annotate_final_steps):
                sol.solve()
                Up.assign(U)

        cd_v = float(fd.assemble(of_cd))
        pc_v = float(fd.assemble(of_pc))
        snap = tuple(d.data_ro.copy() for d in U.dat)
        return cd_v, pc_v, snap, ctx

    # ------------------------------------------------------------------
    # Step 1: Compute target at TRUE params (no annotation)
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"V18 ADJOINT-vs-FD CHECK at V_RHE = {V_TEST} V")
    print("=" * 70)
    print(f"True params: k0_1={K0_HAT_R1:.4e}, k0_2={K0_HAT_R2:.4e}, "
          f"a_1={ALPHA_R1:.4f}, a_2={ALPHA_R2:.4f}")
    print()
    print("Step 1: target at true params (cold start, no tape)...", flush=True)
    t0 = time.time()
    with adj.stop_annotating():
        target_cd, target_pc, true_U_snap, _ = solve_to_steady(
            K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, annotate_final_steps=0)
    print(f"  target cd={target_cd:.6e}, pc={target_pc:.6e} ({time.time()-t0:.1f}s)")

    if target_cd is None:
        print("FATAL: target solve did not converge at TRUE params.")
        return

    # ------------------------------------------------------------------
    # Step 2: Adjoint gradient at +20% offset
    # ------------------------------------------------------------------
    offset = 0.20
    init_k0_1 = K0_HAT_R1 * (1 + offset)
    init_k0_2 = K0_HAT_R2 * (1 + offset)
    init_a_1 = ALPHA_R1 * (1 + offset)
    init_a_2 = ALPHA_R2 * (1 + offset)
    print()
    print(f"Step 2: adjoint gradient at +{offset*100:.0f}% offset...", flush=True)
    print(f"  k0_1={init_k0_1:.4e}, k0_2={init_k0_2:.4e}, "
          f"a_1={init_a_1:.4f}, a_2={init_a_2:.4f}")

    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    t0 = time.time()
    cd_off, pc_off, _, ctx = solve_to_steady(
        init_k0_1, init_k0_2, init_a_1, init_a_2,
        annotate_final_steps=3, warm_ic=true_U_snap)
    if cd_off is None:
        adj.pause_annotation()
        print("FATAL: warm-started solve at +20% offset did not converge.")
        return
    print(f"  cd={cd_off:.6e}, pc={pc_off:.6e} ({time.time()-t0:.1f}s, warm-start from TRUE)")

    of_cd = _build_bv_observable_form(ctx, mode="current_density",
                                       reaction_index=None, scale=-I_SCALE)
    of_pc = _build_bv_observable_form(ctx, mode="peroxide_current",
                                       reaction_index=None, scale=-I_SCALE)
    # fd.assemble of a scalar form returns AdjFloat (an OverloadedType)
    # so arithmetic with plain Python floats preserves tape.
    cd_assembled = fd.assemble(of_cd)
    pc_assembled = fd.assemble(of_pc)

    # Inverse-variance weights
    sigma_cd = 0.02 * abs(float(target_cd))
    sigma_pc = 0.02 * abs(float(target_pc))
    inv_var_cd = 1.0 / (sigma_cd ** 2)
    inv_var_pc = 1.0 / (sigma_pc ** 2)
    target_cd_f = float(target_cd)
    target_pc_f = float(target_pc)
    J = 0.5 * inv_var_cd * (cd_assembled - target_cd_f) ** 2 \
        + 0.5 * inv_var_pc * (pc_assembled - target_pc_f) ** 2

    J_val = float(J)
    print(f"  J at offset = {J_val:.6e}  (type {type(J).__name__})")

    k0_funcs = list(ctx["bv_k0_funcs"])
    alpha_funcs = list(ctx["bv_alpha_funcs"])
    print(f"  ctx exposes {len(k0_funcs)} k0 controls, {len(alpha_funcs)} alpha controls")

    if len(k0_funcs) < 2 or len(alpha_funcs) < 2:
        adj.pause_annotation()
        print(f"FATAL: expected 2 k0 and 2 alpha controls (one per reaction); "
              f"got {len(k0_funcs)} k0, {len(alpha_funcs)} alpha.")
        return

    controls = [adj.Control(f) for f in k0_funcs[:2] + alpha_funcs[:2]]
    rf = adj.ReducedFunctional(J, controls)

    t0 = time.time()
    try:
        gradient = rf.derivative()
    except Exception as e:
        adj.pause_annotation()
        print(f"FATAL: adjoint derivative raised: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return
    t_adj = time.time() - t0

    grad_vals = []
    for g in gradient:
        if hasattr(g, "dat"):
            grad_vals.append(float(g.dat[0].data_ro[0]))
        elif hasattr(g, "values"):
            grad_vals.append(float(g.values()[0]))
        else:
            grad_vals.append(float(g))

    adj.pause_annotation()
    tape.clear_tape()

    print(f"  adjoint gradient ({t_adj:.1f}s):")
    names = ["k0_r1", "k0_r2", "alpha_r1", "alpha_r2"]
    for n, g in zip(names, grad_vals):
        print(f"    dJ/d{n:>8} = {g:+.6e}")

    # ------------------------------------------------------------------
    # Step 3: Central-difference FD gradient at +20% offset
    #
    # Use relative step in the direction of each parameter.  Rel step
    # 1e-3 keeps round-off below truncation while staying in the linear
    # regime for these magnitudes.
    # ------------------------------------------------------------------
    print()
    print("Step 3: central-difference gradient (8 forward solves)...", flush=True)

    eps_rel = 1e-3
    base = np.array([init_k0_1, init_k0_2, init_a_1, init_a_2])

    def J_at(theta):
        with adj.stop_annotating():
            cd_v, pc_v, _, _ = solve_to_steady(
                float(theta[0]), float(theta[1]),
                float(theta[2]), float(theta[3]),
                annotate_final_steps=0, warm_ic=true_U_snap)
        if cd_v is None or not np.isfinite(cd_v) or not np.isfinite(pc_v):
            return float("nan")
        return 0.5 * ((cd_v - target_cd) / sigma_cd) ** 2 \
             + 0.5 * ((pc_v - target_pc) / sigma_pc) ** 2

    fd_grad = np.zeros(4)
    t_fd = time.time()
    for i in range(4):
        eps = eps_rel * abs(base[i])
        if eps == 0: eps = eps_rel
        plus = base.copy(); plus[i] += eps
        minus = base.copy(); minus[i] -= eps
        Jp = J_at(plus)
        Jm = J_at(minus)
        if not (np.isfinite(Jp) and np.isfinite(Jm)):
            fd_grad[i] = float("nan")
            print(f"  d/d{names[i]:>8}: NaN (Jp={Jp}, Jm={Jm})", flush=True)
            continue
        fd_grad[i] = (Jp - Jm) / (2 * eps)
        print(f"  d/d{names[i]:>8} = {fd_grad[i]:+.6e}  "
              f"(Jp={Jp:.4e}, Jm={Jm:.4e}, eps={eps:.2e})", flush=True)
    t_fd = time.time() - t_fd
    print(f"  FD elapsed: {t_fd:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: Compare
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  {'Param':>10} {'Adjoint':>14} {'FD':>14} {'rel err':>10} {'verdict':>8}")
    pass_all = True
    pass_threshold = 0.05  # 5% tolerance (FD truncation + SNES noise)
    for i, n in enumerate(names):
        a = grad_vals[i]; f = fd_grad[i]
        if not np.isfinite(f):
            verdict = "FD-NAN"
            pass_all = False
            print(f"  {n:>10} {a:+.6e}        NaN       NaN  {verdict:>8}")
            continue
        if abs(f) < 1e-12:
            rel = abs(a - f)
            verdict = "PASS" if rel < 1e-6 else "FAIL"
        else:
            rel = abs(a - f) / abs(f)
            verdict = "PASS" if rel < pass_threshold else "FAIL"
        if verdict == "FAIL":
            pass_all = False
        print(f"  {n:>10} {a:+.6e} {f:+.6e}  {rel:>8.2%}  {verdict:>8}")

    print()
    if pass_all:
        print("OVERALL: PASS — adjoint matches FD; safe to use L-BFGS-B with adjoint gradients")
    else:
        print("OVERALL: FAIL — adjoint disagrees with FD; do NOT use adjoint until fixed")
    print()
    print(f"Speed: 1 adjoint = {t_adj:.1f}s, 8 FD = {t_fd:.1f}s ({t_fd/t_adj:.1f}x)")


if __name__ == "__main__":
    main()
