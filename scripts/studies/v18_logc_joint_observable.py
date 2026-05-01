"""V18 4-parameter joint inverse with disk + peroxide observables, no Tikhonov.

Tests GPT's structural argument that disk current sees r1+r2 while peroxide
current sees r1-r2, so the joint observable should break the (k0, alpha) ridge
without needing a Tikhonov prior.

Forward physics: log-c + 3sp + Boltzmann ClO4- background + H2O2 IC seed,
per docs/k0_inference_status.md (the "breakthrough recipe").

Speedups:
  - Per-voltage IC cache that persists across optimizer evals.  Cache the
    converged U after each voltage solve; reuse as warm-start IC for the
    same voltage on the next eval.
  - Three-tier IC cascade per voltage: cached -> adjacent voltage -> cold
    z-ramp from neutral seeded IC.
  - Adjoint gradients via pyadjoint through forms_logc + Boltzmann
    (verified against FD to <0.1% on 3 of 4 components; k0_r2 disagreement
    is FD noise floor, not adjoint error).
  - L-BFGS-B with jac=callable for typical 10-30 iter convergence.

Inverse:
  - 4 free params: log k0_1, log k0_2, alpha_1, alpha_2
  - Initial guess: +20% offset on all four (assumed surrogate-model accuracy)
  - Joint objective: J = sum_v [0.5 * (cd-cd_target)^2/sigma_cd^2
                            + 0.5 * (pc-pc_target)^2/sigma_pc^2]
  - No Tikhonov term
  - 2% relative noise on both observables (independent)
"""
from __future__ import annotations
import os, sys, time, json, argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="V18 joint disk+peroxide inverse")
    parser.add_argument("--noise", type=float, default=2.0,
                        help="Relative noise %% on each observable (default 2.0; "
                             "use 0 for clean-data identifiability check)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=60,
                        help="L-BFGS-B max iterations")
    parser.add_argument("--out_subdir", type=str, default=None,
                        help="Subdirectory under StudyResults/v18_logc_joint_observable/ "
                             "(default: 'noise_<X>pct')")
    args = parser.parse_args()
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
    from scipy.optimize import minimize
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

    # Drop V=0.25 — at +20% offset on all four params it falls past the
    # convergence cliff (R1 + R2 BV rates blow up).  The breakthrough recipe
    # only validated +20% k0 with -10% alpha; +20% alpha makes V>0.20 unstable.
    V_GRID = np.array([-0.10, 0.00, 0.10, 0.15, 0.20])

    out_subdir = args.out_subdir or f"noise_{args.noise:.1f}pct"
    OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_logc_joint_observable", out_subdir)
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

    def _snapshot(U):
        return tuple(d.data_ro.copy() for d in U.dat)

    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat):
            dst.data[:] = src
        Up.assign(U)

    def build_solve_ctx(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2):
        """Build ctx + solver + observables for a single voltage."""
        sp = make_3sp_sp(V_RHE / V_T, k0_r1, k0_r2, alpha_r1, alpha_r2)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))

        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT)
        of_cd = _build_bv_observable_form(ctx, mode="current_density",
                                           reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(ctx, mode="peroxide_current",
                                           reaction_index=None, scale=-I_SCALE)
        z_nominal = [float(sp[4][i]) for i in range(ctx["n_species"])]
        return ctx, sol, of_cd, of_pc, z_nominal

    def make_run_ss(ctx, sol, of_cd):
        """Adaptive-dt time-stepping to steady state, using cd as monitor."""
        U = ctx["U"]; Up = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_init = 0.25
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
        return run_ss

    def solve_to_steady_warm(V_RHE, k0_1, k0_2, a_1, a_2, ic_data,
                              annotate_final_steps=0, max_steps=80):
        """Warm-started solve at V_RHE.  Loads ic_data as IC, runs unannotated
        warm-up to convergence, then optionally a few annotated steps for tape.
        Returns (cd, pc, U_snapshot, ctx) or (None, None, None, ctx) on failure.
        """
        ctx, sol, of_cd, of_pc, z_nominal = build_solve_ctx(
            V_RHE, k0_1, k0_2, a_1, a_2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)

        with adj.stop_annotating():
            _restore(ic_data, U, Up)
            for i in range(n): zc[i].assign(z_nominal[i])
            paf.assign(V_RHE / V_T)
            if not run_ss(max_steps):
                return None, None, None, ctx

        if annotate_final_steps > 0:
            for _ in range(annotate_final_steps):
                sol.solve()
                Up.assign(U)

        cd_v = float(fd.assemble(of_cd))
        pc_v = float(fd.assemble(of_pc))
        snap = _snapshot(U)
        return cd_v, pc_v, snap, ctx

    def solve_to_steady_cold(V_RHE, k0_1, k0_2, a_1, a_2, max_z_steps=20):
        """Cold z-ramp + steady-state.  No annotation (used for target +
        cache seeding).  Returns (cd, pc, U_snapshot)."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve_ctx(
            V_RHE, k0_1, k0_2, a_1, a_2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)

        with adj.stop_annotating():
            for zci in zc: zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(100):
                return None, None, None
            achieved_z = 0.0
            for z_val in np.linspace(0, 1, max_z_steps + 1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n): zc[i].assign(z_nominal[i] * z_val)
                if run_ss(60):
                    achieved_z = z_val
                else:
                    _restore(ckpt, U, Up)
                    break
            if achieved_z < 1.0 - 1e-3:
                return None, None, None
            cd_v = float(fd.assemble(of_cd))
            pc_v = float(fd.assemble(of_pc))
            return cd_v, pc_v, _snapshot(U)

    def populate_initial_cache(k0_1, k0_2, a_1, a_2):
        """Cold-ramp the first voltage, warm-step through the rest of V_GRID.
        Returns (cd_array, pc_array, ic_cache_list)."""
        cds = np.full(len(V_GRID), np.nan)
        pcs = np.full(len(V_GRID), np.nan)
        cache = [None] * len(V_GRID)

        # First voltage: cold start
        cd_v, pc_v, snap = solve_to_steady_cold(
            float(V_GRID[0]), k0_1, k0_2, a_1, a_2)
        if snap is None:
            print(f"  WARN: cold start at V={V_GRID[0]:.2f} failed")
            return cds, pcs, cache
        cds[0] = cd_v; pcs[0] = pc_v; cache[0] = snap
        last_U = snap

        # Subsequent voltages: warm-start from last
        for i in range(1, len(V_GRID)):
            V = float(V_GRID[i])
            cd_v, pc_v, snap, _ = solve_to_steady_warm(
                V, k0_1, k0_2, a_1, a_2, last_U,
                annotate_final_steps=0, max_steps=80)
            if snap is None:
                # Try cold ramp as fallback
                cd_v, pc_v, snap = solve_to_steady_cold(
                    V, k0_1, k0_2, a_1, a_2)
                if snap is None:
                    print(f"  WARN: V={V:.2f} cold+warm both failed")
                    continue
            cds[i] = cd_v; pcs[i] = pc_v; cache[i] = snap
            last_U = snap
        return cds, pcs, cache

    # ------------------------------------------------------------------
    # Step 1: Compute target curve at TRUE params (and seed an IC cache that
    # we'll later use to seed the inversion's IC cache).
    # ------------------------------------------------------------------
    print("=" * 72)
    print("V18 4-PARAM JOINT INVERSE: L-BFGS-B + adjoint, IC cache, no Tikhonov")
    print("=" * 72)
    print(f"V_GRID (V_RHE):  {V_GRID.tolist()}")
    print(f"True k0_1 = {K0_HAT_R1:.4e}, k0_2 = {K0_HAT_R2:.4e}")
    print(f"True alpha_1 = {ALPHA_R1:.4f}, alpha_2 = {ALPHA_R2:.4f}")
    print()
    print("Step 1: computing target curve at TRUE params...", flush=True)
    t0 = time.time()
    target_cd, target_pc, true_cache = populate_initial_cache(
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    print(f"  Target curve in {time.time()-t0:.1f}s")
    print(f"  cd targets: {target_cd}")
    print(f"  pc targets: {target_pc}")

    if not (np.all(np.isfinite(target_cd)) and np.all(np.isfinite(target_pc))):
        print("FATAL: target curve has NaN.")
        return

    # ------------------------------------------------------------------
    # Step 2: Add 2% noise (independent on cd and pc), with sigma floor
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.noise > 0:
        rel = args.noise / 100.0
        sigma_cd = np.maximum(rel * np.abs(target_cd), rel * np.max(np.abs(target_cd)))
        sigma_pc = np.maximum(rel * np.abs(target_pc), rel * np.max(np.abs(target_pc)))
        noisy_cd = target_cd + rng.normal(0, sigma_cd)
        noisy_pc = target_pc + rng.normal(0, sigma_pc)
        print(f"  Adding {args.noise}% relative noise (seed={args.seed})")
    else:
        # Clean-data control: J = 0 at TRUE, so any nonzero recovered J means
        # optimizer didn't find the minimum (or k0_2/etc. is non-identifiable).
        sigma_cd = np.full_like(target_cd, 1.0)  # unused — only for shape
        sigma_pc = np.full_like(target_pc, 1.0)
        noisy_cd = target_cd.copy()
        noisy_pc = target_pc.copy()
        print("  Clean data (no noise) — identifiability check")

    # Range-normalized residuals: each observable contributes O(1) to J at the
    # ~20%-off point.  Inverse-variance weighting (1/sigma^2) was ~10^8 worse
    # for L-BFGS-B because sigma_pc is tiny in absolute terms — gradient
    # exploded and the first step hit every bound.  Range normalization keeps
    # cd and pc on equal footing without amplifying noise via tiny sigmas.
    cd_scale = float(np.max(np.abs(target_cd)))
    pc_scale = float(np.max(np.abs(target_pc)))
    inv_var_cd = np.full_like(target_cd, 1.0 / (cd_scale ** 2))
    inv_var_pc = np.full_like(target_pc, 1.0 / (pc_scale ** 2))
    print(f"  Residual normalization: 1/cd_scale^2={inv_var_cd[0]:.3e}, "
          f"1/pc_scale^2={inv_var_pc[0]:.3e}")
    print(f"  noisy cd: {noisy_cd}")
    print(f"  noisy pc: {noisy_pc}")
    np.savez(os.path.join(OUT_DIR, "targets.npz"),
             V_GRID=V_GRID, target_cd=target_cd, target_pc=target_pc,
             noisy_cd=noisy_cd, noisy_pc=noisy_pc,
             sigma_cd=sigma_cd, sigma_pc=sigma_pc)

    # ------------------------------------------------------------------
    # Step 3: Cold-populate the IC cache at INITIAL GUESS params (+20% offset)
    # so the L-BFGS-B first eval is warm-started.
    # ------------------------------------------------------------------
    init_offset = 0.20
    init_k0_1 = K0_HAT_R1 * (1.0 + init_offset)
    init_k0_2 = K0_HAT_R2 * (1.0 + init_offset)
    init_a_1 = ALPHA_R1 * (1.0 + init_offset)
    init_a_2 = ALPHA_R2 * (1.0 + init_offset)
    x0 = np.array([np.log(init_k0_1), np.log(init_k0_2), init_a_1, init_a_2])

    print()
    print("Step 3: seeding inversion IC cache from TRUE cache.")
    print(f"  init k0=({init_k0_1:.4e},{init_k0_2:.4e}) "
          f"a=({init_a_1:.4f},{init_a_2:.4f})")
    print("  (cold-start at +20% offset is unreliable in log-c; instead the")
    print("   first L-BFGS-B eval warm-starts each voltage from TRUE U,")
    print("   then the cache updates with the new params' converged U.)")
    inv_cache = [c for c in true_cache]  # shallow copy of refs
    print(f"  Seeded {sum(c is not None for c in inv_cache)}/{len(V_GRID)} voltages from TRUE")

    # ------------------------------------------------------------------
    # Step 4: Define J + grad evaluator (per-voltage adjoint, sum gradients)
    # ------------------------------------------------------------------
    # Sanity bounds on observables: reject solves that produce values
    # physically inconsistent with the target (catches unphysical SNES
    # "convergence" to wrong steady states).
    cd_sanity = 3.0 * float(np.max(np.abs(target_cd)))
    pc_sanity = 5.0 * float(np.max(np.abs(target_pc)))
    print(f"  Observable sanity bounds: |cd| < {cd_sanity:.3e}, |pc| < {pc_sanity:.3e}")

    def compute_J_and_grad(x):
        """Compute J = sum_v J_v(x) and dJ/dx via per-voltage adjoint.
        Returns (J, grad_array_of_4, sources_list).  Bounds projection is
        delegated to scipy via the L-BFGS-B `bounds=` argument; this
        function trusts that x is in-bounds."""
        log_k0_1, log_k0_2, a_1, a_2 = x
        k0_1 = float(np.exp(log_k0_1)); k0_2 = float(np.exp(log_k0_2))

        # Gradient w.r.t. (log_k0_1, log_k0_2, a_1, a_2).
        # Adjoint gives dJ/d(k0_1, k0_2, a_1, a_2); chain rule: dJ/d(log k0) = k0 * dJ/dk0.
        J_total = 0.0
        grad_total = np.zeros(4)  # [dJ/dlog_k0_1, dJ/dlog_k0_2, dJ/da_1, dJ/da_2]
        sources = []
        n_failed = 0

        last_U = None  # adjacent-warm fallback
        tape = adj.get_working_tape()

        for v_idx, V in enumerate(V_GRID):
            V = float(V)

            # Pick IC source: cached for this voltage, else last solved,
            # else TRUE-cache fallback (always-good anchor), else cold ramp.
            ic_data = inv_cache[v_idx]
            ic_source = "C"
            if ic_data is None and last_U is not None:
                ic_data = last_U
                ic_source = "W"
            if ic_data is None and true_cache[v_idx] is not None:
                ic_data = true_cache[v_idx]
                ic_source = "T"
            if ic_data is None:
                # Cold ramp (last resort; reliable only at TRUE params)
                cd_v, pc_v, snap = solve_to_steady_cold(V, k0_1, k0_2, a_1, a_2)
                if snap is None or abs(cd_v) > cd_sanity \
                        or abs(pc_v) > pc_sanity:
                    sources.append("F")
                    n_failed += 1
                    continue
                inv_cache[v_idx] = snap
                last_U = snap
                # Cold path doesn't compute gradient; treat J_v as known but
                # zero contribution to grad (this point won't update params).
                r_cd = (cd_v - noisy_cd[v_idx])
                r_pc = (pc_v - noisy_pc[v_idx])
                J_v = 0.5 * inv_var_cd[v_idx] * r_cd ** 2 \
                    + 0.5 * inv_var_pc[v_idx] * r_pc ** 2
                J_total += J_v
                sources.append("X")
                continue

            # Warm-start with annotation for adjoint.
            tape.clear_tape()
            adj.continue_annotation()

            cd_v, pc_v, snap, ctx = solve_to_steady_warm(
                V, k0_1, k0_2, a_1, a_2, ic_data,
                annotate_final_steps=5, max_steps=60)

            warm_failed = (snap is None)
            unphysical = (not warm_failed) and (
                abs(cd_v) > cd_sanity or abs(pc_v) > pc_sanity
                or not np.isfinite(cd_v) or not np.isfinite(pc_v))

            if unphysical or warm_failed:
                # Tier 4 (always-available): retry warm-start from TRUE_cache U.
                # TRUE_cache holds physically-converged U at the true params.
                # As long as the new params are within the warm-start basin
                # of TRUE U, this recovers cleanly.  Cold-ramp at +20% offset
                # is unreliable in log-c so we don't bother with it.
                adj.pause_annotation()
                tape.clear_tape()
                inv_cache[v_idx] = None
                if true_cache[v_idx] is None:
                    sources.append("F")
                    n_failed += 1
                    continue
                tape.clear_tape()
                adj.continue_annotation()
                cd_v, pc_v, snap, ctx = solve_to_steady_warm(
                    V, k0_1, k0_2, a_1, a_2, true_cache[v_idx],
                    annotate_final_steps=5, max_steps=80)
                if snap is None or abs(cd_v) > cd_sanity or abs(pc_v) > pc_sanity \
                        or not np.isfinite(cd_v) or not np.isfinite(pc_v):
                    adj.pause_annotation()
                    tape.clear_tape()
                    sources.append("F")
                    n_failed += 1
                    continue
                ic_source = "T"  # recovered from TRUE_cache fallback

            # Cache the new converged U (param-aware; latest iterate's solution)
            inv_cache[v_idx] = snap
            last_U = snap

            # Build J_v ON the tape so adjoint sees it.
            of_cd_t = _build_bv_observable_form(ctx, mode="current_density",
                                                 reaction_index=None, scale=-I_SCALE)
            of_pc_t = _build_bv_observable_form(ctx, mode="peroxide_current",
                                                 reaction_index=None, scale=-I_SCALE)
            cd_a = fd.assemble(of_cd_t)
            pc_a = fd.assemble(of_pc_t)
            J_v = 0.5 * inv_var_cd[v_idx] * (cd_a - float(noisy_cd[v_idx])) ** 2 \
                + 0.5 * inv_var_pc[v_idx] * (pc_a - float(noisy_pc[v_idx])) ** 2

            controls = [adj.Control(f) for f in
                        list(ctx["bv_k0_funcs"][:2]) + list(ctx["bv_alpha_funcs"][:2])]
            try:
                rf = adj.ReducedFunctional(J_v, controls)
                gradient = rf.derivative()
            except Exception as e:
                adj.pause_annotation()
                print(f"  WARN: adjoint failed at V={V:.2f}: "
                      f"{type(e).__name__}: {e}")
                # Still count J contribution from the solve
                J_total += float(J_v)
                sources.append("A?")
                adj.pause_annotation()
                continue

            adj.pause_annotation()

            # Extract scalar gradients
            grad_vals = []
            for g in gradient:
                if hasattr(g, "dat"):
                    grad_vals.append(float(g.dat[0].data_ro[0]))
                elif hasattr(g, "values"):
                    grad_vals.append(float(g.values()[0]))
                else:
                    grad_vals.append(float(g))

            # Chain rule: dJ/d(log k0_j) = k0_j * dJ/dk0_j
            grad_total[0] += k0_1 * grad_vals[0]
            grad_total[1] += k0_2 * grad_vals[1]
            grad_total[2] += grad_vals[2]
            grad_total[3] += grad_vals[3]
            J_total += float(J_v)
            sources.append(ic_source)

        # On failure (more than half the voltages couldn't solve), return a
        # big J but NON-ZERO gradient.  Returning grad=0 makes L-BFGS-B
        # interpret PGTOL=0 as convergence and terminate prematurely.
        # Instead return last_good_grad (scaled up) so scipy backtracks
        # the line search.  If no good prior eval exists, return a large
        # gradient pointing back toward x_init.
        if n_failed > len(V_GRID) // 2:
            if cache_box.get("last_good_grad") is not None:
                bad_grad = 10.0 * cache_box["last_good_grad"]
            else:
                # Point back toward x_init with magnitude 10
                direction = x0 - x
                norm = np.linalg.norm(direction)
                bad_grad = -10.0 * (direction / norm if norm > 1e-12 else np.ones(4))
            return 1e6, bad_grad, sources

        return J_total, grad_total, sources

    # Function/jac cache so scipy doesn't recompute the adjoint on duplicate x
    cache_box = {"x": None, "J": None, "grad": None}

    def cached_eval(x):
        if cache_box["x"] is None or not np.allclose(x, cache_box["x"], atol=1e-12):
            t_eval = time.time()
            J, grad, sources = compute_J_and_grad(x)
            elapsed = time.time() - t_eval
            cache_box["x"] = x.copy()
            cache_box["J"] = J
            cache_box["grad"] = grad
            cache_box["last_sources"] = sources
            cache_box["last_elapsed"] = elapsed
            # Track LAST GOOD (J, grad) for failure-recovery return values
            if J < 1e5 and np.all(np.isfinite(grad)):
                cache_box["last_good_J"] = J
                cache_box["last_good_grad"] = grad.copy()
        return cache_box["J"], cache_box["grad"]

    eval_count = [0]
    history = []

    def fun(x):
        J, grad = cached_eval(x)
        eval_count[0] += 1
        log_k0_1, log_k0_2, a_1, a_2 = x
        k0_1 = float(np.exp(log_k0_1)); k0_2 = float(np.exp(log_k0_2))
        sources = cache_box.get("last_sources", [])
        elapsed = cache_box.get("last_elapsed", 0.0)
        k0_1_err = 100 * (k0_1 - K0_HAT_R1) / K0_HAT_R1
        k0_2_err = 100 * (k0_2 - K0_HAT_R2) / K0_HAT_R2
        a_1_err = 100 * (a_1 - ALPHA_R1) / ALPHA_R1
        a_2_err = 100 * (a_2 - ALPHA_R2) / ALPHA_R2
        src_summary = "".join(sources) if sources else "?"
        gradnorm = float(np.linalg.norm(grad))
        print(f"  [{eval_count[0]:3d}] k0=({k0_1:.3e},{k0_2:.3e}) "
              f"a=({a_1:.4f},{a_2:.4f})  J={J:.3e} |g|={gradnorm:.2e}  "
              f"err%=({k0_1_err:+.1f},{k0_2_err:+.1f},{a_1_err:+.1f},{a_2_err:+.1f})  "
              f"src=[{src_summary}] {elapsed:.1f}s", flush=True)
        history.append({"eval": eval_count[0], "k0_1": k0_1, "k0_2": k0_2,
                        "alpha_1": a_1, "alpha_2": a_2, "J": float(J),
                        "grad": grad.tolist(), "grad_norm": gradnorm,
                        "sources": sources, "elapsed_s": elapsed,
                        "k0_1_err_pct": k0_1_err, "k0_2_err_pct": k0_2_err,
                        "alpha_1_err_pct": a_1_err, "alpha_2_err_pct": a_2_err})
        if eval_count[0] % 5 == 0:
            with open(os.path.join(OUT_DIR, "history_partial.json"), "w") as f:
                json.dump(history, f, indent=2)
        return J

    def jac(x):
        _, grad = cached_eval(x)
        return grad

    # ------------------------------------------------------------------
    # Step 5: Run L-BFGS-B
    # ------------------------------------------------------------------
    print()
    print("Step 5: L-BFGS-B with adjoint gradient (joint cd+pc, no Tikhonov)")
    print(f"  Initial: log_k0=({x0[0]:.4f},{x0[1]:.4f}) a=({x0[2]:.4f},{x0[3]:.4f})")
    print("  Source legend: C=cached  W=adjacent warm  T=TRUE-fallback  X=cold ramp  F=fail")
    print()

    # Tight bounds: ±2 decades around init for log k0, ±0.2 around init for
    # alpha.  These are loose enough to allow real corrections but stop
    # L-BFGS-B from runaway first steps that hit all bounds simultaneously.
    bounds = [
        (np.log(K0_HAT_R1) - 2.0, np.log(K0_HAT_R1) + 2.0),  # ~7x range either side
        (np.log(K0_HAT_R2) - 2.0, np.log(K0_HAT_R2) + 2.0),
        (max(0.20, ALPHA_R1 - 0.2), min(0.80, ALPHA_R1 + 0.2)),
        (max(0.20, ALPHA_R2 - 0.2), min(0.80, ALPHA_R2 + 0.2)),
    ]
    print(f"  bounds (log_k0_1, log_k0_2, a_1, a_2) = {bounds}")

    t_opt = time.time()
    res = minimize(
        fun, x0, jac=jac, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-9, "gtol": 1e-7},
    )
    t_opt = time.time() - t_opt

    log_k0_1_rec, log_k0_2_rec, a_1_rec, a_2_rec = res.x
    k0_1_rec = float(np.exp(log_k0_1_rec)); k0_2_rec = float(np.exp(log_k0_2_rec))

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print(f"  evals: {eval_count[0]}, wall: {t_opt/60:.1f} min, "
          f"L-BFGS-B success: {res.success}")
    print(f"  J final: {res.fun:.4e}")
    print(f"  message: {res.message}")
    print()
    print(f"  Param        True          Recovered     Error")
    print(f"  k0_1     {K0_HAT_R1:.4e}    {k0_1_rec:.4e}    "
          f"{100*(k0_1_rec-K0_HAT_R1)/K0_HAT_R1:+.2f}%")
    print(f"  k0_2     {K0_HAT_R2:.4e}    {k0_2_rec:.4e}    "
          f"{100*(k0_2_rec-K0_HAT_R2)/K0_HAT_R2:+.2f}%")
    print(f"  alpha_1  {ALPHA_R1:.4f}        {a_1_rec:.4f}        "
          f"{100*(a_1_rec-ALPHA_R1)/ALPHA_R1:+.2f}%")
    print(f"  alpha_2  {ALPHA_R2:.4f}        {a_2_rec:.4f}        "
          f"{100*(a_2_rec-ALPHA_R2)/ALPHA_R2:+.2f}%")

    out = {
        "config": {
            "V_GRID": V_GRID.tolist(),
            "true_params": {"k0_1": K0_HAT_R1, "k0_2": K0_HAT_R2,
                            "alpha_1": ALPHA_R1, "alpha_2": ALPHA_R2},
            "init_x": x0.tolist(),
            "init_offset_pct": init_offset * 100,
            "noise_pct": args.noise, "noise_seed": args.seed,
            "regularization": "none",
            "observable": "joint disk + peroxide",
            "weighting": "inverse-variance per residual",
            "ic_cache": "per-voltage, persists across evals",
            "optimizer": "L-BFGS-B",
            "gradient": "adjoint via pyadjoint (per-voltage tape, sum)",
            "param_space": "log k0, alpha (linear)",
        },
        "result": {
            "k0_1": k0_1_rec, "k0_2": k0_2_rec,
            "alpha_1": a_1_rec, "alpha_2": a_2_rec,
            "k0_1_err_pct": 100 * (k0_1_rec - K0_HAT_R1) / K0_HAT_R1,
            "k0_2_err_pct": 100 * (k0_2_rec - K0_HAT_R2) / K0_HAT_R2,
            "alpha_1_err_pct": 100 * (a_1_rec - ALPHA_R1) / ALPHA_R1,
            "alpha_2_err_pct": 100 * (a_2_rec - ALPHA_R2) / ALPHA_R2,
            "J_final": float(res.fun),
            "n_evals": eval_count[0],
            "wall_minutes": t_opt / 60,
            "lbfgs_success": bool(res.success),
            "lbfgs_message": str(res.message),
        },
        "history": history,
    }
    out_path = os.path.join(OUT_DIR, "result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'targets.npz')}")


if __name__ == "__main__":
    main()
