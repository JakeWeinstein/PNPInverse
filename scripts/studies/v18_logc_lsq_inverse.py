"""V18 4-param joint inverse via scipy.optimize.least_squares + per-observable adjoint Jacobian.

Phase A diagnostics established:
  - Adding peroxide does increase Fisher rank by ~7 OOM (sv_min: 4.5e-8 -> 3e-2).
  - Clean-data L-BFGS-B endpoint had a MONOTONE descending path to TRUE,
    so the previous "non-identifiability" result was actually OPTIMIZER FAILURE.

This script tests whether a sum-of-squares-aware optimizer can navigate the
heavily ill-conditioned landscape (cond F ~ 1e11 even with PC):
  - scipy.optimize.least_squares with method='trf' (trust-region reflective,
    supports bounds, handles ill-conditioning naturally via LM damping).
  - Residual vector: r = [(cd_v - cd_target_v)/sigma_cd, (pc_v - pc_target_v)/sigma_pc] for v in V_GRID
  - Jacobian: per voltage, one forward solve + TWO adjoint solves (cd and pc),
    producing rows of J that respect the sum-of-squares structure exactly.
  - Per GPT's correction: use noise-whitened residuals (sigma_cd, sigma_pc),
    not range-normalized.  This preserves statistical meaning.
  - No Tikhonov.

Initial guess: +20% offset on all 4 params.
Forward: log-c + 3sp + Boltzmann + H2O2 IC seed, V_GRID = [-0.10, 0.20] V_RHE.
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
    parser = argparse.ArgumentParser(description="V18 LSQ inverse with adjoint Jacobian")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Relative noise %% (default 0 = clean identifiability test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["trf", "lm"], default="trf",
                        help="trf=trust-region (bounds-aware); lm=Levenberg-Marquardt")
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--precision", choices=["standard", "tight"], default="standard",
                        help="standard: snes_rtol=1e-10, ss_rtol=1e-4. "
                             "tight: snes_rtol=1e-13, ss_rtol=1e-6 (3-5x slower)")
    parser.add_argument("--mesh_ny", type=int, default=200,
                        help="Mesh Ny (default 200; 400 for finer with 4x runtime)")
    parser.add_argument("--forward_diag_only", action="store_true",
                        help="Only run forward TRUE solve at this precision; skip inverse")
    parser.add_argument("--out_subdir", type=str, default=None)
    parser.add_argument("--log-rate", action="store_true",
                        help="Stage 2: enable bv_log_rate=True in BV evaluation.")
    parser.add_argument("--v-grid", nargs="+", type=float, default=None,
                        help="Override V_GRID. Default: -0.10 0.0 0.10 0.15 0.20.")
    parser.add_argument("--init", choices=[
        "plus20", "minus20", "k0high_alow", "k0low_ahigh", "true"
    ], default="plus20",
                        help="Initial guess pattern (clean-data multi-init test).")
    parser.add_argument("--out-base", type=str,
                        default="v18_logc_lsq_inverse",
                        help="Top-level results subdir name.")
    parser.add_argument("--prior-sigma-log-k0", type=float, default=0.0,
                        help="Tikhonov prior std on log(k0_j). 0 (default) "
                             "= no prior. Suggested: log(3)~1.099 for "
                             "factor-of-3 uncertainty; log(10)~2.303 for "
                             "factor-of-10. Adds 2 prior residual rows: "
                             "(log_k0_j - log_k0_prior_center_j)/sigma.")
    parser.add_argument("--prior-center", choices=["true"], default="true",
                        help="Prior center: 'true' = literature/EIS "
                             "measurement at the actual k0 values.")
    parser.add_argument("--start-theta", nargs=4, type=float, default=None,
                        metavar=("LOG_K0_1", "LOG_K0_2", "ALPHA_1", "ALPHA_2"),
                        help="Override --init with explicit starting theta. "
                             "If --anchor-tafel is set, --start-theta is "
                             "interpreted as PHYSICAL coords (log_k0, alpha) "
                             "and transformed to (beta, alpha) internally.")
    parser.add_argument("--anchor-tafel", action="store_true",
                        help="Optimize in anchored Tafel coords (beta_j, "
                             "alpha_j) where beta_j = log_k0_j - alpha_j * "
                             "n_e * (V_anchor_j - E_eq_j) / V_T. The forward "
                             "solver still uses (log_k0_j, alpha_j); the "
                             "transform is applied at the optimizer interface.")
    parser.add_argument("--v-anchor-1", type=float, default=0.30,
                        help="V_RHE anchor for R1 (default 0.30 V).")
    parser.add_argument("--v-anchor-2", type=float, default=0.50,
                        help="V_RHE anchor for R2 (default 0.50 V).")
    parser.add_argument("--c-o2-hat", type=float, default=None,
                        help="Override bulk c_O2 (in HAT/normalized units; "
                             "1.0 = C_O2_HAT default).  Used for "
                             "multi-experiment FIM design with O2 variation.")
    parser.add_argument("--save-fim-at-true", action="store_true",
                        help="Skip the inverse; just compute and save the "
                             "Jacobian (sensitivity matrix) at TRUE params "
                             "with TRUE-cache as IC.  Output: fim_at_true.json "
                             "with S_cd_raw, S_pc_raw, observables, sigma.")
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
    from scipy.optimize import least_squares
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
    C_O2_HAT_USED = C_O2_HAT if args.c_o2_hat is None else float(args.c_o2_hat)
    THREE_SPECIES_C0 = [C_O2_HAT_USED, H2O2_SEED, C_HP_HAT]
    if args.c_o2_hat is not None:
        print(f"  --c-o2-hat: bulk c_O2 = {C_O2_HAT_USED:.4f} (HAT)")
    V_GRID = (np.array(args.v_grid, dtype=float) if args.v_grid is not None
              else np.array([-0.10, 0.00, 0.10, 0.15, 0.20]))
    NV = len(V_GRID)
    PARAM_NAMES = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]
    USE_LOG_RATE = bool(args.log_rate)
    USE_PRIOR = args.prior_sigma_log_k0 > 0
    SIGMA_LOG_K0 = float(args.prior_sigma_log_k0) if USE_PRIOR else None
    LOG_K0_PRIOR_CENTER = (np.array([np.log(K0_HAT_R1), np.log(K0_HAT_R2)])
                           if USE_PRIOR else None)
    USE_ANCHOR = bool(args.anchor_tafel)
    if USE_ANCHOR:
        # Per HANDOFF_11 §4.1: log_rate_j ≈ log_k0_j - alpha_j*n_e*(V-E_eq)/V_T.
        # Anchor at V_anchor_j folds the alpha-induced shift at that voltage
        # into beta_j, decoupling beta from alpha within the anchor's
        # neighborhood.  The forward solver still takes log_k0; we transform
        # at the optimizer interface.
        C_ANCHOR_1 = float(N_ELECTRONS) * (args.v_anchor_1 - E_EQ_R1) / V_T
        C_ANCHOR_2 = float(N_ELECTRONS) * (args.v_anchor_2 - E_EQ_R2) / V_T
        # x_phys = M_ANC @ x_anc; M_ANC[i,j] = ∂phys_i/∂anc_j.
        M_ANC = np.array([
            [1.0, 0.0, C_ANCHOR_1, 0.0],
            [0.0, 1.0, 0.0,        C_ANCHOR_2],
            [0.0, 0.0, 1.0,        0.0],
            [0.0, 0.0, 0.0,        1.0],
        ])
        print(f"  Anchor: V_anchor=({args.v_anchor_1:.2f}, {args.v_anchor_2:.2f}), "
              f"C=({C_ANCHOR_1:.3f}, {C_ANCHOR_2:.3f})")
    else:
        C_ANCHOR_1 = C_ANCHOR_2 = 0.0
        M_ANC = np.eye(4)

    def _phys_from_x(x):
        """Map optimizer x → (log_k0_1, log_k0_2, alpha_1, alpha_2)."""
        if USE_ANCHOR:
            return M_ANC @ np.asarray(x, dtype=float)
        return np.asarray(x, dtype=float)

    def _x_from_phys(phys):
        """Inverse map physical → optimizer coords."""
        if USE_ANCHOR:
            log_k0_1, log_k0_2, a_1, a_2 = phys
            return np.array([log_k0_1 - a_1 * C_ANCHOR_1,
                             log_k0_2 - a_2 * C_ANCHOR_2,
                             a_1, a_2])
        return np.asarray(phys, dtype=float)

    out_subdir = args.out_subdir or (
        f"lsq_{args.method}_noise_{args.noise:.1f}pct_prec{args.precision}"
        f"_ny{args.mesh_ny}_init{args.init}"
        f"{'_lograte' if USE_LOG_RATE else ''}"
        f"{f'_priorsig{SIGMA_LOG_K0:.3f}' if USE_PRIOR else ''}"
        f"{f'_anc{args.v_anchor_1:.2f}_{args.v_anchor_2:.2f}' if USE_ANCHOR else ''}")
    OUT_DIR = os.path.join(_ROOT, "StudyResults", args.out_base, out_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    # Apply precision overrides
    if args.precision == "tight":
        SP_DICT["snes_atol"] = 1e-10  # was 1e-7
        SP_DICT["snes_rtol"] = 1e-13  # was 1e-10
        SP_DICT["snes_stol"] = 1e-14  # was 1e-12
        SP_DICT["snes_max_it"] = 600  # tighter tols may need more iters
        ss_rel_tol = 1e-6  # was 1e-4
        ss_abs_tol = 1e-10  # was 1e-8
        ss_consec = 5  # was 4
    else:
        ss_rel_tol = 1e-4
        ss_abs_tol = 1e-8
        ss_consec = 4
    print(f"  precision={args.precision}: snes_rtol={SP_DICT.get('snes_rtol')}, "
          f"snes_atol={SP_DICT.get('snes_atol')}, ss_rel={ss_rel_tol}, mesh_ny={args.mesh_ny}")

    def make_3sp_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = _make_bv_convergence_cfg(log_rate=USE_LOG_RATE)
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

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(args.mesh_ny), beta=3.0)

    def _snapshot(U): return tuple(d.data_ro.copy() for d in U.dat)
    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat): dst.data[:] = src
        Up.assign(U)

    def build_solve(V_RHE, k0_1, k0_2, a_1, a_2):
        sp = make_3sp_sp(V_RHE / V_T, k0_1, k0_2, a_1, a_2)
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
        U = ctx["U"]; Up = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_init = 0.25
        def run_ss(max_steps):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for s in range(1, max_steps + 1):
                try: sol.solve()
                except Exception: return False
                Up.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    d = abs(fv - prev_flux); sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
                    if d/sv <= ss_rel_tol or d <= ss_abs_tol: sc += 1
                    else: sc = 0
                    if prev_delta and d > 0:
                        r = prev_delta/d
                        dt_val = (min(dt_val*min(r,4), dt_init*20)
                                  if r > 1 else max(dt_val*0.5, dt_init))
                        dt_const.assign(dt_val)
                    prev_delta = d
                prev_flux = fv
                if sc >= ss_consec: return True
            return False
        return run_ss

    # Bumped max_steps from 120 to 250 to handle tight-precision SS convergence
    # (rel tol 1e-6 takes ~2x more steps than 1e-4 to reach).
    def solve_warm_annotated(V_RHE, k0_1, k0_2, a_1, a_2, ic_data,
                              annotate_final_steps=5, max_steps=250):
        """Warm-start solve at V_RHE, annotated for adjoint.
        Returns (cd_assembled, pc_assembled, snap, ctx) where assembled
        values are AdjFloat (for adjoint) and snap is a numpy snapshot."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_RHE, k0_1, k0_2, a_1, a_2)
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
        # Final annotated steps
        for _ in range(annotate_final_steps):
            sol.solve()
            Up.assign(U)
        cd_a = fd.assemble(of_cd); pc_a = fd.assemble(of_pc)
        return cd_a, pc_a, _snapshot(U), ctx

    def solve_warm_unann(V_RHE, k0_1, k0_2, a_1, a_2, ic_data, max_steps=250):
        """Warm-start solve, no annotation. Returns (cd_float, pc_float, snap)."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_RHE, k0_1, k0_2, a_1, a_2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            _restore(ic_data, U, Up)
            for i in range(n): zc[i].assign(z_nominal[i])
            paf.assign(V_RHE / V_T)
            if not run_ss(max_steps):
                return None, None, None
        return float(fd.assemble(of_cd)), float(fd.assemble(of_pc)), _snapshot(U)

    def solve_cold(V_RHE, k0_1, k0_2, a_1, a_2, max_z_steps=20):
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_RHE, k0_1, k0_2, a_1, a_2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            for zci in zc: zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(200): return None, None, None
            achieved_z = 0.0
            for z_val in np.linspace(0, 1, max_z_steps+1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n): zc[i].assign(z_nominal[i] * z_val)
                if run_ss(120): achieved_z = z_val
                else:
                    _restore(ckpt, U, Up); break
            if achieved_z < 1.0 - 1e-3: return None, None, None
        return float(fd.assemble(of_cd)), float(fd.assemble(of_pc)), _snapshot(U)

    # ----------------------------------------------------------------
    # Step 1: TRUE curve (cold, with cold-fallback per-V)
    # ----------------------------------------------------------------
    print("=" * 72)
    print(f"V18 LSQ inverse: method={args.method}, noise={args.noise}%")
    print("=" * 72)
    print(f"V_GRID: {V_GRID.tolist()}")
    print()
    print("Step 1: TRUE curve (cold-with-fallback)...", flush=True)
    t0 = time.time()
    true_cache = [None] * NV
    target_cd = np.zeros(NV); target_pc = np.zeros(NV)
    for i, V in enumerate(V_GRID):
        t_v = time.time()
        cd, pc, snap = solve_cold(float(V), K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        if snap is None:
            print(f"  FATAL: cold solve at TRUE failed at V={V}")
            return
        target_cd[i] = cd; target_pc[i] = pc; true_cache[i] = snap
        print(f"  V={V:+.2f}: cd={cd:+.6e}, pc={pc:+.6e}  ({time.time()-t_v:.1f}s)", flush=True)
    print(f"  TRUE curve in {time.time()-t0:.1f}s")

    # Save TRUE curve at this precision so we can diff vs other precision runs
    np.savez(os.path.join(OUT_DIR, "true_curve.npz"),
             V_GRID=V_GRID, target_cd=target_cd, target_pc=target_pc,
             precision=args.precision, mesh_ny=args.mesh_ny)
    if args.forward_diag_only:
        print(f"\n  --forward_diag_only set; skipping inverse.")
        print(f"  Saved: {os.path.join(OUT_DIR, 'true_curve.npz')}")
        return

    # ----------------------------------------------------------------
    # Step 2: noise + sigma
    # ----------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.noise > 0:
        rel = args.noise / 100.0
        sigma_cd = np.maximum(rel * np.abs(target_cd), rel * np.max(np.abs(target_cd)))
        sigma_pc = np.maximum(rel * np.abs(target_pc), rel * np.max(np.abs(target_pc)))
        noisy_cd = target_cd + rng.normal(0, sigma_cd)
        noisy_pc = target_pc + rng.normal(0, sigma_pc)
    else:
        # For LSQ we need a sigma to whiten by even with clean data.
        # Use the noise-equivalent: 2% of max as a conservative scale.
        sigma_cd = np.full_like(target_cd, 0.02 * float(np.max(np.abs(target_cd))))
        sigma_pc = np.full_like(target_pc, 0.02 * float(np.max(np.abs(target_pc))))
        noisy_cd = target_cd.copy()
        noisy_pc = target_pc.copy()
        print("  Clean data; whitening sigma set to 2%*max for numerical conditioning")

    print(f"  sigma_cd: {sigma_cd}")
    print(f"  sigma_pc: {sigma_pc}")
    np.savez(os.path.join(OUT_DIR, "targets.npz"),
             V_GRID=V_GRID, target_cd=target_cd, target_pc=target_pc,
             noisy_cd=noisy_cd, noisy_pc=noisy_pc,
             sigma_cd=sigma_cd, sigma_pc=sigma_pc)

    # ----------------------------------------------------------------
    # Step 2.5: cold-solve at INIT parameters → init_cache
    # ----------------------------------------------------------------
    # The optimizer's per-V IC cache (`inv_cache`) needs valid warm-start
    # snapshots near the optimizer's current state.  Pre-populating it from
    # `true_cache` only works when init is close to TRUE; far inits fail
    # to warm-start because the IC's c_O2_surf / c_H2O2_surf differ too much.
    # Cold-solve at INIT params first; then TRF iterations can warm-start
    # from `inv_cache` along the optimizer's trajectory.
    if args.init == "plus20":
        init_offset = 0.20
        init_k0_1 = K0_HAT_R1 * (1 + init_offset)
        init_k0_2 = K0_HAT_R2 * (1 + init_offset)
        init_a_1 = ALPHA_R1 * (1 + init_offset)
        init_a_2 = ALPHA_R2 * (1 + init_offset)
    elif args.init == "minus20":
        init_offset = 0.20
        init_k0_1 = K0_HAT_R1 * (1 - init_offset)
        init_k0_2 = K0_HAT_R2 * (1 - init_offset)
        init_a_1 = ALPHA_R1 * (1 - init_offset)
        init_a_2 = ALPHA_R2 * (1 - init_offset)
    elif args.init == "k0high_alow":
        init_offset = 0.20
        init_k0_1 = K0_HAT_R1 * (1 + init_offset)
        init_k0_2 = K0_HAT_R2 * (1 + init_offset)
        init_a_1 = ALPHA_R1 * (1 - init_offset)
        init_a_2 = ALPHA_R2 * (1 - init_offset)
    elif args.init == "k0low_ahigh":
        init_offset = 0.20
        init_k0_1 = K0_HAT_R1 * (1 - init_offset)
        init_k0_2 = K0_HAT_R2 * (1 - init_offset)
        init_a_1 = ALPHA_R1 * (1 + init_offset)
        init_a_2 = ALPHA_R2 * (1 + init_offset)
    else:  # "true"
        init_k0_1 = K0_HAT_R1; init_k0_2 = K0_HAT_R2
        init_a_1 = ALPHA_R1; init_a_2 = ALPHA_R2

    print()
    if args.save_fim_at_true:
        # Skip cold-solve at INIT; use TRUE-cache as IC for the FIM-at-TRUE
        # computation (no inverse, no offset start).
        print("Step 2.5: --save-fim-at-true; skipping INIT cold-solve, "
              "init_cache = true_cache.", flush=True)
        init_cache = list(true_cache)
    else:
        print("Step 2.5: cold-solve at INIT parameters for warm-start anchor...",
              flush=True)
        print(f"  init: k0_1={init_k0_1:.4e}, k0_2={init_k0_2:.4e}, "
              f"a_1={init_a_1:.4f}, a_2={init_a_2:.4f}")
        t0 = time.time()
        init_cache = [None] * NV
        for i, V in enumerate(V_GRID):
            t_v = time.time()
            cd_i, pc_i, snap_i = solve_cold(float(V),
                                             init_k0_1, init_k0_2,
                                             init_a_1, init_a_2)
            if snap_i is None:
                print(f"  WARN: cold solve at INIT failed at V={V}; "
                      f"will fall back to TRUE-cache for this voltage",
                      flush=True)
                init_cache[i] = None
            else:
                init_cache[i] = snap_i
                print(f"  V={V:+.2f}: cd={cd_i:+.6e}, pc={pc_i:+.6e}  "
                      f"({time.time()-t_v:.1f}s)", flush=True)
        print(f"  INIT cold-ramp done in {time.time()-t0:.1f}s; "
              f"{sum(1 for c in init_cache if c is not None)}/{NV} voltages converged.")

    # ----------------------------------------------------------------
    # Step 3: residual + Jacobian via per-observable adjoint
    # ----------------------------------------------------------------
    # Use init_cache as the optimizer's starting IC (instead of true_cache).
    # Fall back to true_cache when a voltage's init-cache failed.
    inv_cache = [c if c is not None else true_cache[i]
                 for i, c in enumerate(init_cache)]
    cd_sanity = 3.0 * float(np.max(np.abs(target_cd)))
    pc_sanity = 5.0 * float(np.max(np.abs(target_pc)))

    eval_count = [0]
    history = []

    def compute_residuals_and_jacobian(x, want_jac=True):
        """Return (residuals_10, J_10x4) or (residuals_10, None) if want_jac=False.
        Residual ordering: [r_cd_v0, r_cd_v1, ..., r_cd_v4, r_pc_v0, ..., r_pc_v4].
        On failure: returns (large_residuals, jac_with_back_to_x0_direction)."""
        log_k0_1, log_k0_2, a_1, a_2 = x
        k0_1 = float(np.exp(log_k0_1)); k0_2 = float(np.exp(log_k0_2))

        # Guard against unphysical params before invoking forward solver.
        # Forward/bv_solver/config.py validates alpha in (0, 1] and will
        # raise ValueError otherwise.  Method='lm' has no bounds, so an
        # overshoot step can land here.  Return huge residuals + a
        # back-pointing Jacobian so LM/TRF takes a step away from the
        # boundary rather than crashing the run.
        eps_a = 1e-3
        log_k0_radius = 5.0
        a1_low = a_1 <= eps_a;          a1_high = a_1 > 1.0
        a2_low = a_2 <= eps_a;          a2_high = a_2 > 1.0
        k1_oob = (not np.isfinite(log_k0_1)) or abs(log_k0_1 - np.log(K0_HAT_R1)) > log_k0_radius
        k2_oob = (not np.isfinite(log_k0_2)) or abs(log_k0_2 - np.log(K0_HAT_R2)) > log_k0_radius
        if a1_low or a1_high or a2_low or a2_high or k1_oob or k2_oob:
            print(f"  GUARD: unphysical x=[lk1={log_k0_1:.3f}, lk2={log_k0_2:.3f}, "
                  f"a1={a_1:.3f}, a2={a_2:.3f}] -> huge-r fallback", flush=True)
            big = np.full(2 * NV, 1e3)
            if want_jac:
                # Sign rule for α: if a_j < eps, want Δa_j > 0 -> J col < 0.
                # If a_j > 1, want Δa_j < 0 -> J col > 0.
                # Sign rule for log_k0: push back toward TRUE.
                bad_J = np.zeros((2 * NV, 4))
                bad_J[:, 0] = (+1e1 if log_k0_1 > np.log(K0_HAT_R1) else -1e1) if k1_oob else 1e0
                bad_J[:, 1] = (+1e1 if log_k0_2 > np.log(K0_HAT_R2) else -1e1) if k2_oob else 1e0
                if a1_low:    bad_J[:, 2] = -1e2
                elif a1_high: bad_J[:, 2] = +1e2
                if a2_low:    bad_J[:, 3] = -1e2
                elif a2_high: bad_J[:, 3] = +1e2
                return big, bad_J
            return big, None

        residuals = np.zeros(2 * NV)
        if want_jac:
            J = np.zeros((2 * NV, 4))

        n_failed = 0
        last_U = None
        tape = adj.get_working_tape()

        for v_idx, V in enumerate(V_GRID):
            V = float(V)
            # IC cascade: cached -> last solved -> TRUE-cache -> cold
            ic_data = inv_cache[v_idx]
            if ic_data is None and last_U is not None:
                ic_data = last_U
            if ic_data is None and true_cache[v_idx] is not None:
                ic_data = true_cache[v_idx]
            if ic_data is None:
                cd, pc, snap = solve_cold(V, k0_1, k0_2, a_1, a_2)
                if snap is None or abs(cd) > cd_sanity or abs(pc) > pc_sanity:
                    n_failed += 1
                    continue
                inv_cache[v_idx] = snap; last_U = snap
                # Cold path: no adjoint, only residual
                residuals[v_idx] = (cd - noisy_cd[v_idx]) / sigma_cd[v_idx]
                residuals[NV + v_idx] = (pc - noisy_pc[v_idx]) / sigma_pc[v_idx]
                continue

            # Annotated warm-start
            tape.clear_tape()
            adj.continue_annotation()

            if want_jac:
                cd_a, pc_a, snap, ctx = solve_warm_annotated(
                    V, k0_1, k0_2, a_1, a_2, ic_data,
                    annotate_final_steps=5, max_steps=120)
            else:
                cd_v, pc_v, snap = solve_warm_unann(
                    V, k0_1, k0_2, a_1, a_2, ic_data, max_steps=120)
                cd_a = cd_v; pc_a = pc_v

            warm_failed = (snap is None)
            cd_val = float(cd_a) if not warm_failed else float("nan")
            pc_val = float(pc_a) if not warm_failed else float("nan")
            unphysical = (not warm_failed) and (
                abs(cd_val) > cd_sanity or abs(pc_val) > pc_sanity
                or not np.isfinite(cd_val) or not np.isfinite(pc_val))

            if warm_failed or unphysical:
                # IC fallback cascade: try init_cache (cold-solved at the
                # optimizer's init params), then true_cache (only useful
                # if optimizer is near TRUE).
                adj.pause_annotation(); tape.clear_tape()
                inv_cache[v_idx] = None
                fallback_ic = (init_cache[v_idx] if init_cache[v_idx] is not None
                               else true_cache[v_idx])
                if fallback_ic is None:
                    n_failed += 1; continue
                tape.clear_tape(); adj.continue_annotation()
                if want_jac:
                    cd_a, pc_a, snap, ctx = solve_warm_annotated(
                        V, k0_1, k0_2, a_1, a_2, fallback_ic,
                        annotate_final_steps=5, max_steps=200)
                else:
                    cd_v, pc_v, snap = solve_warm_unann(
                        V, k0_1, k0_2, a_1, a_2, fallback_ic, max_steps=200)
                    cd_a = cd_v; pc_a = pc_v
                if snap is None:
                    adj.pause_annotation(); tape.clear_tape()
                    n_failed += 1; continue
                cd_val = float(cd_a); pc_val = float(pc_a)
                if abs(cd_val) > cd_sanity or abs(pc_val) > pc_sanity:
                    adj.pause_annotation(); tape.clear_tape()
                    n_failed += 1; continue

            inv_cache[v_idx] = snap
            last_U = snap

            r_cd = (cd_val - noisy_cd[v_idx]) / sigma_cd[v_idx]
            r_pc = (pc_val - noisy_pc[v_idx]) / sigma_pc[v_idx]
            residuals[v_idx] = r_cd
            residuals[NV + v_idx] = r_pc

            if want_jac:
                # Per-observable adjoint: build SAME tape, two ReducedFunctionals
                k0_funcs = list(ctx["bv_k0_funcs"][:2])
                alpha_funcs = list(ctx["bv_alpha_funcs"][:2])
                controls = [adj.Control(f) for f in k0_funcs + alpha_funcs]
                try:
                    rf_cd = adj.ReducedFunctional(cd_a, controls)
                    dcd_vals = rf_cd.derivative()
                    rf_pc = adj.ReducedFunctional(pc_a, controls)
                    dpc_vals = rf_pc.derivative()
                except Exception as e:
                    adj.pause_annotation()
                    print(f"    WARN adjoint failed at V={V}: {type(e).__name__}: {e}")
                    # Skip this voltage's Jacobian rows (already zero)
                    continue

                def _scalar(g):
                    if hasattr(g, "dat"): return float(g.dat[0].data_ro[0])
                    if hasattr(g, "values"): return float(g.values()[0])
                    return float(g)

                dcd = [_scalar(g) for g in dcd_vals]
                dpc = [_scalar(g) for g in dpc_vals]

                # Chain rule: d/d(log_k0) = k0 * d/dk0 ; alpha unchanged
                # Then divide by sigma to get d r_i / d theta_j
                J[v_idx, 0] = k0_1 * dcd[0] / sigma_cd[v_idx]
                J[v_idx, 1] = k0_2 * dcd[1] / sigma_cd[v_idx]
                J[v_idx, 2] = dcd[2] / sigma_cd[v_idx]
                J[v_idx, 3] = dcd[3] / sigma_cd[v_idx]
                J[NV + v_idx, 0] = k0_1 * dpc[0] / sigma_pc[v_idx]
                J[NV + v_idx, 1] = k0_2 * dpc[1] / sigma_pc[v_idx]
                J[NV + v_idx, 2] = dpc[2] / sigma_pc[v_idx]
                J[NV + v_idx, 3] = dpc[3] / sigma_pc[v_idx]

            adj.pause_annotation()

        if n_failed > NV // 2:
            # Return huge residuals + Jacobian pointing back to TRUE
            big = np.full(2 * NV, 1e3)
            if want_jac:
                # Synthesize Jacobian pointing residuals toward zero at TRUE
                # Just use identity scaled (least_squares will damp it)
                bad_J = np.zeros((2 * NV, 4))
                # crude: each residual depends on each param identically
                bad_J[:] = 1e2 * np.tile(np.array([1, 1, 1, 1]), (2 * NV, 1))
                return big, bad_J
            return big, None

        return residuals, J if want_jac else None

    # Cache for least_squares to avoid recomputing when fun and jac called at same x
    last_cache = {"x": None, "r": None, "J": None}

    def _eval_with_prior(x):
        """Run forward + adjoint, then optionally append Tikhonov prior rows.
        When USE_ANCHOR, x is in anchored coords; the forward solver and
        prior live in physical coords; J is chain-ruled back to anc."""
        x_phys = _phys_from_x(x)
        r_data, J_data_phys = compute_residuals_and_jacobian(x_phys, want_jac=True)
        # Chain rule: J_anc = J_phys @ M_ANC (no-op when not anchored).
        J_data = J_data_phys @ M_ANC if USE_ANCHOR else J_data_phys
        if USE_PRIOR:
            # Prior is on physical log_k0_j.  Residual stays the same; J is
            # chain-ruled by the prior's physical-coord Jacobian.
            log_k0_1, log_k0_2 = float(x_phys[0]), float(x_phys[1])
            r_prior = np.array([
                (log_k0_1 - LOG_K0_PRIOR_CENTER[0]) / SIGMA_LOG_K0,
                (log_k0_2 - LOG_K0_PRIOR_CENTER[1]) / SIGMA_LOG_K0,
            ])
            J_prior_phys = np.zeros((2, 4))
            J_prior_phys[0, 0] = 1.0 / SIGMA_LOG_K0
            J_prior_phys[1, 1] = 1.0 / SIGMA_LOG_K0
            J_prior = J_prior_phys @ M_ANC if USE_ANCHOR else J_prior_phys
            r = np.concatenate([r_data, r_prior])
            J = np.vstack([J_data, J_prior])
        else:
            r = r_data; J = J_data; r_prior = None
        return r, J, r_data, r_prior

    def fun(x):
        if last_cache["x"] is None or not np.allclose(x, last_cache["x"], atol=1e-14):
            t0 = time.time()
            r, J, r_data, r_prior = _eval_with_prior(x)
            last_cache["x"] = x.copy(); last_cache["r"] = r; last_cache["J"] = J
            last_cache["r_data"] = r_data; last_cache["r_prior"] = r_prior
            last_cache["elapsed"] = time.time() - t0
        eval_count[0] += 1
        r = last_cache["r"]
        r_data = last_cache["r_data"]; r_prior = last_cache["r_prior"]
        cost = 0.5 * float(np.sum(r ** 2))
        cost_data = 0.5 * float(np.sum(r_data ** 2))
        cost_prior = (0.5 * float(np.sum(r_prior ** 2))
                      if r_prior is not None else 0.0)
        x_phys = _phys_from_x(x)
        log_k0_1, log_k0_2, a_1, a_2 = x_phys
        k0_1 = float(np.exp(log_k0_1)); k0_2 = float(np.exp(log_k0_2))
        e_k0_1 = 100*(k0_1-K0_HAT_R1)/K0_HAT_R1
        e_k0_2 = 100*(k0_2-K0_HAT_R2)/K0_HAT_R2
        e_a_1 = 100*(a_1-ALPHA_R1)/ALPHA_R1
        e_a_2 = 100*(a_2-ALPHA_R2)/ALPHA_R2
        elapsed = last_cache.get("elapsed", 0)
        cost_str = (f"cost={cost:.4e}(d={cost_data:.3e},p={cost_prior:.3e})"
                    if USE_PRIOR else f"cost={cost:.4e}")
        anc_str = (f"  beta=({x[0]:.3f},{x[1]:.3f})" if USE_ANCHOR else "")
        print(f"  [{eval_count[0]:3d}] k0=({k0_1:.3e},{k0_2:.3e}) "
              f"a=({a_1:.4f},{a_2:.4f}){anc_str}  {cost_str}  "
              f"err%=({e_k0_1:+.2f},{e_k0_2:+.2f},{e_a_1:+.2f},{e_a_2:+.2f})  "
              f"{elapsed:.1f}s", flush=True)
        history.append({
            "eval": eval_count[0], "k0_1": k0_1, "k0_2": k0_2,
            "alpha_1": a_1, "alpha_2": a_2,
            "beta_1": float(x[0]) if USE_ANCHOR else None,
            "beta_2": float(x[1]) if USE_ANCHOR else None,
            "cost": cost, "cost_data": cost_data, "cost_prior": cost_prior,
            "k0_1_err_pct": e_k0_1, "k0_2_err_pct": e_k0_2,
            "alpha_1_err_pct": e_a_1, "alpha_2_err_pct": e_a_2,
            "residual_norm": float(np.linalg.norm(r)),
            "elapsed_s": elapsed,
        })
        if eval_count[0] % 5 == 0:
            with open(os.path.join(OUT_DIR, "history_partial.json"), "w") as f:
                json.dump(history, f, indent=2)
        return r

    def jac(x):
        if last_cache["x"] is None or not np.allclose(x, last_cache["x"], atol=1e-14):
            r, J, r_data, r_prior = _eval_with_prior(x)
            last_cache["x"] = x.copy(); last_cache["r"] = r; last_cache["J"] = J
            last_cache["r_data"] = r_data; last_cache["r_prior"] = r_prior
        return last_cache["J"]

    if args.save_fim_at_true:
        # Compute the whitened sensitivity matrix at TRUE params and save.
        # This is the per-experiment building block for multi-experiment FIM.
        print("\nStep 3.5: --save-fim-at-true; computing J at TRUE params.",
              flush=True)
        x_true_phys = np.array([np.log(K0_HAT_R1), np.log(K0_HAT_R2),
                                ALPHA_R1, ALPHA_R2])
        x_true_optcoord = _x_from_phys(x_true_phys)
        t0 = time.time()
        r_true, J_true_white = compute_residuals_and_jacobian(x_true_phys,
                                                              want_jac=True)
        print(f"  J at TRUE in {time.time()-t0:.1f}s; "
              f"r||={np.linalg.norm(r_true):.3e}, J shape={J_true_white.shape}")
        # J_true_white = (∂r/∂phys) / sigma; r = (cd_modeled - cd_target)/sigma.
        # At TRUE, r should be ~0 (any residual = numerical noise).  J rows are
        # ordered [cd@V0..NV-1, pc@V0..NV-1].  Convert back to raw S = J × σ:
        sigma_stack = np.concatenate([sigma_cd, sigma_pc])
        S_white_full = J_true_white                # shape (2*NV, 4)
        S_raw = J_true_white * sigma_stack[:, None]  # raw ∂y/∂theta (no σ)
        S_cd_raw = S_raw[:NV, :]
        S_pc_raw = S_raw[NV:, :]
        out_fim = {
            "config": {
                "V_GRID": V_GRID.tolist(),
                "c_O2_hat": float(C_O2_HAT_USED),
                "use_log_rate": USE_LOG_RATE,
                "use_anchor_tafel": USE_ANCHOR,
                "params": ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"],
                "true_params": {"k0_1": K0_HAT_R1, "k0_2": K0_HAT_R2,
                                "alpha_1": ALPHA_R1, "alpha_2": ALPHA_R2},
                "sigma_model": "global 2% max" if args.noise == 0 else f"{args.noise}% relative",
            },
            "observables_at_true": {
                "cd": target_cd.tolist(),
                "pc": target_pc.tolist(),
            },
            "sigma": {
                "sigma_cd": sigma_cd.tolist(),
                "sigma_pc": sigma_pc.tolist(),
            },
            "S_white": S_white_full.tolist(),    # already whitened
            "S_cd_raw": S_cd_raw.tolist(),       # not whitened
            "S_pc_raw": S_pc_raw.tolist(),
            "residuals_at_true_norm": float(np.linalg.norm(r_true)),
        }
        path = os.path.join(OUT_DIR, "fim_at_true.json")
        with open(path, "w") as f:
            json.dump(out_fim, f, indent=2)
        print(f"  Saved: {path}")
        return

    # ----------------------------------------------------------------
    # Step 4: initial guess + bounds
    # ----------------------------------------------------------------
    init_offset = 0.20
    if args.init == "plus20":
        x0 = np.array([
            np.log(K0_HAT_R1 * (1 + init_offset)),
            np.log(K0_HAT_R2 * (1 + init_offset)),
            ALPHA_R1 * (1 + init_offset),
            ALPHA_R2 * (1 + init_offset),
        ])
    elif args.init == "minus20":
        x0 = np.array([
            np.log(K0_HAT_R1 * (1 - init_offset)),
            np.log(K0_HAT_R2 * (1 - init_offset)),
            ALPHA_R1 * (1 - init_offset),
            ALPHA_R2 * (1 - init_offset),
        ])
    elif args.init == "k0high_alow":
        x0 = np.array([
            np.log(K0_HAT_R1 * (1 + init_offset)),
            np.log(K0_HAT_R2 * (1 + init_offset)),
            ALPHA_R1 * (1 - init_offset),
            ALPHA_R2 * (1 - init_offset),
        ])
    elif args.init == "k0low_ahigh":
        x0 = np.array([
            np.log(K0_HAT_R1 * (1 - init_offset)),
            np.log(K0_HAT_R2 * (1 - init_offset)),
            ALPHA_R1 * (1 + init_offset),
            ALPHA_R2 * (1 + init_offset),
        ])
    else:  # "true"
        x0 = np.array([
            np.log(K0_HAT_R1), np.log(K0_HAT_R2), ALPHA_R1, ALPHA_R2,
        ])

    if args.start_theta is not None:
        # --start-theta is in PHYSICAL coords by convention; transform if anchored.
        x0 = np.array(args.start_theta, dtype=float)
        print(f"  --start-theta overrides --init={args.init}: physical x0={x0.tolist()}")
        if USE_ANCHOR:
            x0 = _x_from_phys(x0)
            print(f"  --anchor-tafel: x0 transformed to anchored = {x0.tolist()}")
    else:
        # --init logic above produced x0 in PHYSICAL coords; transform if anchored.
        if USE_ANCHOR:
            x0 = _x_from_phys(x0)

    if args.method == "trf":
        if USE_ANCHOR:
            # beta_j = log_k0_j - alpha_j * C_j with C_2 ≈ -100 means a 20%
            # alpha perturbation shifts beta_2 by ~20, so the standard 4 inits
            # (plus20, minus20, k0high_alow, k0low_ahigh) start ~10 units
            # from beta_TRUE.  Bound radius needs to accommodate the inits
            # PLUS room for TRF to explore log_k0 in TRUE ± 2 across the full
            # alpha range.  Per-reaction radius:
            #   r_j ≥ 2 (log_k0 swing) + 0.2 * |C_j| (alpha swing)
            # = 8.7 (R1), 21.9 (R2). Use 12 / 25 with margin.
            # The unphysical-param GUARD catches log_k0 outside ±5 of TRUE.
            beta_true_1 = np.log(K0_HAT_R1) - ALPHA_R1 * C_ANCHOR_1
            beta_true_2 = np.log(K0_HAT_R2) - ALPHA_R2 * C_ANCHOR_2
            beta_r1 = max(12.0, 2.0 + 0.25 * abs(C_ANCHOR_1) + 1.0)
            beta_r2 = max(25.0, 2.0 + 0.25 * abs(C_ANCHOR_2) + 1.0)
            bounds_lower = np.array([
                beta_true_1 - beta_r1,
                beta_true_2 - beta_r2,
                max(0.20, ALPHA_R1 - 0.2),
                max(0.20, ALPHA_R2 - 0.2),
            ])
            bounds_upper = np.array([
                beta_true_1 + beta_r1,
                beta_true_2 + beta_r2,
                min(0.80, ALPHA_R1 + 0.2),
                min(0.80, ALPHA_R2 + 0.2),
            ])
            print(f"  bounds (anchored): beta_1 [{bounds_lower[0]:.2f}, {bounds_upper[0]:.2f}], "
                  f"beta_2 [{bounds_lower[1]:.2f}, {bounds_upper[1]:.2f}], "
                  f"a_1 [{bounds_lower[2]:.3f}, {bounds_upper[2]:.3f}], "
                  f"a_2 [{bounds_lower[3]:.3f}, {bounds_upper[3]:.3f}]")
        else:
            bounds_lower = np.array([
                np.log(K0_HAT_R1) - 2.0,
                np.log(K0_HAT_R2) - 2.0,
                max(0.20, ALPHA_R1 - 0.2),
                max(0.20, ALPHA_R2 - 0.2),
            ])
            bounds_upper = np.array([
                np.log(K0_HAT_R1) + 2.0,
                np.log(K0_HAT_R2) + 2.0,
                min(0.80, ALPHA_R1 + 0.2),
                min(0.80, ALPHA_R2 + 0.2),
            ])
            print(f"  bounds: log_k0_1 [{bounds_lower[0]:.3f}, {bounds_upper[0]:.3f}], "
                  f"log_k0_2 [{bounds_lower[1]:.3f}, {bounds_upper[1]:.3f}], "
                  f"a_1 [{bounds_lower[2]:.3f}, {bounds_upper[2]:.3f}], "
                  f"a_2 [{bounds_lower[3]:.3f}, {bounds_upper[3]:.3f}]")
        bounds_arg = (bounds_lower, bounds_upper)
    else:
        bounds_arg = (-np.inf, np.inf)
        print("  no bounds (LM)")

    print()
    print(f"Step 4: scipy.optimize.least_squares method={args.method}, "
          f"x0={x0.tolist()}")
    print()
    t_opt = time.time()
    res = least_squares(
        fun, x0, jac=jac, method=args.method, bounds=bounds_arg,
        max_nfev=args.maxiter * 4,  # LSQ counts function evals not iters
        ftol=1e-9, xtol=1e-9, gtol=1e-8,
        verbose=2,
    )
    t_opt = time.time() - t_opt

    # res.x lives in optimizer coords (anchored if --anchor-tafel else physical).
    # Convert to physical for downstream reporting.
    x_phys_rec = _phys_from_x(res.x)
    log_k0_1_rec, log_k0_2_rec, a_1_rec, a_2_rec = x_phys_rec
    k0_1_rec = float(np.exp(log_k0_1_rec)); k0_2_rec = float(np.exp(log_k0_2_rec))

    print()
    print("=" * 72)
    print("FINAL")
    print("=" * 72)
    print(f"  evals: {eval_count[0]}, wall: {t_opt/60:.1f} min, "
          f"status: {res.status}, msg: {res.message}")
    print(f"  cost final: {res.cost:.4e}, residual norm: {np.linalg.norm(res.fun):.4e}")
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
            "init_x": x0.tolist(), "init_offset_pct": init_offset * 100,
            "noise_pct": args.noise, "noise_seed": args.seed,
            "method": args.method,
            "observable": "joint disk + peroxide, sigma-whitened residuals",
            "ic_cache": "per-V, persists across evals; TRUE-cache fallback",
            "gradient": "per-observable adjoint via pyadjoint (10 RFs per Jacobian)",
            "use_prior": USE_PRIOR,
            "prior_sigma_log_k0": SIGMA_LOG_K0,
            "prior_center": args.prior_center,
            "log_k0_prior_center": (LOG_K0_PRIOR_CENTER.tolist()
                                    if USE_PRIOR else None),
            "use_anchor_tafel": USE_ANCHOR,
            "v_anchor_1": args.v_anchor_1 if USE_ANCHOR else None,
            "v_anchor_2": args.v_anchor_2 if USE_ANCHOR else None,
            "C_anchor_1": C_ANCHOR_1 if USE_ANCHOR else None,
            "C_anchor_2": C_ANCHOR_2 if USE_ANCHOR else None,
        },
        "result": {
            "k0_1": k0_1_rec, "k0_2": k0_2_rec,
            "alpha_1": a_1_rec, "alpha_2": a_2_rec,
            "k0_1_err_pct": 100 * (k0_1_rec - K0_HAT_R1) / K0_HAT_R1,
            "k0_2_err_pct": 100 * (k0_2_rec - K0_HAT_R2) / K0_HAT_R2,
            "alpha_1_err_pct": 100 * (a_1_rec - ALPHA_R1) / ALPHA_R1,
            "alpha_2_err_pct": 100 * (a_2_rec - ALPHA_R2) / ALPHA_R2,
            "x_optimizer": res.x.tolist(),  # anchored coords if USE_ANCHOR
            "x_physical": x_phys_rec.tolist(),
            "beta_1": float(res.x[0]) if USE_ANCHOR else None,
            "beta_2": float(res.x[1]) if USE_ANCHOR else None,
            "cost_final": float(res.cost),
            "residual_norm_final": float(np.linalg.norm(res.fun)),
            "n_evals": eval_count[0],
            "wall_minutes": t_opt / 60,
            "status": int(res.status),
            "message": str(res.message),
        },
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'result.json')}")

    # ----------------------------------------------------------------
    # Line profile from recovered to TRUE
    # ----------------------------------------------------------------
    # Per GPT's PNP Log Rate Next Steps Handoff Task 4: J(t) along
    # theta(t) = (1-t)*theta_recovered + t*theta_true should be checked.
    # If J(t) decreases monotonically from recovered to TRUE, the optimizer
    # stalled even if the endpoint looks decent.
    print()
    print("Step 5: line profile from recovered to TRUE")
    theta_rec = np.array([log_k0_1_rec, log_k0_2_rec, a_1_rec, a_2_rec])
    theta_true = np.array([
        np.log(K0_HAT_R1), np.log(K0_HAT_R2), ALPHA_R1, ALPHA_R2
    ])
    ts = [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]
    profile = []
    for t in ts:
        theta = (1 - t) * theta_rec + t * theta_true
        k0_1 = float(np.exp(theta[0]))
        k0_2 = float(np.exp(theta[1]))
        a_1 = float(theta[2]); a_2 = float(theta[3])
        cds_t = np.full(NV, np.nan); pcs_t = np.full(NV, np.nan)
        for i_v, V in enumerate(V_GRID):
            ic = inv_cache[i_v] if inv_cache[i_v] is not None else true_cache[i_v]
            cd_v, pc_v, _ = solve_warm_unann(float(V), k0_1, k0_2, a_1, a_2, ic)
            if cd_v is not None:
                cds_t[i_v] = cd_v; pcs_t[i_v] = pc_v
        if np.any(np.isnan(cds_t)) or np.any(np.isnan(pcs_t)):
            J_t = float("nan"); J_cd = float("nan"); J_pc = float("nan")
        else:
            r_cd = (cds_t - target_cd) / sigma_cd
            r_pc = (pcs_t - target_pc) / sigma_pc
            J_cd = 0.5 * float(np.sum(r_cd**2))
            J_pc = 0.5 * float(np.sum(r_pc**2))
            J_t = J_cd + J_pc
        profile.append({
            "t": float(t), "J": J_t, "J_cd": J_cd, "J_pc": J_pc,
            "theta": theta.tolist(),
        })
        print(f"  t={t:.2f}  J={J_t:.4e}  J_cd={J_cd:.4e}  J_pc={J_pc:.4e}")

    # Monotonicity check: is J(t) monotonic from t=0 to t=1?
    Js = [p["J"] for p in profile if np.isfinite(p["J"])]
    monotone_descent = all(Js[i] >= Js[i+1] for i in range(len(Js) - 1))
    print(f"  Monotone descent recovered→TRUE: {monotone_descent}")
    print(f"  J(recovered)={profile[0]['J']:.4e}, J(TRUE)={profile[-1]['J']:.4e}")

    profile_path = os.path.join(OUT_DIR, "line_profile.json")
    with open(profile_path, "w") as f:
        json.dump({
            "theta_recovered": theta_rec.tolist(),
            "theta_true": theta_true.tolist(),
            "monotone_descent_recovered_to_true": monotone_descent,
            "profile": profile,
        }, f, indent=2)
    print(f"  Saved: {profile_path}")


if __name__ == "__main__":
    main()
