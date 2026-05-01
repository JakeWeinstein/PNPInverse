"""V18 k0 inference vs noise level.

Runs the log-c inference at multiple noise levels (0%, 0.5%, 1%, 2%)
with a fixed seed. If k0 error is noise-limited (scales with noise),
then we can estimate the fundamental precision available from onset data.

If k0 error is LARGE even at 0% noise, it's not noise -- it's ridge degeneracy.
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
    import pyadjoint as adj
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

    V_GRID = np.array([-0.10, 0.00, 0.10, 0.15, 0.20, 0.25])

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

    def solve_point(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, z_steps=20):
        phi_hat = V_RHE / V_T
        sp = make_3sp_sp(phi_hat, k0_r1, k0_r2, alpha_r1, alpha_r2)
        with adj.stop_annotating():
            ctx = build_context_logc(list(sp), mesh=mesh)
            ctx = build_forms_logc(ctx, list(sp))
            ctx = add_boltzmann(ctx)
            set_initial_conditions_logc(ctx, list(sp))

        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
        sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
        of = _build_bv_observable_form(ctx, mode="current_density",
                                        reaction_index=None, scale=-I_SCALE)

        dt_init = 0.25
        def run_ss(max_steps=60):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for s in range(1, max_steps+1):
                try: sol.solve()
                except Exception: return False
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
                if sc >= 4: return True
            return False

        for zci in zc: zci.assign(0.0)
        paf.assign(phi_hat)
        with adj.stop_annotating():
            if not run_ss(100):
                return np.nan

        z_nominal = [float(sp[4][i]) for i in range(n)]
        achieved_z = 0.0
        with adj.stop_annotating():
            for z_val in np.linspace(0, 1, z_steps+1)[1:]:
                U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
                for i in range(n): zc[i].assign(z_nominal[i]*z_val)
                if run_ss(60):
                    achieved_z = z_val
                else:
                    for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                    Up.assign(U)
                    break

        if achieved_z < 1.0 - 1e-3:
            return np.nan
        return float(fd.assemble(of))

    def solve_curve(k0_r1, k0_r2, alpha_r1, alpha_r2):
        cds = np.zeros(len(V_GRID))
        for i, V in enumerate(V_GRID):
            cds[i] = solve_point(V, k0_r1, k0_r2, alpha_r1, alpha_r2)
        return cds

    # Generate target ONCE
    print("Computing target at true parameters...", flush=True)
    t0 = time.time()
    target_cd = solve_curve(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    print(f"Target computed in {time.time()-t0:.1f}s")

    # Run inference at multiple noise levels
    results = []
    for noise_pct in [0.0, 0.5, 1.0, 2.0]:
        print(f"\n{'='*60}\nNOISE LEVEL: {noise_pct}%\n{'='*60}")
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_pct/100, size=target_cd.shape) * np.abs(target_cd)
        noisy = target_cd + noise

        eval_count = [0]
        def objective(x):
            log_k0, alpha = x
            k0 = np.exp(log_k0)
            if not (0.1 < alpha < 0.9): return 1e6
            if not (1e-6 < k0 < 1.0): return 1e6
            eval_count[0] += 1
            cds = solve_curve(k0, K0_HAT_R2, alpha, ALPHA_R2)
            if not np.all(np.isfinite(cds)):
                n_nan = np.sum(~np.isfinite(cds))
                if n_nan > 2: return 1e6
                finite = np.isfinite(cds)
                diff = cds[finite] - noisy[finite]
            else:
                diff = cds - noisy
            J = 0.5 * np.sum(diff**2)
            print(f"  [{eval_count[0]}] k0={k0:.3e} alpha={alpha:.4f} J={J:.4e}", flush=True)
            return J

        x0 = [np.log(K0_HAT_R1 * 1.2), ALPHA_R1 * 0.9]
        t0 = time.time()
        res = minimize(objective, x0, method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 30})
        k0_rec = np.exp(res.x[0])
        alpha_rec = res.x[1]
        k0_err = 100 * (k0_rec - K0_HAT_R1) / K0_HAT_R1
        alpha_err = 100 * (alpha_rec - ALPHA_R1) / ALPHA_R1
        results.append((noise_pct, k0_rec, alpha_rec, k0_err, alpha_err, res.fun, eval_count[0], time.time()-t0))
        print(f"  Recovered: k0={k0_rec:.3e} ({k0_err:+.1f}%), alpha={alpha_rec:.4f} ({alpha_err:+.1f}%), J={res.fun:.2e}")

    print(f"\n{'='*60}\nNOISE LIMIT ANALYSIS\n{'='*60}")
    print(f"  {'noise':>7}  {'k0_rec':>11}  {'k0 err':>8}  {'alpha':>7}  {'α err':>7}  {'J':>10}  {'evals':>5}")
    for noise_pct, k0, alpha, k0_err, alpha_err, J, n, t in results:
        print(f"  {noise_pct:>6.1f}%  {k0:>11.3e}  {k0_err:>+7.1f}%  {alpha:>7.4f}  {alpha_err:>+6.1f}%  {J:>10.2e}  {n:>5}")


if __name__ == "__main__":
    main()
