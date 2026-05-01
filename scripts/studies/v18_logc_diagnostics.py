"""V18 log-c + Boltzmann: Phase A diagnostics (FIM + line profile).

Per GPT's recommendation in docs/PNP Peroxide Negative Result Next Steps.md.
Runs two independent diagnostics on the same forward model used in
v18_logc_joint_observable.py:

1. **Sensitivity / Fisher Information Matrix** — compute S[m,j] = dy_m/dtheta_j
   via central FD with two step sizes for stability check.  Whiten by an
   assumed 2% relative noise model.  Compare F = S_white^T S_white for:
     - CD only (5 observables)
     - PC only (5 observables)
     - CD + PC (10 observables)
   Report singular values, eigenvalues of F, condition number, and the
   smallest eigenvector of F (the "weak direction" — the parameter combo
   the data is least informative about).  Output: JSON + CSV.

2. **Line profile** from theta_recovered (clean-data L-BFGS-B endpoint) to
   theta_true.  Evaluate J(t), J_cd(t), J_pc(t) at t in [0, 0.05, 0.10, ..., 1.0].
   - If J(t) decreases monotonically, the recovered point was an optimizer/
     conditioning failure, NOT a true local minimum.
   - If J(t) rises before falling, there's a real basin between the two.

All forward solves use the breakthrough recipe:
  forms_logc + 3-species + Boltzmann ClO4- + H2O2 IC seed,
  V_GRID = [-0.10, 0.00, +0.10, +0.15, +0.20] V_RHE.

Warm-starts every per-theta solve from the TRUE-params converged U (an
"anchor cache") so all forward solves take ~1-3 s each.  Total runtime
~15-20 min for both diagnostics.
"""
from __future__ import annotations
import os, sys, time, json

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
    V_GRID = np.array([-0.10, 0.00, 0.10, 0.15, 0.20])
    NV = len(V_GRID)
    PARAM_NAMES = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]

    OUT_FIM = os.path.join(_ROOT, "StudyResults", "v18_logc_sensitivity_fim")
    OUT_PROFILE = os.path.join(_ROOT, "StudyResults", "v18_logc_joint_profile_checks")
    os.makedirs(OUT_FIM, exist_ok=True)
    os.makedirs(OUT_PROFILE, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    # ---- Forward-model machinery (mirrors v18_logc_joint_observable.py) ----

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
                    d = abs(fv - prev_flux); sv = max(abs(fv), abs(prev_flux), 1e-8)
                    if d/sv <= 1e-4 or d <= 1e-8: sc += 1
                    else: sc = 0
                    if prev_delta and d > 0:
                        r = prev_delta/d
                        dt_val = (min(dt_val*min(r,4), dt_init*20)
                                  if r > 1 else max(dt_val*0.5, dt_init))
                        dt_const.assign(dt_val)
                    prev_delta = d
                prev_flux = fv
                if sc >= 4: return True
            return False
        return run_ss

    def solve_warm(V_RHE, k0_1, k0_2, a_1, a_2, ic_data, max_steps=80):
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
        cd_v = float(fd.assemble(of_cd)); pc_v = float(fd.assemble(of_pc))
        return cd_v, pc_v, _snapshot(U)

    def solve_cold(V_RHE, k0_1, k0_2, a_1, a_2, max_z_steps=20):
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_RHE, k0_1, k0_2, a_1, a_2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            for zci in zc: zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(100): return None, None, None
            achieved_z = 0.0
            for z_val in np.linspace(0, 1, max_z_steps+1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n): zc[i].assign(z_nominal[i] * z_val)
                if run_ss(60): achieved_z = z_val
                else:
                    _restore(ckpt, U, Up); break
            if achieved_z < 1.0 - 1e-3: return None, None, None
        cd_v = float(fd.assemble(of_cd)); pc_v = float(fd.assemble(of_pc))
        return cd_v, pc_v, _snapshot(U)

    def solve_curve_warm(theta_log, true_cache):
        """Solve all 5 voltages by warm-starting each from TRUE_cache U.
        theta_log = [log_k0_1, log_k0_2, alpha_1, alpha_2].
        Returns (cd_array, pc_array)."""
        k0_1 = float(np.exp(theta_log[0])); k0_2 = float(np.exp(theta_log[1]))
        a_1 = float(theta_log[2]); a_2 = float(theta_log[3])
        cds = np.full(NV, np.nan); pcs = np.full(NV, np.nan)
        for i, V in enumerate(V_GRID):
            cd, pc, _ = solve_warm(float(V), k0_1, k0_2, a_1, a_2, true_cache[i])
            if cd is not None:
                cds[i] = cd; pcs[i] = pc
        return cds, pcs

    # ----------------------------------------------------------------
    # Step 1: cold-solve TRUE curve to populate the warm-start anchor.
    # ----------------------------------------------------------------
    print("=" * 72)
    print("V18 log-c diagnostics: FIM + line profile")
    print("=" * 72)
    print(f"V_GRID (V_RHE): {V_GRID.tolist()}")
    print()
    print("Step 1: cold-solving TRUE curve as warm-start anchor...", flush=True)
    t0 = time.time()
    true_cache = [None] * NV
    target_cd = np.zeros(NV); target_pc = np.zeros(NV)
    last_U = None
    for i, V in enumerate(V_GRID):
        t_v = time.time()
        if last_U is None:
            cd, pc, snap = solve_cold(float(V), K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
            ic_kind = "cold"
        else:
            cd, pc, snap = solve_warm(float(V), K0_HAT_R1, K0_HAT_R2,
                                       ALPHA_R1, ALPHA_R2, last_U, max_steps=200)
            ic_kind = "warm"
        if snap is None:
            # Fallback: cold ramp at this voltage
            print(f"  V={V:+.2f}: {ic_kind} failed; falling back to cold ramp...",
                  flush=True)
            cd, pc, snap = solve_cold(float(V), K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
            ic_kind = f"{ic_kind}->cold"
        if snap is None:
            print(f"  FATAL: TRUE solve failed at V={V} (both warm and cold)")
            return
        target_cd[i] = cd; target_pc[i] = pc; true_cache[i] = snap; last_U = snap
        print(f"  V={V:+.2f}: cd={cd:+.4e}, pc={pc:+.4e}  [{ic_kind}] "
              f"({time.time()-t_v:.1f}s)", flush=True)
    print(f"  TRUE curve in {time.time()-t0:.1f}s")

    theta_true = np.array([np.log(K0_HAT_R1), np.log(K0_HAT_R2),
                           ALPHA_R1, ALPHA_R2])

    # ----------------------------------------------------------------
    # Step 2: FIM diagnostic
    #
    # Sensitivity matrix S[m, j] = dy_m / dtheta_j computed via central FD.
    # m ranges over (NV cd values) + (NV pc values).
    # Two step sizes for stability check.
    # Whiten by sigma_m = 2% * max(|target_obs|).
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("Step 2: FIM diagnostic (sensitivity matrix at TRUE params)")
    print("=" * 72)

    # GPT recommended step ranges:
    # log_k0_j: 1e-3 to 1e-2 (absolute step in log-space)
    # alpha_j:  1e-4 to 1e-3 (absolute step in alpha)
    step_sizes = {
        "small": np.array([1e-3, 1e-3, 1e-4, 1e-4]),
        "large": np.array([1e-2, 1e-2, 1e-3, 1e-3]),
    }

    # Whitening: 2% relative noise, sigma = 2% * max(|target|) per observable type
    sigma_cd_w = 0.02 * float(np.max(np.abs(target_cd)))
    sigma_pc_w = 0.02 * float(np.max(np.abs(target_pc)))
    print(f"  Whitening: sigma_cd = {sigma_cd_w:.4e}, sigma_pc = {sigma_pc_w:.4e}")

    fim_results = {}  # keyed by step size

    for step_label, h_vec in step_sizes.items():
        print(f"\n  --- Step size '{step_label}': h = {h_vec.tolist()} ---", flush=True)
        S_cd = np.zeros((NV, 4))   # 5 cd observables vs 4 params
        S_pc = np.zeros((NV, 4))   # 5 pc observables vs 4 params

        for j in range(4):
            t_j = time.time()
            theta_plus = theta_true.copy(); theta_plus[j] += h_vec[j]
            theta_minus = theta_true.copy(); theta_minus[j] -= h_vec[j]
            cd_plus, pc_plus = solve_curve_warm(theta_plus, true_cache)
            cd_minus, pc_minus = solve_curve_warm(theta_minus, true_cache)

            ok_plus = np.all(np.isfinite(cd_plus)) and np.all(np.isfinite(pc_plus))
            ok_minus = np.all(np.isfinite(cd_minus)) and np.all(np.isfinite(pc_minus))
            if not (ok_plus and ok_minus):
                print(f"    d/d{PARAM_NAMES[j]:>10}: FAILED "
                      f"(plus_ok={ok_plus}, minus_ok={ok_minus})", flush=True)
                S_cd[:, j] = np.nan; S_pc[:, j] = np.nan
                continue

            S_cd[:, j] = (cd_plus - cd_minus) / (2 * h_vec[j])
            S_pc[:, j] = (pc_plus - pc_minus) / (2 * h_vec[j])
            print(f"    d/d{PARAM_NAMES[j]:>10}: |dcd|={np.linalg.norm(S_cd[:,j]):.3e}, "
                  f"|dpc|={np.linalg.norm(S_pc[:,j]):.3e} "
                  f"({time.time()-t_j:.1f}s)", flush=True)

        # Whiten
        S_cd_w = S_cd / sigma_cd_w
        S_pc_w = S_pc / sigma_pc_w
        S_both_w = np.vstack([S_cd_w, S_pc_w])

        # Save raw + whitened
        np.savetxt(os.path.join(OUT_FIM, f"S_cd_raw_{step_label}.csv"), S_cd,
                   header=",".join(PARAM_NAMES), delimiter=",", comments="")
        np.savetxt(os.path.join(OUT_FIM, f"S_pc_raw_{step_label}.csv"), S_pc,
                   header=",".join(PARAM_NAMES), delimiter=",", comments="")

        # Compute SVD + FIM eigendecomposition for each observable subset
        per_subset = {}
        for name, S_w in [("cd", S_cd_w), ("pc", S_pc_w), ("both", S_both_w)]:
            if not np.all(np.isfinite(S_w)):
                per_subset[name] = {"error": "non-finite sensitivity"}
                continue
            try:
                U_svd, sv, VT = np.linalg.svd(S_w, full_matrices=False)
                F = S_w.T @ S_w
                evals, evecs = np.linalg.eigh(F)  # ascending
                # smallest eigenvector = weak direction
                weak_eigvec = evecs[:, 0]
                strong_eigvec = evecs[:, -1]
                cond = float(evals[-1] / max(evals[0], 1e-30))
                per_subset[name] = {
                    "singular_values": sv.tolist(),
                    "fim_eigenvalues": evals.tolist(),
                    "condition_number": cond,
                    "weak_eigenvector": weak_eigvec.tolist(),
                    "strong_eigenvector": strong_eigvec.tolist(),
                    "weak_eigvec_components": dict(zip(PARAM_NAMES, weak_eigvec.tolist())),
                    "fim_diagonal": np.diag(F).tolist(),
                }
                print(f"    {name:>4}: sv={[f'{s:.2e}' for s in sv]}, "
                      f"cond(F)={cond:.2e}")
                print(f"          weak eigvec ({PARAM_NAMES}) = "
                      f"{[f'{v:+.3f}' for v in weak_eigvec.tolist()]}")
            except Exception as e:
                per_subset[name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"    {name:>4}: ERROR {e}")

        fim_results[step_label] = {
            "h": h_vec.tolist(),
            "sigma_cd_whiten": sigma_cd_w,
            "sigma_pc_whiten": sigma_pc_w,
            "subsets": per_subset,
        }

    # Compare CD-only vs CD+PC: does adding PC increase the smallest singular
    # value?  This is the central diagnostic GPT asked for.
    print()
    print("  --- Rank comparison (smaller step) ---")
    cd_subset = fim_results["small"]["subsets"]["cd"]
    both_subset = fim_results["small"]["subsets"]["both"]
    if "singular_values" in cd_subset and "singular_values" in both_subset:
        sv_cd = np.array(cd_subset["singular_values"])
        sv_both = np.array(both_subset["singular_values"])
        print(f"    smallest sv (cd only):  {sv_cd[-1]:.3e}")
        print(f"    smallest sv (cd + pc):  {sv_both[-1]:.3e}")
        rank_gain = sv_both[-1] / max(sv_cd[-1], 1e-30)
        print(f"    sv_min ratio (both/cd): {rank_gain:.3f}x")
        print(f"    cond(F) cd only:        {cd_subset['condition_number']:.3e}")
        print(f"    cond(F) cd + pc:        {both_subset['condition_number']:.3e}")
        cond_improve = cd_subset["condition_number"] / max(both_subset["condition_number"], 1e-30)
        print(f"    cond improvement:       {cond_improve:.3f}x lower with PC")
        fim_results["rank_comparison_small_step"] = {
            "sv_min_cd_only": float(sv_cd[-1]),
            "sv_min_cd_plus_pc": float(sv_both[-1]),
            "sv_min_ratio_both_over_cd": float(rank_gain),
            "cond_cd_only": cd_subset["condition_number"],
            "cond_cd_plus_pc": both_subset["condition_number"],
            "cond_improvement_factor": float(cond_improve),
        }

    fim_path = os.path.join(OUT_FIM, "fim_results.json")
    with open(fim_path, "w") as f:
        json.dump(fim_results, f, indent=2)
    print(f"\n  Saved: {fim_path}")

    # ----------------------------------------------------------------
    # Step 3: Line profile from theta_recovered to theta_true
    #
    # theta_recovered comes from the clean-data L-BFGS-B endpoint
    # (StudyResults/v18_logc_joint_observable/noise_0.0pct/result.json).
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("Step 3: Line profile from theta_recovered (clean) to theta_true")
    print("=" * 72)

    # Hardcoded from the clean-data inversion result (commit-ready)
    theta_recovered = np.array([
        np.log(1.5223e-3),  # log_k0_1
        np.log(6.2236e-5),  # log_k0_2
        0.5897,             # alpha_1
        0.4766,             # alpha_2
    ])
    print(f"  theta_recovered = {theta_recovered.tolist()}")
    print(f"  theta_true      = {theta_true.tolist()}")
    print(f"  delta           = {(theta_true - theta_recovered).tolist()}")

    # Use the EXACT objective from v18_logc_joint_observable.py to make J
    # values directly comparable: range-normalized residuals against the
    # same noisy targets.  But also report unweighted residuals so the
    # "true minimum at TRUE" interpretation is clean.
    cd_scale = float(np.max(np.abs(target_cd)))
    pc_scale = float(np.max(np.abs(target_pc)))
    inv_var_cd = 1.0 / (cd_scale ** 2)
    inv_var_pc = 1.0 / (pc_scale ** 2)
    # Targets are CLEAN (noise=0) so J=0 at TRUE by construction
    target_cd_use = target_cd; target_pc_use = target_pc

    t_grid = np.concatenate([np.linspace(0, 1, 21)])
    profile_rows = []
    print(f"\n  Evaluating J(t) at {len(t_grid)} points along the line...", flush=True)
    for k, t in enumerate(t_grid):
        theta_t = (1 - t) * theta_recovered + t * theta_true
        t0 = time.time()
        cds, pcs = solve_curve_warm(theta_t, true_cache)
        ok = np.isfinite(cds) & np.isfinite(pcs)
        n_ok = int(ok.sum())
        if n_ok < 3:
            J_cd = J_pc = J_total = float("nan")
            print(f"  t={t:.3f}: only {n_ok} pts converged  ({time.time()-t0:.1f}s)",
                  flush=True)
        else:
            r_cd = (cds[ok] - target_cd_use[ok])
            r_pc = (pcs[ok] - target_pc_use[ok])
            J_cd = 0.5 * inv_var_cd * float(np.sum(r_cd ** 2))
            J_pc = 0.5 * inv_var_pc * float(np.sum(r_pc ** 2))
            J_total = J_cd + J_pc
            print(f"  t={t:.3f}: J={J_total:.4e} (cd={J_cd:.3e}, pc={J_pc:.3e})  "
                  f"{n_ok}/{NV}  ({time.time()-t0:.1f}s)", flush=True)
        profile_rows.append({
            "t": float(t),
            "J_total": float(J_total) if np.isfinite(J_total) else None,
            "J_cd": float(J_cd) if np.isfinite(J_cd) else None,
            "J_pc": float(J_pc) if np.isfinite(J_pc) else None,
            "n_ok": n_ok,
            "theta": theta_t.tolist(),
            "cds": cds.tolist(),
            "pcs": pcs.tolist(),
        })

    # Diagnose: monotone descent (= optimizer failure) or rise-then-fall (= true basin)?
    finite_J = [r["J_total"] for r in profile_rows if r["J_total"] is not None]
    if len(finite_J) >= 5:
        J_arr = np.array(finite_J)
        # Monotone if max(J) is at one of the endpoints (no interior peak)
        argmax = int(np.argmax(J_arr))
        is_monotone = (argmax == 0 or argmax == len(J_arr) - 1)
        diagnosis = ("MONOTONE: J(t) descends from recovered (t=0) to TRUE (t=1) "
                     "without barrier — recovered point is OPTIMIZER FAILURE, "
                     "not a true local minimum"
                     if is_monotone and J_arr[0] > J_arr[-1]
                     else "BARRIER: J(t) has interior maximum at t={:.2f} — "
                          "recovered point is in a SEPARATE BASIN from TRUE".format(
                              t_grid[argmax]))
        print()
        print(f"  J(0) [recovered] = {J_arr[0]:.4e}")
        print(f"  J(1) [TRUE]      = {J_arr[-1]:.4e}")
        print(f"  J(0.5) midpoint  = {J_arr[len(J_arr)//2]:.4e}")
        print(f"  argmax J at t = {t_grid[argmax]:.3f}")
        print(f"\n  >>> {diagnosis}")
    else:
        diagnosis = f"insufficient finite J values ({len(finite_J)}/{len(profile_rows)})"
        print(f"\n  >>> Diagnostic INCONCLUSIVE: {diagnosis}")

    profile_path = os.path.join(OUT_PROFILE, "profile.json")
    with open(profile_path, "w") as f:
        json.dump({
            "theta_recovered": theta_recovered.tolist(),
            "theta_true": theta_true.tolist(),
            "param_names": PARAM_NAMES,
            "V_GRID": V_GRID.tolist(),
            "target_cd": target_cd.tolist(),
            "target_pc": target_pc.tolist(),
            "weighting": {"inv_var_cd": inv_var_cd, "inv_var_pc": inv_var_pc,
                          "scheme": "range-normalized (1/scale^2)"},
            "diagnosis": diagnosis,
            "profile": profile_rows,
        }, f, indent=2)
    print(f"\n  Saved: {profile_path}")

    print()
    print("=" * 72)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 72)
    print(f"  FIM:     {OUT_FIM}/")
    print(f"  Profile: {OUT_PROFILE}/")


if __name__ == "__main__":
    main()
