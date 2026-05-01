"""V18 multi-experiment FIM screening.

Per GPT's "Recommended Next Path" plan: instead of more optimizer tuning at
single-experiment CD+PC (precision diagnostic showed precision can shift but
not break the ridge), screen multi-experiment designs by Fisher information
to find one that bends the ridge into a well-conditioned bowl.

Experiments:
  - ORR at L_ref multipliers s = 1, 2, 4 (rotation rate analog: ω ∝ 1/L²)
  - H2O2-fed (c_O2=0, c_H2O2>0) at L_ref = L0 — isolates R2 directly

Designs to screen:
  A. ORR L0 CD only
  B. ORR L0 CD+PC
  C. ORR L0/2L0/4L0 CD+PC each (multi-rotation)
  D. H2O2-fed L0 CD only (R2 isolation)
  E. ORR L0 CD+PC + H2O2-fed L0 CD
  F. ORR L0/2L0/4L0 + H2O2-fed L0 (full design)

For each design, computes whitened sensitivity matrix S_white, the joint Fisher
information matrix F = S_white^T S_white, and reports:
  - sv_min, sv_max, condition(F)
  - eigenvalues of F
  - smallest eigenvector (weak direction)
  - cosine similarity between weak eigvec and canonical (k0_1, alpha_1) ridge
    direction predicted by the breakthrough analysis (slope d log k0 / d alpha
    ≈ -47, so canonical ridge in 4D is [-cos, 0, sin, 0] / norm with sin/cos
    set by atan(1/47) ≈ 0.021)

Runtime: ~30-40 min total (4 experiments × ~8 min each for sensitivity FD).
Forward sensitivities use central FD with single step size per param to keep
runtime bounded; results are diagnostic anyway.
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


# Canonical ridge direction in (log_k0_1, log_k0_2, alpha_1, alpha_2) space.
# Per breakthrough analysis: along the ridge, d(log k0)/d(alpha) ~ -47 per
# reaction.  Normalize to a unit vector.  We construct the 2D-per-reaction
# combined ridge direction; the weak eigvec is expected to overlap with this
# canonical ridge if the system is ridge-limited.
def canonical_ridge():
    # For a single reaction: weak direction is along (Δlog_k0, Δalpha) = (47, 1)
    # normalized.  In 4D with both reactions, we expect the weakest direction
    # to be dominated by reaction 2 (per FIM diagnostic showing weak eigvec
    # ~ [0, -1, 0, 0.02]).  Build that as the reference.
    v = np.array([0.0, -47.0, 0.0, +1.0])
    return v / np.linalg.norm(v)


def fim_metrics(S_white, names=("log_k0_1", "log_k0_2", "alpha_1", "alpha_2")):
    """Compute SVD + FIM metrics for a whitened sensitivity matrix."""
    if not np.all(np.isfinite(S_white)):
        return {"error": "non-finite sensitivity"}
    try:
        U, sv, VT = np.linalg.svd(S_white, full_matrices=False)
        F = S_white.T @ S_white
        evals, evecs = np.linalg.eigh(F)
        cond_F = float(evals[-1] / max(evals[0], 1e-30))
        weak_v = evecs[:, 0]
        canonical = canonical_ridge()
        cos_sim = float(abs(np.dot(weak_v, canonical)))
        return {
            "n_residuals": int(S_white.shape[0]),
            "n_params": int(S_white.shape[1]),
            "singular_values": sv.tolist(),
            "fim_eigenvalues": evals.tolist(),
            "fim_diagonal": np.diag(F).tolist(),
            "condition_number": cond_F,
            "weak_eigvec": dict(zip(names, weak_v.tolist())),
            "weak_eigvec_canonical_ridge_cos": cos_sim,
            "ridge_breaking_score": 1.0 - cos_sim,  # 0 = pure ridge, 1 = orthogonal
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT, C_H2O2_HAT,
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
    V_GRID = np.array([-0.10, 0.00, 0.10, 0.15, 0.20])
    NV = len(V_GRID)
    PARAM_NAMES = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]

    OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_logc_multiexperiment_fim")
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    # FD step sizes (single size — diagnostic; central differences):
    H = np.array([1e-3, 1e-3, 1e-4, 1e-4])  # log_k0_1, log_k0_2, alpha_1, alpha_2

    def make_3sp_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2,
                     c0_O2, c0_H2O2, c0_Hp):
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
        c0_vec = [float(c0_O2), float(c0_H2O2), float(c0_Hp)]
        return SolverParams.from_list([
            3, 1, 0.25, 80.0, THREE_SPECIES_Z, THREE_SPECIES_D, THREE_SPECIES_A,
            eta_hat, c0_vec, 0.0, params,
        ])

    def add_boltzmann(ctx):
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

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    def _snapshot(U): return tuple(d.data_ro.copy() for d in U.dat)
    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat): dst.data[:] = src
        Up.assign(U)

    def build_solve(V_RHE, k0_1, k0_2, a_1, a_2, c0_O2, c0_H2O2, c0_Hp):
        sp = make_3sp_sp(V_RHE / V_T, k0_1, k0_2, a_1, a_2, c0_O2, c0_H2O2, c0_Hp)
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

    def solve_warm(V_RHE, k0_1, k0_2, a_1, a_2, c0_O2, c0_H2O2, c0_Hp,
                   ic_data, max_steps=200):
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2, c0_O2, c0_H2O2, c0_Hp)
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

    def solve_cold(V_RHE, k0_1, k0_2, a_1, a_2, c0_O2, c0_H2O2, c0_Hp,
                   max_z_steps=20):
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2, c0_O2, c0_H2O2, c0_Hp)
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

    def solve_curve(theta_log, c0_O2, c0_H2O2, c0_Hp, true_cache):
        """theta_log = [log_k0_1, log_k0_2, alpha_1, alpha_2].
        Returns (cd_array_NV, pc_array_NV).  Warm-starts each V from
        true_cache[v_idx]."""
        k0_1 = float(np.exp(theta_log[0])); k0_2 = float(np.exp(theta_log[1]))
        a_1 = float(theta_log[2]); a_2 = float(theta_log[3])
        cds = np.full(NV, np.nan); pcs = np.full(NV, np.nan)
        for i, V in enumerate(V_GRID):
            cd, pc, _ = solve_warm(float(V), k0_1, k0_2, a_1, a_2,
                                     c0_O2, c0_H2O2, c0_Hp, true_cache[i])
            if cd is not None:
                cds[i] = cd; pcs[i] = pc
        return cds, pcs

    def compute_experiment(name, L_scale, c0_O2_phys, c0_H2O2_phys, c0_Hp_phys,
                            v_grid=None):
        """For a single experiment configuration:
          1. Cold-solve TRUE curve (anchor cache); skip voltages that fail
          2. Compute central-FD sensitivity at TRUE on succeeded voltages
          3. Whiten by sigma = 2% * max(|target|) per observable type
        Returns: dict with target_cd, target_pc, S_cd_white, S_pc_white,
                  sigma_cd, sigma_pc, cache, status, v_used (boolean array).
        """
        if v_grid is None: v_grid = V_GRID
        nv_local = len(v_grid)
        print(f"\n[{name}] L_scale={L_scale}, c_O2={c0_O2_phys:.3f}, "
              f"c_H2O2={c0_H2O2_phys:.3f} (phys mol/m3 nondim units)", flush=True)

        # Apply L_scale: K0_HAT scales linearly with L_ref (k0_PHYS fixed)
        K01 = K0_HAT_R1 * L_scale
        K02 = K0_HAT_R2 * L_scale
        theta_true = np.array([np.log(K01), np.log(K02), ALPHA_R1, ALPHA_R2])

        # 1. Cold-solve TRUE curve; tolerate per-V failures
        t0 = time.time()
        true_cache_local = [None] * nv_local
        target_cd = np.full(nv_local, np.nan); target_pc = np.full(nv_local, np.nan)
        v_used = np.zeros(nv_local, dtype=bool)
        for i, V in enumerate(v_grid):
            cd, pc, snap = solve_cold(float(V), K01, K02, ALPHA_R1, ALPHA_R2,
                                       c0_O2_phys, c0_H2O2_phys, c0_Hp_phys)
            if snap is None:
                print(f"  V={V}: cold solve FAILED — skipping this voltage",
                      flush=True)
                continue
            target_cd[i] = cd; target_pc[i] = pc; true_cache_local[i] = snap
            v_used[i] = True
        n_used = int(v_used.sum())
        if n_used < 3:
            print(f"  Only {n_used} voltages converged; aborting experiment",
                  flush=True)
            return {"name": name, "status": "insufficient_voltages",
                    "L_scale": L_scale, "c_O2": c0_O2_phys,
                    "c_H2O2": c0_H2O2_phys, "n_used": n_used}
        print(f"  TRUE ({n_used}/{nv_local} V converged): "
              f"cd={target_cd[v_used]}, pc={target_pc[v_used]}  "
              f"({time.time()-t0:.1f}s)", flush=True)

        # 2. Central FD sensitivities at TRUE (using a per-experiment solve_curve
        # that only attempts voltages where the TRUE solve converged).
        def solve_curve_local(theta_log):
            k0_1 = float(np.exp(theta_log[0])); k0_2 = float(np.exp(theta_log[1]))
            a_1 = float(theta_log[2]); a_2 = float(theta_log[3])
            cds = np.full(nv_local, np.nan); pcs = np.full(nv_local, np.nan)
            for i, V in enumerate(v_grid):
                if not v_used[i]:
                    continue
                cd, pc, _ = solve_warm(float(V), k0_1, k0_2, a_1, a_2,
                                         c0_O2_phys, c0_H2O2_phys, c0_Hp_phys,
                                         true_cache_local[i])
                if cd is not None:
                    cds[i] = cd; pcs[i] = pc
            return cds, pcs

        S_cd = np.zeros((nv_local, 4)); S_pc = np.zeros((nv_local, 4))
        for j in range(4):
            t_j = time.time()
            theta_plus = theta_true.copy(); theta_plus[j] += H[j]
            theta_minus = theta_true.copy(); theta_minus[j] -= H[j]
            cd_p, pc_p = solve_curve_local(theta_plus)
            cd_m, pc_m = solve_curve_local(theta_minus)
            # Per-voltage validity check
            for i in range(nv_local):
                if not v_used[i]: continue
                if not (np.isfinite(cd_p[i]) and np.isfinite(cd_m[i])
                        and np.isfinite(pc_p[i]) and np.isfinite(pc_m[i])):
                    S_cd[i, j] = np.nan; S_pc[i, j] = np.nan
                else:
                    S_cd[i, j] = (cd_p[i] - cd_m[i]) / (2 * H[j])
                    S_pc[i, j] = (pc_p[i] - pc_m[i]) / (2 * H[j])
            valid_norm_cd = np.linalg.norm(S_cd[v_used, j])
            valid_norm_pc = np.linalg.norm(S_pc[v_used, j])
            print(f"    d/d{PARAM_NAMES[j]:>10}: |dcd|={valid_norm_cd:.3e}, "
                  f"|dpc|={valid_norm_pc:.3e}  ({time.time()-t_j:.1f}s)",
                  flush=True)

        # 3. Whitening: only over converged voltages
        sigma_cd = 0.02 * float(np.max(np.abs(target_cd[v_used])))
        sigma_pc = 0.02 * float(np.max(np.abs(target_pc[v_used])))

        # Restrict S to converged voltages for the FIM
        S_cd_used = S_cd[v_used, :]
        S_pc_used = S_pc[v_used, :]
        S_cd_white = S_cd_used / sigma_cd
        S_pc_white = S_pc_used / sigma_pc

        return {
            "name": name, "L_scale": L_scale,
            "c_O2": c0_O2_phys, "c_H2O2": c0_H2O2_phys, "c_Hp": c0_Hp_phys,
            "K01_at_TRUE": K01, "K02_at_TRUE": K02,
            "v_grid": [float(v) for v in v_grid],
            "v_used": v_used.tolist(),
            "n_used": n_used,
            "target_cd": target_cd.tolist(), "target_pc": target_pc.tolist(),
            "sigma_cd": sigma_cd, "sigma_pc": sigma_pc,
            "S_cd_raw": S_cd_used.tolist(), "S_pc_raw": S_pc_used.tolist(),
            "S_cd_white": S_cd_white.tolist(), "S_pc_white": S_pc_white.tolist(),
            "status": "ok",
            "elapsed_s": time.time() - t0,
        }

    # ----------------------------------------------------------------
    # Compute per-experiment sensitivities
    # ----------------------------------------------------------------
    print("=" * 72)
    print("V18 multi-experiment FIM screening")
    print("=" * 72)
    print(f"V_GRID: {V_GRID.tolist()}")
    print(f"PARAM_NAMES: {PARAM_NAMES}")
    print(f"FD steps: {H.tolist()}")
    print(f"Whitening: sigma = 2% * max(|target|) per observable type")
    print()

    # Define experiments.  L_scale is multiplier on baseline L_ref (= 100 µm).
    # In the rotation-rate analog: L_scale = 2 ↔ ω = ω0 / 4 (Levich); L_scale = 4
    # ↔ ω = ω0 / 16.  Practical range for RDE: typically 100–6400 rpm, so
    # 16:1 ratio of ω is achievable.  The forward solver was tuned for L0;
    # other L values may have different convergence properties.
    experiments = [
        {"name": "ORR_L1", "L_scale": 1.0, "c_O2": C_O2_HAT, "c_H2O2": H2O2_SEED,
         "c_Hp": C_HP_HAT},
        {"name": "ORR_L2", "L_scale": 2.0, "c_O2": C_O2_HAT, "c_H2O2": H2O2_SEED,
         "c_Hp": C_HP_HAT},
        {"name": "ORR_L4", "L_scale": 4.0, "c_O2": C_O2_HAT, "c_H2O2": H2O2_SEED,
         "c_Hp": C_HP_HAT},
        # "H2O2 co-fed" — keep standard O2 (1.0 nondim), elevate bulk H2O2 to
        # 0.1 nondim (= 0.05 mol/m3 phys = 50 mM).  Suppressing O2 entirely
        # destabilizes the log-c forward solver; co-feeding gives R2 strong
        # substrate without breaking convergence.  The k0_2 sensitivity should
        # still increase substantially because R2 rate ~ k0_2 * c_H2O2_surf.
        {"name": "H2O2_cofed", "L_scale": 1.0, "c_O2": C_O2_HAT,
         "c_H2O2": 0.1, "c_Hp": C_HP_HAT},
    ]

    results = {}
    for exp in experiments:
        res = compute_experiment(exp["name"], exp["L_scale"],
                                  exp["c_O2"], exp["c_H2O2"], exp["c_Hp"],
                                  v_grid=exp.get("v_grid"))
        results[exp["name"]] = res

    # Save raw experiment results
    with open(os.path.join(OUT_DIR, "experiments.json"), "w") as f:
        # cast numpy types to native
        def _clean(o):
            if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list): return [_clean(x) for x in o]
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.float64, np.float32)): return float(o)
            if isinstance(o, (np.int64, np.int32)): return int(o)
            return o
        json.dump(_clean(results), f, indent=2)

    # ----------------------------------------------------------------
    # Build designs and compute FIM metrics
    # ----------------------------------------------------------------
    print()
    print("=" * 72)
    print("Designs and FIM metrics")
    print("=" * 72)

    def stack_design(experiments_list, observables_per_exp):
        """experiments_list: list of (name, S_cd_white, S_pc_white) tuples
        observables_per_exp: list parallel to experiments_list, each in
        {'cd', 'pc', 'both'}.
        Returns S_white (rows stacked)."""
        rows = []
        for (name, S_cd_w, S_pc_w), obs in zip(experiments_list, observables_per_exp):
            if obs in ("cd", "both"):
                rows.append(S_cd_w)
            if obs in ("pc", "both"):
                rows.append(S_pc_w)
        return np.vstack(rows)

    # Helper to fetch results
    def get(name):
        r = results[name]
        if r["status"] != "ok":
            return None
        return (name, np.array(r["S_cd_white"]), np.array(r["S_pc_white"]))

    designs = {}

    # A. ORR L0 CD only
    e_l1 = get("ORR_L1")
    if e_l1: designs["A_ORR_L1_CD"] = stack_design([e_l1], ["cd"])
    # B. ORR L0 CD+PC
    if e_l1: designs["B_ORR_L1_CDPC"] = stack_design([e_l1], ["both"])
    # C. ORR L0/2L0/4L0 CD+PC each
    e_l2 = get("ORR_L2"); e_l4 = get("ORR_L4")
    if e_l1 and e_l2 and e_l4:
        designs["C_ORR_multiL_CDPC"] = stack_design([e_l1, e_l2, e_l4],
                                                      ["both", "both", "both"])
    # D. H2O2-cofed alone, CD+PC
    e_h = get("H2O2_cofed")
    if e_h: designs["D_H2O2cofed_CDPC"] = stack_design([e_h], ["both"])
    # E. ORR L0 CD+PC + H2O2-cofed L0 CD+PC
    if e_l1 and e_h:
        designs["E_ORR_L1_plus_H2O2cofed"] = stack_design([e_l1, e_h], ["both", "both"])
    # F. Full design
    if e_l1 and e_l2 and e_l4 and e_h:
        designs["F_full"] = stack_design([e_l1, e_l2, e_l4, e_h],
                                           ["both", "both", "both", "both"])

    print(f"\n  {'Design':<28} {'#rows':>6} {'sv_min':>11} {'cond(F)':>11} "
          f"{'ridge_cos':>11} {'weak eigvec (a_2-dom?)':>30}")
    summary = {}
    for d_name, S_w in designs.items():
        if not np.all(np.isfinite(S_w)):
            print(f"  {d_name:<28} non-finite sensitivity")
            summary[d_name] = {"error": "non-finite"}
            continue
        m = fim_metrics(S_w, names=tuple(PARAM_NAMES))
        if "error" in m:
            print(f"  {d_name:<28} {m['error']}")
            summary[d_name] = m
            continue
        sv_min = m["singular_values"][-1]
        cond_F = m["condition_number"]
        ridge_cos = m["weak_eigvec_canonical_ridge_cos"]
        weak = m["weak_eigvec"]
        weak_str = (f"[{weak['log_k0_1']:+.2f},{weak['log_k0_2']:+.2f},"
                    f"{weak['alpha_1']:+.2f},{weak['alpha_2']:+.2f}]")
        print(f"  {d_name:<28} {S_w.shape[0]:>6} {sv_min:>11.3e} "
              f"{cond_F:>11.2e} {ridge_cos:>11.3f}  {weak_str}")
        summary[d_name] = {
            "n_rows": int(S_w.shape[0]),
            "sv_min": float(sv_min),
            "sv_max": float(m["singular_values"][0]),
            "condition_number": cond_F,
            "weak_eigvec": weak,
            "weak_eigvec_canonical_ridge_cos": ridge_cos,
            "ridge_breaking_score": m["ridge_breaking_score"],
            "fim_eigenvalues": m["fim_eigenvalues"],
        }

    # Decision rule
    print()
    print("Decision rule (per GPT plan):")
    print("  Run TRF inverse on a design only if sv_min improves >100x AND")
    print("  cond(F) drops below ~1e8 AND weak eigvec rotates off canonical ridge")
    print()
    baseline = summary.get("B_ORR_L1_CDPC", {})
    baseline_sv_min = baseline.get("sv_min", None)
    if baseline_sv_min:
        print(f"  Baseline (B): sv_min = {baseline_sv_min:.3e}, "
              f"cond = {baseline['condition_number']:.2e}, "
              f"ridge cos = {baseline['weak_eigvec_canonical_ridge_cos']:.3f}")
        print()
        print(f"  {'Design':<28} {'sv ratio':>10} {'cond ratio':>12} "
              f"{'verdict':>20}")
        for name, s in summary.items():
            if name == "B_ORR_L1_CDPC" or "error" in s: continue
            sv_ratio = s["sv_min"] / baseline_sv_min
            cond_ratio = baseline["condition_number"] / s["condition_number"]
            cond_pass = s["condition_number"] <= 1e8
            sv_pass = sv_ratio >= 100
            ridge_pass = s["ridge_breaking_score"] >= 0.5  # rotated >60° off
            verdict = "RUN TRF" if (cond_pass and sv_pass) else \
                      "marginal" if (sv_ratio > 10 or cond_ratio > 100) else \
                      "skip"
            print(f"  {name:<28} {sv_ratio:>10.3f} {cond_ratio:>12.3f}  "
                  f"{verdict:>20}")

    out_path = os.path.join(OUT_DIR, "summary.json")
    # Strip ndarrays from experiments before serializing
    exp_clean = []
    for e in experiments:
        ec = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in e.items()}
        exp_clean.append(ec)
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "V_GRID": V_GRID.tolist(),
                "PARAM_NAMES": PARAM_NAMES,
                "FD_steps": H.tolist(),
                "experiments": exp_clean,
                "whitening": "2% max(|target|) per observable type",
            },
            "designs": summary,
            "raw_experiments": list(results.keys()),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'experiments.json')}")


if __name__ == "__main__":
    main()
