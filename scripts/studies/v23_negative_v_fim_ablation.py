"""V23 -- Negative-V FIM ablation (extend grid to V <= -0.30).

Per ``docs/TODO_extend_inverse_v_range_negative.md`` (2026-04-30):

  The existing V20 unified data
  (``StudyResults/v20_voltage_grid_fim_ablation/unified_13V_cap50_lograte/``)
  has FD sensitivity rows at 11 voltages, but V = -0.50 and V = -0.30
  failed cold-ramp at the time. Voltage continuation now succeeds at all
  V in [-0.5, +0.6] (see ``scripts/plot_iv_curves_3sp_true.py``). The
  IV plot at TRUE shows PC carries ~5 decades of additional information
  at V <= -0.10 -- a regime where R1 sits in its anodic /
  peroxide-consuming branch with R2 essentially off, which should rotate
  the post-rebuild log_k0_1 weak Fisher direction (V20 / V22 found
  |log_k0_1| >= 0.99 across all grids tested in V >= -0.20).

  This script:
    1. Cold-anchors at V = 0, sweeps down to V = -0.50 with warm
       voltage continuation, saving the converged snapshot at each V.
    2. Computes central-FD sensitivities of (CD, PC, R2) w.r.t.
       (log_k0_1, log_k0_2, alpha_1, alpha_2) at each new V via warm
       restarts from that V's snapshot. Same H, same conventions as
       ``scripts/studies/v19_bv_clip_audit.py``.
    3. Merges the new rows with the existing V20 unified-13V rows.
    4. Runs V20-style FIM ablation across grids:
         G0       = [-0.10, +0.10, +0.20, +0.30, +0.40, +0.50, +0.60]
         G_neg1   = G0 + [-0.20]
         G_neg2   = G0 + [-0.20, -0.30]
         G_neg3   = G0 + [-0.20, -0.30, -0.40, -0.50]
       under both global_max and local_rel noise models.

  Pass criterion (per the TODO):
    weak eigvec |log_k0_1| component drops below 0.95 under at least
    one noise model on at least one extended grid.

  Caveat: under global_max, |PC| at V = -0.50 (~0.18) inflates sigma_pc
  by ~1e4x and demotes every V > 0 PC row (V20 already showed this trap
  with V = -0.20: cond went 1.79e7 -> 7.77e8). local_rel is the
  publishable noise model for this comparison; global_max is reported
  only to show the failure mode.

Output: ``StudyResults/v23_negative_v_fim_ablation/``
    sensitivities_extended.json   (S rows per V incl. new + reused)
    fim_by_grid.json              (FIM metrics per grid x noise model)
    weak_eigvec_by_grid.csv
    leverage_by_voltage.csv
    summary.md

Usage:
    python scripts/studies/v23_negative_v_fim_ablation.py
    python scripts/studies/v23_negative_v_fim_ablation.py --extra-vs -0.45 -0.35
    python scripts/studies/v23_negative_v_fim_ablation.py --skip-solves \
        --extended-json <existing>/sensitivities_extended.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


PARAM_NAMES: tuple[str, ...] = ("log_k0_1", "log_k0_2", "alpha_1", "alpha_2")
H_FD = np.array([1e-3, 1e-3, 1e-4, 1e-4])

# G0 baseline grid (HANDOFF_10/11; matches V20 G0_current).
G0 = (-0.10, +0.10, +0.20, +0.30, +0.40, +0.50, +0.60)
# Default new voltages to compute via continuation (missing from V20 unified).
DEFAULT_NEW_VS = (-0.50, -0.40, -0.30)


def canonical_ridge() -> np.ndarray:
    """Old (k0_2, alpha_2) ridge direction; kept for ridge_cos comparability."""
    v = np.array([0.0, -47.0, 0.0, +1.0])
    return v / np.linalg.norm(v)


def fim_metrics(S_white: np.ndarray) -> dict[str, Any]:
    if not np.all(np.isfinite(S_white)):
        return {"error": "non-finite sensitivity"}
    _, sv, _ = np.linalg.svd(S_white, full_matrices=False)
    F = S_white.T @ S_white
    evals, evecs = np.linalg.eigh(F)
    cond_F = float(evals[-1] / max(evals[0], 1e-300))
    weak_v = evecs[:, 0]
    cos_sim = float(abs(np.dot(weak_v, canonical_ridge())))
    diag = np.diag(F)
    corr = F / np.sqrt(np.outer(diag, diag) + 1e-300)
    return {
        "n_residuals": int(S_white.shape[0]),
        "n_params": int(S_white.shape[1]),
        "singular_values": sv.tolist(),
        "fim_eigenvalues": evals.tolist(),
        "fim_eigenvectors": evecs.tolist(),
        "fim_diagonal": diag.tolist(),
        "correlation_matrix": corr.tolist(),
        "condition_number": cond_F,
        "weak_eigvec": dict(zip(PARAM_NAMES, weak_v.tolist())),
        "weak_eigvec_canonical_ridge_cos": cos_sim,
        "weak_eigvec_log_k0_1_component": float(abs(weak_v[0])),
    }


def per_voltage_leverage(S_white: np.ndarray, n_obs: int) -> list[float]:
    NV = S_white.shape[0] // n_obs
    rn2 = (S_white ** 2).sum(axis=1)
    total = rn2.sum()
    return [
        float(rn2[[i + k * NV for k in range(n_obs)]].sum() / total)
        for i in range(NV)
    ]


def per_voltage_weak_dir_contribution(
    S_white: np.ndarray, n_obs: int
) -> list[float]:
    U, _, _ = np.linalg.svd(S_white, full_matrices=False)
    weak_left = U[:, -1]
    NV = S_white.shape[0] // n_obs
    return [
        float((weak_left[[i + k * NV for k in range(n_obs)]] ** 2).sum())
        for i in range(NV)
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--new-vs", type=float, nargs="+", default=list(DEFAULT_NEW_VS),
        help=(
            "Voltages at which to compute new FD sensitivity rows via "
            "voltage continuation. Default: -0.50 -0.40 -0.30."
        ),
    )
    p.add_argument(
        "--extra-vs", type=float, nargs="*", default=[],
        help="Additional voltages to add (e.g. -0.45 -0.35 -0.25).",
    )
    p.add_argument(
        "--anchor-v", type=float, default=0.0,
        help="Cold-start anchor for voltage continuation (default 0.0).",
    )
    p.add_argument(
        "--max-z-steps", type=int, default=20,
        help="z-ramp steps in cold solve at the anchor (default 20).",
    )
    p.add_argument(
        "--unified-json", type=str,
        default=os.path.join(
            _ROOT, "StudyResults", "v20_voltage_grid_fim_ablation",
            "unified_13V_cap50_lograte", "fim_by_cap.json",
        ),
        help="Path to V20 unified fim_by_cap.json to merge (cap=50, lograte).",
    )
    p.add_argument(
        "--skip-solves", action="store_true",
        help=(
            "Skip the forward+FD step. Reuse a previously saved "
            "sensitivities_extended.json from --extended-json."
        ),
    )
    p.add_argument(
        "--extended-json", type=str, default=None,
        help="Path to a previously saved sensitivities_extended.json.",
    )
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(
            _ROOT, "StudyResults", "v23_negative_v_fim_ablation",
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Forward solver setup (mirrors v19 + plot_iv conventions exactly).
# ---------------------------------------------------------------------------

def _build_solver_factory():
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, _make_bv_convergence_cfg, SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78
    H2O2_SEED = 1e-4
    THREE_SPECIES_Z = [0, 0, 1]
    THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.params import SolverParams

    SP_DICT = {
        k: v for k, v in dict(SNES_OPTS_CHARGED).items()
        if k.startswith(("snes_", "ksp_", "pc_", "mat_"))
    }
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    def make_3sp_sp(eta_hat, k0_1, k0_2, a_1, a_2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = _make_bv_convergence_cfg(log_rate=True)
        params["nondim"] = _make_nondim_cfg()
        rxn1 = {
            "k0": k0_1, "alpha": a_1,
            "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
            "reversible": True, "E_eq_v": E_EQ_R1,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
            ],
        }
        rxn2 = {
            "k0": k0_2, "alpha": a_2,
            "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
            "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
            "reversible": False, "E_eq_v": E_EQ_R2,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
            ],
        }
        params["bv_bc"] = {
            "reactions": [rxn1, rxn2],
            "k0": [k0_1] * 3, "alpha": [a_1] * 3,
            "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
            "E_eq_v": 0.0,
            "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
        }
        c0_vec = [float(C_O2_HAT), float(H2O2_SEED), float(C_HP_HAT)]
        return SolverParams.from_list([
            3, 1, 0.25, 80.0, THREE_SPECIES_Z, THREE_SPECIES_D, THREE_SPECIES_A,
            eta_hat, c0_vec, 0.0, params,
        ])

    def add_boltzmann(ctx):
        W = ctx["W"]; U = ctx["U"]
        scaling = ctx["nondim"]
        phi = fd.split(U)[-1]
        w = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=ctx["mesh"])
        charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(
            fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0),
        )
        ctx["F_res"] -= (
            charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
        )
        ctx["J_form"] = fd.derivative(ctx["F_res"], U)
        return ctx

    def _snapshot(U):
        return tuple(d.data_ro.copy() for d in U.dat)

    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat):
            dst.data[:] = src
        Up.assign(U)

    def build_solve(V_RHE, k0_1, k0_2, a_1, a_2):
        sp = make_3sp_sp(V_RHE / V_T, k0_1, k0_2, a_1, a_2)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"],
        )
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT)
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        of_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )
        of_r1 = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=1.0,
        )
        of_r2 = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=1, scale=1.0,
        )
        z_nominal = [float(sp[4][i]) for i in range(ctx["n_species"])]
        return ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal

    def make_run_ss(ctx, sol, of_cd):
        U = ctx["U"]; Up = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_init = 0.25

        def run_ss(max_steps: int) -> bool:
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for _ in range(1, max_steps + 1):
                try:
                    sol.solve()
                except Exception:
                    return False
                Up.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    d = abs(fv - prev_flux)
                    sv_loc = max(abs(fv), abs(prev_flux), 1e-8)
                    if d / sv_loc <= 1e-4 or d <= 1e-8:
                        sc += 1
                    else:
                        sc = 0
                    if prev_delta and d > 0:
                        r = prev_delta / d
                        dt_val = (
                            min(dt_val * min(r, 4), dt_init * 20)
                            if r > 1
                            else max(dt_val * 0.5, dt_init)
                        )
                        dt_const.assign(dt_val)
                    prev_delta = d
                prev_flux = fv
                if sc >= 4:
                    return True
            return False

        return run_ss

    def assemble_obs(of_cd, of_pc, of_r1, of_r2):
        return (
            float(fd.assemble(of_cd)),
            float(fd.assemble(of_pc)),
            float(fd.assemble(of_r1)),
            float(fd.assemble(of_r2)),
        )

    def solve_cold(V_RHE, k0_1, k0_2, a_1, a_2, max_z_steps=20):
        ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2,
        )
        U = ctx["U"]; zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        achieved_z = 0.0
        with adj.stop_annotating():
            for zci in zc:
                zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(200):
                return None
            for z_val in np.linspace(0, 1, max_z_steps + 1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                if run_ss(120):
                    achieved_z = float(z_val)
                else:
                    _restore(ckpt, U, ctx["U_prev"])
                    break
            if achieved_z < 1.0 - 1e-3:
                return None
        cd, pc, r1, r2 = assemble_obs(of_cd, of_pc, of_r1, of_r2)
        return {"cd": cd, "pc": pc, "r1": r1, "r2": r2,
                "snap": _snapshot(U), "z_achieved": achieved_z}

    def solve_warm(V_RHE, k0_1, k0_2, a_1, a_2, ic_data, max_steps=200):
        ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2,
        )
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            _restore(ic_data, U, Up)
            for i in range(n):
                zc[i].assign(z_nominal[i])
            paf.assign(V_RHE / V_T)
            if not run_ss(max_steps):
                return None
        cd, pc, r1, r2 = assemble_obs(of_cd, of_pc, of_r1, of_r2)
        return {"cd": cd, "pc": pc, "r1": r1, "r2": r2, "snap": _snapshot(U)}

    return {
        "solve_cold": solve_cold,
        "solve_warm": solve_warm,
        "K0_HAT_R1": K0_HAT_R1,
        "K0_HAT_R2": K0_HAT_R2,
        "ALPHA_R1": ALPHA_R1,
        "ALPHA_R2": ALPHA_R2,
    }


# ---------------------------------------------------------------------------
# Forward sweep: anchor cold + warm-sweep down through requested V's, then
# at each V compute central-FD sensitivities via warm restarts at TRUE.
# ---------------------------------------------------------------------------

def compute_extended_rows(
    factory: dict[str, Any],
    new_vs: list[float],
    anchor_v: float,
    max_z_steps: int,
) -> dict[str, Any]:
    solve_cold = factory["solve_cold"]
    solve_warm = factory["solve_warm"]
    K0_1 = factory["K0_HAT_R1"]
    K0_2 = factory["K0_HAT_R2"]
    A_1 = factory["ALPHA_R1"]
    A_2 = factory["ALPHA_R2"]
    theta_true = np.array([np.log(K0_1), np.log(K0_2), A_1, A_2])

    print("\n--- Forward continuation ---", flush=True)
    print(f"Anchor cold-solve at V = {anchor_v:+.3f}", flush=True)
    t0 = time.time()
    res_anchor = solve_cold(anchor_v, K0_1, K0_2, A_1, A_2, max_z_steps)
    if res_anchor is None:
        raise RuntimeError(
            f"Anchor cold-solve at V = {anchor_v:+.3f} failed.",
        )
    print(
        f"  V = {anchor_v:+.3f}: cd = {res_anchor['cd']:+.4e}, "
        f"pc = {res_anchor['pc']:+.4e}  ({time.time() - t0:.1f}s)",
        flush=True,
    )

    # Walk monotonically toward more negative V's, warm-starting from
    # the previous snapshot. We assume new_vs is sorted descending from
    # nearest-to-anchor to farthest-from-anchor. Insert intermediate
    # voltages at 0.05 spacing if the user-requested step is large,
    # to keep continuation stable (mirrors plot_iv 0.05 spacing).
    step = 0.05
    vs_sorted = sorted(set(float(v) for v in new_vs))
    vs_negative = [v for v in vs_sorted if v < anchor_v - 1e-6]
    vs_positive = [v for v in vs_sorted if v > anchor_v + 1e-6]

    def make_path(target_vs: list[float], descending: bool):
        if not target_vs:
            return []
        target_vs = sorted(target_vs, reverse=descending)
        path: list[float] = []
        cur = anchor_v
        for target in target_vs:
            n_steps = max(int(round(abs(target - cur) / step)), 1)
            grid = np.linspace(cur, target, n_steps + 1)[1:]
            for vv in grid:
                path.append(float(vv))
            cur = target
        return path

    path_neg = make_path(vs_negative, descending=True)
    path_pos = make_path(vs_positive, descending=False)
    requested = set(round(v, 3) for v in new_vs)

    snaps_at_requested: dict[float, Any] = {}
    obs_at_requested: dict[float, dict[str, float]] = {}

    def walk_path(path: list[float], start_snap: Any) -> None:
        last_snap = start_snap
        for v in path:
            t_v = time.time()
            res = solve_warm(v, K0_1, K0_2, A_1, A_2, last_snap)
            if res is None:
                print(
                    f"  V = {v:+.3f}: warm FAILED, retry cold "
                    f"({time.time() - t_v:.1f}s)",
                    flush=True,
                )
                res = solve_cold(v, K0_1, K0_2, A_1, A_2, max_z_steps)
                if res is None:
                    print(
                        f"  V = {v:+.3f}: BOTH FAILED, abandoning path",
                        flush=True,
                    )
                    return
            last_snap = res["snap"]
            print(
                f"  V = {v:+.3f}: cd = {res['cd']:+.4e}, "
                f"pc = {res['pc']:+.4e}  ({time.time() - t_v:.1f}s)",
                flush=True,
            )
            if round(v, 3) in requested:
                snaps_at_requested[round(v, 3)] = last_snap
                obs_at_requested[round(v, 3)] = {
                    "cd": float(res["cd"]),
                    "pc": float(res["pc"]),
                    "r1": float(res["r1"]),
                    "r2": float(res["r2"]),
                }

    walk_path(path_neg, res_anchor["snap"])
    walk_path(path_pos, res_anchor["snap"])

    # ---- FD sensitivities at TRUE for each requested V ----
    print("\n--- FD sensitivities at TRUE (warm restart) ---", flush=True)
    rows: dict[float, dict[str, list[float]]] = {}
    for v in sorted(snaps_at_requested.keys()):
        snap = snaps_at_requested[v]
        S_cd = np.zeros(4); S_pc = np.zeros(4); S_r2 = np.zeros(4)
        ok = True
        for j in range(4):
            t_j = time.time()
            tp = theta_true.copy(); tp[j] += H_FD[j]
            tm = theta_true.copy(); tm[j] -= H_FD[j]
            kp = (float(np.exp(tp[0])), float(np.exp(tp[1])),
                  float(tp[2]), float(tp[3]))
            km = (float(np.exp(tm[0])), float(np.exp(tm[1])),
                  float(tm[2]), float(tm[3]))
            res_p = solve_warm(v, *kp, snap)
            res_m = solve_warm(v, *km, snap)
            if res_p is None or res_m is None:
                print(
                    f"  V = {v:+.3f}, d/d{PARAM_NAMES[j]:>10}: FD FAILED "
                    f"(p={'OK' if res_p else 'FAIL'}, "
                    f"m={'OK' if res_m else 'FAIL'})",
                    flush=True,
                )
                ok = False
                break
            S_cd[j] = (res_p["cd"] - res_m["cd"]) / (2 * H_FD[j])
            S_pc[j] = (res_p["pc"] - res_m["pc"]) / (2 * H_FD[j])
            S_r2[j] = (res_p["r2"] - res_m["r2"]) / (2 * H_FD[j])
            print(
                f"  V = {v:+.3f}, d/d{PARAM_NAMES[j]:>10}: "
                f"dcd = {S_cd[j]:+.3e}  dpc = {S_pc[j]:+.3e}  "
                f"dr2 = {S_r2[j]:+.3e}  ({time.time() - t_j:.1f}s)",
                flush=True,
            )
        if ok:
            rows[v] = {
                "S_cd": S_cd.tolist(),
                "S_pc": S_pc.tolist(),
                "S_r2": S_r2.tolist(),
                "cd": obs_at_requested[v]["cd"],
                "pc": obs_at_requested[v]["pc"],
                "r1": obs_at_requested[v]["r1"],
                "r2": obs_at_requested[v]["r2"],
            }

    return {
        "anchor_v": float(anchor_v),
        "new_vs": [float(v) for v in sorted(rows.keys())],
        "rows": {str(v): rows[v] for v in sorted(rows.keys())},
    }


# ---------------------------------------------------------------------------
# Merge new rows with existing v20 unified-13V data, then run V20-style
# FIM ablation across grids x noise models.
# ---------------------------------------------------------------------------

def load_unified_v20(unified_json: str) -> dict[str, Any]:
    """Load V20 unified-13V cap=50 fim rows.

    The unified data is split between ``fim_by_cap.json`` (S_cd_raw,
    S_pc_raw, S_r2_raw -- only the converged subset of voltages) and
    ``observables_by_cap.csv`` (per-V cd, pc, converged flag). Mirrors
    the read pattern used in ``v20_voltage_grid_fim_ablation.py``.
    """
    with open(unified_json) as f:
        d = json.load(f)
    cap_data = d["50"]
    S_cd = np.array(cap_data["S_cd_raw"])
    S_pc = np.array(cap_data["S_pc_raw"])
    S_r2 = np.array(cap_data["S_r2_raw"])

    obs_path = os.path.join(os.path.dirname(unified_json), "observables_by_cap.csv")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(
            f"Companion observables CSV not found: {obs_path}",
        )
    voltages: list[float] = []
    cd_vals: list[float] = []
    pc_vals: list[float] = []
    with open(obs_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row["cap"] != "50":
                continue
            if row["converged"] != "True":
                continue
            voltages.append(float(row["V_RHE"]))
            cd_vals.append(float(row["cd"]))
            pc_vals.append(float(row["pc"]))
    if S_cd.shape != (len(voltages), 4):
        raise ValueError(
            f"S_cd shape {S_cd.shape} != ({len(voltages)}, 4); "
            f"unified data is inconsistent.",
        )
    return {
        "v": voltages,
        "cd": cd_vals,
        "pc": pc_vals,
        "S_cd": S_cd,
        "S_pc": S_pc,
        "S_r2": S_r2,
    }


def assemble_combined_table(
    unified: dict[str, Any], extended: dict[str, Any],
) -> dict[str, np.ndarray]:
    v_combined = list(unified["v"])
    cd_combined = list(unified["cd"])
    pc_combined = list(unified["pc"])
    S_cd_combined = list(unified["S_cd"])
    S_pc_combined = list(unified["S_pc"])
    S_r2_combined = list(unified["S_r2"])
    for v_str, row in extended["rows"].items():
        v = float(v_str)
        if any(abs(v - v_existing) < 1e-6 for v_existing in v_combined):
            continue
        v_combined.append(v)
        cd_combined.append(float(row["cd"]))
        pc_combined.append(float(row["pc"]))
        S_cd_combined.append(np.array(row["S_cd"]))
        S_pc_combined.append(np.array(row["S_pc"]))
        S_r2_combined.append(np.array(row["S_r2"]))
    order = np.argsort(v_combined)
    v_arr = np.array(v_combined)[order]
    return {
        "v": v_arr,
        "cd": np.array(cd_combined)[order],
        "pc": np.array(pc_combined)[order],
        "S_cd": np.array(S_cd_combined)[order],
        "S_pc": np.array(S_pc_combined)[order],
        "S_r2": np.array(S_r2_combined)[order],
    }


def select_grid_rows(
    combined: dict[str, np.ndarray], grid: tuple[float, ...],
) -> dict[str, np.ndarray]:
    idx = []
    missing = []
    for vt in grid:
        match = np.where(np.abs(combined["v"] - vt) < 1e-3)[0]
        if match.size == 0:
            missing.append(vt)
        else:
            idx.append(int(match[0]))
    if missing:
        return {"missing": missing}  # caller handles
    idx = np.array(idx)
    return {
        "v": combined["v"][idx],
        "cd": combined["cd"][idx],
        "pc": combined["pc"][idx],
        "S_cd": combined["S_cd"][idx],
        "S_pc": combined["S_pc"][idx],
    }


def whiten_rows(
    sub: dict[str, np.ndarray], noise_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cd = sub["cd"]; pc = sub["pc"]
    S_cd = sub["S_cd"]; S_pc = sub["S_pc"]
    if noise_model == "global_max":
        sigma_cd = 0.02 * float(np.max(np.abs(cd)))
        sigma_pc = 0.02 * float(np.max(np.abs(pc)))
        S_cd_w = S_cd / sigma_cd
        S_pc_w = S_pc / sigma_pc
        sigmas = np.array([sigma_cd] * len(cd) + [sigma_pc] * len(pc))
    elif noise_model == "local_rel":
        # 2% relative + small absolute floor to keep tiny-|y| rows honest;
        # matches V20 'local_rel' interpretation.
        FLOOR = 1e-8
        sigma_cd_v = np.maximum(0.02 * np.abs(cd), FLOOR)
        sigma_pc_v = np.maximum(0.02 * np.abs(pc), FLOOR)
        S_cd_w = S_cd / sigma_cd_v[:, None]
        S_pc_w = S_pc / sigma_pc_v[:, None]
        sigmas = np.concatenate([sigma_cd_v, sigma_pc_v])
    else:
        raise ValueError(f"Unknown noise_model: {noise_model}")
    S_white = np.vstack([S_cd_w, S_pc_w])
    return S_white, sigmas, np.array([
        float(np.max(sigmas[: len(cd)])),
        float(np.max(sigmas[len(cd):])),
    ])


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Forward + FD (or reuse) -----------------------------------
    extended_path = os.path.join(out_dir, "sensitivities_extended.json")
    if args.skip_solves:
        src = args.extended_json or extended_path
        if not os.path.exists(src):
            print(
                f"FATAL: --skip-solves requested but {src} not found.",
                file=sys.stderr,
            )
            sys.exit(2)
        with open(src) as f:
            extended = json.load(f)
        print(f"Reusing extended sensitivities from {src}")
    else:
        new_vs = list(args.new_vs) + list(args.extra_vs)
        new_vs = sorted(set(round(float(v), 3) for v in new_vs))
        print(f"V20 unified rows source: {args.unified_json}")
        print(f"New voltages to compute: {new_vs}")
        print(f"Anchor voltage: {args.anchor_v:+.3f}")
        factory = _build_solver_factory()
        extended = compute_extended_rows(
            factory,
            new_vs=new_vs,
            anchor_v=float(args.anchor_v),
            max_z_steps=int(args.max_z_steps),
        )
        with open(extended_path, "w") as f:
            json.dump(extended, f, indent=2)
        print(f"Saved: {extended_path}")

    # --- 2. Merge with V20 unified ------------------------------------
    unified = load_unified_v20(args.unified_json)
    combined = assemble_combined_table(unified, extended)
    print(
        f"\nCombined V grid ({len(combined['v'])} rows): "
        f"{combined['v'].tolist()}"
    )

    # --- 3. FIM ablation ---------------------------------------------
    grids: dict[str, tuple[float, ...]] = {
        "G0": tuple(G0),
        "G_neg1": tuple(sorted(set(list(G0) + [-0.20]))),
        "G_neg2": tuple(sorted(set(list(G0) + [-0.20, -0.30]))),
        "G_neg3": tuple(sorted(set(list(G0) + [-0.20, -0.30, -0.40, -0.50]))),
    }
    noise_models = ("global_max", "local_rel")
    fim_by_grid: dict[str, Any] = {}
    weak_rows: list[dict[str, Any]] = []

    leverage_records: list[dict[str, Any]] = []

    for grid_name, grid in grids.items():
        sub = select_grid_rows(combined, grid)
        if "missing" in sub:
            print(
                f"\n[{grid_name}] SKIPPED — missing rows for "
                f"{sub['missing']}"
            )
            continue
        fim_by_grid[grid_name] = {"voltages": list(grid)}
        for nm in noise_models:
            S_white, sigmas, sigma_max = whiten_rows(sub, nm)
            metrics = fim_metrics(S_white)
            metrics["sigma_cd_max"] = float(sigma_max[0])
            metrics["sigma_pc_max"] = float(sigma_max[1])
            metrics["sv_min"] = float(metrics["singular_values"][-1])
            metrics["sv_max"] = float(metrics["singular_values"][0])
            fim_by_grid[grid_name][nm] = metrics
            weak_rows.append({
                "grid": grid_name,
                "noise_model": nm,
                "NV": len(grid),
                "sv_min": metrics["sv_min"],
                "cond_F": metrics["condition_number"],
                "ridge_cos": metrics["weak_eigvec_canonical_ridge_cos"],
                "log_k0_1_component": metrics["weak_eigvec_log_k0_1_component"],
                **metrics["weak_eigvec"],
            })
            lev = per_voltage_leverage(S_white, n_obs=2)
            wdc = per_voltage_weak_dir_contribution(S_white, n_obs=2)
            for vi, v in enumerate(grid):
                leverage_records.append({
                    "grid": grid_name,
                    "noise_model": nm,
                    "V": float(v),
                    "leverage": float(lev[vi]),
                    "weak_dir_contribution": float(wdc[vi]),
                })

    # --- 4. Persist outputs ------------------------------------------
    with open(os.path.join(out_dir, "fim_by_grid.json"), "w") as f:
        json.dump(fim_by_grid, f, indent=2)

    with open(os.path.join(out_dir, "weak_eigvec_by_grid.csv"), "w") as f:
        if weak_rows:
            w = csv.DictWriter(f, fieldnames=list(weak_rows[0].keys()))
            w.writeheader()
            for r in weak_rows:
                w.writerow(r)

    with open(os.path.join(out_dir, "leverage_by_voltage.csv"), "w") as f:
        if leverage_records:
            w = csv.DictWriter(f, fieldnames=list(leverage_records[0].keys()))
            w.writeheader()
            for r in leverage_records:
                w.writerow(r)

    # --- 5. Summary md ------------------------------------------------
    write_summary_md(
        out_dir, combined, fim_by_grid, grids, noise_models,
        anchor_v=float(extended.get("anchor_v", args.anchor_v)),
    )
    print(f"\nDone. Outputs in {out_dir}")


def write_summary_md(
    out_dir: str,
    combined: dict[str, np.ndarray],
    fim_by_grid: dict[str, Any],
    grids: dict[str, tuple[float, ...]],
    noise_models: tuple[str, ...],
    anchor_v: float,
) -> None:
    lines: list[str] = []
    lines.append("# V23 -- Negative-V FIM ablation (extend grid to V <= -0.30)\n")
    lines.append(
        "Per ``docs/TODO_extend_inverse_v_range_negative.md``: extend the "
        "V20 unified FIM dataset with FD sensitivity rows at the missing "
        "negative voltages (V20 cold-ramp failed at V = -0.30 and V = -0.50, "
        "which voltage continuation now reaches in 1-2 s per V) and rerun a "
        "V20-style ablation across G0 + extended grids. Pass criterion: "
        "weak eigvec |log_k0_1| component drops below 0.95 under at least "
        "one noise model on at least one extended grid.\n"
    )
    lines.append(
        f"Anchor cold-solve at V = {anchor_v:+.3f}, warm-continuation outward.\n"
    )

    lines.append("\n## Combined V grid (rows available)\n")
    lines.append("| V | cd | pc |\n|---:|:---|:---|\n")
    for vi, v in enumerate(combined["v"]):
        lines.append(
            f"| {v:+.3f} | {combined['cd'][vi]:+.4e} | "
            f"{combined['pc'][vi]:+.4e} |\n"
        )

    for nm in noise_models:
        lines.append(f"\n## Results -- noise model: {nm}\n")
        lines.append(
            "| grid | NV | sv_min | cond(F) | ridge_cos | "
            "|k0_1| weak | weak_eigvec |\n"
            "|---|---:|---:|---:|---:|---:|---|\n"
        )
        for grid_name in grids:
            if grid_name not in fim_by_grid:
                lines.append(
                    f"| {grid_name} | -- | (skipped) | -- | -- | -- | -- |\n"
                )
                continue
            block = fim_by_grid[grid_name].get(nm)
            if block is None:
                continue
            wv = block["weak_eigvec"]
            lines.append(
                f"| {grid_name} | {block['n_residuals'] // 2} | "
                f"{block['sv_min']:.3e} | "
                f"{block['condition_number']:.2e} | "
                f"{block['weak_eigvec_canonical_ridge_cos']:.3f} | "
                f"{block['weak_eigvec_log_k0_1_component']:.3f} | "
                f"[{wv['log_k0_1']:+.3f}, {wv['log_k0_2']:+.3f}, "
                f"{wv['alpha_1']:+.3f}, {wv['alpha_2']:+.3f}] |\n"
            )

    # ------------------------------------------------------------
    # Verdict.
    #
    # The TODO's pass criterion was "weak |log_k0_1| drops below 0.95
    # under at least one noise model on at least one extended grid."
    # That criterion is necessary but NOT sufficient: a global_max flip
    # can be a noise-model artifact (V <= -0.30 inflates sigma_pc,
    # demoting V > 0 PC rows -- cond gets WORSE while the weak direction
    # mathematically rotates). V20 already documented this trap with
    # V = -0.20.
    #
    # The publishable test is local_rel only:
    #   STRONG PASS    weak |log_k0_1| < 0.95 AND cond improves
    #   WEAK PASS      weak |log_k0_1| stays >= 0.95 BUT
    #                  cond improves >= 2x AND sv_min improves >= 1.5x
    #   FAIL           neither.
    # ------------------------------------------------------------
    lines.append("\n## Verdict\n")
    PASS_K0_THRESH = 0.95
    PASS_COND_RATIO = 2.0
    PASS_SVMIN_RATIO = 1.5
    g0 = fim_by_grid.get("G0", {})

    def block_for(grid_name: str, nm: str) -> Optional[dict[str, Any]]:
        return fim_by_grid.get(grid_name, {}).get(nm)

    g0_local = block_for("G0", "local_rel")
    g0_global = block_for("G0", "global_max")

    strong_pass: list[str] = []
    weak_pass: list[str] = []
    failures: list[str] = []
    artifact_flips: list[str] = []

    for grid_name in grids:
        if grid_name == "G0":
            continue
        for nm in noise_models:
            block = block_for(grid_name, nm)
            if block is None:
                continue
            base = block_for("G0", nm)
            if base is None:
                continue
            ratio_cond = base["condition_number"] / max(
                block["condition_number"], 1e-300,
            )
            ratio_svmin = block["sv_min"] / max(base["sv_min"], 1e-300)
            label = f"{grid_name} [{nm}]"
            k0_below = (
                block["weak_eigvec_log_k0_1_component"] < PASS_K0_THRESH
            )
            cond_better = ratio_cond >= 1.0
            cond_meets = ratio_cond >= PASS_COND_RATIO
            sv_meets = ratio_svmin >= PASS_SVMIN_RATIO
            if k0_below and cond_better:
                strong_pass.append(
                    f"{label} (cond/{ratio_cond:.2f}x, "
                    f"sv_min x{ratio_svmin:.2f})"
                )
            elif k0_below and not cond_better:
                artifact_flips.append(
                    f"{label} (cond x{1 / max(ratio_cond, 1e-300):.2f} "
                    f"WORSE -- noise-model artifact)"
                )
            elif (not k0_below) and cond_meets and sv_meets:
                weak_pass.append(
                    f"{label} (cond/{ratio_cond:.2f}x, "
                    f"sv_min x{ratio_svmin:.2f})"
                )
            else:
                failures.append(
                    f"{label} (cond/{ratio_cond:.2f}x, "
                    f"sv_min x{ratio_svmin:.2f})"
                )

    if strong_pass:
        lines.append(
            "**STRONG PASS** -- weak |log_k0_1| < "
            f"{PASS_K0_THRESH} *and* cond(F) improves on: "
            f"{', '.join(strong_pass)}.\n"
        )
    if weak_pass:
        lines.append(
            "**WEAK PASS** -- weak direction stays log_k0_1 but cond(F) "
            f"improves >= {PASS_COND_RATIO}x and sv_min improves "
            f">= {PASS_SVMIN_RATIO}x on: {', '.join(weak_pass)}. "
            "Geometry is unchanged but local curvature along log_k0_1 "
            "is sharper -- worth a single inverse run on the extended "
            "grid as a sanity test, but unlikely to break basin barriers "
            "by itself.\n"
        )
    if artifact_flips:
        lines.append(
            "**Artifact flips (do not count)** -- weak direction "
            "rotated but cond(F) got WORSE on: "
            f"{', '.join(artifact_flips)}. This is the V20 "
            "sigma_pc-inflation trap: V <= -0.30 has |PC| ~ 0.06-0.18, "
            "which under global_max sets sigma_pc by 2% x max|pc| and "
            "demotes every V > 0 PC row by ~10^4. The flip is "
            "mathematical, not informative.\n"
        )
    if failures:
        lines.append(
            f"FAIL on: {', '.join(failures)}.\n"
        )

    if not strong_pass and not weak_pass:
        lines.append(
            "\n## Recommendation\n"
            "Single-experiment grid extension to V <= -0.30 does NOT "
            "rotate the weak Fisher direction off log_k0_1 under the "
            "publishable noise model. The hypothesis from "
            "``docs/TODO_extend_inverse_v_range_negative.md`` is not "
            "supported. Move to HANDOFF_11 Phase 2A (multi-experiment "
            "FIM with bulk-O2 variation) -- changing c_O2_bulk perturbs "
            "R1's transport/kinetic balance directly and is the most "
            "targeted lever for the remaining log_k0_1 ambiguity.\n"
        )
    elif weak_pass and not strong_pass:
        lines.append(
            "\n## Recommendation\n"
            "Geometry is essentially unchanged but conditioning "
            "improves under local_rel. Suggested order: (a) optionally "
            "rerun the multi-init clean inverse on the best extended "
            "grid as a low-cost sanity check (TODO Step 2), but do not "
            "expect it to recover all 4 parameters; (b) proceed to "
            "HANDOFF_11 Phase 2A (bulk-O2 variation) regardless, since "
            "the weak direction is unchanged.\n"
        )
    else:
        lines.append(
            "\n## Recommendation\n"
            "Proceed with TODO Step 2: add voltage continuation to "
            "v18 Step 1 and rerun the multi-init clean inverse on "
            "the best extended grid. Compare against G0 baseline.\n"
        )

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
