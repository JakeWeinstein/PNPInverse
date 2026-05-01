"""V19 Stage 1 — BV exponent clip audit.

Sweeps `exponent_clip` over [50, 60, 70, 80, 100, None] with the log-c + 3sp +
Boltzmann breakthrough recipe.  For each cap, cold-solves the TRUE-parameter
forward model on V_GRID = [-0.10, 0.0, 0.10, 0.15, 0.20], records observables
(cd, pc, r1, r2), H2O2 surface and domain extrema, and computes central-FD
sensitivities of cd / pc / r2 w.r.t. (log_k0_1, log_k0_2, alpha_1, alpha_2).
Then computes whitened FIM metrics per cap.

Per the handoff (`docs/PNP Anodic Solver Handoff.md`):

  Acceptance test --- if increasing the cap from 50 to 70/80/100 changes any of:
    dPC/dalpha_2, weak eigenvector, sv_min, ridge_cos
  then the current k0_2 failure is at least partly a clip artifact.

  If nothing changes and the solver fails badly without the cap, proceed to
  the FV/SG prototype (Stage 4).

Outputs in StudyResults/v19_bv_clip_audit/:
  convergence_by_cap.json
  fim_by_cap.json
  observables_by_cap.csv
  rates_by_cap.csv
  h2o2_min_by_cap.csv

Usage:
  python scripts/studies/v19_bv_clip_audit.py                    # default sweep
  python scripts/studies/v19_bv_clip_audit.py --caps 50          # smoke test
  python scripts/studies/v19_bv_clip_audit.py --caps 50 70 none  # subset
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


# ---------------------------------------------------------------------------
# Canonical (k0_2, alpha_2) ridge direction in 4D parameter space.
# Per breakthrough analysis the slope d(log k0)/d(alpha) ~ -47 along the ridge.
# ---------------------------------------------------------------------------
def canonical_ridge() -> np.ndarray:
    v = np.array([0.0, -47.0, 0.0, +1.0])
    return v / np.linalg.norm(v)


def fim_metrics(S_white: np.ndarray,
                names: tuple[str, ...] = ("log_k0_1", "log_k0_2", "alpha_1", "alpha_2")
                ) -> dict[str, Any]:
    if not np.all(np.isfinite(S_white)):
        return {"error": "non-finite sensitivity"}
    try:
        _, sv, _ = np.linalg.svd(S_white, full_matrices=False)
        F = S_white.T @ S_white
        evals, evecs = np.linalg.eigh(F)
        cond_F = float(evals[-1] / max(evals[0], 1e-30))
        weak_v = evecs[:, 0]
        cos_sim = float(abs(np.dot(weak_v, canonical_ridge())))
        return {
            "n_residuals": int(S_white.shape[0]),
            "n_params": int(S_white.shape[1]),
            "singular_values": sv.tolist(),
            "fim_eigenvalues": evals.tolist(),
            "fim_diagonal": np.diag(F).tolist(),
            "condition_number": cond_F,
            "weak_eigvec": dict(zip(names, weak_v.tolist())),
            "weak_eigvec_canonical_ridge_cos": cos_sim,
            "ridge_breaking_score": 1.0 - cos_sim,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--caps", nargs="+", default=None,
        help="Caps to sweep (e.g. 50 60 70 80 100 none). Default: full sweep.",
    )
    p.add_argument(
        "--out-dir", default=None, help="Override output directory.",
    )
    p.add_argument(
        "--max-z-steps", type=int, default=20,
        help="Number of z-ramp steps in cold solve (default 20).",
    )
    p.add_argument(
        "--log-rate", action="store_true",
        help="Stage 2: enable log-rate BV evaluation (bv_log_rate=True).",
    )
    p.add_argument(
        "--v-grid", nargs="+", type=float, default=None,
        help="Override V_GRID. Default: -0.10 0.0 0.10 0.15 0.20.",
    )
    p.add_argument(
        "--u-clamp", type=float, default=30.0,
        help="Symmetric clamp on u=ln(c) for bulk PDE terms (default 30).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, SNES_OPTS_CHARGED,
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

    # ---- Configuration -----------------------------------------------------
    H2O2_SEED = 1e-4
    THREE_SPECIES_Z = [0, 0, 1]
    THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
    V_GRID = (np.array(args.v_grid, dtype=float) if args.v_grid is not None
              else np.array([-0.10, 0.00, 0.10, 0.15, 0.20]))
    NV = len(V_GRID)
    USE_LOG_RATE = bool(args.log_rate)
    U_CLAMP = float(args.u_clamp)

    PARAM_NAMES = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]
    H = np.array([1e-3, 1e-3, 1e-4, 1e-4])

    if args.caps is None:
        caps_raw: list[str] = ["50", "60", "70", "80", "100", "none"]
    else:
        caps_raw = list(args.caps)
    caps: list[Optional[float]] = []
    for c in caps_raw:
        cs = str(c).lower()
        if cs in ("none", "null", "off"):
            caps.append(None)
        else:
            caps.append(float(c))

    OUT_DIR = args.out_dir or os.path.join(_ROOT, "StudyResults", "v19_bv_clip_audit")
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    # ---- Per-cap convergence cfg -------------------------------------------
    def make_conv_cfg(cap: Optional[float]) -> dict[str, Any]:
        return {
            "clip_exponent": cap is not None,
            "exponent_clip": float(cap) if cap is not None else 50.0,
            "regularize_concentration": True,
            "conc_floor": 1e-12,
            "use_eta_in_bv": True,
            "bv_log_rate": USE_LOG_RATE,
            "u_clamp": U_CLAMP,
        }

    # ---- SolverParams factory ---------------------------------------------
    def make_3sp_sp(eta_hat: float, k0_1: float, k0_2: float,
                    a_1: float, a_2: float, cap: Optional[float]) -> SolverParams:
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = make_conv_cfg(cap)
        params["nondim"] = _make_nondim_cfg()
        reaction_1 = {
            "k0": k0_1, "alpha": a_1,
            "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
            "reversible": True, "E_eq_v": E_EQ_R1,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
            ],
        }
        reaction_2 = {
            "k0": k0_2, "alpha": a_2,
            "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
            "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
            "reversible": False, "E_eq_v": E_EQ_R2,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
            ],
        }
        params["bv_bc"] = {
            "reactions": [reaction_1, reaction_2],
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

    def add_boltzmann(ctx: dict[str, Any]) -> dict[str, Any]:
        mesh = ctx["mesh"]; W = ctx["W"]; U = ctx["U"]
        scaling = ctx["nondim"]
        phi = fd.split(U)[-1]
        w = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=mesh)
        charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)),
                              fd.Constant(50.0))
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

    def build_solve(V_RHE: float, k0_1: float, k0_2: float,
                    a_1: float, a_2: float, cap: Optional[float]):
        sp = make_3sp_sp(V_RHE / V_T, k0_1, k0_2, a_1, a_2, cap)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT)
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        of_r1 = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=1.0)
        of_r2 = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=1, scale=1.0)
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
                    sv = max(abs(fv), abs(prev_flux), 1e-8)
                    if d / sv <= 1e-4 or d <= 1e-8:
                        sc += 1
                    else:
                        sc = 0
                    if prev_delta and d > 0:
                        r = prev_delta / d
                        dt_val = (min(dt_val * min(r, 4), dt_init * 20)
                                  if r > 1 else max(dt_val * 0.5, dt_init))
                        dt_const.assign(dt_val)
                    prev_delta = d
                prev_flux = fv
                if sc >= 4:
                    return True
            return False
        return run_ss

    def domain_h2o2_extrema(U) -> tuple[float, float]:
        """Min/max of c_H2O2 = exp(u_H2O2) over CG1 DOF values."""
        u_h2o2 = U.sub(1).dat.data_ro
        c_h2o2 = np.exp(np.clip(u_h2o2, -30.0, 30.0))
        return float(c_h2o2.min()), float(c_h2o2.max())

    def surface_h2o2(ctx) -> float:
        """Average c_H2O2 over electrode boundary."""
        electrode_marker = int(ctx["bv_settings"]["electrode_marker"])
        ds = fd.Measure("ds", domain=ctx["mesh"])
        ci = ctx["ci_exprs"][1]   # H2O2
        area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
        if area <= 0:
            return float("nan")
        return float(fd.assemble(ci * ds(electrode_marker))) / area

    def assemble_obs(of_cd, of_pc, of_r1, of_r2):
        return (
            float(fd.assemble(of_cd)),
            float(fd.assemble(of_pc)),
            float(fd.assemble(of_r1)),
            float(fd.assemble(of_r2)),
        )

    def solve_warm(V_RHE, k0_1, k0_2, a_1, a_2, cap, ic_data, max_steps=200):
        ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2, cap)
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
        c_min, c_max = domain_h2o2_extrema(U)
        c_surf = surface_h2o2(ctx)
        return {
            "cd": cd, "pc": pc, "r1": r1, "r2": r2,
            "c_h2o2_surf": c_surf,
            "c_h2o2_min": c_min, "c_h2o2_max": c_max,
            "snap": _snapshot(U),
        }

    def solve_cold(V_RHE, k0_1, k0_2, a_1, a_2, cap, max_z_steps=20):
        ctx, sol, of_cd, of_pc, of_r1, of_r2, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2, cap)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        achieved_z = 0.0
        with adj.stop_annotating():
            for zci in zc:
                zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(200):
                return None, achieved_z
            for z_val in np.linspace(0, 1, max_z_steps + 1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                if run_ss(120):
                    achieved_z = float(z_val)
                else:
                    _restore(ckpt, U, Up); break
            if achieved_z < 1.0 - 1e-3:
                return None, achieved_z
        cd, pc, r1, r2 = assemble_obs(of_cd, of_pc, of_r1, of_r2)
        c_min, c_max = domain_h2o2_extrema(U)
        c_surf = surface_h2o2(ctx)
        return {
            "cd": cd, "pc": pc, "r1": r1, "r2": r2,
            "c_h2o2_surf": c_surf,
            "c_h2o2_min": c_min, "c_h2o2_max": c_max,
            "snap": _snapshot(U),
            "z_achieved": achieved_z,
        }, achieved_z

    # ---- Per-cap pipeline -------------------------------------------------
    def run_cap(cap: Optional[float]) -> dict[str, Any]:
        cap_label = "none" if cap is None else f"{cap:g}"
        print(f"\n[cap={cap_label}] Cold-solving TRUE on V_GRID", flush=True)
        t0 = time.time()
        per_v = []
        true_cache: list[Any] = [None] * NV
        for i, V in enumerate(V_GRID):
            t_v = time.time()
            res, z_ach = solve_cold(float(V), K0_HAT_R1, K0_HAT_R2,
                                    ALPHA_R1, ALPHA_R2, cap,
                                    max_z_steps=args.max_z_steps)
            entry = {
                "V_RHE": float(V),
                "converged": res is not None,
                "z_achieved": float(z_ach),
                "elapsed_s": round(time.time() - t_v, 2),
            }
            if res is not None:
                for k in ("cd", "pc", "r1", "r2",
                          "c_h2o2_surf", "c_h2o2_min", "c_h2o2_max"):
                    entry[k] = res[k]
                true_cache[i] = res["snap"]
                print(
                    f"  V={V:+.2f}  cd={res['cd']:+.4e}  pc={res['pc']:+.4e}  "
                    f"r1={res['r1']:.3e}  r2={res['r2']:.3e}  "
                    f"H2O2[{res['c_h2o2_min']:.2e},{res['c_h2o2_max']:.2e}]  "
                    f"z={z_ach:.3f}  ({entry['elapsed_s']}s)",
                    flush=True,
                )
            else:
                print(
                    f"  V={V:+.2f}  FAILED (z={z_ach:.3f})  "
                    f"({entry['elapsed_s']}s)",
                    flush=True,
                )
            per_v.append(entry)

        n_used = sum(1 for e in per_v if e["converged"])
        print(f"  cold solve: {n_used}/{NV} V converged in "
              f"{time.time() - t0:.1f}s", flush=True)

        if n_used < 3:
            return {"cap": cap_label, "status": "insufficient_voltages",
                    "n_used": n_used, "per_v": per_v,
                    "elapsed_s": round(time.time() - t0, 2)}

        # FD sensitivities at TRUE
        v_used = np.array([e["converged"] for e in per_v], dtype=bool)
        target_cd = np.array([e.get("cd", np.nan) for e in per_v])
        target_pc = np.array([e.get("pc", np.nan) for e in per_v])
        theta_true = np.array([np.log(K0_HAT_R1), np.log(K0_HAT_R2),
                               ALPHA_R1, ALPHA_R2])

        def solve_curve(theta_log) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            k0_1 = float(np.exp(theta_log[0]))
            k0_2 = float(np.exp(theta_log[1]))
            a_1 = float(theta_log[2]); a_2 = float(theta_log[3])
            cds = np.full(NV, np.nan); pcs = np.full(NV, np.nan)
            r2s = np.full(NV, np.nan)
            for i, V in enumerate(V_GRID):
                if not v_used[i]:
                    continue
                res = solve_warm(float(V), k0_1, k0_2, a_1, a_2, cap,
                                 true_cache[i])
                if res is not None:
                    cds[i] = res["cd"]; pcs[i] = res["pc"]
                    r2s[i] = res["r2"]
            return cds, pcs, r2s

        S_cd = np.zeros((NV, 4)); S_pc = np.zeros((NV, 4))
        S_r2 = np.zeros((NV, 4))
        print(f"[cap={cap_label}] FD sensitivities at TRUE", flush=True)
        for j in range(4):
            t_j = time.time()
            tp = theta_true.copy(); tp[j] += H[j]
            tm = theta_true.copy(); tm[j] -= H[j]
            cd_p, pc_p, r2_p = solve_curve(tp)
            cd_m, pc_m, r2_m = solve_curve(tm)
            for i in range(NV):
                if not v_used[i]:
                    S_cd[i, j] = np.nan; S_pc[i, j] = np.nan
                    S_r2[i, j] = np.nan
                    continue
                vals = (cd_p[i], cd_m[i], pc_p[i], pc_m[i], r2_p[i], r2_m[i])
                if not all(np.isfinite(x) for x in vals):
                    S_cd[i, j] = np.nan; S_pc[i, j] = np.nan
                    S_r2[i, j] = np.nan
                else:
                    S_cd[i, j] = (cd_p[i] - cd_m[i]) / (2 * H[j])
                    S_pc[i, j] = (pc_p[i] - pc_m[i]) / (2 * H[j])
                    S_r2[i, j] = (r2_p[i] - r2_m[i]) / (2 * H[j])
            print(
                f"    d/d{PARAM_NAMES[j]:>10}: "
                f"|dcd|={np.linalg.norm(S_cd[v_used, j]):.3e}  "
                f"|dpc|={np.linalg.norm(S_pc[v_used, j]):.3e}  "
                f"|dr2|={np.linalg.norm(S_r2[v_used, j]):.3e}  "
                f"({time.time() - t_j:.1f}s)",
                flush=True,
            )

        sigma_cd = 0.02 * float(np.max(np.abs(target_cd[v_used])))
        sigma_pc = 0.02 * float(np.max(np.abs(target_pc[v_used])))

        S_cd_used = S_cd[v_used, :]; S_pc_used = S_pc[v_used, :]
        S_r2_used = S_r2[v_used, :]
        S_cd_white = S_cd_used / sigma_cd
        S_pc_white = S_pc_used / sigma_pc
        S_both = np.vstack([S_cd_white, S_pc_white])

        fim = fim_metrics(S_both)
        # Per-V dPC/d(alpha_2) and dr2/d(alpha_2): index 3 in S
        dpc_dalpha2 = S_pc[:, 3].tolist()
        dr2_dalpha2 = S_r2[:, 3].tolist()

        return {
            "cap": cap_label,
            "status": "ok",
            "n_used": int(n_used),
            "per_v": per_v,
            "v_used": v_used.tolist(),
            "sigma_cd": sigma_cd, "sigma_pc": sigma_pc,
            "S_cd_raw": S_cd_used.tolist(),
            "S_pc_raw": S_pc_used.tolist(),
            "S_r2_raw": S_r2_used.tolist(),
            "dPC_dalpha2_per_v": dpc_dalpha2,
            "dR2_dalpha2_per_v": dr2_dalpha2,
            "fim": fim,
            "elapsed_s": round(time.time() - t0, 2),
        }

    # ---- Run -------------------------------------------------------------
    print("=" * 72)
    print("V19 Stage 1 — BV exponent clip audit")
    print("=" * 72)
    print(f"V_GRID:    {V_GRID.tolist()}")
    print(f"caps:      {[('none' if c is None else c) for c in caps]}")
    print(f"log_rate:  {USE_LOG_RATE}")
    print(f"u_clamp:   {U_CLAMP}")
    print(f"out_dir:   {OUT_DIR}")

    results: dict[str, Any] = {}
    for cap in caps:
        cap_label = "none" if cap is None else f"{cap:g}"
        try:
            results[cap_label] = run_cap(cap)
        except Exception as e:
            print(f"[cap={cap_label}] EXCEPTION: {type(e).__name__}: {e}",
                  flush=True)
            results[cap_label] = {"cap": cap_label, "status": "exception",
                                  "error": f"{type(e).__name__}: {e}"}

    # ---- Outputs --------------------------------------------------------
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float64, np.float32)):
            return float(o)
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        return o

    convergence = {}
    for cap_label, r in results.items():
        if "per_v" in r:
            convergence[cap_label] = {
                "status": r["status"],
                "n_used": r.get("n_used", 0),
                "per_v": [
                    {"V_RHE": e["V_RHE"],
                     "converged": e["converged"],
                     "z_achieved": e.get("z_achieved", 0.0),
                     "elapsed_s": e.get("elapsed_s", 0.0)}
                    for e in r["per_v"]
                ],
            }
        else:
            convergence[cap_label] = {
                "status": r["status"],
                "error": r.get("error", "unknown"),
            }
    with open(os.path.join(OUT_DIR, "convergence_by_cap.json"), "w") as f:
        json.dump(_clean(convergence), f, indent=2)

    fim_out = {}
    for cap_label, r in results.items():
        if r.get("status") == "ok" and "fim" in r:
            fim_out[cap_label] = {
                "fim": r["fim"],
                "sigma_cd": r["sigma_cd"], "sigma_pc": r["sigma_pc"],
                "dPC_dalpha2_per_v": r["dPC_dalpha2_per_v"],
                "dR2_dalpha2_per_v": r["dR2_dalpha2_per_v"],
                "S_cd_raw": r["S_cd_raw"],
                "S_pc_raw": r["S_pc_raw"],
                "S_r2_raw": r["S_r2_raw"],
                "v_used": r["v_used"],
                "n_used": r["n_used"],
            }
        else:
            fim_out[cap_label] = {
                "status": r.get("status", "unknown"),
                "error": r.get("error"),
            }
    with open(os.path.join(OUT_DIR, "fim_by_cap.json"), "w") as f:
        json.dump(_clean(fim_out), f, indent=2)

    with open(os.path.join(OUT_DIR, "observables_by_cap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap", "V_RHE", "converged", "z_achieved",
                    "cd", "pc", "r1", "r2"])
        for cap_label, r in results.items():
            for e in r.get("per_v", []):
                w.writerow([cap_label, e["V_RHE"], e["converged"],
                            e.get("z_achieved", ""),
                            e.get("cd", ""), e.get("pc", ""),
                            e.get("r1", ""), e.get("r2", "")])

    with open(os.path.join(OUT_DIR, "rates_by_cap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap", "V_RHE", "converged", "r1", "r2",
                    "dPC_dalpha2", "dR2_dalpha2"])
        for cap_label, r in results.items():
            dpc_v = r.get("dPC_dalpha2_per_v", [None] * NV)
            dr2_v = r.get("dR2_dalpha2_per_v", [None] * NV)
            for i, e in enumerate(r.get("per_v", [])):
                w.writerow([
                    cap_label, e["V_RHE"], e["converged"],
                    e.get("r1", ""), e.get("r2", ""),
                    dpc_v[i] if i < len(dpc_v) else "",
                    dr2_v[i] if i < len(dr2_v) else "",
                ])

    with open(os.path.join(OUT_DIR, "h2o2_min_by_cap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap", "V_RHE", "converged", "c_h2o2_surf",
                    "c_h2o2_min_domain", "c_h2o2_max_domain"])
        for cap_label, r in results.items():
            for e in r.get("per_v", []):
                w.writerow([
                    cap_label, e["V_RHE"], e["converged"],
                    e.get("c_h2o2_surf", ""),
                    e.get("c_h2o2_min", ""), e.get("c_h2o2_max", ""),
                ])

    # Console summary
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  {'cap':>6}  {'n_V':>3}  {'sv_min':>10}  {'cond(F)':>10}  "
          f"{'ridge_cos':>9}  {'weak_eigvec'}")
    for cap_label, r in results.items():
        if r.get("status") != "ok":
            print(f"  {cap_label:>6}  ----  status={r.get('status')}")
            continue
        f_data = r["fim"]
        if "error" in f_data:
            print(f"  {cap_label:>6}  {r['n_used']:>3}  fim_error: "
                  f"{f_data['error']}")
            continue
        sv = f_data["singular_values"][-1]
        cond = f_data["condition_number"]
        ridge = f_data["weak_eigvec_canonical_ridge_cos"]
        we = f_data["weak_eigvec"]
        we_str = (f"[{we['log_k0_1']:+.2f},{we['log_k0_2']:+.2f},"
                  f"{we['alpha_1']:+.2f},{we['alpha_2']:+.2f}]")
        print(f"  {cap_label:>6}  {r['n_used']:>3}  {sv:>10.3e}  {cond:>10.2e}  "
              f"{ridge:>9.3f}  {we_str}")
    print()
    for fname in ("convergence_by_cap.json", "fim_by_cap.json",
                  "observables_by_cap.csv", "rates_by_cap.csv",
                  "h2o2_min_by_cap.csv"):
        print(f"Saved: {os.path.join(OUT_DIR, fname)}")


if __name__ == "__main__":
    main()
