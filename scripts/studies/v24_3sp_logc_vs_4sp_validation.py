"""V24 — Apples-to-apples 4sp standard PNP vs 3sp+Boltzmann log-c forward solver.

Purpose
-------
The Apr 27 writeup claims:

    "In the overlap regime where both the 4-species and 3-species+Boltzmann
     formulations converged, the observables matched closely within the
     validation tolerances used for the inverse work."

The previously archived numerical evidence for that claim was thin (one PNG
mislabelled as 3sp-vs-4sp; a 4-point stdout comparison against hard-coded
reference values).  This script regenerates the comparison from scratch
using the *current production* forward solver — exactly the build path used
inside ``v18_logc_lsq_inverse.py`` (the most recent inverse script) — against
the canonical 4sp standard PNP solver via ``solve_grid_with_charge_continuation``
(the same recipe that produced ``StudyResults/diagnostic_eeq_sweep/...``).

Design
------
* Same graded mesh, same K0/α/E_eq/diffusivities/bulk concentrations, same
  ``I_SCALE`` for both solvers.  The only difference is Change 1 (drop ClO4-
  as a dynamic species, add an analytic Boltzmann factor in Poisson) plus
  Change 2 (log-concentration primary variable).  Change 3 (log-rate BV) is
  controlled by ``--log-rate``; default OFF so this is a clean
  reduction-only comparison.
* Voltage grid defaults to the documented overlap window where the standard
  4sp z-ramp reaches z=1: V_RHE ∈ {-0.30, -0.20, -0.10, 0.00, +0.05, +0.10}.
* Reports raw observables, absolute and relative errors, and a per-voltage
  PASS/FLAG verdict against a 5%-of-max threshold (matches the F2 diffusion-
  limit tolerance used elsewhere in the validation framework).
* Outputs ``raw_values.json``, ``comparison.csv``, ``comparison.png``, and
  ``summary.md`` under ``StudyResults/v24_3sp_logc_vs_4sp_validation/``.

Run
---
    ../venv-firedrake/bin/python scripts/studies/v24_3sp_logc_vs_4sp_validation.py
    # Optional: --log-rate to compare the full production stack vs 4sp.
    # Optional: --v-grid -0.30 -0.20 -0.10 0.0 0.05 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v-grid", nargs="+", type=float,
                   default=[-0.50, -0.40, -0.30, -0.20, -0.10,
                            0.00, 0.05, 0.10],
                   help="V_RHE points (V).  Default spans the 4sp z-ramp "
                        "full-convergence window plus the cathodic extension "
                        "we now reach via 3sp warm-start continuation.")
    p.add_argument("--log-rate", action="store_true",
                   help="Enable bv_log_rate=True on the 3sp logc side. "
                        "Default OFF so the comparison isolates Changes 1+2 "
                        "(Boltzmann reduction + log-c) and not Change 3.")
    p.add_argument("--mesh-ny", type=int, default=200)
    p.add_argument("--warm-substeps", type=int, default=4,
                   help="3sp warm-start: number of paf substeps when marching "
                        "from a converged anchor to a new voltage.")
    p.add_argument("--bisect-depth", type=int, default=3,
                   help="3sp warm-start: max bisection depth when a substep "
                        "ramp fails (each level halves the voltage gap).")
    p.add_argument("--out-subdir", type=str, default=None)
    p.add_argument("--rel-tol-pct", type=float, default=5.0,
                   help="Per-voltage PASS threshold as %% of max|observable|.")
    return p.parse_args()


def main():
    args = _parse_args()

    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
        C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
        SNES_OPTS_CHARGED,
        _make_nondim_cfg, _make_bv_convergence_cfg,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_with_charge_continuation,
    )
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.params import SolverParams

    V_GRID = np.array(args.v_grid, dtype=float)
    NV = len(V_GRID)
    USE_LOG_RATE = bool(args.log_rate)

    out_subdir = args.out_subdir or (
        f"v24_3sp_logc_vs_4sp_validation"
        f"{'_lograte' if USE_LOG_RATE else ''}"
    )
    OUT_DIR = os.path.join(_ROOT, "StudyResults", out_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("V24: 3sp+Boltzmann log-c forward solver vs 4sp standard PNP")
    print("=" * 72)
    print(f"  V_GRID:       {V_GRID.tolist()}")
    print(f"  log-rate:     {USE_LOG_RATE}")
    print(f"  mesh Ny:      {args.mesh_ny}")
    print(f"  K0_HAT_R1:    {K0_HAT_R1:.6e}")
    print(f"  K0_HAT_R2:    {K0_HAT_R2:.6e}")
    print(f"  ALPHA_R1/R2:  {ALPHA_R1} / {ALPHA_R2}")
    print(f"  E_eq R1/R2:   {E_EQ_R1} / {E_EQ_R2} V")
    print(f"  Output:       {OUT_DIR}")
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(args.mesh_ny), beta=3.0)

    # ------------------------------------------------------------------
    # 4-species standard solver via grid_charge_continuation
    # (same recipe as scripts/studies/diagnostic_eeq_voltage_sweep.py)
    # ------------------------------------------------------------------
    print("--- 4sp standard PNP via solve_grid_with_charge_continuation ---")
    snes_opts_4sp = {
        "snes_type":                 "newtonls",
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type":                  "preonly",
        "pc_type":                   "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8":         77,
        "mat_mumps_icntl_14":        80,
    }
    sp_4sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=snes_opts_4sp,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
        c_hp_hat=C_HP_HAT,
    )

    cd_4sp = np.full(NV, np.nan)
    pc_4sp = np.full(NV, np.nan)

    def _extract_4sp(orig_idx, _phi_app, ctx):
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_4sp[orig_idx] = float(fd.assemble(form_cd))
        pc_4sp[orig_idx] = float(fd.assemble(form_pc))

    phi_hat_grid = V_GRID / V_T
    t_4sp = time.time()
    with adj.stop_annotating():
        solve_grid_with_charge_continuation(
            sp_4sp,
            phi_applied_values=phi_hat_grid,
            charge_steps=20,
            mesh=mesh,
            max_eta_gap=2.0,
            min_delta_z=0.002,
            per_point_callback=_extract_4sp,
        )
    t_4sp = time.time() - t_4sp
    n_4sp_ok = int(np.sum(~np.isnan(cd_4sp)))
    print(f"  4sp converged at {n_4sp_ok}/{NV} points ({t_4sp:.1f}s)")
    for i, V in enumerate(V_GRID):
        print(f"    V={V:+.3f}: cd={cd_4sp[i]:+.6e}  pc={pc_4sp[i]:+.6e}")

    # ------------------------------------------------------------------
    # 3-species + Boltzmann log-c solver
    # (lifted verbatim from v18_logc_lsq_inverse.py — current production)
    # ------------------------------------------------------------------
    print()
    print("--- 3sp + Boltzmann log-c (current production solver) ---")

    H2O2_SEED = 1e-4
    THREE_SPECIES_Z = [0, 0, 1]
    THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
    THREE_SPECIES_C0 = [C_O2_HAT, H2O2_SEED, C_HP_HAT]

    SP_DICT_3SP = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    ss_rel_tol = 1e-4
    ss_abs_tol = 1e-8
    ss_consec = 4

    def make_3sp_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = _make_bv_convergence_cfg(log_rate=USE_LOG_RATE)
        params["nondim"] = _make_nondim_cfg()
        reaction_1 = {
            "k0": k0_r1, "alpha": alpha_r1,
            "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
            "reversible": True, "E_eq_v": E_EQ_R1,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
        }
        reaction_2 = {
            "k0": k0_r2, "alpha": alpha_r2,
            "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
            "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
            "reversible": False, "E_eq_v": E_EQ_R2,
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
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
        scaling = ctx["nondim"]
        W = ctx["W"]; U = ctx["U"]; mesh_ = ctx["mesh"]
        phi = fd.split(U)[-1]
        w = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=mesh_)
        charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)),
                              fd.Constant(50.0))
        ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
        ctx["J_form"] = fd.derivative(ctx["F_res"], U)
        return ctx

    def _snapshot(U): return tuple(d.data_ro.copy() for d in U.dat)
    def _restore(snap, U, Up):
        for src, dst in zip(snap, U.dat):
            dst.data[:] = src
        Up.assign(U)

    def build_solve(V_RHE):
        sp = make_3sp_sp(V_RHE / V_T, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = add_boltzmann(ctx)
        set_initial_conditions_logc(ctx, list(sp))
        prob = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT_3SP)
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
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
                try:
                    sol.solve()
                except Exception:
                    return False
                Up.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    d = abs(fv - prev_flux)
                    sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
                    if d / sv <= ss_rel_tol or d <= ss_abs_tol:
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
                if sc >= ss_consec:
                    return True
            return False
        return run_ss

    def solve_cold_3sp(V_RHE, max_z_steps=20):
        """Cold start (zero-charge IC) + z-ramp.  Returns (cd, pc, snap, z)."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_RHE)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)
        with adj.stop_annotating():
            for zci in zc:
                zci.assign(0.0)
            paf.assign(V_RHE / V_T)
            if not run_ss(200):
                return None, None, None, 0.0
            achieved_z = 0.0
            for z_val in np.linspace(0, 1, max_z_steps + 1)[1:]:
                ckpt = _snapshot(U)
                for i in range(n):
                    zc[i].assign(z_nominal[i] * z_val)
                if run_ss(120):
                    achieved_z = z_val
                else:
                    _restore(ckpt, U, Up)
                    break
            if achieved_z < 1.0 - 1e-3:
                return None, None, None, achieved_z
        snap = _snapshot(U)
        return (float(fd.assemble(of_cd)),
                float(fd.assemble(of_pc)),
                snap, achieved_z)

    def solve_warm_3sp_step(V_target, V_anchor, ic_data,
                             n_substeps=args.warm_substeps,
                             bisect_depth=args.bisect_depth):
        """Warm-start at full charge from ic_data (converged at V_anchor),
        ramp paf in n_substeps from V_anchor to V_target, run SS at each.
        On any failure, recursively bisect down to bisect_depth levels.
        Returns (cd, pc, snap) on full success or (None, None, None)."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(V_target)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        run_ss = make_run_ss(ctx, sol, of_cd)

        with adj.stop_annotating():
            _restore(ic_data, U, Up)
            for i in range(n):
                zc[i].assign(z_nominal[i])  # full charge from the start

            def _march(v0, v1, depth):
                substeps = np.linspace(v0, v1, n_substeps + 1)[1:]
                ckpt_outer = _snapshot(U)
                for v_sub in substeps:
                    ckpt_inner = _snapshot(U)
                    paf.assign(float(v_sub) / V_T)
                    if not run_ss(150):
                        # Roll back the failed substep
                        _restore(ckpt_inner, U, Up)
                        if depth >= bisect_depth:
                            _restore(ckpt_outer, U, Up)
                            return False
                        # Bisect: walk to the midpoint first, then to v_sub
                        v_prev = float(paf.dat.data_ro[0]) * V_T
                        v_mid = 0.5 * (v_prev + float(v_sub))
                        if not _march(v_prev, v_mid, depth + 1):
                            _restore(ckpt_outer, U, Up)
                            return False
                        if not _march(v_mid, float(v_sub), depth + 1):
                            _restore(ckpt_outer, U, Up)
                            return False
                return True

            ok = _march(float(V_anchor), float(V_target), 0)
            if not ok:
                return None, None, None
            # Final SS to make sure we are firmly at V_target
            paf.assign(float(V_target) / V_T)
            if not run_ss(200):
                return None, None, None

        snap = _snapshot(U)
        return (float(fd.assemble(of_cd)),
                float(fd.assemble(of_pc)),
                snap)

    cd_3sp = np.full(NV, np.nan)
    pc_3sp = np.full(NV, np.nan)
    z_3sp = np.full(NV, np.nan)
    method_3sp = ["MISSING"] * NV
    snaps_3sp: list = [None] * NV
    t_3sp = time.time()

    # Phase 1: cold-start each V independently (records baseline)
    print("  [3sp Phase 1] cold-start z-ramp at each V_RHE...")
    for i, V in enumerate(V_GRID):
        t_v = time.time()
        cd_v, pc_v, snap_v, z_a = solve_cold_3sp(float(V))
        z_3sp[i] = z_a
        if cd_v is not None:
            cd_3sp[i] = cd_v
            pc_3sp[i] = pc_v
            snaps_3sp[i] = snap_v
            method_3sp[i] = "cold"
            print(f"    V={V:+.3f}: cd={cd_v:+.6e}  pc={pc_v:+.6e}  "
                  f"z={z_a:.3f}  cold  ({time.time()-t_v:.1f}s)")
        else:
            print(f"    V={V:+.3f}: cold FAILED (z={z_a:.3f})  "
                  f"({time.time()-t_v:.1f}s)")

    # Phase 2: warm-start continuation outward from cold successes
    cold_idxs = sorted([i for i in range(NV) if snaps_3sp[i] is not None])
    if cold_idxs:
        print("  [3sp Phase 2] warm-start continuation from cold anchors "
              f"(substeps={args.warm_substeps}, "
              f"bisect_depth={args.bisect_depth})...")

        # Cathodic walk: from min cold-success index, march toward index 0
        anchor_lo = cold_idxs[0]
        for i in range(anchor_lo - 1, -1, -1):
            if snaps_3sp[i] is not None:
                continue
            # Pick the nearest converged neighbor (i+1 always exists here)
            j = i + 1
            while j < NV and snaps_3sp[j] is None:
                j += 1
            if j >= NV:
                break
            t_v = time.time()
            cd_v, pc_v, snap_v = solve_warm_3sp_step(
                float(V_GRID[i]), float(V_GRID[j]), snaps_3sp[j])
            if cd_v is not None:
                cd_3sp[i] = cd_v
                pc_3sp[i] = pc_v
                snaps_3sp[i] = snap_v
                z_3sp[i] = 1.0
                method_3sp[i] = f"warm←{V_GRID[j]:+.2f}"
                print(f"    V={V_GRID[i]:+.3f}: cd={cd_v:+.6e}  "
                      f"pc={pc_v:+.6e}  warm←{V_GRID[j]:+.2f}  "
                      f"({time.time()-t_v:.1f}s)")
            else:
                print(f"    V={V_GRID[i]:+.3f}: warm-from {V_GRID[j]:+.2f} "
                      f"FAILED  ({time.time()-t_v:.1f}s)")
                break  # don't try further-cathodic if the nearer step failed

        # Anodic walk: from max cold-success index, march toward index NV-1
        anchor_hi = cold_idxs[-1]
        for i in range(anchor_hi + 1, NV):
            if snaps_3sp[i] is not None:
                continue
            j = i - 1
            while j >= 0 and snaps_3sp[j] is None:
                j -= 1
            if j < 0:
                break
            t_v = time.time()
            cd_v, pc_v, snap_v = solve_warm_3sp_step(
                float(V_GRID[i]), float(V_GRID[j]), snaps_3sp[j])
            if cd_v is not None:
                cd_3sp[i] = cd_v
                pc_3sp[i] = pc_v
                snaps_3sp[i] = snap_v
                z_3sp[i] = 1.0
                method_3sp[i] = f"warm←{V_GRID[j]:+.2f}"
                print(f"    V={V_GRID[i]:+.3f}: cd={cd_v:+.6e}  "
                      f"pc={pc_v:+.6e}  warm←{V_GRID[j]:+.2f}  "
                      f"({time.time()-t_v:.1f}s)")
            else:
                print(f"    V={V_GRID[i]:+.3f}: warm-from {V_GRID[j]:+.2f} "
                      f"FAILED  ({time.time()-t_v:.1f}s)")
                break

    t_3sp = time.time() - t_3sp
    n_3sp_ok = int(np.sum(~np.isnan(cd_3sp)))
    print(f"  3sp converged at {n_3sp_ok}/{NV} points "
          f"(cold+warm continuation, {t_3sp:.1f}s)")

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    cd_max = float(np.nanmax(np.abs(cd_4sp))) if n_4sp_ok else float("nan")
    pc_max = float(np.nanmax(np.abs(pc_4sp))) if n_4sp_ok else float("nan")
    rel_tol = args.rel_tol_pct / 100.0

    rows = []
    for i, V in enumerate(V_GRID):
        if np.isnan(cd_4sp[i]) or np.isnan(cd_3sp[i]):
            rows.append({
                "V_RHE": float(V),
                "cd_4sp": (None if np.isnan(cd_4sp[i]) else float(cd_4sp[i])),
                "cd_3sp": (None if np.isnan(cd_3sp[i]) else float(cd_3sp[i])),
                "pc_4sp": (None if np.isnan(pc_4sp[i]) else float(pc_4sp[i])),
                "pc_3sp": (None if np.isnan(pc_3sp[i]) else float(pc_3sp[i])),
                "z_3sp": float(z_3sp[i]),
                "method_3sp": method_3sp[i],
                "abs_dcd": None, "abs_dpc": None,
                "cd_err_pct_of_max": None, "pc_err_pct_of_max": None,
                "verdict": "MISSING",
            })
            continue
        d_cd = float(cd_3sp[i] - cd_4sp[i])
        d_pc = float(pc_3sp[i] - pc_4sp[i])
        cd_err_pct_of_max = 100.0 * abs(d_cd) / max(cd_max, 1e-30)
        pc_err_pct_of_max = 100.0 * abs(d_pc) / max(pc_max, 1e-30)
        cd_pass = cd_err_pct_of_max <= args.rel_tol_pct
        pc_pass = pc_err_pct_of_max <= args.rel_tol_pct
        verdict = "PASS" if (cd_pass and pc_pass) else "FLAG"
        rows.append({
            "V_RHE": float(V),
            "cd_4sp": float(cd_4sp[i]), "cd_3sp": float(cd_3sp[i]),
            "pc_4sp": float(pc_4sp[i]), "pc_3sp": float(pc_3sp[i]),
            "z_3sp": float(z_3sp[i]),
            "method_3sp": method_3sp[i],
            "abs_dcd": abs(d_cd), "abs_dpc": abs(d_pc),
            "cd_err_pct_of_max": cd_err_pct_of_max,
            "pc_err_pct_of_max": pc_err_pct_of_max,
            "verdict": verdict,
        })

    overlap = [r for r in rows if r["verdict"] != "MISSING"]
    cd_errs = [r["cd_err_pct_of_max"] for r in overlap]
    pc_errs = [r["pc_err_pct_of_max"] for r in overlap]
    summary_stats = {
        "n_overlap": len(overlap),
        "cd_max_err_pct_of_max": (max(cd_errs) if cd_errs else None),
        "cd_mean_err_pct_of_max": (float(np.mean(cd_errs)) if cd_errs else None),
        "pc_max_err_pct_of_max": (max(pc_errs) if pc_errs else None),
        "pc_mean_err_pct_of_max": (float(np.mean(pc_errs)) if pc_errs else None),
        "cd_max_abs_4sp": cd_max, "pc_max_abs_4sp": pc_max,
        "n_pass": sum(1 for r in overlap if r["verdict"] == "PASS"),
        "n_flag": sum(1 for r in overlap if r["verdict"] == "FLAG"),
        "rel_tol_pct": args.rel_tol_pct,
    }

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    raw = {
        "config": {
            "v_grid": V_GRID.tolist(), "log_rate": USE_LOG_RATE,
            "mesh_ny": args.mesh_ny, "rel_tol_pct": args.rel_tol_pct,
            "K0_HAT_R1": K0_HAT_R1, "K0_HAT_R2": K0_HAT_R2,
            "ALPHA_R1": ALPHA_R1, "ALPHA_R2": ALPHA_R2,
            "E_EQ_R1": E_EQ_R1, "E_EQ_R2": E_EQ_R2,
            "I_SCALE": I_SCALE, "V_T": V_T,
        },
        "rows": rows,
        "summary": summary_stats,
        "wall_seconds": {"4sp": t_4sp, "3sp": t_3sp},
    }
    with open(os.path.join(OUT_DIR, "raw_values.json"), "w") as f:
        json.dump(raw, f, indent=2)

    with open(os.path.join(OUT_DIR, "comparison.csv"), "w") as f:
        f.write("V_RHE,cd_4sp,cd_3sp,pc_4sp,pc_3sp,z_3sp,method_3sp,"
                "abs_dcd,abs_dpc,cd_err_pct_of_max,pc_err_pct_of_max,verdict\n")
        for r in rows:
            def _fmt(v):
                return "" if v is None else f"{v:.8e}"
            f.write(",".join([
                f"{r['V_RHE']:.4f}",
                _fmt(r["cd_4sp"]), _fmt(r["cd_3sp"]),
                _fmt(r["pc_4sp"]), _fmt(r["pc_3sp"]),
                f"{r['z_3sp']:.4f}",
                r["method_3sp"],
                _fmt(r["abs_dcd"]), _fmt(r["abs_dpc"]),
                _fmt(r["cd_err_pct_of_max"]),
                _fmt(r["pc_err_pct_of_max"]),
                r["verdict"],
            ]) + "\n")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 9), sharex=True)
        I = -I_SCALE  # mA/cm^2 conversion sign-matched to observable form
        ax = axes[0]
        ax.plot(V_GRID, cd_4sp * I, "b-o", label="4sp standard PNP", markersize=6)
        ax.plot(V_GRID, cd_3sp * I, "r--s", label="3sp+Boltzmann log-c", markersize=5)
        ax.set_ylabel("Total current density (mA/cm²)")
        ax.set_title(f"Forward-solver agreement on overlap window  "
                     f"(log_rate={USE_LOG_RATE})")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(V_GRID, pc_4sp * I, "b-o", label="4sp standard PNP", markersize=6)
        ax.plot(V_GRID, pc_3sp * I, "r--s", label="3sp+Boltzmann log-c", markersize=5)
        ax.set_xlabel("V vs RHE (V)")
        ax.set_ylabel("Peroxide current density (mA/cm²)")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  plot: {plot_path}")
    except Exception as e:
        print(f"  plot skipped: {e}")

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    md = []
    md.append("# V24 — 3sp+Boltzmann log-c vs 4sp standard PNP\n")
    md.append("Apples-to-apples regeneration of the writeup's overlap claim "
              "using the *current production* forward solver from "
              "`scripts/studies/v18_logc_lsq_inverse.py` (most recent inverse) "
              "and the 4sp standard PNP via "
              "`Forward.bv_solver.solve_grid_with_charge_continuation`.\n")
    md.append("## Setup\n")
    md.append(f"- V_GRID = {V_GRID.tolist()}")
    md.append(f"- log-rate (3sp side) = {USE_LOG_RATE}")
    md.append(f"- mesh: graded rectangle Nx=8, Ny={args.mesh_ny}, beta=3.0")
    md.append(f"- TRUE params: K0_HAT_R1={K0_HAT_R1:.6e}, "
              f"K0_HAT_R2={K0_HAT_R2:.6e}, "
              f"α1={ALPHA_R1}, α2={ALPHA_R2}")
    md.append(f"- E_eq R1/R2 = {E_EQ_R1} / {E_EQ_R2} V (RHE)")
    md.append(f"- pass threshold: |Δobs|/max(|obs_4sp|) ≤ "
              f"{args.rel_tol_pct:.2f}% per voltage (matches F2 tolerance "
              f"used in `Forward/bv_solver/validation.py`)\n")
    md.append("## Per-voltage comparison\n")
    md.append("| V_RHE (V) | cd_4sp | cd_3sp | |Δcd|/cd_max% | "
              "pc_4sp | pc_3sp | |Δpc|/pc_max% | 3sp method | verdict |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
    def _fmt_obs(v):
        return "" if v is None else f"{v:+.4e}"
    for r in rows:
        if r["verdict"] == "MISSING":
            md.append(
                f"| {r['V_RHE']:+.3f} | "
                f"{_fmt_obs(r['cd_4sp'])} | {_fmt_obs(r['cd_3sp'])} | — | "
                f"{_fmt_obs(r['pc_4sp'])} | {_fmt_obs(r['pc_3sp'])} | — | "
                f"{r['method_3sp']} | MISSING |"
            )
        else:
            md.append(
                f"| {r['V_RHE']:+.3f} | "
                f"{r['cd_4sp']:+.4e} | {r['cd_3sp']:+.4e} | "
                f"{r['cd_err_pct_of_max']:.3f} | "
                f"{r['pc_4sp']:+.4e} | {r['pc_3sp']:+.4e} | "
                f"{r['pc_err_pct_of_max']:.3f} | "
                f"{r['method_3sp']} | {r['verdict']} |"
            )
    md.append("")
    md.append("## Aggregate\n")
    s = summary_stats
    md.append(f"- overlap voltages where both solvers converged: "
              f"{s['n_overlap']} / {NV}")
    if s["n_overlap"]:
        md.append(f"- max |Δcd|/cd_max: {s['cd_max_err_pct_of_max']:.3f}%")
        md.append(f"- mean |Δcd|/cd_max: {s['cd_mean_err_pct_of_max']:.3f}%")
        md.append(f"- max |Δpc|/pc_max: {s['pc_max_err_pct_of_max']:.3f}%")
        md.append(f"- mean |Δpc|/pc_max: {s['pc_mean_err_pct_of_max']:.3f}%")
        md.append(f"- per-voltage verdicts: PASS={s['n_pass']}, "
                  f"FLAG={s['n_flag']}")
        all_pass = (s["n_flag"] == 0)
        md.append("")
        md.append("## Verdict\n")
        if all_pass:
            md.append(
                f"All {s['n_overlap']} overlap voltages pass the "
                f"{args.rel_tol_pct:.1f}% F2-style tolerance on both CD and "
                f"PC.  This regenerates and quantifies the writeup's "
                f"\"matched closely within the validation tolerances\" "
                f"claim from real data and saves the artefacts under "
                f"`{out_subdir}/`.")
        else:
            md.append(
                f"{s['n_flag']} of {s['n_overlap']} overlap voltages exceed "
                f"the {args.rel_tol_pct:.1f}% threshold.  See `comparison.csv` "
                f"and `comparison.png` for per-voltage detail; the writeup's "
                f"overlap claim should be qualified to the subset that "
                f"passes, or the threshold revisited.")
    md.append("")
    md.append(f"Wall time: 4sp = {t_4sp:.1f}s, 3sp logc = {t_3sp:.1f}s.\n")
    md.append("Artefacts:\n")
    md.append(f"- `raw_values.json`")
    md.append(f"- `comparison.csv`")
    md.append(f"- `comparison.png`")

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if summary_stats["n_overlap"]:
        print(f"  overlap points: {summary_stats['n_overlap']}/{NV}")
        print(f"  CD max  err %: {summary_stats['cd_max_err_pct_of_max']:.3f}")
        print(f"  CD mean err %: {summary_stats['cd_mean_err_pct_of_max']:.3f}")
        print(f"  PC max  err %: {summary_stats['pc_max_err_pct_of_max']:.3f}")
        print(f"  PC mean err %: {summary_stats['pc_mean_err_pct_of_max']:.3f}")
        print(f"  PASS: {summary_stats['n_pass']}  "
              f"FLAG: {summary_stats['n_flag']}")
    else:
        print("  No overlap points — both solvers must converge at "
              "at least one voltage for a comparison.")
    print(f"  Saved: {OUT_DIR}/{{summary.md, comparison.csv, "
          f"comparison.png, raw_values.json}}")


if __name__ == "__main__":
    main()
