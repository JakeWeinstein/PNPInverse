"""Plot CD and PC I-V curves at TRUE parameters with the production
3-species + Boltzmann + log-c + log-rate solver.

Voltage continuation strategy: cold-start at V=0 (where R1 is mild), then
warm-start outward in both directions (toward +0.6 and toward -0.5).

Output:
    StudyResults/plot_iv_curves_3sp_true/
        iv_curves.npz   (V, cd, pc arrays; NaN for failed voltages)
        iv_linear.png   (CD vs V, PC vs V; linear scales)
        iv_loglinear.png (|CD| and |PC| vs V on semilog y)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-min", type=float, default=-0.5)
    parser.add_argument("--v-max", type=float, default=+0.6)
    parser.add_argument("--v-step", type=float, default=0.05)
    parser.add_argument("--mesh-ny", type=int, default=200)
    parser.add_argument("--out-base", type=str,
                        default="plot_iv_curves_3sp_true")
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

    OUT_DIR = os.path.join(_ROOT, "StudyResults", args.out_base)
    os.makedirs(OUT_DIR, exist_ok=True)

    SP_DICT = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    ss_rel_tol, ss_abs_tol, ss_consec = 1e-4, 1e-8, 4

    def make_3sp_sp(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = _make_bv_convergence_cfg(log_rate=True)
        params["nondim"] = _make_nondim_cfg()
        rxn1 = {
            "k0": k0_r1, "alpha": alpha_r1,
            "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
            "reversible": True, "E_eq_v": E_EQ_R1,
            "cathodic_conc_factors": [{"species": 2, "power": 2,
                                       "c_ref_nondim": C_HP_HAT}],
        }
        rxn2 = {
            "k0": k0_r2, "alpha": alpha_r2,
            "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
            "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
            "reversible": False, "E_eq_v": E_EQ_R2,
            "cathodic_conc_factors": [{"species": 2, "power": 2,
                                       "c_ref_nondim": C_HP_HAT}],
        }
        params["bv_bc"] = {
            "reactions": [rxn1, rxn2],
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
        W = ctx["W"]; U = ctx["U"]
        scaling = ctx["nondim"]
        phi = fd.split(U)[-1]
        w = fd.TestFunctions(W)[-1]
        dx = fd.Measure("dx", domain=ctx["mesh"])
        charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)),
                              fd.Constant(50.0))
        ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
        ctx["J_form"] = fd.derivative(ctx["F_res"], U)
        return ctx

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(args.mesh_ny), beta=3.0)

    def _snapshot(U): return tuple(d.data_ro.copy() for d in U.dat)
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
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=SP_DICT)
        of_cd = _build_bv_observable_form(ctx, mode="current_density",
                                           reaction_index=None, scale=-I_SCALE)
        of_pc = _build_bv_observable_form(ctx, mode="peroxide_current",
                                           reaction_index=None, scale=-I_SCALE)
        z_nominal = [float(sp[4][i]) for i in range(ctx["n_species"])]
        return ctx, sol, of_cd, of_pc, z_nominal

    def run_ss(ctx, sol, of_cd, max_steps=250):
        U = ctx["U"]; Up = ctx["U_prev"]; dt_const = ctx["dt_const"]
        dt_val = 0.25; dt_const.assign(dt_val)
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
                if d / sv <= ss_rel_tol or d <= ss_abs_tol: sc += 1
                else: sc = 0
                if prev_delta and d > 0:
                    r = prev_delta / d
                    dt_val = (min(dt_val * min(r, 4), 0.25 * 20)
                              if r > 1 else max(dt_val * 0.5, 0.25))
                    dt_const.assign(dt_val)
                prev_delta = d
            prev_flux = fv
            if sc >= ss_consec:
                return True
        return False

    def solve_cold(V_RHE, k0_1, k0_2, a_1, a_2, max_z_steps=20):
        """Cold-start with z-ramp from neutral to charged."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2)
        n = ctx["n_species"]; zc = ctx["z_consts"]
        with adj.stop_annotating():
            for i in range(n): zc[i].assign(0.0)
            paf = ctx["phi_applied_func"]; paf.assign(V_RHE / V_T)
            if not run_ss(ctx, sol, of_cd):
                return None, None, None
            # Ramp z 0 -> 1
            for k in range(1, max_z_steps + 1):
                z_factor = k / max_z_steps
                for i in range(n): zc[i].assign(z_nominal[i] * z_factor)
                if not run_ss(ctx, sol, of_cd):
                    # Bisect retry
                    z_factor = (k - 0.5) / max_z_steps
                    for i in range(n): zc[i].assign(z_nominal[i] * z_factor)
                    if not run_ss(ctx, sol, of_cd):
                        return None, None, None
                    for i in range(n): zc[i].assign(z_nominal[i] * (k / max_z_steps))
                    if not run_ss(ctx, sol, of_cd):
                        return None, None, None
        cd = float(fd.assemble(of_cd))
        pc = float(fd.assemble(of_pc))
        return cd, pc, _snapshot(ctx["U"])

    def solve_warm(V_RHE, k0_1, k0_2, a_1, a_2, ic_data, max_steps=250):
        """Warm-start solve, no annotation."""
        ctx, sol, of_cd, of_pc, z_nominal = build_solve(
            V_RHE, k0_1, k0_2, a_1, a_2)
        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]; n = ctx["n_species"]
        paf = ctx["phi_applied_func"]
        with adj.stop_annotating():
            _restore(ic_data, U, Up)
            for i in range(n): zc[i].assign(z_nominal[i])
            paf.assign(V_RHE / V_T)
            if not run_ss(ctx, sol, of_cd, max_steps=max_steps):
                return None, None, None
        cd = float(fd.assemble(of_cd))
        pc = float(fd.assemble(of_pc))
        return cd, pc, _snapshot(U)

    # ----------------------------------------------------------------
    # Voltage grid + continuation strategy
    # ----------------------------------------------------------------
    n_pts = int(round((args.v_max - args.v_min) / args.v_step)) + 1
    V_GRID = np.linspace(args.v_min, args.v_max, n_pts)
    NV = len(V_GRID)
    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)

    print(f"V grid ({NV} pts): "
          f"[{V_GRID.min():.3f}, {V_GRID.max():.3f}] step {args.v_step:.3f}")

    # Find anchor index nearest V=0 for cold-start
    anchor_idx = int(np.argmin(np.abs(V_GRID)))
    print(f"Anchor at V={V_GRID[anchor_idx]:+.3f} (cold-start)")
    t0 = time.time()
    cd, pc, anchor_snap = solve_cold(float(V_GRID[anchor_idx]),
                                      K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    if cd is None:
        print(f"  ANCHOR FAILED — aborting")
        sys.exit(2)
    cd_arr[anchor_idx] = cd; pc_arr[anchor_idx] = pc
    print(f"  V={V_GRID[anchor_idx]:+.3f}: cd={cd:+.4e}, pc={pc:+.4e}  "
          f"({time.time()-t0:.1f}s)")

    # Sweep upward (anchor+1, anchor+2, ..., NV-1)
    last_snap = anchor_snap
    for i in range(anchor_idx + 1, NV):
        V = float(V_GRID[i])
        t_v = time.time()
        cd, pc, snap = solve_warm(V, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
                                   last_snap)
        if cd is None:
            cd, pc, snap = solve_cold(V, K0_HAT_R1, K0_HAT_R2,
                                       ALPHA_R1, ALPHA_R2)
        if cd is None:
            print(f"  V={V:+.3f}: FAILED  ({time.time()-t_v:.1f}s)")
            continue
        cd_arr[i] = cd; pc_arr[i] = pc; last_snap = snap
        print(f"  V={V:+.3f}: cd={cd:+.4e}, pc={pc:+.4e}  "
              f"({time.time()-t_v:.1f}s)")

    # Sweep downward (anchor-1, anchor-2, ..., 0)
    last_snap = anchor_snap
    for i in range(anchor_idx - 1, -1, -1):
        V = float(V_GRID[i])
        t_v = time.time()
        cd, pc, snap = solve_warm(V, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
                                   last_snap)
        if cd is None:
            cd, pc, snap = solve_cold(V, K0_HAT_R1, K0_HAT_R2,
                                       ALPHA_R1, ALPHA_R2)
        if cd is None:
            print(f"  V={V:+.3f}: FAILED  ({time.time()-t_v:.1f}s)")
            continue
        cd_arr[i] = cd; pc_arr[i] = pc; last_snap = snap
        print(f"  V={V:+.3f}: cd={cd:+.4e}, pc={pc:+.4e}  "
              f"({time.time()-t_v:.1f}s)")

    # ----------------------------------------------------------------
    # Save and plot
    # ----------------------------------------------------------------
    n_ok = int(np.sum(~np.isnan(cd_arr)))
    print(f"\n{n_ok}/{NV} voltages converged.")

    npz_path = os.path.join(OUT_DIR, "iv_curves.npz")
    np.savez(npz_path, V=V_GRID, cd=cd_arr, pc=pc_arr,
             true_params={"k0_1": K0_HAT_R1, "k0_2": K0_HAT_R2,
                          "alpha_1": ALPHA_R1, "alpha_2": ALPHA_R2})
    print(f"Data: {npz_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Linear ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ok = ~np.isnan(cd_arr)
    ax1.plot(V_GRID[ok], cd_arr[ok], "o-", color="C0", lw=1.5, ms=4)
    ax1.axhline(0, color="0.6", lw=0.5)
    ax1.set_ylabel(r"CD = current density (nondim)")
    ax1.set_title("3sp + Boltzmann + log-c + log-rate, TRUE parameters")
    ax1.grid(True, alpha=0.3)

    ax2.plot(V_GRID[ok], pc_arr[ok], "s-", color="C1", lw=1.5, ms=4)
    ax2.axhline(0, color="0.6", lw=0.5)
    ax2.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax2.set_ylabel(r"PC = peroxide current (nondim)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    lin_path = os.path.join(OUT_DIR, "iv_linear.png")
    fig.savefig(lin_path, dpi=160)
    plt.close(fig)
    print(f"Linear plot: {lin_path}")

    # --- Tafel-style log|i| ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax1.semilogy(V_GRID[ok], np.abs(cd_arr[ok]), "o-", color="C0", lw=1.5, ms=4)
    ax1.set_ylabel(r"$|$CD$|$ (nondim)")
    ax1.set_title("Tafel view: |observable| vs $V_{\\mathrm{RHE}}$")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogy(V_GRID[ok], np.abs(pc_arr[ok]), "s-", color="C1", lw=1.5, ms=4)
    ax2.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")
    ax2.set_ylabel(r"$|$PC$|$ (nondim)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    log_path = os.path.join(OUT_DIR, "iv_loglinear.png")
    fig.savefig(log_path, dpi=160)
    plt.close(fig)
    print(f"Log plot:    {log_path}")


if __name__ == "__main__":
    main()
