"""V18 k0 sensitivity test using log-concentration 3sp+Boltzmann solver.

If k0_1 is identifiable from I-V curves in the working voltage range,
different k0 values should produce DISTINGUISHABLE cd curves (beyond noise).
This test measures that sensitivity directly.

Goal: determine if k0 inference is feasible from the onset-region data
that log-c can reliably produce (V = -0.10 to +0.30V).
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

    # Voltage range where log-c was shown to converge reliably
    V_GRID = np.array([-0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30])

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
        W = ctx["W"]; U = ctx["U"]; n = ctx["n_species"]
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

    def solve_point(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh, z_steps=20):
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
        opc = _build_bv_observable_form(ctx, mode="peroxide_current",
                                         reaction_index=None, scale=-I_SCALE)

        dt_init = 0.25
        def run_ss(max_steps=60):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; sc = 0
            for s in range(1, max_steps+1):
                try: sol.solve()
                except Exception: return False, s-1
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
                if sc >= 4: return True, s
            return False, max_steps

        # z=0
        for zci in zc: zci.assign(0.0)
        paf.assign(phi_hat)
        with adj.stop_annotating():
            ok0, _ = run_ss(100)
        if not ok0:
            return None, None, 0.0

        # z-ramp
        z_nominal = [float(sp[4][i]) for i in range(n)]
        achieved_z = 0.0
        with adj.stop_annotating():
            for z_val in np.linspace(0, 1, z_steps+1)[1:]:
                U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
                for i in range(n): zc[i].assign(z_nominal[i]*z_val)
                ok_z, _ = run_ss(60)
                if ok_z:
                    achieved_z = z_val
                else:
                    for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                    Up.assign(U)
                    break

        if achieved_z < 1.0 - 1e-3:
            return None, None, achieved_z

        cd = float(fd.assemble(of))
        pc = float(fd.assemble(opc))
        return cd, pc, achieved_z

    # -- Run sensitivity sweep --
    multipliers = [0.2, 0.5, 1.0, 2.0, 5.0]
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    results = {}
    for mult in multipliers:
        k0_r1 = K0_HAT_R1 * mult
        print(f"\n{'='*60}\n  k0_r1 x{mult} = {k0_r1:.3e}\n{'='*60}")
        cd_arr = np.full(len(V_GRID), np.nan)
        pc_arr = np.full(len(V_GRID), np.nan)
        z_arr = np.full(len(V_GRID), 0.0)
        for i, V in enumerate(V_GRID):
            t0 = time.time()
            cd, pc, z = solve_point(V, k0_r1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, mesh)
            dt = time.time() - t0
            if cd is not None:
                cd_arr[i] = cd
                pc_arr[i] = pc
                z_arr[i] = z
                print(f"  V={V:+.3f}V: cd={cd:+.6f}, pc={pc:+.6f}, z={z:.3f}, t={dt:.1f}s")
            else:
                print(f"  V={V:+.3f}V: FAILED (z={z:.3f}), t={dt:.1f}s")
        results[mult] = {"cd": cd_arr, "pc": pc_arr, "z": z_arr}

    # -- Summary --
    print(f"\n{'='*80}")
    print("K0 SENSITIVITY SUMMARY (log-c 3sp+Boltzmann)")
    print(f"{'='*80}")
    hdr = f"  {'V_RHE':>7}"
    for m in multipliers:
        hdr += f"  {'x'+str(m)+' cd':>11}"
    print(hdr)
    for i, V in enumerate(V_GRID):
        row = f"  {V:+7.3f}"
        for m in multipliers:
            cd = results[m]["cd"][i]
            row += f"  {cd:+11.5f}" if np.isfinite(cd) else f"  {'  NaN':>11}"
        print(row)

    # Sensitivity: max |Δcd| across k0 sweep
    print(f"\n  {'V_RHE':>7}  {'max |Δcd|':>11}  {'rel Δ (%)':>10}  k0-sensitive?")
    for i, V in enumerate(V_GRID):
        cds = np.array([results[m]["cd"][i] for m in multipliers])
        valid = np.isfinite(cds)
        if valid.sum() < 2:
            print(f"  {V:+7.3f}  {'N/A':>11}  {'N/A':>10}")
            continue
        delta = cds[valid].max() - cds[valid].min()
        rel = 100 * delta / max(abs(cds[valid]).max(), 1e-10)
        flag = "*** YES" if rel > 5 else ("partial" if rel > 1 else "no")
        print(f"  {V:+7.3f}  {delta:+11.5f}  {rel:10.2f}  {flag}")


if __name__ == "__main__":
    main()
