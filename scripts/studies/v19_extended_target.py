"""Build the extended hybrid I-V target: 4sp v-chain cathodic + logc onset.

Combines the two proven forward paths into a single target spanning
V=[-1.2, +0.25]V. Saves pre-computed U arrays per voltage for use as
seed IC cache in the inference driver.

Outputs:
  StudyResults/v19_extended/target.npz      (V, cd, pc, solver_used)
  StudyResults/v19_extended/U_seed.pkl      (per-V initial-condition arrays)
"""
from __future__ import annotations
import os, sys, time, pickle

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


E_EQ_R1 = 0.68
E_EQ_R2 = 1.78
H2O2_SEED = 1e-4

# Voltage grid: cathodic (4sp v-chain) + onset (logc cold z-ramp)
V_CATHODIC = np.array([
    -1.20, -1.00, -0.80, -0.60, -0.50, -0.40, -0.35, -0.30,
    -0.25, -0.20, -0.15, -0.10, -0.05,
])
V_ONSET = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])


def _run_ss(sol, U, Up, dt_const, form, max_steps=60, rel_tol=1e-4, abs_tol=1e-8):
    import firedrake as fd
    import pyadjoint as adj
    dt_val = 0.25; dt_const.assign(dt_val)
    prev_flux = None; prev_delta = None; sc = 0
    for _ in range(max_steps):
        try: sol.solve()
        except Exception: return False
        with adj.stop_annotating():
            Up.assign(U)
        fv = float(fd.assemble(form))
        if prev_flux is not None:
            d = abs(fv - prev_flux); sv = max(abs(fv), abs(prev_flux), 1e-8)
            if d / sv <= rel_tol or d <= abs_tol: sc += 1
            else: sc = 0
            if prev_delta and d > 0:
                r = prev_delta / d
                dt_val = (min(dt_val * min(r, 4), 5.0) if r > 1
                          else max(dt_val * 0.5, 0.25))
                dt_const.assign(dt_val)
            prev_delta = d
        prev_flux = fv
        if sc >= 4: return True
    return False


def _build_4sp_solver(k0_1, k0_2, a1, a2, V_hat, mesh, include_z_ramp=True):
    """Create ctx for 4sp, initialize ICs. Returns (ctx, sp, sol, form_cd, form_pc)."""
    import firedrake as fd
    import pyadjoint as adj
    from scripts._bv_common import (
        V_T, I_SCALE,
        FOUR_SPECIES_CHARGED, make_bv_solver_params, SNES_OPTS_CHARGED,
    )
    from Forward.bv_solver.forms import (
        build_context, build_forms, set_initial_conditions,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    sp = make_bv_solver_params(
        eta_hat=V_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_1, k0_hat_r2=k0_2,
        alpha_r1=a1, alpha_r2=a2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    with adj.stop_annotating():
        ctx = build_context(list(sp), mesh=mesh)
        ctx = build_forms(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    sp_dict["snes_error_if_not_converged"] = True
    sp_dict["snes_linesearch_maxlambda"] = 0.3
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
    return ctx, sp, sol, form_cd, form_pc


def _build_logc_solver(k0_1, k0_2, a1, a2, V_hat, mesh):
    """Create ctx for logc+Boltzmann. Returns (ctx, sp, sol, form_cd, form_pc)."""
    import firedrake as fd
    import pyadjoint as adj
    from scripts._bv_common import (
        V_T, I_SCALE,
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT, C_O2_HAT, C_HP_HAT, C_CLO4_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, _make_bv_convergence_cfg, SNES_OPTS_CHARGED,
    )
    from Forward.params import SolverParams
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    params = dict(SNES_OPTS_CHARGED)
    params["bv_convergence"] = _make_bv_convergence_cfg()
    params["nondim"] = _make_nondim_cfg()
    r1 = {
        "k0": k0_1, "alpha": a1,
        "cathodic_species": 0, "anodic_species": 1, "c_ref": 1.0,
        "stoichiometry": [-1, +1, -2], "n_electrons": N_ELECTRONS,
        "reversible": True, "E_eq_v": E_EQ_R1,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    r2 = {
        "k0": k0_2, "alpha": a2,
        "cathodic_species": 1, "anodic_species": None, "c_ref": 0.0,
        "stoichiometry": [0, -1, -2], "n_electrons": N_ELECTRONS,
        "reversible": False, "E_eq_v": E_EQ_R2,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    params["bv_bc"] = {
        "reactions": [r1, r2],
        "k0": [k0_1] * 3, "alpha": [a1] * 3,
        "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    sp = SolverParams.from_list([
        3, 1, 0.25, 80.0,
        [0, 0, 1], [D_O2_HAT, D_H2O2_HAT, D_HP_HAT], [A_DEFAULT] * 3,
        V_hat, [C_O2_HAT, H2O2_SEED, C_HP_HAT], 0.0, params,
    ])
    with adj.stop_annotating():
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        # Boltzmann monkey-patch
        phi = fd.split(ctx["U"])[-1]
        w = fd.TestFunctions(ctx["W"])[-1]
        dx = fd.Measure("dx", domain=mesh)
        charge_rhs = fd.Constant(float(ctx["nondim"]["charge_rhs_prefactor"]))
        c_bulk = fd.Constant(C_CLO4_HAT)
        phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0))
        ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
        ctx["J_form"] = fd.derivative(ctx["F_res"], ctx["U"])
        set_initial_conditions_logc(ctx, list(sp))

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    sp_dict["snes_error_if_not_converged"] = True
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
    return ctx, sp, sol, form_cd, form_pc


def main():
    from scripts._bv_common import (
        setup_firedrake_env, V_T,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
    )
    setup_firedrake_env()
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh

    out_dir = os.path.join(_ROOT, "StudyResults", "v19_extended")
    os.makedirs(out_dir, exist_ok=True)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    print("=" * 70)
    print("EXTENDED TARGET: 4sp v-chain cathodic + logc onset")
    print(f"  Cathodic: {len(V_CATHODIC)} points V in [{V_CATHODIC.min():+.2f}, "
          f"{V_CATHODIC.max():+.2f}]")
    print(f"  Onset:    {len(V_ONSET)} points V in [{V_ONSET.min():+.2f}, "
          f"{V_ONSET.max():+.2f}]")
    print("=" * 70)

    # -------- 4sp v-chain for cathodic range --------
    # Walk from V=-0.05 (cold-start target) down through all V_CATHODIC values.
    # Start at V_CHAIN_START which MUST be easy to cold-start.
    V_CHAIN_START = -0.05
    print(f"\n[cathodic] cold-start at V={V_CHAIN_START:+.3f}")
    ctx_c, sp_c, sol_c, form_cd_c, form_pc_c = _build_4sp_solver(
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, V_CHAIN_START / V_T, mesh,
    )
    U_c = ctx_c["U"]; Up_c = ctx_c["U_prev"]
    zc = ctx_c["z_consts"]; n = ctx_c["n_species"]
    dt_const = ctx_c["dt_const"]; paf = ctx_c["phi_applied_func"]

    with adj.stop_annotating():
        for zci in zc: zci.assign(0.0)
        paf.assign(V_CHAIN_START / V_T)
        if not _run_ss(sol_c, U_c, Up_c, dt_const, form_cd_c, max_steps=100):
            print("FAIL: z=0 cold start at chain start")
            return
        z_nominal = [float(sp_c[4][i]) for i in range(n)]
        achieved = 0.0
        for z_val in np.linspace(0, 1, 21)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U_c.dat)
            for i in range(n): zc[i].assign(z_nominal[i] * z_val)
            if _run_ss(sol_c, U_c, Up_c, dt_const, form_cd_c, max_steps=60):
                achieved = z_val
            else:
                for src, dst in zip(U_ckpt, U_c.dat): dst.data[:] = src
                Up_c.assign(U_c); break
        if achieved < 1.0 - 1e-3:
            print(f"FAIL: z-ramp stalled at z={achieved:.3f}")
            return

    # Walk the v-chain in descending order through V_CATHODIC
    V_sorted = sorted(V_CATHODIC.tolist(), reverse=True)  # -0.05 ... -1.20
    cathodic_results = []
    V_cur = V_CHAIN_START
    step_size = 0.05  # nominal step
    for V_target in V_sorted:
        # Step from V_cur down to V_target in sub-steps if needed.
        sub_step = step_size
        while V_cur > V_target + 1e-6:
            V_next = max(V_cur - sub_step, V_target)
            U_ckpt = tuple(d.data_ro.copy() for d in U_c.dat)
            paf.assign(V_next / V_T)
            with adj.stop_annotating():
                ok = _run_ss(sol_c, U_c, Up_c, dt_const, form_cd_c, max_steps=60)
            if ok:
                V_cur = V_next
                sub_step = min(sub_step * 1.3, 0.1)
            else:
                for src, dst in zip(U_ckpt, U_c.dat): dst.data[:] = src
                Up_c.assign(U_c); paf.assign(V_cur / V_T)
                sub_step *= 0.5
                if sub_step < 1e-3:
                    print(f"  stalled walking to V={V_target:+.3f}, giving up at V={V_cur:+.4f}")
                    break
        # Record at V_target (if reached)
        if abs(V_cur - V_target) < 1e-4:
            cd = float(fd.assemble(form_cd_c))
            pc = float(fd.assemble(form_pc_c))
            U_arrays = tuple(np.asarray(d.data_ro, dtype=float).copy() for d in U_c.dat)
            cathodic_results.append({
                "V": V_target, "cd": cd, "pc": pc, "solver": "4sp", "U": U_arrays,
            })
            print(f"  V={V_target:+.4f} (4sp)  cd={cd:+.6f}  pc={pc:+.6f}")

    # -------- logc for onset range (cold-start at each V, all easy) --------
    print(f"\n[onset] logc cold-start at each voltage")
    onset_results = []
    for V in V_ONSET:
        ctx_l, sp_l, sol_l, form_cd_l, form_pc_l = _build_logc_solver(
            K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, V / V_T, mesh,
        )
        U_l = ctx_l["U"]; Up_l = ctx_l["U_prev"]
        zc_l = ctx_l["z_consts"]; n_l = ctx_l["n_species"]
        dt_l = ctx_l["dt_const"]; paf_l = ctx_l["phi_applied_func"]

        with adj.stop_annotating():
            for zci in zc_l: zci.assign(0.0)
            paf_l.assign(V / V_T)
            if not _run_ss(sol_l, U_l, Up_l, dt_l, form_cd_l, max_steps=100):
                print(f"  V={V:+.4f} (logc): z=0 FAIL")
                continue
            z_nominal_l = [float(sp_l[4][i]) for i in range(n_l)]
            achieved_l = 0.0
            for z_val in np.linspace(0, 1, 21)[1:]:
                U_ck = tuple(d.data_ro.copy() for d in U_l.dat)
                for i in range(n_l): zc_l[i].assign(z_nominal_l[i] * z_val)
                if _run_ss(sol_l, U_l, Up_l, dt_l, form_cd_l, max_steps=60):
                    achieved_l = z_val
                else:
                    for src, dst in zip(U_ck, U_l.dat): dst.data[:] = src
                    Up_l.assign(U_l); break
            if achieved_l < 1.0 - 1e-3:
                print(f"  V={V:+.4f} (logc): z-ramp stalled at {achieved_l:.3f}")
                continue

        cd = float(fd.assemble(form_cd_l))
        pc = float(fd.assemble(form_pc_l))
        U_arrays = tuple(np.asarray(d.data_ro, dtype=float).copy() for d in U_l.dat)
        onset_results.append({
            "V": float(V), "cd": cd, "pc": pc, "solver": "logc", "U": U_arrays,
        })
        print(f"  V={V:+.4f} (logc)  cd={cd:+.6f}  pc={pc:+.6f}")

    # -------- Combine, save --------
    all_results = sorted(cathodic_results + onset_results, key=lambda r: r["V"])
    V_arr = np.array([r["V"] for r in all_results])
    cd_arr = np.array([r["cd"] for r in all_results])
    pc_arr = np.array([r["pc"] for r in all_results])
    solver_arr = np.array([r["solver"] for r in all_results])

    print("\n" + "=" * 70)
    print("FULL I-V TARGET")
    print("=" * 70)
    print(f"  {'V_RHE':>8} {'solver':>6} {'cd':>12} {'pc':>12} {'|PC/CD|':>8}")
    for V, cd, pc, solv in zip(V_arr, cd_arr, pc_arr, solver_arr):
        ratio = f"{abs(pc/cd):.4f}" if abs(cd) > 1e-12 else "NaN"
        print(f"  {V:+8.4f} {solv:>6} {cd:+12.6f} {pc:+12.6f} {ratio:>8}")

    out_npz = os.path.join(out_dir, "target.npz")
    np.savez(
        out_npz,
        V_RHE=V_arr, cd=cd_arr, pc=pc_arr, solver_used=solver_arr,
        k0_r1=K0_HAT_R1, k0_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
    )

    out_pkl = os.path.join(out_dir, "U_seed.pkl")
    U_seed = {r["V"]: r["U"] for r in all_results}
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "V_RHE": V_arr.tolist(),
            "solver_used": solver_arr.tolist(),
            "U_per_V": [r["U"] for r in all_results],
        }, f)

    print(f"\nSaved target -> {out_npz}")
    print(f"Saved U seed -> {out_pkl}")


if __name__ == "__main__":
    main()
