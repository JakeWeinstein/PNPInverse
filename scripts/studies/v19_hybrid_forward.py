"""Hybrid forward solver for CD + PC across the full V range.

Per-voltage dispatch:
  V_RHE >= V_THRESHOLD: log-c 3sp + Boltzmann ClO4-  (z=1 direct)
  V_RHE <  V_THRESHOLD: standard 4sp                  (z=1 via charge continuation)

Both paths return (cd, pc) in the same scale (I_SCALE). Agreement at the
boundary (V ~ V_THRESHOLD) is a sanity check on the physics match between
the two models (they differ in ClO4- treatment: dynamic vs Boltzmann).

Outputs: StudyResults/v19_hybrid/iv_target.npz with V_GRID, cd, pc, z_ach, solver_used.
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


E_EQ_R1 = 0.68
E_EQ_R2 = 1.78
H2O2_SEED = 1e-4  # log-c seed; see docs/k0_inference_status.md
V_THRESHOLD = 0.0  # V vs RHE: logc at/above, 4sp below
Z_STEPS = 20


def _build_logc_solver_params(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
    """3sp (O2, H2O2, H+) solver params for log-c + Boltzmann path."""
    from scripts._bv_common import (
        D_O2_HAT, D_H2O2_HAT, D_HP_HAT, C_O2_HAT, C_HP_HAT,
        A_DEFAULT, N_ELECTRONS,
        _make_nondim_cfg, _make_bv_convergence_cfg, SNES_OPTS_CHARGED,
    )
    from Forward.params import SolverParams

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
    z = [0, 0, 1]
    D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
    A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
    c0 = [C_O2_HAT, H2O2_SEED, C_HP_HAT]
    return SolverParams.from_list([
        3, 1, 0.25, 80.0, z, D, A, eta_hat, c0, 0.0, params,
    ])


def _build_4sp_solver_params(eta_hat, k0_r1, k0_r2, alpha_r1, alpha_r2):
    """4sp (O2, H2O2, H+, ClO4-) solver params for standard path."""
    from scripts._bv_common import (
        FOUR_SPECIES_CHARGED, make_bv_solver_params, SNES_OPTS_CHARGED,
    )
    return make_bv_solver_params(
        eta_hat=eta_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )


def _add_boltzmann_clo4(ctx):
    """Monkey-patch F_res with Boltzmann ClO4- term (for 3sp logc only)."""
    import firedrake as fd
    from scripts._bv_common import C_CLO4_HAT

    mesh = ctx["mesh"]
    W = ctx["W"]; U = ctx["U"]
    scaling = ctx["nondim"]
    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh)
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    c_bulk = fd.Constant(C_CLO4_HAT)
    phi_cl = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0))
    # z_ClO4 = -1, so ρ contribution = -c_bulk*exp(phi). Residual subtracts
    # (charge_rhs * z * c_ClO4), so we subtract charge_rhs*(-1)*c_bulk*exp(phi):
    ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_bulk * fd.exp(phi_cl) * w * dx
    ctx["J_form"] = fd.derivative(ctx["F_res"], U)
    return ctx


def _run_to_ss(sol, U, Up, dt_const, obs_form, dt_init=0.25, max_steps=60,
               rel_tol=1e-4, abs_tol=1e-8):
    """Pseudo-transient continuation: assign Up, solve, track flux until steady."""
    import firedrake as fd
    dt_val = dt_init; dt_const.assign(dt_val)
    prev_flux = None; prev_delta = None; sc = 0
    for _ in range(1, max_steps + 1):
        try:
            sol.solve()
        except Exception:
            return False
        Up.assign(U)
        fv = float(fd.assemble(obs_form))
        if prev_flux is not None:
            d = abs(fv - prev_flux)
            sv = max(abs(fv), abs(prev_flux), 1e-8)
            if d / sv <= rel_tol or d <= abs_tol:
                sc += 1
            else:
                sc = 0
            if prev_delta and d > 0:
                r = prev_delta / d
                dt_val = (min(dt_val * min(r, 4), dt_init * 20) if r > 1
                          else max(dt_val * 0.5, dt_init))
                dt_const.assign(dt_val)
            prev_delta = d
        prev_flux = fv
        if sc >= 4:
            return True
    return False


def solve_point_logc(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh, z_steps=Z_STEPS):
    """Log-c 3sp+Boltzmann forward at a single voltage. Returns (cd, pc, z_ach)."""
    import firedrake as fd
    import pyadjoint as adj
    from scripts._bv_common import V_T, I_SCALE, SNES_OPTS_CHARGED
    from Forward.bv_solver.forms_logc import (
        build_context_logc, build_forms_logc, set_initial_conditions_logc,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    phi_hat = V_RHE / V_T
    sp = _build_logc_solver_params(phi_hat, k0_r1, k0_r2, alpha_r1, alpha_r2)
    with adj.stop_annotating():
        ctx = build_context_logc(list(sp), mesh=mesh)
        ctx = build_forms_logc(ctx, list(sp))
        ctx = _add_boltzmann_clo4(ctx)
        set_initial_conditions_logc(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

    # z=0 steady
    for zci in zc: zci.assign(0.0)
    paf.assign(phi_hat)
    with adj.stop_annotating():
        if not _run_to_ss(sol, U, Up, dt_const, form_cd, max_steps=100):
            return np.nan, np.nan, 0.0

    # z-ramp to z=1
    z_nominal = [float(sp[4][i]) for i in range(n)]
    achieved_z = 0.0
    with adj.stop_annotating():
        for z_val in np.linspace(0, 1, z_steps + 1)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n):
                zc[i].assign(z_nominal[i] * z_val)
            if _run_to_ss(sol, U, Up, dt_const, form_cd, max_steps=60):
                achieved_z = z_val
            else:
                for src, dst in zip(U_ckpt, U.dat):
                    dst.data[:] = src
                Up.assign(U)
                break

    if achieved_z < 1.0 - 1e-3:
        return np.nan, np.nan, achieved_z
    return float(fd.assemble(form_cd)), float(fd.assemble(form_pc)), achieved_z


def solve_point_4sp(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh, z_steps=Z_STEPS):
    """Standard 4sp forward at a single voltage. Returns (cd, pc, z_ach)."""
    import firedrake as fd
    import pyadjoint as adj
    from scripts._bv_common import V_T, I_SCALE, SNES_OPTS_CHARGED
    from Forward.bv_solver.forms import (
        build_context, build_forms, set_initial_conditions,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    phi_hat = V_RHE / V_T
    sp = _build_4sp_solver_params(phi_hat, k0_r1, k0_r2, alpha_r1, alpha_r2)
    with adj.stop_annotating():
        ctx = build_context(list(sp), mesh=mesh)
        ctx = build_forms(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)
    form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)

    for zci in zc: zci.assign(0.0)
    paf.assign(phi_hat)
    with adj.stop_annotating():
        if not _run_to_ss(sol, U, Up, dt_const, form_cd, max_steps=100):
            return np.nan, np.nan, 0.0

    z_nominal = [float(sp[4][i]) for i in range(n)]
    achieved_z = 0.0
    with adj.stop_annotating():
        for z_val in np.linspace(0, 1, z_steps + 1)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n):
                zc[i].assign(z_nominal[i] * z_val)
            if _run_to_ss(sol, U, Up, dt_const, form_cd, max_steps=60):
                achieved_z = z_val
            else:
                for src, dst in zip(U_ckpt, U.dat):
                    dst.data[:] = src
                Up.assign(U)
                break

    if achieved_z < 1.0 - 1e-3:
        return np.nan, np.nan, achieved_z
    return float(fd.assemble(form_cd)), float(fd.assemble(form_pc)), achieved_z


def solve_point_hybrid(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh,
                      v_threshold=V_THRESHOLD):
    """Dispatch to the appropriate solver based on V_RHE."""
    if V_RHE >= v_threshold:
        cd, pc, z = solve_point_logc(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh)
        return cd, pc, z, "logc"
    else:
        cd, pc, z = solve_point_4sp(V_RHE, k0_r1, k0_r2, alpha_r1, alpha_r2, mesh)
        return cd, pc, z, "4sp"


def main():
    from scripts._bv_common import (
        setup_firedrake_env, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
    )
    setup_firedrake_env()

    from Forward.bv_solver import make_graded_rectangle_mesh

    out_dir = os.path.join(_ROOT, "StudyResults", "v19_hybrid")
    os.makedirs(out_dir, exist_ok=True)

    # Voltage grid: cathodic + onset, plus overlap points at V=-0.05 and V=0.05
    # for physics-match check near the threshold.
    V_GRID = np.array([
        -0.50, -0.40, -0.30, -0.20, -0.10, -0.05,
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
    ])

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    print(f"{'='*70}")
    print("HYBRID FORWARD: logc 3sp+Boltzmann (V>=0) + standard 4sp (V<0)")
    print(f"{'='*70}")
    print(f"V threshold = {V_THRESHOLD:+.3f} V")
    print(f"True params: k0_r1={K0_HAT_R1:.4e}, k0_r2={K0_HAT_R2:.4e}, "
          f"α1={ALPHA_R1:.4f}, α2={ALPHA_R2:.4f}")
    print(f"V_GRID ({len(V_GRID)} points): {V_GRID.tolist()}")

    cd = np.full(len(V_GRID), np.nan)
    pc = np.full(len(V_GRID), np.nan)
    z_ach = np.full(len(V_GRID), 0.0)
    solver_used = ["" for _ in V_GRID]
    times = np.zeros(len(V_GRID))

    for i, V in enumerate(V_GRID):
        t0 = time.time()
        cd_i, pc_i, z_i, s_i = solve_point_hybrid(
            V, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, mesh)
        times[i] = time.time() - t0
        cd[i] = cd_i; pc[i] = pc_i; z_ach[i] = z_i; solver_used[i] = s_i
        status = "OK" if np.isfinite(cd_i) else "FAIL"
        print(f"  V={V:+.3f}V [{s_i:>4}]: cd={cd_i:+.6f}, pc={pc_i:+.6f}, "
              f"z={z_i:.3f}, t={times[i]:.1f}s [{status}]")

    # Physics-match check at V=0.0 (both solvers exist adjacent; the 4sp at
    # V=-0.05 and logc at V=0.00 are adjacent and should produce similar cd/pc)
    print(f"\n{'='*70}")
    print("PHYSICS-MATCH CHECK at threshold region")
    print(f"{'='*70}")
    idx_m05 = np.where(V_GRID == -0.05)[0][0]
    idx_p00 = np.where(V_GRID == 0.00)[0][0]
    idx_p05 = np.where(V_GRID == 0.05)[0][0]
    print(f"  V=-0.05 (4sp):  cd={cd[idx_m05]:+.6f}, pc={pc[idx_m05]:+.6f}")
    print(f"  V=+0.00 (logc): cd={cd[idx_p00]:+.6f}, pc={pc[idx_p00]:+.6f}")
    print(f"  V=+0.05 (logc): cd={cd[idx_p05]:+.6f}, pc={pc[idx_p05]:+.6f}")
    # If the two physics are close, linearly interpolating the 4sp to V=0
    # from V=-0.05 and V=-0.10 should match the logc at V=0 within small error.
    idx_m10 = np.where(V_GRID == -0.10)[0][0]
    if np.isfinite(cd[idx_m05]) and np.isfinite(cd[idx_m10]):
        cd_4sp_at_0 = cd[idx_m05] + (cd[idx_m05] - cd[idx_m10])  # linear extrap
        if np.isfinite(cd[idx_p00]):
            mismatch = 100 * abs(cd_4sp_at_0 - cd[idx_p00]) / abs(cd[idx_p00])
            print(f"  4sp linear-extrap to V=0: cd≈{cd_4sp_at_0:+.6f}")
            print(f"  logc vs extrap-4sp mismatch at V=0: {mismatch:.2f}%")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'V_RHE':>8} {'solver':>6} {'cd':>12} {'pc':>12} {'|PC/CD|':>8} {'z':>6}")
    for i, V in enumerate(V_GRID):
        ratio = f"{abs(pc[i]/cd[i]):.4f}" if (np.isfinite(pc[i]) and np.isfinite(cd[i])
                                               and abs(cd[i]) > 1e-12) else "NaN"
        print(f"  {V:+8.3f} {solver_used[i]:>6} {cd[i]:+12.6f} {pc[i]:+12.6f} "
              f"{ratio:>8} {z_ach[i]:6.3f}")

    out_path = os.path.join(out_dir, "iv_target.npz")
    np.savez(
        out_path,
        V_RHE=V_GRID, cd=cd, pc=pc, z_achieved=z_ach,
        solver_used=np.array(solver_used), times=times,
        k0_r1=K0_HAT_R1, k0_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        v_threshold=V_THRESHOLD,
    )
    print(f"\nSaved I-V curve to {out_path}")
    print(f"Total time: {np.sum(times):.1f}s ({np.sum(times)/60:.1f} min)")


if __name__ == "__main__":
    main()
