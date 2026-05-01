"""V18: 3-species model with Boltzmann background for ClO4-.

Drops ClO4- as a dynamic species. Instead, uses c_ClO4 = c_bulk*exp(phi)
(Boltzmann equilibrium) in the Poisson source. This:
1. Eliminates the species that causes convergence failure
2. Maintains Debye layer screening through the PB exp(phi) term
3. Is physically justified: ClO4- is a spectator ion in equilibrium
4. Keeps full H+ dynamics (consumed + replenished by diffusion+migration)

System: O2(z=0), H2O2(z=0), H+(z=+1), phi with PB background.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    make_bv_solver_params, SNES_OPTS_CHARGED,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
    D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
    C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT,
    A_DEFAULT, N_ELECTRONS,
    D_REF, C_SCALE, L_REF, K_SCALE,
    _make_nondim_cfg, _make_bv_convergence_cfg,
)
setup_firedrake_env()

import numpy as np
import time
import firedrake as fd
import pyadjoint as adj
from Forward.bv_solver import make_graded_rectangle_mesh
from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
from Forward.bv_solver.observables import _build_bv_observable_form
from Forward.bv_solver.validation import validate_solution_state
from Forward.params import SolverParams
from dataclasses import dataclass, field
from typing import Dict, Any, List

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_3species")
os.makedirs(OUT_DIR, exist_ok=True)

# 3-species config: O2, H2O2, H+ (drop ClO4-)
THREE_SPECIES_Z = [0, 0, 1]
THREE_SPECIES_D = [D_O2_HAT, D_H2O2_HAT, D_HP_HAT]
THREE_SPECIES_A = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
THREE_SPECIES_C0 = [C_O2_HAT, C_H2O2_HAT, C_HP_HAT]

# ClO4- bulk concentration for Boltzmann background
C_CLO4_BULK_NONDIM = C_CLO4_HAT  # 0.2 nondim


def make_3sp_solver_params(eta_hat, k0_r1=K0_HAT_R1, k0_r2=K0_HAT_R2,
                           alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2):
    """Build 3-species SolverParams with BV reactions."""
    snes_opts = dict(SNES_OPTS_CHARGED)
    params = dict(snes_opts)
    params["bv_convergence"] = _make_bv_convergence_cfg()
    params["nondim"] = _make_nondim_cfg()

    # BV reactions (same as 4-species but stoichiometry has 3 entries)
    reaction_1 = {
        "k0": k0_r1, "alpha": alpha_r1,
        "cathodic_species": 0, "anodic_species": 1,
        "c_ref": 1.0,
        "stoichiometry": [-1, +1, -2],  # 3 species
        "n_electrons": N_ELECTRONS, "reversible": True,
        "E_eq_v": E_EQ_R1,
        "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    }
    reaction_2 = {
        "k0": k0_r2, "alpha": alpha_r2,
        "cathodic_species": 1, "anodic_species": None,
        "c_ref": 0.0,
        "stoichiometry": [0, -1, -2],  # 3 species
        "n_electrons": N_ELECTRONS, "reversible": False,
        "E_eq_v": E_EQ_R2,
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
        3,  # n_species
        1,  # order
        0.25,  # dt
        80.0,  # t_end
        THREE_SPECIES_Z,
        THREE_SPECIES_D,
        THREE_SPECIES_A,
        eta_hat,
        THREE_SPECIES_C0,
        0.0,  # phi0
        params,
    ])


def add_boltzmann_background(ctx, solver_params):
    """Add Boltzmann ClO4- background to Poisson source in F_res.

    Replaces the missing ClO4- transport equation with:
      c_ClO4(x) = c_ClO4_bulk * exp(-z_ClO4 * phi) = c_bulk * exp(phi)

    Added to Poisson source: -charge_rhs * z_ClO4 * c_ClO4 * w * dx
      = -charge_rhs * (-1) * c_bulk * exp(phi) * w * dx
      = +charge_rhs * c_bulk * exp(phi) * w * dx
    """
    scaling = ctx["nondim"]
    mesh = ctx["mesh"]
    W = ctx["W"]
    U = ctx["U"]
    n = ctx["n_species"]

    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh)

    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    c_clO4_bulk = fd.Constant(C_CLO4_BULK_NONDIM)

    # Boltzmann ClO4-: c = c_bulk * exp(-z*phi) = c_bulk * exp(phi) for z=-1
    # Clip phi to prevent exp overflow
    phi_clipped = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0))
    c_clO4_boltzmann = c_clO4_bulk * fd.exp(phi_clipped)

    # Add to Poisson source: z_ClO4 = -1
    # Standard Poisson: F += eps*grad(phi)·grad(w) - charge_rhs * sum(z_i*c_i) * w
    # The missing ClO4- term: -charge_rhs * (-1) * c_clO4_boltzmann * w
    #                        = +charge_rhs * c_clO4_boltzmann * w
    ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_clO4_boltzmann * w * dx

    # Update Jacobian
    ctx["J_form"] = fd.derivative(ctx["F_res"], U)
    ctx["boltzmann_background"] = {"c_clO4_bulk": C_CLO4_BULK_NONDIM}

    return ctx


def test_point(V_RHE, z_steps=20):
    phi_hat = V_RHE / V_T
    sp = make_3sp_solver_params(phi_hat)
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    with adj.stop_annotating():
        ctx = build_context(list(sp), mesh=mesh)
        ctx = build_forms(ctx, list(sp))
        ctx = add_boltzmann_background(ctx, list(sp))
        set_initial_conditions(ctx, list(sp))

    U = ctx["U"]; Up = ctx["U_prev"]
    zc = ctx["z_consts"]; n = ctx["n_species"]
    dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]

    sp_dict = {k: v for k, v in dict(SNES_OPTS_CHARGED).items()
               if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}
    prob = fd.NonlinearVariationalProblem(
        ctx["F_res"], U, bcs=ctx["bcs"], J=ctx["J_form"])
    sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

    of = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)

    dt_init = 0.25
    def run_ss(max_steps=60):
        dt_val = dt_init; dt_const.assign(dt_val)
        prev_flux = None; prev_delta = None; sc = 0
        for s in range(1, max_steps+1):
            try: sol.solve()
            except Exception as e: return False, s-1, str(e)
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
            if sc >= 4: return True, s, "converged"
        return False, max_steps, "budget"

    # z=0
    for zci in zc: zci.assign(0.0)
    paf.assign(phi_hat)

    print(f"\n--- V_RHE={V_RHE:.2f}V (phi_hat={phi_hat:.2f}) ---")
    with adj.stop_annotating():
        ok_z0, steps_z0, msg_z0 = run_ss(100)

    cd_z0 = float(fd.assemble(of))
    c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
    print(f"  z=0: ok={ok_z0}, cd={cd_z0:.6f}, c_min={[f'{c:.2e}' for c in c_min]}")

    # Physics validation at z=0
    c_bulk = [float(v) for v in sp[8]]
    z_nominal = [float(sp[4][i]) for i in range(n)]
    vr = validate_solution_state(
        U, n_species=n, c_bulk=c_bulk, phi_applied=phi_hat,
        z_vals=[0.0] * n, eps_c=ctx.get("_diag_eps_c", 1e-8),
        exponent_clip=ctx.get("_diag_exponent_clip", 50.0),
        species_names=["O2", "H2O2", "H+"],
    )
    if vr.failures:
        print(f"    [validation] FAILURES: {'; '.join(vr.failures)}")
    if vr.warnings:
        print(f"    [validation] warnings: {'; '.join(vr.warnings)}")

    # z-ramp
    achieved_z = 0.0

    with adj.stop_annotating():
        for z_val in np.linspace(0, 1, z_steps+1)[1:]:
            U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
            for i in range(n): zc[i].assign(z_nominal[i]*z_val)
            ok_z, steps_z, msg_z = run_ss(60)

            if ok_z or steps_z > 0:
                achieved_z = z_val
                cd_z = float(fd.assemble(of))
                c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
                if z_val in [0.05, 0.5, 0.75, 1.0] or not ok_z:
                    print(f"  z={z_val:.3f}: ok={ok_z}, cd={cd_z:.6f}, c_min={[f'{c:.2e}' for c in c_min]}")
                    # Physics validation at this z-step
                    vr = validate_solution_state(
                        U, n_species=n, c_bulk=c_bulk, phi_applied=phi_hat,
                        z_vals=[zi * z_val for zi in z_nominal],
                        eps_c=ctx.get("_diag_eps_c", 1e-8),
                        exponent_clip=ctx.get("_diag_exponent_clip", 50.0),
                        species_names=["O2", "H2O2", "H+"],
                    )
                    if vr.failures:
                        print(f"    [validation] FAILURES: {'; '.join(vr.failures)}")
                    if vr.warnings:
                        print(f"    [validation] warnings: {'; '.join(vr.warnings)}")
            else:
                for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                Up.assign(U)
                c_min = [float(U.dat[i].data_ro.min()) for i in range(n)]
                print(f"  z={z_val:.3f}: FAILED ({msg_z}), c_min={[f'{c:.2e}' for c in c_min]}")
                for z_f in np.linspace(achieved_z, z_val, 6)[1:]:
                    U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                    for i in range(n): zc[i].assign(z_nominal[i]*z_f)
                    ok_f, steps_f, _ = run_ss(60)
                    if ok_f or steps_f > 0: achieved_z = z_f; print(f"    z={z_f:.4f}: ok")
                    else:
                        for src, dst in zip(U_ckpt2, U.dat): dst.data[:] = src
                        Up.assign(U); print(f"    z={z_f:.4f}: FAILED"); break
                break

    cd_final = float(fd.assemble(of))
    print(f"  >> z={achieved_z:.4f}, cd={cd_final:.6f}")
    return achieved_z, cd_final


def main():
    print("V18: 3-SPECIES + BOLTZMANN BACKGROUND (no ClO4- transport)")
    print("="*60)

    # Standard 4-species reference values
    std_ref = {-0.30: -0.181333, -0.10: -0.177529, 0.0: -0.173297, 0.10: -0.162887}

    V_points = [-0.30, -0.10, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
    results = []

    for V in V_points:
        t0 = time.time()
        try:
            z, cd = test_point(V)
            results.append({"V": V, "z": z, "cd": cd, "time": time.time()-t0})
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"V": V, "z": 0, "cd": float("nan"), "error": str(e), "time": time.time()-t0})

    print("\n" + "="*70)
    print("3-SPECIES + BOLTZMANN CONVERGENCE MAP")
    print("="*70)
    print(f"{'V_RHE':>8} {'z':>6} {'cd':>12} {'time':>6} {'status':>10} {'vs 4sp std':>12}")
    print("-"*65)
    for r in results:
        V = r["V"]; z = r["z"]; cd = r["cd"]; t = r["time"]
        status = "FULL" if z >= 0.999 else f"z={z:.2f}"
        ref = std_ref.get(V)
        comp = f"err={abs(cd-ref)/abs(ref)*100:.1f}%" if (ref and z >= 0.999 and np.isfinite(cd)) else ""
        print(f"{V:8.2f} {z:6.3f} {cd:12.6f} {t:5.1f}s {status:>10} {comp:>12}")


if __name__ == "__main__":
    main()
