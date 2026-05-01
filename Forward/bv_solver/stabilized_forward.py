"""Stabilized forward solver: extends z=1 convergence to onset voltages.

Adds artificial diffusion to the co-ion (ClO4-, species 3) to prevent
negativity in the underresolved Debye layer. This is the ONLY species
that goes negative; O2 and H2O2 are uncharged, and H+ accumulates.

The stabilization adds:
  D_art = d_art_scale * h * |z_i * D_i * em| * |∇φ| * ∇c_i · ∇v_i * dx

to the weak form for the specified species. This is a streamline-type
diffusion that prevents oscillations in the drift-dominated regime.

Accuracy: 2.2% vs standard solver at V_RHE=0.10V (the overlap point).
The stabilization is minimal in the bulk (small ∇φ) and only significant
in the thin Debye layer where the co-ion concentrations would otherwise
go negative.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def solve_stabilized_curve(
    solver_params: Any,
    phi_applied_values: np.ndarray,
    observable_scale: float,
    *,
    d_art_scale: float = 0.001,
    stabilized_species: Optional[List[int]] = None,
    z_steps: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve I-V curve with stabilized PNP at full z=1.

    Parameters
    ----------
    solver_params : SolverParams or list
        11-element solver params with physical E_eq.
    phi_applied_values : array
        Dimensionless voltages (phi_hat = V_RHE / V_T).
    observable_scale : float
        Scaling for observables (typically -I_SCALE).
    d_art_scale : float
        Artificial diffusion strength. Default 0.001 (2.2% error at overlap).
    stabilized_species : list of int, optional
        Which species to stabilize. Default: [3] (ClO4- only).
    z_steps : int
        Number of z-ramp steps from 0 to 1.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys: V_RHE, cd, pc, z_achieved, converged_mask
    """
    import time
    import firedrake as fd
    import pyadjoint as adj
    from . import make_graded_rectangle_mesh
    from .forms import build_context, build_forms, set_initial_conditions
    from .observables import _build_bv_observable_form

    if stabilized_species is None:
        stabilized_species = [3]  # ClO4- only

    n_pts = len(phi_applied_values)
    cd_out = np.full(n_pts, np.nan)
    pc_out = np.full(n_pts, np.nan)
    z_out = np.full(n_pts, 0.0)
    U_data_list = [None] * n_pts

    t_total = time.time()

    for pt_idx in range(n_pts):
        phi_hat = float(phi_applied_values[pt_idx])
        V_RHE = phi_hat * 0.02569  # approximate

        sp_list = list(solver_params)
        sp_list[7] = phi_hat  # set phi_applied
        n = sp_list[0]

        mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

        with adj.stop_annotating():
            ctx = build_context(sp_list, mesh=mesh)
            ctx = build_forms(ctx, sp_list)
            set_initial_conditions(ctx, sp_list)

        U = ctx["U"]; Up = ctx["U_prev"]
        zc = ctx["z_consts"]
        dt_const = ctx["dt_const"]; paf = ctx["phi_applied_func"]
        W = ctx["W"]

        dx = fd.Measure("dx", domain=mesh)
        ci = fd.split(U)[:-1]
        phi = fd.split(U)[-1]
        v_tests = fd.TestFunctions(W)

        F_res = ctx["F_res"]
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        z_vals = sp_list[4]
        D_vals = scaling["D_model_vals"]

        # Add artificial diffusion for specified species
        h = fd.CellSize(mesh)
        for i in stabilized_species:
            z_i = float(z_vals[i])
            if abs(z_i) > 0:
                D_i = float(D_vals[i])
                drift_speed = fd.Constant(abs(z_i) * D_i * em)
                D_art = fd.Constant(d_art_scale) * h * drift_speed * fd.sqrt(
                    fd.dot(fd.grad(phi), fd.grad(phi)) + fd.Constant(1e-10))
                F_res += D_art * fd.dot(fd.grad(ci[i]), fd.grad(v_tests[i])) * dx

        J_stab = fd.derivative(F_res, U)
        params = sp_list[10]
        sp_dict = {k: v for k, v in params.items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))} if isinstance(params, dict) else {}
        prob = fd.NonlinearVariationalProblem(F_res, U, bcs=ctx["bcs"], J=J_stab)
        sol = fd.NonlinearVariationalSolver(prob, solver_parameters=sp_dict)

        ocd = _build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=observable_scale)
        opc = _build_bv_observable_form(ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)

        dt_init = float(scaling["dt_model"])

        def run_ss(max_steps=60):
            dt_val = dt_init; dt_const.assign(dt_val)
            prev_flux = None; prev_delta = None; steady_count = 0
            for s in range(1, max_steps+1):
                try:
                    sol.solve()
                except:
                    return False, s-1
                Up.assign(U)
                fv = float(fd.assemble(ocd))
                if prev_flux is not None:
                    delta = abs(fv-prev_flux); sc = max(abs(fv),abs(prev_flux),1e-8)
                    if delta/sc <= 1e-4 or delta <= 1e-8: steady_count += 1
                    else: steady_count = 0
                    if prev_delta and delta > 0:
                        r = prev_delta/delta
                        dt_val = min(dt_val*min(r,4),dt_init*20) if r>1 else max(dt_val*0.5,dt_init)
                        dt_const.assign(dt_val)
                    prev_delta = delta
                prev_flux = fv
                if steady_count >= 4: return True, s
            return False, max_steps

        # z=0
        for zci in zc: zci.assign(0.0)
        paf.assign(phi_hat)
        with adj.stop_annotating():
            run_ss(100)

        # z-ramp
        z_nominal = [float(z_vals[i]) for i in range(n)]
        achieved_z = 0.0
        with adj.stop_annotating():
            for z_val in np.linspace(0, 1, z_steps+1)[1:]:
                U_ckpt = tuple(d.data_ro.copy() for d in U.dat)
                for i in range(n): zc[i].assign(z_nominal[i]*z_val)
                ok, steps = run_ss(60)
                if ok or steps > 0:
                    achieved_z = z_val
                else:
                    for src, dst in zip(U_ckpt, U.dat): dst.data[:] = src
                    Up.assign(U)
                    for z_f in np.linspace(achieved_z, z_val, 6)[1:]:
                        U_ckpt2 = tuple(d.data_ro.copy() for d in U.dat)
                        for i in range(n): zc[i].assign(z_nominal[i]*z_f)
                        ok_f, steps_f = run_ss(60)
                        if ok_f or steps_f > 0: achieved_z = z_f
                        else:
                            for src, dst in zip(U_ckpt2, U.dat): dst.data[:] = src
                            Up.assign(U); break
                    break

        cd_out[pt_idx] = float(fd.assemble(ocd))
        pc_out[pt_idx] = float(fd.assemble(opc))
        z_out[pt_idx] = achieved_z
        U_data_list[pt_idx] = tuple(d.data_ro.copy() for d in U.dat)

        if verbose and (pt_idx % 5 == 0 or achieved_z < 0.999):
            status = "OK" if achieved_z >= 0.999 else f"z={achieved_z:.2f}"
            print(f"  [{pt_idx+1}/{n_pts}] phi={phi_hat:+7.2f}: "
                  f"cd={cd_out[pt_idx]:.6f}, z={achieved_z:.3f} [{status}]")

    elapsed = time.time() - t_total
    n_full = int(np.sum(z_out >= 0.999))
    if verbose:
        print(f"\nStabilized solve: {n_full}/{n_pts} full z=1 in {elapsed:.1f}s")

    return {
        "phi_applied": phi_applied_values,
        "cd": cd_out,
        "pc": pc_out,
        "z_achieved": z_out,
        "U_data_list": U_data_list,
        "n_converged": n_full,
        "n_total": n_pts,
        "elapsed": elapsed,
    }
