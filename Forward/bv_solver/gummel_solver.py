"""Gummel (operator-split) solver for the BV-PNP system.

Instead of solving the fully coupled NP + Poisson system monolithically
(which creates an ill-conditioned Jacobian near the Debye layer), this
solver alternates between:

  Step A: Fix φ, solve NP for concentrations (advection-diffusion + BV BC)
  Step B: Fix c, solve Poisson for φ (linear)

Each subproblem is well-conditioned individually. The iteration converges
for moderate coupling and can be accelerated with Anderson mixing.

This approach is standard in semiconductor simulation (Gummel's method,
1964) and avoids the singular perturbation that kills the monolithic solver
at onset voltages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def solve_gummel_steady(
    solver_params: Any,
    *,
    mesh: Any = None,
    max_gummel_iter: int = 200,
    gummel_rtol: float = 1e-6,
    gummel_atol: float = 1e-8,
    omega: float = 0.5,
    np_substeps: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the BV-PNP system to steady state using Gummel iteration.

    Parameters
    ----------
    solver_params : list or SolverParams
        11-element solver parameter set.
    mesh : firedrake.Mesh, optional
        Pre-built mesh.
    max_gummel_iter : int
        Maximum Gummel iterations.
    gummel_rtol, gummel_atol : float
        Convergence tolerances on the L2 norm of the concentration update.
    omega : float
        Under-relaxation for Poisson update (0 < omega <= 1).
    np_substeps : int
        Time steps within each NP solve (sub-step to steady state).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys: U, ctx, converged, n_iter, cd, pc
    """
    import firedrake as fd
    import pyadjoint as adj

    from .forms import build_context, build_forms, set_initial_conditions
    from . import make_graded_rectangle_mesh

    if mesh is None:
        mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    sp_list = list(solver_params)
    n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = sp_list

    n = n_species
    z_nominal = [float(z_vals[i]) for i in range(n)]

    # Extract SNES options
    sp_dict = {}
    if isinstance(params, dict):
        sp_dict = {k: v for k, v in params.items()
                   if k.startswith(("snes_", "ksp_", "pc_", "mat_"))}

    # Build context with FULL z (not z=0!)
    with adj.stop_annotating():
        ctx = build_context(sp_list, mesh=mesh)
        ctx = build_forms(ctx, sp_list)
        set_initial_conditions(ctx, sp_list)

    U_full = ctx["U"]
    U_prev = ctx["U_prev"]
    W = ctx["W"]
    zc = ctx["z_consts"]
    paf = ctx["phi_applied_func"]
    dt_const = ctx["dt_const"]

    # We need separate function spaces for the split approach.
    # But Firedrake uses mixed function spaces, so we work within
    # the mixed system but freeze certain components.
    #
    # Approach: Use the monolithic form but with a TWO-STAGE solve:
    #   Stage A: zero out z_consts → NP becomes pure diffusion+reaction
    #            (decoupled from Poisson). Solve for a few timesteps.
    #   Stage B: freeze concentrations, solve Poisson directly.
    #
    # This is a pragmatic Gummel: not a true split (which would need
    # separate function spaces), but achieves the same effect of
    # alternating between NP and Poisson updates.

    # Actually, let's build truly separate problems.
    V_scalar = ctx["V_scalar"]

    # --- Concentration-only function space ---
    W_c = fd.MixedFunctionSpace([V_scalar for _ in range(n)])
    C = fd.Function(W_c, name="concentrations")
    C_prev = fd.Function(W_c, name="c_prev")

    # --- Potential-only function space ---
    PHI = fd.Function(V_scalar, name="phi")
    PHI_prev = fd.Function(V_scalar, name="phi_prev")

    # Extract scaling/config from ctx
    scaling = ctx["nondim"]
    bv_cfg = ctx["bv_settings"]
    conv_cfg = ctx["bv_convergence"]

    electrode_marker = bv_cfg["electrode_marker"]
    concentration_marker = bv_cfg["concentration_marker"]
    ground_marker = bv_cfg["ground_marker"]

    dx = fd.Measure("dx", domain=mesh)
    ds = fd.Measure("ds", domain=mesh)
    R_space = fd.FunctionSpace(mesh, "R", 0)

    # Log-diffusivity (same as forms.py)
    D = []
    for i in range(n):
        D_val = float(scaling["D_model_vals"][i])
        D.append(fd.Constant(D_val))

    em = float(scaling["electromigration_prefactor"])
    dt_c = fd.Constant(float(scaling["dt_model"]))

    phi_applied_val = fd.Function(R_space, name="phi_app")
    phi_applied_val.assign(float(scaling["phi_applied_model"]))

    E_eq_global = fd.Constant(float(scaling["bv_E_eq_model"]))
    bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))

    # --- Build NP residual (Step A): fixed PHI ---
    ci = fd.split(C)
    ci_prev = fd.split(C_prev)
    v_tests_c = fd.TestFunctions(W_c)

    eps_c = fd.Constant(float(conv_cfg["conc_floor"]))

    def _eta_clipped(E_eq_const):
        eta_raw = phi_applied_val - E_eq_const
        eta_scaled = bv_exp_scale * eta_raw
        clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
        return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)

    F_np = 0
    for i in range(n):
        c_i = ci[i]
        c_old = ci_prev[i]
        v = v_tests_c[i]
        z_i = fd.Constant(float(z_nominal[i]))

        # Drift from FIXED potential
        drift = em * z_i * PHI
        Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift))

        F_np += ((c_i - c_old) / dt_c) * v * dx
        F_np += fd.dot(Jflux, fd.grad(v)) * dx

    # BV boundary terms (same as forms.py reactions path)
    c_surf = [fd.max_value(ci[i], eps_c) for i in range(n)]

    bv_k0_funcs = []
    bv_alpha_funcs = []
    bv_rate_exprs = []

    from .config import _get_bv_reactions_cfg
    reactions_cfg = _get_bv_reactions_cfg(params, n)
    use_reactions = reactions_cfg is not None

    if use_reactions:
        rxns_scaled = scaling["bv_reactions"]
        for j, rxn in enumerate(rxns_scaled):
            k0_j = fd.Function(R_space, name=f"g_k0_{j}")
            k0_j.assign(float(rxn["k0_model"]))
            bv_k0_funcs.append(k0_j)
            alpha_j = fd.Function(R_space, name=f"g_alpha_{j}")
            alpha_j.assign(float(rxn["alpha"]))
            bv_alpha_funcs.append(alpha_j)
            n_e_j = fd.Constant(float(rxn["n_electrons"]))
            cat_idx = rxn["cathodic_species"]

            E_eq_j_val = rxn.get("E_eq_model", None)
            if E_eq_j_val is not None and E_eq_j_val != 0.0:
                eta_j = _eta_clipped(fd.Constant(float(E_eq_j_val)))
            else:
                eta_j = _eta_clipped(E_eq_global)

            cathodic = k0_j * c_surf[cat_idx] * fd.exp(-alpha_j * n_e_j * eta_j)
            for factor in rxn.get("cathodic_conc_factors", []):
                sp_idx = factor["species"]
                power = factor["power"]
                c_ref_f = fd.Constant(max(float(factor["c_ref_nondim"]), 1e-12))
                cathodic = cathodic * (c_surf[sp_idx] / c_ref_f) ** power

            if rxn["reversible"] and rxn["anodic_species"] is not None:
                anod_idx = rxn["anodic_species"]
                anodic = k0_j * c_surf[anod_idx] * fd.exp(
                    (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
                )
            elif rxn["reversible"]:
                c_ref_j = fd.Constant(float(rxn["c_ref_model"]))
                anodic = k0_j * c_ref_j * fd.exp(
                    (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
                )
            else:
                anodic = fd.Constant(0.0)

            R_j = cathodic - anodic
            bv_rate_exprs.append(R_j)

            stoi = rxn["stoichiometry"]
            for i in range(n):
                if stoi[i] != 0:
                    F_np -= fd.Constant(float(stoi[i])) * R_j * v_tests_c[i] * ds(electrode_marker)

    # NP BCs: concentration BCs at bulk
    c0_model = scaling.get("c0_model_vals", [1.0]*n)
    bc_np = [
        fd.DirichletBC(W_c.sub(i), fd.Constant(float(c0_model[i])), concentration_marker)
        for i in range(n)
    ]

    J_np = fd.derivative(F_np, C)
    np_problem = fd.NonlinearVariationalProblem(F_np, C, bcs=bc_np, J=J_np)

    np_sp = dict(sp_dict)
    np_sp["snes_max_it"] = 100
    np_sp["snes_linesearch_maxlambda"] = 0.5
    np_solver = fd.NonlinearVariationalSolver(np_problem, solver_parameters=np_sp)

    # --- Build Poisson residual (Step B): fixed concentrations ---
    phi_trial = fd.Function(V_scalar, name="phi_new")
    w_test = fd.TestFunction(V_scalar)

    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))

    F_poisson = eps_coeff * fd.dot(fd.grad(phi_trial), fd.grad(w_test)) * dx
    # Charge source uses FIXED concentrations from C
    for i in range(n):
        z_i_val = float(z_nominal[i])
        if z_i_val != 0.0:
            F_poisson -= charge_rhs * fd.Constant(z_i_val) * ci[i] * w_test * dx

    # Poisson BCs
    bc_poisson = [
        fd.DirichletBC(V_scalar, phi_applied_val, electrode_marker),
        fd.DirichletBC(V_scalar, fd.Constant(0.0), ground_marker),
    ]

    # This is actually linear in phi_trial, so we can use a linear solver
    # But we need to be careful — F_poisson uses phi_trial as the unknown
    # and ci from C (fixed). Since ci enters as coefficients (not unknowns),
    # the Jacobian w.r.t. phi_trial is just the Laplacian.
    J_poisson = fd.derivative(F_poisson, phi_trial)
    poisson_problem = fd.NonlinearVariationalProblem(
        F_poisson, phi_trial, bcs=bc_poisson, J=J_poisson
    )
    poisson_sp = {
        "snes_type": "ksponly",  # linear problem
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    poisson_solver = fd.NonlinearVariationalSolver(
        poisson_problem, solver_parameters=poisson_sp
    )

    # --- Initialize ---
    # Set initial concentrations and potential
    c0_vals = scaling.get("c0_model_vals", [1.0]*n)
    for i in range(n):
        C.sub(i).assign(fd.Constant(float(c0_vals[i])))
        C_prev.sub(i).assign(fd.Constant(float(c0_vals[i])))

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()
    spatial_var = coords[1] if ndim >= 2 else coords[0]
    phi_app_model = float(scaling["phi_applied_model"])
    PHI.interpolate(fd.Constant(phi_app_model) * (1.0 - spatial_var))
    PHI_prev.assign(PHI)

    # --- Observable forms ---
    # Build from bv_rate_exprs (which use ci from C)
    scale_const = fd.Constant(1.0)
    ocd_form = sum(R for R in bv_rate_exprs) * ds(electrode_marker) if bv_rate_exprs else fd.Constant(0.0) * ds(electrode_marker)
    if len(bv_rate_exprs) >= 2:
        opc_form = (bv_rate_exprs[0] - bv_rate_exprs[1]) * ds(electrode_marker)
    else:
        opc_form = ocd_form

    # --- Gummel Iteration ---
    converged = False
    with adj.stop_annotating():
        for gummel_k in range(max_gummel_iter):
            # Save current state for convergence check
            C_old_data = [C.dat[i].data_ro.copy() for i in range(n)]
            PHI_old_data = PHI.dat[0].data_ro.copy()

            # Step A: NP solve with fixed PHI
            C_prev.assign(C)
            dt_c.assign(float(scaling["dt_model"]))
            np_failed = False
            for sub in range(np_substeps):
                try:
                    np_solver.solve()
                except Exception as e:
                    if verbose:
                        print(f"  [Gummel {gummel_k}] NP solve failed at substep {sub}: {e}")
                    np_failed = True
                    break
                C_prev.assign(C)

            if np_failed:
                # Restore and try with smaller substeps
                for i in range(n):
                    C.dat[i].data[:] = C_old_data[i]
                C_prev.assign(C)
                break

            # Step B: Poisson solve with fixed C
            try:
                poisson_solver.solve()
            except Exception as e:
                if verbose:
                    print(f"  [Gummel {gummel_k}] Poisson solve failed: {e}")
                break

            # Under-relax PHI update
            PHI.dat[0].data[:] = (
                omega * phi_trial.dat[0].data_ro
                + (1.0 - omega) * PHI_old_data
            )

            # Convergence check
            c_diff = 0.0
            c_norm = 0.0
            for i in range(n):
                diff = C.dat[i].data_ro - C_old_data[i]
                c_diff += float(np.sum(diff**2))
                c_norm += float(np.sum(C.dat[i].data_ro**2))

            phi_diff = float(np.sum((PHI.dat[0].data_ro - PHI_old_data)**2))
            phi_norm = float(np.sum(PHI.dat[0].data_ro**2))

            rel_c = np.sqrt(c_diff / max(c_norm, 1e-30))
            rel_phi = np.sqrt(phi_diff / max(phi_norm, 1e-30))
            abs_change = np.sqrt(c_diff + phi_diff)

            if verbose and (gummel_k % 10 == 0 or gummel_k < 5):
                c_mins = [float(C.dat[i].data_ro.min()) for i in range(n)]
                cd_val = float(fd.assemble(ocd_form))
                print(f"  [Gummel {gummel_k:3d}] rel_c={rel_c:.2e}, rel_phi={rel_phi:.2e}, "
                      f"cd={cd_val:.6e}, c_min={[f'{c:.2e}' for c in c_mins]}")

            if rel_c < gummel_rtol and rel_phi < gummel_rtol:
                converged = True
                if verbose:
                    print(f"  [Gummel] Converged at iter {gummel_k} "
                          f"(rel_c={rel_c:.2e}, rel_phi={rel_phi:.2e})")
                break

            if abs_change < gummel_atol:
                converged = True
                if verbose:
                    print(f"  [Gummel] Converged (abs) at iter {gummel_k}")
                break

    # Extract observables
    from scripts._bv_common import I_SCALE
    cd_val = float(fd.assemble(ocd_form)) * (-I_SCALE)
    pc_val = float(fd.assemble(opc_form)) * (-I_SCALE)
    c_mins = [float(C.dat[i].data_ro.min()) for i in range(n)]

    return {
        "C": C, "PHI": PHI, "ctx": ctx, "mesh": mesh,
        "converged": converged, "n_iter": gummel_k + 1,
        "cd": cd_val, "pc": pc_val, "c_min": c_mins,
    }
