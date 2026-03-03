"""Context/form building and initial conditions for the BV PNP solver."""

from __future__ import annotations

from typing import Any

import numpy as np
import firedrake as fd

from Nondim.transform import build_model_scaling, _get_nondim_cfg, _as_list, _bool, _pos

from .config import _get_bv_cfg, _get_bv_convergence_cfg, _get_bv_reactions_cfg
from .nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform


# ---------------------------------------------------------------------------
# Public solver API
# ---------------------------------------------------------------------------

def build_context(solver_params: Any, *, mesh: Any = None) -> dict[str, Any]:
    """Build mesh and function spaces for the Butler-Volmer PNP solver.

    Parameters
    ----------
    solver_params : list or SolverParams
        11-element parameter list.
    mesh : firedrake.Mesh, optional
        Pre-built mesh.  If ``None`` (default), a ``UnitSquareMesh(32, 32)``
        is created for backward compatibility.  Pass a
        :func:`make_graded_interval_mesh` for 1-D EDL problems.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    if not (len(z_vals) == len(D_vals) == len(a_vals) == n_species):
        raise ValueError(
            f"z_vals, D_vals, and a_vals must all have length n_species ({n_species}); "
            f"got lengths {len(z_vals)}, {len(D_vals)}, {len(a_vals)}"
        )

    if mesh is None:
        mesh = fd.UnitSquareMesh(32, 32)
    V_scalar = fd.FunctionSpace(mesh, "CG", order)
    W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])
    U = fd.Function(W)
    U_prev = fd.Function(W)

    return {
        "mesh": mesh,
        "V_scalar": V_scalar,
        "W": W,
        "U": U,
        "U_prev": U_prev,
        "n_species": n_species,
    }


def build_forms(ctx: dict[str, Any], solver_params: Any) -> dict[str, Any]:
    """Assemble weak forms for the Butler-Volmer PNP problem.

    The BV boundary condition is applied at the electrode boundary::

        J_i · n  =  s_i · k̂₀_i · [max(ĉ_i, ε) · exp(−α_i · clip(η̂ − Ê_eq, ±50))
                                   − ĉ_ref_i     · exp((1−α_i) · clip(η̂ − Ê_eq, ±50))]

    The overpotential η̂ is taken from ``phi_applied_func`` (the Dirichlet BC value),
    not from the interior φ field — this keeps the Newton Jacobian well-conditioned.

    Parameters
    ----------
    ctx:
        Context dict from :func:`build_context`.
    solver_params:
        11-element SolverParams / list.  ``solver_params[10]`` must contain a
        ``"bv_bc"`` key (see module docstring).

    Returns
    -------
    dict
        Updated context including ``F_res``, ``J_form``, ``bcs``,
        ``bv_settings``, ``bv_scaling``, ``nondim``.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    c0_raw = [float(v) for v in (
        [c0] * n_species if np.isscalar(c0) else list(c0)
    )][:n_species]

    mesh = ctx["mesh"]
    W = ctx["W"]
    n = ctx["n_species"]

    # Parse BV config and convergence options.
    bv_cfg = _get_bv_cfg(params, n)
    conv_cfg = _get_bv_convergence_cfg(params)
    reactions_cfg = _get_bv_reactions_cfg(params, n)
    use_reactions = reactions_cfg is not None

    # Nondimensionalization (standard path, reusing robin transform).
    # Pass a dummy robin dict so the function doesn't crash; BV params are
    # handled separately via _add_bv_*_scaling_to_transform.
    dummy_robin = {
        "kappa_vals": [1.0] * n,
        "c_inf_vals": bv_cfg["c_ref_vals"],
        "electrode_marker": bv_cfg["electrode_marker"],
        "concentration_marker": bv_cfg["concentration_marker"],
        "ground_marker": bv_cfg["ground_marker"],
    }
    base_scaling = build_model_scaling(
        params=params,
        n_species=n,
        dt=dt,
        t_end=t_end,
        D_vals=D_vals,
        c0_vals=c0_raw,
        phi_applied=phi_applied,
        phi0=phi0,
        robin=dummy_robin,
    )

    nondim_cfg = _get_nondim_cfg(params)
    nondim_enabled = _bool(nondim_cfg.get("enabled", False))
    kappa_inputs_dimless = _bool(nondim_cfg.get("kappa_inputs_are_dimensionless", True))
    concentration_inputs_dimless = _bool(
        nondim_cfg.get("concentration_inputs_are_dimensionless", False)
    )

    if use_reactions:
        E_eq_v = float(bv_cfg.get("E_eq_v", 0.0))
        scaling = _add_bv_reactions_scaling_to_transform(
            base_scaling, reactions_cfg,
            nondim_enabled=nondim_enabled,
            kappa_inputs_dimless=kappa_inputs_dimless,
            concentration_inputs_dimless=concentration_inputs_dimless,
            E_eq_v=E_eq_v,
        )
    else:
        scaling = _add_bv_scaling_to_transform(
            base_scaling, bv_cfg,
            n_species=n,
            nondim_enabled=nondim_enabled,
            kappa_inputs_dimless=kappa_inputs_dimless,
            concentration_inputs_dimless=concentration_inputs_dimless,
        )

    electrode_marker = bv_cfg["electrode_marker"]
    concentration_marker = bv_cfg["concentration_marker"]
    ground_marker = bv_cfg["ground_marker"]

    dx = fd.Measure("dx", domain=mesh)
    ds = fd.Measure("ds", domain=mesh)
    R_space = fd.FunctionSpace(mesh, "R", 0)

    # Log-diffusivity controls (same as robin_solver).
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(scaling["D_model_vals"][i])))
    D = [fd.exp(m[i]) for i in range(n)]

    z = [fd.Constant(float(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    ci_prev = fd.split(U_prev)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    em = float(scaling["electromigration_prefactor"])
    dt_const = fd.Constant(float(scaling["dt_model"]))  # mutable — enables adaptive dt

    # Electrode potential constant (for BV exponent).
    phi_applied_func = fd.Function(R_space, name="phi_applied")
    phi_applied_func.assign(float(scaling["phi_applied_model"]))
    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(float(scaling["phi0_model"]))

    # E_eq offset in model space.
    E_eq_model = fd.Constant(float(scaling["bv_E_eq_model"]))
    bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))

    # Build the η̂ expression used in BV exponent.
    if conv_cfg["use_eta_in_bv"]:
        eta_hat_raw = phi_applied_func - E_eq_model
    else:
        eta_hat_raw = phi - E_eq_model

    eta_hat_scaled = bv_exp_scale * eta_hat_raw

    # Exponent clipping to prevent float overflow.
    if conv_cfg["clip_exponent"]:
        clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
        eta_clipped = fd.min_value(fd.max_value(eta_hat_scaled, -clip_val), clip_val)
    else:
        eta_clipped = eta_hat_scaled

    # Steric chemical potential (Bikerman model): mu_steric = ln(1 - sum_j a_j c_j)
    # Using fd.Function(R_space) instead of fd.Constant so steric `a` values
    # are adjoint-compatible (pyadjoint can differentiate through them).
    a_vals_list = [float(v) for v in a_vals]
    steric_active = any(v != 0.0 for v in a_vals_list)
    steric_a_funcs = [fd.Function(R_space, name=f"steric_a_{i}") for i in range(n)]
    for i in range(n):
        steric_a_funcs[i].assign(float(a_vals_list[i]))
    if steric_active:
        packing_floor = float(conv_cfg.get("packing_floor", 1e-8))
        packing = fd.max_value(
            fd.Constant(1.0) - sum(steric_a_funcs[j] * ci[j] for j in range(n)),
            fd.Constant(packing_floor),
        )
        mu_steric = fd.ln(packing)

    # Regularized surface concentrations (shared by both BV paths).
    # Two regularization options:
    #   1. max_value (default): c_surf = max(c_i, eps_c)
    #      Pros: simple, enforces a strict floor.
    #      Cons: non-smooth (kink at c_i = eps_c), can cause Jacobian issues.
    #   2. softplus: c_surf = eps_c * ln(1 + exp(c_i / eps_c))
    #      Pros: C-infinity smooth, better Jacobian conditioning.
    #      Cons: slightly overestimates small concentrations.
    #      For c_i >> eps_c: softplus ~ c_i.  For c_i << 0: softplus ~ eps_c * exp(c_i/eps_c) ~ 0.
    #      At c_i = 0: softplus = eps_c * ln(2) ~ 0.693 * eps_c.
    use_softplus = conv_cfg.get("softplus_regularization", False)
    if conv_cfg["regularize_concentration"]:
        eps_c = fd.Constant(float(conv_cfg["conc_floor"]))
        if use_softplus:
            # Softplus regularization: smooth approximation to max(c_i, eps_c).
            # Avoids the non-differentiable kink of max_value, improving
            # Newton convergence for the Jacobian near c_i ~ 0.
            c_surf = [eps_c * fd.ln(fd.Constant(1.0) + fd.exp(ci[i] / eps_c))
                      for i in range(n)]
        else:
            c_surf = [fd.max_value(ci[i], eps_c) for i in range(n)]
    else:
        c_surf = list(ci)

    # Residual: Nernst-Planck time-stepping + BV electrode BC.
    F_res = 0

    for i in range(n):
        c_i = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]

        # Electromigration drift (zero for neutral species, z=0).
        drift = em * z[i] * phi
        if steric_active:
            Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift) + c_i * fd.grad(mu_steric))
        else:
            Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift))

        # Time-stepping residual.
        F_res += ((c_i - c_old) / dt_const) * v * dx
        F_res += fd.dot(Jflux, fd.grad(v)) * dx

    # Butler-Volmer boundary flux.
    bv_k0_funcs = []
    bv_alpha_funcs = []
    if use_reactions:
        # ---- Multi-reaction path ----
        # Each reaction j has its own rate R_j; stoichiometry matrix couples
        # reactions to species.
        bv_rate_exprs = []
        rxns_scaled = scaling["bv_reactions"]
        for j, rxn in enumerate(rxns_scaled):
            k0_j = fd.Function(R_space, name=f"bv_k0_rxn{j}")
            k0_j.assign(float(rxn["k0_model"]))
            bv_k0_funcs.append(k0_j)
            alpha_j = fd.Function(R_space, name=f"bv_alpha_rxn{j}")
            alpha_j.assign(float(rxn["alpha"]))
            bv_alpha_funcs.append(alpha_j)
            cat_idx = rxn["cathodic_species"]

            # Cathodic term: k0 * c_cat * exp(-alpha * eta)
            cathodic = k0_j * c_surf[cat_idx] * fd.exp(-alpha_j * eta_clipped)

            # Apply optional cathodic concentration factors: e.g. (c_H+/c_ref)^power
            for factor in rxn.get("cathodic_conc_factors", []):
                sp_idx = factor["species"]
                power = factor["power"]
                c_ref_f = fd.Constant(float(factor["c_ref_nondim"]))
                cathodic = cathodic * (c_surf[sp_idx] / c_ref_f) ** power

            # Anodic term (if reversible)
            if rxn["reversible"] and rxn["anodic_species"] is not None:
                c_ref_j = fd.Constant(float(rxn["c_ref_model"]))
                anodic = k0_j * c_ref_j * fd.exp(
                    (fd.Constant(1.0) - alpha_j) * eta_clipped
                )
            else:
                anodic = fd.Constant(0.0)

            R_j = cathodic - anodic
            bv_rate_exprs.append(R_j)

            # Apply stoichiometry: F_res -= s_ij * R_j * v_i * ds(electrode)
            stoi = rxn["stoichiometry"]
            for i in range(n):
                if stoi[i] != 0:
                    F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)

    else:
        # ---- Legacy per-species path ----
        bv_rate_exprs = []
        for i in range(n):
            alpha_i = fd.Function(R_space, name=f"bv_alpha_sp{i}")
            alpha_i.assign(float(scaling["bv_alpha_vals"][i]))
            bv_alpha_funcs.append(alpha_i)
            stoi_i = int(scaling["bv_stoichiometry"][i])
            k0_i = fd.Function(R_space, name=f"bv_k0_sp{i}")
            k0_i.assign(float(scaling["bv_k0_model_vals"][i]))
            bv_k0_funcs.append(k0_i)
            c_ref_i = fd.Constant(float(scaling["bv_c_ref_model_vals"][i]))

            bv_flux_i = k0_i * (
                c_surf[i] * fd.exp(-alpha_i * eta_clipped)
                - c_ref_i * fd.exp((fd.Constant(1.0) - alpha_i) * eta_clipped)
            )
            bv_rate_exprs.append(bv_flux_i)
            # Weak-form sign: F_res -= stoi_i × bv_flux_i × v × ds
            F_res -= fd.Constant(float(stoi_i)) * bv_flux_i * v_list[i] * ds(electrode_marker)

    # Poisson equation.
    suppress_poisson_source = _bool(nondim_cfg.get("suppress_poisson_source", False))
    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
    if not suppress_poisson_source:
        F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx

    # Dirichlet BCs.
    bc_phi_electrode = fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    bc_ci = [
        fd.DirichletBC(
            W.sub(i),
            fd.Constant(float(scaling["c0_model_vals"][i])),
            concentration_marker,
        )
        for i in range(n)
    ]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]

    J_form = fd.derivative(F_res, U)

    ctx.update({
        "F_res": F_res,
        "J_form": J_form,
        "bcs": bcs,
        "logD_funcs": m,
        "D_consts": D,
        "z_consts": z,
        "dt_const": dt_const,
        "phi_applied_func": phi_applied_func,
        "phi0_func": phi0_func,
        "bv_settings": bv_cfg,
        "bv_convergence": conv_cfg,
        "bv_rate_exprs": bv_rate_exprs,
        "bv_k0_funcs": bv_k0_funcs,
        "bv_alpha_funcs": bv_alpha_funcs,
        "steric_a_funcs": steric_a_funcs,
        "nondim": scaling,
    })
    return ctx


def set_initial_conditions(ctx: dict[str, Any], solver_params: Any, blob: bool = False) -> None:
    """Set initial conditions for the BV solver.

    Default ``blob=False``: uniform concentrations + linear potential profile.
    For BV problems, starting near equilibrium (η ≈ 0) significantly helps
    Newton convergence at the first timestep.

    Parameters
    ----------
    blob:
        If True, superimpose a Gaussian concentration perturbation.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    scaling = ctx.get("nondim", {})

    c0_raw = [float(v) for v in ([c0] * n if np.isscalar(c0) else list(c0))][:n]
    c0_model = scaling.get("c0_model_vals", c0_raw)
    phi_applied_model = scaling.get("phi_applied_model", float(phi_applied))

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()

    if blob and ndim >= 2:
        x, y = coords[0], coords[1]
        A = fd.Constant(0.1)
        x0 = fd.Constant(0.5)
        y0 = fd.Constant(0.2)
        sigma = fd.Constant(0.08)
        gaussian_blob = A * fd.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        for i in range(n):
            U_prev.sub(i).interpolate(fd.Constant(float(c0_model[i])) + gaussian_blob)
    else:
        for i in range(n):
            U_prev.sub(i).assign(fd.Constant(float(c0_model[i])))

    # Linear potential profile: phi_applied at electrode, 0 at bulk.
    # 1D: electrode at x=0, bulk at x=1 → phi = phi_applied * (1 - x)
    # 2D: electrode at y=0, bulk at y=1 → phi = phi_applied * (1 - y)
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]
    U_prev.sub(n).interpolate(fd.Constant(float(phi_applied_model)) * (1.0 - spatial_var))
    ctx["U"].assign(U_prev)
