"""Log-concentration transform forms for the BV PNP solver.

Replaces the standard concentration variables c_i with u_i = ln(c_i).
Physical concentrations are recovered as c_i = exp(u_i), which is
guaranteed positive for any finite u_i.

This transform addresses the EDL depletion zone convergence failure:
- c_i ranges from ~0.018 to 1.0 → u_i ranges from -4.0 to 0.0 (linear)
- No near-zero concentrations in the Jacobian
- The entropy barrier ln(c) → -∞ as c → 0 prevents depletion to zero
- Well-proven in literature (Metti, Xu, Liu 2016)

The interface mirrors forms.py: build_context_log, build_forms_log,
set_initial_conditions_log produce a ctx dict compatible with the
existing solver and observable infrastructure.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import firedrake as fd

from Nondim.transform import build_model_scaling, _get_nondim_cfg, _as_list, _bool, _pos
from .config import _get_bv_cfg, _get_bv_convergence_cfg, _get_bv_reactions_cfg
from .nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context_log(solver_params: Any, *, mesh: Any = None) -> dict[str, Any]:
    """Build mesh and function spaces for the log-transformed BV PNP solver.

    The mixed function space has n_species + 1 components:
    [u_0, u_1, ..., u_{n-1}, phi] where u_i = ln(c_i).

    This is identical to build_context in forms.py — the function space
    structure is the same (CG elements for each component).
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
        "log_transform": True,  # Flag so downstream code knows
    }


def build_forms_log(ctx: dict[str, Any], solver_params: Any) -> dict[str, Any]:
    """Assemble weak forms for the LOG-TRANSFORMED BV PNP problem.

    Primary unknowns: u_i = ln(c_i) for species, phi for potential.
    Physical concentrations: c_i = exp(u_i).

    The Nernst-Planck equation in log variables:

        exp(u_i) * du_i/dt = div(D_i * exp(u_i) * (grad(u_i) + z_i * em * grad(phi)))

    Weak form (test function v):

        exp(u_i) * (u_i - u_i_prev)/dt * v * dx
        + D_i * exp(u_i) * (grad(u_i) + z_i * em * grad(phi)) . grad(v) * dx
        - BV_flux * v * ds(electrode)
        = 0

    The key difference from standard forms: the NP flux uses
    D_i * exp(u_i) * grad(u_i) instead of D_i * grad(c_i).
    The drift term D_i * c_i * grad(z*em*phi) becomes
    D_i * exp(u_i) * z_i * em * grad(phi) — same form but with exp(u_i).
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

    # Parse BV config and convergence options (same as forms.py)
    bv_cfg = _get_bv_cfg(params, n)
    conv_cfg = _get_bv_convergence_cfg(params)
    reactions_cfg = _get_bv_reactions_cfg(params, n)
    use_reactions = reactions_cfg is not None

    dummy_robin = {
        "kappa_vals": [1.0] * n,
        "c_inf_vals": bv_cfg["c_ref_vals"],
        "electrode_marker": bv_cfg["electrode_marker"],
        "concentration_marker": bv_cfg["concentration_marker"],
        "ground_marker": bv_cfg["ground_marker"],
    }
    base_scaling = build_model_scaling(
        params=params, n_species=n, dt=dt, t_end=t_end,
        D_vals=D_vals, c0_vals=c0_raw,
        phi_applied=phi_applied, phi0=phi0, robin=dummy_robin,
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
        stern_raw = bv_cfg.get("stern_capacitance_f_m2")
        if stern_raw is not None and float(stern_raw) > 0:
            if not nondim_enabled:
                scaling["bv_stern_capacitance_model"] = float(stern_raw)
            else:
                from Nondim.constants import FARADAY_CONSTANT as _F
                length_scale = scaling.get("length_scale_m", 1.0)
                potential_scale_v = scaling.get("potential_scale_v", 1.0)
                concentration_scale = scaling.get("concentration_scale_mol_m3", 1.0)
                scaling["bv_stern_capacitance_model"] = (
                    float(stern_raw) * potential_scale_v
                    / (_F * concentration_scale * length_scale)
                )
        else:
            scaling.setdefault("bv_stern_capacitance_model", None)
    else:
        scaling = _add_bv_scaling_to_transform(
            base_scaling, bv_cfg, n_species=n,
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

    # Log-diffusivity controls (same as forms.py — for adjoint compatibility)
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(scaling["D_model_vals"][i])))
    D = [fd.exp(m[i]) for i in range(n)]

    z = [fd.Constant(float(z_vals[i])) for i in range(n)]

    # --- KEY DIFFERENCE: u_i = ln(c_i), c_i = exp(u_i) ---
    U = ctx["U"]
    U_prev = ctx["U_prev"]
    ui = fd.split(U)[:-1]       # u_0, ..., u_{n-1} (log-concentrations)
    phi = fd.split(U)[-1]       # electrostatic potential
    ui_prev = fd.split(U_prev)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    # Physical concentrations from log variables
    ci = [fd.exp(ui[i]) for i in range(n)]       # c_i = exp(u_i)
    ci_prev = [fd.exp(ui_prev[i]) for i in range(n)]

    em = float(scaling["electromigration_prefactor"])
    dt_const = fd.Constant(float(scaling["dt_model"]))

    phi_applied_func = fd.Function(R_space, name="phi_applied")
    phi_applied_func.assign(float(scaling["phi_applied_model"]))
    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(float(scaling["phi0_model"]))

    E_eq_model_global = fd.Constant(float(scaling["bv_E_eq_model"]))
    bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))

    stern_capacitance_model = scaling.get("bv_stern_capacitance_model")
    use_stern = stern_capacitance_model is not None and float(stern_capacitance_model) > 0

    def _build_eta_clipped(E_eq_const):
        if use_stern:
            eta_raw = phi_applied_func - phi - E_eq_const
        elif conv_cfg["use_eta_in_bv"]:
            eta_raw = phi_applied_func - E_eq_const
        else:
            eta_raw = phi - E_eq_const
        eta_scaled = bv_exp_scale * eta_raw
        if conv_cfg["clip_exponent"]:
            clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
            return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)
        return eta_scaled

    eta_clipped = _build_eta_clipped(E_eq_model_global)

    # Steric chemical potential (Bikerman): mu_steric = ln(1 - sum a_j * c_j)
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

    # --- Surface concentrations for BV ---
    # With log-transform, c_i = exp(u_i) is always positive.
    # We still apply a floor for numerical safety in BV terms.
    eps_c = fd.Constant(float(conv_cfg.get("conc_floor", 1e-12)))
    c_surf = [fd.max_value(ci[i], eps_c) for i in range(n)]

    # ===================================================================
    # RESIDUAL: Log-transformed Nernst-Planck + Poisson + BV
    # ===================================================================
    F_res = 0

    for i in range(n):
        u_i = ui[i]
        u_old = ui_prev[i]
        c_i = ci[i]          # exp(u_i)
        c_old = ci_prev[i]   # exp(u_i_prev)
        v = v_list[i]

        # Drift term (electromigration)
        drift = em * z[i] * phi

        # NP flux in log variables:
        # J_i = D_i * c_i * (grad(u_i) + z_i * em * grad(phi) + grad(mu_steric))
        #      = D_i * exp(u_i) * (grad(u_i) + grad(drift) + grad(mu_steric))
        if steric_active:
            Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift) + fd.grad(mu_steric))
        else:
            Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift))

        # Time-stepping residual.
        # d(c_i)/dt = d(exp(u_i))/dt = exp(u_i) * du_i/dt
        # Linearized: exp(u_i) * (u_i - u_i_prev) / dt
        # This is equivalent to (c_i - c_old)/dt when u changes are small,
        # but is more numerically stable because u varies linearly.
        F_res += (c_i * (u_i - u_old) / dt_const) * v * dx
        F_res += fd.dot(Jflux, fd.grad(v)) * dx

    # Butler-Volmer boundary flux — same as forms.py but using c_surf = exp(u)
    bv_k0_funcs = []
    bv_alpha_funcs = []
    if use_reactions:
        bv_rate_exprs = []
        rxns_scaled = scaling["bv_reactions"]
        for j, rxn in enumerate(rxns_scaled):
            k0_j = fd.Function(R_space, name=f"bv_k0_rxn{j}")
            k0_j.assign(float(rxn["k0_model"]))
            bv_k0_funcs.append(k0_j)
            alpha_j = fd.Function(R_space, name=f"bv_alpha_rxn{j}")
            alpha_j.assign(float(rxn["alpha"]))
            bv_alpha_funcs.append(alpha_j)
            n_e_j = fd.Constant(float(rxn["n_electrons"]))
            cat_idx = rxn["cathodic_species"]

            E_eq_j_val = rxn.get("E_eq_model", None)
            if E_eq_j_val is not None and E_eq_j_val != 0.0:
                E_eq_j = fd.Constant(float(E_eq_j_val))
                eta_j = _build_eta_clipped(E_eq_j)
            else:
                eta_j = eta_clipped

            cathodic = k0_j * c_surf[cat_idx] * fd.exp(-alpha_j * n_e_j * eta_j)
            for factor in rxn.get("cathodic_conc_factors", []):
                sp_idx = factor["species"]
                power = factor["power"]
                c_ref_f = fd.Constant(max(float(factor["c_ref_nondim"]), 1e-12))
                cathodic = cathodic * (c_surf[sp_idx] / c_ref_f) ** power

            if rxn["reversible"] and rxn["anodic_species"] is not None:
                anod_idx = rxn["anodic_species"]
                anodic = k0_j * c_surf[anod_idx] * fd.exp(
                    (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j)
            elif rxn["reversible"]:
                c_ref_j = fd.Constant(float(rxn["c_ref_model"]))
                anodic = k0_j * c_ref_j * fd.exp(
                    (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j)
            else:
                anodic = fd.Constant(0.0)

            R_j = cathodic - anodic
            bv_rate_exprs.append(R_j)

            stoi = rxn["stoichiometry"]
            for i in range(n):
                if stoi[i] != 0:
                    F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)
    else:
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
            F_res -= fd.Constant(float(stoi_i)) * bv_flux_i * v_list[i] * ds(electrode_marker)

    # Poisson equation — uses c_i = exp(u_i) for charge source
    suppress_poisson_source = _bool(nondim_cfg.get("suppress_poisson_source", False))
    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
    if not suppress_poisson_source:
        F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx

    # Stern layer Robin BC (same as forms.py)
    if use_stern:
        stern_coeff = fd.Constant(float(stern_capacitance_model))
        F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)

    # Dirichlet BCs — concentration BCs are in LOG SPACE: u_i = ln(c_i_bulk)
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    _LOG_BC_FLOOR = 1e-10  # BC floor can be tighter than IC floor
    bc_ui = []
    for i in range(n):
        c_bulk = float(scaling["c0_model_vals"][i])
        if c_bulk < _LOG_BC_FLOOR:
            c_bulk = _LOG_BC_FLOOR
        u_bulk = np.log(c_bulk)
        bc_ui.append(
            fd.DirichletBC(W.sub(i), fd.Constant(u_bulk), concentration_marker)
        )

    if use_stern:
        bcs = bc_ui + [bc_phi_ground]
    else:
        bc_phi_electrode = fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)
        bcs = bc_ui + [bc_phi_electrode, bc_phi_ground]

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
        "use_stern": use_stern,
        "log_transform": True,
        # Store ci (exp(u_i)) expressions for observable extraction
        "ci_exprs": ci,
    })
    return ctx


def set_initial_conditions_log(
    ctx: dict[str, Any], solver_params: Any, blob: bool = False,
) -> None:
    """Set initial conditions in LOG SPACE.

    u_i = ln(c_i_bulk) for each species, phi = linear profile.
    For species with zero bulk concentration, uses a small floor.
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

    # Set u_i = ln(c_i_bulk), with floor for zero concentrations.
    # IMPORTANT: Use a generous floor (1e-4) rather than machine-epsilon.
    # Too-small concentrations make the exp(u)-weighted mass matrix nearly
    # zero, destroying pseudo-transient regularization.
    _LOG_IC_FLOOR = 1e-4  # ~0.02% of typical bulk concentration
    for i in range(n):
        c_val = float(c0_model[i])
        if c_val < _LOG_IC_FLOOR:
            c_val = _LOG_IC_FLOOR
        u_val = np.log(c_val)
        U_prev.sub(i).assign(fd.Constant(u_val))

    # Linear potential profile (same as forms.py)
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]
    U_prev.sub(n).interpolate(fd.Constant(float(phi_applied_model)) * (1.0 - spatial_var))
    ctx["U"].assign(U_prev)
