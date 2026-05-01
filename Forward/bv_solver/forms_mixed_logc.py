"""Mixed log-concentration transform: log(c) ONLY for ClO4- (species 3).

The standard CG1 solver fails because ClO4- goes negative in the Debye
layer despite the mesh being fine enough (0.01nm elements vs 30nm Debye
length). This is a positivity violation inherent to CG1 approximation
of exponentially-depleting profiles.

Fix: use u_3 = ln(c_3) as the unknown for ClO4- only. This:
  1. Guarantees c_3 = exp(u_3) > 0 always
  2. Linearizes the exponential Debye-layer profile (better CG1 approx)
  3. Doesn't affect O2/H2O2 (uncharged) or H+ (accumulates, no issue)
  4. No artificial diffusion or fake physics

The mixed function space:
  Sub 0: c_O2     (standard)
  Sub 1: c_H2O2   (standard)
  Sub 2: c_H+     (standard)
  Sub 3: u_ClO4   = ln(c_ClO4)  (log-transformed)
  Sub 4: phi      (standard)

NP equation for ClO4- in log-space:
  exp(u) * du/dt + ∇·(-D*exp(u)*(∇u + z*∇φ)) = 0

Poisson source uses c_ClO4 = exp(u_3).

This module provides build_forms_mixed_logc as a drop-in replacement for
build_forms. build_context and set_initial_conditions are wrapped to
handle the log-transform bookkeeping.
"""
from __future__ import annotations

from typing import Any, List, Optional, Set

import numpy as np
import firedrake as fd

from .forms import build_context as _std_build_context
from .forms import build_forms as _std_build_forms


# Which species to log-transform (indices). Default: only ClO4- (index 3).
_DEFAULT_LOG_SPECIES: Set[int] = {3}

# Floor for ln(0) avoidance in initial conditions and BCs
_C_FLOOR = 1e-20

# Clamp for exp(u) to prevent overflow
_U_CLAMP = 50.0


def build_context_mixed_logc(
    solver_params: Any,
    *,
    mesh: Any = None,
    log_species: Optional[Set[int]] = None,
) -> dict[str, Any]:
    """Build context — identical to standard, plus log-species metadata."""
    ctx = _std_build_context(solver_params, mesh=mesh)
    ctx["log_species"] = log_species if log_species is not None else _DEFAULT_LOG_SPECIES
    return ctx


def build_forms_mixed_logc(ctx: dict[str, Any], solver_params: Any) -> dict[str, Any]:
    """Build weak forms with selective log-concentration transform.

    For species in ctx["log_species"], the unknown is u_i = ln(c_i).
    For all other species, the unknown is c_i directly.

    The weak form is modified only for log-transformed species:
      Standard:  (c - c_old)/dt * v + D*(∇c + z*c*∇φ)·∇v = BV
      Log-space: (exp(u) - exp(u_old))/dt * v + D*exp(u)*(∇u + z*∇φ)·∇v = BV
    """
    # First build the STANDARD forms (gets all the config, scaling, BV terms)
    ctx = _std_build_forms(ctx, solver_params)

    log_species = ctx.get("log_species", _DEFAULT_LOG_SPECIES)
    if not log_species:
        return ctx  # Nothing to transform

    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception:
        sp = list(solver_params)
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = sp

    n = ctx["n_species"]
    mesh = ctx["mesh"]
    W = ctx["W"]
    U = ctx["U"]
    U_prev = ctx["U_prev"]
    scaling = ctx["nondim"]

    dx = fd.Measure("dx", domain=mesh)
    ds = fd.Measure("ds", domain=mesh)

    # We need to REBUILD F_res for the log-transformed species.
    # The standard F_res used c_i directly. For log-species, we replace
    # the transport terms with the log-space equivalents.
    #
    # Strategy: rebuild F_res from scratch, using the standard forms
    # for non-log species and log-space forms for log species.

    ui_all = fd.split(U)       # first n are species, last is phi
    ui_prev_all = fd.split(U_prev)
    phi = ui_all[-1]
    v_tests = fd.TestFunctions(W)

    em = float(scaling["electromigration_prefactor"])
    dt_const = ctx["dt_const"]

    # Get diffusivities from ctx (these are the exp(logD) expressions)
    D = ctx["D_consts"]
    z_consts = ctx["z_consts"]

    # Steric model (from standard build)
    steric_a_funcs = ctx["steric_a_funcs"]
    a_vals_list = [float(v) for v in a_vals]
    steric_active = any(v != 0.0 for v in a_vals_list)

    # Rebuild F_res
    F_res_new = fd.Constant(0.0) * v_tests[0] * dx  # start with zero

    for i in range(n):
        v = v_tests[i]
        is_log = (i in log_species)

        if is_log:
            # Log-transformed species: unknown is u_i = ln(c_i)
            u_i = ui_all[i]        # this IS u_i (log-concentration)
            u_old = ui_prev_all[i]
            # c_i = exp(u_i), clamped to prevent overflow
            c_i = fd.exp(fd.min_value(fd.max_value(u_i, fd.Constant(-_U_CLAMP)),
                                       fd.Constant(_U_CLAMP)))
            c_old = fd.exp(fd.min_value(fd.max_value(u_old, fd.Constant(-_U_CLAMP)),
                                         fd.Constant(_U_CLAMP)))

            # NP in log-space:
            # (exp(u) - exp(u_old))/dt * v + D*exp(u)*(∇u + z*∇φ)·∇v = BV
            drift = em * z_consts[i] * phi
            if steric_active:
                packing_floor = 1e-8
                ci_all = []
                for j in range(n):
                    if j in log_species:
                        ci_all.append(fd.exp(fd.min_value(fd.max_value(
                            ui_all[j], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP))))
                    else:
                        ci_all.append(ui_all[j])
                packing = fd.max_value(
                    fd.Constant(1.0) - sum(steric_a_funcs[j] * ci_all[j] for j in range(n)),
                    fd.Constant(packing_floor))
                mu_steric = fd.ln(packing)
                Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift) + fd.grad(mu_steric))
            else:
                Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift))

            F_res_new += ((c_i - c_old) / dt_const) * v * dx
            F_res_new += fd.dot(Jflux, fd.grad(v)) * dx

        else:
            # Standard species: unchanged from forms.py
            c_i = ui_all[i]
            c_old = ui_prev_all[i]
            drift = em * z_consts[i] * phi
            if steric_active:
                ci_all_for_steric = []
                for j in range(n):
                    if j in log_species:
                        ci_all_for_steric.append(fd.exp(fd.min_value(fd.max_value(
                            ui_all[j], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP))))
                    else:
                        ci_all_for_steric.append(ui_all[j])
                packing = fd.max_value(
                    fd.Constant(1.0) - sum(steric_a_funcs[j] * ci_all_for_steric[j] for j in range(n)),
                    fd.Constant(1e-8))
                mu_steric = fd.ln(packing)
                Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift) + c_i * fd.grad(mu_steric))
            else:
                Jflux = D[i] * (fd.grad(c_i) + c_i * fd.grad(drift))

            F_res_new += ((c_i - c_old) / dt_const) * v * dx
            F_res_new += fd.dot(Jflux, fd.grad(v)) * dx

    # Poisson equation: uses c_i for standard species, exp(u_i) for log species
    from Nondim.transform import _get_nondim_cfg, _bool
    nondim_cfg = _get_nondim_cfg(params)
    suppress_poisson_source = _bool(nondim_cfg.get("suppress_poisson_source", False))
    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    w = v_tests[-1]  # potential test function

    F_res_new += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
    if not suppress_poisson_source:
        for i in range(n):
            if i in log_species:
                c_i_poisson = fd.exp(fd.min_value(fd.max_value(
                    ui_all[i], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP)))
            else:
                c_i_poisson = ui_all[i]
            F_res_new -= charge_rhs * z_consts[i] * c_i_poisson * w * dx

    # BV boundary terms: need c_surf for all species
    # For log species, c_surf = exp(u_i) at the boundary
    bv_cfg = ctx["bv_settings"]
    conv_cfg = ctx["bv_convergence"]
    electrode_marker = bv_cfg["electrode_marker"]

    # Rebuild c_surf for BV
    if conv_cfg["regularize_concentration"]:
        eps_c = fd.Constant(float(conv_cfg["conc_floor"]))
        c_surf = []
        for i in range(n):
            if i in log_species:
                c_raw = fd.exp(fd.min_value(fd.max_value(
                    ui_all[i], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP)))
                c_surf.append(c_raw)  # exp(u) is always positive, no floor needed
            else:
                c_surf.append(fd.max_value(ui_all[i], eps_c))
    else:
        c_surf = []
        for i in range(n):
            if i in log_species:
                c_surf.append(fd.exp(fd.min_value(fd.max_value(
                    ui_all[i], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP))))
            else:
                c_surf.append(ui_all[i])

    # Rebuild BV rate expressions with the new c_surf
    from .config import _get_bv_reactions_cfg
    reactions_cfg = _get_bv_reactions_cfg(params, n)
    bv_rate_exprs = []
    bv_k0_funcs = ctx["bv_k0_funcs"]
    bv_alpha_funcs = ctx["bv_alpha_funcs"]

    if reactions_cfg is not None:
        rxns_scaled = scaling["bv_reactions"]
        bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))
        E_eq_global = fd.Constant(float(scaling["bv_E_eq_model"]))
        phi_applied_func = ctx["phi_applied_func"]

        def _eta_clipped(E_eq_const):
            if conv_cfg["use_eta_in_bv"]:
                eta_raw = phi_applied_func - E_eq_const
            else:
                eta_raw = phi - E_eq_const
            eta_scaled = bv_exp_scale * eta_raw
            if conv_cfg["clip_exponent"]:
                clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
                return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)
            return eta_scaled

        for j, rxn in enumerate(rxns_scaled):
            k0_j = bv_k0_funcs[j]
            alpha_j = bv_alpha_funcs[j]
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
                    F_res_new -= fd.Constant(float(stoi[i])) * R_j * v_tests[i] * ds(electrode_marker)

    # Stern layer (if active)
    if ctx.get("use_stern", False):
        stern_capacitance_model = scaling.get("bv_stern_capacitance_model")
        stern_coeff = fd.Constant(float(stern_capacitance_model))
        phi_applied_func = ctx["phi_applied_func"]
        F_res_new -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)

    # Update BCs: log-species need ln(c0) at bulk boundary
    c0_model = scaling.get("c0_model_vals", [1.0] * n)
    concentration_marker = bv_cfg["concentration_marker"]
    ground_marker = bv_cfg["ground_marker"]

    new_bcs = []
    for i in range(n):
        if i in log_species:
            c0_i = max(float(c0_model[i]), _C_FLOOR)
            new_bcs.append(fd.DirichletBC(W.sub(i), fd.Constant(np.log(c0_i)),
                                           concentration_marker))
        else:
            new_bcs.append(fd.DirichletBC(W.sub(i), fd.Constant(float(c0_model[i])),
                                           concentration_marker))

    # Phi BCs (same as standard)
    if not ctx.get("use_stern", False):
        phi_applied_func = ctx["phi_applied_func"]
        new_bcs.append(fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker))
    new_bcs.append(fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker))

    # Update context
    ctx["F_res"] = F_res_new
    ctx["J_form"] = fd.derivative(F_res_new, U)
    ctx["bcs"] = new_bcs
    ctx["bv_rate_exprs"] = bv_rate_exprs
    ctx["mixed_logc"] = True

    return ctx


def set_initial_conditions_mixed_logc(
    ctx: dict[str, Any],
    solver_params: Any,
    blob: bool = False,
) -> None:
    """Set initial conditions with log-transform for specified species."""
    from .forms import set_initial_conditions as _std_set_ic
    _std_set_ic(ctx, solver_params, blob=blob)

    # Convert log-species ICs from c to ln(c)
    log_species = ctx.get("log_species", _DEFAULT_LOG_SPECIES)
    U_prev = ctx["U_prev"]

    for i in log_species:
        c_data = U_prev.dat[i].data
        c_data[:] = np.log(np.maximum(c_data, _C_FLOOR))

    ctx["U"].assign(U_prev)
