"""Hybrid proton-electrochemical-potential transform for the BV PNP solver.

Experimental ``formulation="logc_muh"`` formulation.  Mirrors
``forms_logc.py`` but stores the proton (H+) primary variable as the
electrochemical potential

    mu_H = u_H + em * z_H * phi    where    em*z_H = +1 in the production scaling

so that ``c_H = exp(mu_H - em*z_H*phi)`` is reconstructed from the muh
unknown plus the resolved phi.  In Debye/Boltzmann regions ``mu_H`` is
nearly flat (``log(c_H_bulk)``), even when ``u_H`` and ``phi`` each vary
by tens of log-units, giving Newton a much smoother primary variable.
The Nernst-Planck flux for the muh species reduces to

    J_H = -D_H * c_H * grad(mu_H)

with no separate ``em*z*grad(phi)`` drift term.

Non-mu species (O2, H2O2 — neutrals) are unchanged from ``forms_logc.py``:
their primary variables remain ``u_i = log(c_i)``.

This module is a near-clone of ``forms_logc.py`` with substitutions on the
five ``c_H``-touch sites (NP flux, BV surface concentration, BV log-rate,
Bikerman packing, Poisson source) plus the IC pair.  The duplication is
intentional for the experimental phase; consolidation is tracked for after
Phase 7 (charged-species mu) lands.

Critical implementation notes:

* ``c_H_old`` for the time term must reconstruct using ``phi_prev =
  U_prev.sub(n)``, not the current ``phi``.  Otherwise the time
  derivative is silently wrong on transient runs.
* The ``u_clamp`` is applied to the reconstructed ``log(c_H) = mu_H -
  em*z_H*phi``, NOT to the raw ``mu_H``.  Raw ``mu_H`` is offset by
  ``phi`` (O(40) at high V), so a symmetric clamp on raw mu_H imposes a
  different physical bound than for ``forms_logc``.
* The ``debye_boltzmann`` IC for muh exhibits an analytic cancellation:
  with ``em*z_H = 1``, ``mu_H_init = u_H_init + em*z_H*phi_init =
  log(H_outer) - psi + log(H_outer/c_clo4_bulk) + psi = 2*log(H_outer) -
  log(c_clo4_bulk)``.  The diffuse-layer ``psi`` cancels exactly --
  Boltzmann equilibrium -> mu_H is smooth in y.

See:

* ``docs/electrochemical_potential_solver_plan.md`` (scoping doc and
  multi-phase landing plan with traps + sequencing).
* ``docs/PNP_BV_Analytical_Simplifications.md`` -- the rigorous reason
  ``mu_i`` is the smooth variable in the Debye layer.
* ``docs/4sp_bikerman_ic_option_2b_results.md`` -- the May 4 sweep
  exercising this backend end-to-end on the production target stack.
"""

from __future__ import annotations

from typing import Any

import math
import numpy as np
import firedrake as fd

from Nondim.transform import build_model_scaling, _get_nondim_cfg, _bool

from .config import (
    _get_bv_cfg,
    _get_bv_convergence_cfg,
    _get_bv_reactions_cfg,
    _get_bv_boltzmann_counterions_cfg,
)
from .nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform
from .boltzmann import (
    add_boltzmann_counterion_residual,
    build_steric_boltzmann_expressions,
)
from .forms_logc import build_context_logc, set_initial_conditions_logc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_mu_h_index(z_vals: list) -> int:
    """Return the index of the proton species (z = +1) for the muh formulation.

    Phase 2 of the muh landing is *hybrid* -- only H+ becomes the mu
    variable; ClO4- (z = -1, when dynamic) stays as u-log-c.  Charged-species
    mu for 4sp+Bikerman is Phase 7 and lives in a separate forms file.
    """
    candidates = [i for i, zv in enumerate(z_vals) if abs(float(zv) - 1.0) < 1e-12]
    if not candidates:
        raise ValueError(
            "formulation='logc_muh' requires a species with z=+1 (H+); none found "
            f"in z_vals={list(z_vals)}."
        )
    if len(candidates) > 1:
        raise ValueError(
            "formulation='logc_muh' requires exactly one species with z=+1 (H+); "
            f"found {len(candidates)} (indices {candidates})."
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context_logc_muh(solver_params: Any, *, mesh: Any = None) -> dict[str, Any]:
    """Build mesh and function spaces for muh BV PNP solver.

    Identical mixed-space layout to ``build_context_logc`` -- the muh
    transform changes only the *interpretation* of ``U.sub(mu_h_idx)``, not
    the function space.  Marks the context with ``logc_muh_transform=True``
    so downstream diagnostics know to read concentrations through
    ``ctx['ci_exprs']`` rather than ``np.exp(U.sub(i).dat...)``.
    """
    ctx = build_context_logc(solver_params, mesh=mesh)
    ctx["logc_muh_transform"] = True
    return ctx


def build_forms_logc_muh(ctx: dict[str, Any], solver_params: Any) -> dict[str, Any]:
    """Assemble weak forms with the proton electrochemical-potential transform.

    Primary unknowns: ``u_i = log(c_i)`` for non-mu species, ``mu_H =
    u_H + em*z_H*phi`` for the proton, plus ``phi``.  Concentrations
    reconstructed pointwise as

        c_H    = exp(min/max(mu_H - em*z_H*phi,    +/- u_clamp))
        c_H,old= exp(min/max(mu_H_old - em*z_H*phi_old, +/- u_clamp))   <- phi_prev!

    The clamp is applied to reconstructed ``log(c_H)`` (not raw ``mu_H``);
    see module docstring.
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

    # Nondimensionalization (same as forms_logc.py).
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
        # Stern layer capacitance (same as forms_logc.py)
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

    # Log-diffusivity controls (same as forms_logc.py).
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(scaling["D_model_vals"][i])))
    D = [fd.exp(m[i]) for i in range(n)]

    z = [fd.Constant(float(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]

    # Resolve mu species (Phase 2: H+ only).
    mu_h_idx = _resolve_mu_h_index(list(z_vals))
    mu_species = [mu_h_idx]

    # Split mixed function: u_i (non-mu) or mu_H (mu_h_idx) for species,
    # phi for potential.  Note: ui[mu_h_idx] is mu_H, NOT u_H.
    ui = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    ui_prev = fd.split(U_prev)[:-1]
    phi_prev = fd.split(U_prev)[-1]   # NEW for muh: c_H_old reconstruction
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    # Electromigration prefactor and dt -- moved up from forms_logc.py
    # (line 214) so we can reference em when reconstructing log(c_H) for
    # the mu-species.
    em = float(scaling["electromigration_prefactor"])
    dt_const = fd.Constant(float(scaling["dt_model"]))

    # ---------------------------------------------------------------
    # Reconstruct log-concentrations and concentrations
    # ---------------------------------------------------------------
    # For non-mu species:    u_expr(i)      = ui[i]
    # For mu species (H+):   u_expr(i)      = mu_H - em*z_H*phi              <- recover log(c_H)
    #                        u_expr_prev(i) = mu_H_prev - em*z_H*phi_prev    <- USE phi_prev
    #
    # The clamp is applied to the reconstructed log(c_H), not raw mu_H,
    # so the floating-point overflow protection on c_H = exp(...) is
    # preserved for the muh formulation.
    _U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))
    _U_CLAMP_C = fd.Constant(_U_CLAMP)
    _NEG_U_CLAMP_C = fd.Constant(-_U_CLAMP)

    def _u_expr(i: int, ui_split, phi_var):
        if i in mu_species:
            return ui_split[i] - fd.Constant(em) * z[i] * phi_var
        return ui_split[i]

    u_exprs      = [_u_expr(i, ui,      phi)      for i in range(n)]
    u_prev_exprs = [_u_expr(i, ui_prev, phi_prev) for i in range(n)]

    ci      = [fd.exp(fd.min_value(fd.max_value(u_exprs[i],      _NEG_U_CLAMP_C), _U_CLAMP_C))
               for i in range(n)]
    ci_prev = [fd.exp(fd.min_value(fd.max_value(u_prev_exprs[i], _NEG_U_CLAMP_C), _U_CLAMP_C))
               for i in range(n)]

    # Electrode potential constant.
    phi_applied_func = fd.Function(R_space, name="phi_applied")
    phi_applied_func.assign(float(scaling["phi_applied_model"]))
    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(float(scaling["phi0_model"]))

    # Global E_eq.
    E_eq_model_global = fd.Constant(float(scaling["bv_E_eq_model"]))
    bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))

    # Stern layer flag.
    stern_capacitance_model = scaling.get("bv_stern_capacitance_model")
    use_stern = stern_capacitance_model is not None and float(stern_capacitance_model) > 0

    def _build_eta_clipped(E_eq_const):
        """Build clipped overpotential expression for a given E_eq.

        See ``forms_logc.py`` for the full semantics of the clip.
        """
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

    # Steric chemical potential (Bikerman model) -- ci[mu_h_idx] is the
    # muh-reconstructed c_H, so packing automatically uses the right c_H.
    a_vals_list = [float(v) for v in a_vals]
    steric_a_funcs = [fd.Function(R_space, name=f"steric_a_{i}") for i in range(n)]
    for i in range(n):
        steric_a_funcs[i].assign(float(a_vals_list[i]))

    # Steric-aware analytic Boltzmann counterion(s) — same Option 2
    # wiring as forms_logc.py: closure expression must enter BOTH
    # Poisson AND the dynamic species' total packing fraction.
    c0_model_for_steric = scaling.get("c0_model_vals", c0_raw)
    steric_boltz = build_steric_boltzmann_expressions(
        ctx=ctx,
        params=params,
        ci=ci,
        a_dyn_funcs=steric_a_funcs,
        a_dyn_floats=a_vals_list,
        c0_dyn=[float(v) for v in c0_model_for_steric[:n]],
        z_dyn=[int(v) for v in z_vals[:n]],
        phi=phi,
        R_space=R_space,
    )

    steric_active = any(v != 0.0 for v in a_vals_list) or (steric_boltz is not None)
    if steric_active:
        packing_floor = float(conv_cfg.get("packing_floor", 1e-8))
        A_dyn = sum(steric_a_funcs[j] * ci[j] for j in range(n))
        if steric_boltz is not None:
            theta_inner = (
                fd.Constant(1.0) - A_dyn
                - steric_boltz.z_scale * steric_boltz.packing_contribution
            )
        else:
            theta_inner = fd.Constant(1.0) - A_dyn
        packing = fd.max_value(theta_inner, fd.Constant(packing_floor))
        mu_steric = -fd.ln(packing)

    # ---------------------------------------------------------------
    # Residual: Nernst-Planck with muh transform
    # ---------------------------------------------------------------
    # For non-mu species (u_i = log(c_i)):
    #   J_i = -D_i * c_i * (grad(u_i) + em*z_i*grad(phi))
    # For mu species (mu_H = u_H + em*z_H*phi):
    #   J_H = -D_H * c_H * grad(mu_H)
    # because grad(u_H) + em*z_H*grad(phi) = grad(mu_H) by definition.
    F_res = 0

    for i in range(n):
        c_i = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]

        if i in mu_species:
            # mu-species flux: D*c*grad(mu) (+ steric activity)
            ideal_grad = fd.grad(ui[i])
        else:
            # log-c flux: D*c*(grad(u) + em*z*grad(phi))
            drift = fd.Constant(em) * z[i] * phi
            ideal_grad = fd.grad(u_exprs[i]) + fd.grad(drift)

        if steric_active:
            Jflux = D[i] * c_i * (ideal_grad + fd.grad(mu_steric))
        else:
            Jflux = D[i] * c_i * ideal_grad

        # Time-stepping residual: (c - c_old)/dt
        F_res += ((c_i - c_old) / dt_const) * v * dx
        F_res += fd.dot(Jflux, fd.grad(v)) * dx

    # ---------------------------------------------------------------
    # Butler-Volmer boundary flux
    # ---------------------------------------------------------------
    # Surface concentrations are c_surf[i] = ci[i] -- already muh-aware
    # because ci[mu_h_idx] is the reconstruction.
    c_surf = ci

    bv_log_rate = bool(conv_cfg.get("bv_log_rate", False))

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

            # Per-reaction E_eq
            E_eq_j_val = rxn.get("E_eq_model", None)
            if E_eq_j_val is not None and E_eq_j_val != 0.0:
                E_eq_j = fd.Constant(float(E_eq_j_val))
                eta_j = _build_eta_clipped(E_eq_j)
            else:
                eta_j = eta_clipped

            if bv_log_rate:
                # Cathodic: log_r = ln(k0) + u_cat + sum power*(u_sp - ln c_ref)
                #                   - alpha * n_e * eta_clipped
                # MUH: u_exprs[i] is log(c_i) for both branches (recovered
                # from mu for the proton).
                log_cathodic = (
                    fd.ln(k0_j) + u_exprs[cat_idx]
                    - alpha_j * n_e_j * eta_j
                )
                for factor in rxn.get("cathodic_conc_factors", []):
                    sp_idx = factor["species"]
                    power = fd.Constant(float(factor["power"]))
                    c_ref_log = fd.ln(fd.Constant(
                        max(float(factor["c_ref_nondim"]), 1e-12)
                    ))
                    log_cathodic = log_cathodic + power * (u_exprs[sp_idx] - c_ref_log)
                cathodic = fd.exp(log_cathodic)

                # Anodic: log_r = ln(k0) + u_anod + (1-alpha) * n_e * eta_clipped
                if rxn["reversible"] and rxn["anodic_species"] is not None:
                    anod_idx = rxn["anodic_species"]
                    log_anodic = (
                        fd.ln(k0_j) + u_exprs[anod_idx]
                        + (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
                    )
                    anodic = fd.exp(log_anodic)
                elif rxn["reversible"] and float(rxn["c_ref_model"]) > 1e-30:
                    c_ref_j_log = fd.ln(fd.Constant(float(rxn["c_ref_model"])))
                    log_anodic = (
                        fd.ln(k0_j) + c_ref_j_log
                        + (fd.Constant(1.0) - alpha_j) * n_e_j * eta_j
                    )
                    anodic = fd.exp(log_anodic)
                else:
                    anodic = fd.Constant(0.0)
            else:
                # Cathodic term: k0 * c_cat * exp(-alpha * n_e * eta)
                # c_surf[mu_h_idx] is muh-reconstructed; transparent here.
                cathodic = k0_j * c_surf[cat_idx] * fd.exp(-alpha_j * n_e_j * eta_j)

                # Cathodic concentration factors
                for factor in rxn.get("cathodic_conc_factors", []):
                    sp_idx = factor["species"]
                    power = factor["power"]
                    c_ref_f = fd.Constant(max(float(factor["c_ref_nondim"]), 1e-12))
                    cathodic = cathodic * (c_surf[sp_idx] / c_ref_f) ** power

                # Anodic term (if reversible)
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
                    F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)

    else:
        # Legacy per-species path (concentration-form BV).  Surface c is
        # via c_surf[i] = ci[i] which is muh-aware.
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

    # ---------------------------------------------------------------
    # Poisson equation (uses c_i = exp(...) -- ci[mu_h_idx] is muh-reconstructed)
    # ---------------------------------------------------------------
    suppress_poisson_source = _bool(nondim_cfg.get("suppress_poisson_source", False))
    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
    if not suppress_poisson_source:
        F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
        if steric_boltz is not None:
            # Bikerman counterion charge density (mirror of forms_logc.py).
            F_res -= (
                steric_boltz.z_scale * charge_rhs
                * steric_boltz.charge_density * w * dx
            )

    # Stern layer Robin BC
    if use_stern:
        stern_coeff = fd.Constant(float(stern_capacitance_model))
        F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)

    # ---------------------------------------------------------------
    # Boundary conditions
    # ---------------------------------------------------------------
    # Concentration BCs:
    #   non-mu species:  u_i        = ln(c0_i)
    #   mu species:      mu_H_bulk  = ln(c0_H) + em*z_H*phi_bulk
    #
    # The concentration_marker (== ground_marker in the no-Stern case, or
    # the bulk for Stern) is where phi = 0 by construction, so the muh BC
    # value is numerically identical to ln(c0_H).  The explicit em*z*0
    # term is kept for documentation -- a future Robin or non-zero bulk
    # phi would activate it.
    c0_model = scaling.get("c0_model_vals", c0_raw)
    _C_FLOOR = 1e-20
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    bc_ui = []
    phi_bulk_at_ground = 0.0  # by Dirichlet at ground_marker
    for i in range(n):
        c0_i = max(float(c0_model[i]), _C_FLOOR)
        if i in mu_species:
            bc_val = fd.Constant(
                np.log(c0_i) + em * float(z_vals[i]) * phi_bulk_at_ground
            )
        else:
            bc_val = fd.Constant(np.log(c0_i))
        bc_ui.append(fd.DirichletBC(W.sub(i), bc_val, concentration_marker))
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
        "ci_exprs": ci,
        "u_exprs": u_exprs,           # NEW: log(c_i) for diagnostics in muh runs
        "mu_species": list(mu_species),  # NEW: which raw U.sub(i) is mu, not u
        "logc_transform": True,        # downstream code may key off this
        "logc_muh_transform": True,    # NEW: muh-specific code path flag
        # Diagnostic metadata mirroring forms_logc.py.
        "_diag_bv_exp_scale": float(bv_exp_scale),
        "_diag_exponent_clip": float(conv_cfg["exponent_clip"]),
        "_diag_eps_c": float(conv_cfg.get("conc_floor", 1e-8)),
        "steric_boltzmann": steric_boltz,
    })
    # Ideal-path Boltzmann counterions (bikerman entries handled above
    # by build_steric_boltzmann_expressions).
    add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)
    return ctx


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def set_initial_conditions_logc_muh(ctx: dict[str, Any], solver_params: Any) -> None:
    """Linear-phi IC for the muh formulation.

    Sets:

      non-mu species:   u_i(y) = ln(c0_i)              (constant, same as logc)
      mu species:       mu_H(y) = ln(c0_H) + em*z_H*phi_init(y)
      phi:              phi_init(y) = phi_applied * (1 - y)

    Pointwise reconstruction of c_H from these IC fields recovers
    c_H_bulk exactly:

      exp(mu_H(y) - em*z_H*phi(y)) = c_H_bulk

    so the initial concentration is bulk-uniform, identical to the
    ``set_initial_conditions_logc`` initial state but stored in muh
    coordinates.
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
    em = float(scaling.get("electromigration_prefactor", 1.0))

    mu_h_idx = _resolve_mu_h_index(list(z_vals))
    mu_species = {mu_h_idx}

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]

    phi_init_expr = fd.Constant(float(phi_applied_model)) * (1.0 - spatial_var)

    _C_FLOOR = 1e-20
    for i in range(n):
        c0_i = max(float(c0_model[i]), _C_FLOOR)
        if i in mu_species:
            # mu_H_init(y) = ln(c0_H) + em*z_H*phi_init(y)
            # -> reconstructs c_H = c0_H pointwise.
            U_prev.sub(i).interpolate(
                fd.Constant(np.log(c0_i))
                + fd.Constant(em * float(z_vals[i])) * phi_init_expr
            )
        else:
            U_prev.sub(i).assign(fd.Constant(np.log(c0_i)))

    U_prev.sub(n).interpolate(phi_init_expr)
    ctx["U"].assign(U_prev)


def set_initial_conditions_debye_boltzmann_logc_muh(
    ctx: dict[str, Any], solver_params: Any
) -> None:
    """Matched-asymptotic IC for the muh stack at high anodic phi.

    Identical Picard outer loop + Gouy-Chapman analytical Debye layer to
    the logc variant (``set_initial_conditions_debye_boltzmann_logc``);
    the only difference is the assignment for the proton species, which
    writes the muh primary variable

        mu_H_init(y) = u_H_init(y) + em*z_H*phi_init(y)

    With the production scaling ``em*z_H = 1`` the Debye-layer ``psi``
    cancels analytically (Boltzmann equilibrium), so ``mu_H_init`` reduces
    to ``2*log(H_outer) - log(c_clo4_bulk)``  -- exactly the smooth
    initial variable Newton wants in the diffuse layer.

    On Picard non-convergence or a degenerate config falls back to the
    muh linear-phi IC and sets ``ctx['initializer_fallback']=True``.

    Wraps the body in ``firedrake.adjoint.stop_annotating()`` so the
    Picard iterations and IC interpolations do not reach the pyadjoint
    tape.
    """
    import firedrake.adjoint as adj

    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    with adj.stop_annotating():
        ok, reason, picard_iters = _try_debye_boltzmann_ic_muh(
            ctx, solver_params, params, phi_applied, c0, n_species
        )
        if ok:
            ctx["initializer_fallback"] = False
            ctx["initializer_picard_iters"] = picard_iters
            return
        ctx["initializer_fallback"] = True
        ctx["initializer_fallback_reason"] = reason
        ctx["initializer_picard_iters"] = picard_iters
        set_initial_conditions_logc_muh(ctx, solver_params)


def _try_debye_boltzmann_ic_muh(
    ctx: dict[str, Any],
    solver_params: Any,
    params: Any,
    phi_applied: float,
    c0: Any,
    n_species: int,
) -> tuple[bool, str, int]:
    """Picard + Gouy-Chapman IC body for the muh formulation.

    Returns ``(success, reason, picard_iters)``.  On failure the caller
    should fall back to the muh linear-phi IC.  Must be called inside
    ``firedrake.adjoint.stop_annotating()``.

    The Picard outer loop and analytic Debye-layer profile mirror the
    logc variant exactly -- the Picard iteration is in scalar/python
    space and produces the same converged ``H_o``, ``psi_D``, ``O_s``,
    ``P_s``, ``c_clo4_bulk``.  Only the final FE assignment line for the
    proton species differs:

        logc:   U_prev.sub(mu_h_idx).interpolate(ln(H_outer) - psi)        # u_H
        muh:    U_prev.sub(mu_h_idx).interpolate(
                    (ln(H_outer) - psi)                                    # u_H
                    + em*z_H * (ln(H_outer/c_clo4_bulk) + psi)             # em*z_H*phi_init
                )                                                          # = mu_H_init

    With em*z_H = 1, ``psi`` cancels and ``mu_H_init = 2*ln(H_outer) -
    ln(c_clo4_bulk)``.
    """
    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    scaling = ctx.get("nondim", {})
    conv_cfg = ctx.get("bv_convergence", {})

    if n < 3:
        return False, f"n_species_lt_3 (n={n})", 0

    bv_reactions = scaling.get("bv_reactions", [])
    if len(bv_reactions) < 2:
        return False, f"fewer_than_2_reactions ({len(bv_reactions)})", 0

    # ----- Read scalar inputs ------------------------------------------
    phi_applied_model = float(scaling.get("phi_applied_model", float(phi_applied)))
    poisson_coefficient = float(scaling.get("poisson_coefficient", 1.0))
    lambda_D = math.sqrt(max(poisson_coefficient, 1e-300))
    em = float(scaling.get("electromigration_prefactor", 1.0))

    z_vals_full = list(solver_params[4])
    mu_h_idx = _resolve_mu_h_index(z_vals_full[:n])
    z_h_val = float(z_vals_full[mu_h_idx])

    D_model_vals = [float(v) for v in scaling.get("D_model_vals", [1.0] * n)]
    if np.isscalar(c0):
        c0_raw = [float(c0)] * n
    else:
        c0_raw = [float(v) for v in list(c0)][:n]
    c0_model = [max(float(v), 1e-300) for v in scaling.get("c0_model_vals", c0_raw)]

    # ----- Locate the Boltzmann counterion bulk concentration ---------
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    synthesised_4sp_counterion = False
    if not counterions:
        if (
            n == 4
            and len(z_vals_full) >= 4
            and int(z_vals_full[3]) == -1
            and len(c0_model) >= 4
        ):
            counterions = [{
                "z": -1,
                "c_bulk_nondim": float(c0_model[3]),
            }]
            synthesised_4sp_counterion = True
    if not counterions:
        return False, "no_boltzmann_counterion", 0

    # Phase 2 hybrid muh -- proton at index 2 (production 3sp+Boltzmann)
    # and matched-asymptotic Picard layout (O2, H2O2, H+ at indices
    # 0, 1, 2) is required.  Reject configurations where mu_h_idx != 2,
    # or fall through to logc IC.  The orchestrator's fallback handles
    # this gracefully.
    if mu_h_idx != 2:
        return False, f"mu_h_idx_unsupported ({mu_h_idx} != 2)", 0

    O_b = c0_model[0]
    P_b = c0_model[1]
    H_b = c0_model[2]
    D_O = max(D_model_vals[0], 1e-30)
    D_P = max(D_model_vals[1], 1e-30)
    D_H = max(D_model_vals[2], 1e-30)
    P_FLOOR = max(P_b, 1e-30)

    c_clo4_bulk = max(float(counterions[0]["c_bulk_nondim"]), 1e-300)

    bv_exp_scale = float(scaling.get("bv_exponent_scale", 1.0))
    exponent_clip = float(conv_cfg.get("exponent_clip", 50.0))
    clip_exponent = bool(conv_cfg.get("clip_exponent", True))

    rxn1 = bv_reactions[0]
    rxn2 = bv_reactions[1]
    k1 = float(rxn1["k0_model"])
    k2 = float(rxn2["k0_model"])
    a1 = float(rxn1["alpha"])
    a2 = float(rxn2["alpha"])
    n_e = float(rxn1["n_electrons"])
    E1 = float(rxn1.get("E_eq_model", 0.0))
    E2 = float(rxn2.get("E_eq_model", 0.0))

    h_factor1 = rxn1.get("cathodic_conc_factors", [])
    h_factor2 = rxn2.get("cathodic_conc_factors", [])

    def _eta_clipped(E: float) -> float:
        eta = bv_exp_scale * (phi_applied_model - E)
        if clip_exponent:
            return max(min(eta, exponent_clip), -exponent_clip)
        return eta

    eta1 = _eta_clipped(E1)
    eta2 = _eta_clipped(E2)

    def _h_factor_log(H_val: float, factors: list) -> float:
        total = 0.0
        H_log = math.log(max(H_val, 1e-300))
        for f in factors:
            if int(f["species"]) != 2:
                continue
            power = float(f["power"])
            c_ref_log = math.log(max(float(f["c_ref_nondim"]), 1e-30))
            total += power * (H_log - c_ref_log)
        return total

    def _safe_exp(x: float) -> float:
        if not math.isfinite(x):
            return float("inf") if x > 0 else 0.0
        if x > 700.0:
            return math.exp(700.0)
        if x < -700.0:
            return 0.0
        return math.exp(x)

    # ----- Picard outer loop -------------------------------------------
    OMEGA = 0.5
    MAX_ITERS = 50
    TOL = 1e-6

    R1 = 0.0
    R2 = 0.0
    H_o = H_b
    phi_o = 0.0
    psi_D = phi_applied_model - phi_o
    O_s = O_b
    P_s = P_b

    delta = float("inf")
    converged = False
    picard_iters = 0
    for k in range(1, MAX_ITERS + 1):
        R1_old, R2_old = R1, R2

        H_s = max(H_o * _safe_exp(-psi_D), 1e-300)

        log_h_factor1 = _h_factor_log(H_s, h_factor1)
        log_h_factor2 = _h_factor_log(H_s, h_factor2)
        log_A1 = math.log(k1) + log_h_factor1 - a1 * n_e * eta1
        log_B1 = math.log(k1) + (1.0 - a1) * n_e * eta1
        log_A2 = math.log(k2) + log_h_factor2 - a2 * n_e * eta2

        A1 = _safe_exp(log_A1)
        B1 = _safe_exp(log_B1)
        A2 = _safe_exp(log_A2)

        m11 = 1.0 + A1 / D_O + B1 / D_P
        m12 = -B1 / D_P
        m21 = -A2 / D_P
        m22 = 1.0 + A2 / D_P
        rhs1 = A1 * O_b - B1 * P_b
        rhs2 = A2 * P_b
        det = m11 * m22 - m12 * m21
        if not math.isfinite(det) or abs(det) < 1e-300:
            return False, f"singular_jacobian_iter_{k}_det={det:.3g}", k
        R1_new = (m22 * rhs1 - m12 * rhs2) / det
        R2_new = (-m21 * rhs1 + m11 * rhs2) / det
        if not (math.isfinite(R1_new) and math.isfinite(R2_new)):
            return False, f"non_finite_R_iter_{k}", k

        R1 = (1.0 - OMEGA) * R1 + OMEGA * R1_new
        R2 = (1.0 - OMEGA) * R2 + OMEGA * R2_new

        O_s = max(O_b - R1 / D_O, 1e-300)
        P_s = max(P_b + (R1 - R2) / D_P, P_FLOOR)
        H_o = max(H_b - (R1 + R2) / D_H, 1e-300)
        phi_o = math.log(H_o / c_clo4_bulk)
        psi_D = phi_applied_model - phi_o

        denom1 = max(abs(R1), 1e-30)
        denom2 = max(abs(R2), 1e-30)
        delta = abs(R1 - R1_old) / denom1 + abs(R2 - R2_old) / denom2
        picard_iters = k
        if delta < TOL:
            converged = True
            break

    if not converged:
        return False, f"picard_max_iters_delta={delta:.4g}", picard_iters

    # ----- Numerically-safe Gouy-Chapman psi baseline (always built) ---
    EPS_TANH = 1e-15
    T = math.tanh(psi_D / 4.0)
    T_clamp = math.copysign(min(abs(T), 1.0 - EPS_TANH), T)

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()
    y = coords[0] if ndim == 1 else coords[1]

    E_expr = fd.exp(-y / fd.Constant(lambda_D))
    arg = fd.Constant(T_clamp) * E_expr
    arg_safe = fd.min_value(
        fd.max_value(arg, fd.Constant(-1.0 + EPS_TANH)),
        fd.Constant(1.0 - EPS_TANH),
    )
    psi_gc = fd.Constant(2.0) * fd.ln(
        (fd.Constant(1.0) + arg_safe) / (fd.Constant(1.0) - arg_safe)
    )

    O_outer = fd.Constant(O_s) + (fd.Constant(O_b) - fd.Constant(O_s)) * y
    P_outer = fd.max_value(
        fd.Constant(P_s) + (fd.Constant(P_b) - fd.Constant(P_s)) * y,
        fd.Constant(P_FLOOR),
    )
    H_outer = fd.max_value(
        fd.Constant(H_o) + (fd.Constant(H_b) - fd.Constant(H_o)) * y,
        fd.Constant(1e-300),
    )

    # Bikerman-consistent IC paths (synthesised 4sp ClO4-, or explicit
    # ``boltzmann_counterions=[{steric_mode='bikerman', ...}]``).  Mirrors
    # forms_logc.py:_try_debye_boltzmann_ic; the only muh-specific bit is
    # the proton stored as mu_H = u_H + em*z_H*phi (psi cancels with em*z_H = 1,
    # log_gamma propagates through u_H into mu_H).
    bikerman_in_counterions = bool(counterions) and any(
        e.get("steric_mode", "ideal") == "bikerman" for e in counterions
    )
    apply_bikerman_ic = synthesised_4sp_counterion or bikerman_in_counterions

    if apply_bikerman_ic:
        a_vals_full = list(solver_params[6])
        a_h = float(a_vals_full[2])
        if synthesised_4sp_counterion:
            a_cl = float(a_vals_full[3])
            c_cl_anchor = H_outer
        else:
            bikerman_entry = next(
                e for e in counterions
                if e.get("steric_mode", "ideal") == "bikerman"
            )
            a_cl = float(bikerman_entry["a_nondim"])
            c_cl_anchor = fd.Constant(c_clo4_bulk)

        # Composite psi (BKSA matched-asymptotic): saturated zone +
        # outer exponential decay.  See forms_logc.py for the full
        # comment block.
        nu_charged = 2.0 * a_cl * c_clo4_bulk
        psi_d_abs = abs(float(psi_D))
        if nu_charged <= 0.0 or psi_d_abs < 1e-6:
            psi = psi_gc
        else:
            psi_sat_val = math.log(2.0 / nu_charged)
            sign_psi_D = 1.0 if psi_D >= 0.0 else -1.0
            if psi_d_abs <= psi_sat_val * (1.0 - 1e-3):
                psi = fd.Constant(float(psi_D)) * fd.exp(
                    -y / fd.Constant(lambda_D)
                )
            else:
                arg_cosh = math.cosh(psi_d_abs)
                alpha_d = math.sqrt(
                    (2.0 / (nu_charged * lambda_D ** 2))
                    * math.log(1.0 + nu_charged * (arg_cosh - 1.0))
                )
                y_match_val = (psi_d_abs - psi_sat_val) / alpha_d
                psi_zone1 = fd.Constant(sign_psi_D) * (
                    fd.Constant(psi_d_abs) - fd.Constant(alpha_d) * y
                )
                psi_zone2 = fd.Constant(sign_psi_D * psi_sat_val) * fd.exp(
                    -(y - fd.Constant(y_match_val)) / fd.Constant(lambda_D)
                )
                psi = fd.conditional(
                    y < fd.Constant(y_match_val), psi_zone1, psi_zone2
                )

        phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi

        gamma_psi = fd.Constant(1.0) / (
            fd.Constant(1.0)
            + fd.Constant(a_h)  * H_outer    * (fd.exp(-psi) - fd.Constant(1.0))
            + fd.Constant(a_cl) * c_cl_anchor * (fd.exp(+psi) - fd.Constant(1.0))
        )
        log_gamma = fd.ln(gamma_psi)

        U_prev.sub(0).interpolate(fd.ln(O_outer) + log_gamma)
        U_prev.sub(1).interpolate(fd.ln(P_outer) + log_gamma)

        # Mu species (proton): mu_H = u_H + em*z_H*phi.  With em*z_H = 1,
        # the (-psi) in u_H cancels the (+psi) in phi, leaving the
        # log_gamma offset as a smooth additive shift on mu_H:
        #   mu_H_init = (ln H_outer - psi + log_gamma) + (ln(H_outer/c_clo4_bulk) + psi)
        #             = 2*ln(H_outer) - ln(c_clo4_bulk) + log_gamma.
        u_h_init_expr  = fd.ln(H_outer) - psi + log_gamma
        mu_h_init_expr = (
            u_h_init_expr
            + fd.Constant(em * z_h_val) * phi_init_expr
        )
        U_prev.sub(2).interpolate(mu_h_init_expr)
        U_prev.sub(n).interpolate(phi_init_expr)

        if synthesised_4sp_counterion:
            # 4sp dynamic ClO4- stays as u_3 = ln(c_ClO4) (Phase 2 hybrid
            # muh treats only H+ as mu).  Apply gamma correction here too.
            U_prev.sub(3).interpolate(
                fd.Constant(math.log(c_clo4_bulk)) + phi_init_expr + log_gamma
            )
    else:
        # 3sp + ideal counterion (legacy): byte-identical to pre-2b.
        # ``apply_bikerman_ic = False`` here implies
        # ``synthesised_4sp_counterion = False``, so no u_3 seed is needed.
        psi = psi_gc
        phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi

        U_prev.sub(0).interpolate(fd.ln(O_outer))
        U_prev.sub(1).interpolate(fd.ln(P_outer))

        u_h_init_expr  = fd.ln(H_outer) - psi
        mu_h_init_expr = (
            u_h_init_expr
            + fd.Constant(em * z_h_val) * phi_init_expr
        )
        U_prev.sub(2).interpolate(mu_h_init_expr)
        U_prev.sub(n).interpolate(phi_init_expr)

    ctx["U"].assign(U_prev)
    return True, "", picard_iters
