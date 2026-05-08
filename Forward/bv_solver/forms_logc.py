"""Log-concentration transform for the BV PNP solver.

Uses u_i = ln(c_i) as primary unknowns instead of c_i.
Key advantages:
  1. c_i = exp(u_i) is ALWAYS positive (no concentration floor needed)
  2. Exponential Debye-layer variation becomes linear in u (better conditioned)
  3. Eliminates the Jacobian singularity that kills z-ramp at onset voltages

The weak form (Nernst-Planck):
  ∫ (exp(u) - exp(u_old))/dt · v dx + ∫ D·exp(u)·(∇u + z·∇φ)·∇v dx = BV terms

Poisson:
  ∫ ε·∇φ·∇w dx - ∫ charge_rhs·Σ z_i·exp(u_i)·w dx = 0

This module mirrors forms.py (build_context_logc, build_forms_logc, set_initial_conditions_logc)
but with the log-concentration substitution throughout.
"""

from __future__ import annotations

from typing import Any

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context_logc(solver_params: Any, *, mesh: Any = None) -> dict[str, Any]:
    """Build mesh and function spaces for log-c BV PNP solver.

    The mixed function space is [V_scalar^{n_species+1}] where the first
    n_species components are u_i = ln(c_i) and the last is phi.
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
        "logc_transform": True,  # flag for downstream code
    }


def build_forms_logc(ctx: dict[str, Any], solver_params: Any) -> dict[str, Any]:
    """Assemble weak forms using log-concentration transform.

    Primary unknowns: u_i = ln(c_i), phi.
    Concentrations reconstructed as c_i = exp(u_i).
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

    # Nondimensionalization (same as forms.py).
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
        # Stern layer capacitance (same as forms.py)
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

    # Log-diffusivity controls (same as forms.py).
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(scaling["D_model_vals"][i])))
    D = [fd.exp(m[i]) for i in range(n)]

    z = [fd.Constant(float(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]

    # Split: u_i = ln(c_i) for species, phi for potential
    ui = fd.split(U)[:-1]      # log-concentrations
    phi = fd.split(U)[-1]
    ui_prev = fd.split(U_prev)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    # Reconstruct concentrations from log-transform.
    # c_i = exp(u_i), clamped symmetrically to avoid floating-point overflow
    # in either direction.  Default 30: exp(±30) covers [9.4e-14, 1.07e+13],
    # adequate for V_RHE in [-0.5, +0.6] V at SS.
    #
    # *** WIDEN to u_clamp=100 for V_RHE > +0.30 V *** — at higher voltages
    # R2 consumes H2O2 faster than R1 produces it and the SS u_H2O2 itself
    # approaches the lower clamp, so it starts binding *at SS* (not just on
    # transient Newton iterates) and distorts the bulk PDE coefficient.
    # exp(±100) ~ 2.7e±43 is still inside double precision.
    #
    # Log-rate eliminated the analogous clamp inside the BV boundary
    # residual (c_surf = exp(clamp(u, +/-30))) but CANNOT eliminate this
    # bulk one: exp(u_i) appears as a *coefficient* multiplying gradients /
    # test functions in the PDE (D*c*grad(u)*grad(v), Poisson source
    # sum z_i*c_i*w) — not as the argument of one deferred exp — so no
    # algebraic rearrangement absorbs it into a single exp(scalar_sum).
    # See docs/clipping_conventions.md for the full discussion.
    _U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))
    ci = [fd.exp(fd.min_value(fd.max_value(ui[i], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP)))
          for i in range(n)]
    ci_prev = [fd.exp(fd.min_value(fd.max_value(ui_prev[i], fd.Constant(-_U_CLAMP)), fd.Constant(_U_CLAMP)))
               for i in range(n)]

    em = float(scaling["electromigration_prefactor"])
    dt_const = fd.Constant(float(scaling["dt_model"]))

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

        Note: clip is applied to ``eta_scaled = (V - E_eq) / V_T``, NOT to the
        full BV exponent ``alpha * n_e * eta_scaled``. So a reaction's BV
        exponent unclips when ``|V - E_eq| < exponent_clip * V_T``.
        For exponent_clip=50, V_T=0.02569 V, E_eq_2=1.78 V, this gives
        V_unclip_2 = 1.78 - 50*0.02569 = +0.495 V (R2 unclipping threshold).
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

    # Steric chemical potential (Bikerman model)
    a_vals_list = [float(v) for v in a_vals]
    steric_a_funcs = [fd.Function(R_space, name=f"steric_a_{i}") for i in range(n)]
    for i in range(n):
        steric_a_funcs[i].assign(float(a_vals_list[i]))

    # Steric-aware analytic Boltzmann counterion(s) — built BEFORE the
    # dynamic-species mu_steric so the closure expression enters BOTH
    # Poisson AND the dynamic species' total packing fraction.  The
    # legacy add_boltzmann_counterion_residual call at the end of this
    # function is invoked with skip_bikerman=True so it doesn't double-
    # count.  Returns None when no bikerman entries are configured —
    # the legacy ideal path is unchanged.
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
            # Multiply the bikerman counterion's packing contribution
            # by the same ``boltzmann_z_scale`` Function the Poisson
            # source uses, so Strategy-B / C+D z-ramps zero out BOTH
            # contributions consistently at z=0.  Without this, at z=0
            # with a high-phi IC the closure saturates locally near the
            # electrode and ``packing_floor`` activates with a huge
            # gradient ``a/packing ~ 1e6``, breaking Newton at the
            # first ramp step.
            theta_inner = (
                fd.Constant(1.0) - A_dyn
                - steric_boltz.z_scale * steric_boltz.packing_contribution
            )
        else:
            theta_inner = fd.Constant(1.0) - A_dyn
        packing = fd.max_value(theta_inner, fd.Constant(packing_floor))
        mu_steric = -fd.ln(packing)

    # ---------------------------------------------------------------
    # Residual: Nernst-Planck with log-concentration transform
    # ---------------------------------------------------------------
    # For c_i = exp(u_i):
    #   ∇c_i = c_i · ∇u_i
    #   J_i = -D_i · (∇c_i + z_i · c_i · ∇φ) = -D_i · c_i · (∇u_i + z_i · ∇φ)
    #
    # Weak form of ∂c/∂t + ∇·J = 0:
    #   ∫ (c - c_old)/dt · v dx + ∫ D·c·(∇u + z·∇φ)·∇v dx = boundary terms
    #
    # Note: c_i = exp(u_i) enters as a coefficient. The concentration
    # gradient is absorbed into the u_i gradient, giving a cleaner
    # Jacobian structure.
    F_res = 0

    for i in range(n):
        c_i = ci[i]
        c_old = ci_prev[i]
        u_i = ui[i]
        v = v_list[i]

        # Electromigration drift
        drift = em * z[i] * phi
        if steric_active:
            # J = D·c·(∇u + ∇drift + ∇μ_steric)
            Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift) + fd.grad(mu_steric))
        else:
            # J = D·c·(∇u + z·∇φ) — the key log-transform simplification
            Jflux = D[i] * c_i * (fd.grad(u_i) + fd.grad(drift))

        # Time-stepping residual: (c - c_old)/dt
        F_res += ((c_i - c_old) / dt_const) * v * dx
        F_res += fd.dot(Jflux, fd.grad(v)) * dx

    # ---------------------------------------------------------------
    # Butler-Volmer boundary flux
    # ---------------------------------------------------------------
    # Surface concentrations are simply c_surf[i] = exp(u_i) at the boundary.
    # No regularization needed — exp(u_i) is always positive!
    c_surf = ci  # already exp(u_i)

    # Stage 2: log-rate BV evaluation. When enabled, build the rate as
    # exp(log k0 + u_cat + sum(power*(u_sp - log c_ref)) - alpha*n_e*eta).
    # This uses ui[i] (unclamped) instead of c_surf[i] (= exp(clamp(ui))),
    # which removes the artificial R2 sink that the lower _U_CLAMP creates
    # when c_H2O2 underflows during Newton iteration.
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

            # k0 <= 0 means disabled; skip to avoid fd.ln(0) in log-rate branch.
            if float(rxn["k0_model"]) <= 0.0:
                R_j = fd.Constant(0.0)
                bv_rate_exprs.append(R_j)
                continue

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
                log_cathodic = (
                    fd.ln(k0_j) + ui[cat_idx]
                    - alpha_j * n_e_j * eta_j
                )
                for factor in rxn.get("cathodic_conc_factors", []):
                    sp_idx = factor["species"]
                    power = fd.Constant(float(factor["power"]))
                    c_ref_log = fd.ln(fd.Constant(
                        max(float(factor["c_ref_nondim"]), 1e-12)
                    ))
                    log_cathodic = log_cathodic + power * (ui[sp_idx] - c_ref_log)
                cathodic = fd.exp(log_cathodic)

                # Anodic: log_r = ln(k0) + u_anod + (1-alpha) * n_e * eta_clipped
                if rxn["reversible"] and rxn["anodic_species"] is not None:
                    anod_idx = rxn["anodic_species"]
                    log_anodic = (
                        fd.ln(k0_j) + ui[anod_idx]
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
        # Legacy per-species path
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
    # Poisson equation (uses c_i = exp(u_i))
    # ---------------------------------------------------------------
    suppress_poisson_source = _bool(nondim_cfg.get("suppress_poisson_source", False))
    eps_coeff = fd.Constant(float(scaling["poisson_coefficient"]))
    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
    if not suppress_poisson_source:
        F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
        if steric_boltz is not None:
            # Steric-aware analytic counterion contribution to Poisson.
            # Mirrors the ideal-path expression in
            # add_boltzmann_counterion_residual but uses the closure
            # ``c_steric`` (bounded at 1/a_b) instead of the unbounded
            # ``c_b * exp(-z*phi)``.  Multiplied by the shared
            # ``boltzmann_z_scale`` so Strategy-B z-ramps continue to
            # zero out both ideal and bikerman counterion charges
            # together.
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
    # Concentration BCs: u_i = ln(c0_i) at bulk boundary
    # NOTE: c0 = 0 for product species (H2O2) — use a small positive floor
    c0_model = scaling.get("c0_model_vals", c0_raw)
    _C_FLOOR = 1e-20  # floor for ln(0) avoidance
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    bc_ui = []
    for i in range(n):
        c0_i = max(float(c0_model[i]), _C_FLOOR)
        bc_ui.append(
            fd.DirichletBC(W.sub(i), fd.Constant(np.log(c0_i)), concentration_marker)
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
        "ci_exprs": ci,   # exp(u_i) expressions for observable extraction
        # Diagnostic metadata mirroring forms.py so downstream validation /
        # observable code that reads these keys works in the logc path too.
        "_diag_bv_exp_scale": float(bv_exp_scale),
        "_diag_exponent_clip": float(conv_cfg["exponent_clip"]),
        "_diag_eps_c": float(conv_cfg.get("conc_floor", 1e-8)),
        "logc_transform": True,
        "steric_boltzmann": steric_boltz,  # None for ideal-only configs
    })
    # Ideal-path Boltzmann counterions (bikerman entries are wired above
    # via build_steric_boltzmann_expressions; skip them here).
    add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)
    return ctx


def set_initial_conditions_logc(ctx: dict[str, Any], solver_params: Any) -> None:
    """Set initial conditions in log-concentration space.

    Sets u_i = ln(c0_i) for species, linear phi profile.

    When the residual is Stern-aware, the linear phi anchors at the
    OHP-side surface potential ``phi_applied - psi_S`` rather than
    ``phi_applied`` -- otherwise every fallback row sees the residual
    eta collapse to ``-E_eq`` (handoff #12 §6, Codex's Fix 3).
    """
    import math
    from .picard_ic import solve_stern_split

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

    # Stern-aware anchoring (Phase F).  When use_stern is active, solve
    # for ``psi_S`` at bulk outer values so the linear phi profile does
    # not collapse to phi(0) = phi_applied (which would give residual
    # eta = -E_eq for every fallback row).
    stern_capacitance_model = scaling.get("bv_stern_capacitance_model")
    use_stern_at_ic = (
        stern_capacitance_model is not None
        and float(stern_capacitance_model) > 0
    )
    phi_surface = phi_applied_model
    if use_stern_at_ic and n >= 3 and len(c0_model) >= 3:
        # Bulk outer reference: phi_o_bulk = log(H_b / c_clo4_bulk).
        # Need a counterion bulk concentration estimate.  Pull from the
        # config (analytic boltzmann_counterions or synthesised 4sp).
        counterions = _get_bv_boltzmann_counterions_cfg(params)
        c_clo4_bulk = None
        a_cl_bulk = 0.0
        if counterions:
            c_clo4_bulk = max(float(counterions[0]["c_bulk_nondim"]), 1e-300)
            for e in counterions:
                if e.get("steric_mode", "ideal") == "bikerman":
                    a_cl_bulk = float(e.get("a_nondim", 0.0))
                    break
        elif n == 4 and len(c0_model) >= 4:
            c_clo4_bulk = max(float(c0_model[3]), 1e-300)
            a_vals_full = list(solver_params[6])
            if len(a_vals_full) >= 4:
                a_cl_bulk = float(a_vals_full[3])
        if c_clo4_bulk is not None:
            H_b = max(float(c0_model[2]), 1e-300)
            phi_o_bulk = math.log(H_b / c_clo4_bulk)
            poisson_coefficient = float(scaling.get("poisson_coefficient", 1.0))
            lambda_D_bulk = math.sqrt(max(poisson_coefficient, 1e-300))
            psi_S, _, phi_surface_split = solve_stern_split(
                phi_applied_model=float(phi_applied_model),
                phi_o=phi_o_bulk,
                lambda_D=lambda_D_bulk,
                c_clo4_bulk=c_clo4_bulk,
                a_cl=a_cl_bulk,
                stern_coeff_nondim=float(stern_capacitance_model),
                eps_nondim=poisson_coefficient,
            )
            phi_surface = phi_surface_split

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()

    _C_FLOOR = 1e-20
    for i in range(n):
        c0_i = max(float(c0_model[i]), _C_FLOOR)
        U_prev.sub(i).assign(fd.Constant(np.log(c0_i)))

    # Linear potential profile
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]
    U_prev.sub(n).interpolate(fd.Constant(float(phi_surface)) * (1.0 - spatial_var))
    ctx["U"].assign(U_prev)


def set_initial_conditions_debye_boltzmann_logc(
    ctx: dict[str, Any], solver_params: Any
) -> None:
    """Matched-asymptotic IC for the log-c stack at high anodic phi.

    Outer ambipolar 2*D_H proton transport + 2x2 algebraic surface-rate
    Picard solve + Gouy-Chapman analytical Debye layer.  Seeds Newton on
    the depleted-H+ / enriched-counterion manifold so the solver does not
    have to discover the O(40) Boltzmann depletion in a single step.
    See ``docs/PNP_BV_Analytical_Simplifications.md``.

    On Picard non-convergence (max iters, NaN, singular Jacobian), or on
    a degenerate config (no Boltzmann counterion, n_species < 3), falls
    back to the linear-phi IC and sets ``ctx['initializer_fallback']=True``
    with a string in ``ctx['initializer_fallback_reason']``.

    The whole body runs under ``firedrake.adjoint.stop_annotating()`` so
    Picard iterations and IC interpolations never reach the pyadjoint tape
    -- the IC is an initial guess, not a control.
    """
    import math
    import firedrake.adjoint as adj

    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    with adj.stop_annotating():
        ok, reason, picard_iters = _try_debye_boltzmann_ic(
            ctx, solver_params, params, phi_applied, c0, n_species
        )
        if ok:
            ctx["initializer_fallback"] = False
            ctx["initializer_picard_iters"] = picard_iters
            return
        ctx["initializer_fallback"] = True
        ctx["initializer_fallback_reason"] = reason
        ctx["initializer_picard_iters"] = picard_iters
        set_initial_conditions_logc(ctx, solver_params)


def _try_debye_boltzmann_ic(
    ctx: dict[str, Any],
    solver_params: Any,
    params: Any,
    phi_applied: float,
    c0: Any,
    n_species: int,
) -> tuple[bool, str, int]:
    """Picard + Gouy-Chapman IC body. Returns (success, reason, picard_iters).

    Wraps the shared scalar Picard outer loop in
    ``Forward.bv_solver.picard_ic.picard_outer_loop``.  This function
    handles config unpacking, counterion detection, mesh / coordinate
    setup, and the post-Picard FE interpolation (Bikerman composite-psi
    + multispecies-gamma seed, or legacy ideal-counterion GC).  All
    scalar Picard algebra lives in ``picard_ic``.

    On Picard failure the caller should fall back to linear-phi.  Must
    be called inside ``firedrake.adjoint.stop_annotating()``.
    """
    import math
    from .picard_ic import picard_outer_loop_general

    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    scaling = ctx.get("nondim", {})
    conv_cfg = ctx.get("bv_convergence", {})

    if n < 3:
        return False, f"n_species_lt_3 (n={n})", 0

    bv_reactions = scaling.get("bv_reactions", [])
    if len(bv_reactions) < 1:
        return False, f"empty_reactions ({len(bv_reactions)})", 0

    # M3a.3 topology dispatch (per
    # ``docs/picard_general_topology_derivation.md`` v3): the legacy
    # SEQUENTIAL reaction set (rxn1 produces H₂O₂, rxn2 consumes it)
    # uses the post-loop closed-form ``P_s`` reconstruction (§8) which
    # eliminates a near-cancellation in the diffusion-limited regime.
    # The Ruggiero PARALLEL 2e/4e topology has rxn2_stoich[H2O2]=0
    # (R_4e doesn't touch H₂O₂) — there is no cancellation source and
    # the naive flux balance is robust.  We dispatch via topology_hint
    # rather than rejecting non-sequential configs (the M3a.2 gate);
    # the generalized N-reaction Picard handles both.
    rxn1_stoich = list(bv_reactions[0].get("stoichiometry", []))
    rxn2_stoich = (
        list(bv_reactions[1].get("stoichiometry", []))
        if len(bv_reactions) >= 2 else []
    )
    rxn2_reversible = (
        bool(bv_reactions[1].get("reversible", False))
        if len(bv_reactions) >= 2 else False
    )
    is_sequential_template = (
        len(bv_reactions) == 2
        and len(rxn1_stoich) >= 2 and len(rxn2_stoich) >= 2
        and int(rxn1_stoich[1]) > 0
        and int(rxn2_stoich[1]) < 0
        and not rxn2_reversible
    )
    topology_hint_picard = (
        "sequential_2e_h2o2" if is_sequential_template else "general"
    )

    # ----- Read scalar inputs ------------------------------------------
    phi_applied_model = float(scaling.get("phi_applied_model", float(phi_applied)))
    poisson_coefficient = float(scaling.get("poisson_coefficient", 1.0))
    lambda_D = math.sqrt(max(poisson_coefficient, 1e-300))

    D_model_vals = [float(v) for v in scaling.get("D_model_vals", [1.0] * n)]
    if np.isscalar(c0):
        c0_raw = [float(c0)] * n
    else:
        c0_raw = [float(v) for v in list(c0)][:n]
    c0_model = [max(float(v), 1e-300) for v in scaling.get("c0_model_vals", c0_raw)]

    # ----- Locate the Boltzmann counterion bulk concentration ---------
    # Primary path: an explicit ``boltzmann_counterions`` config entry
    # (the 3sp+Boltzmann production stack).  Fallback: 4sp dynamic with
    # an inert ClO4- as species 3 (z=-1).  In that case the same
    # Picard + Gouy-Chapman analytical IC physics apply -- ClO4- in
    # steady state with no BV reaction and Dirichlet bulk BC is exactly
    # the Boltzmann distribution (the equivalence proven by
    # ``tests/test_solver_equivalence.py``) -- so we synthesise an
    # equivalent counterion entry from ``c0_model[3]`` and let the IC
    # fire instead of falling back to linear-phi.
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    synthesised_4sp_counterion = False
    if not counterions:
        z_vals_full = list(solver_params[4])
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

    O_b = c0_model[0]
    P_b = c0_model[1]
    H_b = c0_model[2]
    D_O = max(D_model_vals[0], 1e-30)
    D_P = max(D_model_vals[1], 1e-30)
    D_H = max(D_model_vals[2], 1e-30)
    # P_FLOOR: small absolute floor to prevent log(0) underflow.  The
    # legacy ``max(P_b, 1e-30)`` was too aggressive -- in the diffusion-
    # limited regime (large A2), the matched-asymptotic balance
    # ``P_s = R2 / A2`` gives ``P_s << P_b`` (e.g. ~1e-16 at V=+0.5 V
    # with Stern + bikerman).  Clamping at P_b broke residual rate
    # consistency for R2 (cathodic_R2 in residual ~ A2 * c_P(IC) =
    # A2 * P_s_clamped * gamma, which is many orders of magnitude
    # larger than Picard's R2 = A2 * P_s_unclamped).  A pure 1e-30
    # numerical floor preserves the diffusion-limited consistency.
    P_FLOOR = 1e-30

    c_clo4_bulk = max(float(counterions[0]["c_bulk_nondim"]), 1e-300)

    bv_exp_scale = float(scaling.get("bv_exponent_scale", 1.0))
    exponent_clip = float(conv_cfg.get("exponent_clip", 100.0))
    clip_exponent = bool(conv_cfg.get("clip_exponent", True))

    # Per-reaction scalars (k0, alpha, n_electrons, E_eq, cathodic_conc_factors,
    # cathodic_species, anodic_species, reversible, c_ref_model, stoichiometry)
    # are read directly from ``bv_reactions`` by ``picard_outer_loop_general``;
    # the M3a.2-era unpacking into ``k1, k2, a1, a2, n_e, E1, E2, h_factor1,
    # h_factor2`` was tied to the legacy 2x2 ``picard_outer_loop`` and is
    # removed in M3a.3 (per v3 §9 contract item 2: per-reaction n_e read
    # from ``reactions[j]``, not ``reactions[0]`` only).

    # Two paths reach the bikerman-consistent IC.
    bikerman_in_counterions = bool(counterions) and any(
        e.get("steric_mode", "ideal") == "bikerman" for e in counterions
    )
    apply_bikerman_ic = synthesised_4sp_counterion or bikerman_in_counterions

    # Bikerman size parameters for the Picard's surface gamma.  For
    # ``a_h = a_cl = 0`` (ideal counterion) the loop reduces to the
    # legacy gamma-free Picard.
    if apply_bikerman_ic:
        a_vals_full_for_picard = list(solver_params[6])
        a_h_picard = float(a_vals_full_for_picard[2])
        if synthesised_4sp_counterion:
            # 4sp dynamic ClO4-: a_cl is the dynamic ClO4- size; outer
            # anchor is H_o (electroneutrality with the proton).
            a_cl_picard = float(a_vals_full_for_picard[3])
            c_cl_anchor_kind = "synthesised_4sp"
        else:
            # 3sp + analytic bikerman counterion: a_cl is the
            # ``a_nondim`` of the bikerman entry; anchor = c_clo4_bulk.
            bikerman_entry_for_picard = next(
                e for e in counterions
                if e.get("steric_mode", "ideal") == "bikerman"
            )
            a_cl_picard = float(bikerman_entry_for_picard["a_nondim"])
            c_cl_anchor_kind = "bulk"
    else:
        a_h_picard = 0.0
        a_cl_picard = 0.0
        c_cl_anchor_kind = "bulk"

    # Stern split for Phase E (Bug #1 fix).  When the residual is
    # Stern-aware (``bv_stern_capacitance_model > 0``), the Picard's eta
    # must use ``eta_drop = psi_S`` instead of ``phi_applied`` to match
    # the residual ``eta_raw = phi_applied - phi - E_eq``.  The returned
    # ``psi_D`` is the post-Stern-split diffuse-layer drop, which the FE
    # composite-psi profile picks up automatically (so phi(y=0) =
    # phi_applied - psi_S after the IC interpolation).
    stern_capacitance_model = scaling.get("bv_stern_capacitance_model")
    use_stern_at_ic = (
        stern_capacitance_model is not None
        and float(stern_capacitance_model) > 0
    )
    if use_stern_at_ic:
        stern_split_picard = {
            "lambda_D": lambda_D,
            "stern_coeff": float(stern_capacitance_model),
            "eps": poisson_coefficient,
        }
    else:
        stern_split_picard = None

    # ----- Run shared scalar Picard outer loop -------------------------
    # Generalized N-reaction loop with topology_hint dispatch.  For the
    # sequential template (2 rxns, rxn1 produces H₂O₂, rxn2 consumes it,
    # rxn2 irreversible) ``topology_hint='sequential_2e_h2o2'`` enables
    # the legacy closed-form P_s/O_s reconstruction (byte-equivalent to
    # the pre-M3a.3 2x2 ``picard_outer_loop`` for that topology).  All
    # other topologies (parallel 2e/4e, future N-reaction sets) use the
    # naive signed flux balance (v3 §2/§8).
    bulk_concs = [O_b, P_b, H_b]
    diffusivities = [D_O, D_P, D_H]
    species_floors = [1e-300, P_FLOOR, 1e-300]
    ok, reason, picard_iters, picard_state = picard_outer_loop_general(
        reactions=bv_reactions,
        bulk_concs=bulk_concs,
        diffusivities=diffusivities,
        species_floors=species_floors,
        h_idx=2,
        c_clo4_bulk=c_clo4_bulk,
        phi_applied_model=phi_applied_model,
        bv_exp_scale=bv_exp_scale,
        exponent_clip=exponent_clip,
        clip_exponent=clip_exponent,
        a_h=a_h_picard,
        a_cl=a_cl_picard,
        c_cl_anchor_kind=c_cl_anchor_kind,
        stern_split=stern_split_picard,
        topology_hint=topology_hint_picard,
    )
    # Stash converged scalar state for downstream diagnostics
    # (rate-consistency check; see scripts/diagnose_db_ic_distance.py
    # and Codex's verification protocol in handoff #13 response).
    ctx["initializer_picard_state"] = picard_state

    if not ok:
        return False, reason, picard_iters

    R1 = picard_state["R1"]
    R2 = picard_state["R2"]
    O_s = picard_state["O_s"]
    P_s = picard_state["P_s"]
    H_o = picard_state["H_o"]
    psi_D = picard_state["psi_D"]

    # ----- Numerically-safe Gouy-Chapman psi baseline (always built) ---
    # psi_gc(y) = 4*atanh(tanh(psi_D/4)*exp(-y/lambda_D))
    # rewritten as 2*ln((1+T*E)/(1-T*E)) to avoid atanh saturation at psi_D > ~30.
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

    # Two paths reach the bikerman-consistent IC: the legacy synthesised
    # 4sp dynamic ClO4- (z=-1, no explicit counterion config) and an
    # explicit ``boltzmann_counterions=[{steric_mode='bikerman', ...}]``
    # entry (the 3sp+bikerman production target).  Both share the same
    # closed-form gamma + composite-psi seeding; they differ only in the
    # outer-anchor for the analytic counterion in the gamma denominator.
    # ``apply_bikerman_ic`` was already computed above.
    if apply_bikerman_ic:
        a_vals_full = list(solver_params[6])
        a_h = float(a_vals_full[2])
        if synthesised_4sp_counterion:
            # 4sp dynamic: ClO4- size from a_vals[3]; outer-region
            # electroneutrality combined with phi_o = ln(H_outer/c_clo4_bulk)
            # gives c_ClO4_outer(y) = H_outer(y).
            a_cl = float(a_vals_full[3])
            c_cl_anchor = H_outer
        else:
            # 3sp + bikerman counterion: a_cl is the counterion's
            # ``a_nondim``; the outer-region anchor for the analytic
            # counterion is its bulk concentration.
            bikerman_entry = next(
                e for e in counterions
                if e.get("steric_mode", "ideal") == "bikerman"
            )
            a_cl = float(bikerman_entry["a_nondim"])
            c_cl_anchor = fd.Constant(c_clo4_bulk)

        # Composite psi: BKSA matched-asymptotic profile (saturated zone
        # near the electrode + outer exponential decay).  Falls through
        # to psi_gc when the linear-Debye approximation is sufficient
        # (|psi_D| <= psi_sat or nu = 0).  See
        # docs/4sp_bikerman_ic_option_2b_plan.md and
        # tests/test_steric_psi_profile.py for the formula derivation.
        nu_charged = 2.0 * a_cl * c_clo4_bulk
        psi_d_abs = abs(float(psi_D))
        if nu_charged <= 0.0 or psi_d_abs < 1e-6:
            psi = psi_gc
        else:
            psi_sat_val = math.log(2.0 / nu_charged)
            sign_psi_D = 1.0 if psi_D >= 0.0 else -1.0
            if psi_d_abs <= psi_sat_val * (1.0 - 1e-3):
                # Below saturation: pure linear-Debye exponential.
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

        # Multispecies Bikerman gamma (matches the sign-corrected residual
        # mu_steric = -ln(packing)).  Closed-form zero-flux equilibrium
        # gives c_i(y) = c_outer_i(y) * gamma(psi) * exp(-z_i*psi), with
        #
        #   gamma(psi) = 1 / [1 + sum_j a_j * c_anchor_j * (exp(-z_j*psi) - 1)].
        #
        # Neutral species drop out (exp(0) - 1 = 0).  c_anchor_j is
        # H_outer(y) for the dynamic-4sp ClO4- (electroneutrality), or
        # c_clo4_bulk for the analytic-3sp counterion (outer = bulk).
        gamma_psi = fd.Constant(1.0) / (
            fd.Constant(1.0)
            + fd.Constant(a_h)  * H_outer    * (fd.exp(-psi) - fd.Constant(1.0))
            + fd.Constant(a_cl) * c_cl_anchor * (fd.exp(+psi) - fd.Constant(1.0))
        )
        log_gamma = fd.ln(gamma_psi)

        U_prev.sub(0).interpolate(fd.ln(O_outer) + log_gamma)
        U_prev.sub(1).interpolate(fd.ln(P_outer) + log_gamma)
        U_prev.sub(2).interpolate(fd.ln(H_outer) - psi + log_gamma)
        U_prev.sub(n).interpolate(phi_init_expr)
        if synthesised_4sp_counterion:
            # 4sp dynamic: ClO4- is u_3 with gamma-corrected seed.
            U_prev.sub(3).interpolate(
                fd.Constant(math.log(c_clo4_bulk)) + phi_init_expr + log_gamma
            )
        # 3sp + bikerman: no dynamic ClO4-; analytic counterion enters
        # via build_steric_boltzmann_expressions.
    else:
        # 3sp + ideal counterion (legacy): byte-identical to pre-2a'.
        psi = psi_gc
        phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi
        U_prev.sub(0).interpolate(fd.ln(O_outer))
        U_prev.sub(1).interpolate(fd.ln(P_outer))
        U_prev.sub(2).interpolate(fd.ln(H_outer) - psi)
        U_prev.sub(n).interpolate(phi_init_expr)

    ctx["U"].assign(U_prev)
    return True, "", picard_iters


# ---------------------------------------------------------------------------
# Observable extraction (log-c aware)
# ---------------------------------------------------------------------------

def build_bv_observable_form_logc(ctx: dict[str, Any], *, mode: str = "current_density",
                                  reaction_index: int | None = None,
                                  scale: float = 1.0) -> Any:
    """Build observable form using log-concentration transform.

    Same interface as observables._build_bv_observable_form, but uses
    the ci_exprs (= exp(u_i)) stored in ctx by build_forms_logc.
    """
    # The BV rate expressions in ctx already use c_surf = exp(u_i),
    # so the standard observable builder works IF we pass the right context.
    # The rate expressions are already assembled in build_forms_logc.
    from .observables import _build_bv_observable_form
    return _build_bv_observable_form(ctx, mode=mode, reaction_index=reaction_index, scale=scale)
