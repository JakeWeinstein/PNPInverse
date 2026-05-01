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

from .config import _get_bv_cfg, _get_bv_convergence_cfg, _get_bv_reactions_cfg
from .nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform


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
    # adequate for typical EDL profiles.  When BV evaluation drives c_H2O2
    # below 1e-13 during Newton iteration (V > +0.30 V), widen via the
    # `u_clamp` BV-convergence config option (e.g. u_clamp=100).  Note that
    # the log-rate BV path uses ui[i] directly and bypasses this clamp on the
    # boundary residual; the clamp here only affects bulk PDE terms (time
    # derivative, diffusion).
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
    })
    return ctx


def set_initial_conditions_logc(ctx: dict[str, Any], solver_params: Any) -> None:
    """Set initial conditions in log-concentration space.

    Sets u_i = ln(c0_i) for species, linear phi profile.
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

    _C_FLOOR = 1e-20
    for i in range(n):
        c0_i = max(float(c0_model[i]), _C_FLOOR)
        U_prev.sub(i).assign(fd.Constant(np.log(c0_i)))

    # Linear potential profile
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]
    U_prev.sub(n).interpolate(fd.Constant(float(phi_applied_model)) * (1.0 - spatial_var))
    ctx["U"].assign(U_prev)


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
