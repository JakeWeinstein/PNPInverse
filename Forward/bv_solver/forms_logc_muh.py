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
    _get_species_roles,
)
from .nondim import _add_bv_scaling_to_transform, _add_bv_reactions_scaling_to_transform
from .boltzmann import (
    add_boltzmann_counterion_residual,
    build_steric_boltzmann_expressions,
)
from .water_ionization import (
    build_proton_condition_flux,
    build_water_ionization_terms,
    is_water_ionization_enabled,
    resolve_h_index,
)
from .cation_hydrolysis import (
    build_cation_hydrolysis_terms,
    build_pka_shift,
    build_proton_boundary_source,
    is_cation_hydrolysis_enabled,
)
from .forms_logc import build_context_logc, set_initial_conditions_logc
from .forms_indexing import unpack_dof_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_mu_h_index(z_vals: list, *, roles=None) -> int:
    """Return the index of the proton species for the muh formulation.

    Two paths:

    * ``roles=None`` (legacy) — infer from ``z_vals``: exactly one
      species with ``z = +1`` is required.  Same fail-loudly behavior as
      ``resolve_h_index``.
    * ``roles is not None`` (Phase 6β v9 Gate 1) — pick the unique index
      whose role label is ``"proton"``.  Required for the K2SO4 stack
      where both H⁺ and K⁺ carry ``z = +1``.

    Phase 2 of the muh landing is *hybrid* -- only H+ becomes the mu
    variable; ClO4- (z = -1, when dynamic) stays as u-log-c.  Charged-species
    mu for 4sp+Bikerman is Phase 7 and lives in a separate forms file.
    """
    if roles is not None:
        if len(roles) != len(z_vals):
            raise ValueError(
                f"_resolve_mu_h_index: roles length {len(roles)} != z_vals "
                f"length {len(z_vals)}; pass one role per dynamic species."
            )
        candidates = [
            i for i, r in enumerate(roles)
            if str(r).strip().lower() == "proton"
        ]
        if not candidates:
            raise ValueError(
                "_resolve_mu_h_index: no species with role='proton' in "
                f"roles={list(roles)}."
            )
        if len(candidates) > 1:
            raise ValueError(
                "_resolve_mu_h_index: multiple species with role='proton' "
                f"in roles={list(roles)} (indices {candidates}); roles must "
                "be unique."
            )
        return candidates[0]

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
        # Stern layer capacitance (same as forms_logc.py).  Phase 6β v9
        # Gate 2: also store the phys→nondim conversion factor so the
        # C_S continuation orchestrator can reassign the live Constant
        # given a physical F/m² target.
        stern_raw = bv_cfg.get("stern_capacitance_f_m2")
        if stern_raw is not None and float(stern_raw) > 0:
            if not nondim_enabled:
                scaling["bv_stern_capacitance_model"] = float(stern_raw)
                scaling["bv_stern_phys_to_nondim_factor"] = 1.0
            else:
                from Nondim.constants import FARADAY_CONSTANT as _F
                length_scale = scaling.get("length_scale_m", 1.0)
                potential_scale_v = scaling.get("potential_scale_v", 1.0)
                concentration_scale = scaling.get("concentration_scale_mol_m3", 1.0)
                conv_factor = potential_scale_v / (
                    _F * concentration_scale * length_scale
                )
                scaling["bv_stern_capacitance_model"] = (
                    float(stern_raw) * conv_factor
                )
                scaling["bv_stern_phys_to_nondim_factor"] = float(conv_factor)
        else:
            scaling.setdefault("bv_stern_capacitance_model", None)
            scaling.setdefault("bv_stern_phys_to_nondim_factor", 1.0)
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

    # Resolve mu species (Phase 2: H+ only).  Phase 6β v9 Gate 1: thread
    # ``species_roles`` so K2SO4-style stacks with two z=+1 species (H⁺
    # + K⁺) can disambiguate via explicit role labels.
    species_roles = _get_species_roles(params, n)
    mu_h_idx = _resolve_mu_h_index(list(z_vals), roles=species_roles)
    mu_species = [mu_h_idx]

    # Split mixed function: u_i (non-mu) or mu_H (mu_h_idx) for species,
    # phi for potential.  Note: ui[mu_h_idx] is mu_H, NOT u_H.
    # Phase 6β v9 Gate 3A: use the layout-aware index helper so the
    # mixed-space slicing works for both the legacy ``species + phi``
    # layout and the Γ-augmented ``species + phi + Γ`` layout.  The
    # indices object lives on ctx (set by ``build_context_logc``).
    indices = ctx.get("mixed_space_indices") or unpack_dof_indices(
        has_gamma=False
    )
    ui = fd.split(U)[indices.species_slice]
    phi = fd.split(U)[indices.phi_index]
    ui_prev = fd.split(U_prev)[indices.species_slice]
    phi_prev = fd.split(U_prev)[indices.phi_index]   # NEW for muh: c_H_old reconstruction
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[indices.species_slice]
    w = v_tests[indices.phi_index]

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

    # Phase 6α — water self-ionization closure for the muh backend.
    # ``u_exprs[h_idx_water]`` is the muh-reconstructed log(c_H) =
    # μ_H − em·z_H·φ; pass the unclamped reconstruction to the helper so
    # the symmetric clamp on c_OH = Kw_eff·exp(-u_H) lines up with the
    # symmetric clamp on c_H = exp(u_H).
    water_ion_enabled = is_water_ionization_enabled(conv_cfg)
    if water_ion_enabled:
        h_idx_water = resolve_h_index(list(z_vals), roles=species_roles)
        if h_idx_water != mu_h_idx:
            raise ValueError(
                "enable_water_ionization=True requires the water-ionization "
                "H+ index to match the muh formulation's mu species index "
                f"(h_idx_water={h_idx_water}, mu_h_idx={mu_h_idx}); the "
                "muh stack only supports H+ as the proton-condition variable."
            )
        water_bundle = build_water_ionization_terms(
            ctx=ctx,
            conv_cfg=conv_cfg,
            z_vals=z_vals,
            u_h_unclamped=u_exprs[h_idx_water],
            u_h_unclamped_prev=u_prev_exprs[h_idx_water],
            ci_h_clamped=ci[h_idx_water],
            ci_h_prev_clamped=ci_prev[h_idx_water],
            R_space=R_space,
            u_clamp=_U_CLAMP,
            roles=species_roles,
        )
    else:
        h_idx_water = None
        water_bundle = None

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

    steric_active = (
        any(v != 0.0 for v in a_vals_list)
        or bool(steric_boltz)
        or (water_ion_enabled and float(water_bundle.a_oh_const) != 0.0)
    )
    if steric_active:
        packing_floor = float(conv_cfg.get("packing_floor", 1e-8))
        A_dyn = sum(steric_a_funcs[j] * ci[j] for j in range(n))
        if water_ion_enabled and float(water_bundle.a_oh_const) != 0.0:
            # OH⁻ Bikerman packing contribution (shadow species).
            A_dyn = A_dyn + water_bundle.a_oh_const * water_bundle.c_oh_expr
        if steric_boltz:
            # Multi-counterion: shared z_scale, sum packing_contribution
            # over every bikerman bundle.  Single-counterion case (the
            # legacy path) reduces to the same UFL algebra.
            z_scale_shared = steric_boltz[0].z_scale
            packing_total = sum(b.packing_contribution for b in steric_boltz)
            theta_inner = (
                fd.Constant(1.0) - A_dyn - z_scale_shared * packing_total
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
            # mu-species flux: D*c*grad(mu) (+ steric activity).  In muh,
            # the proton's μ_H is the primary, so ∇μ_H_ideal = ∇ui[i].
            ideal_grad = fd.grad(ui[i])
        else:
            # log-c flux: D*c*(grad(u) + em*z*grad(phi))
            drift = fd.Constant(em) * z[i] * phi
            ideal_grad = fd.grad(u_exprs[i]) + fd.grad(drift)

        if water_ion_enabled and i == h_idx_water:
            # Phase 6α — proton-condition residual on E = c_H − c_OH for
            # the muh backend.  ``ideal_grad`` here is ∇μ_H (the primary).
            mu_steric_grad = fd.grad(mu_steric) if steric_active else None
            Jflux = build_proton_condition_flux(
                bundle=water_bundle,
                D_h=D[i],
                c_h=c_i,
                ideal_grad_h=ideal_grad,
                mu_steric_grad=mu_steric_grad,
                steric_active=steric_active,
            )
            F_res += (
                (water_bundle.e_var_expr - water_bundle.e_var_prev_expr)
                / dt_const
            ) * v * dx
            F_res += fd.dot(Jflux, fd.grad(v)) * dx
            continue

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

            # k0 <= 0 OR enabled=False means disabled; skip to avoid fd.ln(0)
            # in log-rate branch. bv_k0_funcs/bv_alpha_funcs are populated
            # above so Stage-4 k0-continuation can still .assign(...) them.
            if float(rxn["k0_model"]) <= 0.0 or bool(rxn.get("enabled", True)) is False:
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
        if water_ion_enabled:
            # OH⁻ Poisson contribution (z = −1).
            F_res -= charge_rhs * (
                fd.Constant(-1.0) * water_bundle.c_oh_expr
            ) * w * dx
        if steric_boltz:
            # Bikerman counterion charge density: shared z_scale, sum
            # charge_density across every bundle.  Single-counterion
            # case reduces to the same UFL algebra as the legacy path.
            z_scale_shared = steric_boltz[0].z_scale
            charge_density_total = sum(b.charge_density for b in steric_boltz)
            F_res -= (
                z_scale_shared * charge_rhs * charge_density_total * w * dx
            )

    # Stern layer Robin BC.  Phase 6β v9 Gate 2: ``stern_coeff`` is
    # promoted to the ctx so the C_S continuation ladder in
    # ``solve_anchor_with_continuation`` can reassign it via
    # ``set_stern_capacitance_model``.
    if use_stern:
        stern_coeff = fd.Constant(float(stern_capacitance_model))
        F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
    else:
        stern_coeff = None

    # ---------------------------------------------------------------
    # Phase 6β v9 Gate 3 — cation hydrolysis (Γ as external coefficient)
    # ---------------------------------------------------------------
    cation_hydrolysis_bundle = None
    if is_cation_hydrolysis_enabled(conv_cfg):
        h_idx_for_cation = (
            h_idx_water
            if water_ion_enabled
            else resolve_h_index(list(z_vals), roles=species_roles)
        )
        cation_hydrolysis_bundle = build_cation_hydrolysis_terms(
            ctx=ctx,
            conv_cfg=conv_cfg,
            z_vals=z_vals,
            roles=species_roles,
            h_idx=h_idx_for_cation,
            R_space=R_space,
        )

        # σ_S for the Singh formula — bare Stern surface charge per R4#5
        # in **physical** C/m² (Singh's Eq. (4) is unit-specific).
        if stern_coeff is not None:
            from Nondim.constants import FARADAY_CONSTANT as _F
            length_scale = float(scaling.get("length_scale_m", 1.0))
            concentration_scale = float(
                scaling.get("concentration_scale_mol_m3", 1.0)
            )
            sigma_phys_per_nondim = float(_F) * concentration_scale * length_scale
            sigma_S_expr = (
                stern_coeff
                * (phi_applied_func - phi)
                * fd.Constant(sigma_phys_per_nondim)
            )
        else:
            sigma_S_expr = fd.Constant(0.0)
        ctx["_cation_hydrolysis_sigma_S_expr"] = sigma_S_expr

        # On the muh backend the proton primary is mu_H, but the
        # boundary concentration we want is c_H = exp(u_exprs[mu_h_idx])
        # which is ci[mu_h_idx] — same convention as Phase 6α.
        c_M_bdy_expr = ci[cation_hydrolysis_bundle.counterion_idx]
        c_H_bdy_expr = ci[h_idx_for_cation]

        # Phase 6β step 6 plumbing-ablation flags — see
        # Forward/bv_solver/config.py for cross-validation rules
        # (manufactured_R_inj required when half-physical;
        # override_sigma_singh_counts_pm2 mutually exclusive with
        # manufactured_R_inj).
        manufactured_R_inj = conv_cfg.get("manufactured_R_inj", None)
        apply_h_source = bool(conv_cfg.get("apply_h_source", True))
        apply_k_sink = bool(conv_cfg.get("apply_k_sink", True))
        sigma_singh_override = conv_cfg.get(
            "override_sigma_singh_counts_pm2", None
        )

        if sigma_singh_override is not None:
            # σ_singh_post_clamp = max(0, -σ_S_signed) · 6.2415e-6.
            # To force σ_singh_post_clamp = override (counts/pm²),
            # pass σ_S_signed = -override / 6.2415e-6 (C/m², negative
            # ⇒ cathodic) so the existing anode clamp + conversion
            # in build_pka_shift collapse to the override.
            inv_factor_C_m2_per_count_pm2 = 1.602176634e-19 / 1.0e-24
            fake_signed_sigma_S = fd.Constant(
                -float(sigma_singh_override) * inv_factor_C_m2_per_count_pm2
            )
            pka_shift_expr = build_pka_shift(
                cation_params=cation_hydrolysis_bundle.cation_params,
                sigma_S=fake_signed_sigma_S,
                r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
                beta_offset_pm2_func=cation_hydrolysis_bundle.beta_offset_pm2_func,
            )
            pka_sigma_S_for_storage = fake_signed_sigma_S
        else:
            pka_shift_expr = build_pka_shift(
                cation_params=cation_hydrolysis_bundle.cation_params,
                sigma_S=sigma_S_expr,
                r_H_El_func=cation_hydrolysis_bundle.r_H_El_pm_func,
                beta_offset_pm2_func=cation_hydrolysis_bundle.beta_offset_pm2_func,
            )
            pka_sigma_S_for_storage = sigma_S_expr

        if manufactured_R_inj is not None:
            R_net = fd.Constant(float(manufactured_R_inj))
        else:
            R_net = build_proton_boundary_source(
                bundle=cation_hydrolysis_bundle,
                c_M_bdy_expr=c_M_bdy_expr,
                c_H_bdy_expr=c_H_bdy_expr,
                pka_shift_expr=pka_shift_expr,
            )

        # Proton + cation boundary residuals — wrapped in
        # ``λ_hydrolysis`` so λ=0 byte-zeros every hydrolysis
        # contribution.  Combined with ``bundle.gamma_func = 0`` at
        # λ=0 this restores the disabled-feature baseline up to
        # discretisation error.  Step 6 stages each term as a local
        # so they can be stored on ctx for diagnostic / consistency
        # checks (single source of truth — residual + diagnostics
        # consume the same UFL expression object).
        lam_func = cation_hydrolysis_bundle.lambda_hydrolysis_func
        H_residual_term = (
            lam_func * R_net
            * v_list[h_idx_for_cation]
            * ds(electrode_marker)
        )
        K_residual_term = (
            lam_func * (-R_net)
            * v_list[cation_hydrolysis_bundle.counterion_idx]
            * ds(electrode_marker)
        )
        R_net_scalar_form = lam_func * R_net * ds(electrode_marker)
        H_flux_scalar_form = R_net_scalar_form
        K_flux_scalar_form = lam_func * (-R_net) * ds(electrode_marker)

        if apply_h_source:
            F_res -= H_residual_term
        if apply_k_sink:
            F_res -= K_residual_term

        # Step 6 canonical artifacts — single source of truth for
        # diagnostics, Picard, and slow slot-wiring tests.  Inserted
        # into ctx via the ctx.update block below.
        _step6_cation_hydrolysis_artifacts = {
            "_cation_hydrolysis_R_net_expr": R_net,
            "_cation_hydrolysis_pka_shift_expr": pka_shift_expr,
            "_cation_hydrolysis_pka_sigma_S_expr": pka_sigma_S_for_storage,
            "_cation_hydrolysis_H_residual_term": H_residual_term,
            "_cation_hydrolysis_K_residual_term": K_residual_term,
            "_cation_hydrolysis_R_net_scalar_form": R_net_scalar_form,
            "_cation_hydrolysis_H_flux_scalar_form": H_flux_scalar_form,
            "_cation_hydrolysis_K_flux_scalar_form": K_flux_scalar_form,
        }
    else:
        _step6_cation_hydrolysis_artifacts = {}

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
        # Phase 6β v9 Gate 2: live FE Constant for the Stern capacitance
        # in nondim units.  None when use_stern is False.  Allows
        # set_stern_capacitance_model() to update the residual without a
        # form rebuild.
        "stern_coeff_const": stern_coeff,
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
        "water_ionization": water_bundle,  # None when feature disabled
        "cation_hydrolysis": cation_hydrolysis_bundle,  # None when feature disabled
    })
    # Phase 6β step 6 — canonical cation-hydrolysis artifacts.
    ctx.update(_step6_cation_hydrolysis_artifacts)
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
      phi:              phi_init(y) = phi_surface * (1 - y)

    Pointwise reconstruction of c_H from these IC fields recovers
    c_H_bulk exactly when phi_o_bulk = 0:

      exp(mu_H(y) - em*z_H*phi(y)) = c_H_bulk * exp(-em*z_H*(phi(y) - phi_init(y)))

    For Stern-aware configs, ``phi_surface = phi_applied - psi_S``
    instead of ``phi_applied`` (Phase F fix; see forms_logc.py
    counterpart for full rationale).
    """
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
    em = float(scaling.get("electromigration_prefactor", 1.0))

    species_roles = _get_species_roles(params, n)
    mu_h_idx = _resolve_mu_h_index(list(z_vals), roles=species_roles)
    mu_species = {mu_h_idx}

    # Stern-aware anchoring (Phase F).
    stern_capacitance_model = scaling.get("bv_stern_capacitance_model")
    use_stern_at_ic = (
        stern_capacitance_model is not None
        and float(stern_capacitance_model) > 0
    )
    phi_surface = float(phi_applied_model)
    if use_stern_at_ic and n >= 3 and len(c0_model) >= 3:
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
            _, _, phi_surface = solve_stern_split(
                phi_applied_model=float(phi_applied_model),
                phi_o=phi_o_bulk,
                lambda_D=lambda_D_bulk,
                c_clo4_bulk=c_clo4_bulk,
                a_cl=a_cl_bulk,
                stern_coeff_nondim=float(stern_capacitance_model),
                eps_nondim=poisson_coefficient,
            )

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()
    if ndim == 1:
        spatial_var = coords[0]
    else:
        spatial_var = coords[1]

    # ``y_norm = y / domain_height_hat`` so phi_init lands at 0 at the
    # mesh top regardless of the L_eff sweep extent.
    bv_conv = params.get("bv_convergence", {}) if isinstance(params, dict) else {}
    domain_height_hat = float(bv_conv.get("domain_height_hat", 1.0))
    y_norm = spatial_var / fd.Constant(domain_height_hat)

    phi_init_expr = fd.Constant(phi_surface) * (1.0 - y_norm)

    _C_FLOOR = 1e-20
    for i in range(n):
        c0_i = max(float(c0_model[i]), _C_FLOOR)
        if i in mu_species:
            # mu_H_init(y) = ln(c0_H) + em*z_H*phi_init(y)
            # -> reconstructs c_H = c0_H pointwise (when bulk phi=0).
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

    Wraps the shared scalar Picard outer loop in
    ``Forward.bv_solver.picard_ic.picard_outer_loop`` -- the scalar
    Picard algebra is byte-identical to the logc variant.  This wrapper
    handles muh-specific FE assignment for the proton species:

        logc:   U_prev.sub(mu_h_idx).interpolate(ln(H_outer) - psi)        # u_H
        muh:    U_prev.sub(mu_h_idx).interpolate(
                    (ln(H_outer) - psi)                                    # u_H
                    + em*z_H * (ln(H_outer/c_clo4_bulk) + psi)             # em*z_H*phi_init
                )                                                          # = mu_H_init

    With em*z_H = 1, ``psi`` cancels and ``mu_H_init = 2*ln(H_outer) -
    ln(c_clo4_bulk)``.
    """
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
    # ``docs/picard_general_topology_derivation.md`` v3): see
    # ``forms_logc.py:_try_debye_boltzmann_ic`` for the rationale.
    # Sequential template uses the legacy closed-form P_s/O_s
    # reconstruction (byte-equivalent to the pre-M3a.3 2x2 path);
    # parallel and other N-reaction topologies use the naive signed
    # flux balance (v3 §2/§8).
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
    em = float(scaling.get("electromigration_prefactor", 1.0))

    z_vals_full = list(solver_params[4])
    species_roles = _get_species_roles(params, n)
    mu_h_idx = _resolve_mu_h_index(z_vals_full[:n], roles=species_roles)
    z_h_val = float(z_vals_full[mu_h_idx])

    D_model_vals = [float(v) for v in scaling.get("D_model_vals", [1.0] * n)]
    if np.isscalar(c0):
        c0_raw = [float(c0)] * n
    else:
        c0_raw = [float(v) for v in list(c0)][:n]
    c0_model = [max(float(v), 1e-300) for v in scaling.get("c0_model_vals", c0_raw)]

    # ----- Locate the Boltzmann counterion bulk concentration ---------
    # Phase 6β v9 Gate 1: when ``species_roles`` are provided, use them
    # to locate the dynamic counterion (necessary for K2SO4-style stacks
    # where K⁺ at idx 3 has z=+1, not z=-1).  Without roles, fall back
    # to the legacy ``z=-1`` check at idx 3 (preserves the ClO₄⁻ stack).
    counterions = _get_bv_boltzmann_counterions_cfg(params)
    synthesised_4sp_counterion = False
    if not counterions:
        if n == 4 and len(z_vals_full) >= 4 and len(c0_model) >= 4:
            idx_counterion = None
            if species_roles is not None:
                role_matches = [
                    i for i, r in enumerate(species_roles)
                    if str(r).strip().lower() == "counterion"
                ]
                if len(role_matches) == 1:
                    idx_counterion = role_matches[0]
            if idx_counterion is None and int(z_vals_full[3]) == -1:
                idx_counterion = 3
            if idx_counterion is not None and 0 <= idx_counterion < n:
                counterions = [{
                    "z": int(z_vals_full[idx_counterion]),
                    "c_bulk_nondim": float(c0_model[idx_counterion]),
                }]
                synthesised_4sp_counterion = True
    if not counterions:
        return False, "no_boltzmann_counterion", 0

    # Phase 2 hybrid muh -- proton at index 2 (production 3sp+Boltzmann)
    # and matched-asymptotic Picard layout (O2, H2O2, H+ at indices
    # 0, 1, 2) is required.  Reject configurations where mu_h_idx != 2,
    # or fall through to logc IC.  The orchestrator's fallback handles
    # this gracefully.
    #
    # Phase 6β v9 Gate 1: K2SO4 stacks with explicit ``species_roles``
    # preserve the species ordering (O₂, H₂O₂, H⁺, K⁺), so mu_h_idx == 2
    # holds and the reject does not fire.  Lifting the constraint to
    # arbitrary proton positions requires refactoring the Picard's
    # hardcoded h_idx=2 — out of Gate 1/2 scope.
    if mu_h_idx != 2:
        return False, f"mu_h_idx_unsupported ({mu_h_idx} != 2)", 0

    O_b = c0_model[0]
    P_b = c0_model[1]
    H_b = c0_model[2]
    D_O = max(D_model_vals[0], 1e-30)
    D_P = max(D_model_vals[1], 1e-30)
    D_H = max(D_model_vals[2], 1e-30)
    # P_FLOOR: 1e-30 (numerical floor only).  See forms_logc.py
    # counterpart for the rate-consistency rationale.
    P_FLOOR = 1e-30

    c_clo4_bulk = max(float(counterions[0]["c_bulk_nondim"]), 1e-300)

    bv_exp_scale = float(scaling.get("bv_exponent_scale", 1.0))
    exponent_clip = float(conv_cfg.get("exponent_clip", 100.0))
    clip_exponent = bool(conv_cfg.get("clip_exponent", True))

    # Per-reaction scalars are read directly from ``bv_reactions`` by
    # ``picard_outer_loop_general``; M3a.2 unpacking removed (per v3
    # §9 contract item 2).  See ``forms_logc.py`` counterpart.

    # Two paths reach the bikerman-consistent IC.
    bikerman_in_counterions = bool(counterions) and any(
        e.get("steric_mode", "ideal") == "bikerman" for e in counterions
    )
    apply_bikerman_ic = synthesised_4sp_counterion or bikerman_in_counterions

    # Bikerman size parameters for the Picard's surface gamma; same
    # branching as forms_logc.py.  ``a_h = a_cl = 0`` -> legacy
    # gamma-free Picard.
    if apply_bikerman_ic:
        a_vals_full_for_picard = list(solver_params[6])
        a_h_picard = float(a_vals_full_for_picard[2])
        if synthesised_4sp_counterion:
            a_cl_picard = float(a_vals_full_for_picard[3])
            c_cl_anchor_kind = "synthesised_4sp"
        else:
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

    # Stern split (Bug #1 fix); see forms_logc.py counterpart for the
    # rationale.  The post-Stern psi_D returned by picard_outer_loop
    # automatically gets picked up by the BKSA composite-psi profile,
    # so phi(y=0) = phi_applied - psi_S after IC interpolation.  The
    # muh formulation's mu_h_init = u_h_init + em*z_H*phi_init inherits
    # the shift via phi_init (psi cancellation algebra still holds).
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
    # Generalized N-reaction loop with topology_hint dispatch.  See
    # forms_logc.py counterpart and v3 §9 for the contract.
    bulk_concs = [O_b, P_b, H_b]
    diffusivities = [D_O, D_P, D_H]
    species_floors = [1e-300, P_FLOOR, 1e-300]

    # Phase 5α T7: when 2+ bikerman counterions are configured (Cs⁺ +
    # SO₄²⁻ etc.), build the multi-ion ctx BEFORE the Picard call and
    # pass it as ``multi_ion_ctx`` so the helpers ``_solve_phi_o``,
    # ``_compute_picard_gamma_s``, ``_solve_picard_stern_split`` each
    # use the multi-ion shared-theta closure.
    bikerman_entries_pre = [
        e for e in (counterions or [])
        if e.get("steric_mode", "ideal") == "bikerman"
    ]
    if len(bikerman_entries_pre) > 1:
        from .multi_ion import build_counterion_ctx as _build_counterion_ctx_pre
        a_vals_full_pre = list(solver_params[6])
        ctx_mion_pre = _build_counterion_ctx_pre(
            counterions=counterions,
            a_dyn=[float(v) for v in a_vals_full_pre[:n]],
            c_dyn_bulk=[float(O_b), float(P_b), float(H_b)],
            z_dyn=[int(v) for v in z_vals_full[:n]],
        )
    else:
        ctx_mion_pre = None

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
        topology_hint=topology_hint_picard,
        a_h=a_h_picard,
        a_cl=a_cl_picard,
        c_cl_anchor_kind=c_cl_anchor_kind,
        stern_split=stern_split_picard,
        multi_ion_ctx=ctx_mion_pre,
        poisson_coefficient=poisson_coefficient,
    )
    ctx["initializer_picard_state"] = picard_state

    if not ok:
        return False, reason, picard_iters

    # R1/R2 historically read here but never used downstream; tolerate
    # single-reaction picard_state (no "R2" key) so single-Tafel setups
    # (Jithin §4.8 emulation) can use this IC path.
    R1 = picard_state.get("R1", 0.0)
    R2 = picard_state.get("R2", 0.0)
    O_s = picard_state["O_s"]
    P_s = picard_state["P_s"]
    H_o = picard_state["H_o"]
    psi_D = picard_state["psi_D"]

    # ----- Numerically-safe Gouy-Chapman psi baseline (always built) ---
    EPS_TANH = 1e-15
    T = math.tanh(psi_D / 4.0)
    T_clamp = math.copysign(min(abs(T), 1.0 - EPS_TANH), T)

    coords = fd.SpatialCoordinate(mesh)
    ndim = mesh.geometric_dimension()
    y = coords[0] if ndim == 1 else coords[1]

    # ``y_norm = y / domain_height_hat`` (∈ [0, 1]) drives the outer
    # surface→bulk linear interpolation so the bulk anchor lands at the
    # mesh top regardless of the L_eff sweep mesh extent.  Debye-layer
    # terms keep using ``y`` directly (lambda_D shares its nondim units).
    domain_height_hat = float(conv_cfg.get("domain_height_hat", 1.0))
    y_norm = y / fd.Constant(domain_height_hat)

    E_expr = fd.exp(-y / fd.Constant(lambda_D))
    arg = fd.Constant(T_clamp) * E_expr
    arg_safe = fd.min_value(
        fd.max_value(arg, fd.Constant(-1.0 + EPS_TANH)),
        fd.Constant(1.0 - EPS_TANH),
    )
    psi_gc = fd.Constant(2.0) * fd.ln(
        (fd.Constant(1.0) + arg_safe) / (fd.Constant(1.0) - arg_safe)
    )

    O_outer = fd.Constant(O_s) + (fd.Constant(O_b) - fd.Constant(O_s)) * y_norm
    P_outer = fd.max_value(
        fd.Constant(P_s) + (fd.Constant(P_b) - fd.Constant(P_s)) * y_norm,
        fd.Constant(P_FLOOR),
    )
    H_outer = fd.max_value(
        fd.Constant(H_o) + (fd.Constant(H_b) - fd.Constant(H_o)) * y_norm,
        fd.Constant(1e-300),
    )

    # ----- Multi-ion IC branch (plan §2.4) -------------------------------
    # When 2+ bikerman counterions are configured (Cs⁺ + SO₄²⁻ etc.), the
    # 1:1 BKSA composite-psi closure no longer applies; use the multi-ion
    # shared-theta machinery in ``Forward.bv_solver.multi_ion``.  Mirrors
    # forms_logc.py multi-ion branch with the muh-specific
    # ``mu_H = u_H + em*z_H*phi`` accumulation for the proton.
    bikerman_entries_in_counterions = [
        e for e in counterions
        if e.get("steric_mode", "ideal") == "bikerman"
    ]
    multi_ion_mode = len(bikerman_entries_in_counterions) > 1
    if multi_ion_mode:
        from .multi_ion import (
            build_counterion_ctx, solve_outer_phi_multiion,
            effective_debye_length_local,
        )
        a_vals_full_mion = list(solver_params[6])
        a_dyn_for_ctx = [float(v) for v in a_vals_full_mion[:n]]
        c_dyn_bulk_for_ctx = [float(O_b), float(P_b), float(H_b)]
        z_dyn_for_ctx = [int(v) for v in z_vals_full[:n]]
        ctx_mion = build_counterion_ctx(
            counterions=counterions,
            a_dyn=a_dyn_for_ctx,
            c_dyn_bulk=c_dyn_bulk_for_ctx,
            z_dyn=z_dyn_for_ctx,
        )
        c_dyn_outer_at_ohp = [float(O_s), float(P_s), float(H_o)]
        try:
            phi_o_local = solve_outer_phi_multiion(
                ctx=ctx_mion, c_dyn_outer=c_dyn_outer_at_ohp,
            )
        except ValueError as e:
            return False, f"multi_ion_phi_o_solve_failed: {e}", picard_iters
        lambda_eff = effective_debye_length_local(
            phi_o=phi_o_local, ctx=ctx_mion,
            c_dyn_outer=c_dyn_outer_at_ohp,
            poisson_coeff=poisson_coefficient,
        )
        full_drop = float(phi_applied_model) - phi_o_local
        if use_stern_at_ic:
            stern_coeff_val = float(stern_capacitance_model)
            psi_D_local = (
                stern_coeff_val * full_drop * lambda_eff
                / (poisson_coefficient + stern_coeff_val * lambda_eff)
            )
        else:
            psi_D_local = full_drop

        psi = fd.Constant(psi_D_local) * fd.exp(-y / fd.Constant(lambda_eff))
        phi_outer = fd.Constant(phi_o_local) * (
            fd.Constant(1.0) - fd.min_value(y_norm, fd.Constant(1.0))
        )
        phi_init_expr = phi_outer + psi

        A_dyn_outer = sum(
            a * c for a, c in zip(a_dyn_for_ctx, c_dyn_outer_at_ohp)
        )
        denom_outer = ctx_mion["theta_b"] + sum(
            ion["a_nondim"] * ion["c_bulk_nondim"]
            * math.exp(-ion["z"] * phi_o_local)
            for ion in ctx_mion["ions"]
            if ion.get("steric_mode", "ideal") == "bikerman"
        )
        theta_outer_const = max(
            (1.0 - A_dyn_outer) * ctx_mion["theta_b"] / max(denom_outer, 1e-300),
            1e-30,
        )
        c_outer_y_list = [O_outer, P_outer, H_outer]
        A_dyn_y = sum(
            fd.Constant(a) * c_y for a, c_y in zip(a_dyn_for_ctx, c_outer_y_list)
        )
        denom_y = fd.Constant(ctx_mion["theta_b"]) + sum(
            fd.Constant(ion["a_nondim"] * ion["c_bulk_nondim"])
            * fd.exp(fd.Constant(-float(ion["z"])) * (phi_outer + psi))
            for ion in ctx_mion["ions"]
            if ion.get("steric_mode", "ideal") == "bikerman"
        )
        theta_y = (
            (fd.Constant(1.0) - A_dyn_y)
            * fd.Constant(ctx_mion["theta_b"]) / denom_y
        )
        gamma_psi = theta_y / fd.Constant(theta_outer_const)
        log_gamma = fd.ln(fd.max_value(gamma_psi, fd.Constant(1e-30)))

        # Spatial IC for each species — muh formulation stores H+ as
        # mu_H = u_H + em*z_H*phi (with em*z_H=1, the -psi shift in u_H
        # cancels the +psi in phi inside mu_H, leaving log_gamma propagated).
        z_dyn_int = [int(v) for v in z_vals_full[:n]]
        for i in range(n):
            log_c_outer_i = fd.ln(c_outer_y_list[i])
            u_i_init = (
                log_c_outer_i
                - fd.Constant(float(z_dyn_int[i])) * psi
                + log_gamma
            )
            if i == mu_h_idx:
                # H+ stored as mu_H = u_H + em*z_H*phi.
                mu_h_init = (
                    u_i_init
                    + fd.Constant(em * z_h_val) * phi_init_expr
                )
                U_prev.sub(i).interpolate(mu_h_init)
            else:
                U_prev.sub(i).interpolate(u_i_init)
        U_prev.sub(n).interpolate(phi_init_expr)
        ctx["U"].assign(U_prev)
        return True, "", picard_iters

    # Bikerman-consistent IC paths (synthesised 4sp ClO4-, or explicit
    # ``boltzmann_counterions=[{steric_mode='bikerman', ...}]``).  Mirrors
    # forms_logc.py:_try_debye_boltzmann_ic; the only muh-specific bit is
    # the proton stored as mu_H = u_H + em*z_H*phi (psi cancels with em*z_H = 1,
    # log_gamma propagates through u_H into mu_H).
    # ``apply_bikerman_ic`` was already computed above.
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
            # Phase 6β v9 Gate 2: when the explicit Bikerman counterion
            # is analytic (e.g. SO₄²⁻), seed any extra dynamic species
            # at idx 3+ with their own Boltzmann-anchored bulk profile.
            # Otherwise U.sub(3+) stays at zero ⇒ c_i(IC) = 1.0 nondim,
            # which is wildly off-bulk for the K2SO4 stack (c_K_bulk ≈ 167).
            # Treats them as inert dynamic species (no BV depletion ⇒ outer
            # = bulk), with the Boltzmann anchor c_i(y) = c_i_bulk · γ(y) ·
            # exp(-z_i · phi(y)).
            for i_extra in range(3, n):
                c0_i = max(float(c0_model[i_extra]), 1e-300)
                z_i_extra = float(z_vals_full[i_extra])
                U_prev.sub(i_extra).interpolate(
                    fd.Constant(math.log(c0_i))
                    - fd.Constant(z_i_extra) * phi_init_expr
                    + log_gamma
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
