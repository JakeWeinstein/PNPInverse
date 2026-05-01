"""Nondimensionalization helpers for Butler-Volmer BV parameters."""

from __future__ import annotations

from Nondim.constants import FARADAY_CONSTANT, GAS_CONSTANT


def _add_bv_scaling_to_transform(
    scaling: dict,
    bv_cfg: dict,
    *,
    n_species: int,
    nondim_enabled: bool,
    kappa_inputs_dimless: bool,
    concentration_inputs_dimless: bool = False,
) -> dict:
    """Augment the ModelScaling dict with BV-specific nondimensional quantities.

    Returns a new dict (does not mutate scaling).

    Parameters
    ----------
    concentration_inputs_dimless:
        If True, ``bv_bc.c_ref`` values are already dimensionless (divided by the
        concentration scale) and must NOT be scaled again.  If False (default),
        c_ref values are in physical units (mol/m³) and will be divided by
        ``concentration_scale_mol_m3``.  Mirror of ``kappa_inputs_dimless``.
    """
    kappa_scale = scaling.get("kappa_scale_m_s", 1.0)
    thermal_voltage_v = scaling.get("thermal_voltage_v", 0.02569)
    conc_scale = scaling.get("concentration_scale_mol_m3", 1.0)
    potential_scale = scaling.get("potential_scale_v", 1.0)

    k0_raw = bv_cfg["k0_vals"]
    c_ref_raw = bv_cfg["c_ref_vals"]
    E_eq_raw = bv_cfg["E_eq_v"]

    if not nondim_enabled:
        # Dimensional: keep physical values, apply F/(RT) as EM prefactor for exponents
        k0_model = k0_raw
        c_ref_model = c_ref_raw
        # Exponent in BV: exp(±α · F · η / RT).  In dimensional mode phi is in Volts.
        # The BV exponent prefactor = F/(RT).
        bv_exponent_scale = FARADAY_CONSTANT / (GAS_CONSTANT * scaling["temperature_k"])
        E_eq_model = E_eq_raw
    else:
        # Nondimensional: k0 → k0/κ_scale, c_ref → c_ref/c_scale
        # Exponent: exp(±α · η̂) where η̂ = η/V_T (dimensionless potential already)
        # Since potential is already in V_T units, exponent prefactor = 1.
        bv_exponent_scale = 1.0
        if kappa_inputs_dimless:
            k0_model = k0_raw  # already dimensionless
        else:
            k0_model = [v / kappa_scale for v in k0_raw]
        if concentration_inputs_dimless:
            c_ref_model = c_ref_raw  # already dimensionless, do not divide by c_scale
        else:
            c_ref_model = [v / conc_scale for v in c_ref_raw]
        # E_eq in thermal voltage units
        E_eq_model = E_eq_raw / potential_scale

    # Stern layer capacitance nondimensionalization.
    # Physical BC: epsilon * grad(phi).n = C_stern * (phi_m - phi)
    #
    # In the weak form, the IBP boundary integral is eps_coeff * grad(phi_hat).n.
    # The Stern BC replaces this, so the weak-form coefficient must satisfy:
    #   stern_model * (phi_m_hat - phi_hat) = eps_coeff * grad(phi_hat).n
    #   = eps_coeff * (C_stern * L / epsilon) * (phi_m_hat - phi_hat)
    #
    # In dimensional mode:
    #   eps_coeff = permittivity, boundary integral = permittivity * grad(phi).n
    #   Physical BC: permittivity * grad(phi).n = C_stern * (phi_m - phi)
    #   stern_model = C_stern  (direct physical value, used without eps_coeff)
    #
    # In nondim mode:
    #   eps_coeff = epsilon * V_T / (F * c_ref * L^2)
    #   stern_model = eps_coeff * C_stern * L / epsilon
    #              = C_stern * V_T / (F * c_ref * L)  [dimensionless]
    stern_raw = bv_cfg.get("stern_capacitance_f_m2")
    if stern_raw is not None and float(stern_raw) > 0:
        if not nondim_enabled:
            stern_model = float(stern_raw)
        else:
            length_scale = scaling.get("length_scale_m", 1.0)
            concentration_scale = scaling.get("concentration_scale_mol_m3", 1.0)
            stern_model = (
                float(stern_raw) * potential_scale
                / (FARADAY_CONSTANT * concentration_scale * length_scale)
            )
    else:
        stern_model = None

    out = dict(scaling)
    out["bv_k0_model_vals"] = k0_model
    out["bv_c_ref_model_vals"] = c_ref_model
    out["bv_exponent_scale"] = bv_exponent_scale
    out["bv_E_eq_model"] = E_eq_model
    out["bv_alpha_vals"] = bv_cfg["alpha_vals"]
    out["bv_stoichiometry"] = bv_cfg["stoichiometry"]
    out["bv_stern_capacitance_model"] = stern_model
    return out


def _add_bv_reactions_scaling_to_transform(
    scaling: dict,
    reactions: list[dict],
    *,
    nondim_enabled: bool,
    kappa_inputs_dimless: bool,
    concentration_inputs_dimless: bool = False,
    E_eq_v: float = 0.0,
) -> dict:
    """Augment ModelScaling with multi-reaction BV quantities.

    Each reaction's k0 and c_ref are nondimensionalized individually.
    """
    kappa_scale = scaling.get("kappa_scale_m_s", 1.0)
    conc_scale = scaling.get("concentration_scale_mol_m3", 1.0)
    potential_scale = scaling.get("potential_scale_v", 1.0)

    if not nondim_enabled:
        bv_exponent_scale = FARADAY_CONSTANT / (GAS_CONSTANT * scaling["temperature_k"])
        E_eq_model = E_eq_v
    else:
        bv_exponent_scale = 1.0
        E_eq_model = E_eq_v / potential_scale

    scaled_reactions = []
    for rxn in reactions:
        srxn = dict(rxn)
        if not nondim_enabled:
            srxn["k0_model"] = rxn["k0"]
            srxn["c_ref_model"] = rxn["c_ref"]
            srxn["E_eq_model"] = rxn.get("E_eq_v", 0.0)
        else:
            if kappa_inputs_dimless:
                srxn["k0_model"] = rxn["k0"]
            else:
                srxn["k0_model"] = rxn["k0"] / kappa_scale
            if concentration_inputs_dimless:
                srxn["c_ref_model"] = rxn["c_ref"]
            else:
                srxn["c_ref_model"] = rxn["c_ref"] / conc_scale
            srxn["E_eq_model"] = rxn.get("E_eq_v", 0.0) / potential_scale

        # Scale cathodic_conc_factors c_ref_nondim values
        scaled_factors = []
        for f_cfg in rxn.get("cathodic_conc_factors", []):
            sf = dict(f_cfg)
            if nondim_enabled and not concentration_inputs_dimless:
                sf["c_ref_nondim"] = f_cfg["c_ref_nondim"] / conc_scale
            else:
                sf["c_ref_nondim"] = f_cfg["c_ref_nondim"]
            scaled_factors.append(sf)
        srxn["cathodic_conc_factors"] = scaled_factors

        scaled_reactions.append(srxn)

    # Stern layer capacitance is not set via the reactions path directly;
    # it comes from the bv_cfg which is handled by build_forms.  Store None
    # so downstream code can check uniformly.
    out = dict(scaling)
    out["bv_reactions"] = scaled_reactions
    out["bv_exponent_scale"] = bv_exponent_scale
    out["bv_E_eq_model"] = E_eq_model
    if "bv_stern_capacitance_model" not in out:
        out["bv_stern_capacitance_model"] = None
    return out
