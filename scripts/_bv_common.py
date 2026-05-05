"""Shared constants, scales, and configuration for BV scripts.

After the May 2026 cleanup, this module supports only the production
log-c + Boltzmann counterion + log-rate BV stack.  It centralizes:

- Firedrake cache environment setup (``setup_firedrake_env``)
- Physical constants (Faraday, gas constant, thermal voltage)
- Species diffusivities and bulk concentrations (Mangan2025, pH 4)
- Dimensionless scaling and ``I_SCALE`` current-density conversion
- SNES solver options (``SNES_OPTS``, ``SNES_OPTS_CHARGED``)
- ``THREE_SPECIES_LOGC_BOLTZMANN`` species preset
- ``DEFAULT_CLO4_BOLTZMANN_COUNTERION`` ClO4- counterion entry
- ``make_bv_solver_params`` SolverParams factory

Usage from any script in ``scripts/*/``::

    import sys, os
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from scripts._bv_common import (
        setup_firedrake_env,
        F_CONST, V_T, N_ELECTRONS,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        make_bv_solver_params, compute_i_scale,
        SNES_OPTS, SNES_OPTS_CHARGED,
    )
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Environment / path setup
# ---------------------------------------------------------------------------

def setup_firedrake_env() -> None:
    """Set Firedrake/PyOP2 cache environment variables."""
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Physical constants (single source of truth: Nondim.constants)
# ---------------------------------------------------------------------------

from Nondim.constants import (
    FARADAY_CONSTANT as F_CONST,
    GAS_CONSTANT as R_GAS,
    DEFAULT_TEMPERATURE_K as T_REF,
)

V_T = R_GAS * T_REF / F_CONST  # Thermal voltage at 25 °C, ~0.025693 V
N_ELECTRONS = 2             # Electrons transferred per BV reaction


# ---------------------------------------------------------------------------
# Species physical properties (Mangan2025, pH 4)
# ---------------------------------------------------------------------------

# Diffusivities (m²/s)
D_O2 = 1.9e-9
D_H2O2 = 1.6e-9
D_HP = 9.311e-9     # H⁺
D_CLO4 = 1.792e-9   # ClO₄⁻

# Bulk concentrations (mol/m³)
C_O2 = 0.5
C_H2O2 = 0.0        # product, initially absent
C_HP = 0.1           # pH 4 → 1e-4 mol/L = 0.1 mol/m³
C_CLO4 = 0.1         # electroneutrality partner

# BV kinetics
K0_PHYS_R1 = 2.4e-8  # m/s, O₂ → H₂O₂
ALPHA_R1 = 0.627
K0_PHYS_R2 = 1e-9    # m/s, H₂O₂ → H₂O (irreversible)
ALPHA_R2 = 0.5


# ---------------------------------------------------------------------------
# Reference scales
# ---------------------------------------------------------------------------

L_REF = 1.0e-4              # m (100 µm)
D_REF = D_O2                # m²/s (O₂ diffusivity as reference)
C_SCALE = C_O2              # mol/m³ (O₂ bulk concentration as reference)
K_SCALE = D_REF / L_REF     # m/s (velocity scale for k0 nondimensionalization)


# ---------------------------------------------------------------------------
# Dimensionless species properties
# ---------------------------------------------------------------------------

D_O2_HAT = D_O2 / D_REF
D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT = D_HP / D_REF
D_CLO4_HAT = D_CLO4 / D_REF

C_O2_HAT = C_O2 / C_SCALE
C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT = C_HP / C_SCALE
C_CLO4_HAT = C_CLO4 / C_SCALE

K0_HAT_R1 = K0_PHYS_R1 / K_SCALE
K0_HAT_R2 = K0_PHYS_R2 / K_SCALE

# Steric (Bikerman) parameters
A_DEFAULT = 0.01


# ---------------------------------------------------------------------------
# Current density scale
# ---------------------------------------------------------------------------

def compute_i_scale(
    n_electrons: int = N_ELECTRONS,
    d_ref: float = D_REF,
    c_scale: float = C_SCALE,
    l_ref: float = L_REF,
) -> float:
    """Compute current density conversion: dimensionless rate → mA/cm².

    ``I_SCALE = n_e × F × D_ref × c_scale / L_ref × 0.1``
    """
    return n_electrons * F_CONST * d_ref * c_scale / l_ref * 0.1


I_SCALE = compute_i_scale()


# ---------------------------------------------------------------------------
# SNES solver options
# ---------------------------------------------------------------------------

SNES_OPTS: Dict[str, Any] = {
    "snes_type":                 "newtonls",
    "snes_max_it":               200,
    "snes_atol":                 1e-7,
    "snes_rtol":                 1e-10,
    "snes_stol":                 1e-12,
    "snes_linesearch_type":      "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type":                  "preonly",
    "pc_type":                   "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8":         77,
    "mat_mumps_icntl_14":        80,
}

# Charged-system variant: more SNES iterations for Poisson-coupled stiffness
SNES_OPTS_CHARGED: Dict[str, Any] = {
    **SNES_OPTS,
    "snes_max_it": 300,
}


# ---------------------------------------------------------------------------
# Species configuration presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeciesConfig:
    """Immutable species configuration for BV solver params construction."""

    n_species: int
    z_vals: List[int]
    d_vals_hat: List[float]
    a_vals_hat: List[float]
    c0_vals_hat: List[float]
    stoichiometry_r1: List[int]
    stoichiometry_r2: List[int]
    # Legacy per-species fallback for _get_bv_cfg
    k0_legacy: List[float] = field(default_factory=list)
    alpha_legacy: List[float] = field(default_factory=list)
    stoichiometry_legacy: List[int] = field(default_factory=list)
    c_ref_legacy: List[float] = field(default_factory=list)


# 3-species + analytic-Boltzmann counterion preset — the production
# log-c stack used by scripts/plot_iv_curve_unified.py and
# scripts/studies/v18_logc_lsq_inverse.py.  The fourth (ClO4-) ion is
# moved out of the dynamic Nernst--Planck system into a Boltzmann
# residual on Poisson; see _make_bv_bc_cfg(boltzmann_counterions=...).
# H2O2 is initialised at a small positive seed (1e-4) to keep ln(c0)
# finite in the log-c primary variable.
H2O2_SEED_NONDIM = 1e-4
THREE_SPECIES_LOGC_BOLTZMANN = SpeciesConfig(
    n_species=3,
    z_vals=[0, 0, 1],
    d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
    a_vals_hat=[A_DEFAULT] * 3,
    c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT],
    stoichiometry_r1=[-1, +1, -2],
    stoichiometry_r2=[0, -1, -2],
    k0_legacy=[K0_HAT_R1] * 3,
    alpha_legacy=[ALPHA_R1] * 3,
    stoichiometry_legacy=[-1, -1, -1],
    c_ref_legacy=[1.0, 0.0, 1.0],
)


# 4-species fully-dynamic preset (no Boltzmann reduction).  ClO4- is
# tracked as a 4th NP species with z=-1, mirroring how the legacy
# concentration backend handled it.  Pair with
# ``boltzmann_counterions=None`` in ``make_bv_solver_params``.  H+ and
# ClO4- have matching bulk concentration (C_HP_HAT == C_CLO4_HAT == 0.2)
# so the bulk is electroneutral.  Used by the equivalence test that
# verifies the Boltzmann reduction against the dynamic formulation.
FOUR_SPECIES_LOGC_DYNAMIC = SpeciesConfig(
    n_species=4,
    z_vals=[0, 0, 1, -1],
    d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
    a_vals_hat=[A_DEFAULT] * 4,
    c0_vals_hat=[C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT, C_CLO4_HAT],
    stoichiometry_r1=[-1, +1, -2, 0],   # ClO4- inert in R1
    stoichiometry_r2=[ 0, -1, -2, 0],   # ClO4- inert in R2
    k0_legacy=[K0_HAT_R1] * 4,
    alpha_legacy=[ALPHA_R1] * 4,
    stoichiometry_legacy=[-1, -1, -1, 0],
    c_ref_legacy=[1.0, 0.0, 1.0, 1.0],
)


# ---------------------------------------------------------------------------
# BV convergence + nondim sub-configs
# ---------------------------------------------------------------------------

def _make_bv_convergence_cfg(*, softplus: bool = False,
                              log_rate: bool = False,
                              u_clamp: float = 100.0,
                              formulation: str = "concentration",
                              initializer: str = "linear_phi") -> Dict[str, Any]:
    """Standard BV convergence config sub-dict.

    Parameters
    ----------
    softplus:
        If True, adds ``softplus_regularization`` to soften the eps_c floor.
    log_rate:
        If True, sets ``bv_log_rate=True`` so the BV residual is built in
        log-rate form (see Change 3 in WeekOfApr27/PNP Inverse Solver
        Revised.tex).  Compatible with both formulations but only useful
        in tandem with the log-c primary variable.
    u_clamp:
        Symmetric clamp on ``u_i = ln c_i`` to prevent overflow during
        Newton iteration in the log-c bulk terms.  Has no effect when
        ``formulation="concentration"``.
    formulation:
        ``"concentration"`` (default, legacy), ``"logc"`` (production), or
        ``"logc_muh"`` (experimental — proton electrochemical-potential
        primary variable; Phase 1 routes to ``logc`` with a warning).
        Selects which weak-form backend the dispatcher in
        ``Forward.bv_solver`` uses.
    """
    cfg: Dict[str, Any] = {
        "clip_exponent": True,
        # exponent_clip raised from 50.0 -> 100.0 on 2026-05-04: clip=50
        # sign-flips PC at V_RHE < -0.1 V (see clip_observable_investigation.md
        # §5.2).  At clip=100 the production V grid is fully unclipped.
        "exponent_clip": 100.0,
        "regularize_concentration": True,
        "conc_floor": 1e-12,
        "use_eta_in_bv": True,
        "bv_log_rate": log_rate,
        "u_clamp": u_clamp,
        "formulation": str(formulation).strip().lower(),
        "initializer": str(initializer).strip().lower(),
    }
    if softplus:
        cfg["softplus_regularization"] = True
    return cfg


def _make_nondim_cfg() -> Dict[str, Any]:
    """Standard nondimensionalization config sub-dict."""
    return {
        "enabled": True,
        "diffusivity_scale_m2_s": D_REF,
        "concentration_scale_mol_m3": C_SCALE,
        "length_scale_m": L_REF,
        "potential_scale_v": V_T,
        "kappa_scale_m_s": K_SCALE,
        "time_scale_s": L_REF**2 / D_REF,
        "kappa_inputs_are_dimensionless": True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless": True,
        "time_inputs_are_dimensionless": True,
    }


def _make_bv_bc_cfg(
    species: SpeciesConfig,
    *,
    k0_hat_r1: float = K0_HAT_R1,
    k0_hat_r2: float = K0_HAT_R2,
    alpha_r1: float = ALPHA_R1,
    alpha_r2: float = ALPHA_R2,
    E_eq_r1: float = 0.0,
    E_eq_r2: float = 0.0,
    c_hp_hat: float = C_HP_HAT,
    electrode_marker: int = 3,
    concentration_marker: int = 4,
    ground_marker: int = 4,
    boltzmann_counterions: Optional[Sequence[Dict[str, Any]]] = None,
    stern_capacitance_f_m2: Optional[float] = None,
    include_h_factor: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the ``bv_bc`` config sub-dict for 2-reaction BV.

    Parameters
    ----------
    boltzmann_counterions:
        Optional list of analytic-Boltzmann counterion entries for Poisson
        (Poisson-Boltzmann-Nernst-Planck reduction).  Each entry is a dict
        with keys ``z`` and ``c_bulk_nondim``.  When provided, the forms
        modules add a residual ``-charge_rhs * z * c_bulk * exp(-z*phi)``
        per entry to Poisson's equation.  See
        ``Forward/bv_solver/boltzmann.py``.
    stern_capacitance_f_m2:
        Optional compact-layer Stern capacitance in physical units
        ``F/m²`` (1 F/m² = 100 µF/cm²).  When ``None`` the cfg key is
        omitted entirely and the no-Stern Dirichlet BC ``phi_s = phi_m``
        is used (the idealised C_S → ∞ limit).  When set to a positive
        value, ``forms_logc.py`` switches to a Robin BC that lets the
        compact-layer voltage drop ``phi_m - phi_s`` be solved for, and
        the BV overpotential becomes ``eta = phi_applied - phi - E_eq``.
        A value of ``0.0`` is written through but is runtime-inactive
        (``forms_logc.py`` requires ``> 0`` to activate Stern).
    include_h_factor:
        Whether to attach H+ stoichiometric concentration factors to each
        reaction's ``cathodic_conc_factors``.  Defaults to True for
        ``species.n_species >= 3`` (works for both the 4sp charged preset
        and the 3sp logc+Boltzmann preset).
    """
    reaction_1: Dict[str, Any] = {
        "k0": k0_hat_r1,
        "alpha": alpha_r1,
        "cathodic_species": 0,
        "anodic_species": 1,
        "c_ref": 1.0,
        "stoichiometry": list(species.stoichiometry_r1),
        "n_electrons": N_ELECTRONS,
        "reversible": True,
        "E_eq_v": E_eq_r1,
    }
    reaction_2: Dict[str, Any] = {
        "k0": k0_hat_r2,
        "alpha": alpha_r2,
        "cathodic_species": 1,
        "anodic_species": None,
        "c_ref": 0.0,
        "stoichiometry": list(species.stoichiometry_r2),
        "n_electrons": N_ELECTRONS,
        "reversible": False,
        "E_eq_v": E_eq_r2,
    }

    # H+ stoichiometric concentration factor.  Apply automatically when the
    # species set has at least 3 species (i.e. tracks H+ explicitly), unless
    # the caller overrides via include_h_factor.
    if include_h_factor is None:
        attach_h = species.n_species >= 3
    else:
        attach_h = bool(include_h_factor)
    if attach_h:
        h_factor = [{"species": 2, "power": 2, "c_ref_nondim": c_hp_hat}]
        reaction_1["cathodic_conc_factors"] = h_factor
        reaction_2["cathodic_conc_factors"] = [dict(f) for f in h_factor]

    cfg: Dict[str, Any] = {
        "reactions": [reaction_1, reaction_2],
        # Legacy per-species fallback (used by _get_bv_cfg for markers)
        "k0": list(species.k0_legacy),
        "alpha": list(species.alpha_legacy),
        "stoichiometry": list(species.stoichiometry_legacy),
        "c_ref": list(species.c_ref_legacy),
        "E_eq_v": 0.0,
        "electrode_marker": electrode_marker,
        "concentration_marker": concentration_marker,
        "ground_marker": ground_marker,
    }
    if boltzmann_counterions:
        cfg["boltzmann_counterions"] = [dict(entry) for entry in boltzmann_counterions]
    if stern_capacitance_f_m2 is not None:
        cfg["stern_capacitance_f_m2"] = float(stern_capacitance_f_m2)
    return cfg


# Convenience: the standard ClO4- counterion entry that pairs with the
# 3-species log-c preset (matches the inline `add_boltzmann()` helper used
# inside scripts/studies/v18_logc_lsq_inverse.py).
DEFAULT_CLO4_BOLTZMANN_COUNTERION: Dict[str, Any] = {
    "z": -1,
    "c_bulk_nondim": C_CLO4_HAT,
    "phi_clamp": 50.0,
}


# Steric-aware variant of the ClO4- counterion: uses the Bikerman
# closure  c_b * exp(phi) * (1-A_dyn) / (theta_b + a_b * c_b * exp(phi))
# instead of the unbounded ideal Boltzmann.  Drop-in replacement for
# DEFAULT_CLO4_BOLTZMANN_COUNTERION when the residual should respect
# steric saturation at high anodic V_RHE.  See
# docs/steric_analytic_clo4_reduction_handoff.md for the derivation.
DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC: Dict[str, Any] = {
    **DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    "steric_mode": "bikerman",
    "a_nondim": A_DEFAULT,
}


# ---------------------------------------------------------------------------
# SolverParams factory
# ---------------------------------------------------------------------------

def make_bv_solver_params(
    *,
    eta_hat: float,
    dt: float,
    t_end: float,
    species: SpeciesConfig = THREE_SPECIES_LOGC_BOLTZMANN,
    snes_opts: Optional[Dict[str, Any]] = None,
    softplus: bool = False,
    c_hp_hat: float = C_HP_HAT,
    k0_hat_r1: float = K0_HAT_R1,
    k0_hat_r2: float = K0_HAT_R2,
    alpha_r1: float = ALPHA_R1,
    alpha_r2: float = ALPHA_R2,
    E_eq_r1: float = 0.0,
    E_eq_r2: float = 0.0,
    electrode_marker: int = 3,
    concentration_marker: int = 4,
    ground_marker: int = 4,
    formulation: str = "logc",
    log_rate: bool = False,
    boltzmann_counterions: Optional[Sequence[Dict[str, Any]]] = None,
    stern_capacitance_f_m2: Optional[float] = None,
    u_clamp: float = 100.0,
    initializer: str = "linear_phi",
    include_h_factor: Optional[bool] = None,
) -> "SolverParams":
    """Build SolverParams for multi-species BV with graded rectangle mesh markers.

    Parameters
    ----------
    eta_hat:
        Applied overpotential (dimensionless).
    dt, t_end:
        Time step / end time (both dimensionless).
    species:
        Species configuration preset.  Only ``THREE_SPECIES_LOGC_BOLTZMANN``
        is supported by the production stack; other presets were removed
        with the legacy concentration backend.
    snes_opts:
        Override SNES options (defaults to ``SNES_OPTS``).
    softplus:
        Enable softplus regularization in BV convergence.
    c_hp_hat:
        Nondimensional H⁺ concentration (for cathodic_conc_factors).
    k0_hat_r1, k0_hat_r2:
        Nondimensional k0 for reactions 1 and 2.
    alpha_r1, alpha_r2:
        Transfer coefficients for reactions 1 and 2.
    electrode_marker, concentration_marker, ground_marker:
        Mesh boundary markers.
    formulation:
        ``"concentration"`` (legacy), ``"logc"`` (production default), or
        ``"logc_muh"`` (experimental — proton electrochemical-potential
        primary variable; see ``docs/electrochemical_potential_solver_plan.md``).
        Selects the backend that the dispatcher in ``Forward.bv_solver``
        uses.  See the writeup
        ``writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`` for the
        formulation choices and
        ``docs/4sp_bikerman_ic_option_2b_results.md`` for the May 4
        production-target sweep that uses this backend end-to-end.
    log_rate:
        Enable log-rate Butler-Volmer evaluation (Change 3 in the
        writeup).  Compatible with both formulations but only useful in
        the log-c primary variable.
    boltzmann_counterions:
        Optional analytic-Boltzmann counterions (Change 1 in the writeup,
        the PBNP reduction).  For the standard ClO4- supporting
        electrolyte pass ``[DEFAULT_CLO4_BOLTZMANN_COUNTERION]``.
    stern_capacitance_f_m2:
        Optional compact-layer Stern capacitance, ``F/m²`` (1 F/m² = 100
        µF/cm²).  ``None`` (default) → no-Stern Dirichlet BC; positive
        value → Robin BC with finite compact-layer drop.  See
        ``docs/stern_layer_physics_and_next_steps.md``.
    u_clamp:
        Symmetric clamp on ``u_i = ln(c_i)`` in the log-c bulk forms.
    include_h_factor:
        Override the auto-detection in :func:`_make_bv_bc_cfg`.

    Returns
    -------
    SolverParams
    """
    from Forward.params import SolverParams

    params = dict(snes_opts or SNES_OPTS)
    params["bv_convergence"] = _make_bv_convergence_cfg(
        softplus=softplus, log_rate=log_rate, u_clamp=u_clamp,
        formulation=formulation, initializer=initializer,
    )
    params["nondim"] = _make_nondim_cfg()
    params["bv_bc"] = _make_bv_bc_cfg(
        species,
        k0_hat_r1=k0_hat_r1,
        k0_hat_r2=k0_hat_r2,
        alpha_r1=alpha_r1,
        alpha_r2=alpha_r2,
        E_eq_r1=E_eq_r1,
        E_eq_r2=E_eq_r2,
        c_hp_hat=c_hp_hat,
        electrode_marker=electrode_marker,
        concentration_marker=concentration_marker,
        ground_marker=ground_marker,
        boltzmann_counterions=boltzmann_counterions,
        stern_capacitance_f_m2=stern_capacitance_f_m2,
        include_h_factor=include_h_factor,
    )

    c0 = list(species.c0_vals_hat)
    # Override H⁺ concentration if custom c_hp_hat provided (3- or 4-species)
    if species.n_species >= 3 and c_hp_hat != C_HP_HAT:
        c0[2] = c_hp_hat
        # Maintain electroneutrality with counterion (only the 4-species
        # set carries an explicit ClO4- entry in c0; 3sp + Boltzmann
        # tracks the counterion analytically inside Poisson).
        if species.n_species > 3:
            c0[3] = c_hp_hat

    return SolverParams.from_list([
        species.n_species,
        1,  # FE order
        dt,
        t_end,
        list(species.z_vals),
        list(species.d_vals_hat),
        list(species.a_vals_hat),
        eta_hat,
        c0,
        0.0,  # phi0: intentionally zero for BV path — equilibrium potential is in E_eq_v of bv_bc config
        params,
    ])


