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
# C_O2 = 1.2 mol/m³ is the Ruggiero 2022 §2.4 deck-correct value at pH 5-13
# (1.1 at pH 2).  Migrated 2026-05-07 from the legacy 0.5 (M3a.2.1 in
# `docs/ruggiero_realignment_plan.md`).  Because C_SCALE = C_O2, the flip
# rescales C_HP_HAT and C_CLO4_HAT (0.2 → 0.0833) and lifts I_SCALE by
# 2.4× (0.183 → 0.44 mA/cm²).  Studies anchored at the legacy value
# (Run C, m3a0 audit) are pre-fix references; the legacy constant is
# retained below for those comparison plots only.
C_O2 = 1.2
C_O2_PHYS_LEGACY = 0.5  # mol/m³ — pre-M3a.2.1 value; do not use for new runs
C_H2O2 = 0.0        # product, initially absent
C_HP = 0.1           # pH 4 → 1e-4 mol/L = 0.1 mol/m³
C_CLO4 = 0.1         # electroneutrality partner

# BV kinetics
K0_PHYS_R1 = 2.4e-8  # m/s, O₂ → H₂O₂
ALPHA_R1 = 0.627
K0_PHYS_R2 = 1e-9    # m/s, H₂O₂ → H₂O (irreversible peroxide reduction; legacy
                     # sequential-topology rate. Per Ruggiero 2022 the deck
                     # uses parallel 2e/4e ORR — see K0_PHYS_R4E below for the
                     # parallel 4e direct-to-water rate that replaces R_2.)
ALPHA_R2 = 0.5

# Parallel 2e/4e ORR kinetics (Ruggiero 2022, M3a.2 — 2026-05-07).
# R_2e: O₂ + 2H⁺ + 2e⁻ → H₂O₂ (peroxide-producing 2e channel).
# R_4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O (direct 4e-to-water channel).  Both
# pathways consume O₂ and H⁺ but only R_2e produces free H₂O₂; the 4e
# channel proceeds via adsorbed *-OOH → *-OH → H₂O without releasing
# free peroxide (Ruggiero §1).
K0_PHYS_R2E = K0_PHYS_R1     # same rate as legacy R_1 (the 2e channel)
ALPHA_R2E = ALPHA_R1
K0_PHYS_R4E = K0_PHYS_R1     # PRIOR-SELECTED placeholder (= K0_PHYS_R1).  Per
                              # H18 V4: k0_4e is weakly identified from page-15
                              # peroxide alone; calibrate against disk current
                              # / selectivity / Tafel slope in M4.  Sweep
                              # K0_PHYS_R4E ∈ {1, 5, 10} × K0_PHYS_R1 in
                              # M3a.2 if magnitude is far off experimental.
ALPHA_R4E = 0.5              # default placeholder; revisit in M4 with Tafel.

# Physical equilibrium potentials (V vs RHE).  CLAUDE.md Hard Rule 4:
# "Use physical E_eq (R1 = 0.68 V, R2 = 1.78 V vs RHE), never E_eq = 0."
E_EQ_R1_V = 0.68
E_EQ_R2_V = 1.78
# Ruggiero §1 Eqs 1-2 — parallel 2e/4e equilibrium potentials.
# Refines the legacy E_EQ_R1_V = 0.68 V to 0.695 V for the 2e channel.
E_EQ_R2E_V = 0.695   # V vs RHE, Ruggiero 2022 §1 (2e: O₂ + 2H⁺ + 2e⁻ → H₂O₂)
E_EQ_R4E_V = 1.23    # V vs RHE, Ruggiero 2022 §1 (4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O)


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
# Parallel 2e/4e ORR nondimensional rate constants.
K0_HAT_R2E = K0_PHYS_R2E / K_SCALE
K0_HAT_R4E = K0_PHYS_R4E / K_SCALE

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
        # exponent_clip = 100.0 is the only PC-trustworthy setting:
        # at clip=50 (production until 2026-05-04) PC is fictitious
        # below V_RHE = -0.1 V (sign-flipped, 3-4 OOM off; CD is OK).
        # Do not lower this for forward runs whose PC will be compared
        # against experiment.  See docs/clipping_conventions.md.
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
    E_eq_r1: float = E_EQ_R1_V,
    E_eq_r2: float = E_EQ_R2_V,
    c_hp_hat: float = C_HP_HAT,
    electrode_marker: int = 3,
    concentration_marker: int = 4,
    ground_marker: int = 4,
    boltzmann_counterions: Optional[Sequence[Dict[str, Any]]] = None,
    stern_capacitance_f_m2: Optional[float] = None,
    include_h_factor: Optional[bool] = None,
    bv_reactions: Optional[Sequence[Dict[str, Any]]] = None,
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
    bv_reactions:
        Optional list of fully-specified reaction dicts.  When set, takes
        precedence over the legacy ``k0_hat_r{1,2}, alpha_r{1,2},
        E_eq_r{1,2}`` keyword bundle and the species ``stoichiometry_r{1,2}``
        fields — the BV residual loop iterates over this list directly.
        Each entry must provide ``k0`` (nondim), ``alpha``, ``n_electrons``,
        ``E_eq_v`` (V vs RHE), ``cathodic_species``, ``anodic_species``
        (or None), ``stoichiometry`` (per-species int list of length
        ``n_species``), ``c_ref``, ``reversible``, and optionally
        ``cathodic_conc_factors``.  Used by the parallel-2e/4e topology
        (M3a.2, 2026-05-07) to bypass the hardcoded sequential R_1/R_2
        construction; see ``PARALLEL_2E_4E_REACTIONS`` below.
    """
    if bv_reactions is not None:
        # Caller-supplied reactions list takes precedence over the legacy
        # R_1/R_2 keyword construction.  Deep-copy each entry so internal
        # mutation (e.g. nondimensionalization in
        # ``_add_bv_reactions_scaling_to_transform``) does not leak
        # back to the caller's literal.
        reactions_out = []
        for rxn in bv_reactions:
            rxn_copy = dict(rxn)
            # Deep-copy nested factor lists so they cannot share state.
            if "cathodic_conc_factors" in rxn_copy and rxn_copy["cathodic_conc_factors"]:
                rxn_copy["cathodic_conc_factors"] = [
                    dict(f) for f in rxn_copy["cathodic_conc_factors"]
                ]
            if "stoichiometry" in rxn_copy:
                rxn_copy["stoichiometry"] = list(rxn_copy["stoichiometry"])
            reactions_out.append(rxn_copy)
    else:
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

        reactions_out = [reaction_1, reaction_2]

    cfg: Dict[str, Any] = {
        "reactions": reactions_out,
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
# Parallel 2e/4e ORR reaction set (Ruggiero 2022, M3a.2 — 2026-05-07)
# ---------------------------------------------------------------------------
#
# Used by passing ``bv_reactions=PARALLEL_2E_4E_REACTIONS`` to
# ``make_bv_solver_params`` to override the legacy sequential R_1 + R_2
# topology with the deck/paper-aligned parallel 2e + 4e topology.
#
# Stoichiometry conventions (per-species int list, length n_species=3):
#   index 0 = O₂, index 1 = H₂O₂, index 2 = H⁺
#   negative = consumed at cathode, positive = produced at cathode
#
# R_2e: O₂ + 2H⁺ + 2e⁻ → H₂O₂        stoichiometry [-1, +1, -2], n_e = 2
# R_4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O        stoichiometry [-1,  0, -4], n_e = 4
#
# H+ stoichiometric concentration factor is included on each reaction
# with ``power = n_electrons`` (acid-form ORR consumes one H+ per electron).
# Ruggiero §1 also lists alkaline-form rate laws; not modeled here.
#
# k0_R4e is a PRIOR-SELECTED placeholder (= K0_HAT_R2E) — not calibrated
# to any data.  Per H18 V4: k0_4e is weakly identified from page-15
# peroxide alone; calibrate against disk current / selectivity / Tafel
# slope in M4.  Sweep K0_HAT_R4E ∈ {1, 5, 10} × K0_HAT_R2E in M3a.2 if
# magnitude is far off experimental.

PARALLEL_2E_4E_REACTIONS: List[Dict[str, Any]] = [
    {
        "k0": K0_HAT_R2E,
        "alpha": ALPHA_R2E,
        "cathodic_species": 0,           # O₂
        "anodic_species": 1,             # H₂O₂ (reverse direction)
        "c_ref": 1.0,
        "stoichiometry": [-1, +1, -2],   # O₂ consumed, H₂O₂ produced, 2 H⁺ consumed
        "n_electrons": 2,
        "reversible": True,
        "E_eq_v": E_EQ_R2E_V,            # 0.695 V vs RHE
        "cathodic_conc_factors": [
            {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
        ],
    },
    {
        "k0": K0_HAT_R4E,
        "alpha": ALPHA_R4E,
        "cathodic_species": 0,           # O₂
        "anodic_species": None,          # irreversible to water
        "c_ref": 0.0,
        "stoichiometry": [-1,  0, -4],   # O₂ consumed, H₂O₂ untouched, 4 H⁺ consumed
        "n_electrons": 4,
        "reversible": False,
        "E_eq_v": E_EQ_R4E_V,            # 1.23 V vs RHE
        "cathodic_conc_factors": [
            {"species": 2, "power": 4, "c_ref_nondim": C_HP_HAT},
        ],
    },
]


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
    E_eq_r1: float = E_EQ_R1_V,
    E_eq_r2: float = E_EQ_R2_V,
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
    bv_reactions: Optional[Sequence[Dict[str, Any]]] = None,
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
    bv_reactions:
        Optional fully-specified reactions list.  When set, takes
        precedence over the legacy ``k0_hat_r{1,2}, alpha_r{1,2},
        E_eq_r{1,2}`` keyword bundle and the species ``stoichiometry_r{1,2}``
        fields.  Used by the parallel-2e/4e topology (M3a.2,
        2026-05-07) — pass ``PARALLEL_2E_4E_REACTIONS`` for the
        Ruggiero-aligned parallel set.

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
        bv_reactions=bv_reactions,
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


# ---------------------------------------------------------------------------
# Mangan 2025 alignment — experiment metadata (M1)
# ---------------------------------------------------------------------------
#
# Every Mangan-aligned study output JSON carries a top-level
# ``experiment_metadata`` block describing how the run maps onto the deck.
# Fields that depend on M0 extraction (target curve, source authority,
# RRDE collection efficiency, acceptance tier) default to honest
# placeholders so downstream consumers can cleanly distinguish
# internal-baseline runs from deck-comparable ones.
#
# The placeholder convention is anchored in
# ``memory/project_mangan_m1_deferred_parameters.md`` and gates promotion
# of ``comparison_status`` past ``"internal_baseline_only"`` -- see that
# memory entry before overriding any default here in a study script.

@dataclass(frozen=True)
class ExperimentMetadata:
    """Provenance + comparison-status metadata for one study run.

    Sentinel values
    ---------------
    - ``source_authority="memory"`` and ``comparison_status="internal_baseline_only"``
      together flag a run as not-yet-deck-comparable.  Promotion past
      these values requires M0 extraction (digitised target curves,
      acceptance-tier decision, source citations).
    - ``N_collection`` and ``target_curve`` default to ``None``: the
      study script must set them explicitly, and a ``None`` after the
      run finishes indicates an unresolved M0 dependency.
    """

    catalyst: str                       # e.g. "generic_carbon", "CMK-3"
    geometry: str                       # "RRDE" | "RDE" | "stagnant_film"
    pH_bulk: float
    cation: Optional[str]               # None for protonic-only solvers
    anion_model: str                    # "ClO4_protonic_surrogate" | "sulfate_effective_z2" | ...
    rotation_rate_rpm: Optional[float]
    L_eff_m: Optional[float]            # Levich effective transport length
    N_collection: Optional[float]       # RRDE collection efficiency, (0, 1]
    electrolyte_model: str              # "pH_countercharge_surrogate" | "ideal_salt_pair" | ...
    comparison_status: str              # "internal_baseline_only" | "deck_proxy" | "deck_quantitative_candidate"
    source_authority: str               # "Mangan2025_deck" | "Ruggiero_manuscript" | "memory"
    target_curve: Optional[str]         # filled in M0 (digitised target identifier)
    acceptance_tier: Optional[str]      # "trend" | "semi_quant" | "quant"


def make_experiment_metadata(
    *,
    catalyst: str = "generic_carbon",
    geometry: str = "stagnant_film",
    pH_bulk: float = 4.0,
    cation: Optional[str] = None,
    anion_model: str = "ClO4_protonic_surrogate",
    rotation_rate_rpm: Optional[float] = None,
    L_eff_m: Optional[float] = None,
    N_collection: Optional[float] = None,
    electrolyte_model: str = "pH_countercharge_surrogate",
    comparison_status: str = "internal_baseline_only",
    source_authority: str = "memory",
    target_curve: Optional[str] = None,
    acceptance_tier: Optional[str] = "trend",
) -> ExperimentMetadata:
    """Construct an ExperimentMetadata describing how a study maps to deck.

    Defaults reflect the *current* solver's actual configuration:
    pH-countercharge surrogate (no real cation, no real supporting anion),
    stagnant-film geometry (no rotation, no Levich correction), no IrOx
    calibration applied to the surface-pH proxy.  Fields that depend on
    M0 extraction default to honest placeholders so downstream consumers
    can cleanly distinguish internal-baseline runs from deck-comparable
    ones.  See ``memory/project_mangan_m1_deferred_parameters.md`` for
    the placeholder convention and how it gates promotion of
    ``comparison_status``.
    """
    if pH_bulk < 0.0:
        raise ValueError(f"pH_bulk must be non-negative, got {pH_bulk}")
    if N_collection is not None and not (0.0 < N_collection <= 1.0):
        raise ValueError(
            f"N_collection must be in (0, 1] when set, got {N_collection}"
        )
    if rotation_rate_rpm is not None and rotation_rate_rpm < 0.0:
        raise ValueError(
            f"rotation_rate_rpm must be non-negative when set, got {rotation_rate_rpm}"
        )
    if L_eff_m is not None and L_eff_m <= 0.0:
        raise ValueError(
            f"L_eff_m must be positive when set, got {L_eff_m}"
        )
    return ExperimentMetadata(
        catalyst=str(catalyst),
        geometry=str(geometry),
        pH_bulk=float(pH_bulk),
        cation=cation,
        anion_model=str(anion_model),
        rotation_rate_rpm=(None if rotation_rate_rpm is None else float(rotation_rate_rpm)),
        L_eff_m=(None if L_eff_m is None else float(L_eff_m)),
        N_collection=(None if N_collection is None else float(N_collection)),
        electrolyte_model=str(electrolyte_model),
        comparison_status=str(comparison_status),
        source_authority=str(source_authority),
        target_curve=target_curve,
        acceptance_tier=acceptance_tier,
    )

