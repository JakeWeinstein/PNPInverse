"""Shared constants, scales, and configuration for BV inference scripts.

This module eliminates the ~6,400 lines of copy-pasted boilerplate across
35+ inference scripts by centralizing:

- Environment / path setup
- Physical constants (Faraday, gas constant, thermal voltage)
- Species diffusivities and bulk concentrations
- Dimensionless scaling computations
- Base SNES solver options
- SolverParams factory for 2-species neutral and 4-species charged systems
- Current-density scale (I_SCALE) computation
- ForwardRecoveryConfig factory

Usage from any script in ``scripts/*/``::

    import sys, os
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from scripts._bv_common import (
        setup_firedrake_env,
        F_CONST, V_T, N_ELECTRONS,
        TWO_SPECIES_NEUTRAL, FOUR_SPECIES_CHARGED,
        make_bv_solver_params, compute_i_scale,
        SNES_OPTS, make_recovery_config,
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

def setup_pnpinverse_env(script_file: str) -> str:
    """Add PNPInverse root to sys.path and set Firedrake cache env vars.

    Parameters
    ----------
    script_file:
        Pass ``__file__`` from the calling script.

    Returns
    -------
    str
        Absolute path to the PNPInverse root directory.
    """
    this_dir = os.path.dirname(os.path.abspath(script_file))
    # scripts are 2 levels deep: scripts/<category>/<script>.py
    root = os.path.dirname(os.path.dirname(this_dir))
    if root not in sys.path:
        sys.path.insert(0, root)
    setup_firedrake_env()
    return root


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

# Neutral-system reference scales (D_O2 = 2.10e-9 m²/s from water at 25°C)
D_O2_NEUTRAL = 2.10e-9
D_REF_NEUTRAL = D_O2_NEUTRAL
K_SCALE_NEUTRAL = D_REF_NEUTRAL / L_REF


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

# Neutral-system dimensionless values
D_O2_HAT_NEUTRAL = D_O2_NEUTRAL / D_REF_NEUTRAL  # = 1.0
D_H2O2_HAT_NEUTRAL = D_H2O2 / D_REF_NEUTRAL
K0_HAT_R1_NEUTRAL = K0_PHYS_R1 / K_SCALE_NEUTRAL
K0_HAT_R2_NEUTRAL = K0_PHYS_R2 / K_SCALE_NEUTRAL
I_SCALE_NEUTRAL = compute_i_scale(d_ref=D_REF_NEUTRAL)


# ---------------------------------------------------------------------------
# Base SNES solver options (all 7 convergence strategies)
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

# Stricter variant for multi-pH / elevated H⁺ robustness
SNES_OPTS_STRICT: Dict[str, Any] = {
    **SNES_OPTS,
    "snes_max_it": 300,
    "snes_linesearch_maxlambda": 0.3,
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


TWO_SPECIES_NEUTRAL = SpeciesConfig(
    n_species=2,
    z_vals=[0, 0],
    d_vals_hat=[D_O2_HAT_NEUTRAL, D_H2O2_HAT_NEUTRAL],
    a_vals_hat=[A_DEFAULT, A_DEFAULT],
    c0_vals_hat=[1.0, 0.0],
    stoichiometry_r1=[-1, +1],
    stoichiometry_r2=[0, -1],
    k0_legacy=[K0_HAT_R1_NEUTRAL, K0_HAT_R2_NEUTRAL],
    alpha_legacy=[ALPHA_R1, ALPHA_R2],
    stoichiometry_legacy=[-1, -1],
    c_ref_legacy=[1.0, 0.0],
)

FOUR_SPECIES_CHARGED = SpeciesConfig(
    n_species=4,
    z_vals=[0, 0, 1, -1],
    d_vals_hat=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
    a_vals_hat=[A_DEFAULT] * 4,
    c0_vals_hat=[C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
    stoichiometry_r1=[-1, +1, -2, 0],
    stoichiometry_r2=[0, -1, -2, 0],
    k0_legacy=[K0_HAT_R1] * 4,
    alpha_legacy=[ALPHA_R1] * 4,
    stoichiometry_legacy=[-1, +1, -2, 0],
    c_ref_legacy=[1.0, 0.0, 1.0, 1.0],  # H2O2 c_ref=0 matches per-reaction config
)


# ---------------------------------------------------------------------------
# BV convergence + nondim sub-configs
# ---------------------------------------------------------------------------

def _make_bv_convergence_cfg(*, softplus: bool = False) -> Dict[str, Any]:
    """Standard BV convergence config sub-dict."""
    cfg: Dict[str, Any] = {
        "clip_exponent": True,
        "exponent_clip": 50.0,
        "regularize_concentration": True,
        "conc_floor": 1e-12,
        "use_eta_in_bv": True,
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
    c_hp_hat: float = C_HP_HAT,
    electrode_marker: int = 3,
    concentration_marker: int = 4,
    ground_marker: int = 4,
) -> Dict[str, Any]:
    """Build the ``bv_bc`` config sub-dict for 2-reaction BV."""
    reaction_1: Dict[str, Any] = {
        "k0": k0_hat_r1,
        "alpha": alpha_r1,
        "cathodic_species": 0,
        "anodic_species": 1,
        "c_ref": 1.0,
        "stoichiometry": list(species.stoichiometry_r1),
        "n_electrons": N_ELECTRONS,
        "reversible": True,
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
    }

    # 4-species charged system uses cathodic_conc_factors for H⁺ dependence
    if species.n_species >= 4:
        h_factor = [{"species": 2, "power": 2, "c_ref_nondim": c_hp_hat}]
        reaction_1["cathodic_conc_factors"] = h_factor
        reaction_2["cathodic_conc_factors"] = list(h_factor)

    return {
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


# ---------------------------------------------------------------------------
# SolverParams factory
# ---------------------------------------------------------------------------

def make_bv_solver_params(
    *,
    eta_hat: float,
    dt: float,
    t_end: float,
    species: SpeciesConfig = FOUR_SPECIES_CHARGED,
    snes_opts: Optional[Dict[str, Any]] = None,
    softplus: bool = False,
    c_hp_hat: float = C_HP_HAT,
    k0_hat_r1: float = K0_HAT_R1,
    k0_hat_r2: float = K0_HAT_R2,
    alpha_r1: float = ALPHA_R1,
    alpha_r2: float = ALPHA_R2,
    electrode_marker: int = 3,
    concentration_marker: int = 4,
    ground_marker: int = 4,
) -> "SolverParams":
    """Build SolverParams for multi-species BV with graded rectangle mesh markers.

    Parameters
    ----------
    eta_hat:
        Applied overpotential (dimensionless).
    dt:
        Time step (dimensionless).
    t_end:
        End time (dimensionless).
    species:
        Species configuration preset (2-species neutral or 4-species charged).
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

    Returns
    -------
    SolverParams
    """
    from Forward.params import SolverParams

    params = dict(snes_opts or SNES_OPTS)
    params["bv_convergence"] = _make_bv_convergence_cfg(softplus=softplus)
    params["nondim"] = _make_nondim_cfg()
    params["bv_bc"] = _make_bv_bc_cfg(
        species,
        k0_hat_r1=k0_hat_r1,
        k0_hat_r2=k0_hat_r2,
        alpha_r1=alpha_r1,
        alpha_r2=alpha_r2,
        c_hp_hat=c_hp_hat,
        electrode_marker=electrode_marker,
        concentration_marker=concentration_marker,
        ground_marker=ground_marker,
    )

    c0 = list(species.c0_vals_hat)
    # Override H⁺ concentration if custom c_hp_hat provided (4-species)
    if species.n_species >= 4 and c_hp_hat != C_HP_HAT:
        c0[2] = c_hp_hat
        # Maintain electroneutrality with counterion
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
# ForwardRecoveryConfig factory
# ---------------------------------------------------------------------------

def make_recovery_config(
    *,
    max_attempts: int = 6,  # 6 (not class default 8): enough for typical BV solves, avoids wasting time on hopeless points
    max_it_only_attempts: int = 2,
    anisotropy_only_attempts: int = 1,
    tolerance_relax_attempts: int = 2,
    max_it_growth: float = 1.5,
    max_it_cap: int = 600,
    atol_relax_factor: float = 10.0,
    rtol_relax_factor: float = 10.0,
    ksp_rtol_relax_factor: float = 10.0,
    line_search_schedule: Sequence[str] = ("bt", "l2", "cp", "basic"),
    anisotropy_target_ratio: float = 3.0,
    anisotropy_blend: float = 0.5,
) -> "ForwardRecoveryConfig":
    """Build ForwardRecoveryConfig with project-standard defaults."""
    from FluxCurve.config import ForwardRecoveryConfig

    return ForwardRecoveryConfig(
        max_attempts=max_attempts,
        max_it_only_attempts=max_it_only_attempts,
        anisotropy_only_attempts=anisotropy_only_attempts,
        tolerance_relax_attempts=tolerance_relax_attempts,
        max_it_growth=max_it_growth,
        max_it_cap=max_it_cap,
        atol_relax_factor=atol_relax_factor,
        rtol_relax_factor=rtol_relax_factor,
        ksp_rtol_relax_factor=ksp_rtol_relax_factor,
        line_search_schedule=tuple(line_search_schedule),
        anisotropy_target_ratio=anisotropy_target_ratio,
        anisotropy_blend=anisotropy_blend,
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def print_params_summary() -> None:
    """Print dimensionless parameter summary for log output."""
    print(f"[params] D_ref = {D_REF:.3e} m²/s")
    print(f"[params] K_scale = {K_SCALE:.3e} m/s")
    print(f"[params] k0_hat (R1) = {K0_HAT_R1:.6f}")
    print(f"[params] k0_hat (R2) = {K0_HAT_R2:.8f}")
    print(f"[params] I_scale = {I_SCALE:.4f} mA/cm²")


def print_redimensionalized_results(
    best_k0: "np.ndarray",
    true_k0: "np.ndarray",
    *,
    best_alpha: "np.ndarray | None" = None,
    true_alpha: "np.ndarray | None" = None,
) -> None:
    """Print redimensionalized inference results."""
    import numpy as np

    best_k0_phys = best_k0 * K_SCALE
    true_k0_phys = true_k0 * K_SCALE

    print("\n=== Redimensionalized Results ===")
    print(f"K_scale = {K_SCALE:.6e} m/s")
    for i in range(len(best_k0)):
        print(f"  k0_{i+1} true={true_k0_phys[i]:.4e}  est={best_k0_phys[i]:.4e} m/s")
    rel_err = np.abs(best_k0 - true_k0) / np.maximum(np.abs(true_k0), 1e-16)
    print(f"  k0 relative error: {rel_err.tolist()}")

    if best_alpha is not None and true_alpha is not None:
        alpha_err = np.abs(best_alpha - true_alpha) / np.maximum(np.abs(true_alpha), 1e-16)
        for i in range(len(best_alpha)):
            print(f"  alpha_{i+1} true={true_alpha[i]:.6f}  est={best_alpha[i]:.6f}")
        print(f"  alpha relative error: {alpha_err.tolist()}")
