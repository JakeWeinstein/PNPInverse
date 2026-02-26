"""Physical reference scale computation for PNP nondimensionalization.

Given bulk physical parameters (diffusivities, concentration, length, temperature,
permittivity), :func:`build_physical_scales` computes the characteristic scales
needed to nondimensionalize the PNP equations.

Typical usage
-------------
>>> scales = build_physical_scales(
...     d_species_m2_s=(9.311e-9, 5.273e-9),
...     c_bulk_m=0.1,            # mol/L
...     length_scale_m=1.0e-4,   # 100 µm
... )
>>> print(f"Debye length: {scales.debye_length_m:.3e} m")
>>> print(f"Time scale:   {scales.time_s:.3e} s")

The :class:`NondimScales` dataclass returned here feeds directly into
:func:`Nondim.transform.build_model_scaling` via the ``nondim`` key in
``solver_params[10]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from Nondim.constants import (
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    DEFAULT_TEMPERATURE_K,
    VACUUM_PERMITTIVITY_F_PER_M,
    DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    MOLAR_TO_MOL_PER_M3,
)


@dataclass
class NondimScales:
    """All physical reference scales derived from bulk parameters.

    Attributes
    ----------
    d_species_m2_s:
        Per-species diffusivities [m²/s] used to compute ``d_ref_m2_s``.
    d_ref_m2_s:
        Reference diffusivity (geometric mean of species diffusivities) [m²/s].
    bulk_concentration_mol_m3:
        Bulk concentration converted to mol/m³.
    c_inf_mol_m3:
        Boundary (infinity) concentration in mol/m³.
    temperature_k:
        Temperature [K].
    thermal_voltage_v:
        Thermal voltage RT/F [V].
    permittivity_f_m:
        Absolute permittivity ε = ε_r · ε_0 [F/m].
    length_scale_m:
        Reference length L_ref [m].
    diffusivity_scale_m2_s:
        Equal to ``d_ref_m2_s`` [m²/s].
    concentration_scale_mol_m3:
        Equal to ``bulk_concentration_mol_m3`` [mol/m³].
    time_s:
        Diffusion time scale L²/D [s].
    kappa_m_s:
        Transfer-coefficient scale D/L [m/s].
    molar_flux_mol_m2_s:
        Molar flux scale D·c/L [mol/(m²·s)].
    current_density_a_m2:
        Current-density scale F·D·c/L [A/m²].
    debye_length_m:
        Debye screening length λ_D [m].
    debye_ratio:
        λ_D / L_ref (dimensionless).
    """

    d_species_m2_s: list
    d_ref_m2_s: float
    bulk_concentration_mol_m3: float
    c_inf_mol_m3: float
    temperature_k: float
    thermal_voltage_v: float
    permittivity_f_m: float
    length_scale_m: float
    diffusivity_scale_m2_s: float
    concentration_scale_mol_m3: float
    time_s: float
    kappa_m_s: float
    molar_flux_mol_m2_s: float
    current_density_a_m2: float
    debye_length_m: float
    debye_ratio: float

    def to_dict(self) -> dict:
        """Return a plain dict (useful for logging / CSV export)."""
        return {
            "d_species_m2_s": self.d_species_m2_s,
            "d_ref_m2_s": self.d_ref_m2_s,
            "bulk_concentration_mol_m3": self.bulk_concentration_mol_m3,
            "c_inf_mol_m3": self.c_inf_mol_m3,
            "temperature_k": self.temperature_k,
            "thermal_voltage_v": self.thermal_voltage_v,
            "permittivity_f_m": self.permittivity_f_m,
            "length_scale_m": self.length_scale_m,
            "diffusivity_scale_m2_s": self.diffusivity_scale_m2_s,
            "concentration_scale_mol_m3": self.concentration_scale_mol_m3,
            "time_scale_s": self.time_s,
            "kappa_scale_m_s": self.kappa_m_s,
            "molar_flux_scale_mol_m2_s": self.molar_flux_mol_m2_s,
            "current_density_scale_a_m2": self.current_density_a_m2,
            "current_density_scale_a_cm2": self.current_density_a_m2 / 1.0e4,
            "debye_length_m": self.debye_length_m,
            "debye_to_length_ratio": self.debye_ratio,
        }


def build_physical_scales(
    *,
    d_species_m2_s: Sequence[float] = (9.311e-9, 5.273e-9),
    c_bulk_m: float = 0.1,
    c_inf_m: float = 0.01,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    relative_permittivity: float = DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    length_scale_m: float = 1.0e-4,
) -> NondimScales:
    """Compute physical reference scales for PNP nondimensionalization.

    Parameters
    ----------
    d_species_m2_s:
        Per-species diffusivities in m²/s (typically 2 species).
        Reference diffusivity is taken as the geometric mean.
    c_bulk_m:
        Bulk concentration in **mol/L** (molar).  Converted internally to mol/m³.
    c_inf_m:
        Boundary ("infinity") concentration in mol/L.
    temperature_k:
        Temperature in Kelvin.
    relative_permittivity:
        Dimensionless relative permittivity of the solvent (ε_r).
    length_scale_m:
        Reference length L_ref in metres.

    Returns
    -------
    NondimScales
        Dataclass holding all derived reference scales.

    Characteristic scales derived
    -----------------------------
    - D_ref    = geometric mean of d_species_m2_s
    - c_ref    = c_bulk_m * 1000 mol/m³
    - V_T      = RT/F  (thermal voltage, ~25.7 mV at 25 °C)
    - t_ref    = L² / D_ref
    - κ_ref    = D_ref / L
    - j_ref    = F · D_ref · c_ref / L  (current density scale)
    - λ_D      = sqrt(ε · V_T / (F · c_ref))  (Debye length)
    """
    d_list = [float(v) for v in d_species_m2_s]
    if any(v <= 0.0 for v in d_list):
        raise ValueError("d_species_m2_s must contain strictly positive diffusivities.")
    if c_bulk_m <= 0.0:
        raise ValueError(f"c_bulk_m must be > 0; got {c_bulk_m}.")
    if c_inf_m < 0.0:
        raise ValueError(f"c_inf_m must be >= 0; got {c_inf_m}.")
    if temperature_k <= 0.0:
        raise ValueError(f"temperature_k must be > 0; got {temperature_k}.")
    if relative_permittivity <= 0.0:
        raise ValueError(f"relative_permittivity must be > 0; got {relative_permittivity}.")
    if length_scale_m <= 0.0:
        raise ValueError(f"length_scale_m must be > 0; got {length_scale_m}.")

    d_ref = float(np.exp(np.mean(np.log(np.asarray(d_list, dtype=float)))))
    c_bulk_mol_m3 = float(c_bulk_m) * MOLAR_TO_MOL_PER_M3
    c_inf_mol_m3 = float(c_inf_m) * MOLAR_TO_MOL_PER_M3
    eps_f_m = float(relative_permittivity) * VACUUM_PERMITTIVITY_F_PER_M
    v_thermal = (GAS_CONSTANT * float(temperature_k)) / FARADAY_CONSTANT
    L = float(length_scale_m)

    time_s = L * L / d_ref
    kappa_m_s = d_ref / L
    molar_flux = d_ref * c_bulk_mol_m3 / L
    current_density = FARADAY_CONSTANT * molar_flux
    debye_m = float(np.sqrt((eps_f_m * v_thermal) / (FARADAY_CONSTANT * c_bulk_mol_m3)))

    return NondimScales(
        d_species_m2_s=d_list,
        d_ref_m2_s=d_ref,
        bulk_concentration_mol_m3=c_bulk_mol_m3,
        c_inf_mol_m3=c_inf_mol_m3,
        temperature_k=float(temperature_k),
        thermal_voltage_v=v_thermal,
        permittivity_f_m=eps_f_m,
        length_scale_m=L,
        diffusivity_scale_m2_s=d_ref,
        concentration_scale_mol_m3=c_bulk_mol_m3,
        time_s=time_s,
        kappa_m_s=kappa_m_s,
        molar_flux_mol_m2_s=molar_flux,
        current_density_a_m2=current_density,
        debye_length_m=debye_m,
        debye_ratio=debye_m / L,
    )
