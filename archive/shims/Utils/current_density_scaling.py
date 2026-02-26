"""Nondimensionalization helpers for Robin current-density workflows.

``build_physical_scales`` is now canonical in ``Nondim.scales`` and returns a
``NondimScales`` dataclass.  This shim wraps it to return a plain dict for
backward compatibility with callers that use ``scales["kappa_scale_m_s"]`` etc.

``build_solver_options`` is experiment-specific and stays here.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from Nondim.scales import NondimScales  # noqa: F401
from Nondim.scales import build_physical_scales as _nondim_build_physical_scales
from Nondim.constants import (  # noqa: F401
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    DEFAULT_TEMPERATURE_K,
    VACUUM_PERMITTIVITY_F_PER_M,
    DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    MOLAR_TO_MOL_PER_M3,
)


def build_physical_scales(
    *,
    d_species_m2_s: Sequence[float] = (9.311e-9, 5.273e-9),
    c_bulk_m: float = 0.1,
    c_inf_m: float = 0.01,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    relative_permittivity: float = DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    length_scale_m: float = 1.0e-4,
) -> Dict[str, Any]:
    """Backward-compat wrapper returning a plain dict.

    New code should import ``build_physical_scales`` from ``Nondim.scales``
    to get a typed ``NondimScales`` dataclass with named attributes.
    """
    return _nondim_build_physical_scales(
        d_species_m2_s=d_species_m2_s,
        c_bulk_m=c_bulk_m,
        c_inf_m=c_inf_m,
        temperature_k=temperature_k,
        relative_permittivity=relative_permittivity,
        length_scale_m=length_scale_m,
    ).to_dict()


def build_solver_options(scales: Dict[str, Any]) -> Dict[str, Any]:
    """Build PETSc/SNES + Robin + nondim option block for current-density studies."""
    return {
        "snes_type": "newtonls",
        "snes_max_it": 100,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_linesearch_type": "bt",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "robin_bc": {
            "kappa": [0.8, 0.8],
            "c_inf": [float(scales["c_inf_mol_m3"]), float(scales["c_inf_mol_m3"])],
            "electrode_marker": 1,
            "concentration_marker": 3,
            "ground_marker": 3,
        },
        "nondim": {
            "enabled": True,
            "temperature_k": float(scales["temperature_k"]),
            "potential_scale_v": float(scales["thermal_voltage_v"]),
            "length_scale_m": float(scales["length_scale_m"]),
            "concentration_scale_mol_m3": float(scales["concentration_scale_mol_m3"]),
            "diffusivity_scale_m2_s": float(scales["diffusivity_scale_m2_s"]),
            "permittivity_f_m": float(scales["permittivity_f_m"]),
            "kappa_scale_m_s": float(scales["kappa_scale_m_s"]),
            "time_scale_s": float(scales["time_scale_s"]),
            "diffusivity_inputs_are_dimensionless": False,
            "concentration_inputs_are_dimensionless": False,
            "potential_inputs_are_dimensionless": False,
            "time_inputs_are_dimensionless": False,
            # In current Robin kappa studies, inference variables are in model units.
            "kappa_inputs_are_dimensionless": True,
        },
    }
