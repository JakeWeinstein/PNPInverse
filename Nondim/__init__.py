"""Nondimensionalization package for PNP solvers.

Provides a single, audited home for all scaling logic.  Import from here::

    from Nondim import build_physical_scales, build_model_scaling, NondimScales
    from Nondim.constants import FARADAY_CONSTANT, GAS_CONSTANT

Sub-modules
-----------
constants
    Physical constants (F, R, ε₀, …) — single source of truth.
scales
    :func:`build_physical_scales` and :class:`NondimScales` dataclass.
transform
    :func:`build_model_scaling` — converts physical solver inputs to
    model-space values consumed by the forward PDE solver.
    Also :func:`verify_model_params` for sanity checks.
"""

from Nondim.constants import (
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    DEFAULT_TEMPERATURE_K,
    VACUUM_PERMITTIVITY_F_PER_M,
    DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    MOLAR_TO_MOL_PER_M3,
)
from Nondim.scales import NondimScales, build_physical_scales
from Nondim.transform import build_model_scaling, verify_model_params

__all__ = [
    # constants
    "FARADAY_CONSTANT",
    "GAS_CONSTANT",
    "DEFAULT_TEMPERATURE_K",
    "VACUUM_PERMITTIVITY_F_PER_M",
    "DEFAULT_RELATIVE_PERMITTIVITY_WATER",
    "MOLAR_TO_MOL_PER_M3",
    # scales
    "NondimScales",
    "build_physical_scales",
    # transform
    "build_model_scaling",
    "verify_model_params",
]
