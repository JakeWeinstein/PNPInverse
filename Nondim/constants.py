"""Physical constants used throughout the PNP nondimensionalization.

Single source of truth — import from here rather than redefining in each module.
"""

FARADAY_CONSTANT = 96485.3329           # C / mol
GAS_CONSTANT = 8.314462618              # J / (mol · K)
DEFAULT_TEMPERATURE_K = 298.15          # K  (25 °C)
VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12   # F / m  (ε₀)
DEFAULT_RELATIVE_PERMITTIVITY_WATER = 78.5        # dimensionless (ε_r at ~25 °C)
MOLAR_TO_MOL_PER_M3 = 1_000.0          # 1 M = 1000 mol/m³
