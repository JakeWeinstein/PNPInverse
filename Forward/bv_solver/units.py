"""Unit-conversion helpers shared by the BV-PNP forward solver.

Phase 6β v10a introduces this module to host the σ unit-conversion
helper consumed by the cation-hydrolysis ΔpKa machinery and diagnostic
plumbing.  The helper is *signed* on purpose: callers that need the
Singh-convention anode-clamped magnitude apply
``max(0.0, -signed)`` themselves at the call site.

References
----------
* ``docs/phase6/singh_2016_pka_formula.md`` §5.2 — the σ-local-Stern
  mapping convention.  Singh writes σ as counts/pm² (cathode-side
  magnitude); the forward solver carries σ in physical C/m² and only
  converts at the Singh boundary.
* :func:`Forward.bv_solver.cation_hydrolysis._build_singh_2016_eq_4_pka_shift`
  is the in-form UFL caller that performs the same conversion via
  ``fd.Constant``; the helper here is the matching Python-scalar
  equivalent used by diagnostics + tests so the two conventions stay
  in lockstep.
"""
from __future__ import annotations

# CODATA 2018 elementary charge.  Kept in lockstep with
# ``cation_hydrolysis._INVERSE_ELEMENTARY_CHARGE`` so the in-form UFL
# and the Python diagnostic helper agree to floating-point precision.
_ELEMENTARY_CHARGE_C: float = 1.602176634e-19


def sigma_C_m2_to_counts_pm2(sigma_signed: float) -> float:
    """Convert a *signed* surface charge density from C/m² to counts/pm².

    The conversion factor is::

        σ_counts_per_pm² = σ_C_per_m² · (N_A / F) · (1 m² / 1e24 pm²)
                         = σ_C_per_m² · (1 / e) · 1e-24
                         ≈ σ_C_per_m² · 6.2415e-6

    where ``N_A · e = F`` (definitional) is used to keep this consistent
    with the Faraday-constant value in ``Nondim.constants``.

    The return value is **signed**: the Singh ΔpKa formula applies the
    anode-clamp magnitude convention ``σ_singh = max(0.0, -sigma_signed)``
    at the call site, not here.  Diagnostics and tests want both the
    signed value and the Singh-clamped magnitude, so keeping the
    primitive signed lets each caller pick.

    Parameters
    ----------
    sigma_signed
        Signed Stern surface charge density in C/m².  Positive →
        anodic (Stern excess positive charge); negative → cathodic.

    Returns
    -------
    float
        Signed surface charge density in counts/pm².
    """
    factor_per_C_per_m2: float = 1.0 / _ELEMENTARY_CHARGE_C * 1.0e-24
    return float(sigma_signed) * factor_per_C_per_m2


__all__ = ["sigma_C_m2_to_counts_pm2"]
