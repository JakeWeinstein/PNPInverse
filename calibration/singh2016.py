"""Singh 2016 SI cation hydrolysis constants (Firedrake-free leaf module).

Phase 6β v9 Gate 4A introduced the Singh 2016 SI Eq. (3)/(4) ΔpKa formula
for cation hydrolysis at the polarized OHP.  Through v9 → v10b these
constants lived in :mod:`scripts._bv_common`.  Phase 6β step 10 (Phase
D) promotes them to a top-level Firedrake-free module so calibration
code (target-ΔpKa-effect bracket construction, β_per_cation derivations,
unit tests of Singh's geometric coefficient) can import the dict
without dragging Firedrake or the BV solver package into ``sys.modules``.

Provenance: ``docs/phase6/singh_2016_pka_formula.md``.  Constants
verified against Singh 2016 SI Tables S1/S3 within rounding.

The Singh formula structure is::

    pKa_bulk(M)   =  B  −  A · z_eff² / r_M-O
    ΔpKa(σ)      =  +2 · A · z_eff · σ_singh · r_H_El · (1 − r_M-O² / r_H_El²)

with ``r_M-O = r_M + r_O``, ``z_eff`` Singh's effective cation charge,
``A`` Singh's slope (pm), ``B`` Singh's intercept (dimensionless), and
``r_H_El`` the hydration-shell-H to electrode distance (pm).  Per
``singh_2016_pka_formula.md`` §3.4, the cathodic case (``σ_S < 0`` and
``r_H_El < r_M-O``) gives ``ΔpKa < 0`` — lowering the hydrolysis pKa,
producing protons at the OHP.

This module is intentionally leaf-level: it imports only the standard
library.  Any caller that needs Firedrake symbols (UFL expressions,
``Function`` objects) must build them outside this module — Singh's
parameters are pure numerics.

Public API
----------

* :data:`SINGH_A_PM`, :data:`SINGH_B`, :data:`SINGH_R_O_PM` — global
  Singh constants.
* :data:`SINGH_2016_CATION_PARAMS` — per-cation Table S1 row plus the
  Cu r_H_El back-fit (``r_H_El_pm_Cu``).
* :func:`compute_beta_per_cation` — Phase 6β step 10 Phase D helper.
  Returns the per-cation ``β`` coefficient in pm² so the Phase D
  ``Δ_β`` fit can parameterize Δβ in target-ΔpKa-effect space.
"""

from __future__ import annotations

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Singh 2016 SI Eq. (3)/(4) global constants
# ---------------------------------------------------------------------------

SINGH_A_PM: float = 620.32
"""Singh 2016 SI Eq. (3) slope, units of pm.

Per ``singh_2016_pka_formula.md`` §3.  Used in both Eq. (3) (pKa_bulk)
and Eq. (4) (ΔpKa)."""

SINGH_B: float = 17.154
"""Singh 2016 SI Eq. (3) intercept, dimensionless."""

SINGH_R_O_PM: float = 63.0
"""O-atom radius in pm.  ``r_M-O = r_M + r_O`` per Singh Eq. (3) +
Eq. (4)."""


# ---------------------------------------------------------------------------
# Per-cation Singh Table S1 row + Cu r_H_El back-fit
# ---------------------------------------------------------------------------
#
# Canonical keys are charged-form (``"K+"``, ``"Cs+"``, etc.) per the
# Phase 6β v9 + v10a convention (verified at
# ``scripts/_bv_common.py:913-933``).  Callers must use the exact key
# string; no alias support — see :func:`compute_beta_per_cation` for the
# ``ValueError`` contract.

SINGH_2016_CATION_PARAMS: Dict[str, Dict[str, float]] = {
    "Li+": {
        "r_M_pm": 69.0,  "z_eff": 0.864, "n_hyd": 5.2,
        "pKa_bulk": 13.6, "r_H_El_pm_Cu": 132.00,
    },
    "Na+": {
        "r_M_pm": 102.0, "z_eff": 0.900, "n_hyd": 3.5,
        "pKa_bulk": 14.2, "r_H_El_pm_Cu": 164.99,
    },
    "K+": {
        "r_M_pm": 138.0, "z_eff": 0.919, "n_hyd": 2.6,
        "pKa_bulk": 14.5, "r_H_El_pm_Cu": 200.98,
    },
    "Rb+": {
        "r_M_pm": 149.0, "z_eff": 0.923, "n_hyd": 2.4,
        "pKa_bulk": 14.6, "r_H_El_pm_Cu": 211.98,
    },
    "Cs+": {
        "r_M_pm": 170.0, "z_eff": 0.930, "n_hyd": 2.1,
        "pKa_bulk": 14.8, "r_H_El_pm_Cu": 232.97,
    },
}


# ---------------------------------------------------------------------------
# β_per_cation helper (Phase 6β step 10 Phase D)
# ---------------------------------------------------------------------------


def compute_beta_per_cation(
    cation: str,
    r_H_El_pm: Optional[float] = None,
) -> float:
    """Return the per-cation ``β`` coefficient in pm².

    Phase 6β step 10 Phase D parameterises the cation-hydrolysis Singh
    ΔpKa shift as ``ΔpKa = β_per_cation · σ_singh`` (Singh σ in
    counts/pm², ``β`` in pm²).  From Singh 2016 SI Eq. (4):

    .. math::

        \\beta_{\\rm per\\ cation}  =  2 \\cdot A \\cdot z_{\\rm eff}
            \\cdot r_{H\\text{-}El}
            \\cdot \\left(1 - \\frac{r_{M\\text{-}O}^{2}}{r_{H\\text{-}El}^{2}}\\right)

    with ``r_M-O = r_M + r_O``.

    Cathodic / Singh-geometry case (``r_H_El < r_M-O``) gives the
    geometric factor ``< 0``, hence ``β < 0`` — the sign convention
    used by the residual at ``cation_hydrolysis.py:541`` (``β · σ_singh
    < 0`` with the cathode-clamped positive ``σ_singh``).

    For ``"K+"`` at the Cu default ``r_H_El_pm_Cu = 200.98``, returns
    exactly ``-45.608196 pm²`` to 6 decimal places (see the Phase D
    plan §3.0 for the bracket construction that depends on this value).

    Parameters
    ----------
    cation
        Charged-form key (``"Li+"``, ``"Na+"``, ``"K+"``, ``"Rb+"``,
        ``"Cs+"``).  Bare-element strings (``"K"``, ``"Cs"``) raise
        ``ValueError`` — no alias support, since the canonical key
        convention is locked at ``_bv_common.py:913-933``.
    r_H_El_pm
        Optional override for the hydration-shell H to electrode
        distance (pm).  When ``None`` (default), falls back to the
        cation's Cu back-fit value ``r_H_El_pm_Cu`` from
        :data:`SINGH_2016_CATION_PARAMS`.

    Returns
    -------
    float
        ``β_per_cation`` in pm².  Negative under Singh's cathodic
        geometry for every entry in the table.

    Raises
    ------
    ValueError
        ``cation`` not present in :data:`SINGH_2016_CATION_PARAMS`.
    """
    if cation not in SINGH_2016_CATION_PARAMS:
        known = sorted(SINGH_2016_CATION_PARAMS.keys())
        raise ValueError(
            f"compute_beta_per_cation: unknown cation {cation!r}; "
            f"canonical charged-form keys are {known}.  No alias support."
        )
    params = SINGH_2016_CATION_PARAMS[cation]
    z = float(params["z_eff"])
    r_M = float(params["r_M_pm"])
    r_M_O = r_M + SINGH_R_O_PM

    if r_H_El_pm is None:
        r_H_El = float(params["r_H_El_pm_Cu"])
    else:
        r_H_El = float(r_H_El_pm)

    if r_H_El <= 0.0:
        raise ValueError(
            f"compute_beta_per_cation: r_H_El_pm must be positive (got "
            f"{r_H_El!r}); the Singh geometric factor "
            f"(1 − r_M-O²/r_H_El²) diverges at r_H_El → 0."
        )

    geometric = 1.0 - (r_M_O * r_M_O) / (r_H_El * r_H_El)
    return 2.0 * SINGH_A_PM * z * r_H_El * geometric


__all__ = [
    "SINGH_A_PM",
    "SINGH_B",
    "SINGH_R_O_PM",
    "SINGH_2016_CATION_PARAMS",
    "compute_beta_per_cation",
]
