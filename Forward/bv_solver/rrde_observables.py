"""Pure-Python RRDE-style observables for the BV-PNP forward solver.

Post-processing on per-voltage assembled scalars (``j_disk``, ``j_h2o2_disk``)
and a single surface scalar (``c_H_surface_mean``).  No Firedrake imports;
all inputs are plain ``float``.

Observables produced here mirror the experimental quantities reported in
Mangan 2025-style RRDE measurements: a surface-pH proxy, a model ring
current under an idealised disk-to-ring transport with collection
efficiency ``N``, and the standard RRDE selectivity ``S_H2O2`` and
apparent electron count ``n_e``.

These observables are independent of the electrolyte-physics decisions
deferred to M2.  Per-run provenance metadata at
``scripts._bv_common.ExperimentMetadata`` records the comparison status
and source authority for any individual run; without that block,
RRDE-shaped output should not be interpreted as deck-comparable.

Sign conventions (locked across the M1 plan)
--------------------------------------------
- ``j_disk_model`` and ``j_h2o2_disk_model`` follow the existing solver
  convention used by ``Forward.bv_solver.observables`` with
  ``scale=-I_SCALE``: cathodic = negative.
- ``j_ring_model`` is positive when ``j_h2o2_disk_model`` is negative
  (cathodic peroxide production at the disk → anodic peroxide oxidation
  at the ring).  The ``abs(j_h2o2_disk_model)`` factor handles the sign
  flip explicitly; F7 (peroxide-current sign-violation) in
  ``Forward.bv_solver.validation`` flags upstream sign anomalies
  separately.
- ``S_H2O2_percent`` and ``n_e_rrde`` use ``abs(j_disk_model)`` to keep
  the percentages physical.

Formulas
--------
``surface_pH_proxy = -log10(c_H_surface_nondim * C_scale_mol_m3 / 1000)``
(factor 1000 converts mol/m³ → mol/L; the proxy is *not* yet
activity-coefficient-corrected — that correction is deferred until M0
extracts the IrOx calibration protocol.)

``j_ring_model     = N * |j_h2o2_disk_model|``
``S_H2O2_percent   = 200 * (j_ring_model/N) / (|j_disk_model| + j_ring_model/N)``
``n_e_rrde         = 4 * |j_disk_model| / (|j_disk_model| + j_ring_model/N)``

The factor 200 (not 100) in ``S_H2O2_percent`` comes from the 2-electron
peroxide pathway in a 4-electron-vs-2-electron disk-current decomposition:
perfect 2e gives 100% (and ``n_e=2``), perfect 4e gives 0% (and ``n_e=4``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = [
    "RRDEObservables",
    "compute_surface_pH_proxy",
    "compute_ring_current",
    "compute_selectivity_percent",
    "compute_n_e_rrde",
    "assemble_rrde_observables",
]


@dataclass(frozen=True)
class RRDEObservables:
    """One voltage's worth of RRDE-style observables.

    All fields are scalar ``float``.  See module docstring for sign
    conventions and unit expectations.
    """

    j_disk_model: float        # mA/cm², cathodic = negative
    j_h2o2_disk_model: float   # mA/cm², cathodic = negative
    j_ring_model: float        # mA/cm², anodic = positive
    surface_pH_proxy: float    # dimensionless
    S_H2O2_percent: float      # 0..100 for physical inputs
    n_e_rrde: float            # 2..4 for physical inputs


def compute_surface_pH_proxy(
    c_H_surface_nondim: float,
    C_scale_mol_m3: float,
) -> float:
    """Surface-pH proxy from nondim H+ surface mean.

    Returns ``-log10(c_H_surface_mol_per_L)`` where the dimensional
    surface concentration is ``c_H_surface_nondim * C_scale_mol_m3 / 1000``.
    Returns NaN when ``c_H_surface_nondim <= 0`` (numerical noise pushing
    H+ surface mean below zero) so failed-solve points propagate cleanly
    through the per-voltage loop.

    No activity-coefficient correction is applied; the result is a *proxy*
    until M0 extracts the IrOx calibration protocol.
    """
    if C_scale_mol_m3 <= 0.0:
        raise ValueError(
            f"C_scale_mol_m3 must be positive, got {C_scale_mol_m3}"
        )
    if not math.isfinite(c_H_surface_nondim) or c_H_surface_nondim <= 0.0:
        return float("nan")
    c_h_mol_per_l = c_H_surface_nondim * C_scale_mol_m3 / 1000.0
    return -math.log10(c_h_mol_per_l)


def compute_ring_current(
    j_h2o2_disk_mA_cm2: float,
    N_collection: float,
) -> float:
    """Model ring current from disk peroxide current and collection efficiency.

    ``j_ring_model = N * |j_h2o2_disk_model|``.  The absolute value is
    deliberate: it keeps the ring current positive (anodic) regardless of
    the disk-side sign convention.  Upstream sign anomalies are caught by
    F7 in ``validate_observables``, not here.
    """
    _validate_collection(N_collection)
    return N_collection * abs(j_h2o2_disk_mA_cm2)


def compute_selectivity_percent(
    j_disk_mA_cm2: float,
    j_ring_mA_cm2: float,
    N_collection: float,
) -> float:
    """RRDE selectivity to H2O2 in percent.

    ``S_H2O2 = 200 * (I_ring/N) / (|I_disk| + I_ring/N)``.  Returns NaN
    when ``|I_disk| + I_ring/N == 0`` (no current to apportion).  Values
    outside ``[0, 100]`` for physical inputs indicate a sign-convention
    mismatch upstream and should be cross-validated against F3 in
    ``validate_observables``.
    """
    _validate_collection(N_collection)
    abs_disk = abs(j_disk_mA_cm2)
    ring_over_n = j_ring_mA_cm2 / N_collection
    denom = abs_disk + ring_over_n
    if denom == 0.0:
        return float("nan")
    return 200.0 * ring_over_n / denom


def compute_n_e_rrde(
    j_disk_mA_cm2: float,
    j_ring_mA_cm2: float,
    N_collection: float,
) -> float:
    """Apparent electron count per O2 from disk + ring currents.

    ``n_e = 4 * |I_disk| / (|I_disk| + I_ring/N)``.  Pure 2e (peroxide-only)
    gives 2; pure 4e (water-only) gives 4.  Returns NaN when both currents
    are zero.
    """
    _validate_collection(N_collection)
    abs_disk = abs(j_disk_mA_cm2)
    ring_over_n = j_ring_mA_cm2 / N_collection
    denom = abs_disk + ring_over_n
    if denom == 0.0:
        return float("nan")
    return 4.0 * abs_disk / denom


def assemble_rrde_observables(
    j_disk: float,
    j_h2o2_disk: float,
    c_H_surface_nondim: float,
    C_scale_mol_m3: float,
    N_collection: float,
) -> RRDEObservables:
    """Assemble all RRDE-style observables for one voltage point.

    Parameters
    ----------
    j_disk, j_h2o2_disk
        Disk-side currents in mA/cm² (already dimensionalized via
        ``I_SCALE`` upstream).  Cathodic = negative per the existing
        ``scale=-I_SCALE`` observable convention.
    c_H_surface_nondim
        Surface mean of nondim H+ concentration, typically read from the
        diagnostics dict's ``c{H_idx}_surface_mean`` entry populated by
        ``Forward.bv_solver.diagnostics.surface_field_means``.
    C_scale_mol_m3
        Concentration scale used for nondimensionalization
        (``scripts._bv_common.C_SCALE``).
    N_collection
        RRDE collection efficiency in ``(0, 1]``.
    """
    j_ring = compute_ring_current(j_h2o2_disk, N_collection)
    return RRDEObservables(
        j_disk_model=float(j_disk),
        j_h2o2_disk_model=float(j_h2o2_disk),
        j_ring_model=float(j_ring),
        surface_pH_proxy=compute_surface_pH_proxy(
            c_H_surface_nondim, C_scale_mol_m3
        ),
        S_H2O2_percent=compute_selectivity_percent(
            j_disk, j_ring, N_collection
        ),
        n_e_rrde=compute_n_e_rrde(j_disk, j_ring, N_collection),
    )


def _validate_collection(N_collection: float) -> None:
    if not math.isfinite(N_collection):
        raise ValueError(
            f"N_collection must be a finite float in (0, 1], got {N_collection}"
        )
    if not (0.0 < N_collection <= 1.0):
        raise ValueError(
            f"N_collection must be in (0, 1], got {N_collection}"
        )
