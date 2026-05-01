"""Shared physics-validation module for the PNP-BV solver.

Solutions that pass SNES convergence checks can still contain non-physical
artifacts: negative concentrations, currents exceeding diffusion limits,
regularization floors masking real depletion, etc.  This module catches them.

Classification
--------------
Failures (Fxx) -- the solution is non-physical and must not be trusted:
    F1  Negative concentration
    F2  Current exceeds diffusion limit
    F3  Peroxide selectivity > 100%
    F4  Concentration floor domination (eps_c regularization masks physics)
    F6  H2O2 exceeds stoichiometric limit (more H2O2 than source O2)
    F7  Peroxide current wrong sign

Warnings (Wxx) -- suspicious but may be acceptable:
    W1  Clip saturation (BV exponent near clipping threshold)
    W2  Concentration exceeds bulk in interior
    W3  Potential overshoot
    W5  H+ (or other cation) depletion in the bulk region
    W8  Bulk integral still drifting despite flux steady-state
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a validation pass."""

    valid: bool                    # False if any FAIL triggered
    failures: list[str] = field(default_factory=list)   # e.g. ["F1: negative c_O2, min=-0.42"]
    warnings: list[str] = field(default_factory=list)   # e.g. ["W1: clip saturated at V=0.3"]


# ---------------------------------------------------------------------------
# Function 1 -- solution-state checks
# ---------------------------------------------------------------------------

def validate_solution_state(
    U,
    *,
    n_species: int,
    c_bulk: Sequence[float],
    phi_applied: float,
    z_vals: Sequence[float],
    eps_c: float,
    exponent_clip: float,
    species_names: Optional[Sequence[str]] = None,
) -> ValidationResult:
    """Validate a Firedrake mixed-space solution [c_0, ..., c_{n-1}, phi].

    All checks use numpy on ``U.dat[i].data_ro`` -- no FEM assembly.
    """
    import numpy as np

    fails: list[str] = []
    warns: list[str] = []
    names = species_names or [f"species_{i}" for i in range(n_species)]

    # --- F1: negative concentration ---
    # Guard against floating-point noise below zero (|c| < eps_c treated as 0).
    for i in range(n_species):
        c_min = float(U.dat[i].data_ro.min())
        if c_min < -eps_c:
            fails.append(f"F1: negative {names[i]}, min={c_min:.4g}")

    # --- F4: concentration floor domination ---
    # Only meaningful for species whose bulk concentration is substantial --
    # a species that legitimately starts at c=0 (e.g. H2O2) is not
    # "floor-dominated" merely because it equals its initial value.
    # Skip species where F1 already fired (negative c is a distinct failure).
    for i in range(min(n_species, 2)):
        if c_bulk[i] <= eps_c * 10:  # species with no meaningful bulk source
            continue
        c_min = float(U.dat[i].data_ro.min())
        if 0.0 <= c_min <= eps_c * 2.0:
            fails.append(
                f"F4: {names[i]} floor-dominated, min={c_min:.4g} <= 2*eps_c={eps_c * 2.0:.4g}"
            )

    # --- F6: H2O2 exceeds stoichiometric limit ---
    if n_species >= 2 and c_bulk[1] < c_bulk[0] * 0.01:
        c_h2o2_max = float(U.dat[1].data_ro.max())
        if c_h2o2_max > c_bulk[0] * 1.05:
            fails.append(
                f"F6: {names[1]} exceeds O2 bulk, max={c_h2o2_max:.4g} > "
                f"{c_bulk[0] * 1.05:.4g}"
            )

    # --- W2: concentration exceeds bulk in interior ---
    # Charged species (z != 0) can legitimately accumulate at the electrode
    # via Boltzmann-style attraction; only flag large overshoots to catch
    # CG1 oscillations rather than physical accumulation.
    for i in range(n_species):
        c_max = float(U.dat[i].data_ro.max())
        if c_bulk[i] <= 0:
            continue
        # Neutral species: 20% tolerance.  Charged species: 5x tolerance
        # (charged species can accumulate by exp(-z*phi) near the electrode).
        tol_factor = 1.20 if z_vals[i] == 0 else 5.0
        if c_max > c_bulk[i] * tol_factor:
            warns.append(
                f"W2: {names[i]} exceeds {int(tol_factor*100)}% bulk, "
                f"max={c_max:.4g} vs bulk={c_bulk[i]:.4g}"
            )

    # --- W3: potential overshoot ---
    # The field phi can legitimately exceed phi_applied magnitude in the
    # Debye layer (charge accumulation), and the driving overpotential
    # |eta| = |phi_applied - E_eq| can be O(20+) in nondim units for
    # physical E_eq values.  Use a generous absolute floor (25) so W3
    # only fires on gross CG1 oscillations, not on legitimate Debye/BV
    # physics.
    phi_data = U.dat[n_species].data_ro
    phi_min = float(phi_data.min())
    phi_max = float(phi_data.max())
    tol = max(abs(phi_applied) * 2.0, 25.0)
    phi_lo = min(phi_applied, 0.0) - tol
    phi_hi = max(phi_applied, 0.0) + tol
    if phi_min < phi_lo:
        warns.append(
            f"W3: phi undershoot, min={phi_min:.4g} < {phi_lo:.4g}"
        )
    if phi_max > phi_hi:
        warns.append(
            f"W3: phi overshoot, max={phi_max:.4g} > {phi_hi:.4g}"
        )

    # --- W5: cation depletion in bulk region ---
    for i in range(n_species):
        if z_vals[i] > 0 and c_bulk[i] > 0:
            coords = U.function_space().mesh().coordinates.dat.data_ro
            y_coords = coords[:, -1] if coords.ndim == 2 else coords
            y_median = float(np.median(y_coords))
            c_data = U.dat[i].data_ro
            top_mask = y_coords >= y_median
            if np.any(top_mask):
                c_top_min = float(c_data[top_mask].min())
                if c_top_min < c_bulk[i] * 1e-3:
                    warns.append(
                        f"W5: {names[i]} depleted in bulk region, "
                        f"min={c_top_min:.4g} < {c_bulk[i] * 1e-3:.4g}"
                    )

    valid = len(fails) == 0
    if warns:
        for w in warns:
            warnings.warn(w, stacklevel=2)
    return ValidationResult(valid=valid, failures=fails, warnings=warns)


# ---------------------------------------------------------------------------
# Function 2 -- observable checks
# ---------------------------------------------------------------------------

def validate_observables(
    cd: float,
    pc: float,
    *,
    I_lim: float,
    phi_applied: float,
    V_T: float,
) -> ValidationResult:
    """Validate assembled scalar observables (current density, peroxide current)."""
    fails: list[str] = []
    warns: list[str] = []

    # --- F2: current exceeds diffusion limit ---
    if abs(cd) > abs(I_lim) * 1.05:
        fails.append(
            f"F2: |cd|={abs(cd):.4g} exceeds 1.05*|I_lim|={abs(I_lim) * 1.05:.4g}"
        )

    # --- F3: peroxide selectivity > 100% ---
    if abs(cd) > 1e-10:
        selectivity = abs(pc / cd)
        if selectivity > 1.05:
            fails.append(
                f"F3: peroxide selectivity={selectivity:.4g} > 1.05"
            )

    # --- F7: peroxide current wrong sign ---
    if cd < 0.0:
        # Cathodic current: pc should be <= 0 (or negligible positive noise).
        if pc > abs(cd) * 0.01:
            fails.append(
                f"F7: peroxide wrong sign, pc={pc:.4g} > 0 while cd={cd:.4g} < 0"
            )

    valid = len(fails) == 0
    if warns:
        for w in warns:
            warnings.warn(w, stacklevel=2)
    return ValidationResult(valid=valid, failures=fails, warnings=warns)


# ---------------------------------------------------------------------------
# Function 3 -- steady-state checks
# ---------------------------------------------------------------------------

def validate_steady_state(
    flux_history: Sequence[float],
    *,
    bulk_integral_history: Optional[Sequence[float]] = None,
) -> ValidationResult:
    """Check whether bulk integrals are still drifting despite flux convergence."""
    fails: list[str] = []
    warns: list[str] = []

    # --- W8: bulk integral still drifting ---
    if bulk_integral_history is not None and len(bulk_integral_history) >= 4:
        last4 = list(bulk_integral_history[-4:])
        ref = abs(last4[-1])
        if ref < 1e-30:
            ref = 1.0
        max_rel_change = 0.0
        for k in range(1, 4):
            rel = abs(last4[k] - last4[k - 1]) / ref
            if rel > max_rel_change:
                max_rel_change = rel
        if max_rel_change > 1e-3:
            warns.append(
                f"W8: bulk integral still drifting, max relative change="
                f"{max_rel_change:.4g} over last 4 entries"
            )

    valid = len(fails) == 0
    if warns:
        for w in warns:
            warnings.warn(w, stacklevel=2)
    return ValidationResult(valid=valid, failures=fails, warnings=warns)


# ---------------------------------------------------------------------------
# Helper -- diffusion-limited current from solver_params
# ---------------------------------------------------------------------------

def compute_i_lim_from_params(solver_params: Sequence[object]) -> float:
    """Compute the nondim diffusion-limited current from solver_params.

    The 11-element SolverParams layout has c0 (bulk concentrations) at
    index 8.  Nondim I_lim ~ 2 * max(c0) for the standard BV scaling
    where D, L are scaled to O(1) and n_e=2 for the ORR stoichiometry.
    """
    c0 = solver_params[8]
    c0_vals = list(c0) if hasattr(c0, "__iter__") else [c0]
    return 2.0 * max(float(v) for v in c0_vals)


# ---------------------------------------------------------------------------
# Function 4 -- exponent clip saturation
# ---------------------------------------------------------------------------

def check_clip_saturation(
    eta_raw: float,
    *,
    exponent_clip: float,
    bv_exp_scale: float,
    alpha_vals: Sequence[float],
    n_e_vals: Sequence[int],
) -> list[str]:
    """Return W1 warnings for reactions whose BV exponent is near the clip.

    Parameters
    ----------
    eta_raw : float
        Overpotential (dimensional or non-dimensional, matching *bv_exp_scale*).
    exponent_clip : float
        Maximum allowed exponent magnitude before clipping is applied.
    bv_exp_scale : float
        Typically ``F / (R T)`` or ``1 / V_T`` depending on nondimensionalisation.
    alpha_vals : sequence of float
        Transfer coefficients per reaction.
    n_e_vals : sequence of int
        Number of electrons per reaction.
    """
    result: list[str] = []
    threshold = exponent_clip * 0.95
    for j, (alpha_j, n_e_j) in enumerate(zip(alpha_vals, n_e_vals)):
        arg = abs(alpha_j * n_e_j * bv_exp_scale * eta_raw)
        if arg >= threshold:
            result.append(
                f"W1: reaction {j} clip-saturated, |arg|={arg:.4g} >= "
                f"0.95*clip={threshold:.4g}"
            )
    return result
