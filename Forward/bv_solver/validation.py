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
    is_logc: bool = False,
    mu_species: Optional[Sequence[int]] = None,
    em: float = 1.0,
    reaction_e_eq: Optional[Sequence[float]] = None,
    bv_exp_scale: float = 1.0,
    w1_margin: float = 1.0,
) -> ValidationResult:
    """Validate a Firedrake mixed-space solution.

    Concentration formulation: ``U`` carries ``[c_0, ..., c_{n-1}, phi]``
    and concentration data is read directly from ``U.dat[i].data_ro``.

    Log-concentration formulation (``is_logc=True``): ``U`` carries
    ``[u_0, ..., u_{n-1}, phi]`` with ``u_i = ln(c_i)``.  Concentration
    samples are recovered as ``c_i = exp(u_i)`` before the standard
    checks run.  ``F1`` (negative concentration) is unreachable in this
    mode since ``exp(u_i) > 0`` exactly.

    Muh formulation (``is_logc=True`` and ``mu_species=[...]``):
    ``U.dat[i]`` for ``i in mu_species`` carries ``mu_i = u_i + em*z_i*phi``
    instead of ``u_i``.  The per-dof ``log(c_i)`` is recovered via

        log(c_i) = U.dat[i] - em*z_i*U.dat[n_species]

    before exponentiation.  This matches the proton-electrochemical-
    potential transform in ``Forward/bv_solver/forms_logc_muh.py``.  The
    same physics checks then apply unchanged.  Pass ``em`` from
    ``ctx['nondim']['electromigration_prefactor']`` (= 1.0 in the
    production scaling).

    All checks use numpy on dat data plus a single coordinate-interpolation
    in W5 (so the y-mask aligns with each species' DOF layout for any
    element order).

    W1 (clip saturation) fires when ``|η_scaled_j|`` for any reaction ``j``
    reaches within ``w1_margin`` of ``exponent_clip`` somewhere in the
    domain, where ``η_scaled_j = bv_exp_scale * (phi_applied - phi - E_eq_j)``.
    Pass ``reaction_e_eq = [r["E_eq_model"] for r in scaling["bv_reactions"]]``
    and ``bv_exp_scale = scaling["bv_exponent_scale"]`` from the live
    scaling dict.  When ``reaction_e_eq`` is ``None``, W1 is skipped.
    """
    import numpy as np

    fails: list[str] = []
    warns: list[str] = []
    names = species_names or [f"species_{i}" for i in range(n_species)]
    mu_set = frozenset(int(i) for i in mu_species) if mu_species else frozenset()

    def _conc_array(i: int):
        """Return concentration samples for species ``i`` as a numpy array.

        For non-mu species this is ``exp(u_i)`` (logc) or ``c_i`` raw
        (concentration formulation).  For mu species (proton in the muh
        formulation), recovers ``log(c_i) = mu_i - em*z_i*phi`` first.
        """
        raw = U.dat[i].data_ro
        if is_logc:
            if i in mu_set:
                phi_data = U.dat[n_species].data_ro
                z_i = float(z_vals[i])
                return np.exp(raw - em * z_i * phi_data)
            return np.exp(raw)
        return raw

    # --- F1: negative concentration ---
    # Guard against floating-point noise below zero (|c| < eps_c treated as 0).
    # In logc mode this loop is structurally unreachable (exp(u) > 0) but is
    # kept for symmetry with the concentration path.
    for i in range(n_species):
        c_min = float(_conc_array(i).min())
        if c_min < -eps_c:
            fails.append(f"F1: negative {names[i]}, min={c_min:.4g}")

    # --- F4: concentration floor domination ---
    # Only meaningful for species whose bulk concentration is substantial --
    # a species that legitimately starts at c=0 (e.g. H2O2) is not
    # "floor-dominated" merely because it equals its initial value.
    # Skip in logc mode: there is no eps_c floor; small concentrations are
    # represented exactly via large negative u_i.  Skip species where F1
    # already fired (negative c is a distinct failure).
    if not is_logc:
        for i in range(min(n_species, 2)):
            if c_bulk[i] <= eps_c * 10:  # species with no meaningful bulk source
                continue
            c_min = float(_conc_array(i).min())
            if 0.0 <= c_min <= eps_c * 2.0:
                fails.append(
                    f"F4: {names[i]} floor-dominated, min={c_min:.4g} <= 2*eps_c={eps_c * 2.0:.4g}"
                )

    # --- F6: H2O2 exceeds stoichiometric limit ---
    if n_species >= 2 and c_bulk[1] < c_bulk[0] * 0.01:
        c_h2o2_max = float(_conc_array(1).max())
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
        c_max = float(_conc_array(i).max())
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

    # --- W1: BV exponent saturates the clip ---
    # eta_scaled_j = bv_exp_scale * (phi_applied - phi - E_eq_j) is the
    # quantity actually fed to the BV exponent before the alpha*n_e
    # multiplication.  forms_logc[*]._build_eta_clipped clips it to
    # [-exponent_clip, +exponent_clip].  If |eta_scaled| reaches within
    # w1_margin of exponent_clip *anywhere* phi varies in the domain, the
    # BV form is locally saturated and the partial current at that
    # reaction is artefacted.  Use (phi_min, phi_max) as the bounding box
    # for the per-DOF eta range -- O(1) numpy work, no FEM assembly.
    if reaction_e_eq is not None and exponent_clip > 0:
        phi_min_dof = float(phi_data.min())
        phi_max_dof = float(phi_data.max())
        threshold = float(exponent_clip) - float(w1_margin)
        for j, e_eq_j in enumerate(reaction_e_eq):
            e_eq_f = float(e_eq_j)
            eta_at_phi_min = bv_exp_scale * (phi_applied - phi_min_dof - e_eq_f)
            eta_at_phi_max = bv_exp_scale * (phi_applied - phi_max_dof - e_eq_f)
            max_abs_eta = max(abs(eta_at_phi_min), abs(eta_at_phi_max))
            if max_abs_eta >= threshold:
                warns.append(
                    f"W1: clip near saturation in rxn {j} "
                    f"(|eta_scaled|>={max_abs_eta:.3g}, "
                    f"clip-margin={threshold:.3g})"
                )

    # --- W5: cation depletion in bulk region ---
    # The species' DOF layout is the layout of U.dat[i].data_ro, which
    # equals the mesh-vertex count for CG1 but exceeds it for CG2+.
    # Interpolate y onto each species' subspace so the mask aligns with
    # the DOF count of c_data for any element order.
    if any(z_vals[i] > 0 and c_bulk[i] > 0 for i in range(n_species)):
        import firedrake as fd

        V_full = U.function_space()
        mesh = V_full.mesh()
        ydim = mesh.geometric_dimension() - 1
        for i in range(n_species):
            if not (z_vals[i] > 0 and c_bulk[i] > 0):
                continue
            try:
                V_i = V_full.sub(i).collapse()
                y_fn = fd.Function(V_i)
                y_fn.interpolate(fd.SpatialCoordinate(mesh)[ydim])
                y_coords = y_fn.dat.data_ro
            except Exception:
                # Fallback for unusual function-space configurations:
                # use mesh vertex coordinates (correct for CG1 only).
                coords = mesh.coordinates.dat.data_ro
                y_coords = coords[:, ydim] if coords.ndim == 2 else coords
            c_data = _conc_array(i)
            if y_coords.shape != c_data.shape:
                warns.append(
                    f"W5: skipped for {names[i]} (DOF/coord shape mismatch: "
                    f"y={y_coords.shape} vs c={c_data.shape})"
                )
                continue
            y_median = float(np.median(y_coords))
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
