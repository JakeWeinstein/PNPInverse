"""Observable form builders for BV-boundary current-density computation.

Moved from FluxCurve/bv_observables.py so that Forward-layer modules
(e.g. grid_charge_continuation) can use observable-based convergence
without importing from FluxCurve.
"""

from __future__ import annotations

from typing import Dict, Optional


def _build_bv_observable_form(
    ctx: Dict[str, object],
    *,
    mode: str,
    reaction_index: Optional[int],
    scale: float,
) -> object:
    """Build scalar UFL observable from BV reaction rate expressions.

    The observable is current density = scale * (combination of R_j) * ds(electrode).

    Supported modes:
    - ``"current_density"``: total current = scale * sum_j R_j * ds
    - ``"peroxide_current"``: net peroxide = scale * (R_0 - R_1) * ds
    - ``"reaction"``: single reaction = scale * R_{reaction_index} * ds
    """
    import firedrake as fd

    mode_norm = str(mode).strip().lower()
    bv_rate_exprs = list(ctx["bv_rate_exprs"])
    bv_cfg = ctx.get("bv_settings", {})
    electrode_marker = int(bv_cfg.get("electrode_marker", 1))
    ds = fd.Measure("ds", domain=ctx["mesh"])
    scale_const = fd.Constant(float(scale))

    if mode_norm == "current_density":
        # Total current: sum of all reaction rates
        rate_sum = 0
        for R_j in bv_rate_exprs:
            rate_sum += R_j
        return scale_const * rate_sum * ds(electrode_marker)

    elif mode_norm == "peroxide_current":
        # Net peroxide current: R_0 - R_1 (production - consumption)
        if len(bv_rate_exprs) < 2:
            raise ValueError(
                "peroxide_current mode requires at least 2 reactions; "
                f"got {len(bv_rate_exprs)}."
            )
        return scale_const * (bv_rate_exprs[0] - bv_rate_exprs[1]) * ds(electrode_marker)

    elif mode_norm == "reaction":
        if reaction_index is None:
            raise ValueError("reaction_index must be set when mode='reaction'.")
        idx = int(reaction_index)
        if idx < 0 or idx >= len(bv_rate_exprs):
            raise ValueError(
                f"reaction_index {idx} out of bounds for {len(bv_rate_exprs)} reactions."
            )
        return scale_const * bv_rate_exprs[idx] * ds(electrode_marker)

    else:
        raise ValueError(
            f"Unknown BV observable mode '{mode}'. "
            "Use 'current_density', 'peroxide_current', or 'reaction'."
        )


def assemble_observable_validated(
    form: object,
    *,
    I_lim: float,
    phi_applied: float,
    V_T: float,
    mode: str = "current_density",
) -> tuple[float, object]:
    """Assemble an observable form and validate the result.

    Parameters
    ----------
    form : ufl.Form
        The observable form to assemble.
    I_lim : float
        Diffusion-limited current (same units as assembled value).
    phi_applied : float
        Applied potential (nondimensional).
    V_T : float
        Thermal voltage for sign convention.
    mode : str
        "current_density" or "peroxide_current" — affects which checks run.

    Returns
    -------
    tuple[float, ValidationResult]
        (assembled_value, validation_result)
    """
    import firedrake as fd
    from .validation import ValidationResult

    value = float(fd.assemble(form))

    # Only check F2 (magnitude exceeds diffusion limit) for single-observable
    # validation.  Cross-observable checks (F3, F7) require both cd and pc,
    # which happens at the pipeline level (training.py, bv_curve_eval.py).
    failures = []
    warnings_list = []

    if mode in ("current_density", "peroxide_current"):
        if abs(value) > abs(I_lim) * 1.05:
            failures.append(f"F2: |{mode}|={abs(value):.4g} exceeds I_lim={abs(I_lim):.4g}")

    result = ValidationResult(valid=len(failures) == 0, failures=failures, warnings=warnings_list)
    return value, result
