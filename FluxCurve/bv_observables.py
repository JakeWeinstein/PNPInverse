"""Observable form builders for BV-boundary current-density inference."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


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


def _build_bv_scalar_target_in_control_space(
    ctx: Dict[str, object],
    value: float,
    *,
    name: str,
    control_mode: str = "k0",
):
    """Create a scalar Function in the same R-space used by BV controls.

    Parameters
    ----------
    control_mode : str
        ``"k0"`` uses ``bv_k0_funcs``, ``"alpha"`` uses ``bv_alpha_funcs``,
        ``"joint"`` uses ``bv_k0_funcs`` (both live in R-space anyway).
    """
    import firedrake as fd

    if control_mode == "alpha":
        funcs = list(ctx.get("bv_alpha_funcs", []))
        if not funcs:
            raise ValueError("Context has no bv_alpha control functions.")
    else:
        funcs = list(ctx["bv_k0_funcs"])
        if not funcs:
            raise ValueError("Context has no bv_k0 control functions.")
    control_space = funcs[0].function_space()
    target = fd.Function(control_space, name=name)
    target.assign(float(value))
    return target


def _bv_gradient_controls_to_array(
    raw_gradient: object, n_controls: int
) -> np.ndarray:
    """Convert adjoint gradient output to dense numpy vector."""
    if isinstance(raw_gradient, (list, tuple)):
        grads = list(raw_gradient)
    else:
        grads = [raw_gradient]

    if len(grads) != n_controls:
        import warnings
        warnings.warn(
            f"Expected {n_controls} control gradients but got {len(grads)}; "
            f"truncating/padding to match",
            stacklevel=2,
        )

    out = np.zeros(n_controls, dtype=float)
    for i in range(min(n_controls, len(grads))):
        gi = grads[i]
        if hasattr(gi, "dat"):
            out[i] = float(gi.dat.data_ro[0])
        else:
            out[i] = float(gi)
    return out
