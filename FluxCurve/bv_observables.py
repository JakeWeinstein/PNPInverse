"""Observable form builders for BV-boundary current-density inference.

The core _build_bv_observable_form function now lives in
Forward.bv_solver.observables.  This module re-exports it for backward
compatibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from Forward.bv_solver.observables import _build_bv_observable_form  # noqa: F401 — re-export


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
        raise ValueError(
            f"Expected {n_controls} gradient components but got {len(grads)}"
        )

    out = np.zeros(n_controls, dtype=float)
    for i in range(n_controls):
        gi = grads[i]
        if hasattr(gi, "dat"):
            out[i] = float(gi.dat.data_ro[0])
        else:
            out[i] = float(gi)
    return out
