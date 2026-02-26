"""Observable form builders for Robin-boundary flux inference."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from Nondim.constants import FARADAY_CONSTANT


def _build_species_boundary_flux_forms(
    ctx: Dict[str, object],
    *,
    state: Optional[object] = None,
) -> List[object]:
    """Build integrated Robin-boundary flux forms for a selected mixed state."""
    import firedrake as fd

    n_species = int(ctx["n_species"])
    robin = ctx["robin_settings"]
    electrode_marker = int(robin["electrode_marker"])
    c_inf_vals = [float(v) for v in robin["c_inf_vals"]]
    kappa_funcs = list(ctx["kappa_funcs"])
    mixed_state = ctx["U"] if state is None else state
    ci = fd.split(mixed_state)[:-1]
    ds = fd.Measure("ds", domain=ctx["mesh"])

    forms: List[object] = []
    for i in range(n_species):
        forms.append(kappa_funcs[i] * (ci[i] - fd.Constant(c_inf_vals[i])) * ds(electrode_marker))
    return forms


def _build_observable_form(
    ctx: Dict[str, object],
    *,
    mode: str,
    species_index: Optional[int],
    scale: float,
    state: Optional[object] = None,
) -> object:
    """Build scalar observable form from species boundary flux forms.

    Supported modes:
    - ``total_species``: sum_i flux_i
    - ``total_charge``: F * sum_i z_i * flux_i
    - ``charge_proxy_no_f``: sum_i z_i * flux_i (Faraday scaling omitted)
    - ``species``: flux_species_index
    """
    import firedrake as fd

    mode_norm = str(mode).strip().lower()
    forms = _build_species_boundary_flux_forms(ctx, state=state)
    n_species = len(forms)

    if mode_norm == "total_species":
        out = 0
        for form_i in forms:
            out += form_i
    elif mode_norm == "total_charge":
        z_consts = list(ctx.get("z_consts", []))
        if len(z_consts) != n_species:
            raise ValueError(
                f"z_consts length {len(z_consts)} does not match species count {n_species}."
            )
        out = 0
        for i in range(n_species):
            out += z_consts[i] * forms[i]
        out = fd.Constant(float(FARADAY_CONSTANT)) * out
    elif mode_norm == "charge_proxy_no_f":
        z_consts = list(ctx.get("z_consts", []))
        if len(z_consts) != n_species:
            raise ValueError(
                f"z_consts length {len(z_consts)} does not match species count {n_species}."
            )
        # Provisional observable for parameter studies: charge-weighted species
        # flux without Faraday scaling. Units are intentionally treated as a.u.
        # until physical current-density calibration is finalized.
        out = 0
        for i in range(n_species):
            out += z_consts[i] * forms[i]
    elif mode_norm == "species":
        if species_index is None:
            raise ValueError("species_index must be set when observable_mode='species'.")
        idx = int(species_index)
        if idx < 0 or idx >= n_species:
            raise ValueError(
                f"species_index {idx} out of bounds for n_species={n_species}."
            )
        out = forms[idx]
    else:
        raise ValueError(
            f"Unknown observable_mode '{mode}'. "
            "Use 'total_species', 'total_charge', 'charge_proxy_no_f', or 'species'."
        )

    return fd.Constant(float(scale)) * out


def _build_scalar_target_in_control_space(
    ctx: Dict[str, object], value: float, *, name: str
):
    """Create a scalar Function in the same R-space used by kappa controls."""
    import firedrake as fd

    kappa_funcs = list(ctx["kappa_funcs"])
    if not kappa_funcs:
        raise ValueError("Context has no kappa control functions.")
    control_space = kappa_funcs[0].function_space()
    target = fd.Function(control_space, name=name)
    target.assign(float(value))
    return target


def _gradient_controls_to_array(raw_gradient: object, n_species: int) -> np.ndarray:
    """Convert adjoint gradient output to dense numpy vector."""
    if isinstance(raw_gradient, (list, tuple)):
        grads = list(raw_gradient)
    else:
        grads = [raw_gradient]

    out = np.zeros(n_species, dtype=float)
    for i in range(min(n_species, len(grads))):
        gi = grads[i]
        if hasattr(gi, "dat"):
            out[i] = float(gi.dat.data_ro[0])
        else:
            out[i] = float(gi)
    return out
