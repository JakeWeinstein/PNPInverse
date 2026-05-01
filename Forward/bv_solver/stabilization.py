"""Artificial diffusion stabilization for the BV-PNP solver.

Adds a streamline-diffusion term to prevent co-ion negativity in the
underresolved Debye layer. Compatible with pyadjoint annotation —
the stabilization term is added to F_res as a standard UFL form,
so adjoint gradients propagate through it correctly.

Usage::

    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    ctx = add_stabilization(ctx, sp)  # <-- adds D_art term
    set_initial_conditions(ctx, sp)

    # Everything downstream (solver, adjoint, etc.) uses the stabilized F_res.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import firedrake as fd


def add_stabilization(
    ctx: Dict[str, Any],
    solver_params: Sequence[Any],
    *,
    d_art_scale: float = 0.001,
    stabilized_species: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Add artificial diffusion stabilization to an existing BV-PNP context.

    This function modifies ``ctx["F_res"]`` in-place by adding a streamline
    artificial diffusion term for the specified species. The term is:

        D_art = d_art_scale * h * |z_i * D_i * em| * |∇φ|

    added as ``D_art * ∇c_i · ∇v_i * dx`` to the weak form.

    Parameters
    ----------
    ctx : dict
        Context from ``build_forms``. Modified in-place.
    solver_params : list or SolverParams
        11-element solver parameter set.
    d_art_scale : float
        Artificial diffusion strength. Default 0.001 gives ~2% error
        vs unstabilized at the overlap voltage (V_RHE=0.10V).
    stabilized_species : list of int, optional
        Which species indices to stabilize. Default: ``[3]`` (ClO4- only).
        Only species with z != 0 actually get stabilization.

    Returns
    -------
    dict
        The same ``ctx`` with modified ``F_res`` and ``J_form``.
    """
    if stabilized_species is None:
        stabilized_species = [3]  # ClO4- only by default

    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception:
        # SolverParams object
        sp = list(solver_params)
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = sp

    mesh = ctx["mesh"]
    W = ctx["W"]
    U = ctx["U"]
    n = ctx["n_species"]
    scaling = ctx["nondim"]

    em = float(scaling["electromigration_prefactor"])
    D_model_vals = scaling["D_model_vals"]

    dx = fd.Measure("dx", domain=mesh)
    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    v_tests = fd.TestFunctions(W)

    h = fd.CellSize(mesh)
    F_res = ctx["F_res"]

    for i in stabilized_species:
        if i >= n:
            continue
        z_i = float(z_vals[i])
        if abs(z_i) == 0:
            continue

        D_i = float(D_model_vals[i])
        drift_speed = fd.Constant(abs(z_i) * D_i * em)

        # Streamline artificial diffusion: proportional to mesh size * drift velocity
        # |∇φ| is the potential gradient magnitude
        grad_phi_mag = fd.sqrt(fd.dot(fd.grad(phi), fd.grad(phi)) + fd.Constant(1e-10))
        D_art = fd.Constant(d_art_scale) * h * drift_speed * grad_phi_mag

        F_res += D_art * fd.dot(fd.grad(ci[i]), fd.grad(v_tests[i])) * dx

    # Update the Jacobian to include the stabilization term
    ctx["F_res"] = F_res
    ctx["J_form"] = fd.derivative(F_res, U)
    # Mark that stabilization was applied
    ctx["stabilization"] = {
        "d_art_scale": d_art_scale,
        "stabilized_species": stabilized_species,
    }

    return ctx
