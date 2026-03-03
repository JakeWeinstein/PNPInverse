"""Robin-boundary steady-state solve and voltage sweep.

This module handles Robin-condition flux experiments where the measured output
is the per-species flux on the Robin boundary at steady state.
"""

from __future__ import annotations

from typing import Any, Sequence

import firedrake as fd
import numpy as np

from Inverse.solver_interface import as_species_list, deep_copy_solver_params
from Forward.robin_solver import build_context, build_forms, set_initial_conditions

from .common import (
    SteadyStateConfig,
    SteadyStateResult,
    observed_flux_from_species_flux,
    _maybe_stop_annotating,
)


def configure_robin_solver_params(
    base_solver_params: Sequence[Any],
    *,
    phi_applied: float,
    kappa_values: Sequence[float] | None = None,
) -> list[Any]:
    """Return a solver-parameter copy with applied voltage and optional kappa.

    Notes
    -----
    In the Robin forward solver, boundary voltage is read from slot 7
    (``phi_applied``). We also mirror it into slot 9 (``phi0``) for interface
    consistency and easier inspection in logs.
    """
    import copy as _copy
    params = deep_copy_solver_params(base_solver_params)

    from Forward.params import SolverParams as _SP
    if isinstance(params, _SP):
        import dataclasses
        opts = _copy.deepcopy(params.solver_options)
        n_species = params.n_species
        if kappa_values is not None:
            if not isinstance(opts, dict):
                raise ValueError("solver_options must be a dict when setting robin kappa.")
            robin_cfg = opts.setdefault("robin_bc", {})
            if not isinstance(robin_cfg, dict):
                raise ValueError("solver_options['robin_bc'] must be a dict when setting robin kappa.")
            robin_cfg["kappa"] = as_species_list(kappa_values, n_species, "robin_kappa")
        params = dataclasses.replace(
            params,
            phi_applied=float(phi_applied),
            phi0=float(phi_applied),
            solver_options=opts,
        )
    else:
        n_species = int(params[0])
        params[7] = float(phi_applied)  # Applied boundary voltage used by robin_solver.
        params[9] = float(phi_applied)  # Kept in sync for human readability.
        if kappa_values is not None:
            p = params[10]
            if not isinstance(p, dict):
                raise ValueError("solver_params[10] must be a dict when setting robin kappa.")
            robin_cfg = p.setdefault("robin_bc", {})
            if not isinstance(robin_cfg, dict):
                raise ValueError("params['robin_bc'] must be a dict when setting robin kappa.")
            robin_cfg["kappa"] = as_species_list(kappa_values, n_species, "robin_kappa")
    return params


def compute_species_flux_on_robin_boundary(ctx: dict[str, Any]) -> list[float]:
    """Assemble outward species fluxes across the configured Robin boundary.

    The Robin condition in this code is:
        ``J_i · n = kappa_i * (c_i - c_inf_i)``
    so each species flux is assembled directly from this boundary expression.
    """
    n = int(ctx["n_species"])
    robin = ctx["robin_settings"]
    electrode_marker = int(robin["electrode_marker"])
    c_inf_vals = [float(v) for v in robin["c_inf_vals"]]
    kappa_funcs = list(ctx["kappa_funcs"])
    ci = fd.split(ctx["U"])[:-1]
    ds = fd.Measure("ds", domain=ctx["mesh"])

    fluxes: list[float] = []
    for i in range(n):
        form = kappa_funcs[i] * (ci[i] - fd.Constant(c_inf_vals[i])) * ds(electrode_marker)
        fluxes.append(float(fd.assemble(form)))
    return fluxes


def solve_to_steady_state_for_phi_applied(
    solver_params: Sequence[Any],
    *,
    steady: SteadyStateConfig,
    blob_initial_condition: bool = False,
) -> SteadyStateResult:
    """Run one Robin forward solve until flux-defined steady state is reached.

    Steady-state metric:
    - Per step, compute per-species fluxes on the Robin boundary.
    - Let ``delta = |J_n - J_{n-1}|`` (vector).
    - Relative metric is ``max(delta / max(|J_n|, |J_{n-1}|, abs_tol))``.
    - Absolute metric is ``max(delta)``.
    - A step is considered "steady" when either metric passes tolerance:
      ``rel_metric <= relative_tolerance`` OR
      ``abs_metric <= absolute_tolerance``.
    - Convergence requires ``consecutive_steps`` steady steps in a row.
    """
    params = deep_copy_solver_params(solver_params)
    dt = float(params.dt)
    phi_applied = float(params.phi_applied)
    z_vals = as_species_list(params.z_vals, params.n_species, "z_vals")

    with _maybe_stop_annotating():
        ctx = build_context(params)
        ctx = build_forms(ctx, params)
        set_initial_conditions(ctx, params, blob=blob_initial_condition)

        U = ctx["U"]
        U_prev = ctx["U_prev"]
        F_res = ctx["F_res"]
        bcs = ctx["bcs"]

        jac = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params.solver_options)

        prev_flux: np.ndarray | None = None
        steady_count = 0
        rel_metric: float | None = None
        abs_metric: float | None = None
        species_flux = np.zeros(params.n_species, dtype=float)

        for step in range(1, max(1, int(steady.max_steps)) + 1):
            try:
                solver.solve()
            except Exception as exc:
                return SteadyStateResult(
                    phi_applied=phi_applied,
                    converged=False,
                    steps_taken=step,
                    final_time=step * dt,
                    species_flux=species_flux.tolist(),
                    observed_flux=float("nan"),
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )

            U_prev.assign(U)

            species_flux = np.asarray(compute_species_flux_on_robin_boundary(ctx), dtype=float)

            if prev_flux is not None:
                delta = np.abs(species_flux - prev_flux)
                scale = np.maximum(
                    np.maximum(np.abs(species_flux), np.abs(prev_flux)),
                    float(max(steady.absolute_tolerance, 1e-16)),
                )
                rel_metric = float(np.max(delta / scale))
                abs_metric = float(np.max(delta))

                is_steady = (
                    rel_metric <= float(steady.relative_tolerance)
                    or abs_metric <= float(steady.absolute_tolerance)
                )
                steady_count = steady_count + 1 if is_steady else 0
            else:
                steady_count = 0

            if steady.verbose and int(steady.print_every) > 0 and step % int(steady.print_every) == 0:
                obs = observed_flux_from_species_flux(
                    species_flux,
                    z_vals=z_vals,
                    flux_observable=steady.flux_observable,
                    species_index=steady.species_index,
                )
                print(
                    f"[steady] phi_applied={phi_applied:>9.4f} step={step:>4d} "
                    f"rel_change={(rel_metric if rel_metric is not None else float('nan')):>10.3e} "
                    f"abs_change={(abs_metric if abs_metric is not None else float('nan')):>10.3e} "
                    f"flux={obs:>12.6f}"
                )

            prev_flux = species_flux.copy()

            if steady_count >= int(max(1, steady.consecutive_steps)):
                observed = observed_flux_from_species_flux(
                    species_flux,
                    z_vals=z_vals,
                    flux_observable=steady.flux_observable,
                    species_index=steady.species_index,
                )
                return SteadyStateResult(
                    phi_applied=phi_applied,
                    converged=True,
                    steps_taken=step,
                    final_time=step * dt,
                    species_flux=species_flux.tolist(),
                    observed_flux=float(observed),
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    failure_reason="",
                )

        observed = observed_flux_from_species_flux(
            species_flux,
            z_vals=z_vals,
            flux_observable=steady.flux_observable,
            species_index=steady.species_index,
        )
        return SteadyStateResult(
            phi_applied=phi_applied,
            converged=False,
            steps_taken=int(max(1, steady.max_steps)),
            final_time=float(max(1, steady.max_steps)) * dt,
            species_flux=species_flux.tolist(),
            observed_flux=float(observed),
            final_relative_change=rel_metric,
            final_absolute_change=abs_metric,
            failure_reason="steady-state criterion not satisfied before max_steps",
        )


def sweep_phi_applied_steady_flux(
    base_solver_params: Sequence[Any],
    *,
    phi_applied_values: Sequence[float],
    steady: SteadyStateConfig,
    kappa_values: Sequence[float] | None = None,
    blob_initial_condition: bool = False,
) -> list[SteadyStateResult]:
    """Generate a phi_applied-vs-steady-state-flux curve for one kappa setting."""
    out: list[SteadyStateResult] = []
    for phi_applied in phi_applied_values:
        params = configure_robin_solver_params(
            base_solver_params,
            phi_applied=float(phi_applied),
            kappa_values=kappa_values,
        )
        result = solve_to_steady_state_for_phi_applied(
            params,
            steady=steady,
            blob_initial_condition=blob_initial_condition,
        )
        out.append(result)
    return out
