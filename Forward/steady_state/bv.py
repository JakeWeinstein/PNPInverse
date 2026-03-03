"""Butler-Volmer (BV) steady-state solve and voltage sweep.

This module handles BV-condition experiments where the measured output is
the current density on the electrode boundary at steady state.
"""

from __future__ import annotations

from typing import Any, Sequence

import firedrake as fd
import numpy as np

from Inverse.solver_interface import as_species_list, deep_copy_solver_params

from .common import (
    SteadyStateConfig,
    SteadyStateResult,
    _maybe_stop_annotating,
)


def configure_bv_solver_params(
    base_solver_params: Sequence[Any],
    *,
    phi_applied: float,
    k0_values: Sequence[float] | None = None,
    alpha_values: Sequence[float] | None = None,
    a_values: Sequence[float] | None = None,
) -> Any:
    """Return a solver-parameter copy with applied voltage and optional k0/alpha/a.

    Sets ``phi_applied`` and ``phi0`` to the given voltage.  If *k0_values*
    is given, injects them into the ``bv_bc`` config (multi-reaction
    ``reactions[j].k0`` or legacy ``bv_bc.k0``).  If *alpha_values* is given,
    injects them into ``reactions[j].alpha`` or legacy ``bv_bc.alpha``.
    If *a_values* is given, sets the Bikerman steric parameter (a_vals).
    """
    import copy as _copy
    params = deep_copy_solver_params(base_solver_params)

    # Access solver_options via named attribute or index for backward compat.
    from Forward.params import SolverParams as _SP
    if isinstance(params, _SP):
        opts = _copy.deepcopy(params.solver_options)

        if k0_values is not None:
            if not isinstance(opts, dict):
                raise ValueError("solver_options must be a dict when setting BV k0.")
            bv_cfg = opts.get("bv_bc", {})
            if not isinstance(bv_cfg, dict):
                raise ValueError("solver_options['bv_bc'] must be a dict when setting BV k0.")
            k0_list = [float(v) for v in k0_values]
            reactions = bv_cfg.get("reactions")
            if reactions is not None and isinstance(reactions, list):
                for j, rxn in enumerate(reactions):
                    if j < len(k0_list):
                        rxn["k0"] = k0_list[j]
            if "k0" in bv_cfg:
                existing = bv_cfg["k0"]
                if isinstance(existing, list) and len(existing) == len(k0_list):
                    bv_cfg["k0"] = k0_list

        if alpha_values is not None:
            if not isinstance(opts, dict):
                raise ValueError("solver_options must be a dict when setting BV alpha.")
            bv_cfg = opts.get("bv_bc", {})
            if not isinstance(bv_cfg, dict):
                raise ValueError("solver_options['bv_bc'] must be a dict when setting BV alpha.")
            alpha_list = [float(v) for v in alpha_values]
            reactions = bv_cfg.get("reactions")
            if reactions is not None and isinstance(reactions, list):
                for j, rxn in enumerate(reactions):
                    if j < len(alpha_list):
                        rxn["alpha"] = alpha_list[j]
            if "alpha" in bv_cfg:
                existing = bv_cfg["alpha"]
                if isinstance(existing, list) and len(existing) == len(alpha_list):
                    bv_cfg["alpha"] = alpha_list

        new_a_vals = params.a_vals
        if a_values is not None:
            n_species = params.n_species
            a_list = [float(v) for v in a_values]
            if len(a_list) != n_species:
                raise ValueError(
                    f"a_values length {len(a_list)} != n_species {n_species}"
                )
            new_a_vals = a_list

        import dataclasses
        params = dataclasses.replace(
            params,
            phi_applied=float(phi_applied),
            phi0=float(phi_applied),
            a_vals=new_a_vals,
            solver_options=opts,
        )
    else:
        # Legacy list path
        params[7] = float(phi_applied)
        params[9] = float(phi_applied)

        if k0_values is not None:
            p = params[10]
            if not isinstance(p, dict):
                raise ValueError("solver_params[10] must be a dict when setting BV k0.")
            bv_cfg = p.get("bv_bc", {})
            if not isinstance(bv_cfg, dict):
                raise ValueError("params['bv_bc'] must be a dict when setting BV k0.")
            k0_list = [float(v) for v in k0_values]
            reactions = bv_cfg.get("reactions")
            if reactions is not None and isinstance(reactions, list):
                for j, rxn in enumerate(reactions):
                    if j < len(k0_list):
                        rxn["k0"] = k0_list[j]
            if "k0" in bv_cfg:
                existing = bv_cfg["k0"]
                if isinstance(existing, list) and len(existing) == len(k0_list):
                    bv_cfg["k0"] = k0_list

        if alpha_values is not None:
            p = params[10]
            if not isinstance(p, dict):
                raise ValueError("solver_params[10] must be a dict when setting BV alpha.")
            bv_cfg = p.get("bv_bc", {})
            if not isinstance(bv_cfg, dict):
                raise ValueError("params['bv_bc'] must be a dict when setting BV alpha.")
            alpha_list = [float(v) for v in alpha_values]
            reactions = bv_cfg.get("reactions")
            if reactions is not None and isinstance(reactions, list):
                for j, rxn in enumerate(reactions):
                    if j < len(alpha_list):
                        rxn["alpha"] = alpha_list[j]
            if "alpha" in bv_cfg:
                existing = bv_cfg["alpha"]
                if isinstance(existing, list) and len(existing) == len(alpha_list):
                    bv_cfg["alpha"] = alpha_list

        if a_values is not None:
            n_species = int(params[0])
            a_list = [float(v) for v in a_values]
            if len(a_list) != n_species:
                raise ValueError(
                    f"a_values length {len(a_list)} != n_species {n_species}"
                )
            params[6] = a_list
    return params


def compute_bv_reaction_rates(ctx: dict[str, Any]) -> list[float]:
    """Assemble dimensionless BV reaction rates R_j on the electrode boundary.

    Returns a list of floats, one per reaction (multi-reaction) or per species
    (legacy).  Each value is ``R_j * ds(electrode)``.
    """
    bv_rate_exprs = ctx.get("bv_rate_exprs", [])
    bv_cfg = ctx.get("bv_settings", {})
    electrode_marker = int(bv_cfg.get("electrode_marker", 1))
    ds = fd.Measure("ds", domain=ctx["mesh"])

    rates: list[float] = []
    for R_j in bv_rate_exprs:
        rates.append(float(fd.assemble(R_j * ds(electrode_marker))))
    return rates


def compute_bv_current_density(
    ctx: dict[str, Any],
    *,
    i_scale: float = 1.0,
) -> float:
    """Compute total BV current density from assembled reaction rates.

    Returns ``I = -(sum R_j) * i_scale`` in whatever units ``i_scale`` is
    chosen to give (e.g. mA/cm² when ``i_scale = n_e * F * D_ref * c_bulk / L_ref * 0.1``).
    """
    rates = compute_bv_reaction_rates(ctx)
    return -sum(rates) * i_scale


def solve_bv_to_steady_state_for_phi_applied(
    solver_params: Sequence[Any],
    *,
    steady: SteadyStateConfig,
    blob_initial_condition: bool = False,
    mesh: Any = None,
    i_scale: float = 1.0,
) -> SteadyStateResult:
    """Run one BV forward solve until current-density-defined steady state.

    Parameters
    ----------
    solver_params:
        11-element list / SolverParams.
    steady:
        Steady-state convergence settings.
    mesh:
        Optional pre-built mesh (e.g. graded rectangle).
    i_scale:
        Conversion factor from dimensionless BV rate to physical current density.

    Returns
    -------
    SteadyStateResult
        With ``observed_flux`` in current-density units.
    """
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )

    params = deep_copy_solver_params(solver_params)
    dt = float(params.dt)
    phi_applied = float(params.phi_applied)
    n_species = int(params.n_species)

    with _maybe_stop_annotating():
        ctx = bv_build_context(params, mesh=mesh)
        ctx = bv_build_forms(ctx, params)
        bv_set_initial_conditions(ctx, params, blob=blob_initial_condition)

        U = ctx["U"]
        U_prev = ctx["U_prev"]
        F_res = ctx["F_res"]
        bcs = ctx["bcs"]

        jac = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params.solver_options)

        prev_current: float | None = None
        steady_count = 0
        rel_metric: float | None = None
        abs_metric: float | None = None
        current_density = float("nan")

        for step in range(1, max(1, int(steady.max_steps)) + 1):
            try:
                solver.solve()
            except Exception as exc:
                return SteadyStateResult(
                    phi_applied=phi_applied,
                    converged=False,
                    steps_taken=step,
                    final_time=step * dt,
                    species_flux=[],
                    observed_flux=float("nan"),
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )

            U_prev.assign(U)
            current_density = compute_bv_current_density(ctx, i_scale=i_scale)

            if prev_current is not None:
                delta = abs(current_density - prev_current)
                scale = max(abs(current_density), abs(prev_current),
                            float(max(steady.absolute_tolerance, 1e-16)))
                rel_metric = delta / scale
                abs_metric = delta

                is_steady = (
                    rel_metric <= float(steady.relative_tolerance)
                    or abs_metric <= float(steady.absolute_tolerance)
                )
                steady_count = steady_count + 1 if is_steady else 0
            else:
                steady_count = 0

            if (steady.verbose and int(steady.print_every) > 0
                    and step % int(steady.print_every) == 0):
                print(
                    f"[bv_steady] phi_applied={phi_applied:>9.4f} step={step:>4d} "
                    f"rel_change={(rel_metric if rel_metric is not None else float('nan')):>10.3e} "
                    f"I={current_density:>12.6f}"
                )

            prev_current = current_density

            if steady_count >= int(max(1, steady.consecutive_steps)):
                rates = compute_bv_reaction_rates(ctx)
                return SteadyStateResult(
                    phi_applied=phi_applied,
                    converged=True,
                    steps_taken=step,
                    final_time=step * dt,
                    species_flux=rates,
                    observed_flux=float(current_density),
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    failure_reason="",
                )

        rates = compute_bv_reaction_rates(ctx)
        return SteadyStateResult(
            phi_applied=phi_applied,
            converged=False,
            steps_taken=int(max(1, steady.max_steps)),
            final_time=float(max(1, steady.max_steps)) * dt,
            species_flux=rates,
            observed_flux=float(current_density),
            final_relative_change=rel_metric,
            final_absolute_change=abs_metric,
            failure_reason="steady-state criterion not satisfied before max_steps",
        )


def sweep_phi_applied_steady_bv_flux(
    base_solver_params: Sequence[Any],
    *,
    phi_applied_values: Sequence[float],
    steady: SteadyStateConfig,
    k0_values: Sequence[float] | None = None,
    i_scale: float = 1.0,
    mesh: Any = None,
    blob_initial_condition: bool = False,
) -> list[SteadyStateResult]:
    """Generate a phi_applied-vs-steady-state-current-density curve for BV.

    Uses warm-start (voltage continuation): builds context once, sweeps
    phi_applied values sequentially, carrying the converged solution forward
    as IC for the next point.  This is essential for BV convergence at
    large overpotentials.
    """
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )

    # Build context once at the first phi_applied value.
    first_phi = float(phi_applied_values[0]) if len(phi_applied_values) > 0 else 0.0
    params0 = configure_bv_solver_params(
        base_solver_params, phi_applied=first_phi, k0_values=k0_values,
    )
    dt = float(params0.dt)

    with _maybe_stop_annotating():
        ctx = bv_build_context(params0, mesh=mesh)
        ctx = bv_build_forms(ctx, params0)
        bv_set_initial_conditions(ctx, params0, blob=blob_initial_condition)

        U = ctx["U"]
        U_prev = ctx["U_prev"]
        F_res = ctx["F_res"]
        bcs = ctx["bcs"]

        jac = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params0.solver_options)

        U_scratch = U.copy(deepcopy=True)

        out: list[SteadyStateResult] = []
        for phi_applied in phi_applied_values:
            ctx["phi_applied_func"].assign(float(phi_applied))

            prev_current: float | None = None
            steady_count = 0
            rel_metric: float | None = None
            abs_metric: float | None = None
            current_density = float("nan")
            failed = False
            failure_reason = ""
            steps_taken = 0

            for step in range(1, max(1, int(steady.max_steps)) + 1):
                steps_taken = step
                U_scratch.assign(U)
                try:
                    solver.solve()
                except Exception as exc:
                    failed = True
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    U.assign(U_prev)
                    break

                U_prev.assign(U)
                current_density = compute_bv_current_density(ctx, i_scale=i_scale)

                if prev_current is not None:
                    delta = abs(current_density - prev_current)
                    scale_val = max(abs(current_density), abs(prev_current),
                                    float(max(steady.absolute_tolerance, 1e-16)))
                    rel_metric = delta / scale_val
                    abs_metric = delta

                    is_steady = (
                        rel_metric <= float(steady.relative_tolerance)
                        or abs_metric <= float(steady.absolute_tolerance)
                    )
                    steady_count = steady_count + 1 if is_steady else 0
                else:
                    steady_count = 0

                prev_current = current_density

                if steady_count >= int(max(1, steady.consecutive_steps)):
                    break

            converged = (not failed and steady_count >= int(max(1, steady.consecutive_steps)))
            if not failed and not converged:
                failure_reason = "steady-state criterion not satisfied before max_steps"

            rates = compute_bv_reaction_rates(ctx)
            out.append(SteadyStateResult(
                phi_applied=float(phi_applied),
                converged=converged,
                steps_taken=steps_taken,
                final_time=steps_taken * dt,
                species_flux=rates,
                observed_flux=float(current_density),
                final_relative_change=rel_metric,
                final_absolute_change=abs_metric,
                failure_reason=failure_reason,
            ))

        return out
