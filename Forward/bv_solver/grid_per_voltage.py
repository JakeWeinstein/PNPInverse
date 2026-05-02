"""Per-voltage cold-start with internal z-ramp + warm-fallback (C+D).

This module orchestrates the BV-PNP forward solver across a voltage
grid using the strategy that ``writeups/WeekOfApr27/PNP Inverse Solver
Revised.tex`` empirically validates for the production
3sp+Boltzmann+log-c+log-rate stack:

    Phase 1 (C): per-voltage cold-start with internal z-ramp.
        For each V_RHE in the grid (independently), build a fresh
        context at the target voltage, set every charge factor to 0,
        run SS, then ramp z from 0 to 1 in linear steps with
        checkpoint+rollback on failure.  Each voltage stands or falls
        on its own — failures don't cascade.

    Phase 2 (D): warm-walk from nearest converged neighbor.
        For voltages where Phase 1 didn't reach z=1, walk outward
        from the cold-success block at full z by stepping
        phi_applied from V_anchor toward V_target in N substeps,
        recursively bisecting on substep failures up to a depth cap.

This is the orchestration that v24 demonstrates covers
``V_RHE in [-0.50, +0.10] V`` with 8/8 PASS at the 5% F2 tolerance.
The wrapper here generalises it to arbitrary voltage grids and routes
through the formulation dispatcher so it works for both the
concentration backend (``forms.py``) and the log-c backend
(``forms_logc.py`` + Boltzmann counterion + log-rate BV).

Public API
----------
solve_grid_per_voltage_cold_with_warm_fallback(
    solver_params,
    *,
    phi_applied_values,
    mesh=None,
    max_z_steps=20,
    n_substeps_warm=4,
    bisect_depth_warm=3,
    ss_rel_tol=1e-4,
    ss_abs_tol=1e-8,
    ss_consec=4,
    max_ss_steps_cold=200,
    max_ss_steps_z=120,
    max_ss_steps_warm=150,
    max_ss_steps_warm_final=200,
    dt_init=0.25,
    dt_growth_cap=4.0,
    dt_max_ratio=20.0,
    print_interval=1,
    per_point_callback=None,
) -> PerVoltageContinuationResult
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PerVoltagePointResult:
    """Result for a single voltage point."""

    orig_idx: int
    phi_applied: float
    U_data: Optional[tuple]      # tuple(d.data_ro.copy() for d in ctx['U'].dat)
    achieved_z_factor: float
    converged: bool              # True iff reached z = 1
    method: str                  # "cold" | f"warm<-{V_anchor:+.3f}" | "missing"


@dataclasses.dataclass(frozen=True)
class PerVoltageContinuationResult:
    """Aggregate result for the full grid."""

    points: Dict[int, PerVoltagePointResult]
    mesh_dof_count: int

    def all_converged(self) -> bool:
        return all(p.converged for p in self.points.values())

    def get_U_data(self, idx: int) -> Optional[tuple]:
        return self.points[idx].U_data

    def converged_indices(self) -> list[int]:
        return [i for i, p in self.points.items() if p.converged]

    def failed_indices(self) -> list[int]:
        return [i for i, p in self.points.items() if not p.converged]


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def _snapshot_U(U) -> tuple:
    return tuple(d.data_ro.copy() for d in U.dat)


def _restore_U(snap: tuple, U, U_prev) -> None:
    for src, dst in zip(snap, U.dat):
        dst.data[:] = src
    U_prev.assign(U)


def solve_grid_per_voltage_cold_with_warm_fallback(
    solver_params,
    *,
    phi_applied_values: np.ndarray,
    mesh: Any = None,
    max_z_steps: int = 20,
    n_substeps_warm: int = 4,
    bisect_depth_warm: int = 3,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
    max_ss_steps_cold: int = 200,
    max_ss_steps_z: int = 120,
    max_ss_steps_warm: int = 150,
    max_ss_steps_warm_final: int = 200,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
    print_interval: int = 1,
    per_point_callback: Optional[Callable] = None,
) -> PerVoltageContinuationResult:
    """Solve BV-PNP across a voltage grid via per-V cold + warm-walk fallback.

    Strategy
    --------
    1. **Phase 1 (cold + z-ramp per V).** For each voltage in the grid,
       build a fresh context at the target voltage (mesh shared, forms
       per-V), set every charge factor to zero (dynamic species'
       ``z_consts`` and Boltzmann ``boltzmann_z_scale`` if present),
       run steady-state, then ramp z from 0 to 1 in
       ``max_z_steps`` linear steps with checkpoint+rollback on
       failure.
    2. **Phase 2 (warm-walk from cold successes).** For voltages
       Phase 1 didn't reach z=1, walk outward in two branches:
       cathodic-ward from the lowest cold-success index toward index
       0, and anodic-ward from the highest cold-success index toward
       index NV-1.  Each walk warm-starts from the *nearest converged
       neighbor* at full z, then marches ``phi_applied`` from
       ``V_anchor`` to ``V_target`` in ``n_substeps_warm`` linear
       substeps, recursively bisecting (up to ``bisect_depth_warm``)
       on any substep failure.

    Parameters
    ----------
    solver_params
        11-tuple ``SolverParams`` or list.  ``phi_applied`` (index 7)
        and any per-call kinetic parameters should already be set in
        ``solver_params`` (the orchestrator overrides ``phi_applied``
        per voltage but otherwise leaves the params alone).
    phi_applied_values
        Array of dimensionless overpotentials (the full voltage grid,
        in ``V/V_T`` units).
    mesh
        Shared graded mesh; if ``None`` a default is created inside
        ``build_context`` (use ``make_graded_rectangle_mesh`` for
        production).
    max_z_steps
        Linear steps for the per-V z-ramp from 0 to 1.
    n_substeps_warm, bisect_depth_warm
        Warm-walk knobs: substeps per ``ΔV`` and recursion depth
        for bisection on failure.
    ss_rel_tol, ss_abs_tol, ss_consec, max_ss_steps_*, dt_*
        Steady-state convergence settings (matching the v18/v24
        defaults).
    per_point_callback
        Optional ``callback(orig_idx, phi_applied, ctx)`` invoked
        after each per-V solve completes (cold or warm) with the
        live context, so callers can extract observables before the
        next voltage rebuilds the forms.
    """
    import firedrake as fd

    from .dispatch import build_context, build_forms, set_initial_conditions
    from .observables import _build_bv_observable_form
    from .solvers import _clone_params_with_phi
    from Forward.params import SolverParams

    phi_applied_values = np.asarray(phi_applied_values, dtype=float)
    n_points = len(phi_applied_values)

    # Unpack solver_params just to grab n_species and z_nominal.
    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, _, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError(
            "solve_grid_per_voltage_cold_with_warm_fallback expects an "
            "11-element solver_params"
        ) from exc

    z_nominal = [float(v) for v in (
        [z_v] * n_s if np.isscalar(z_v) else list(z_v)
    )][:n_s]

    def _params_with_phi(phi_applied_target: float):
        """Return solver_params with phi_applied (index 7) replaced."""
        if isinstance(solver_params, SolverParams):
            return solver_params.with_phi_applied(float(phi_applied_target))
        return list(_clone_params_with_phi(
            solver_params, phi_applied=float(phi_applied_target)
        ))

    # ------------------------------------------------------------------
    # Per-voltage build helper
    # ------------------------------------------------------------------
    def _build_for_voltage(phi_applied_target: float):
        """Fresh context + forms + IC + solver + observable for V_target.

        Returns ``(ctx, solver, of_cd)`` — observable used for SS
        convergence detection.
        """
        sp = _params_with_phi(phi_applied_target)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        set_initial_conditions(ctx, sp)
        problem = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
        )
        # Filter out non-PETSc sub-dicts (bv_bc, bv_convergence, nondim,
        # robin_bc) so they don't pollute the PETSc options database with
        # flattened "unknown option" warnings.
        _NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
        _items = params.items() if isinstance(params, dict) else []
        solve_opts = {k: v for k, v in _items if k not in _NON_PETSC_KEYS}
        # Force SNES to raise on non-convergence so run_ss's try/except
        # actually catches divergent Newton iterates (Firedrake's default
        # silently accepts them, which would let a non-converged state
        # be declared "steady" by the plateau detector).
        solve_opts.setdefault("snes_error_if_not_converged", True)
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=solve_opts,
        )
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=1.0,
        )
        return ctx, solver, of_cd

    # ------------------------------------------------------------------
    # SS time-stepping loop with SER adaptive dt
    # ------------------------------------------------------------------
    def _make_run_ss(ctx, solver, of_cd):
        U = ctx["U"]
        U_prev = ctx["U_prev"]
        dt_const = ctx["dt_const"]
        dt_max = float(dt_init) * float(dt_max_ratio)

        def run_ss(max_steps: int) -> bool:
            dt_val = float(dt_init)
            dt_const.assign(dt_val)
            prev_flux = None
            prev_delta = None
            steady_count = 0
            for _ in range(1, max_steps + 1):
                try:
                    solver.solve()
                except Exception:
                    return False
                U_prev.assign(U)
                fv = float(fd.assemble(of_cd))
                if prev_flux is not None:
                    delta = abs(fv - prev_flux)
                    sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
                    is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
                    steady_count = steady_count + 1 if is_steady else 0
                    if prev_delta is not None and delta > 0:
                        ratio = prev_delta / delta
                        if ratio > 1.0:
                            grow = min(ratio, dt_growth_cap)
                            dt_val = min(dt_val * grow, dt_max)
                        else:
                            dt_val = max(dt_val * 0.5, float(dt_init))
                        dt_const.assign(dt_val)
                    prev_delta = delta
                else:
                    steady_count = 0
                prev_flux = fv
                if steady_count >= ss_consec:
                    return True
            return False

        return run_ss

    # ------------------------------------------------------------------
    # Helper: scale all charges (dynamic species + Boltzmann ion)
    # ------------------------------------------------------------------
    def _set_z_factor(ctx: dict, z_val: float) -> None:
        for i in range(n_s):
            ctx["z_consts"][i].assign(z_nominal[i] * z_val)
        boltz = ctx.get("boltzmann_z_scale")
        if boltz is not None:
            boltz.assign(float(z_val))

    # ------------------------------------------------------------------
    # Phase 1 (C): per-voltage cold-start with internal z-ramp
    # ------------------------------------------------------------------
    def _solve_cold(orig_idx: int, V_target_eta: float):
        ctx, solver, of_cd = _build_for_voltage(V_target_eta)
        run_ss = _make_run_ss(ctx, solver, of_cd)
        U = ctx["U"]
        U_prev = ctx["U_prev"]
        # IC already linear at V_target_eta from set_initial_conditions.
        # Belt-and-suspenders: set phi_applied_func explicitly too.
        ctx["phi_applied_func"].assign(float(V_target_eta))

        # Step 0: zero all charges, run SS at z=0
        _set_z_factor(ctx, 0.0)
        if not run_ss(max_ss_steps_cold):
            return None, 0.0, ctx

        # Step 1..max_z_steps: linearly ramp z to 1 with checkpoint+rollback
        achieved_z = 0.0
        for z_val in np.linspace(0.0, 1.0, max_z_steps + 1)[1:]:
            ckpt = _snapshot_U(U)
            _set_z_factor(ctx, float(z_val))
            if run_ss(max_ss_steps_z):
                achieved_z = float(z_val)
            else:
                _restore_U(ckpt, U, U_prev)
                _set_z_factor(ctx, achieved_z)
                break

        if achieved_z < 1.0 - 1e-3:
            return None, achieved_z, ctx
        return _snapshot_U(U), achieved_z, ctx

    # ------------------------------------------------------------------
    # Phase 2 (D): warm-walk with paf substepping + bisection
    # ------------------------------------------------------------------
    def _solve_warm(orig_idx: int, V_target_eta: float,
                    V_anchor_eta: float, anchor_snap: tuple):
        ctx, solver, of_cd = _build_for_voltage(V_target_eta)
        run_ss = _make_run_ss(ctx, solver, of_cd)
        U = ctx["U"]
        U_prev = ctx["U_prev"]
        paf = ctx["phi_applied_func"]
        # Restore neighbor's snapshot (already at full z) and force
        # full charge from the start of the walk — no z-ramp here.
        _restore_U(anchor_snap, U, U_prev)
        _set_z_factor(ctx, 1.0)

        def _march(v0: float, v1: float, depth: int) -> bool:
            # Track v_prev_substep explicitly: paf is NOT in U.dat, so
            # _restore_U does not roll it back, and reading paf after a
            # failed substep returns the failed voltage (not the previous
            # successful one).  Using paf as the source of v_prev would
            # collapse the bisection midpoint to the failed voltage and
            # make the recursion degenerate.
            substeps = np.linspace(v0, v1, n_substeps_warm + 1)[1:]
            ckpt_outer = _snapshot_U(U)
            v_prev_substep = float(v0)
            for v_sub in substeps:
                ckpt_inner = _snapshot_U(U)
                paf.assign(float(v_sub))
                if run_ss(max_ss_steps_warm):
                    v_prev_substep = float(v_sub)
                    continue
                _restore_U(ckpt_inner, U, U_prev)
                paf.assign(v_prev_substep)
                if depth >= bisect_depth_warm:
                    _restore_U(ckpt_outer, U, U_prev)
                    paf.assign(float(v0))
                    return False
                v_mid = 0.5 * (v_prev_substep + float(v_sub))
                if not _march(v_prev_substep, v_mid, depth + 1):
                    _restore_U(ckpt_outer, U, U_prev)
                    paf.assign(float(v0))
                    return False
                if not _march(v_mid, float(v_sub), depth + 1):
                    _restore_U(ckpt_outer, U, U_prev)
                    paf.assign(float(v0))
                    return False
                v_prev_substep = float(v_sub)
            return True

        if not _march(float(V_anchor_eta), float(V_target_eta), 0):
            return None, ctx
        # Final SS at V_target to make sure we're firmly at the target
        paf.assign(float(V_target_eta))
        if not run_ss(max_ss_steps_warm_final):
            return None, ctx
        return _snapshot_U(U), ctx

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------
    print(f"[per-voltage] grid of {n_points} points; "
          f"strategy = cold (per-V z-ramp) + warm-walk fallback")

    points: Dict[int, PerVoltagePointResult] = {}
    snapshots: list[Optional[tuple]] = [None] * n_points
    last_dof_count: Optional[int] = None

    # ------- Phase 1: cold-start each V independently -------
    print(f"[per-voltage Phase 1] cold + z-ramp at each V_RHE_eta...")
    for orig_idx in range(n_points):
        eta_target = float(phi_applied_values[orig_idx])
        snap, achieved_z, ctx = _solve_cold(orig_idx, eta_target)
        if last_dof_count is None:
            last_dof_count = int(ctx["U"].function_space().dim())
        if snap is not None:
            snapshots[orig_idx] = snap
            converged = True
            method = "cold"
            if per_point_callback is not None:
                per_point_callback(orig_idx, eta_target, ctx)
        else:
            converged = False
            method = "cold-failed"
        if (orig_idx + 1) % print_interval == 0 or orig_idx == 0:
            tag = "OK" if converged else f"PARTIAL z={achieved_z:.3f}"
            print(
                f"  [Phase 1] {orig_idx + 1}/{n_points}  "
                f"eta={eta_target:+8.4f}  {tag}"
            )
        points[orig_idx] = PerVoltagePointResult(
            orig_idx=orig_idx,
            phi_applied=eta_target,
            U_data=snap,
            achieved_z_factor=achieved_z,
            converged=converged,
            method=method,
        )

    cold_idxs = sorted(i for i, p in points.items() if p.converged)
    n_cold_ok = len(cold_idxs)
    print(f"[per-voltage Phase 1] {n_cold_ok}/{n_points} cold-converged")

    if not cold_idxs:
        print(
            "[per-voltage] no cold successes; warm-walk has nothing to "
            "anchor on, returning Phase-1 results as-is."
        )
        return PerVoltageContinuationResult(
            points=points,
            mesh_dof_count=int(last_dof_count or 0),
        )

    # ------- Phase 2: warm-walk outward -------
    print(
        f"[per-voltage Phase 2] warm-walk from cold anchors "
        f"(n_substeps={n_substeps_warm}, bisect_depth={bisect_depth_warm})"
    )

    anchor_lo = cold_idxs[0]
    anchor_hi = cold_idxs[-1]

    # Cathodic walk: from anchor_lo down toward index 0
    for orig_idx in range(anchor_lo - 1, -1, -1):
        if points[orig_idx].converged:
            continue
        # Find nearest converged neighbor to the right (higher index,
        # but the walk is descending so that's "closer to anchor_lo")
        j = orig_idx + 1
        while j < n_points and not points[j].converged:
            j += 1
        if j >= n_points:
            break  # no anchor available (shouldn't happen given anchor_lo)
        eta_target = float(phi_applied_values[orig_idx])
        eta_anchor = float(phi_applied_values[j])
        snap, ctx = _solve_warm(orig_idx, eta_target, eta_anchor,
                                snapshots[j])  # type: ignore[arg-type]
        if snap is not None:
            snapshots[orig_idx] = snap
            method = f"warm<-{phi_applied_values[j]:+.3f}"
            converged = True
            if per_point_callback is not None:
                per_point_callback(orig_idx, eta_target, ctx)
            points[orig_idx] = PerVoltagePointResult(
                orig_idx=orig_idx,
                phi_applied=eta_target,
                U_data=snap,
                achieved_z_factor=1.0,
                converged=converged,
                method=method,
            )
            print(
                f"  [Phase 2 cathodic] {orig_idx + 1}/{n_points}  "
                f"eta={eta_target:+8.4f}  {method}"
            )
        else:
            print(
                f"  [Phase 2 cathodic] {orig_idx + 1}/{n_points}  "
                f"eta={eta_target:+8.4f}  warm-walk FAILED (broke chain)"
            )
            # Don't try further-cathodic if the nearest step failed —
            # there's nothing closer that converged.
            break

    # Anodic walk: from anchor_hi up toward index NV-1
    for orig_idx in range(anchor_hi + 1, n_points):
        if points[orig_idx].converged:
            continue
        j = orig_idx - 1
        while j >= 0 and not points[j].converged:
            j -= 1
        if j < 0:
            break
        eta_target = float(phi_applied_values[orig_idx])
        eta_anchor = float(phi_applied_values[j])
        snap, ctx = _solve_warm(orig_idx, eta_target, eta_anchor,
                                snapshots[j])  # type: ignore[arg-type]
        if snap is not None:
            snapshots[orig_idx] = snap
            method = f"warm<-{phi_applied_values[j]:+.3f}"
            converged = True
            if per_point_callback is not None:
                per_point_callback(orig_idx, eta_target, ctx)
            points[orig_idx] = PerVoltagePointResult(
                orig_idx=orig_idx,
                phi_applied=eta_target,
                U_data=snap,
                achieved_z_factor=1.0,
                converged=converged,
                method=method,
            )
            print(
                f"  [Phase 2 anodic] {orig_idx + 1}/{n_points}  "
                f"eta={eta_target:+8.4f}  {method}"
            )
        else:
            print(
                f"  [Phase 2 anodic] {orig_idx + 1}/{n_points}  "
                f"eta={eta_target:+8.4f}  warm-walk FAILED (broke chain)"
            )
            break

    n_total_ok = sum(1 for p in points.values() if p.converged)
    print(
        f"[per-voltage] {n_total_ok}/{n_points} converged "
        f"(cold {n_cold_ok}, warm {n_total_ok - n_cold_ok})"
    )

    return PerVoltageContinuationResult(
        points=points,
        mesh_dof_count=int(last_dof_count or 0),
    )
