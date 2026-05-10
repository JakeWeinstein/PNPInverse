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
    diagnostics: Optional[Dict[str, Any]] = None  # populated by collect_diagnostics


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


# ---------------------------------------------------------------------------
# Public aliases (Phase 5γ — used by anchor_continuation.py and external
# callers that need to checkpoint/restore U or run a stand-alone SS loop).
# ---------------------------------------------------------------------------

def snapshot_U(U) -> tuple:
    """Public alias for :func:`_snapshot_U`.

    Returns ``tuple(d.data_ro.copy() for d in U.dat)`` — a deep copy
    suitable for later restoration via :func:`restore_U`.
    """
    return _snapshot_U(U)


def restore_U(snap: tuple, U, U_prev) -> None:
    """Public alias for :func:`_restore_U`.

    Writes ``snap`` into ``U.dat`` and assigns ``U_prev = U`` to keep
    the time-stepping state consistent.
    """
    _restore_U(snap, U, U_prev)


def make_run_ss(
    *,
    ctx,
    solver,
    of_cd,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
) -> Callable[[int], bool]:
    """Build a SER adaptive-dt steady-state loop closure.

    Returns a callable ``run_ss(max_steps: int) -> bool`` that drives
    the supplied ``solver`` for up to ``max_steps`` Newton solves,
    using the SER adaptive-dt rule against the observable form
    ``of_cd``. Returns ``True`` once ``ss_consec`` consecutive steady
    plateau detections are seen, or ``False`` if Newton diverges or
    the step cap is hit.

    This factory makes the SS loop available outside the orchestrator
    (e.g. for k0-continuation rungs in
    :mod:`Forward.bv_solver.anchor_continuation`) without duplicating
    the SER logic.

    Parameters
    ----------
    ctx
        Firedrake context (must expose ``U``, ``U_prev``, ``dt_const``).
    solver
        Pre-built ``NonlinearVariationalSolver``.
    of_cd
        Observable UFL form whose plateau detects steady state.
    dt_init, dt_growth_cap, dt_max_ratio
        SER knobs. ``dt_max = dt_init * dt_max_ratio``.
    ss_rel_tol, ss_abs_tol, ss_consec
        Plateau-detection thresholds.
    """
    import firedrake as fd

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


def warm_walk_phi(
    *,
    ctx,
    solver,
    of_cd,
    v_anchor_eta: float,
    v_target_eta: float,
    n_substeps: int = 4,
    bisect_depth: int = 3,
    max_ss_steps_per_substep: int = 150,
    max_ss_steps_final: int = 200,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
) -> bool:
    """Walk ``ctx['phi_applied_func']`` from anchor to target.

    Performs the substep+bisect from ``v_anchor_eta`` to
    ``v_target_eta`` (mirrors the C+D Phase 2 D logic), then a final
    SS at ``v_target_eta``.

    Pre-conditions (caller's responsibility):
      * ``ctx['U']`` holds a converged state at ``v_anchor_eta``.
      * z-factors are at full (1.0); k0 is at production target.
      * ``solver`` is a ``NonlinearVariationalSolver`` for the same
        ctx; ``of_cd`` is a current-density observable form.

    Returns
    -------
    bool
        ``True`` iff the final SS converges. On failure, restores the
        U snapshot taken at function entry (the anchor state) and
        resets ``ctx['phi_applied_func']`` to ``v_anchor_eta``.
    """
    U = ctx["U"]
    U_prev = ctx["U_prev"]
    paf = ctx["phi_applied_func"]

    run_ss = make_run_ss(
        ctx=ctx,
        solver=solver,
        of_cd=of_cd,
        dt_init=dt_init,
        dt_growth_cap=dt_growth_cap,
        dt_max_ratio=dt_max_ratio,
        ss_rel_tol=ss_rel_tol,
        ss_abs_tol=ss_abs_tol,
        ss_consec=ss_consec,
    )

    def _march(v0: float, v1: float, depth: int) -> bool:
        # Track v_prev_substep explicitly: paf is NOT in U.dat, so
        # _restore_U does not roll it back, and reading paf after a
        # failed substep returns the failed voltage (not the previous
        # successful one).  Using paf as the source of v_prev would
        # collapse the bisection midpoint to the failed voltage and
        # make the recursion degenerate.
        substeps = np.linspace(v0, v1, n_substeps + 1)[1:]
        ckpt_outer = _snapshot_U(U)
        v_prev_substep = float(v0)
        for v_sub in substeps:
            ckpt_inner = _snapshot_U(U)
            paf.assign(float(v_sub))
            if run_ss(max_ss_steps_per_substep):
                v_prev_substep = float(v_sub)
                continue
            _restore_U(ckpt_inner, U, U_prev)
            paf.assign(v_prev_substep)
            if depth >= bisect_depth:
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

    if not _march(float(v_anchor_eta), float(v_target_eta), 0):
        return False
    # Final SS at V_target to make sure we're firmly at the target.
    paf.assign(float(v_target_eta))
    if not run_ss(max_ss_steps_final):
        return False
    return True


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
    from .forms_logc import set_initial_conditions_logc
    from .observables import _build_bv_observable_form
    from .solvers import _clone_params_with_phi
    from .diagnostics import collect_diagnostics
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
        ctx["_last_solver"] = solver
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=1.0,
        )
        return ctx, solver, of_cd

    # ------------------------------------------------------------------
    # SS time-stepping loop with SER adaptive dt
    #
    # Delegates to the public ``make_run_ss`` factory so external callers
    # (anchor_continuation.py) and the orchestrator share a single
    # implementation. The closure here captures the orchestrator's
    # outer kwargs to preserve byte-equivalence with the pre-Phase-5γ
    # version.
    # ------------------------------------------------------------------
    def _make_run_ss(ctx, solver, of_cd):
        return make_run_ss(
            ctx=ctx,
            solver=solver,
            of_cd=of_cd,
            dt_init=dt_init,
            dt_growth_cap=dt_growth_cap,
            dt_max_ratio=dt_max_ratio,
            ss_rel_tol=ss_rel_tol,
            ss_abs_tol=ss_abs_tol,
            ss_consec=ss_consec,
        )

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
    # Initializer flag (read once for orchestrator-level dispatch).
    # ------------------------------------------------------------------
    if isinstance(params, dict):
        _bv_conv_for_init = params.get("bv_convergence", {})
        _initializer_flag = (
            str(_bv_conv_for_init.get("initializer", "linear_phi")).strip().lower()
            if isinstance(_bv_conv_for_init, dict) else "linear_phi"
        )
    else:
        _initializer_flag = "linear_phi"

    # ------------------------------------------------------------------
    # Phase 1 (C): per-voltage cold-start with internal z-ramp
    # ------------------------------------------------------------------
    def _solve_cold(orig_idx: int, V_target_eta: float):
        ctx, solver, of_cd = _build_for_voltage(V_target_eta)
        run_ss = _make_run_ss(ctx, solver, of_cd)
        U = ctx["U"]
        U_prev = ctx["U_prev"]
        # IC already at V_target_eta from set_initial_conditions (linear-phi
        # or debye_boltzmann depending on params).  Belt-and-suspenders:
        # set phi_applied_func explicitly too.
        ctx["phi_applied_func"].assign(float(V_target_eta))

        # Path B: analytical IC -> try direct z=1 SS; the IC already encodes
        # the depleted-H+/enriched-counterion state, so the z=0 stage that
        # the linear-phi path needs (to discover charge balance from a
        # bulk-flat guess) would *erase* the analytical state and is skipped.
        # Fall back to linear-phi z-ramp on Newton failure.
        if _initializer_flag != "linear_phi" and not ctx.get(
            "initializer_fallback", False
        ):
            _set_z_factor(ctx, 1.0)
            if run_ss(max_ss_steps_z):
                return _snapshot_U(U), 1.0, ctx
            # Direct z=1 failed -- reset to linear-phi IC and use the
            # standard z-ramp path below (bypass dispatcher to avoid
            # routing back to the analytical IC).
            ctx["initializer_fallback"] = True
            ctx["initializer_fallback_reason"] = (
                ctx.get("initializer_fallback_reason", "")
                + ";cold_z1_diverged"
            )
            set_initial_conditions_logc(ctx, _params_with_phi(V_target_eta))

        # Path A: linear-phi (or fallback).  Step 0: zero all charges,
        # run SS at z=0 to find the neutral-electrolyte fixed point.
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
    #
    # Delegates the substep+bisect+final-SS to the public
    # ``warm_walk_phi`` factored out in Phase 5γ. Wrapper handles
    # context build, anchor restore, and z=1 flip — preconditions of
    # ``warm_walk_phi`` — then captures the final snapshot on success.
    # ------------------------------------------------------------------
    def _solve_warm(orig_idx: int, V_target_eta: float,
                    V_anchor_eta: float, anchor_snap: tuple):
        ctx, solver, of_cd = _build_for_voltage(V_target_eta)
        U = ctx["U"]
        U_prev = ctx["U_prev"]
        # Restore neighbor's snapshot (already at full z) and force
        # full charge from the start of the walk — no z-ramp here.
        _restore_U(anchor_snap, U, U_prev)
        _set_z_factor(ctx, 1.0)

        ok = warm_walk_phi(
            ctx=ctx,
            solver=solver,
            of_cd=of_cd,
            v_anchor_eta=float(V_anchor_eta),
            v_target_eta=float(V_target_eta),
            n_substeps=n_substeps_warm,
            bisect_depth=bisect_depth_warm,
            max_ss_steps_per_substep=max_ss_steps_warm,
            max_ss_steps_final=max_ss_steps_warm_final,
            ss_rel_tol=ss_rel_tol,
            ss_abs_tol=ss_abs_tol,
            ss_consec=ss_consec,
            dt_init=dt_init,
            dt_growth_cap=dt_growth_cap,
            dt_max_ratio=dt_max_ratio,
        )
        if not ok:
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
        try:
            diags = collect_diagnostics(
                ctx,
                phase="cold_z_ramp" if converged else "cold_failed",
                params=params,
                picard_iters=ctx.get("initializer_picard_iters"),
            )
        except Exception as exc:
            diags = {"phase": "diagnostics_error",
                     "error": f"{type(exc).__name__}: {exc}"}
        points[orig_idx] = PerVoltagePointResult(
            orig_idx=orig_idx,
            phi_applied=eta_target,
            U_data=snap,
            achieved_z_factor=achieved_z,
            converged=converged,
            method=method,
            diagnostics=diags,
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
            try:
                diags = collect_diagnostics(
                    ctx, phase="warm_walk_cathodic", params=params,
                )
            except Exception as exc:
                diags = {"phase": "diagnostics_error",
                         "error": f"{type(exc).__name__}: {exc}"}
            points[orig_idx] = PerVoltagePointResult(
                orig_idx=orig_idx,
                phi_applied=eta_target,
                U_data=snap,
                achieved_z_factor=1.0,
                converged=converged,
                method=method,
                diagnostics=diags,
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
            try:
                diags = collect_diagnostics(
                    ctx, phase="warm_walk_anodic", params=params,
                )
            except Exception as exc:
                diags = {"phase": "diagnostics_error",
                         "error": f"{type(exc).__name__}: {exc}"}
            points[orig_idx] = PerVoltagePointResult(
                orig_idx=orig_idx,
                phi_applied=eta_target,
                U_data=snap,
                achieved_z_factor=1.0,
                converged=converged,
                method=method,
                diagnostics=diags,
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

    # ------- Phase 2 (interior): fill cold-failed gaps between anchors -------
    # The outer cathodic / anodic walks only visit indices outside
    # [anchor_lo, anchor_hi].  Cold-failed points *between* non-contiguous
    # cold successes (e.g. cold succeeded at idx 2 and idx 7 but failed
    # at 3-6) need their own pass.  For each interior failure, warm-walk
    # from the nearest already-converged neighbour (closer side first;
    # fall back to the further side).  Iterate so that just-converged
    # interior points can serve as anchors for further interior fills,
    # and so a longer-distance anchor is retried only after a closer one
    # becomes available.
    if anchor_hi - anchor_lo > 1:
        print(
            f"[per-voltage Phase 2 interior] filling cold-failed gaps "
            f"between anchor_lo={anchor_lo} and anchor_hi={anchor_hi}"
        )
        # Track (j_lo, j_hi) anchor pairs that already failed for each
        # orig_idx.  Skip a retry on a subsequent pass if the anchors
        # haven't changed (no new information available).
        failed_with_anchors: Dict[int, tuple] = {}
        made_progress = True
        while made_progress:
            made_progress = False
            for orig_idx in range(anchor_lo + 1, anchor_hi):
                if points[orig_idx].converged:
                    continue
                # Nearest converged neighbour on each side.
                j_lo: Optional[int] = orig_idx - 1
                while j_lo is not None and j_lo >= 0 and not points[j_lo].converged:
                    j_lo -= 1
                if j_lo is None or j_lo < 0:
                    j_lo = None
                j_hi: Optional[int] = orig_idx + 1
                while (j_hi is not None and j_hi < n_points
                       and not points[j_hi].converged):
                    j_hi += 1
                if j_hi is None or j_hi >= n_points:
                    j_hi = None
                if j_lo is None and j_hi is None:
                    continue
                anchor_key = (j_lo, j_hi)
                if failed_with_anchors.get(orig_idx) == anchor_key:
                    # Same anchors as last failed attempt; nothing to gain.
                    continue
                # Sort by index distance to orig_idx (closer first).
                candidates: list[tuple[int, int]] = []
                if j_lo is not None:
                    candidates.append((orig_idx - j_lo, j_lo))
                if j_hi is not None:
                    candidates.append((j_hi - orig_idx, j_hi))
                candidates.sort(key=lambda x: x[0])
                eta_target = float(phi_applied_values[orig_idx])
                converged_this_attempt = False
                for _, j in candidates:
                    eta_anchor = float(phi_applied_values[j])
                    snap, ctx = _solve_warm(
                        orig_idx, eta_target, eta_anchor,
                        snapshots[j],  # type: ignore[arg-type]
                    )
                    if snap is None:
                        continue
                    snapshots[orig_idx] = snap
                    method = f"warm<-{phi_applied_values[j]:+.3f}"
                    if per_point_callback is not None:
                        per_point_callback(orig_idx, eta_target, ctx)
                    try:
                        diags = collect_diagnostics(
                            ctx, phase="warm_walk_interior", params=params,
                        )
                    except Exception as exc:
                        diags = {"phase": "diagnostics_error",
                                 "error": f"{type(exc).__name__}: {exc}"}
                    points[orig_idx] = PerVoltagePointResult(
                        orig_idx=orig_idx,
                        phi_applied=eta_target,
                        U_data=snap,
                        achieved_z_factor=1.0,
                        converged=True,
                        method=method,
                        diagnostics=diags,
                    )
                    failed_with_anchors.pop(orig_idx, None)
                    print(
                        f"  [Phase 2 interior] {orig_idx + 1}/{n_points}  "
                        f"eta={eta_target:+8.4f}  {method}"
                    )
                    made_progress = True
                    converged_this_attempt = True
                    break
                if not converged_this_attempt:
                    failed_with_anchors[orig_idx] = anchor_key
                    print(
                        f"  [Phase 2 interior] {orig_idx + 1}/{n_points}  "
                        f"eta={eta_target:+8.4f}  warm-walk FAILED "
                        f"from anchors {anchor_key}"
                    )

    n_total_ok = sum(1 for p in points.values() if p.converged)
    print(
        f"[per-voltage] {n_total_ok}/{n_points} converged "
        f"(cold {n_cold_ok}, warm {n_total_ok - n_cold_ok})"
    )

    return PerVoltageContinuationResult(
        points=points,
        mesh_dof_count=int(last_dof_count or 0),
    )


# ---------------------------------------------------------------------------
# solve_grid_with_anchor — Phase 5γ Pass A driver
# ---------------------------------------------------------------------------

def solve_grid_with_anchor(
    solver_params,
    *,
    anchor: "PreconvergedAnchor",
    phi_applied_values: np.ndarray,
    mesh: Any,
    n_substeps_warm: int = 8,
    bisect_depth_warm: int = 5,
    ss_rel_tol: float = 1e-4,
    ss_abs_tol: float = 1e-8,
    ss_consec: int = 4,
    max_ss_steps_warm: int = 150,
    max_ss_steps_warm_final: int = 200,
    dt_init: float = 0.25,
    dt_growth_cap: float = 4.0,
    dt_max_ratio: float = 20.0,
    print_interval: int = 1,
    per_point_callback: Optional[Callable] = None,
) -> PerVoltageContinuationResult:
    """Drive a V_RHE grid by walking outward from a preconverged anchor.

    No cold-start, no z-ramp: every voltage warm-walks from the
    nearest converged neighbour (the anchor itself for the first
    visited V). Use ``solve_anchor_with_continuation`` upstream to
    build the anchor.

    Strategy
    --------
    1. Sort grid indices by ``|phi - anchor.phi_applied_eta|``
       ascending — closest first.
    2. For each target V in that order:
       a. Build a fresh ctx + forms at that V.
       b. Assert ``ctx['U'].function_space().dim() ==
          anchor.mesh_dof_count``.
       c. Set k0 from ``anchor.k0_targets`` via
          :func:`set_reaction_k0_model` (both before and after IC) so
          the FE residual and Picard agree on production rates.
       d. Restore U from the nearest converged neighbour's snapshot
          (the anchor at first; subsequent visited V's eligible).
       e. Set z-factor to 1.0 (anchor is at full z).
       f. Set ``ctx['phi_applied_func']`` to the neighbour's V (NOT
          the target — :func:`warm_walk_phi` advances it).
       g. Build solver + run :func:`warm_walk_phi` from neighbour V
          to target V.
       h. On success: snapshot U, mark converged, optional
          ``per_point_callback``. The new snapshot becomes a candidate
          neighbour for subsequent targets.
       i. On failure: ``PerVoltagePointResult`` with
          ``converged=False`` and
          ``method=f"warm<-{neighbour:+.3f}-FAILED"``.
    3. Wrap the entire loop in ``firedrake.adjoint.stop_annotating()``.

    Parameters
    ----------
    solver_params
        :class:`Forward.params.SolverParams` (or compatible 11-tuple)
        already configured for the production reaction stack
        (k0, alpha, etc. should match ``anchor.k0_targets`` for
        consistency, though the orchestrator re-pins k0 from the
        anchor regardless).
    anchor
        :class:`PreconvergedAnchor` produced by
        :func:`extract_preconverged_anchor` from a successful
        :func:`solve_anchor_with_continuation` run.
    phi_applied_values
        Array of dimensionless overpotentials (V/V_T units).
    mesh
        Shared graded mesh. Must match the mesh that produced the
        anchor (same ``ctx['U'].function_space().dim()``).

    Returns
    -------
    PerVoltageContinuationResult
        Same type as
        :func:`solve_grid_per_voltage_cold_with_warm_fallback`, so
        callers can switch entry points without changing reporting.

    Raises
    ------
    ValueError
        Mesh DOF mismatch between anchor and live ctx; or
        ``solver_params`` is not 11-element.
    TypeError
        ``anchor`` is not a :class:`PreconvergedAnchor`.
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    from .anchor_continuation import (
        NON_PETSC_KEYS,
        PreconvergedAnchor,
        set_reaction_k0_model,
    )
    from .dispatch import build_context, build_forms, set_initial_conditions
    from .observables import _build_bv_observable_form
    from .solvers import _clone_params_with_phi
    from .diagnostics import collect_diagnostics
    from Forward.params import SolverParams

    if not isinstance(anchor, PreconvergedAnchor):
        raise TypeError(
            f"anchor must be PreconvergedAnchor "
            f"(got {type(anchor).__name__})"
        )

    phi_applied_values = np.asarray(phi_applied_values, dtype=float)
    n_points = len(phi_applied_values)

    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, _, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError(
            "solve_grid_with_anchor expects an 11-element solver_params"
        ) from exc

    z_nominal = [float(v) for v in (
        [z_v] * n_s if np.isscalar(z_v) else list(z_v)
    )][:n_s]

    def _params_with_phi(phi_target: float):
        if isinstance(solver_params, SolverParams):
            return solver_params.with_phi_applied(float(phi_target))
        return list(_clone_params_with_phi(
            solver_params, phi_applied=float(phi_target),
        ))

    def _set_z_factor(ctx: dict, z_val: float) -> None:
        for i in range(n_s):
            ctx["z_consts"][i].assign(z_nominal[i] * z_val)
        boltz = ctx.get("boltzmann_z_scale")
        if boltz is not None:
            boltz.assign(float(z_val))

    k0_target_dict = anchor.k0_targets_dict()

    def _build_for_voltage(phi_target: float):
        sp_v = _params_with_phi(phi_target)
        ctx = build_context(sp_v, mesh=mesh)
        ctx = build_forms(ctx, sp_v)
        # Pin k0 to anchor targets BEFORE IC so Picard sees production
        # rates while seeding (matches the anchor's converged regime).
        for j, k_target in k0_target_dict.items():
            set_reaction_k0_model(ctx, j, k_target)
        set_initial_conditions(ctx, sp_v)
        # Re-pin AFTER IC: defensive in case set_initial_conditions
        # internals reseed k0 from the build-time bv_reactions config.
        for j, k_target in k0_target_dict.items():
            set_reaction_k0_model(ctx, j, k_target)
        items = params.items() if isinstance(params, dict) else []
        solve_opts = {k: v for k, v in items if k not in NON_PETSC_KEYS}
        solve_opts.setdefault("snes_error_if_not_converged", True)
        problem = fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
        )
        solver = fd.NonlinearVariationalSolver(
            problem, solver_parameters=solve_opts,
        )
        ctx["_last_solver"] = solver
        of_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=1.0,
        )
        return ctx, solver, of_cd

    # Visit grid in order of distance from the anchor's voltage.
    visit_order = sorted(
        range(n_points),
        key=lambda i: abs(
            float(phi_applied_values[i]) - float(anchor.phi_applied_eta)
        ),
    )

    # Sources: (phi_eta, snapshot) pairs eligible as warm-walk anchors.
    # Start with the anchor itself; append each successful grid solve.
    sources: list[tuple[float, tuple]] = [
        (float(anchor.phi_applied_eta), anchor.U_snapshot),
    ]

    points: Dict[int, PerVoltagePointResult] = {}
    last_dof_count: Optional[int] = None

    print(
        f"[solve_grid_with_anchor] grid of {n_points} points; "
        f"anchor at phi_eta={float(anchor.phi_applied_eta):+.4f}"
    )

    with adj.stop_annotating():
        for visit_n, orig_idx in enumerate(visit_order):
            target_phi = float(phi_applied_values[orig_idx])

            ctx, solver, of_cd = _build_for_voltage(target_phi)

            dof = int(ctx["U"].function_space().dim())
            if last_dof_count is None:
                last_dof_count = dof
            if dof != int(anchor.mesh_dof_count):
                raise ValueError(
                    f"mesh DOF mismatch: ctx['U'].function_space().dim()"
                    f"={dof} but anchor.mesh_dof_count={anchor.mesh_dof_count}. "
                    "The anchor was built on a different mesh than the grid "
                    "run; both must use the same mesh."
                )

            # Pick nearest converged source by absolute phi distance.
            src_phi, src_snap = min(
                sources,
                key=lambda s: abs(float(s[0]) - target_phi),
            )

            U = ctx["U"]
            U_prev = ctx["U_prev"]
            _restore_U(src_snap, U, U_prev)
            _set_z_factor(ctx, 1.0)
            ctx["phi_applied_func"].assign(float(src_phi))

            ok = warm_walk_phi(
                ctx=ctx,
                solver=solver,
                of_cd=of_cd,
                v_anchor_eta=float(src_phi),
                v_target_eta=target_phi,
                n_substeps=n_substeps_warm,
                bisect_depth=bisect_depth_warm,
                max_ss_steps_per_substep=max_ss_steps_warm,
                max_ss_steps_final=max_ss_steps_warm_final,
                ss_rel_tol=ss_rel_tol,
                ss_abs_tol=ss_abs_tol,
                ss_consec=ss_consec,
                dt_init=dt_init,
                dt_growth_cap=dt_growth_cap,
                dt_max_ratio=dt_max_ratio,
            )

            if ok:
                snap = _snapshot_U(U)
                sources.append((target_phi, snap))
                method = f"warm<-{src_phi:+.3f}"
                converged = True
                if per_point_callback is not None:
                    per_point_callback(orig_idx, target_phi, ctx)
            else:
                snap = None
                method = f"warm<-{src_phi:+.3f}-FAILED"
                converged = False

            if (visit_n + 1) % print_interval == 0 or visit_n == 0:
                tag = "OK" if converged else "FAIL"
                print(
                    f"  [{visit_n + 1}/{n_points}] eta={target_phi:+8.4f} "
                    f"src={src_phi:+8.4f} {method} {tag}"
                )

            try:
                diags = collect_diagnostics(
                    ctx, phase="warm_walk_anchor", params=params,
                )
            except Exception as exc:
                diags = {
                    "phase": "diagnostics_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }

            points[orig_idx] = PerVoltagePointResult(
                orig_idx=orig_idx,
                phi_applied=target_phi,
                U_data=snap,
                achieved_z_factor=1.0,
                converged=converged,
                method=method,
                diagnostics=diags,
            )

    n_total_ok = sum(1 for p in points.values() if p.converged)
    print(
        f"[solve_grid_with_anchor] {n_total_ok}/{n_points} converged"
    )

    return PerVoltageContinuationResult(
        points=points,
        mesh_dof_count=int(last_dof_count or anchor.mesh_dof_count),
    )
