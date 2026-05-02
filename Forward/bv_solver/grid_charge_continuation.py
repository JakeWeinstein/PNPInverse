"""Unified grid charge continuation: neutral sweep once, then per-point z-ramp.

Public API
----------
solve_grid_with_charge_continuation(solver_params, *, phi_applied_values, ...)
    -> GridChargeContinuationResult

Design
------
Phase 1 (neutral voltage sweep, z=0):
    Runs ONCE across the full voltage grid using two-branch sweep ordering,
    Lagrange predictor warm-starts, bridge points, and SER adaptive dt.

Phase 2 (per-point charge ramp, z: 0 -> 1):
    Each point independently z-ramps from its neutral solution to full charge.
    Uses an aggressive-first adaptive strategy: try z=1.0 directly, binary
    search for a foothold if that fails, then geometrically accelerate.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

import numpy as np

from .sweep_order import _apply_predictor, _build_sweep_order
from .observables import _build_bv_observable_form


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GridPointResult:
    """Result for a single voltage point after Phase 2 z-ramp."""

    orig_idx: int
    phi_applied: float
    U_data: tuple                # tuple(d.data_ro.copy() for d in ctx["U"].dat)
    achieved_z_factor: float
    converged: bool              # achieved_z >= 1.0 - 1e-6
    validation: object = None    # Optional[ValidationResult] from validation.py


@dataclasses.dataclass(frozen=True)
class GridChargeContinuationResult:
    """Aggregate result for the full grid."""

    points: Dict[int, GridPointResult]
    mesh_dof_count: int

    def all_converged(self) -> bool:
        return all(p.converged for p in self.points.values())

    def get_U_data(self, idx: int) -> tuple:
        return self.points[idx].U_data

    def partial_points(self) -> list:
        return [p for p in self.points.values() if not p.converged]

    def physics_failures(self) -> list:
        """Points where physics validation failed."""
        return [p for p in self.points.values()
                if p.validation is not None and not p.validation.valid]


# ---------------------------------------------------------------------------
# SER / steady-state constants
# ---------------------------------------------------------------------------

_SER_GROWTH_CAP = 4.0       # max dt multiplier per step
_SER_SHRINK = 0.5            # dt multiplier when residual grows
_SER_DT_MAX_RATIO = 20.0    # max dt / dt_initial
_STEADY_REL_TOL = 1e-4       # relative change threshold
_STEADY_ABS_TOL = 1e-8       # absolute change threshold
_STEADY_CONSEC = 4           # consecutive steady steps required
_WARMSTART_MAX_STEPS = 20    # reduced cap for warm-started points
_MAX_COLD_STEPS = 100        # full cap for first point / cold starts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def solve_grid_with_charge_continuation(
    solver_params,
    *,
    phi_applied_values: np.ndarray,
    charge_steps: int = 10,
    mesh: Any = None,
    min_delta_z: float = 0.005,
    max_eta_gap: float = 3.0,
    print_interval: int = 20,
    per_point_callback: Optional[Callable] = None,
) -> GridChargeContinuationResult:
    """Solve BV-PNP across a voltage grid using unified charge continuation.

    Parameters
    ----------
    solver_params : SolverParams or list
        11-element solver parameter set.  phi_applied and z_vals fields
        are overridden internally.
    phi_applied_values : np.ndarray
        Array of dimensionless overpotentials (the full voltage grid).
    charge_steps : int
        Fallback uniform step count for z-ramp (used in fine-step regime).
    mesh : optional
        Shared mesh; if None one is created from solver_params.
    min_delta_z : float
        Minimum z increment before giving up on subdivision.
    max_eta_gap : float
        Maximum eta gap before bridge points are inserted.
    print_interval : int
        Print progress every N points.
    per_point_callback : callable, optional
        Called as ``callback(orig_idx, phi_applied, ctx)`` after each
        Phase 2 point with the live context (for observable extraction).

    Returns
    -------
    GridChargeContinuationResult
    """
    import firedrake as fd

    # Use the formulation dispatcher so the logc + Boltzmann + log-rate
    # stack is reachable through this pipeline whenever solver_params
    # requests it via bv_convergence.formulation.
    from .dispatch import build_context, build_forms, set_initial_conditions
    from .solvers import _clone_params_with_phi

    # ------------------------------------------------------------------
    # Unpack solver_params
    # ------------------------------------------------------------------
    try:
        n_s, order, dt, t_end, z_v, D_v, a_v, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("bv_solver expects a list of 11 solver parameters") from exc

    phi_applied_values = np.asarray(phi_applied_values, dtype=float)

    # Store nominal z for Phase 2
    z_nominal = [float(v) for v in ([z_v] * n_s if np.isscalar(z_v) else list(z_v))][:n_s]

    # ------------------------------------------------------------------
    # Step 1: Build context at z=0, eta=0
    # ------------------------------------------------------------------
    from Forward.params import SolverParams
    if isinstance(solver_params, SolverParams):
        params_neutral = solver_params.with_phi_applied(0.0).with_z_vals([0.0] * n_s)
    else:
        params_neutral = list(_clone_params_with_phi(solver_params, phi_applied=0.0))
        params_neutral[4] = [0.0] * n_s

    ctx = build_context(params_neutral, mesh=mesh)
    ctx = build_forms(ctx, params_neutral)
    set_initial_conditions(ctx, params_neutral, blob=False)

    # When the formulation uses an analytic Boltzmann counterion, also
    # zero its z-scale during Phase 1 so the bulk is truly neutral
    # (otherwise the counterion's z=-1 charge stays full while the
    # dynamic species sit at z=0, leaving an unbalanced charge that
    # breaks the V-sweep bisection).  Phase 2 ramps both back to 1.
    if ctx.get("boltzmann_z_scale") is not None:
        ctx["boltzmann_z_scale"].assign(0.0)

    # ------------------------------------------------------------------
    # Step 2: Build solver and time-stepping constants
    # ------------------------------------------------------------------
    scaling = ctx["nondim"]
    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))

    dt_const = ctx["dt_const"]
    dt_initial = float(dt_const)
    dt_max = dt_initial * _SER_DT_MAX_RATIO

    # Build solver ONCE -- reused across all Phase 1 AND Phase 2 steps
    J = fd.derivative(ctx["F_res"], ctx["U"])
    problem = fd.NonlinearVariationalProblem(ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J)

    solve_opts = dict(params) if isinstance(params, dict) else {}
    solve_opts.setdefault("snes_error_if_not_converged", True)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)

    # ------------------------------------------------------------------
    # Step 3: Sweep order
    # ------------------------------------------------------------------
    sorted_indices = _build_sweep_order(phi_applied_values)

    # ------------------------------------------------------------------
    # Step 4: Pre-allocate checkpoints and predictor state
    # ------------------------------------------------------------------
    U_ckpt = fd.Function(ctx["U"])
    U_prev_ckpt = fd.Function(ctx["U_prev"])

    predictor_prev2 = None
    predictor_prev = None
    predictor_curr = None

    hub_U_data = None
    hub_eta = None
    first_branch_done = False
    prev_solved_eta = None

    has_mixed_signs = (phi_applied_values <= 0).any() and (phi_applied_values > 0).any()
    n_species = int(ctx["n_species"])

    # ------------------------------------------------------------------
    # Step 5a: Observable form for convergence detection
    # ------------------------------------------------------------------
    observable_form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )

    # ------------------------------------------------------------------
    # Step 5b: Shared _run_to_steady_state helper (SER + observable flux)
    # ------------------------------------------------------------------
    def _run_to_steady_state(max_steps):
        """Run PTC with SER adaptive dt until steady-state or max_steps.

        Returns (converged: bool, steps_taken: int).
        """
        nonlocal dt_const

        dt_current = dt_initial
        dt_const.assign(dt_initial)

        prev_flux_val = None
        prev_delta = None
        steady_count = 0
        steps_taken = 0

        try:
            for step in range(1, max_steps + 1):
                steps_taken = step
                solver.solve()
                ctx["U_prev"].assign(ctx["U"])

                # Observable-based convergence metric
                flux_val = float(fd.assemble(observable_form))

                if prev_flux_val is not None:
                    delta = abs(flux_val - prev_flux_val)
                    scale_val = max(abs(flux_val), abs(prev_flux_val), _STEADY_ABS_TOL)
                    rel_metric = delta / scale_val
                    is_steady = (rel_metric <= _STEADY_REL_TOL) or (delta <= _STEADY_ABS_TOL)
                    steady_count = steady_count + 1 if is_steady else 0

                    # SER adaptive dt (paper Eq. 4)
                    if prev_delta is not None and delta > 0:
                        ratio = prev_delta / delta
                        if ratio > 1.0:
                            grow = min(ratio, _SER_GROWTH_CAP)
                            dt_current = min(dt_current * grow, dt_max)
                        else:
                            dt_current = max(dt_current * _SER_SHRINK, dt_initial)
                        dt_const.assign(dt_current)

                    prev_delta = delta
                else:
                    steady_count = 0

                prev_flux_val = flux_val

                if steady_count >= _STEADY_CONSEC:
                    return True, steps_taken

            return False, steps_taken

        except Exception as exc:
            if isinstance(exc, fd.ConvergenceError):
                return False, -1  # sentinel: SNES failure, not budget exhaustion
            import petsc4py.PETSc as PETSc
            if isinstance(exc, PETSc.Error):
                return False, -1
            raise

    # ------------------------------------------------------------------
    # Step 5c: _try_timestep wrapper for Phase 1
    # ------------------------------------------------------------------
    def _try_timestep(eta_try, max_steps=None):
        """Attempt PTC time-stepping at given eta.

        Returns (converged: bool, usable: bool).
        - converged=True: steady-state reached within step budget.
        - converged=False, usable=True: budget exhausted without SNES
          failure — solution is a reasonable warm-start but not fully
          converged.
        - converged=False, usable=False: SNES divergence — solution is
          poisoned, caller must restore checkpoint.
        """
        if max_steps is None:
            max_steps = _MAX_COLD_STEPS
        ctx["phi_applied_func"].assign(float(eta_try))
        converged, steps = _run_to_steady_state(max_steps)
        if converged:
            return True, True
        if steps == max_steps:
            # Budget exhausted but no SNES failure — usable warm-start
            print(f"  WARNING: step budget exhausted ({max_steps} steps) "
                  f"at eta={eta_try:.4f} without steady-state")
            return False, True
        # SNES divergence
        return False, False

    # ------------------------------------------------------------------
    # Step 6a: Shared _bisect_eta helper
    # ------------------------------------------------------------------
    def _bisect_eta(eta_lo, eta_target, U_ckpt_fn, U_prev_ckpt_fn, max_sub=6):
        """Bisect from eta_lo toward eta_target. Returns True if target reached."""
        eta_hi = eta_target
        for sub in range(max_sub):
            ctx["U"].assign(U_ckpt_fn)
            ctx["U_prev"].assign(U_prev_ckpt_fn)
            eta_mid = (eta_lo + eta_hi) / 2.0
            print(f"  [bisect {sub+1}/{max_sub}] eta_mid={eta_mid:.4f}")

            _conv, _usable = _try_timestep(eta_mid)
            if _usable:
                U_ckpt_fn.assign(ctx["U"])
                U_prev_ckpt_fn.assign(ctx["U_prev"])
                eta_lo = eta_mid
                # Try the original target from this better state
                ctx["U"].assign(U_ckpt_fn)
                ctx["U_prev"].assign(U_prev_ckpt_fn)
                _conv2, _usable2 = _try_timestep(eta_target)
                if _usable2:
                    return True
            else:
                eta_hi = eta_mid

        return False

    # ==================================================================
    # PHASE 1: Neutral Voltage Sweep (z=0)
    # ==================================================================
    print(f"[Phase 1] Starting neutral sweep over {len(sorted_indices)} points")
    neutral_solutions = {}

    for sweep_pos, orig_idx in enumerate(sorted_indices):
        eta_i = float(phi_applied_values[orig_idx])

        # --- Branch transition ---
        if (prev_solved_eta is not None
                and not first_branch_done
                and has_mixed_signs):
            prev_sign = prev_solved_eta > 0
            curr_sign = eta_i > 0
            if prev_sign != curr_sign:
                first_branch_done = True
                predictor_prev2 = None
                predictor_prev = None
                predictor_curr = None
                if hub_U_data is not None:
                    for src, dst in zip(hub_U_data, ctx["U"].dat):
                        dst.data[:] = src
                    ctx["U_prev"].assign(ctx["U"])
                    prev_solved_eta = hub_eta

        # --- Bridge sub-steps ---
        if max_eta_gap > 0 and prev_solved_eta is not None:
            gap = abs(eta_i - prev_solved_eta)
            if gap > max_eta_gap:
                n_bridges = int(np.ceil(gap / max_eta_gap)) - 1
                bridge_etas = np.linspace(prev_solved_eta, eta_i, n_bridges + 2)[1:-1]
                for eta_b in bridge_etas:
                    U_ckpt.assign(ctx["U"])
                    U_prev_ckpt.assign(ctx["U_prev"])
                    _conv_b, _usable_b = _try_timestep(eta_b)
                    bridge_ok = _usable_b
                    if not _usable_b:
                        # SNES divergence — restore pre-bridge state and bisect
                        ctx["U"].assign(U_ckpt)
                        ctx["U_prev"].assign(U_prev_ckpt)
                        bridge_ok = _bisect_eta(prev_solved_eta, eta_b, U_ckpt, U_prev_ckpt)
                        if not bridge_ok:
                            # Bisection also failed — restore pre-bridge state
                            ctx["U"].assign(U_ckpt)
                            ctx["U_prev"].assign(U_prev_ckpt)
                            print(f"  WARNING: bridge at eta={eta_b:.4f} did not converge")
                    # Only update predictor history if bridge produced usable state
                    if bridge_ok:
                        bridge_U_data = tuple(d.data_ro.copy() for d in ctx["U"].dat)
                        predictor_prev2 = predictor_prev
                        predictor_prev = predictor_curr
                        predictor_curr = (eta_b, bridge_U_data)
                        prev_solved_eta = eta_b
                        print(f"  [bridge] eta={eta_b:+8.4f} (neutral)")
                    else:
                        print(f"  [bridge] eta={eta_b:+8.4f} SKIPPED (kept prior state)")

        # --- Apply Lagrange predictor ---
        carry_U_data = tuple(d.data_ro.copy() for d in ctx["U"].dat)
        _apply_predictor(
            eta_i, ctx["U"], carry_U_data,
            predictor_prev, predictor_curr, predictor_prev2,
            n_species=n_species,
        )
        ctx["U_prev"].assign(ctx["U"])

        # --- Solve target point ---
        step_cap = _WARMSTART_MAX_STEPS if sweep_pos > 0 else _MAX_COLD_STEPS
        U_ckpt.assign(ctx["U"])
        U_prev_ckpt.assign(ctx["U_prev"])

        _conv_main, _usable_main = _try_timestep(eta_i, max_steps=step_cap)
        if _usable_main:
            pass  # Direct step succeeded (or usable budget-exhaustion)
        else:
            ctx["U"].assign(U_ckpt)
            ctx["U_prev"].assign(U_prev_ckpt)
            eta_lo = prev_solved_eta if prev_solved_eta is not None else 0.0
            if not _bisect_eta(eta_lo, eta_i, U_ckpt, U_prev_ckpt):
                print(f"  WARNING: Phase 1 failed at eta={eta_i:.4f} "
                      f"after bisection (snapshot partial state)")

        # --- Snapshot neutral solution ---
        neutral_solutions[orig_idx] = tuple(d.data_ro.copy() for d in ctx["U"].dat)

        # --- Update predictor history ---
        predictor_prev2 = predictor_prev
        predictor_prev = predictor_curr
        predictor_curr = (eta_i, neutral_solutions[orig_idx])
        prev_solved_eta = eta_i

        # --- Save hub state at first point (nearest equilibrium) ---
        if sweep_pos == 0:
            hub_U_data = neutral_solutions[orig_idx]
            hub_eta = eta_i

        if (sweep_pos + 1) % print_interval == 0 or sweep_pos == 0:
            print(f"[Phase 1] {sweep_pos+1}/{len(sorted_indices)}  "
                  f"eta={eta_i:+8.4f} (neutral, z=0)")

    print(f"[Phase 1] Complete: {len(neutral_solutions)} neutral solutions cached")

    # ==================================================================
    # PHASE 2: Per-Point Adaptive Z-Ramp (0 -> 1)
    # ==================================================================
    U_z_ckpt = fd.Function(ctx["U"])
    U_prev_z_ckpt = fd.Function(ctx["U_prev"])
    results = {}
    n_points = len(phi_applied_values)

    # Boltzmann counterion scale: when present, ramp it in lockstep with
    # the dynamic-species z so Phase 1 (z=0) is truly neutral (no analytic
    # counterion charge contribution either) and Phase 2 ramps both
    # together to reach the physical z=1 solution.
    boltz_z_scale = ctx.get("boltzmann_z_scale", None)

    def _set_z_factor(z_val: float, ns: int) -> None:
        """Set every z scaling (dynamic + Boltzmann) to ``z_val``."""
        for i in range(ns):
            ctx["z_consts"][i].assign(z_nominal[i] * z_val)
        if boltz_z_scale is not None:
            boltz_z_scale.assign(float(z_val))

    def _adaptive_z_ramp(z_nom, ns):
        """Adaptive z-ramp from 0 to 1. Returns achieved_z_factor."""

        def _try_z(z_val):
            """Returns (converged, usable) — same semantics as _try_timestep."""
            _set_z_factor(z_val, ns)
            converged, steps = _run_to_steady_state(_MAX_COLD_STEPS)
            if converged:
                return True, True
            if steps == _MAX_COLD_STEPS:
                print(f"    z={z_val:.4f}: step budget exhausted ({steps} steps)")
                return False, True
            return False, False

        def _checkpoint():
            U_z_ckpt.assign(ctx["U"])
            U_prev_z_ckpt.assign(ctx["U_prev"])

        def _restore():
            ctx["U"].assign(U_z_ckpt)
            ctx["U_prev"].assign(U_prev_z_ckpt)

        achieved_z = 0.0

        def _z_usable(z_val):
            """Try z_val; return True if usable (converged or budget-exhausted)."""
            _c, _u = _try_z(z_val)
            return _u

        # --- Step 1: Try z=1.0 directly (fast path) ---
        _checkpoint()
        if _z_usable(1.0):
            return 1.0

        _restore()

        # --- Step 2: Binary search for initial foothold ---
        search_z = 0.5
        search_lo = 0.0
        search_hi = 1.0
        max_search = 4

        for _ in range(max_search):
            _checkpoint()
            if _z_usable(search_z):
                achieved_z = search_z
                _checkpoint()
                break
            else:
                _restore()
                search_hi = search_z
                search_z = (search_lo + search_z) / 2.0
                if search_z < min_delta_z:
                    break

        if achieved_z < min_delta_z:
            return achieved_z

        # --- Step 3: Geometric acceleration toward z=1.0 ---
        max_accel_iters = 20

        for _ in range(max_accel_iters):
            if achieved_z >= 1.0 - 1e-6:
                break

            remaining = 1.0 - achieved_z
            if remaining < min_delta_z:
                break

            # Try jumping directly to z=1.0
            _checkpoint()
            if _z_usable(1.0):
                achieved_z = 1.0
                break

            # Jump to 1.0 failed -- try midpoint
            _restore()

            mid_z = achieved_z + remaining / 2.0
            _checkpoint()
            if _z_usable(mid_z):
                achieved_z = mid_z
                _checkpoint()
                continue

            # Midpoint also failed -- uniform fine steps in [achieved_z, mid_z]
            _restore()
            n_fine = max(2, charge_steps // 2)
            fine_targets = np.linspace(achieved_z, mid_z, n_fine + 1)[1:]
            for z_t in fine_targets:
                _checkpoint()
                if _z_usable(z_t):
                    achieved_z = z_t
                    _checkpoint()
                else:
                    _restore()
                    inner_gap = z_t - achieved_z
                    if inner_gap < min_delta_z:
                        break
                    inner_mid = achieved_z + inner_gap / 2.0
                    _checkpoint()
                    if _z_usable(inner_mid):
                        achieved_z = inner_mid
                        _checkpoint()
                    else:
                        _restore()
                        break  # Stuck

        return achieved_z

    print(f"[Phase 2] Starting z-ramp for {n_points} points")

    for orig_idx in range(n_points):
        eta_i = float(phi_applied_values[orig_idx])

        # Restore neutral solution from Phase 1
        for src, dst in zip(neutral_solutions[orig_idx], ctx["U"].dat):
            dst.data[:] = src
        ctx["U_prev"].assign(ctx["U"])
        ctx["phi_applied_func"].assign(eta_i)

        # Reset z_consts to 0 (and Boltzmann z_scale if present, so the
        # analytic counterion is also off at the start of the per-point
        # z-ramp).
        _set_z_factor(0.0, n_s)

        # Run adaptive z-ramp
        achieved_z = _adaptive_z_ramp(z_nominal, n_s)

        # Snapshot final state
        U_data = tuple(d.data_ro.copy() for d in ctx["U"].dat)

        # --- Physics validation ---
        from .validation import validate_solution_state
        _c_bulk = [float(v) for v in (list(c0) if hasattr(c0, '__iter__') else [c0] * n_s)]
        _vr = validate_solution_state(
            ctx["U"], n_species=n_s, c_bulk=_c_bulk, phi_applied=eta_i,
            z_vals=[float(v) for v in z_nominal],
            eps_c=ctx.get("_diag_eps_c", 1e-8),
            exponent_clip=ctx.get("_diag_exponent_clip", 50.0),
            is_logc=bool(ctx.get("logc_transform", False)),
        )
        _phys_valid = _vr.valid

        # Optional callback for observable extraction from live ctx
        if per_point_callback is not None:
            per_point_callback(orig_idx, eta_i, ctx)

        results[orig_idx] = GridPointResult(
            orig_idx=orig_idx,
            phi_applied=eta_i,
            U_data=U_data,
            achieved_z_factor=achieved_z,
            converged=(achieved_z >= 1.0 - 1e-6) and _phys_valid,
            validation=_vr,
        )

        tag = "OK" if achieved_z >= 1.0 - 1e-6 else f"PARTIAL z={achieved_z:.3f}"
        if (orig_idx + 1) % print_interval == 0 or orig_idx == 0:
            print(f"[Phase 2] {orig_idx+1}/{n_points}  eta={eta_i:+8.4f}  {tag}")

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    mesh_dof_count = ctx["U"].function_space().dim()

    n_converged = sum(1 for p in results.values() if p.converged)
    print(f"[Done] {n_converged}/{n_points} points fully converged (z=1.0)")

    return GridChargeContinuationResult(
        points=results,
        mesh_dof_count=mesh_dof_count,
    )
