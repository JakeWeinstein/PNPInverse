"""Parallel fast-path executor for BV per-point adjoint solves.

When the checkpoint-restart cache (_all_points_cache) is populated, each
voltage point can be solved independently using its cached IC.  This module
dispatches those independent solves to a ProcessPoolExecutor (spawn context)
for wall-clock parallelism.

Architecture follows the proven Robin pipeline pattern in point_solve.py:
- Frozen config dataclass shipped to workers at pool init
- Per-task tuple with point-specific data (phi, target, control values,
  cached IC arrays)
- Workers import firedrake in their own process, build mesh/forms/solver,
  load cached IC, run forward+adjoint, return serializable result dict

Multi-observable support (v6, 2026-03-02):
  When ``secondary_observable_mode`` is set in the config, each worker
  computes adjoint gradients for BOTH observables after a single forward
  solve.  The "two-tape-pass" approach clears the adjoint tape and
  re-annotates from the converged IC for each observable, requiring only
  1 Newton iteration per tape pass (converged -> converged).  This avoids
  the previous fallback where the secondary observable ran sequentially.

Each worker sets OMP_NUM_THREADS=1 to prevent thread oversubscription.
"""

from __future__ import annotations

import copy
import os
import time as _time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Frozen config shipped to each worker at pool initialization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BVParallelPointConfig:
    """Immutable, pickle-serializable config for BV parallel workers.

    Contains everything a worker needs to build mesh + forms + solver,
    EXCEPT the per-evaluation dynamic data (k0/alpha values, cached ICs,
    targets) which arrive in each task tuple.
    """

    # Solver params as a plain list of primitive Python objects
    base_solver_params: list
    # SteadyStateConfig fields (extracted to primitives for pickling)
    ss_relative_tolerance: float
    ss_absolute_tolerance: float
    ss_consecutive_steps: int
    ss_max_steps: int
    # Mesh parameters
    mesh_Nx: int
    mesh_Ny: int
    mesh_beta: float
    # Solver behavior
    blob_initial_condition: bool
    fail_penalty: float
    warmstart_max_steps: int
    # Observable
    observable_mode: str
    observable_reaction_index: Optional[int]
    observable_scale: float
    # Control
    control_mode: str
    n_controls: int
    # SER adaptive dt constants
    ser_growth_cap: float
    ser_shrink: float
    ser_dt_max_ratio: float
    # Multi-observable support (v6): when set, workers compute gradients for
    # both the primary and secondary observable in a single task.
    secondary_observable_mode: Optional[str] = None
    secondary_observable_reaction_index: Optional[int] = None
    secondary_observable_scale: Optional[float] = None


# Module-level worker state (set by initializer, used by solve function)
_WORKER_CONFIG: Optional[BVParallelPointConfig] = None
_WORKER_MESH: Any = None  # Firedrake mesh, built once per worker


def _bv_worker_init(config: BVParallelPointConfig) -> None:
    """Initialize one worker process with static config.

    Called once when the worker process starts.  Sets OMP_NUM_THREADS=1,
    stores the config, and pre-builds the shared mesh.
    """
    global _WORKER_CONFIG, _WORKER_MESH

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    _WORKER_CONFIG = config

    # Import firedrake and build the mesh once per worker
    from Forward.bv_solver import make_graded_rectangle_mesh
    _WORKER_MESH = make_graded_rectangle_mesh(
        Nx=config.mesh_Nx, Ny=config.mesh_Ny, beta=config.mesh_beta,
    )


def _bv_worker_adjoint_tape_pass(
    *,
    cfg: BVParallelPointConfig,
    params: object,
    k0_list: list,
    alpha_list: Optional[list],
    a_list: Optional[list],
    converged_U_arrays: list,
    target_flux: float,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
) -> Dict[str, object]:
    """Run one adjoint tape pass: annotate, load converged IC, 1 SNES step, adjoint.

    This is the core of the "two-tape-pass" multi-observable approach.  The
    caller has already completed the forward PDE solve and extracted
    ``converged_U_arrays``.  This function:

    1. Clears the adjoint tape and enables annotation.
    2. Rebuilds context/forms using the worker's shared mesh.
    3. Loads the converged solution as IC.
    4. Runs 1 SNES step (converged -> converged, typically 1 Newton iter).
    5. Builds the specified observable form and computes the adjoint gradient.

    Returns a dict with ``simulated_flux``, ``objective``, ``gradient``
    (as list), and ``success`` flag.
    """
    global _WORKER_MESH

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )
    from FluxCurve.bv_observables import (
        _build_bv_observable_form,
        _build_bv_scalar_target_in_control_space,
        _bv_gradient_controls_to_array,
    )

    # Clear tape and enable annotation
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    # Rebuild context and forms (fresh tape)
    ctx = bv_build_context(params, mesh=_WORKER_MESH)
    ctx = bv_build_forms(ctx, params)
    bv_set_initial_conditions(ctx, params, blob=cfg.blob_initial_condition)

    # Load converged IC
    for src_arr, dst in zip(converged_U_arrays, ctx["U"].dat):
        dst.data[:] = src_arr
    ctx["U_prev"].assign(ctx["U"])

    # Assign control values
    k0_funcs = list(ctx["bv_k0_funcs"])
    for j, k0_f in enumerate(k0_funcs):
        if j < len(k0_list):
            k0_f.assign(float(k0_list[j]))
    alpha_funcs = list(ctx.get("bv_alpha_funcs", []))
    if alpha_list is not None:
        for j, alpha_f in enumerate(alpha_funcs):
            if j < len(alpha_list):
                alpha_f.assign(float(alpha_list[j]))
    steric_a_funcs = list(ctx.get("steric_a_funcs", []))
    if a_list is not None:
        for j, a_f in enumerate(steric_a_funcs):
            if j < len(a_list):
                a_f.assign(float(a_list[j]))

    # Select control functions
    if cfg.control_mode == "k0":
        control_funcs = list(k0_funcs)
    elif cfg.control_mode == "alpha":
        control_funcs = list(alpha_funcs)
    elif cfg.control_mode == "joint":
        control_funcs = list(k0_funcs) + list(alpha_funcs)
    elif cfg.control_mode == "steric":
        control_funcs = list(steric_a_funcs)
    elif cfg.control_mode == "full":
        control_funcs = list(k0_funcs) + list(alpha_funcs) + list(steric_a_funcs)
    else:
        control_funcs = list(k0_funcs)

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]

    jac = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
    _bpar_opts = params.solver_options if hasattr(params, 'solver_options') else params[10]
    solve_params = dict(_bpar_opts) if isinstance(_bpar_opts, dict) else {}
    solve_params.setdefault("snes_lag_jacobian", 2)
    solve_params.setdefault("snes_lag_jacobian_persists", True)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_params)

    # 1 SNES step (converged -> converged, ~1 Newton iter)
    try:
        solver.solve()
    except Exception as exc:
        return {
            "success": False,
            "simulated_flux": float("nan"),
            "objective": 0.0,
            "gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
            "reason": f"tape-pass SNES failed: {type(exc).__name__}",
        }
    U_prev.assign(U)

    # Build observable form for this specific observable
    obs_form = _build_bv_observable_form(
        ctx, mode=observable_mode,
        reaction_index=observable_reaction_index,
        scale=float(observable_scale),
    )

    # Compute adjoint gradient
    target_ctrl = _build_bv_scalar_target_in_control_space(
        ctx, target_flux, name="target_flux_value",
        control_mode=cfg.control_mode,
    )
    target_scalar = fd.assemble(target_ctrl * fd.dx(domain=ctx["mesh"]))
    sim_scalar = fd.assemble(obs_form)
    point_objective = 0.5 * (sim_scalar - target_scalar) ** 2

    controls = [adj.Control(cf) for cf in control_funcs]
    rf = adj.ReducedFunctional(point_objective, controls)
    try:
        point_gradient = _bv_gradient_controls_to_array(
            rf.derivative(), cfg.n_controls
        )
    except Exception as exc:
        return {
            "success": False,
            "simulated_flux": float(sim_scalar),
            "objective": 0.0,
            "gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
            "reason": f"tape-pass adjoint failed: {type(exc).__name__}",
        }

    return {
        "success": True,
        "simulated_flux": float(sim_scalar),
        "objective": float(point_objective),
        "gradient": point_gradient.tolist(),
        "reason": "",
    }


def _bv_worker_solve_point(
    task: Tuple,
) -> Dict[str, object]:
    """Worker entrypoint: solve one BV point and return serializable result.

    Task tuple contents (single-observable mode):
        (point_index, phi_applied, target_flux,
         k0_list, alpha_list, a_list,
         cached_U_arrays,   # tuple of numpy arrays (one per sub-function)
         )

    Task tuple contents (multi-observable mode, 9 elements):
        (point_index, phi_applied, target_flux_primary,
         k0_list, alpha_list, a_list,
         cached_U_arrays,
         target_flux_secondary,   # float target for secondary observable
         multi_obs_flag,          # True to signal multi-obs mode
         )

    When multi-obs is active, the worker:
    1. Runs the full forward PDE solve (pseudo-time stepping to steady state).
    2. Extracts converged_U_arrays.
    3. Runs two adjoint tape passes (one per observable) using the converged IC.
    4. Returns both results in the dict (primary + secondary keys).
    """
    global _WORKER_CONFIG, _WORKER_MESH

    if _WORKER_CONFIG is None or _WORKER_MESH is None:
        raise RuntimeError("BV parallel worker not initialized.")

    cfg = _WORKER_CONFIG

    # Parse task tuple -- detect multi-obs by length
    multi_obs = False
    target_flux_secondary = None
    if len(task) >= 9:
        (point_index, phi_applied, target_flux,
         k0_list, alpha_list, a_list,
         cached_U_arrays, target_flux_secondary,
         multi_obs_flag) = task[:9]
        multi_obs = bool(multi_obs_flag)
        if target_flux_secondary is not None:
            target_flux_secondary = float(target_flux_secondary)
    else:
        (point_index, phi_applied, target_flux,
         k0_list, alpha_list, a_list,
         cached_U_arrays) = task[:7]

    point_index = int(point_index)
    phi_applied = float(phi_applied)
    target_flux = float(target_flux)
    k0_list = [float(v) for v in k0_list]
    alpha_list = [float(v) for v in alpha_list] if alpha_list is not None else None
    a_list = [float(v) for v in a_list] if a_list is not None else None

    t_start = _time.perf_counter()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )
    from Forward.steady_state import configure_bv_solver_params
    from FluxCurve.bv_observables import (
        _build_bv_observable_form,
        _build_bv_scalar_target_in_control_space,
        _bv_gradient_controls_to_array,
    )

    # Build solver params for this point
    params = configure_bv_solver_params(
        cfg.base_solver_params,
        phi_applied=phi_applied,
        k0_values=k0_list,
        alpha_values=alpha_list,
        a_values=a_list,
    )

    # Clear tape and enable annotation
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    # Build context and forms using the worker's shared mesh
    ctx = bv_build_context(params, mesh=_WORKER_MESH)
    ctx = bv_build_forms(ctx, params)
    bv_set_initial_conditions(ctx, params, blob=cfg.blob_initial_condition)

    # Load cached IC
    for src_arr, dst in zip(cached_U_arrays, ctx["U"].dat):
        dst.data[:] = src_arr
    ctx["U_prev"].assign(ctx["U"])

    # Assign control values
    k0_funcs = list(ctx["bv_k0_funcs"])
    for j, k0_f in enumerate(k0_funcs):
        if j < len(k0_list):
            k0_f.assign(float(k0_list[j]))
    alpha_funcs = list(ctx.get("bv_alpha_funcs", []))
    if alpha_list is not None:
        for j, alpha_f in enumerate(alpha_funcs):
            if j < len(alpha_list):
                alpha_f.assign(float(alpha_list[j]))
    steric_a_funcs = list(ctx.get("steric_a_funcs", []))
    if a_list is not None:
        for j, a_f in enumerate(steric_a_funcs):
            if j < len(a_list):
                a_f.assign(float(a_list[j]))

    # Select control functions based on control_mode
    if cfg.control_mode == "k0":
        control_funcs = list(k0_funcs)
    elif cfg.control_mode == "alpha":
        control_funcs = list(alpha_funcs)
    elif cfg.control_mode == "joint":
        control_funcs = list(k0_funcs) + list(alpha_funcs)
    elif cfg.control_mode == "steric":
        control_funcs = list(steric_a_funcs)
    elif cfg.control_mode == "full":
        control_funcs = list(k0_funcs) + list(alpha_funcs) + list(steric_a_funcs)
    else:
        control_funcs = list(k0_funcs)

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]

    jac = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
    _bpar_opts = params.solver_options if hasattr(params, 'solver_options') else params[10]
    solve_params = dict(_bpar_opts) if isinstance(_bpar_opts, dict) else {}
    solve_params.setdefault("snes_lag_jacobian", 2)
    solve_params.setdefault("snes_lag_jacobian_persists", True)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_params)

    observable_form = _build_bv_observable_form(
        ctx, mode=cfg.observable_mode,
        reaction_index=cfg.observable_reaction_index,
        scale=float(cfg.observable_scale),
    )

    # SER adaptive dt
    dt_const = ctx.get("dt_const")
    dt_initial = float(dt_const) if dt_const is not None else 1.0
    dt_current = dt_initial
    dt_max = dt_initial * cfg.ser_dt_max_ratio

    abs_tol = float(max(cfg.ss_absolute_tolerance, 1e-16))
    rel_tol = cfg.ss_relative_tolerance
    required_steady = cfg.ss_consecutive_steps
    effective_max_steps = min(cfg.ss_max_steps, cfg.warmstart_max_steps)

    prev_flux_val = None
    steady_count = 0
    rel_metric = None
    abs_metric = None
    simulated_flux = float("nan")
    steps_taken = 0
    failed = False
    prev_delta = None

    for step in range(1, effective_max_steps + 1):
        steps_taken = step
        try:
            solver.solve()
        except Exception:
            failed = True
            break
        U_prev.assign(U)

        with adj.stop_annotating():
            simulated_flux = float(fd.assemble(observable_form))

        if prev_flux_val is not None:
            delta = abs(simulated_flux - prev_flux_val)
            scale_val = max(abs(simulated_flux), abs(prev_flux_val), abs_tol)
            rel_metric = delta / scale_val
            abs_metric = delta
            is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
            steady_count = steady_count + 1 if is_steady else 0

            if dt_const is not None and prev_delta is not None and delta > 0:
                ratio = prev_delta / delta
                if ratio > 1.0:
                    grow = min(ratio, cfg.ser_growth_cap)
                    dt_current = min(dt_current * grow, dt_max)
                else:
                    dt_current = max(dt_current * cfg.ser_shrink, dt_initial)
                dt_const.assign(dt_current)
            prev_delta = delta
        else:
            steady_count = 0
        prev_flux_val = simulated_flux

        if steady_count >= required_steady:
            break

    if dt_const is not None:
        dt_const.assign(dt_initial)

    if failed or steady_count < required_steady:
        # Point failed -- return failure result
        fail_result = {
            "point_index": point_index,
            "success": False,
            "phi_applied": phi_applied,
            "target_flux": target_flux,
            "simulated_flux": float("nan"),
            "objective": float(cfg.fail_penalty),
            "gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
            "converged": False,
            "steps_taken": steps_taken,
            "reason": "forward solve failed or not converged",
            "rel_metric": None,
            "abs_metric": None,
            "converged_U_arrays": None,
        }
        if multi_obs:
            fail_result["secondary_simulated_flux"] = float("nan")
            fail_result["secondary_objective"] = float(cfg.fail_penalty)
            fail_result["secondary_gradient"] = np.zeros(cfg.n_controls, dtype=float).tolist()
        return fail_result

    # ---- Extract converged solution for cache update and tape passes ----
    converged_U_arrays = [d.data_ro.copy() for d in U.dat]

    # ---- Multi-observable mode: two tape passes ----
    if multi_obs and cfg.secondary_observable_mode is not None:
        # Tape pass 1: primary observable
        primary_result = _bv_worker_adjoint_tape_pass(
            cfg=cfg,
            params=params,
            k0_list=k0_list,
            alpha_list=alpha_list,
            a_list=a_list,
            converged_U_arrays=converged_U_arrays,
            target_flux=target_flux,
            observable_mode=cfg.observable_mode,
            observable_reaction_index=cfg.observable_reaction_index,
            observable_scale=float(cfg.observable_scale),
        )
        if not primary_result["success"]:
            fail_result = {
                "point_index": point_index,
                "success": False,
                "phi_applied": phi_applied,
                "target_flux": target_flux,
                "simulated_flux": float("nan"),
                "objective": float(cfg.fail_penalty),
                "gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
                "converged": False,
                "steps_taken": steps_taken,
                "reason": primary_result.get("reason", "primary adjoint tape pass failed"),
                "rel_metric": rel_metric,
                "abs_metric": abs_metric,
                "converged_U_arrays": None,
                "secondary_simulated_flux": float("nan"),
                "secondary_objective": float(cfg.fail_penalty),
                "secondary_gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
            }
            return fail_result

        # Tape pass 2: secondary observable
        sec_target = target_flux_secondary if target_flux_secondary is not None else 0.0
        secondary_result = _bv_worker_adjoint_tape_pass(
            cfg=cfg,
            params=params,
            k0_list=k0_list,
            alpha_list=alpha_list,
            a_list=a_list,
            converged_U_arrays=converged_U_arrays,
            target_flux=sec_target,
            observable_mode=cfg.secondary_observable_mode,
            observable_reaction_index=cfg.secondary_observable_reaction_index,
            observable_scale=float(cfg.secondary_observable_scale),
        )
        if not secondary_result["success"]:
            fail_result = {
                "point_index": point_index,
                "success": False,
                "phi_applied": phi_applied,
                "target_flux": target_flux,
                "simulated_flux": float(primary_result["simulated_flux"]),
                "objective": float(cfg.fail_penalty),
                "gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
                "converged": False,
                "steps_taken": steps_taken,
                "reason": secondary_result.get("reason", "secondary adjoint tape pass failed"),
                "rel_metric": rel_metric,
                "abs_metric": abs_metric,
                "converged_U_arrays": None,
                "secondary_simulated_flux": float("nan"),
                "secondary_objective": float(cfg.fail_penalty),
                "secondary_gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
            }
            return fail_result

        t_elapsed = _time.perf_counter() - t_start

        return {
            "point_index": point_index,
            "success": True,
            "phi_applied": phi_applied,
            "target_flux": target_flux,
            "simulated_flux": float(primary_result["simulated_flux"]),
            "objective": float(primary_result["objective"]),
            "gradient": primary_result["gradient"],
            "converged": True,
            "steps_taken": steps_taken,
            "reason": "",
            "rel_metric": float(rel_metric) if rel_metric is not None else None,
            "abs_metric": float(abs_metric) if abs_metric is not None else None,
            "converged_U_arrays": converged_U_arrays,
            "elapsed": t_elapsed,
            # Secondary observable results
            "secondary_simulated_flux": float(secondary_result["simulated_flux"]),
            "secondary_objective": float(secondary_result["objective"]),
            "secondary_gradient": secondary_result["gradient"],
        }

    # ---- Single-observable mode (original path) ----
    # Compute adjoint gradient
    target_ctrl = _build_bv_scalar_target_in_control_space(
        ctx, target_flux, name="target_flux_value",
        control_mode=cfg.control_mode,
    )
    target_scalar = fd.assemble(target_ctrl * fd.dx(domain=ctx["mesh"]))
    sim_scalar = fd.assemble(observable_form)
    point_objective = 0.5 * (sim_scalar - target_scalar) ** 2

    controls = [adj.Control(cf) for cf in control_funcs]
    rf = adj.ReducedFunctional(point_objective, controls)
    try:
        point_gradient = _bv_gradient_controls_to_array(
            rf.derivative(), cfg.n_controls
        )
    except Exception:
        return {
            "point_index": point_index,
            "success": False,
            "phi_applied": phi_applied,
            "target_flux": target_flux,
            "simulated_flux": float(sim_scalar),
            "objective": float(cfg.fail_penalty),
            "gradient": np.zeros(cfg.n_controls, dtype=float).tolist(),
            "converged": False,
            "steps_taken": steps_taken,
            "reason": "adjoint derivative failed",
            "rel_metric": rel_metric,
            "abs_metric": abs_metric,
            "converged_U_arrays": None,
        }

    t_elapsed = _time.perf_counter() - t_start

    return {
        "point_index": point_index,
        "success": True,
        "phi_applied": phi_applied,
        "target_flux": target_flux,
        "simulated_flux": float(sim_scalar),
        "objective": float(point_objective),
        "gradient": point_gradient.tolist(),
        "converged": True,
        "steps_taken": steps_taken,
        "reason": "",
        "rel_metric": float(rel_metric) if rel_metric is not None else None,
        "abs_metric": float(abs_metric) if abs_metric is not None else None,
        "converged_U_arrays": converged_U_arrays,
        "elapsed": t_elapsed,
    }


# ---------------------------------------------------------------------------
# Pool manager class
# ---------------------------------------------------------------------------

class BVPointSolvePool:
    """Manages a ProcessPoolExecutor for parallel BV fast-path solves.

    Usage::

        pool = BVPointSolvePool(config, n_workers=10)
        # ... in optimizer loop ...
        results = pool.solve_points(tasks)
        # ... when done ...
        pool.close()
    """

    def __init__(
        self,
        config: BVParallelPointConfig,
        n_workers: int = 0,
    ) -> None:
        self.enabled = False
        self.n_workers = 0
        self._executor: Optional[ProcessPoolExecutor] = None
        self._config = config

        if n_workers <= 0:
            # Strategy D: use more workers by default (cpu_count - 1)
            n_workers = max(1, (os.cpu_count() or 4) - 1)

        try:
            ctx = mp.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_bv_worker_init,
                initargs=(config,),
            )
            self.enabled = True
            self.n_workers = n_workers
            print(
                f"[bv-parallel] Pool initialized: {n_workers} workers "
                f"(spawn context)"
            )
        except Exception as exc:
            self.enabled = False
            self.n_workers = 0
            self._executor = None
            print(
                f"[bv-parallel] Pool initialization failed: "
                f"{type(exc).__name__}: {exc}; using serial fast path"
            )

    def close(self) -> None:
        """Shutdown worker pool."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
            self._executor = None
        self.enabled = False

    def solve_points(
        self,
        tasks: List[Tuple],
    ) -> Optional[List[Dict[str, object]]]:
        """Submit all tasks to the pool and collect results.

        Returns a list of result dicts (one per task, in task order),
        or None if the pool is unavailable or any unrecoverable error occurs.
        """
        if not self.enabled or self._executor is None:
            return None

        n_tasks = len(tasks)
        results: List[Optional[Dict[str, object]]] = [None] * n_tasks

        try:
            future_map = {}
            for task in tasks:
                point_index = int(task[0])
                future = self._executor.submit(_bv_worker_solve_point, task)
                future_map[future] = point_index

            for future in as_completed(future_map):
                expected_idx = future_map[future]
                result = future.result()
                actual_idx = int(result["point_index"])
                if actual_idx != expected_idx:
                    raise RuntimeError(
                        f"Worker returned mismatched index "
                        f"(expected {expected_idx}, got {actual_idx})"
                    )
                results[expected_idx] = result

        except Exception as exc:
            print(
                f"[bv-parallel] Execution failed: "
                f"{type(exc).__name__}: {exc}; falling back to serial"
            )
            return None

        # Check all results are populated
        for i, r in enumerate(results):
            if r is None:
                print(f"[bv-parallel] Missing result for point {i}; falling back to serial")
                return None

        return results  # type: ignore[return-value]

    @property
    def is_multi_obs(self) -> bool:
        """True if this pool's config supports multi-observable mode."""
        return (
            self._config is not None
            and self._config.secondary_observable_mode is not None
        )
