"""Training data generation for the BV surrogate model.

Uses the existing FluxCurve BV point solver to compute I-V curves at
sampled parameter sets.  Supports checkpointing for long runs.

Progress reporting
------------------
Every sample prints a line with: ``[N/total] params, status, elapsed, ETA``.
All prints use ``flush=True`` so output streams immediately to the terminal
(critical for monitoring long runs piped through ``tee`` or redirected to a
log file).
"""

from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``Xh Ym Zs``."""
    if seconds < 0:
        return "???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def generate_training_data_single(
    *,
    k0_values: Sequence[float],
    alpha_values: Sequence[float],
    phi_applied_values: np.ndarray,
    base_solver_params: Sequence[object],
    steady: Any,
    observable_scale: float,
    mesh: Any = None,
    max_eta_gap: float = 3.0,
    fail_penalty: float = 1e9,
    initial_solutions: Optional[Dict[int, tuple]] = None,
    return_solutions: bool = False,
    forward_recovery: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one I-V curve at the given (k0, alpha) parameters.

    Uses ``solve_bv_curve_points_with_warmstart`` from FluxCurve to compute
    both current_density and peroxide_current observables.

    Parameters
    ----------
    k0_values : sequence of float
        Rate constants [k0_1, k0_2] (dimensionless).
    alpha_values : sequence of float
        Transfer coefficients [alpha_1, alpha_2].
    phi_applied_values : np.ndarray
        Voltage grid (dimensionless eta_hat).
    base_solver_params : sequence
        11-element SolverParams list.
    steady : SteadyStateConfig
        Steady-state convergence config.
    observable_scale : float
        Scale factor for observable (e.g. -I_SCALE).
    mesh : optional
        Pre-built mesh.
    max_eta_gap : float
        Max gap for bridge point insertion.
    fail_penalty : float
        Penalty for failed points.
    initial_solutions : dict or None
        If provided, seed ``_cross_eval_cache`` with these (index -> U_data)
        entries before the CD sweep.  Enables cross-sample warm-starting.
    return_solutions : bool
        If True, extract converged U_data from the cache and include it in
        the return dict as ``'converged_solutions'``.
    forward_recovery : ForwardRecoveryConfig or None
        Override the default recovery config.  If None, uses a standard
        config with 4 attempts.

    Returns
    -------
    dict with keys:
        'current_density' : np.ndarray of shape (n_eta,)
        'peroxide_current' : np.ndarray of shape (n_eta,)
        'converged_mask' : np.ndarray of bool, shape (n_eta,)
        'n_converged' : int
        'converged_solutions' : dict or None (if return_solutions)
    """
    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )
    from FluxCurve.bv_point_solve import cache as _cache_mod
    from FluxCurve.config import ForwardRecoveryConfig

    if forward_recovery is not None:
        recovery = forward_recovery
    else:
        recovery = ForwardRecoveryConfig(
            max_attempts=4, max_it_only_attempts=2,
            anisotropy_only_attempts=0, tolerance_relax_attempts=1,
            max_it_growth=1.5, max_it_cap=500,
        )

    n_eta = len(phi_applied_values)
    dummy_target = np.zeros(n_eta, dtype=float)
    k0_list = [float(v) for v in k0_values]
    alpha_list = [float(v) for v in alpha_values]

    # Solve for current_density observable
    _clear_caches()
    # Seed cross-eval cache for cross-sample warm-starting
    if initial_solutions is not None:
        for idx, u_data in initial_solutions.items():
            _cache_mod._cross_eval_cache[idx] = u_data

    points_cd = solve_bv_curve_points_with_warmstart(
        base_solver_params=base_solver_params,
        steady=steady,
        phi_applied_values=phi_applied_values,
        target_flux=dummy_target,
        k0_values=k0_list,
        blob_initial_condition=False,
        fail_penalty=fail_penalty,
        forward_recovery=recovery,
        observable_mode="current_density",
        observable_reaction_index=None,
        observable_scale=observable_scale,
        mesh=mesh,
        alpha_values=alpha_list,
        control_mode="joint",
        max_eta_gap=max_eta_gap,
    )

    cd_flux = np.array([float(p.simulated_flux) for p in points_cd], dtype=float)
    cd_converged = np.array([bool(p.converged) for p in points_cd], dtype=bool)

    # Save CD solutions to reuse as IC for PC sweep
    cd_solutions = dict(_cache_mod._cross_eval_cache)

    # Solve for peroxide_current observable -- reuse CD solutions as IC
    _clear_caches()
    for idx, u_data in cd_solutions.items():
        _cache_mod._cross_eval_cache[idx] = u_data

    points_pc = solve_bv_curve_points_with_warmstart(
        base_solver_params=base_solver_params,
        steady=steady,
        phi_applied_values=phi_applied_values,
        target_flux=dummy_target,
        k0_values=k0_list,
        blob_initial_condition=False,
        fail_penalty=fail_penalty,
        forward_recovery=recovery,
        observable_mode="peroxide_current",
        observable_reaction_index=None,
        observable_scale=observable_scale,
        mesh=mesh,
        alpha_values=alpha_list,
        control_mode="joint",
        max_eta_gap=max_eta_gap,
    )

    pc_flux = np.array([float(p.simulated_flux) for p in points_pc], dtype=float)
    pc_converged = np.array([bool(p.converged) for p in points_pc], dtype=bool)

    # Combined convergence mask
    converged_mask = cd_converged & pc_converged

    # Extract solutions for cross-sample warm-start chain
    converged_solutions = None
    if return_solutions:
        converged_solutions = dict(_cache_mod._cross_eval_cache)

    _clear_caches()

    result: Dict[str, Any] = {
        "current_density": cd_flux,
        "peroxide_current": pc_flux,
        "converged_mask": converged_mask,
        "n_converged": int(converged_mask.sum()),
    }
    if return_solutions:
        result["converged_solutions"] = converged_solutions
    return result


def _interpolate_failed_points(
    flux: np.ndarray,
    converged: np.ndarray,
    phi_applied: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate failed (non-converged) points from neighbors.

    Only interpolates if enough points converged (>50%).  Returns a copy
    with failed points filled in.
    """
    result = flux.copy()
    good = converged.astype(bool)
    n_good = good.sum()

    if n_good < 2 or n_good == len(flux):
        return result

    # Use numpy interp for 1D interpolation.
    # np.interp requires xp to be monotonically increasing, but phi_applied
    # may be descending (e.g. sorted voltage grid).  Sort the good points.
    good_idx = np.where(good)[0]
    bad_idx = np.where(~good)[0]

    xp = phi_applied[good_idx]
    fp = flux[good_idx]
    sort_order = np.argsort(xp)

    result[bad_idx] = np.interp(
        phi_applied[bad_idx],
        xp[sort_order],
        fp[sort_order],
    )
    return result


def generate_training_dataset(
    parameter_samples: np.ndarray,
    *,
    phi_applied_values: np.ndarray,
    base_solver_params: Sequence[object],
    steady: Any,
    observable_scale: float,
    mesh: Any = None,
    max_eta_gap: float = 3.0,
    fail_penalty: float = 1e9,
    output_path: str,
    checkpoint_interval: int = 10,
    resume_from: Optional[str] = None,
    min_converged_fraction: float = 0.8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate training data for all parameter samples.

    Loops over each sample, runs the BV solver for both observables,
    and saves results to a ``.npz`` file with periodic checkpointing.

    Progress output streams immediately (``flush=True``) with per-sample
    status, elapsed time, and ETA.

    Parameters
    ----------
    parameter_samples : np.ndarray of shape (N, 4)
        Columns: [k0_1, k0_2, alpha_1, alpha_2].
    phi_applied_values : np.ndarray of shape (n_eta,)
        Voltage grid.
    base_solver_params, steady, observable_scale, mesh, max_eta_gap :
        Passed through to ``generate_training_data_single``.
    output_path : str
        Path for final ``.npz`` output file.
    checkpoint_interval : int
        Save checkpoint every N samples.
    resume_from : str or None
        Path to checkpoint ``.npz`` to resume from.
    min_converged_fraction : float
        Minimum fraction of points that must converge for a sample to be
        included.  Samples below this threshold are skipped.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'parameters' : np.ndarray (N_valid, 4)
        'current_density' : np.ndarray (N_valid, n_eta)
        'peroxide_current' : np.ndarray (N_valid, n_eta)
        'phi_applied' : np.ndarray (n_eta,)
        'n_valid' : int
        'n_total' : int
        'n_failed' : int
        'timings' : np.ndarray (N_total,)
    """
    N = parameter_samples.shape[0]
    n_eta = len(phi_applied_values)

    # Pre-allocate storage
    all_cd = np.full((N, n_eta), np.nan, dtype=float)
    all_pc = np.full((N, n_eta), np.nan, dtype=float)
    all_converged = np.zeros(N, dtype=bool)
    all_timings = np.zeros(N, dtype=float)
    start_idx = 0

    # Resume from checkpoint
    if resume_from is not None and os.path.exists(resume_from):
        if verbose:
            print(f"RESUME: Loading checkpoint: {resume_from}", flush=True)
        ckpt = np.load(resume_from, allow_pickle=True)
        n_done = int(ckpt["n_completed"])
        # Restore completed samples
        all_cd[:n_done] = ckpt["current_density"][:n_done]
        all_pc[:n_done] = ckpt["peroxide_current"][:n_done]
        all_converged[:n_done] = ckpt["converged"][:n_done]
        if "timings" in ckpt:
            all_timings[:n_done] = ckpt["timings"][:n_done]
        start_idx = n_done
        if verbose:
            n_valid_so_far = int(all_converged[:n_done].sum())
            print(f"RESUME: Restored {n_done}/{N} samples "
                  f"({n_valid_so_far} valid, {n_done - n_valid_so_far} failed)",
                  flush=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Summary header
    # ------------------------------------------------------------------
    if verbose:
        k0_1_min, k0_1_max = parameter_samples[:, 0].min(), parameter_samples[:, 0].max()
        k0_2_min, k0_2_max = parameter_samples[:, 1].min(), parameter_samples[:, 1].max()
        a1_min, a1_max = parameter_samples[:, 2].min(), parameter_samples[:, 2].max()
        a2_min, a2_max = parameter_samples[:, 3].min(), parameter_samples[:, 3].max()

        print(f"\n{'='*78}", flush=True)
        print(f"  TRAINING DATA GENERATION", flush=True)
        print(f"  Total samples : {N}", flush=True)
        print(f"  Starting from : {start_idx}", flush=True)
        print(f"  Remaining     : {N - start_idx}", flush=True)
        print(f"  Voltage pts   : {n_eta}", flush=True)
        print(f"  Min converged : {min_converged_fraction*100:.0f}%", flush=True)
        print(f"  Checkpoint    : every {checkpoint_interval} samples", flush=True)
        print(f"  Output        : {output_path}", flush=True)
        print(f"  Parameter ranges:", flush=True)
        print(f"    k0_1  : [{k0_1_min:.4e}, {k0_1_max:.4e}]", flush=True)
        print(f"    k0_2  : [{k0_2_min:.4e}, {k0_2_max:.4e}]", flush=True)
        print(f"    alpha1: [{a1_min:.4f}, {a1_max:.4f}]", flush=True)
        print(f"    alpha2: [{a2_min:.4f}, {a2_max:.4f}]", flush=True)
        print(f"{'='*78}\n", flush=True)

    t_total_start = time.time()
    n_failed = 0
    n_ok = int(all_converged[:start_idx].sum())  # count resumed valid samples
    sample_times: list[float] = []  # rolling window for ETA

    for i in range(start_idx, N):
        t_sample_start = time.time()
        k0 = parameter_samples[i, :2]
        alpha = parameter_samples[i, 2:]

        if verbose:
            # Pre-solve line: identify the sample being worked on
            pct = (i / N) * 100.0
            print(f"[{i+1:>{len(str(N))}}/{N}] ({pct:5.1f}%)  "
                  f"k0=[{k0[0]:.3e},{k0[1]:.3e}]  "
                  f"alpha=[{alpha[0]:.3f},{alpha[1]:.3f}]  ... ",
                  end="", flush=True)

        try:
            result = generate_training_data_single(
                k0_values=k0.tolist(),
                alpha_values=alpha.tolist(),
                phi_applied_values=phi_applied_values,
                base_solver_params=base_solver_params,
                steady=steady,
                observable_scale=observable_scale,
                mesh=mesh,
                max_eta_gap=max_eta_gap,
                fail_penalty=fail_penalty,
            )

            n_conv = result["n_converged"]
            frac = n_conv / n_eta
            elapsed = time.time() - t_sample_start
            all_timings[i] = elapsed
            sample_times.append(elapsed)

            if frac >= min_converged_fraction:
                # Interpolate any failed points
                cd = _interpolate_failed_points(
                    result["current_density"],
                    result["converged_mask"],
                    phi_applied_values,
                )
                pc = _interpolate_failed_points(
                    result["peroxide_current"],
                    result["converged_mask"],
                    phi_applied_values,
                )
                all_cd[i] = cd
                all_pc[i] = pc
                all_converged[i] = True
                n_ok += 1
                status = "OK"
            else:
                n_failed += 1
                status = f"SKIP({frac*100:.0f}%<{min_converged_fraction*100:.0f}%)"

        except Exception as e:
            elapsed = time.time() - t_sample_start
            all_timings[i] = elapsed
            sample_times.append(elapsed)
            n_failed += 1
            n_conv = 0
            frac = 0.0
            status = f"FAIL({e})"

        # Post-solve: status + timing + ETA
        if verbose:
            wall_elapsed = time.time() - t_total_start
            remaining = N - (i + 1)
            # ETA: use rolling average of last 20 samples (or all if < 20)
            window = sample_times[-20:]
            avg_per_sample = sum(window) / len(window) if window else 60.0
            eta_seconds = avg_per_sample * remaining

            print(f"{status}  conv={n_conv}/{n_eta}  "
                  f"dt={elapsed:.1f}s  "
                  f"wall={_fmt_duration(wall_elapsed)}  "
                  f"ETA={_fmt_duration(eta_seconds)}  "
                  f"[valid={n_ok}, fail={n_failed}]",
                  flush=True)

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0 or i == N - 1:
            _save_checkpoint(
                output_path, parameter_samples, all_cd, all_pc,
                all_converged, all_timings, phi_applied_values, i + 1,
            )
            if verbose:
                wall_elapsed = time.time() - t_total_start
                print(f"\n=== CHECKPOINT at sample {i+1}/{N}, "
                      f"elapsed: {_fmt_duration(wall_elapsed)}, "
                      f"valid: {n_ok}, failed: {n_failed} ===\n",
                      flush=True)

    # ------------------------------------------------------------------
    # Completion summary
    # ------------------------------------------------------------------
    valid_mask = all_converged
    n_valid = int(valid_mask.sum())
    total_time = time.time() - t_total_start

    if verbose:
        print(f"\n{'#'*78}", flush=True)
        print(f"  TRAINING DATA GENERATION -- COMPLETE", flush=True)
        print(f"{'#'*78}", flush=True)
        print(f"  Total samples  : {N}", flush=True)
        print(f"  Valid (saved)  : {n_valid}  ({n_valid/N*100:.1f}%)", flush=True)
        print(f"  Failed/skipped : {n_failed}  ({n_failed/N*100:.1f}%)", flush=True)
        print(f"  Total time     : {_fmt_duration(total_time)}", flush=True)
        if N > 0:
            print(f"  Avg per sample : {total_time/N:.1f}s", flush=True)
        if sample_times:
            print(f"  Min/Max sample : {min(sample_times):.1f}s / {max(sample_times):.1f}s", flush=True)
        print(f"  Output file    : {output_path}", flush=True)
        print(f"{'#'*78}\n", flush=True)

    # Save final with valid-only arrays
    np.savez_compressed(
        output_path,
        parameters=parameter_samples[valid_mask],
        current_density=all_cd[valid_mask],
        peroxide_current=all_pc[valid_mask],
        phi_applied=phi_applied_values,
        # Also store full arrays for analysis
        all_parameters=parameter_samples,
        all_current_density=all_cd,
        all_peroxide_current=all_pc,
        all_converged=all_converged,
        all_timings=all_timings,
        n_completed=N,
    )

    return {
        "parameters": parameter_samples[valid_mask],
        "current_density": all_cd[valid_mask],
        "peroxide_current": all_pc[valid_mask],
        "phi_applied": phi_applied_values,
        "n_valid": n_valid,
        "n_total": N,
        "n_failed": n_failed,
        "timings": all_timings,
    }


def _save_checkpoint(
    output_path: str,
    parameters: np.ndarray,
    cd: np.ndarray,
    pc: np.ndarray,
    converged: np.ndarray,
    timings: np.ndarray,
    phi_applied: np.ndarray,
    n_completed: int,
) -> None:
    """Save a checkpoint .npz file atomically."""
    ckpt_path = output_path + ".checkpoint.npz"
    tmp_path = ckpt_path + ".tmp"
    np.savez_compressed(
        tmp_path,
        parameters=parameters,
        current_density=cd,
        peroxide_current=pc,
        converged=converged,
        timings=timings,
        phi_applied=phi_applied,
        n_completed=n_completed,
    )
    os.replace(tmp_path, ckpt_path)  # atomic on POSIX


# ======================================================================
# Parallel training data generation
# ======================================================================

def _order_samples_nearest_neighbor(
    samples: np.ndarray,
    log_space_k0: bool = True,
) -> np.ndarray:
    """Reorder samples by nearest-neighbor greedy chain in parameter space.

    Parameters
    ----------
    samples : np.ndarray of shape (N, 4)
        Columns: [k0_1, k0_2, alpha_1, alpha_2].
    log_space_k0 : bool
        Transform k0 columns to log10 before computing distances.

    Returns
    -------
    np.ndarray
        Permutation indices giving the nearest-neighbor order.
    """
    N = samples.shape[0]
    coords = samples.copy().astype(float)
    if log_space_k0:
        coords[:, 0] = np.log10(np.maximum(coords[:, 0], 1e-20))
        coords[:, 1] = np.log10(np.maximum(coords[:, 1], 1e-20))

    # Normalize each column to [0, 1] for uniform distance weighting
    for col in range(4):
        cmin, cmax = coords[:, col].min(), coords[:, col].max()
        if cmax > cmin:
            coords[:, col] = (coords[:, col] - cmin) / (cmax - cmin)

    # Greedy nearest-neighbor starting from the sample closest to center
    center = coords.mean(axis=0)
    dists_to_center = np.linalg.norm(coords - center, axis=1)
    start = int(np.argmin(dists_to_center))

    visited = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=int)
    order[0] = start
    visited[start] = True

    for i in range(1, N):
        current = order[i - 1]
        dists = np.linalg.norm(coords - coords[current], axis=1)
        dists[visited] = np.inf
        order[i] = int(np.argmin(dists))
        visited[order[i]] = True

    return order


# Module-level worker state (populated by _training_worker_init in each
# spawned subprocess).
_TRAIN_WORKER_STATE: Dict[str, Any] = {}


def _training_worker_init(
    base_solver_params_data: Any,
    steady_data: Any,
    obs_scale: float,
    Nx: int,
    Ny: int,
    beta: float,
) -> None:
    """Initialize a training worker process.

    Sets environment variables, builds the mesh, and stores shared state
    in a module-level dict so ``_training_worker_solve_group`` can access it.
    """
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    _os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    _os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    global _TRAIN_WORKER_STATE
    from Forward.bv_solver import make_graded_rectangle_mesh
    _TRAIN_WORKER_STATE = {
        "base_solver_params": base_solver_params_data,
        "steady": steady_data,
        "observable_scale": obs_scale,
        "mesh": make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta),
    }


def _training_worker_solve_group(
    group_tasks: List[Tuple[int, float, float, float, float, Any]],
    worker_id: int = 0,
) -> List[Dict[str, Any]]:
    """Solve a group of nearby samples sequentially with cross-sample warm-starts.

    Parameters
    ----------
    group_tasks : list of (sample_index, k0_1, k0_2, alpha_1, alpha_2, phi_applied)
    worker_id : int
        Worker identifier for log messages.

    Returns
    -------
    list of result dicts
    """
    import time as _time

    results = []
    prev_solutions: Optional[Dict[int, tuple]] = None

    for task_i, task in enumerate(group_tasks):
        idx, k0_1, k0_2, alpha_1, alpha_2, phi_applied = task
        st = _TRAIN_WORKER_STATE
        t0 = _time.time()

        try:
            result = generate_training_data_single(
                k0_values=[k0_1, k0_2],
                alpha_values=[alpha_1, alpha_2],
                phi_applied_values=np.array(phi_applied),
                base_solver_params=st["base_solver_params"],
                steady=st["steady"],
                observable_scale=st["observable_scale"],
                mesh=st["mesh"],
                initial_solutions=prev_solutions,
                return_solutions=True,
            )
            elapsed = _time.time() - t0
            n_conv = result["n_converged"]
            n_eta = len(phi_applied)

            # Update warm-start chain
            if result.get("converged_solutions") is not None:
                prev_solutions = result["converged_solutions"]

            print(
                f"  [W{worker_id}]  #{task_i+1}/{len(group_tasks)}  "
                f"idx={idx}  k0=[{k0_1:.3e},{k0_2:.3e}]  "
                f"a=[{alpha_1:.2f},{alpha_2:.2f}]  "
                f"conv={n_conv}/{n_eta}  dt={elapsed:.1f}s",
                flush=True,
            )

            results.append({
                "index": idx,
                "current_density": result["current_density"].tolist(),
                "peroxide_current": result["peroxide_current"].tolist(),
                "converged_mask": result["converged_mask"].tolist(),
                "n_converged": n_conv,
                "elapsed": elapsed,
            })

        except Exception as e:
            elapsed = _time.time() - t0
            print(
                f"  [W{worker_id}]  #{task_i+1}/{len(group_tasks)}  "
                f"idx={idx}  FAIL({e})  dt={elapsed:.1f}s",
                flush=True,
            )
            results.append({
                "index": idx,
                "current_density": None,
                "peroxide_current": None,
                "converged_mask": None,
                "n_converged": 0,
                "elapsed": elapsed,
                "error": str(e),
            })
            # Reset warm-start chain on failure
            prev_solutions = None

    return results


def generate_training_dataset_parallel(
    parameter_samples: np.ndarray,
    *,
    phi_applied_values: np.ndarray,
    base_solver_params: Sequence[object],
    steady: Any,
    observable_scale: float,
    mesh_params: Tuple[int, int, float],
    output_path: str,
    n_workers: int = 8,
    min_converged_fraction: float = 0.8,
    verbose: bool = True,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate training data using parallel grouped workers with warm-starts.

    Divides the samples into ``n_workers`` groups using nearest-neighbor
    ordering, then processes each group sequentially within a worker
    (carrying warm-start state) while groups run in parallel.

    Parameters
    ----------
    parameter_samples : np.ndarray of shape (N, 4)
        Columns: [k0_1, k0_2, alpha_1, alpha_2].
    phi_applied_values : np.ndarray
        Voltage grid.
    base_solver_params : sequence
        11-element SolverParams list (picklable).
    steady : SteadyStateConfig
        Convergence config (picklable).
    observable_scale : float
        Scale factor for observable.
    mesh_params : tuple (Nx, Ny, beta)
        Mesh parameters for ``make_graded_rectangle_mesh``.
    output_path : str
        Path for final ``.npz`` output file.
    n_workers : int
        Number of parallel worker processes.
    min_converged_fraction : float
        Minimum converged fraction for a sample to be included.
    verbose : bool
        Print progress.
    resume_from : str or None
        Path to checkpoint ``.npz`` to resume from.
    """
    N = parameter_samples.shape[0]
    n_eta = len(phi_applied_values)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Pre-allocate storage
    all_cd = np.full((N, n_eta), np.nan, dtype=float)
    all_pc = np.full((N, n_eta), np.nan, dtype=float)
    all_converged = np.zeros(N, dtype=bool)
    all_timings = np.zeros(N, dtype=float)

    # Track which groups are already done (for resume)
    completed_indices: set = set()

    # Resume from checkpoint
    if resume_from is not None and os.path.exists(resume_from):
        if verbose:
            print(f"RESUME: Loading checkpoint: {resume_from}", flush=True)
        ckpt = np.load(resume_from, allow_pickle=True)
        # Restore all completed samples
        ckpt_converged = ckpt["converged"].astype(bool)
        ckpt_cd = ckpt["current_density"]
        ckpt_pc = ckpt["peroxide_current"]
        for i in range(N):
            if ckpt_converged[i] or (not np.isnan(ckpt_cd[i]).all()):
                all_cd[i] = ckpt_cd[i]
                all_pc[i] = ckpt_pc[i]
                all_converged[i] = ckpt_converged[i]
                completed_indices.add(i)
        if "timings" in ckpt:
            all_timings[:] = ckpt["timings"]
        if verbose:
            n_valid_so_far = int(all_converged.sum())
            print(f"RESUME: Restored {len(completed_indices)}/{N} completed, "
                  f"{n_valid_so_far} valid", flush=True)

    # Step 1: Order samples by parameter-space proximity
    if verbose:
        print("Ordering samples by nearest-neighbor chain...", flush=True)
    nn_order = _order_samples_nearest_neighbor(parameter_samples)

    # Step 2: Divide into worker groups (contiguous segments of NN chain)
    group_size = (N + n_workers - 1) // n_workers
    groups: List[List[Tuple[int, float, float, float, float, Any]]] = []
    for w in range(n_workers):
        start = w * group_size
        end = min(start + group_size, N)
        if start >= N:
            break
        group_indices = nn_order[start:end]
        # Skip entirely completed groups
        if all(int(i) in completed_indices for i in group_indices):
            if verbose:
                print(f"  Group {w}: all {end-start} samples already completed, skipping",
                      flush=True)
            continue
        group_tasks = [
            (int(i), *parameter_samples[i].tolist(), phi_applied_values.tolist())
            for i in group_indices
        ]
        groups.append(group_tasks)

    if not groups:
        if verbose:
            print("All groups already completed!", flush=True)
    else:
        if verbose:
            print(f"\n{'='*78}", flush=True)
            print(f"  PARALLEL TRAINING DATA GENERATION", flush=True)
            print(f"  Total samples  : {N}", flush=True)
            print(f"  Already done   : {len(completed_indices)}", flush=True)
            print(f"  Groups to run  : {len(groups)}", flush=True)
            print(f"  Workers        : {n_workers}", flush=True)
            print(f"  Voltage points : {n_eta}", flush=True)
            print(f"  Min converged  : {min_converged_fraction*100:.0f}%", flush=True)
            print(f"  Output         : {output_path}", flush=True)
            print(f"{'='*78}\n", flush=True)

        t_total_start = time.time()
        n_valid = int(all_converged.sum())
        n_failed = 0
        n_completed = len(completed_indices)

        # Step 3: Submit groups to worker pool
        ctx = mp.get_context("spawn")
        Nx, Ny, beta = mesh_params
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_training_worker_init,
            initargs=(base_solver_params, steady, observable_scale, Nx, Ny, beta),
        ) as executor:
            future_to_gidx = {}
            for g_idx, group in enumerate(groups):
                future = executor.submit(
                    _training_worker_solve_group, group, g_idx,
                )
                future_to_gidx[future] = g_idx

            pending = set(future_to_gidx.keys())
            while pending:
                done, pending = wait(pending, timeout=300,
                                     return_when=FIRST_COMPLETED)

                if not done:
                    wall = time.time() - t_total_start
                    print(
                        f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')}  "
                        f"waiting for workers...  "
                        f"{n_completed}/{N} complete  "
                        f"wall={_fmt_duration(wall)}",
                        flush=True,
                    )
                    continue

                for future in done:
                    g_idx = future_to_gidx[future]
                    try:
                        group_results = future.result()
                    except Exception as e:
                        print(f"[ERROR] Worker group {g_idx} failed: {e}",
                              flush=True)
                        continue

                    for r in group_results:
                        idx = r["index"]
                        n_completed += 1
                        elapsed = r.get("elapsed", 0.0)
                        all_timings[idx] = elapsed

                        if r["current_density"] is None:
                            n_failed += 1
                            continue

                        cd = np.array(r["current_density"])
                        pc = np.array(r["peroxide_current"])
                        conv_mask = np.array(r["converged_mask"])
                        frac = sum(conv_mask) / n_eta

                        if frac >= min_converged_fraction:
                            all_cd[idx] = _interpolate_failed_points(
                                cd, conv_mask, phi_applied_values)
                            all_pc[idx] = _interpolate_failed_points(
                                pc, conv_mask, phi_applied_values)
                            all_converged[idx] = True
                            n_valid += 1
                        else:
                            n_failed += 1

                    # Progress output
                    if verbose:
                        wall = time.time() - t_total_start
                        remaining = N - n_completed
                        avg = wall / max(n_completed - len(completed_indices), 1)
                        eta = avg * remaining
                        pct = n_completed / N * 100
                        print(
                            f"\n[GROUP {g_idx+1}/{len(groups)}] completed  "
                            f"{n_completed}/{N} ({pct:.1f}%)  "
                            f"valid={n_valid} fail={n_failed}  "
                            f"wall={_fmt_duration(wall)}  "
                            f"ETA={_fmt_duration(eta)}  "
                            f"avg={avg:.1f}s/sample",
                            flush=True,
                        )

                    # Checkpoint after each group
                    _save_checkpoint(
                        output_path, parameter_samples, all_cd, all_pc,
                        all_converged, all_timings, phi_applied_values,
                        n_completed,
                    )
                    if verbose:
                        wall = time.time() - t_total_start
                        print(
                            f"=== CHECKPOINT at {n_completed}/{N}, "
                            f"elapsed: {_fmt_duration(wall)}, "
                            f"valid: {n_valid}, failed: {n_failed} ===\n",
                            flush=True,
                        )

    # Save final output
    valid_mask = all_converged
    n_valid_final = int(valid_mask.sum())

    if verbose:
        print(f"\n{'#'*78}", flush=True)
        print(f"  PARALLEL DATA GENERATION -- COMPLETE", flush=True)
        print(f"  Total samples  : {N}", flush=True)
        print(f"  Valid (saved)  : {n_valid_final}  ({n_valid_final/N*100:.1f}%)",
              flush=True)
        print(f"  Output file    : {output_path}", flush=True)
        print(f"{'#'*78}\n", flush=True)

    np.savez_compressed(
        output_path,
        parameters=parameter_samples[valid_mask],
        current_density=all_cd[valid_mask],
        peroxide_current=all_pc[valid_mask],
        phi_applied=phi_applied_values,
        all_parameters=parameter_samples,
        all_current_density=all_cd,
        all_peroxide_current=all_pc,
        all_converged=all_converged,
        all_timings=all_timings,
        n_completed=N,
    )

    return {
        "parameters": parameter_samples[valid_mask],
        "current_density": all_cd[valid_mask],
        "peroxide_current": all_pc[valid_mask],
        "phi_applied": phi_applied_values,
        "n_valid": n_valid_final,
        "n_total": N,
        "n_failed": N - n_valid_final,
        "timings": all_timings,
    }
