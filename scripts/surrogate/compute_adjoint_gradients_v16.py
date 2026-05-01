#!/usr/bin/env python
"""Deferred adjoint gradient computation for v16 training data.

Loads converged forward solutions produced by ``overnight_train_v16.py`` and
computes adjoint gradients dO/d(k0_1, k0_2, alpha_1, alpha_2) for both
observables (current_density and peroxide_current) at every voltage point.

Architecture:
    - Follows the canonical adjoint pattern from FluxCurve/bv_parallel.py:
      clear tape -> rebuild context -> load ICs -> 1 SNES step -> differentiate
    - Differentiates the RAW observable (ReducedFunctional(sim, controls)),
      NOT a squared residual. This gives dO/dtheta.
    - Parallel workers via ProcessPoolExecutor (spawn context)
    - Incremental checkpointing of gradient arrays

Usage:
    cd PNPInverse
    python scripts/surrogate/compute_adjoint_gradients_v16.py --batch-idx 0

    # Process all batches:
    python scripts/surrogate/compute_adjoint_gradients_v16.py --all

    # Merge all batch gradients:
    python scripts/surrogate/compute_adjoint_gradients_v16.py --merge
"""

from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix libomp conflict between Firedrake/PETSc and PyTorch on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
)
setup_firedrake_env()

import numpy as np


# ======================================================================
# Constants & configuration
# ======================================================================

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v16")

MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0
N_WORKERS = 8
DT = 0.5
MAX_SS_STEPS = 100
T_END = DT * MAX_SS_STEPS  # 50.0

# SNES options for the 1-step adjoint tape pass.
# Jacobian lag settings match the canonical pattern in bv_parallel.py.
ADJOINT_SNES_OPTS: Dict[str, Any] = {
    **SNES_OPTS_CHARGED,
    "snes_lag_jacobian": 2,
    "snes_lag_jacobian_persists": True,
}

# Number of controls: k0_1, k0_2, alpha_1, alpha_2
N_CONTROLS = 4

# Checkpoint frequency: save after this many (sample, eta) pairs complete
CHECKPOINT_INTERVAL = 50


# ======================================================================
# Graceful shutdown
# ======================================================================

_SHUTDOWN_REQUESTED = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Set shutdown flag on SIGINT/SIGTERM."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    sig_name = signal.Signals(signum).name
    print(f"\n[signal] {sig_name} received — will finish current work and exit.")


# ======================================================================
# Worker state (module-level, set by initializer)
# ======================================================================

_WORKER_MESH: Any = None


def _worker_init(mesh_nx: int, mesh_ny: int, mesh_beta: float) -> None:
    """Initialize one worker process: build mesh, set env vars."""
    global _WORKER_MESH

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    # Ignore SIGINT in workers — let the main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from Forward.bv_solver import make_graded_rectangle_mesh
    _WORKER_MESH = make_graded_rectangle_mesh(
        Nx=mesh_nx, Ny=mesh_ny, beta=mesh_beta,
    )


# ======================================================================
# Core adjoint computation: single (sample, eta) point
# ======================================================================

def _compute_gradients_single_point(
    solver_params_list: list,
    u_data_arrays: List[np.ndarray],
    k0_1: float,
    k0_2: float,
    alpha_1: float,
    alpha_2: float,
    observable_mode: str,
) -> Dict[str, Any]:
    """Compute dO/dtheta for one observable at one voltage point.

    Follows the canonical adjoint pattern from FluxCurve/bv_parallel.py:162-262.

    Parameters
    ----------
    solver_params_list:
        SolverParams (as list) with phi_applied already baked in.
    u_data_arrays:
        List of numpy arrays, one per mixed-space component (the converged
        forward solution at this voltage point).
    k0_1, k0_2, alpha_1, alpha_2:
        Parameter values for this sample.
    observable_mode:
        ``"current_density"`` or ``"peroxide_current"``.

    Returns
    -------
    dict with keys:
        ``"success"`` (bool), ``"gradient"`` (ndarray of shape (4,)),
        ``"sim_value"`` (float), ``"reason"`` (str if failed).
    """
    global _WORKER_MESH

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from FluxCurve.bv_observables import _bv_gradient_controls_to_array

    # ------------------------------------------------------------------
    # Step 1: Clear tape and enable annotation
    # ------------------------------------------------------------------
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    # ------------------------------------------------------------------
    # Step 2: Rebuild context + forms from scratch (fresh tape)
    # ------------------------------------------------------------------
    from Forward.params import SolverParams
    params = SolverParams.from_list(solver_params_list)

    ctx = bv_build_context(params, mesh=_WORKER_MESH)
    ctx = bv_build_forms(ctx, params)
    bv_set_initial_conditions(ctx, params, blob=False)

    # ------------------------------------------------------------------
    # Step 3: Load converged U_data as initial guess
    # ------------------------------------------------------------------
    for src_arr, dst in zip(u_data_arrays, ctx["U"].dat):
        dst.data[:] = src_arr
    ctx["U_prev"].assign(ctx["U"])

    # ------------------------------------------------------------------
    # Step 4: Assign k0/alpha as tape-tracked control values
    # ------------------------------------------------------------------
    k0_funcs = list(ctx["bv_k0_funcs"])
    k0_funcs[0].assign(float(k0_1))
    k0_funcs[1].assign(float(k0_2))
    alpha_funcs = list(ctx["bv_alpha_funcs"])
    alpha_funcs[0].assign(float(alpha_1))
    alpha_funcs[1].assign(float(alpha_2))

    # ------------------------------------------------------------------
    # Step 5: Build solver + 1 SNES step ON THE TAPE
    # ------------------------------------------------------------------
    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]

    jac = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=dict(ADJOINT_SNES_OPTS),
    )

    try:
        solver.solve()
    except Exception as exc:
        return {
            "success": False,
            "gradient": np.zeros(N_CONTROLS, dtype=float),
            "sim_value": float("nan"),
            "reason": f"tape-pass SNES failed: {type(exc).__name__}: {exc}",
        }
    U_prev.assign(U)

    # ------------------------------------------------------------------
    # Step 6: Assemble observable form
    # ------------------------------------------------------------------
    obs_form = _build_bv_observable_form(
        ctx,
        mode=observable_mode,
        reaction_index=None,
        scale=-I_SCALE,
    )
    sim_value = fd.assemble(obs_form)

    # ------------------------------------------------------------------
    # Step 7: Differentiate w.r.t. controls -> gradient
    # ------------------------------------------------------------------
    control_funcs = list(k0_funcs) + list(alpha_funcs)
    controls = [adj.Control(cf) for cf in control_funcs]

    # Differentiate the RAW observable, NOT a squared residual.
    # This gives dO/dtheta (the Jacobian row for this observable).
    rf = adj.ReducedFunctional(sim_value, controls)
    try:
        grad = _bv_gradient_controls_to_array(rf.derivative(), N_CONTROLS)
    except Exception as exc:
        return {
            "success": False,
            "gradient": np.zeros(N_CONTROLS, dtype=float),
            "sim_value": float(sim_value) if not isinstance(sim_value, float) else sim_value,
            "reason": f"adjoint derivative failed: {type(exc).__name__}: {exc}",
        }

    # Gradient sanity check: warn on non-finite or suspiciously large values
    control_names = ["k0_1", "k0_2", "alpha_1", "alpha_2"]
    phi_val_str = f"phi={solver_params_list[0] if solver_params_list else '?'}"
    for ci, (cname, gval) in enumerate(zip(control_names, grad)):
        if not np.isfinite(gval):
            warnings.warn(
                f"Non-finite gradient d({observable_mode})/d({cname})={gval} "
                f"at {phi_val_str}"
            )
        elif abs(gval) > 1e6:
            warnings.warn(
                f"Suspiciously large gradient d({observable_mode})/d({cname})"
                f"={gval:.2e} at {phi_val_str}"
            )

    return {
        "success": True,
        "gradient": grad,
        "sim_value": float(sim_value),
        "reason": "",
    }


# ======================================================================
# Worker function: process a batch of (sample, eta) pairs
# ======================================================================

def _worker_compute_gradients(task_args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: compute gradients for assigned (sample, eta) pairs.

    Parameters
    ----------
    task_args: dict with keys
        ``"pairs"`` — list of (sample_idx, eta_idx) tuples
        ``"parameters"`` — ndarray (N, 4) of all sample params
        ``"phi_applied"`` — ndarray (n_eta,) voltage grid
        ``"u_data_keys"`` — dict mapping (sample_idx, eta_idx) -> list of arrays
        ``"dt"`` — float
        ``"t_end"`` — float

    Returns
    -------
    dict with ``"results"`` — list of dicts, one per (sample, eta, observable) triplet.
    """
    pairs = task_args["pairs"]
    parameters = task_args["parameters"]
    phi_applied = task_args["phi_applied"]
    u_data_map = task_args["u_data_map"]
    dt = task_args["dt"]
    t_end = task_args["t_end"]

    results = []

    for sample_idx, eta_idx in pairs:
        k0_1, k0_2, alpha_1, alpha_2 = parameters[sample_idx]
        phi_val = float(phi_applied[eta_idx])

        # Retrieve the flattened U_data arrays for this (sample, eta) pair
        u_data_key = f"u_data_{sample_idx}_{eta_idx}"
        u_data_flat = u_data_map.get(u_data_key)
        if u_data_flat is None:
            results.append({
                "sample_idx": sample_idx,
                "eta_idx": eta_idx,
                "grad_cd": np.zeros(N_CONTROLS, dtype=float),
                "grad_pc": np.zeros(N_CONTROLS, dtype=float),
                "sim_cd": float("nan"),
                "sim_pc": float("nan"),
                "success_cd": False,
                "success_pc": False,
                "reason": f"missing u_data for key {u_data_key}",
            })
            continue

        # Reconstruct per-component arrays from the flat storage.
        # The forward script stores all components concatenated into one array.
        # We need to split it into per-component arrays matching ctx["U"].dat.
        # The number of components = n_species + 1 (phi).
        n_components = FOUR_SPECIES_CHARGED.n_species + 1  # 4 species + 1 potential = 5
        n_dofs_per_component = len(u_data_flat) // n_components
        u_data_arrays = [
            u_data_flat[i * n_dofs_per_component : (i + 1) * n_dofs_per_component]
            for i in range(n_components)
        ]

        # Build solver_params with phi_applied baked in for this voltage point.
        # CRITICAL: phi_applied must be set via make_bv_solver_params BEFORE
        # build_forms() is called, so the BV boundary condition uses the correct
        # overpotential.
        solver_params = make_bv_solver_params(
            eta_hat=phi_val,
            dt=dt,
            t_end=t_end,
            species=FOUR_SPECIES_CHARGED,
            snes_opts=dict(ADJOINT_SNES_OPTS),
            k0_hat_r1=float(k0_1),
            k0_hat_r2=float(k0_2),
            alpha_r1=float(alpha_1),
            alpha_r2=float(alpha_2),
        )
        # Convert SolverParams to plain list for pickling-free rebuild inside
        # _compute_gradients_single_point.
        sp_list = list(solver_params)

        # --- Compute gradient for current_density ---
        res_cd = _compute_gradients_single_point(
            solver_params_list=sp_list,
            u_data_arrays=u_data_arrays,
            k0_1=k0_1,
            k0_2=k0_2,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            observable_mode="current_density",
        )

        # --- Compute gradient for peroxide_current ---
        # Must clear tape and rebuild (steps 1-7 again) for the second observable.
        res_pc = _compute_gradients_single_point(
            solver_params_list=sp_list,
            u_data_arrays=u_data_arrays,
            k0_1=k0_1,
            k0_2=k0_2,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            observable_mode="peroxide_current",
        )

        results.append({
            "sample_idx": sample_idx,
            "eta_idx": eta_idx,
            "grad_cd": res_cd["gradient"],
            "grad_pc": res_pc["gradient"],
            "sim_cd": res_cd["sim_value"],
            "sim_pc": res_pc["sim_value"],
            "success_cd": res_cd["success"],
            "success_pc": res_pc["success"],
            "reason": "; ".join(
                r for r in [res_cd.get("reason", ""), res_pc.get("reason", "")]
                if r
            ),
        })

    return {"results": results}


# ======================================================================
# Batch processing
# ======================================================================

def _load_forward_data(batch_dir: str) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]
]:
    """Load forward data and solutions from a batch directory.

    Returns
    -------
    parameters : ndarray (N, 4)
    converged : ndarray (N,) bool
    phi_applied : ndarray (n_eta,)
    u_data_map : dict mapping ``"u_data_{sample}_{eta}"`` -> flat ndarray
    """
    data_path = os.path.join(batch_dir, "training_data.npz")
    sol_path = os.path.join(batch_dir, "batch_solutions.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    if not os.path.exists(sol_path):
        raise FileNotFoundError(f"Solutions not found: {sol_path}")

    data = np.load(data_path, allow_pickle=False)
    parameters = data["parameters"]       # (N, 4)
    converged = data["converged"]          # (N,) bool
    phi_applied = data["phi_applied"]      # (n_eta,)

    sol = np.load(sol_path, allow_pickle=False)
    u_data_map = {k: sol[k] for k in sol.files}

    print(f"[load] parameters: {parameters.shape}, converged: {converged.sum()}/{len(converged)}")
    print(f"[load] phi_applied: {phi_applied.shape} ({phi_applied.min():.1f} to {phi_applied.max():.1f})")
    print(f"[load] solution keys: {len(u_data_map)}")

    # Re-validate loaded forward data (spot-check first 10 converged samples)
    from Forward.bv_solver.validation import validate_observables

    converged_indices = np.where(converged)[0]
    n_eta = len(phi_applied)
    # The training_data.npz may contain observable arrays; look for them.
    obs_cd_all = data["current_density"] if "current_density" in data.files else None
    obs_pc_all = data["peroxide_current"] if "peroxide_current" in data.files else None

    if obs_cd_all is not None and obs_pc_all is not None:
        for sample_idx in converged_indices[:10]:
            cd_vals = obs_cd_all[sample_idx]
            pc_vals = obs_pc_all[sample_idx]
            for j in range(min(len(cd_vals), n_eta)):
                vr = validate_observables(
                    cd_vals[j], pc_vals[j],
                    I_lim=2.0,  # nondimensional diffusion-limit estimate
                    phi_applied=float(phi_applied[j]),
                    V_T=1.0,
                )
                if not vr.valid:
                    warnings.warn(
                        f"Loaded sample {sample_idx} point {j} has physics "
                        f"violations: {vr.failures}"
                    )
                    break
    else:
        print("[load] obs_cd/obs_pc not in training_data.npz — skipping re-validation")

    return parameters, converged, phi_applied, u_data_map


def _checkpoint_gradients(
    output_path: str,
    grad_cd: np.ndarray,
    grad_pc: np.ndarray,
    valid_indices: np.ndarray,
    phi_applied: np.ndarray,
    completed_mask: np.ndarray,
) -> None:
    """Atomically save gradient checkpoint."""
    tmp_path = output_path + ".tmp"
    np.savez_compressed(
        tmp_path,
        grad_cd=grad_cd,
        grad_pc=grad_pc,
        valid_indices=valid_indices,
        phi_applied=phi_applied,
        completed_mask=completed_mask,
    )
    os.replace(tmp_path, output_path)


def process_batch(
    batch_dir: str,
    output_dir: Optional[str],
    n_workers: int,
) -> Optional[str]:
    """Load forward data, compute adjoint gradients in parallel, checkpoint.

    Parameters
    ----------
    batch_dir : str
        Directory containing ``training_data.npz`` and ``batch_solutions.npz``.
    output_dir : str or None
        Where to write ``batch_gradients.npz``. Defaults to ``batch_dir``.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    str or None
        Path to the output gradients file, or None if no work was done.
    """
    global _SHUTDOWN_REQUESTED

    if output_dir is None:
        output_dir = batch_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "batch_gradients.npz")

    # Load forward data
    parameters, converged, phi_applied, u_data_map = _load_forward_data(batch_dir)
    n_samples, n_params = parameters.shape
    n_eta = len(phi_applied)

    # Filter to converged samples only
    valid_indices = np.where(converged)[0]
    n_valid = len(valid_indices)
    if n_valid == 0:
        print("[warn] No converged samples found. Nothing to do.")
        return None

    print(f"[batch] Processing {n_valid} converged samples x {n_eta} voltage points "
          f"= {n_valid * n_eta} adjoint solves")

    # Initialize gradient arrays
    grad_cd = np.full((n_valid, n_eta, N_CONTROLS), np.nan, dtype=float)
    grad_pc = np.full((n_valid, n_eta, N_CONTROLS), np.nan, dtype=float)
    # Track which (valid_i, eta_j) pairs have been completed
    completed_mask = np.zeros((n_valid, n_eta), dtype=bool)

    # Resume from checkpoint if it exists
    if os.path.exists(output_path):
        print(f"[resume] Loading existing checkpoint: {output_path}")
        ckpt = np.load(output_path, allow_pickle=False)
        if (ckpt["grad_cd"].shape == grad_cd.shape
                and ckpt["grad_pc"].shape == grad_pc.shape):
            grad_cd = ckpt["grad_cd"]
            grad_pc = ckpt["grad_pc"]
            completed_mask = ckpt["completed_mask"]
            n_done = completed_mask.sum()
            print(f"[resume] Restored {n_done}/{n_valid * n_eta} completed points")
        else:
            print("[resume] Shape mismatch — starting fresh")

    # Build work items: all (valid_i, eta_j) pairs that are not yet done
    work_pairs: List[Tuple[int, int]] = []
    for vi, sample_idx in enumerate(valid_indices):
        for ej in range(n_eta):
            if not completed_mask[vi, ej]:
                # Check that solution data exists for this point
                u_key = f"u_data_{sample_idx}_{ej}"
                if u_key in u_data_map:
                    work_pairs.append((vi, ej))

    if not work_pairs:
        print("[batch] All points already completed.")
        return output_path

    print(f"[batch] {len(work_pairs)} points remaining")

    # Distribute work into chunks for workers.
    # Each chunk is processed by one worker call.
    chunk_size = max(1, len(work_pairs) // (n_workers * 4))
    chunks: List[List[Tuple[int, int]]] = []
    for i in range(0, len(work_pairs), chunk_size):
        chunks.append(work_pairs[i : i + chunk_size])

    print(f"[batch] Split into {len(chunks)} chunks (chunk_size ~{chunk_size})")

    # Prepare task arguments for each chunk.
    # To avoid sending the entire u_data_map to every worker (too large),
    # we send only the relevant u_data entries for each chunk.
    def _make_task(chunk: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Build task dict for a chunk of (valid_i, eta_j) pairs."""
        # Map valid_i back to sample_idx for parameter lookup
        pairs_for_worker = []
        chunk_u_data: Dict[str, np.ndarray] = {}
        for vi, ej in chunk:
            sample_idx = int(valid_indices[vi])
            pairs_for_worker.append((sample_idx, ej))
            u_key = f"u_data_{sample_idx}_{ej}"
            if u_key not in chunk_u_data:
                chunk_u_data[u_key] = u_data_map[u_key]

        return {
            "pairs": pairs_for_worker,
            "parameters": parameters,
            "phi_applied": phi_applied,
            "u_data_map": chunk_u_data,
            "dt": DT,
            "t_end": T_END,
        }

    # Build the reverse mapping: (sample_idx, eta_idx) -> (valid_i, eta_j)
    sample_eta_to_vi: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for vi, sample_idx in enumerate(valid_indices):
        for ej in range(n_eta):
            sample_eta_to_vi[(int(sample_idx), ej)] = (vi, ej)

    # Launch parallel workers
    t_start = time.time()
    n_completed_total = int(completed_mask.sum())
    n_failed = 0
    ctx_spawn = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx_spawn,
        initializer=_worker_init,
        initargs=(MESH_NX, MESH_NY, MESH_BETA),
    ) as executor:
        futures = {}
        for ci, chunk in enumerate(chunks):
            if _SHUTDOWN_REQUESTED:
                break
            task = _make_task(chunk)
            fut = executor.submit(_worker_compute_gradients, task)
            futures[fut] = (ci, chunk)

        for fut in as_completed(futures):
            if _SHUTDOWN_REQUESTED:
                print("[shutdown] Stopping submission of new work.")
                break

            ci, chunk = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"[error] Chunk {ci} failed: {type(exc).__name__}: {exc}")
                n_failed += len(chunk)
                continue

            # Store results
            for r in result["results"]:
                si, ej = r["sample_idx"], r["eta_idx"]
                vi_ej = sample_eta_to_vi.get((si, ej))
                if vi_ej is None:
                    continue
                vi, ej_mapped = vi_ej

                if r["success_cd"]:
                    grad_cd[vi, ej_mapped] = r["grad_cd"]
                else:
                    n_failed += 1
                if r["success_pc"]:
                    grad_pc[vi, ej_mapped] = r["grad_pc"]
                else:
                    n_failed += 1

                completed_mask[vi, ej_mapped] = r["success_cd"] and r["success_pc"]
                if completed_mask[vi, ej_mapped]:
                    n_completed_total += 1

            # Incremental checkpoint
            elapsed = time.time() - t_start
            total_work = n_valid * n_eta
            pct = 100.0 * n_completed_total / total_work if total_work > 0 else 0.0
            pace = elapsed / max(n_completed_total, 1)
            remaining = pace * (total_work - n_completed_total)

            print(
                f"[progress] {n_completed_total}/{total_work} ({pct:.1f}%) "
                f"| elapsed {elapsed:.0f}s | pace {pace:.1f}s/pt "
                f"| ETA {remaining:.0f}s | failed {n_failed}"
            )

            _checkpoint_gradients(
                output_path, grad_cd, grad_pc,
                valid_indices, phi_applied, completed_mask,
            )

    # Final checkpoint
    _checkpoint_gradients(
        output_path, grad_cd, grad_pc,
        valid_indices, phi_applied, completed_mask,
    )

    elapsed_total = time.time() - t_start
    print(f"\n[done] Completed {n_completed_total}/{n_valid * n_eta} points "
          f"in {elapsed_total:.1f}s ({n_failed} failures)")
    print(f"[done] Output: {output_path}")
    print(f"[done] grad_cd shape: {grad_cd.shape}, grad_pc shape: {grad_pc.shape}")

    return output_path


# ======================================================================
# Merge utility
# ======================================================================

def merge_batch_gradients(base_dir: str, output_path: Optional[str] = None) -> str:
    """Combine batch gradient files into a single running file.

    Parameters
    ----------
    base_dir : str
        Directory containing ``batch_NNN/`` subdirectories.
    output_path : str or None
        Where to write merged gradients. Defaults to
        ``base_dir/training_gradients_v16_running.npz``.

    Returns
    -------
    str
        Path to the merged output file.
    """
    if output_path is None:
        output_path = os.path.join(base_dir, "training_gradients_v16_running.npz")

    # Discover batch directories
    batch_dirs = sorted(
        d for d in Path(base_dir).iterdir()
        if d.is_dir() and d.name.startswith("batch_")
    )

    all_grad_cd = []
    all_grad_pc = []
    all_valid_indices = []
    all_phi_applied = None
    offset = 0

    for bd in batch_dirs:
        grad_path = bd / "batch_gradients.npz"
        if not grad_path.exists():
            print(f"[merge] Skipping {bd.name} — no batch_gradients.npz")
            continue

        data = np.load(str(grad_path), allow_pickle=False)
        gc = data["grad_cd"]
        gp = data["grad_pc"]
        vi = data["valid_indices"]

        all_grad_cd.append(gc)
        all_grad_pc.append(gp)
        all_valid_indices.append(vi + offset)

        if all_phi_applied is None:
            all_phi_applied = data["phi_applied"]

        # Load the training_data.npz to get total sample count for offset
        td_path = bd / "training_data.npz"
        if td_path.exists():
            td = np.load(str(td_path), allow_pickle=False)
            offset += len(td["parameters"])
        else:
            offset += len(vi)  # fallback

        print(f"[merge] {bd.name}: {gc.shape[0]} samples, "
              f"{gc.shape[1]} eta points")

    if not all_grad_cd:
        print("[merge] No gradient files found.")
        return output_path

    merged_cd = np.concatenate(all_grad_cd, axis=0)
    merged_pc = np.concatenate(all_grad_pc, axis=0)
    merged_vi = np.concatenate(all_valid_indices, axis=0)

    np.savez_compressed(
        output_path,
        grad_cd=merged_cd,
        grad_pc=merged_pc,
        valid_indices=merged_vi,
        phi_applied=all_phi_applied,
    )

    print(f"\n[merge] Merged {len(all_grad_cd)} batches -> {output_path}")
    print(f"[merge] grad_cd: {merged_cd.shape}, grad_pc: {merged_pc.shape}")
    print(f"[merge] Total valid samples: {merged_cd.shape[0]}")

    return output_path


# ======================================================================
# CLI
# ======================================================================

def _find_batch_dirs(base_dir: str) -> List[str]:
    """Auto-detect batch directories under base_dir."""
    dirs = sorted(
        str(d) for d in Path(base_dir).iterdir()
        if d.is_dir() and d.name.startswith("batch_")
        and (d / "training_data.npz").exists()
        and (d / "batch_solutions.npz").exists()
    )
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute adjoint gradients for v16 training data.",
    )
    parser.add_argument(
        "--batch-idx", type=int, default=None,
        help="Process a single batch by index (e.g., --batch-idx 0).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all detected batches.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all batch gradients into a single file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for gradients (default: same as batch dir).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=N_WORKERS,
        help=f"Number of parallel workers (default: {N_WORKERS}).",
    )
    parser.add_argument(
        "--base-dir", type=str, default=OUTPUT_DIR,
        help=f"Base directory for batch data (default: {OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    # Install signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[start] compute_adjoint_gradients_v16.py @ {now}")
    print(f"[config] base_dir={args.base_dir}  n_workers={args.n_workers}")

    if args.merge:
        merge_batch_gradients(args.base_dir, args.output_dir)
        return

    if args.batch_idx is not None:
        # Process a single batch
        batch_dir = os.path.join(args.base_dir, f"batch_{args.batch_idx:03d}")
        if not os.path.isdir(batch_dir):
            print(f"[error] Batch directory not found: {batch_dir}")
            sys.exit(1)
        process_batch(batch_dir, args.output_dir, args.n_workers)

    elif args.all:
        # Process all batches
        batch_dirs = _find_batch_dirs(args.base_dir)
        if not batch_dirs:
            print(f"[error] No batch directories found in {args.base_dir}")
            sys.exit(1)

        print(f"[all] Found {len(batch_dirs)} batches")
        for bi, bd in enumerate(batch_dirs):
            if _SHUTDOWN_REQUESTED:
                print("[shutdown] Stopping batch loop.")
                break
            print(f"\n{'='*60}")
            print(f"[all] Processing batch {bi+1}/{len(batch_dirs)}: {bd}")
            print(f"{'='*60}")
            out_dir = args.output_dir if args.output_dir else None
            process_batch(bd, out_dir, args.n_workers)

        # Auto-merge after processing all batches
        if not _SHUTDOWN_REQUESTED:
            print(f"\n{'='*60}")
            print("[all] Merging all batch gradients...")
            print(f"{'='*60}")
            merge_batch_gradients(args.base_dir)
    else:
        parser.print_help()
        print("\nError: specify --batch-idx N, --all, or --merge")
        sys.exit(1)


if __name__ == "__main__":
    main()
