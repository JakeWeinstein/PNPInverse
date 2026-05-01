#!/usr/bin/env python
"""Re-solve 173 parameter sets that converged but lost curve data when killed.

Uses parallel workers with warm-starting. Each worker writes completed results
to a shared numpy file every 5 samples so progress is never lost.
"""
import os, sys, time, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, I_SCALE, FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED, make_bv_solver_params,
)
setup_firedrake_env()

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime

SNES_OPTS_AGGRESSIVE = {
    **SNES_OPTS_CHARGED,
    "snes_max_it": 500,
    "snes_atol": 1e-6,
    "snes_rtol": 1e-8,
    "snes_linesearch_maxlambda": 0.8,
    "snes_divergence_tolerance": 1e14,
}

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v12")

# Module-level worker state
_WORKER_STATE = {}


def _worker_init(base_solver_params, steady, obs_scale, Nx, Ny, beta):
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    _os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    _os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    global _WORKER_STATE
    from Forward.bv_solver import make_graded_rectangle_mesh
    _WORKER_STATE = {
        "base_solver_params": base_solver_params,
        "steady": steady,
        "observable_scale": obs_scale,
        "mesh": make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta),
    }


def _worker_solve_group(group_tasks, worker_id=0):
    """Solve a group with warm-starts, writing results to disk every 5 samples."""
    import time as _time
    from Surrogate.training import generate_training_data_single, _interpolate_failed_points

    st = _WORKER_STATE
    results = []
    prev_solutions = None
    log_path = os.path.join(OUTPUT_DIR, f"worker_{worker_id}_results.npz")

    for task_i, task in enumerate(group_tasks):
        idx, k0_1, k0_2, alpha_1, alpha_2, phi_applied_list = task
        phi_applied = np.array(phi_applied_list)
        n_eta = len(phi_applied)
        t0 = _time.time()

        try:
            result = generate_training_data_single(
                k0_values=[k0_1, k0_2],
                alpha_values=[alpha_1, alpha_2],
                phi_applied_values=phi_applied,
                base_solver_params=st["base_solver_params"],
                steady=st["steady"],
                observable_scale=st["observable_scale"],
                mesh=st["mesh"],
                initial_solutions=prev_solutions,
                return_solutions=True,
            )
            elapsed = _time.time() - t0
            n_conv = result["n_converged"]

            if result.get("converged_solutions"):
                prev_solutions = result["converged_solutions"]

            cd = _interpolate_failed_points(result["current_density"],
                                            result["converged_mask"], phi_applied)
            pc = _interpolate_failed_points(result["peroxide_current"],
                                            result["converged_mask"], phi_applied)

            print(f"  [W{worker_id}] #{task_i+1}/{len(group_tasks)}  "
                  f"idx={idx}  k0=[{k0_1:.3e},{k0_2:.3e}]  "
                  f"conv={n_conv}/{n_eta}  dt={elapsed:.1f}s", flush=True)

            results.append({
                "index": idx,
                "current_density": cd.tolist(),
                "peroxide_current": pc.tolist(),
                "n_converged": n_conv,
                "elapsed": elapsed,
            })

        except Exception as e:
            elapsed = _time.time() - t0
            print(f"  [W{worker_id}] #{task_i+1}/{len(group_tasks)}  "
                  f"idx={idx}  FAIL({e})  dt={elapsed:.1f}s", flush=True)
            prev_solutions = None
            results.append({
                "index": idx,
                "current_density": None,
                "peroxide_current": None,
                "n_converged": 0,
                "elapsed": elapsed,
            })

        # Save to disk every 5 samples
        if (task_i + 1) % 5 == 0 or task_i == len(group_tasks) - 1:
            _save_worker_log(log_path, results, phi_applied)

    return results


def _save_worker_log(path, results, phi_applied):
    """Save completed results to a per-worker npz file."""
    valid = [r for r in results if r["current_density"] is not None]
    if not valid:
        return
    indices = np.array([r["index"] for r in valid])
    cd = np.array([r["current_density"] for r in valid])
    pc = np.array([r["peroxide_current"] for r in valid])
    n_conv = np.array([r["n_converged"] for r in valid])
    np.savez_compressed(path,
                        indices=indices, current_density=cd,
                        peroxide_current=pc, n_converged=n_conv,
                        phi_applied=phi_applied)


def _fmt_duration(seconds):
    if seconds < 0:
        return "???"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


if __name__ == "__main__":
    from Surrogate.training import (
        _order_samples_nearest_neighbor,
        _interpolate_failed_points,
    )
    from Forward.steady_state import SteadyStateConfig

    ckpt = np.load("StudyResults/surrogate_v12/training_data_gapfill.npz.checkpoint.npz",
                    allow_pickle=True)
    all_params = ckpt["parameters"]
    phi_applied = ckpt["phi_applied"]
    missing_idx = np.load("StudyResults/surrogate_v12/missing_good_indices.npy")

    resolve_params = all_params[missing_idx]
    N = len(resolve_params)
    n_eta = len(phi_applied)
    N_WORKERS = 8

    print(f"Re-solving {N} parameter sets (previously converged)", flush=True)
    print(f"k0_2 range: [{resolve_params[:,1].min():.3e}, {resolve_params[:,1].max():.3e}]",
          flush=True)

    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.5, t_end=50.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_AGGRESSIVE,
    )
    steady = SteadyStateConfig(
        relative_tolerance=5e-4, absolute_tolerance=1e-7,
        consecutive_steps=3, max_steps=150,
        flux_observable="total_species", verbose=False,
    )

    # Clean up stale worker logs
    for w in range(N_WORKERS):
        p = os.path.join(OUTPUT_DIR, f"worker_{w}_results.npz")
        if os.path.exists(p):
            os.remove(p)

    # Order by nearest-neighbor for warm-start
    print("Ordering samples...", flush=True)
    nn_order = _order_samples_nearest_neighbor(resolve_params)

    # Build groups
    group_size = (N + N_WORKERS - 1) // N_WORKERS
    groups = []
    for w in range(N_WORKERS):
        start = w * group_size
        end = min(start + group_size, N)
        if start >= N:
            break
        group_indices = nn_order[start:end]
        group_tasks = [
            (int(missing_idx[i]), *resolve_params[i].tolist(), phi_applied.tolist())
            for i in group_indices
        ]
        groups.append(group_tasks)

    print(f"\n{'='*78}", flush=True)
    print(f"  RE-SOLVE: {N} samples, {len(groups)} groups, {N_WORKERS} workers", flush=True)
    print(f"  Each worker saves to disk every 5 samples", flush=True)
    print(f"{'='*78}\n", flush=True)

    t_start = time.time()
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=N_WORKERS, mp_context=ctx,
        initializer=_worker_init,
        initargs=(base_sp, steady, -I_SCALE, 8, 200, 3.0),
    ) as executor:
        future_to_gidx = {}
        for g_idx, group in enumerate(groups):
            future = executor.submit(_worker_solve_group, group, g_idx)
            future_to_gidx[future] = g_idx

        pending = set(future_to_gidx.keys())
        n_groups_done = 0

        while pending:
            done, pending = wait(pending, timeout=300, return_when=FIRST_COMPLETED)

            if not done:
                wall = time.time() - t_start
                # Count results from worker logs
                n_saved = 0
                for w in range(N_WORKERS):
                    p = os.path.join(OUTPUT_DIR, f"worker_{w}_results.npz")
                    if os.path.exists(p):
                        try:
                            n_saved += len(np.load(p)["indices"])
                        except Exception:
                            pass
                print(f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')}  "
                      f"~{n_saved} samples saved to disk  "
                      f"wall={_fmt_duration(wall)}", flush=True)
                continue

            for future in done:
                g_idx = future_to_gidx[future]
                n_groups_done += 1
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Group {g_idx} failed: {e}", flush=True)

                wall = time.time() - t_start
                print(f"\n[GROUP {n_groups_done}/{len(groups)} done]  "
                      f"wall={_fmt_duration(wall)}", flush=True)

    # Collect all worker results
    print(f"\nCollecting results from worker logs...", flush=True)
    all_indices = []
    all_cd = []
    all_pc = []

    for w in range(N_WORKERS):
        p = os.path.join(OUTPUT_DIR, f"worker_{w}_results.npz")
        if os.path.exists(p):
            wdata = np.load(p)
            all_indices.append(wdata["indices"])
            all_cd.append(wdata["current_density"])
            all_pc.append(wdata["peroxide_current"])
            print(f"  W{w}: {len(wdata['indices'])} samples", flush=True)

    if all_indices:
        all_indices = np.concatenate(all_indices)
        all_cd = np.concatenate(all_cd)
        all_pc = np.concatenate(all_pc)

        # Save final resolve output
        out_path = os.path.join(OUTPUT_DIR, "training_data_resolve.npz")
        np.savez_compressed(out_path,
                            indices=all_indices,
                            parameters=all_params[all_indices],
                            current_density=all_cd,
                            peroxide_current=all_pc,
                            phi_applied=phi_applied)

        wall = time.time() - t_start
        print(f"\n{'#'*78}", flush=True)
        print(f"  RE-SOLVE COMPLETE", flush=True)
        print(f"  Recovered: {len(all_indices)} samples", flush=True)
        print(f"  Wall time: {_fmt_duration(wall)}", flush=True)
        print(f"  Saved to:  {out_path}", flush=True)
        print(f"{'#'*78}", flush=True)
    else:
        print("No results recovered!", flush=True)
