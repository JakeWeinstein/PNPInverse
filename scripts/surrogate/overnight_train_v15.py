#!/usr/bin/env python
"""Overnight surrogate training pipeline v15 — adaptive z-ramping.

Uses a 3-tier solve strategy per sample:
  Tier 1: Standard warm-started solve (fast, works ~80%+ of the time)
  Tier 2: Charge continuation with charge_steps=5 on failed points
  Tier 3: Full charge continuation with charge_steps=15 on remaining failures

Runs indefinitely in batches of 500 samples, checkpointing frequently.

Usage:
    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python \
        scripts/surrogate/overnight_train_v15.py 2>&1 | tee StudyResults/surrogate_v15/run.log

Smoke test:
    ... overnight_train_v15.py --max-batches 1 --batch-size 10

Resume: Re-run the same command. Checkpoints are loaded automatically.
"""

from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
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

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v15")

MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0
N_WORKERS = 8
DT = 0.5
MAX_SS_STEPS = 100
T_END = DT * MAX_SS_STEPS  # 50.0

BATCH_SIZE = 500
N_WIDE_PER_BATCH = 350
N_FOCUSED_PER_BATCH = 150

MIN_CONVERGED_FRACTION = 0.8
GROUP_SIZE = 20  # samples per worker group (controls checkpoint frequency)


def _build_voltage_grid() -> np.ndarray:
    """Build expanded voltage grid with denser positive coverage."""
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0, -0.5,
        -1.0, -2.0, -3.0, -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])
    eta_shallow = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0, -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])
    # New positive-voltage fill-in points
    eta_positive_fill = np.array([+4.0, +2.0, +0.5, 0.0])

    all_eta = np.unique(np.concatenate([
        eta_symmetric, eta_shallow, eta_cathodic, eta_positive_fill,
    ]))
    return np.sort(all_eta)[::-1]  # descending


def _build_base_solver_params():
    """Build base SolverParams for training data generation."""
    return make_bv_solver_params(
        eta_hat=0.0,
        dt=DT,
        t_end=T_END,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
    )


def _build_steady_config():
    """Build SteadyStateConfig for training."""
    from Forward.steady_state import SteadyStateConfig
    return SteadyStateConfig(
        relative_tolerance=1e-4,
        absolute_tolerance=1e-8,
        consecutive_steps=4,
        max_steps=MAX_SS_STEPS,
        flux_observable="total_species",
        verbose=False,
    )


def _save_checkpoint_v15(
    output_path: str,
    parameters: np.ndarray,
    cd: np.ndarray,
    pc: np.ndarray,
    converged: np.ndarray,
    timings: np.ndarray,
    phi_applied: np.ndarray,
    n_completed: int,
) -> None:
    """Save a checkpoint .npz file atomically.

    Works around numpy's auto-append of .npz by ensuring paths end with .npz.
    """
    ckpt_path = output_path + ".checkpoint.npz"
    tmp_path = ckpt_path + ".tmp.npz"  # ends in .npz so numpy won't double-add
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


def _fmt_duration(seconds: float) -> str:
    """Format duration as Xh Ym Zs."""
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


# ======================================================================
# Adaptive z-ramp worker infrastructure
# ======================================================================

_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(
    base_solver_params_data: Any,
    steady_data: Any,
    obs_scale: float,
    Nx: int,
    Ny: int,
    beta: float,
) -> None:
    """Initialize a worker process."""
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    _os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    _os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    global _WORKER_STATE
    from Forward.bv_solver import make_graded_rectangle_mesh
    _WORKER_STATE = {
        "base_solver_params": base_solver_params_data,
        "steady": steady_data,
        "observable_scale": obs_scale,
        "mesh": make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta),
    }


def _solve_single_sample_adaptive(
    k0_1: float,
    k0_2: float,
    alpha_1: float,
    alpha_2: float,
    phi_applied: np.ndarray,
    mesh: Any,
    base_solver_params: Any,
    steady: Any,
    observable_scale: float,
    initial_solutions: Optional[Dict[int, tuple]] = None,
    return_solutions: bool = False,
) -> Dict[str, Any]:
    """Solve one I-V curve with adaptive 3-tier z-ramping.

    Tier 1: Standard warm-started solve via generate_training_data_single.
    Tier 2: For failed points, retry with charge continuation (charge_steps=5).
    Tier 3: For still-failed points, retry with charge_steps=15.

    Returns dict with same structure as generate_training_data_single plus
    'tier_stats' with counts of points resolved at each tier.
    """
    from Surrogate.training import generate_training_data_single
    from Forward.bv_solver.solvers import solve_bv_with_charge_continuation

    n_eta = len(phi_applied)

    # ---- Tier 1: Standard warm-started solve ----
    result = generate_training_data_single(
        k0_values=[k0_1, k0_2],
        alpha_values=[alpha_1, alpha_2],
        phi_applied_values=phi_applied,
        base_solver_params=base_solver_params,
        steady=steady,
        observable_scale=observable_scale,
        mesh=mesh,
        initial_solutions=initial_solutions,
        return_solutions=return_solutions,
    )

    cd = result["current_density"].copy()
    pc = result["peroxide_current"].copy()
    converged = result["converged_mask"].copy()

    tier1_ok = int(converged.sum())
    tier2_ok = 0
    tier3_ok = 0

    failed_indices = np.where(~converged)[0]

    if len(failed_indices) == 0:
        # All converged at Tier 1
        result["tier_stats"] = {
            "tier1": tier1_ok, "tier2": 0, "tier3": 0,
            "still_failed": 0,
        }
        return result

    # ---- Tier 2: Charge continuation (charge_steps=5) on failed points ----
    tier2_remaining = []
    for fi in failed_indices:
        eta_target = float(phi_applied[fi])
        try:
            # Build solver params with this sample's kinetics
            sp = _make_sample_solver_params(
                base_solver_params, k0_1, k0_2, alpha_1, alpha_2, eta_target,
            )
            ctx = solve_bv_with_charge_continuation(
                sp,
                eta_target=eta_target,
                eta_steps=20,
                charge_steps=5,
                print_interval=999,  # suppress per-step output
                mesh=mesh,
                return_ctx=True,
            )
            # Only accept if z-ramp reached full charge coupling
            if ctx.get("achieved_z_factor", 0.0) < 0.999:
                tier2_remaining.append(fi)
                continue
            cd_val, pc_val = _extract_observables_from_ctx(ctx, observable_scale)
            if not (np.isfinite(cd_val) and np.isfinite(pc_val)):
                tier2_remaining.append(fi)
                continue
            cd[fi] = cd_val
            pc[fi] = pc_val
            converged[fi] = True
            tier2_ok += 1
        except Exception:
            tier2_remaining.append(fi)

    if not tier2_remaining:
        result["current_density"] = cd
        result["peroxide_current"] = pc
        result["converged_mask"] = converged
        result["n_converged"] = int(converged.sum())
        result["tier_stats"] = {
            "tier1": tier1_ok, "tier2": tier2_ok, "tier3": 0,
            "still_failed": 0,
        }
        return result

    # ---- Tier 3: Full charge continuation (charge_steps=15) ----
    # Use relaxed SNES options for the hardest cases
    for fi in tier2_remaining:
        eta_target = float(phi_applied[fi])
        try:
            sp = _make_sample_solver_params(
                base_solver_params, k0_1, k0_2, alpha_1, alpha_2, eta_target,
                relaxed=True,
            )
            ctx = solve_bv_with_charge_continuation(
                sp,
                eta_target=eta_target,
                eta_steps=30,
                charge_steps=15,
                print_interval=999,
                mesh=mesh,
                return_ctx=True,
                min_delta_z=0.002,
            )
            if ctx.get("achieved_z_factor", 0.0) < 0.999:
                continue  # partial z-ramp — not physically valid
            cd_val, pc_val = _extract_observables_from_ctx(ctx, observable_scale)
            if not (np.isfinite(cd_val) and np.isfinite(pc_val)):
                continue
            cd[fi] = cd_val
            pc[fi] = pc_val
            converged[fi] = True
            tier3_ok += 1
        except Exception:
            pass  # leave as non-converged

    result["current_density"] = cd
    result["peroxide_current"] = pc
    result["converged_mask"] = converged
    result["n_converged"] = int(converged.sum())
    result["tier_stats"] = {
        "tier1": tier1_ok,
        "tier2": tier2_ok,
        "tier3": tier3_ok,
        "still_failed": int((~converged).sum()),
    }
    return result


def _make_sample_solver_params(
    base_solver_params: Any,
    k0_1: float,
    k0_2: float,
    alpha_1: float,
    alpha_2: float,
    eta_hat: float,
    relaxed: bool = False,
) -> list:
    """Build SolverParams for a specific sample with optional relaxed tolerances."""
    snes_opts = dict(SNES_OPTS_CHARGED)
    if relaxed:
        snes_opts["snes_max_it"] = 500
        snes_opts["snes_atol"] = 1e-6
        snes_opts["snes_rtol"] = 1e-6

    return make_bv_solver_params(
        eta_hat=eta_hat,
        dt=DT,
        t_end=T_END,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=snes_opts,
        k0_hat_r1=k0_1,
        k0_hat_r2=k0_2,
        alpha_r1=alpha_1,
        alpha_r2=alpha_2,
    )


def _extract_observables_from_ctx(
    ctx: Dict[str, Any],
    observable_scale: float,
) -> Tuple[float, float]:
    """Extract current_density and peroxide_current from a solved context.

    Uses the BV rate expressions assembled on the electrode boundary.
    The context from charge continuation already has ``bv_rate_exprs``
    and ``bv_settings`` populated by ``build_forms``.
    """
    from Forward.steady_state.bv import compute_bv_reaction_rates

    rates = compute_bv_reaction_rates(ctx)
    cd_val = sum(rates) * observable_scale
    # peroxide_current = (R_0 - R_1) * scale  (production - consumption)
    pc_val = (rates[0] - rates[1]) * observable_scale if len(rates) >= 2 else 0.0

    return float(cd_val), float(pc_val)


def _worker_solve_group_adaptive(
    group_tasks: List[Tuple[int, float, float, float, float, Any]],
    worker_id: int = 0,
) -> List[Dict[str, Any]]:
    """Solve a group of nearby samples with adaptive z-ramping and warm-starts."""
    import time as _time

    results = []
    prev_solutions: Optional[Dict[int, tuple]] = None

    for task_i, task in enumerate(group_tasks):
        idx, k0_1, k0_2, alpha_1, alpha_2, phi_applied_list = task
        phi_applied = np.array(phi_applied_list)
        st = _WORKER_STATE
        t0 = _time.time()

        try:
            result = _solve_single_sample_adaptive(
                k0_1, k0_2, alpha_1, alpha_2,
                phi_applied=phi_applied,
                mesh=st["mesh"],
                base_solver_params=st["base_solver_params"],
                steady=st["steady"],
                observable_scale=st["observable_scale"],
                initial_solutions=prev_solutions,
                return_solutions=True,
            )
            elapsed = _time.time() - t0
            n_conv = result["n_converged"]
            n_eta = len(phi_applied)
            tier = result.get("tier_stats", {})

            # Update warm-start chain
            if result.get("converged_solutions") is not None:
                prev_solutions = result["converged_solutions"]

            tier_str = (
                f"T1={tier.get('tier1', '?')} "
                f"T2={tier.get('tier2', '?')} "
                f"T3={tier.get('tier3', '?')} "
                f"fail={tier.get('still_failed', '?')}"
            )
            print(
                f"  [W{worker_id}]  #{task_i+1}/{len(group_tasks)}  "
                f"idx={idx}  k0=[{k0_1:.3e},{k0_2:.3e}]  "
                f"a=[{alpha_1:.2f},{alpha_2:.2f}]  "
                f"conv={n_conv}/{n_eta}  {tier_str}  dt={elapsed:.1f}s",
                flush=True,
            )

            results.append({
                "index": idx,
                "current_density": result["current_density"].tolist(),
                "peroxide_current": result["peroxide_current"].tolist(),
                "converged_mask": result["converged_mask"].tolist(),
                "n_converged": n_conv,
                "elapsed": elapsed,
                "tier_stats": tier,
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
            prev_solutions = None

    return results


# ======================================================================
# Batch generation
# ======================================================================

def _generate_batch(
    batch_idx: int,
    phi_applied: np.ndarray,
    base_solver_params: Any,
    steady: Any,
) -> Dict[str, Any]:
    """Generate one batch of training samples with adaptive z-ramping."""
    from Surrogate.sampling import (
        ParameterBounds,
        generate_multi_region_lhs_samples,
    )
    from Surrogate.training import (
        _order_samples_nearest_neighbor,
        _interpolate_failed_points,
    )

    batch_dir = os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}")
    os.makedirs(batch_dir, exist_ok=True)
    output_path = os.path.join(batch_dir, "batch_data.npz")
    checkpoint_path = output_path + ".checkpoint.npz"

    # Check if batch already complete
    if os.path.exists(output_path):
        print(f"\n[Batch {batch_idx}] Already complete at {output_path}, skipping.",
              flush=True)
        data = np.load(output_path)
        return {
            "parameters": data["parameters"],
            "current_density": data["current_density"],
            "peroxide_current": data["peroxide_current"],
            "n_valid": len(data["parameters"]),
            "n_total": int(data.get("n_total", len(data["parameters"]))),
        }

    # ---- Sampling ----
    wide_bounds = ParameterBounds(
        k0_1_range=(1e-6, 1.0),
        k0_2_range=(1e-7, 0.1),
        alpha_1_range=(0.1, 0.9),
        alpha_2_range=(0.1, 0.9),
    )
    focused_bounds = ParameterBounds(
        k0_1_range=(1e-4, 1e-1),
        k0_2_range=(1e-5, 1e-2),
        alpha_1_range=(0.2, 0.7),
        alpha_2_range=(0.2, 0.7),
    )

    # Increment seeds per batch to avoid duplicates
    seed_wide = 1000 + batch_idx * 2
    seed_focused = 1001 + batch_idx * 2

    samples = generate_multi_region_lhs_samples(
        wide_bounds=wide_bounds,
        focused_bounds=focused_bounds,
        n_base=N_WIDE_PER_BATCH,
        n_focused=N_FOCUSED_PER_BATCH,
        seed_base=seed_wide,
        seed_focused=seed_focused,
        log_space_k0=True,
    )

    N = samples.shape[0]
    n_eta = len(phi_applied)

    print(f"\n{'='*78}", flush=True)
    print(f"  BATCH {batch_idx}: {N} samples, {n_eta} voltage points", flush=True)
    print(f"  Seeds: wide={seed_wide}, focused={seed_focused}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    print(f"{'='*78}\n", flush=True)

    # ---- Pre-allocate storage ----
    all_cd = np.full((N, n_eta), np.nan, dtype=float)
    all_pc = np.full((N, n_eta), np.nan, dtype=float)
    all_converged = np.zeros(N, dtype=bool)
    all_timings = np.zeros(N, dtype=float)
    completed_indices: set = set()

    # ---- Resume from checkpoint ----
    if os.path.exists(checkpoint_path):
        print(f"RESUME: Loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = np.load(checkpoint_path, allow_pickle=True)
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
        n_valid_so_far = int(all_converged.sum())
        print(f"RESUME: Restored {len(completed_indices)}/{N} completed, "
              f"{n_valid_so_far} valid", flush=True)

    # ---- Order by nearest-neighbor ----
    nn_order = _order_samples_nearest_neighbor(samples)

    # ---- Divide into small worker groups for frequent checkpointing ----
    n_groups = (N + GROUP_SIZE - 1) // GROUP_SIZE
    groups: List[List[Tuple[int, float, float, float, float, Any]]] = []
    for g in range(n_groups):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, N)
        group_indices = nn_order[start:end]
        if all(int(i) in completed_indices for i in group_indices):
            continue
        group_tasks = [
            (int(i), *samples[i].tolist(), phi_applied.tolist())
            for i in group_indices
        ]
        groups.append(group_tasks)

    if not groups:
        print("All groups already completed!", flush=True)
    else:
        t_batch_start = time.time()
        n_valid = int(all_converged.sum())
        n_failed = 0
        n_completed = len(completed_indices)

        # Aggregate tier stats across batch
        batch_tier_stats = {"tier1": 0, "tier2": 0, "tier3": 0, "still_failed": 0}

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                base_solver_params, steady, -I_SCALE,
                MESH_NX, MESH_NY, MESH_BETA,
            ),
        ) as executor:
            future_to_gidx = {}
            for g_idx, group in enumerate(groups):
                future = executor.submit(
                    _worker_solve_group_adaptive, group, g_idx,
                )
                future_to_gidx[future] = g_idx

            pending = set(future_to_gidx.keys())
            while pending:
                done, pending = wait(
                    pending, timeout=300, return_when=FIRST_COMPLETED,
                )

                if not done:
                    wall = time.time() - t_batch_start
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

                        # Accumulate tier stats
                        ts = r.get("tier_stats", {})
                        for k in batch_tier_stats:
                            batch_tier_stats[k] += ts.get(k, 0)

                        if r["current_density"] is None:
                            n_failed += 1
                            continue

                        cd = np.array(r["current_density"])
                        pc = np.array(r["peroxide_current"])
                        conv_mask = np.array(r["converged_mask"])
                        frac = sum(conv_mask) / n_eta

                        if frac >= MIN_CONVERGED_FRACTION:
                            cd_interp = _interpolate_failed_points(
                                cd, conv_mask, phi_applied)
                            pc_interp = _interpolate_failed_points(
                                pc, conv_mask, phi_applied)
                            if np.all(np.isfinite(cd_interp)) and np.all(np.isfinite(pc_interp)):
                                all_cd[idx] = cd_interp
                                all_pc[idx] = pc_interp
                                all_converged[idx] = True
                                n_valid += 1
                            else:
                                n_failed += 1
                        else:
                            n_failed += 1

                    # Progress
                    wall = time.time() - t_batch_start
                    remaining = N - n_completed
                    avg = wall / max(n_completed - len(completed_indices), 1)
                    eta = avg * remaining
                    pct = n_completed / N * 100
                    print(
                        f"\n[Batch {batch_idx} | GROUP {g_idx+1}/{len(groups)}]  "
                        f"{n_completed}/{N} ({pct:.1f}%)  "
                        f"valid={n_valid} fail={n_failed}  "
                        f"wall={_fmt_duration(wall)}  "
                        f"ETA={_fmt_duration(eta)}",
                        flush=True,
                    )

                    # Checkpoint after each group
                    _save_checkpoint_v15(
                        output_path, samples, all_cd, all_pc,
                        all_converged, all_timings, phi_applied, n_completed,
                    )

        # Print tier breakdown
        print(f"\n  Tier breakdown for batch {batch_idx}:", flush=True)
        print(f"    Tier 1 (warm-start):    {batch_tier_stats['tier1']} points",
              flush=True)
        print(f"    Tier 2 (z-ramp x5):     {batch_tier_stats['tier2']} points",
              flush=True)
        print(f"    Tier 3 (z-ramp x15):    {batch_tier_stats['tier3']} points",
              flush=True)
        print(f"    Still failed:           {batch_tier_stats['still_failed']} points",
              flush=True)

    # ---- Save final batch data ----
    valid_mask = all_converged
    n_valid_final = int(valid_mask.sum())

    tmp_final_path = output_path[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp_final_path,
        parameters=samples[valid_mask],
        current_density=all_cd[valid_mask],
        peroxide_current=all_pc[valid_mask],
        phi_applied=phi_applied,
        all_parameters=samples,
        all_current_density=all_cd,
        all_peroxide_current=all_pc,
        all_converged=all_converged,
        all_timings=all_timings,
        n_completed=N,
        n_total=N,
    )
    os.replace(tmp_final_path, output_path)

    # Remove checkpoint now that final is saved
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"\n  Batch {batch_idx}: {n_valid_final}/{N} valid samples saved to {output_path}",
          flush=True)

    return {
        "parameters": samples[valid_mask],
        "current_density": all_cd[valid_mask],
        "peroxide_current": all_pc[valid_mask],
        "n_valid": n_valid_final,
        "n_total": N,
    }


def _merge_all_batches(phi_applied: np.ndarray) -> None:
    """Merge all completed batch files into a single running file."""
    merged_path = os.path.join(OUTPUT_DIR, "training_data_running.npz")

    all_params = []
    all_cd = []
    all_pc = []

    batch_idx = 0
    while True:
        batch_path = os.path.join(
            OUTPUT_DIR, f"batch_{batch_idx:04d}", "batch_data.npz",
        )
        if not os.path.exists(batch_path):
            break
        data = np.load(batch_path)
        if len(data["parameters"]) > 0:
            all_params.append(data["parameters"])
            all_cd.append(data["current_density"])
            all_pc.append(data["peroxide_current"])
        batch_idx += 1

    if not all_params:
        print("No batch data to merge.", flush=True)
        return

    merged_params = np.concatenate(all_params, axis=0)
    merged_cd = np.concatenate(all_cd, axis=0)
    merged_pc = np.concatenate(all_pc, axis=0)

    # Atomic write — tmp_path must end in .npz to avoid numpy auto-appending
    tmp_path = merged_path[:-4] + ".tmp.npz"
    np.savez_compressed(
        tmp_path,
        parameters=merged_params,
        current_density=merged_cd,
        peroxide_current=merged_pc,
        phi_applied=phi_applied,
        n_batches=batch_idx,
    )
    os.replace(tmp_path, merged_path)

    print(f"\nMerged {batch_idx} batches -> {len(merged_params)} total valid samples "
          f"at {merged_path}", flush=True)


# ======================================================================
# Main: infinite batch loop
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overnight surrogate training v15 with adaptive z-ramping",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="Max batches to run (0 = unlimited)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Samples per batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--start-batch", type=int, default=None,
        help="Starting batch index (auto-detected if not set)",
    )
    args = parser.parse_args()

    global N_WIDE_PER_BATCH, N_FOCUSED_PER_BATCH
    if args.batch_size != BATCH_SIZE:
        # Maintain 70/30 split
        N_WIDE_PER_BATCH = int(args.batch_size * 0.7)
        N_FOCUSED_PER_BATCH = args.batch_size - N_WIDE_PER_BATCH

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    phi_applied = _build_voltage_grid()
    base_sp = _build_base_solver_params()
    steady = _build_steady_config()

    print(f"{'#'*78}", flush=True)
    print(f"  SURROGATE TRAINING v15 — ADAPTIVE Z-RAMPING", flush=True)
    print(f"{'#'*78}", flush=True)
    print(f"  Output dir     : {OUTPUT_DIR}", flush=True)
    print(f"  Voltage points : {len(phi_applied)}", flush=True)
    print(f"  Voltage range  : [{phi_applied.min():.1f}, {phi_applied.max():.1f}]",
          flush=True)
    print(f"  Batch size     : {N_WIDE_PER_BATCH + N_FOCUSED_PER_BATCH} "
          f"({N_WIDE_PER_BATCH} wide + {N_FOCUSED_PER_BATCH} focused)",
          flush=True)
    print(f"  Workers        : {N_WORKERS}", flush=True)
    print(f"  Max batches    : {'unlimited' if args.max_batches == 0 else args.max_batches}",
          flush=True)
    print(f"  Mesh           : {MESH_NX}x{MESH_NY}, beta={MESH_BETA}", flush=True)
    print(f"{'#'*78}\n", flush=True)

    # Auto-detect starting batch index
    if args.start_batch is not None:
        batch_idx = args.start_batch
    else:
        batch_idx = 0
        while os.path.exists(
            os.path.join(OUTPUT_DIR, f"batch_{batch_idx:04d}", "batch_data.npz")
        ):
            batch_idx += 1
        if batch_idx > 0:
            print(f"Auto-detected: {batch_idx} batches already complete, "
                  f"starting at batch {batch_idx}", flush=True)

    t_grand = time.time()
    batches_completed = 0

    while True:
        if args.max_batches > 0 and batches_completed >= args.max_batches:
            print(f"\nReached max batches ({args.max_batches}), stopping.", flush=True)
            break

        t_batch = time.time()
        print(f"\n{'='*78}", flush=True)
        print(f"  STARTING BATCH {batch_idx} "
              f"(#{batches_completed+1}"
              f"{f'/{args.max_batches}' if args.max_batches > 0 else ''})"
              f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              flush=True)
        print(f"{'='*78}", flush=True)

        _generate_batch(batch_idx, phi_applied, base_sp, steady)

        batch_elapsed = time.time() - t_batch
        total_elapsed = time.time() - t_grand

        # Merge all batches into running file
        _merge_all_batches(phi_applied)

        print(f"\n  Batch {batch_idx} elapsed: {_fmt_duration(batch_elapsed)}", flush=True)
        print(f"  Total elapsed: {_fmt_duration(total_elapsed)}", flush=True)

        batch_idx += 1
        batches_completed += 1

    total_elapsed = time.time() - t_grand
    print(f"\n{'#'*78}", flush=True)
    print(f"  DONE: {batches_completed} batches in {_fmt_duration(total_elapsed)}",
          flush=True)
    print(f"{'#'*78}", flush=True)


if __name__ == "__main__":
    main()
