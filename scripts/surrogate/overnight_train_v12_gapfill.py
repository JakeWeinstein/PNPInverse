#!/usr/bin/env python
"""Overnight surrogate training pipeline v12 -- Targeted gap-filling.

Purpose: Fill coverage gaps identified by Phase 1 training data audit.
         The 3,194-sample training set has a max-empty-ball radius of 0.58
         (threshold 0.15), located in the high-k0_1 / low-k0_2 / high-alpha_1
         / low-alpha_2 corner.  Convergence failures cluster heavily in the
         low-k0_2 decades (84 of 251 failures in [1e-7, 1e-6)).

Strategy:
  - Region 1 (500 samples): The largest gap corner
      k0_1 in [0.1, 1.0], k0_2 in [1e-7, 1e-5],
      alpha_1 in [0.6, 0.9], alpha_2 in [0.1, 0.4]
  - Region 2 (300 samples): Failure-depleted low-k0_2 decade (full other params)
      k0_1 in [1e-6, 1.0], k0_2 in [1e-7, 1e-6],
      alpha_1 in [0.1, 0.9], alpha_2 in [0.1, 0.9]
  - Region 3 (200 samples): Wide coverage supplement
      Full default ParameterBounds

Convergence strategy for difficult low-k0_2 samples:
  - Parameter continuation: start from a nearby "easy" point (k0_2 ~ 1e-3) and
    walk toward the target in log-space steps, propagating converged ICs forward.
  - Cross-sample warm-starting via nearest-neighbor ordering within groups.
  - Aggressive retry: 6 recovery attempts with tolerance relaxation, line-search
    cycling, and max-iteration escalation.
  - Per-sample retry with continuation fallback: if direct solve fails, attempt
    continuation from the nearest converged sample.

Usage:
    cd /path/to/PNPInverse
    /path/to/venv-firedrake/bin/python \\
        scripts/surrogate/overnight_train_v12_gapfill.py 2>&1 \\
        | tee StudyResults/surrogate_v12/run.log

Resume: Re-run the same command. Checkpoints are loaded automatically.

Estimated runtime: ~1.5-3h for 1000 samples with continuation retries.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time

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

# Aggressive SNES opts for difficult low-k0_2 samples:
# - 500 Newton iterations (vs 300 default for charged system)
# - Relaxed absolute tolerance 1e-6 (vs 1e-7)
# - Relaxed relative tolerance 1e-8 (vs 1e-10)
# - Backtracking line search with larger max lambda
SNES_OPTS_AGGRESSIVE = {
    **SNES_OPTS_CHARGED,
    "snes_max_it": 500,
    "snes_atol": 1e-6,
    "snes_rtol": 1e-8,
    "snes_linesearch_maxlambda": 0.8,
    "snes_divergence_tolerance": 1e14,
}

import numpy as np

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v12")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Phase flags ----
RUN_PHASE_1 = True   # Gap-filling data generation
RUN_PHASE_2 = True   # Merge with existing data

# ---- Shared constants (match v11) ----
MESH_NX = 8
MESH_NY = 200
MESH_BETA = 3.0
N_WORKERS = 8
DT = 0.5
MAX_SS_STEPS = 100
T_END = DT * MAX_SS_STEPS  # 50.0

# ---- Gap-fill sampling config ----
# Region 1: Largest gap corner (high k0_1, low k0_2, high a1, low a2)
REGION_1_N = 500
REGION_1_SEED = 401

# Region 2: Failure-depleted low k0_2 decade
REGION_2_N = 300
REGION_2_SEED = 402

# Region 3: Wide coverage supplement
REGION_3_N = 200
REGION_3_SEED = 403

# Continuation: parameters for homotopy when direct solve fails
CONTINUATION_ANCHOR_K02 = 1e-3  # "easy" k0_2 value that reliably converges
CONTINUATION_N_STEPS = 8        # log-space steps from anchor to target
CONTINUATION_MAX_RETRIES = 2    # max continuation attempts per failed sample


def _build_aggressive_recovery():
    """Build an aggressive ForwardRecoveryConfig for gap-fill samples.

    Compared to the default (4 attempts, no anisotropy, 1 tolerance relax):
    - 7 total attempts (vs 4)
    - 3 max-it-only attempts (vs 2) -- gives Newton more room before relaxing
    - 0 anisotropy attempts -- not relevant here (we're solving at fixed target
      parameters, not adjusting them like in inference)
    - 3 tolerance relaxation attempts (vs 1) -- allows relaxing to 1e-4 / 1e-5
    - Higher max-it cap: 800 (vs 500)
    - Full line-search cycling: bt -> l2 -> cp -> basic
    """
    from FluxCurve.config import ForwardRecoveryConfig
    return ForwardRecoveryConfig(
        max_attempts=7,
        max_it_only_attempts=3,
        anisotropy_only_attempts=0,
        tolerance_relax_attempts=3,
        max_it_growth=1.5,
        max_it_cap=800,
        atol_relax_factor=10.0,
        rtol_relax_factor=10.0,
        ksp_rtol_relax_factor=10.0,
        line_search_schedule=("bt", "l2", "cp", "basic"),
    )


def _build_voltage_grid() -> np.ndarray:
    """Build the union voltage grid from v7 phases (matches v11)."""
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
    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    return np.sort(all_eta)[::-1]  # descending


def _build_base_solver_params():
    """Build base SolverParams with aggressive SNES for gap-fill regions."""
    return make_bv_solver_params(
        eta_hat=0.0,
        dt=DT,
        t_end=T_END,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_AGGRESSIVE,
    )


def _build_steady_config():
    """Build SteadyStateConfig for gap-fill training.

    Slightly more lenient than v11 defaults:
    - relative_tolerance 5e-4 (vs 1e-4): allows convergence at harder points
    - max_steps 150 (vs 100): more time-steps to reach steady state
    - consecutive_steps 3 (vs 4): declare converged sooner
    """
    from Forward.steady_state import SteadyStateConfig
    return SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=3,
        max_steps=150,
        flux_observable="total_species",
        verbose=False,
    )


def _fmt_duration(seconds: float) -> str:
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


# ==================================================================
# Gap-fill sample generation
# ==================================================================

def _generate_gapfill_samples() -> np.ndarray:
    """Generate targeted LHS samples for the three gap-fill regions."""
    from Surrogate.sampling import ParameterBounds, generate_lhs_samples

    # Region 1: Largest gap corner
    region1_bounds = ParameterBounds(
        k0_1_range=(0.1, 1.0),
        k0_2_range=(1e-7, 1e-5),
        alpha_1_range=(0.6, 0.9),
        alpha_2_range=(0.1, 0.4),
    )
    samples_r1 = generate_lhs_samples(
        region1_bounds, REGION_1_N, seed=REGION_1_SEED, log_space_k0=True,
    )

    # Region 2: Failure-depleted low k0_2 decade
    region2_bounds = ParameterBounds(
        k0_1_range=(1e-6, 1.0),
        k0_2_range=(1e-7, 1e-6),
        alpha_1_range=(0.1, 0.9),
        alpha_2_range=(0.1, 0.9),
    )
    samples_r2 = generate_lhs_samples(
        region2_bounds, REGION_2_N, seed=REGION_2_SEED, log_space_k0=True,
    )

    # Region 3: Wide coverage supplement
    wide_bounds = ParameterBounds(
        k0_1_range=(1e-6, 1.0),
        k0_2_range=(1e-7, 0.1),
        alpha_1_range=(0.1, 0.9),
        alpha_2_range=(0.1, 0.9),
    )
    samples_r3 = generate_lhs_samples(
        wide_bounds, REGION_3_N, seed=REGION_3_SEED, log_space_k0=True,
    )

    all_samples = np.concatenate([samples_r1, samples_r2, samples_r3], axis=0)

    print(f"Gap-fill samples generated:", flush=True)
    print(f"  Region 1 (gap corner):     {REGION_1_N}", flush=True)
    print(f"  Region 2 (low k0_2):       {REGION_2_N}", flush=True)
    print(f"  Region 3 (wide):           {REGION_3_N}", flush=True)
    print(f"  Total:                     {len(all_samples)}", flush=True)
    print(f"  k0_1 range: [{all_samples[:,0].min():.3e}, {all_samples[:,0].max():.3e}]",
          flush=True)
    print(f"  k0_2 range: [{all_samples[:,1].min():.3e}, {all_samples[:,1].max():.3e}]",
          flush=True)

    return all_samples


# ==================================================================
# Continuation-based solver for difficult samples
# ==================================================================

def _solve_single(
    k0_1, k0_2, alpha_1, alpha_2, phi_applied,
    base_solver_params, steady, observable_scale, mesh,
    recovery, prev_solutions=None, max_eta_gap=3.0,
):
    """Thin wrapper around generate_training_data_single with all our knobs."""
    from Surrogate.training import generate_training_data_single
    return generate_training_data_single(
        k0_values=[k0_1, k0_2],
        alpha_values=[alpha_1, alpha_2],
        phi_applied_values=phi_applied,
        base_solver_params=base_solver_params,
        steady=steady,
        observable_scale=observable_scale,
        mesh=mesh,
        initial_solutions=prev_solutions,
        return_solutions=True,
        forward_recovery=recovery,
        max_eta_gap=max_eta_gap,
    )


def _solve_with_continuation(
    target_k0_2: float,
    target_alpha_1: float,
    target_alpha_2: float,
    k0_1: float,
    phi_applied: np.ndarray,
    base_solver_params,
    steady,
    observable_scale: float,
    mesh,
    recovery,
    anchor_k02: float = CONTINUATION_ANCHOR_K02,
    n_steps: int = CONTINUATION_N_STEPS,
    max_eta_gap: float = 3.0,
    walk_alpha: bool = False,
):
    """Solve a difficult sample via parameter continuation.

    Strategy: Start from anchor_k02 (where convergence is reliable) at
    alpha midpoints, then walk in log-space steps toward the target k0_2
    (and optionally alpha), propagating converged ICs forward at each step.

    The PDE solution varies smoothly in log(k0_2), so a converged solution
    at k0_2 = 10^a is an excellent IC for k0_2 = 10^(a - delta) when delta
    is small.

    Parameters
    ----------
    walk_alpha : bool
        If True, simultaneously walk alpha_1 and alpha_2 from 0.5 midpoints
        toward target values.  Useful when target alphas are extreme (near
        0.1 or 0.9) which can cause BV exponential overflow.
    """
    log_anchor = np.log10(anchor_k02)
    log_target = np.log10(target_k0_2)
    n_eta = len(phi_applied)

    # Build schedules
    log_k02_sched = np.linspace(log_anchor, log_target, n_steps + 1)
    k02_sched = 10.0 ** log_k02_sched

    if walk_alpha:
        a1_sched = np.linspace(0.5, target_alpha_1, n_steps + 1)
        a2_sched = np.linspace(0.5, target_alpha_2, n_steps + 1)
    else:
        a1_sched = np.full(n_steps + 1, target_alpha_1)
        a2_sched = np.full(n_steps + 1, target_alpha_2)

    prev_solutions = None
    last_good_result = None
    last_good_step = -1

    for step_i in range(n_steps + 1):
        try:
            result = _solve_single(
                k0_1, k02_sched[step_i], a1_sched[step_i], a2_sched[step_i],
                phi_applied, base_solver_params, steady, observable_scale,
                mesh, recovery, prev_solutions, max_eta_gap,
            )
            n_conv = result["n_converged"]

            if n_conv >= n_eta * 0.5:
                prev_solutions = result.get("converged_solutions")
                last_good_result = result
                last_good_step = step_i
            # If intermediate step has poor convergence, keep going with
            # whatever ICs we have -- don't give up on the chain.

        except Exception:
            pass  # continue chain with previous ICs

    # Accept if the FINAL step (at target params) converged well enough.
    # The caller enforces the 80% threshold, so return anything we got
    # at the target to avoid a redundant re-solve.
    if last_good_step == n_steps and last_good_result is not None:
        return last_good_result

    # Continuation chain didn't reach the target -- one final direct
    # attempt with the best accumulated ICs from the chain.
    try:
        result = _solve_single(
            k0_1, target_k0_2, target_alpha_1, target_alpha_2,
            phi_applied, base_solver_params, steady, observable_scale,
            mesh, recovery, prev_solutions, max_eta_gap,
        )
        return result
    except Exception:
        return None


def _build_lenient_steady_config():
    """Even more lenient steady-state config for last-resort attempts.

    More time-steps (200), relaxed tolerance (1e-3), fewer consecutive
    steps required (2).  This lets the solver grind through stiff transients.
    """
    from Forward.steady_state import SteadyStateConfig
    return SteadyStateConfig(
        relative_tolerance=1e-3,
        absolute_tolerance=1e-6,
        consecutive_steps=2,
        max_steps=200,
        flux_observable="total_species",
        verbose=False,
    )


# ==================================================================
# Worker with escalating retry ladder
# ==================================================================

# Module-level worker state
_TRAIN_WORKER_STATE = {}


def _gapfill_worker_init(
    base_solver_params_data,
    steady_data,
    obs_scale: float,
    Nx: int,
    Ny: int,
    beta: float,
):
    """Initialize a gap-fill worker process."""
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
        "steady_lenient": _build_lenient_steady_config(),
        "observable_scale": obs_scale,
        "mesh": make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta),
        "recovery": _build_aggressive_recovery(),
    }


def _gapfill_worker_solve_group(
    group_tasks,
    worker_id: int = 0,
):
    """Solve a group of samples with escalating retry strategies.

    Escalation ladder for each sample:
      Level 1: Direct solve, warm-started from previous sample, max_eta_gap=3.0
      Level 2: k0_2 continuation (8 steps from anchor 1e-3), max_eta_gap=3.0
      Level 3: k0_2 continuation with denser bridge points, max_eta_gap=1.5
      Level 4: Multi-param continuation (also walk alpha from 0.5), max_eta_gap=1.5
      Level 5: Multi-param continuation, lenient steady-state, max_eta_gap=1.5
      Level 6: Cold start with lenient steady-state, max_eta_gap=1.0
    """
    import time as _time

    st = _TRAIN_WORKER_STATE
    results = []
    prev_solutions = None
    recovery = st["recovery"]
    min_conv = 0.8  # 80% convergence threshold throughout

    for task_i, task in enumerate(group_tasks):
        idx, k0_1, k0_2, alpha_1, alpha_2, phi_applied_list = task
        phi_applied = np.array(phi_applied_list)
        t0 = _time.time()
        n_eta = len(phi_applied)
        min_pts = int(n_eta * min_conv)

        method = "L1-direct"
        final_result = None

        # ---- Level 1: Direct solve with warm-start ----
        try:
            result = _solve_single(
                k0_1, k0_2, alpha_1, alpha_2, phi_applied,
                st["base_solver_params"], st["steady"], st["observable_scale"],
                st["mesh"], recovery, prev_solutions, max_eta_gap=3.0,
            )
            if result["n_converged"] >= min_pts:
                final_result = result
                method = "L1-direct"
        except Exception:
            pass

        # ---- Level 2: k0_2 continuation, standard bridge points ----
        if final_result is None and k0_2 < 1e-3:
            try:
                result = _solve_with_continuation(
                    target_k0_2=k0_2, target_alpha_1=alpha_1,
                    target_alpha_2=alpha_2, k0_1=k0_1,
                    phi_applied=phi_applied,
                    base_solver_params=st["base_solver_params"],
                    steady=st["steady"],
                    observable_scale=st["observable_scale"],
                    mesh=st["mesh"], recovery=recovery,
                    max_eta_gap=3.0, walk_alpha=False,
                )
                if result is not None and result["n_converged"] >= min_pts:
                    final_result = result
                    method = "L2-k02-cont"
            except Exception:
                pass

        # ---- Level 3: k0_2 continuation, denser bridge points ----
        if final_result is None and k0_2 < 1e-3:
            try:
                result = _solve_with_continuation(
                    target_k0_2=k0_2, target_alpha_1=alpha_1,
                    target_alpha_2=alpha_2, k0_1=k0_1,
                    phi_applied=phi_applied,
                    base_solver_params=st["base_solver_params"],
                    steady=st["steady"],
                    observable_scale=st["observable_scale"],
                    mesh=st["mesh"], recovery=recovery,
                    max_eta_gap=1.5, walk_alpha=False,
                    n_steps=12,  # finer steps
                )
                if result is not None and result["n_converged"] >= min_pts:
                    final_result = result
                    method = "L3-dense-bridge"
            except Exception:
                pass

        # ---- Level 4: Multi-parameter continuation (walk alpha too) ----
        if final_result is None and k0_2 < 1e-3:
            # Extreme alphas (near 0.1 or 0.9) cause BV exponential overflow;
            # walking from 0.5 midpoint keeps the exponential tame during continuation.
            alpha_is_extreme = (alpha_1 < 0.2 or alpha_1 > 0.8 or
                                alpha_2 < 0.2 or alpha_2 > 0.8)
            if alpha_is_extreme:
                try:
                    result = _solve_with_continuation(
                        target_k0_2=k0_2, target_alpha_1=alpha_1,
                        target_alpha_2=alpha_2, k0_1=k0_1,
                        phi_applied=phi_applied,
                        base_solver_params=st["base_solver_params"],
                        steady=st["steady"],
                        observable_scale=st["observable_scale"],
                        mesh=st["mesh"], recovery=recovery,
                        max_eta_gap=1.5, walk_alpha=True,
                        n_steps=12,
                    )
                    if result is not None and result["n_converged"] >= min_pts:
                        final_result = result
                        method = "L4-multi-cont"
                except Exception:
                    pass

        # ---- Level 5: Multi-param continuation + lenient steady-state ----
        if final_result is None and k0_2 < 1e-3:
            # Use a closer anchor (1e-4) only when target is below it;
            # otherwise fall back to the standard 1e-3 anchor to ensure
            # we always walk downward from an easier point.
            l5_anchor = 1e-4 if k0_2 < 1e-4 else CONTINUATION_ANCHOR_K02
            try:
                result = _solve_with_continuation(
                    target_k0_2=k0_2, target_alpha_1=alpha_1,
                    target_alpha_2=alpha_2, k0_1=k0_1,
                    phi_applied=phi_applied,
                    base_solver_params=st["base_solver_params"],
                    steady=st["steady_lenient"],
                    observable_scale=st["observable_scale"],
                    mesh=st["mesh"], recovery=recovery,
                    max_eta_gap=1.5, walk_alpha=True,
                    n_steps=16,  # even finer steps
                    anchor_k02=l5_anchor,
                )
                if result is not None and result["n_converged"] >= min_pts:
                    final_result = result
                    method = "L5-lenient-cont"
            except Exception:
                pass

        # ---- Level 6: Cold start, lenient steady-state, densest bridges ----
        if final_result is None:
            try:
                result = _solve_single(
                    k0_1, k0_2, alpha_1, alpha_2, phi_applied,
                    st["base_solver_params"], st["steady_lenient"],
                    st["observable_scale"], st["mesh"], recovery,
                    prev_solutions=None, max_eta_gap=1.0,
                )
                if result["n_converged"] >= min_pts:
                    final_result = result
                    method = "L6-cold-lenient"
            except Exception:
                pass

        elapsed = _time.time() - t0

        # --- Record result ---
        if final_result is not None:
            n_conv = final_result["n_converged"]
            if final_result.get("converged_solutions"):
                prev_solutions = final_result["converged_solutions"]
            print(
                f"  [W{worker_id}]  #{task_i+1}/{len(group_tasks)}  "
                f"idx={idx}  k0=[{k0_1:.3e},{k0_2:.3e}]  "
                f"a=[{alpha_1:.2f},{alpha_2:.2f}]  "
                f"conv={n_conv}/{n_eta}  {method}  dt={elapsed:.1f}s",
                flush=True,
            )
            results.append({
                "index": idx,
                "current_density": final_result["current_density"].tolist(),
                "peroxide_current": final_result["peroxide_current"].tolist(),
                "converged_mask": final_result["converged_mask"].tolist(),
                "n_converged": n_conv,
                "elapsed": elapsed,
                "method": method,
            })
        else:
            print(
                f"  [W{worker_id}]  #{task_i+1}/{len(group_tasks)}  "
                f"idx={idx}  k0=[{k0_1:.3e},{k0_2:.3e}]  "
                f"a=[{alpha_1:.2f},{alpha_2:.2f}]  "
                f"FAIL-ALL  dt={elapsed:.1f}s",
                flush=True,
            )
            results.append({
                "index": idx,
                "current_density": None,
                "peroxide_current": None,
                "converged_mask": None,
                "n_converged": 0,
                "elapsed": elapsed,
                "method": "failed",
                "error": "all 6 levels exhausted",
            })
            # Reset warm-start chain on total failure
            prev_solutions = None

    return results


# ==================================================================
# Phase 1: Gap-filling data generation
# ==================================================================

def phase_1_gapfill_generation():
    """Generate ~1000 gap-filling training samples with continuation retry."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import datetime

    from Surrogate.training import (
        _order_samples_nearest_neighbor,
        _interpolate_failed_points,
        _save_checkpoint,
    )

    t0 = time.time()

    new_data_path = os.path.join(OUTPUT_DIR, "training_data_gapfill.npz")
    checkpoint_path = new_data_path + ".checkpoint.npz"

    # Check if already done
    if os.path.exists(new_data_path):
        tmp = np.load(new_data_path)
        if "parameters" in tmp and len(tmp["parameters"]) > 0:
            print(f"Phase 1: gap-fill data already exists ({len(tmp['parameters'])} samples)",
                  flush=True)
            return

    # Generate samples
    all_samples = _generate_gapfill_samples()
    N = len(all_samples)
    phi_applied = _build_voltage_grid()
    n_eta = len(phi_applied)

    base_sp = _build_base_solver_params()
    steady = _build_steady_config()

    # Pre-allocate
    all_cd = np.full((N, n_eta), np.nan, dtype=float)
    all_pc = np.full((N, n_eta), np.nan, dtype=float)
    all_converged = np.zeros(N, dtype=bool)
    all_timings = np.zeros(N, dtype=float)
    all_methods = np.array(["" for _ in range(N)], dtype=object)
    completed_indices = set()

    # Resume
    if os.path.exists(checkpoint_path):
        print(f"RESUME: Loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        ckpt_converged = ckpt["converged"].astype(bool)
        ckpt_cd = ckpt["current_density"]
        ckpt_pc = ckpt["peroxide_current"]
        for i in range(min(N, len(ckpt_converged))):
            if ckpt_converged[i] or (not np.isnan(ckpt_cd[i]).all()):
                all_cd[i] = ckpt_cd[i]
                all_pc[i] = ckpt_pc[i]
                all_converged[i] = ckpt_converged[i]
                completed_indices.add(i)
        if "timings" in ckpt:
            all_timings[:len(ckpt["timings"])] = ckpt["timings"][:N]
        n_valid = int(all_converged.sum())
        print(f"RESUME: Restored {len(completed_indices)}/{N}, {n_valid} valid", flush=True)

    # Order by nearest-neighbor for warm-start effectiveness
    print("Ordering samples by nearest-neighbor chain...", flush=True)
    nn_order = _order_samples_nearest_neighbor(all_samples)

    # Divide into worker groups
    min_converged_fraction = 0.8  # Same as v11 -- worker escalation handles difficult points
    group_size = (N + N_WORKERS - 1) // N_WORKERS
    groups = []
    for w in range(N_WORKERS):
        start = w * group_size
        end = min(start + group_size, N)
        if start >= N:
            break
        group_indices = nn_order[start:end]
        if all(int(i) in completed_indices for i in group_indices):
            print(f"  Group {w}: all {end-start} samples done, skipping", flush=True)
            continue
        group_tasks = [
            (int(i), *all_samples[i].tolist(), phi_applied.tolist())
            for i in group_indices
        ]
        groups.append(group_tasks)

    method_counts = {}

    if not groups:
        print("All groups already completed!", flush=True)
    else:
        print(f"\n{'='*78}", flush=True)
        print(f"  GAP-FILL PARALLEL DATA GENERATION", flush=True)
        print(f"  Total samples   : {N}", flush=True)
        print(f"  Already done    : {len(completed_indices)}", flush=True)
        print(f"  Groups to run   : {len(groups)}", flush=True)
        print(f"  Workers         : {N_WORKERS}", flush=True)
        print(f"  Voltage points  : {n_eta}", flush=True)
        print(f"  Min converged   : {min_converged_fraction*100:.0f}%", flush=True)
        print(f"  Continuation    : anchor={CONTINUATION_ANCHOR_K02:.0e}, "
              f"steps={CONTINUATION_N_STEPS}", flush=True)
        print(f"{'='*78}\n", flush=True)

        t_start = time.time()
        n_valid = int(all_converged.sum())
        n_failed = 0
        n_completed = len(completed_indices)

        ctx = mp.get_context("spawn")
        Nx, Ny, beta = MESH_NX, MESH_NY, MESH_BETA

        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            mp_context=ctx,
            initializer=_gapfill_worker_init,
            initargs=(base_sp, steady, -I_SCALE, Nx, Ny, beta),
        ) as executor:
            future_to_gidx = {}
            for g_idx, group in enumerate(groups):
                future = executor.submit(
                    _gapfill_worker_solve_group, group, g_idx,
                )
                future_to_gidx[future] = g_idx

            pending = set(future_to_gidx.keys())
            while pending:
                done, pending = wait(pending, timeout=300,
                                     return_when=FIRST_COMPLETED)

                if not done:
                    wall = time.time() - t_start
                    print(
                        f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')}  "
                        f"waiting...  {n_completed}/{N} complete  "
                        f"wall={_fmt_duration(wall)}",
                        flush=True,
                    )
                    continue

                for future in done:
                    g_idx = future_to_gidx[future]
                    try:
                        group_results = future.result()
                    except Exception as e:
                        print(f"[ERROR] Worker group {g_idx} failed: {e}", flush=True)
                        continue

                    for r in group_results:
                        idx = r["index"]
                        if idx not in completed_indices:
                            n_completed += 1
                            completed_indices.add(idx)
                        elapsed = r.get("elapsed", 0.0)
                        all_timings[idx] = elapsed
                        method = r.get("method", "unknown")
                        method_counts[method] = method_counts.get(method, 0) + 1

                        if r["current_density"] is None:
                            n_failed += 1
                            continue

                        cd = np.array(r["current_density"])
                        pc = np.array(r["peroxide_current"])
                        conv_mask = np.array(r["converged_mask"])
                        frac = sum(conv_mask) / n_eta

                        if frac >= min_converged_fraction:
                            all_cd[idx] = _interpolate_failed_points(
                                cd, conv_mask, phi_applied)
                            all_pc[idx] = _interpolate_failed_points(
                                pc, conv_mask, phi_applied)
                            all_converged[idx] = True
                            n_valid += 1
                        else:
                            n_failed += 1

                    # Progress summary
                    wall = time.time() - t_start
                    new_done = n_completed - len(completed_indices)
                    avg = wall / max(new_done, 1)
                    remaining = N - n_completed
                    eta = avg * remaining
                    pct = n_completed / N * 100

                    # Build method breakdown string
                    method_str = "  ".join(
                        f"{k}={v}" for k, v in sorted(method_counts.items())
                    )

                    print(
                        f"\n{'='*78}\n"
                        f"  [GROUP {g_idx+1}/{len(groups)} done]  "
                        f"{n_completed}/{N} ({pct:.1f}%)  "
                        f"valid={n_valid} fail={n_failed}\n"
                        f"  wall={_fmt_duration(wall)}  "
                        f"ETA={_fmt_duration(eta)}  "
                        f"avg={avg:.1f}s/sample\n"
                        f"  methods: {method_str}\n"
                        f"{'='*78}",
                        flush=True,
                    )

                    # Checkpoint
                    _save_checkpoint(
                        new_data_path, all_samples, all_cd, all_pc,
                        all_converged, all_timings, phi_applied,
                        n_completed,
                    )

    # Save final
    valid_mask = all_converged
    n_valid_final = int(valid_mask.sum())

    np.savez_compressed(
        new_data_path,
        parameters=all_samples[valid_mask],
        current_density=all_cd[valid_mask],
        peroxide_current=all_pc[valid_mask],
        phi_applied=phi_applied,
        all_parameters=all_samples,
        all_current_density=all_cd,
        all_peroxide_current=all_pc,
        all_converged=all_converged,
        all_timings=all_timings,
        n_completed=N,
        # Region metadata
        region_1_n=REGION_1_N,
        region_2_n=REGION_2_N,
        region_3_n=REGION_3_N,
    )

    elapsed = time.time() - t0
    print(f"\n{'#'*78}", flush=True)
    print(f"  PHASE 1 COMPLETE: GAP-FILL DATA GENERATION", flush=True)
    print(f"  Total time     : {_fmt_duration(elapsed)}", flush=True)
    print(f"  Total samples  : {N}", flush=True)
    print(f"  Valid (saved)  : {n_valid_final}  ({n_valid_final/N*100:.1f}%)", flush=True)
    print(f"  Failed         : {N - n_valid_final}  ({(N-n_valid_final)/N*100:.1f}%)",
          flush=True)
    print(f"  Saved to       : {new_data_path}", flush=True)
    if method_counts:
        print(f"  Escalation breakdown:", flush=True)
        for method, count in sorted(method_counts.items()):
            print(f"    {method:20s}: {count:4d}", flush=True)
    print(f"{'#'*78}\n", flush=True)


# ==================================================================
# Phase 2: Merge with existing data
# ==================================================================

def phase_2_merge():
    """Merge gap-fill data with existing v11 training data."""
    t0 = time.time()

    gapfill_path = os.path.join(OUTPUT_DIR, "training_data_gapfill.npz")
    existing_path = os.path.join(_ROOT, "data", "surrogate_models",
                                 "training_data_merged.npz")
    merged_path = os.path.join(OUTPUT_DIR, "training_data_merged_v12.npz")

    if os.path.exists(merged_path):
        tmp = np.load(merged_path)
        if "parameters" in tmp and len(tmp["parameters"]) > 0:
            print(f"Phase 2: merged data already exists ({len(tmp['parameters'])} samples)",
                  flush=True)
            return

    if not os.path.exists(gapfill_path):
        print("ERROR: gap-fill data not found. Run Phase 1 first.", flush=True)
        return

    gapfill = np.load(gapfill_path)
    gf_params = gapfill["parameters"]
    gf_cd = gapfill["current_density"]
    gf_pc = gapfill["peroxide_current"]
    gf_phi = gapfill["phi_applied"]

    print(f"Gap-fill data: {len(gf_params)} valid samples", flush=True)

    if not os.path.exists(existing_path):
        print(f"WARNING: existing merged data not found at {existing_path}", flush=True)
        print("Saving gap-fill data as standalone.", flush=True)
        merged_params = gf_params
        merged_cd = gf_cd
        merged_pc = gf_pc
        n_existing = 0
    else:
        existing = np.load(existing_path)
        ex_params = existing["parameters"]
        ex_cd = existing["current_density"]
        ex_pc = existing["peroxide_current"]
        ex_phi = existing["phi_applied"]

        if not np.allclose(ex_phi, gf_phi):
            print("ERROR: voltage grids don't match! Cannot merge.", flush=True)
            return

        n_existing = len(ex_params)
        merged_params = np.concatenate([ex_params, gf_params], axis=0)
        merged_cd = np.concatenate([ex_cd, gf_cd], axis=0)
        merged_pc = np.concatenate([ex_pc, gf_pc], axis=0)

        print(f"Existing data: {n_existing} samples", flush=True)
        print(f"Merged total:  {len(merged_params)} samples", flush=True)

    np.savez_compressed(
        merged_path,
        parameters=merged_params,
        current_density=merged_cd,
        peroxide_current=merged_pc,
        phi_applied=gf_phi,
        n_existing=n_existing,
        n_gapfill=len(gf_params),
    )

    # Also create a new train/test split
    rng = np.random.default_rng(seed=888)
    N_total = len(merged_params)
    n_test = max(50, int(N_total * 0.15))
    perm = rng.permutation(N_total)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    split_path = os.path.join(OUTPUT_DIR, "split_indices_v12.npz")
    np.savez_compressed(
        split_path,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    elapsed = time.time() - t0
    print(f"\n{'#'*78}", flush=True)
    print(f"  PHASE 2 COMPLETE: DATA MERGE", flush=True)
    print(f"  Total time     : {_fmt_duration(elapsed)}", flush=True)
    print(f"  Merged samples : {N_total}", flush=True)
    print(f"  Train / Test   : {len(train_idx)} / {len(test_idx)}", flush=True)
    print(f"  Saved to       : {merged_path}", flush=True)
    print(f"{'#'*78}\n", flush=True)


# ==================================================================
# Main
# ==================================================================

if __name__ == "__main__":
    t_grand = time.time()

    print("=" * 78, flush=True)
    print("  OVERNIGHT TRAINING V12 -- TARGETED GAP-FILL", flush=True)
    print("=" * 78, flush=True)
    print(flush=True)

    if RUN_PHASE_1:
        print("=" * 78, flush=True)
        print("  PHASE 1: GAP-FILL DATA GENERATION (~1000 samples)", flush=True)
        print("=" * 78, flush=True)
        phase_1_gapfill_generation()

    if RUN_PHASE_2:
        print("=" * 78, flush=True)
        print("  PHASE 2: MERGE WITH EXISTING DATA", flush=True)
        print("=" * 78, flush=True)
        phase_2_merge()

    total = time.time() - t_grand
    print(f"\nTotal elapsed: {total/3600:.1f}h ({_fmt_duration(total)})", flush=True)
