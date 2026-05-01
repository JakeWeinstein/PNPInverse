#!/usr/bin/env python3
"""Overnight training data generation v16.

Two-phase solve using unified grid_charge_continuation API.
Global SharedMemory IC cache with KD-tree lookup.
100% strict convergence — no interpolation.

Usage:
    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python \
        scripts/surrogate/overnight_train_v16.py 2>&1 | tee StudyResults/surrogate_v16/run.log

Smoke test:
    ... overnight_train_v16.py --max-batches 1 --batch-size 5 --smoke-test

Resume: Re-run the same command. Checkpoints are loaded automatically.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path bootstrap (must precede project imports) ──────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
from scipy.spatial import KDTree

# ── Project imports (non-Firedrake) ────────────────────────────────────
from scripts._bv_common import (
    setup_firedrake_env,
    FOUR_SPECIES_CHARGED,
    I_SCALE,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
)
setup_firedrake_env()

from Surrogate.sampling import ParameterBounds, generate_multi_region_lhs_samples
from Surrogate.training import _order_samples_nearest_neighbor

# ── Firedrake-heavy imports: DEFERRED ──────────────────────────────────
# These must NOT be imported at top level — they conflict with
# ProcessPoolExecutor(spawn) workers that initialize Firedrake
# independently. Plan-4 and plan-5 will add them as local imports
# inside functions:
#   from Forward.bv_solver.grid_charge_continuation import solve_grid_with_charge_continuation
#   from Forward.bv_solver.forms import build_context, build_forms
#   from Forward.bv_solver.observables import _build_bv_observable_form
#   from Forward.steady_state.bv import compute_bv_reaction_rates, configure_bv_solver_params
#   import firedrake as fd

# ── Constants ──────────────────────────────────────────────────────────
MESH_NX, MESH_NY, MESH_BETA = 8, 200, 3.0
N_WORKERS = 8
DT = 0.5
MAX_SS_STEPS = 100
T_END = DT * MAX_SS_STEPS  # 50.0
BATCH_SIZE = 500  # 350 wide + 150 focused
N_WIDE_PER_BATCH = 350
N_FOCUSED_PER_BATCH = 150
GROUP_SIZE = 5
IC_CACHE_MAX_ENTRIES = 500
MIN_CONVERGED_FRACTION = 1.0  # STRICT: ALL points must converge
OUTPUT_DIR = Path("StudyResults/surrogate_v16")

logger = logging.getLogger(__name__)


# ── Voltage grid ───────────────────────────────────────────────────────

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


# ── SER / steady-state constants (mirrored from grid_charge_continuation) ─
_SER_GROWTH_CAP = 4.0       # max dt multiplier per step
_SER_SHRINK = 0.5            # dt multiplier when residual grows
_SER_DT_MAX_RATIO = 20.0    # max dt / dt_initial
_STEADY_REL_TOL = 1e-4       # relative change threshold
_STEADY_ABS_TOL = 1e-8       # absolute change threshold
_STEADY_CONSEC = 4           # consecutive steady steps required


# ── Fast-path solver ──────────────────────────────────────────────────

def _warm_start_sweep(
    solver_params_base,
    cached_U_data: dict,
    mesh,
    phi_applied_grid: np.ndarray,
    max_steps_per_point: int = 20,
) -> tuple:
    """Warm-start PTC sweep across all voltage points for one sample.

    Builds context/forms/solver ONCE, then iterates over phi_applied_grid,
    loading cached ICs and running a short PTC loop at each point.

    Args:
        solver_params_base: Base SolverParams (phi_applied will be set per point).
        cached_U_data: dict mapping eta_idx -> tuple of numpy arrays (IC per voltage point).
        mesh: Firedrake mesh.
        phi_applied_grid: array of eta values (e.g. 42 points).
        max_steps_per_point: max PTC iterations per voltage point.

    Returns:
        (all_converged, results) where results[i] = dict with keys
        'U_data', 'obs_cd', 'obs_pc' for each voltage point.
        If any point fails, returns (False, None).
    """
    import dataclasses as _dc
    import firedrake as fd
    from Forward.bv_solver.forms import build_context, build_forms, set_initial_conditions
    from Forward.bv_solver.observables import _build_bv_observable_form

    # Build context/forms/solver ONCE with the first voltage point
    first_eta = float(phi_applied_grid[0])
    solver_params = solver_params_base.with_phi_applied(first_eta)
    ctx = build_context(solver_params, mesh=mesh)
    ctx = build_forms(ctx, solver_params)
    set_initial_conditions(ctx, solver_params, blob=False)

    # Build solver once
    J = fd.derivative(ctx["F_res"], ctx["U"])
    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=J,
    )
    solve_opts = dict(solver_params.solver_options)
    solve_opts.setdefault("snes_error_if_not_converged", True)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)

    # Time-stepping constants
    dt_const = ctx["dt_const"]
    dt_initial = float(dt_const)
    dt_max = dt_initial * _SER_DT_MAX_RATIO

    # Observable forms — build ONCE, reuse across voltage points.
    # These reference phi_applied_func symbolically, so they evaluate
    # the current value at assemble-time after .assign().
    conv_obs_form = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0,
    )
    obs_form_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
    )
    obs_form_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
    )

    results = []

    for i, phi_val in enumerate(phi_applied_grid):
        # (a) Update phi_applied_func
        ctx["phi_applied_func"].assign(float(phi_val))

        # (b) Load cached ICs
        if i not in cached_U_data:
            return (False, None)
        for src, dst in zip(cached_U_data[i], ctx["U"].dat):
            dst.data[:] = src

        # (c) Sync U_prev
        ctx["U_prev"].assign(ctx["U"])

        # (d) SER-adaptive PTC loop
        dt_current = dt_initial
        dt_const.assign(dt_initial)

        prev_flux_val = None
        prev_delta = None
        steady_count = 0
        converged = False

        try:
            for step in range(1, max_steps_per_point + 1):
                solver.solve()
                ctx["U_prev"].assign(ctx["U"])

                flux_val = float(fd.assemble(conv_obs_form))

                if prev_flux_val is not None:
                    delta = abs(flux_val - prev_flux_val)
                    scale_val = max(abs(flux_val), abs(prev_flux_val), _STEADY_ABS_TOL)
                    rel_metric = delta / scale_val
                    is_steady = (rel_metric <= _STEADY_REL_TOL) or (delta <= _STEADY_ABS_TOL)
                    steady_count = steady_count + 1 if is_steady else 0

                    # SER adaptive dt
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
                    converged = True
                    break

        except Exception:
            # SNES divergence or PETSc error -> fast path fails
            return (False, None)

        # (e) If NOT converged at this point, bail out
        if not converged:
            return (False, None)

        # (f) Extract observables (forms built once before the loop)
        cd = float(fd.assemble(obs_form_cd))
        pc = float(fd.assemble(obs_form_pc))

        # (g) Save U_data snapshot
        U_data = tuple(d.data_ro.copy() for d in ctx["U"].dat)

        results.append({
            "U_data": U_data,
            "obs_cd": cd,
            "obs_pc": pc,
        })

    return (True, results)


# ── Slow-path observable callback ─────────────────────────────────────

def _make_extract_observables_callback() -> tuple:
    """Create a per-point callback and its accumulator for slow-path solves.

    Returns:
        (callback, accumulator) where accumulator is a dict mapping
        orig_idx -> {'current_density': float, 'peroxide_current': float}.
        The callback captures results into the accumulator since the return
        value from per_point_callback may be discarded by the unified API.
    """
    accumulator: dict = {}

    def _extract_observables_callback(orig_idx: int, eta_i: float, ctx: dict) -> dict:
        """Per-point callback for solve_grid_with_charge_continuation.

        Called by the unified API as: callback(orig_idx, eta_i, ctx).
        Extracts current_density and peroxide_current via observable form assembly.
        """
        from Forward.bv_solver.observables import _build_bv_observable_form
        import firedrake as fd

        obs_form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        obs_form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )
        cd = float(fd.assemble(obs_form_cd))
        pc = float(fd.assemble(obs_form_pc))
        result = {"current_density": cd, "peroxide_current": pc}
        accumulator[orig_idx] = result
        return result

    return _extract_observables_callback, accumulator


# ── Slow-path solver ──────────────────────────────────────────────────

def _slow_path_solve(
    solver_params,
    phi_applied_grid: np.ndarray,
    mesh,
) -> tuple:
    """Full grid solve via unified API with one retry on failure.

    Args:
        solver_params: SolverParams instance.
        phi_applied_grid: Array of voltage points.
        mesh: Firedrake mesh.

    Returns:
        (success, result, observables) where:
        - success: bool
        - result: GridChargeContinuationResult or None
        - observables: dict mapping orig_idx -> {'current_density', 'peroxide_current'}
    """
    import dataclasses as _dc
    from Forward.bv_solver.grid_charge_continuation import (
        solve_grid_with_charge_continuation,
    )

    callback, accumulator = _make_extract_observables_callback()

    # First attempt: standard parameters
    try:
        result = solve_grid_with_charge_continuation(
            solver_params,
            phi_applied_values=phi_applied_grid,
            charge_steps=10,
            mesh=mesh,
            min_delta_z=0.005,
            max_eta_gap=3.0,
            per_point_callback=callback,
        )
        if result.all_converged():
            return (True, result, accumulator)
    except Exception:
        pass

    # Retry: more charge steps + relaxed SNES
    logger.warning("Slow-path first attempt failed; retrying with relaxed parameters")
    relaxed_opts = dict(solver_params.solver_options)
    relaxed_opts["snes_max_it"] = 500
    relaxed_opts["snes_atol"] = 1e-6
    relaxed_opts["snes_rtol"] = 1e-6
    relaxed_params = _dc.replace(solver_params, solver_options=relaxed_opts)

    callback_retry, accumulator_retry = _make_extract_observables_callback()

    try:
        result = solve_grid_with_charge_continuation(
            relaxed_params,
            phi_applied_values=phi_applied_grid,
            charge_steps=15,
            mesh=mesh,
            min_delta_z=0.005,
            max_eta_gap=3.0,
            per_point_callback=callback_retry,
        )
        if result.all_converged():
            return (True, result, accumulator_retry)
    except Exception:
        pass

    return (False, None, {})


# ── Strict validation ─────────────────────────────────────────────────

def _validate_sample(
    converged_flags: np.ndarray,
    z_factors: np.ndarray,
    obs_cd: np.ndarray,
    obs_pc: np.ndarray,
    phi_applied: Optional[np.ndarray] = None,
) -> bool:
    """Return True only if ALL voltage points pass strict validation.

    Requirements:
    - Every point must be converged.
    - Every z_factor >= 1.0 - 1e-6.
    - All observables must be finite (no NaN, no Inf).
    - F2: |cd| must not exceed diffusion-limit estimate.
    - F3: Peroxide selectivity must not exceed 100%.
    - F7: Peroxide current must have correct sign at cathodic voltages.
    - W1: Clip saturation warnings (non-fatal).

    No 80% threshold. No interpolation.
    """
    from Forward.bv_solver.validation import check_clip_saturation

    if not np.all(converged_flags):
        return False
    if not np.all(z_factors >= 1.0 - 1e-6):
        return False
    if not (np.all(np.isfinite(obs_cd)) and np.all(np.isfinite(obs_pc))):
        return False

    # Nondimensional diffusion-limit estimate: 2 * max(c_bulk_hat).
    # FOUR_SPECIES_CHARGED.c0_vals_hat = [1.0, 0.0, 0.2, 0.2] -> max = 1.0
    I_LIM = 2.0 * max(FOUR_SPECIES_CHARGED.c0_vals_hat)

    for i in range(len(obs_cd)):
        cd_i = obs_cd[i]
        pc_i = obs_pc[i]

        # F2: current density exceeds diffusion limit
        if abs(cd_i) > I_LIM * 1.05:
            logger.warning(
                "F2: sample rejected — |cd[%d]|=%.4g exceeds 1.05*I_lim=%.4g",
                i, abs(cd_i), I_LIM * 1.05,
            )
            return False

        # F3: peroxide selectivity > 100%
        if abs(cd_i) > 1e-10 and abs(pc_i / cd_i) > 1.05:
            logger.warning(
                "F3: sample rejected — |pc/cd|=%.4g > 1.05 at point %d",
                abs(pc_i / cd_i), i,
            )
            return False

        # F7: peroxide current wrong sign at cathodic voltages
        if cd_i < 0.0 and pc_i > abs(cd_i) * 0.01:
            logger.warning(
                "F7: sample rejected — pc=%.4g > 0 while cd=%.4g < 0 at point %d",
                pc_i, cd_i, i,
            )
            return False

    # W1: clip saturation warnings (non-fatal)
    if phi_applied is not None:
        for i in range(len(obs_cd)):
            clip_warns = check_clip_saturation(
                eta_raw=float(phi_applied[i]),
                exponent_clip=50.0,
                bv_exp_scale=1.0,
                alpha_vals=[0.627, 0.5],
                n_e_vals=[2, 2],
            )
            for w in clip_warns:
                logger.warning("W1 at voltage point %d: %s", i, w)

    return True


# ── SharedMemory-backed IC cache ──────────────────────────────────────

class SharedICCache:
    """Global IC cache backed by multiprocessing.SharedMemory.

    Layout: fixed-size ring buffer in shared memory.
      - Header: [n_entries (int64), max_entries (int64)]
      - Params block: (max_entries, 4) float64 — [log10(k0_1), log10(k0_2), alpha_1, alpha_2]
      - U_data block: (max_entries, n_eta, n_components, n_dofs) float64

    Workers accept stale KD-tree lookups. A stale miss simply falls through
    to the slow path; this is a performance tradeoff, not a correctness issue.
    """

    def __init__(
        self,
        max_entries: int,
        n_eta: int,
        n_components: int,
        n_dofs: int,
    ) -> None:
        self.max_entries = max_entries
        self.n_eta = n_eta
        self.n_components = n_components
        self.n_dofs = n_dofs
        self.hit_count = 0
        self.miss_count = 0

        # Compute sizes
        header_bytes = 2 * 8  # 2 x int64
        self._params_shape = (max_entries, 4)
        self._udata_shape = (max_entries, n_eta, n_components, n_dofs)
        params_bytes = int(np.prod(self._params_shape)) * 8
        udata_bytes = int(np.prod(self._udata_shape)) * 8
        total_bytes = header_bytes + params_bytes + udata_bytes

        self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        self._header_offset = 0
        self._params_offset = header_bytes
        self._udata_offset = header_bytes + params_bytes

        # Create numpy views into shared memory
        buf = self._shm.buf
        self._header = np.ndarray(
            (2,), dtype=np.int64, buffer=buf[self._header_offset:]
        )
        self._params = np.ndarray(
            self._params_shape, dtype=np.float64,
            buffer=buf[self._params_offset:],
        )
        self._udata = np.ndarray(
            self._udata_shape, dtype=np.float64,
            buffer=buf[self._udata_offset:],
        )

        # Initialize
        # Header layout: [n_entries (capped at max), total_writes (uncapped)]
        self._header[0] = 0  # n_entries (capped at max_entries for tree size)
        self._header[1] = 0  # total_writes (monotonically increasing for ring index)
        self._tree: KDTree | None = None

    @property
    def shm_name(self) -> str:
        """Name of the underlying SharedMemory segment."""
        return self._shm.name

    @property
    def n_entries(self) -> int:
        return int(self._header[0])

    def find_nearest(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
        max_dist: float = 0.3,
    ) -> tuple | None:
        """KD-tree lookup in (log10(k0_1), log10(k0_2), alpha_1, alpha_2) space.

        Returns:
            dict mapping eta_idx -> tuple of numpy arrays if a neighbor is
            found within max_dist, else None.
        """
        n = self.n_entries
        if n == 0 or self._tree is None:
            self.miss_count += 1
            return None

        query = np.array([np.log10(max(k0_1, 1e-30)),
                          np.log10(max(k0_2, 1e-30)),
                          alpha_1, alpha_2])
        dist, idx = self._tree.query(query)
        if dist > max_dist:
            self.miss_count += 1
            return None

        self.hit_count += 1
        # Return U_data as dict: eta_idx -> tuple of component arrays
        cached = {}
        for eta_idx in range(self.n_eta):
            components = tuple(
                self._udata[idx, eta_idx, comp, :].copy()
                for comp in range(self.n_components)
            )
            cached[eta_idx] = components
        return cached

    def store(
        self,
        k0_1: float,
        k0_2: float,
        alpha_1: float,
        alpha_2: float,
        U_data_per_eta: dict,
    ) -> None:
        """Store a new entry. Main process only.

        Args:
            k0_1, k0_2, alpha_1, alpha_2: Physical kinetic parameters.
            U_data_per_eta: dict mapping eta_idx -> tuple of numpy arrays
                (one per function space component).
        """
        total_writes = int(self._header[1])
        write_idx = total_writes % self.max_entries  # ring buffer wrap-around

        # Write data BEFORE incrementing counters (memory ordering safety)
        self._params[write_idx, :] = [
            np.log10(max(k0_1, 1e-30)),
            np.log10(max(k0_2, 1e-30)),
            alpha_1,
            alpha_2,
        ]
        for eta_idx, components in U_data_per_eta.items():
            for comp_idx, arr in enumerate(components):
                self._udata[write_idx, eta_idx, comp_idx, :len(arr)] = arr

        # Update counters: total_writes always increments; n_entries caps at max
        self._header[1] = total_writes + 1
        if self._header[0] < self.max_entries:
            self._header[0] = int(self._header[0]) + 1

        # Rebuild KD-tree from active entries
        active_n = self.n_entries
        self._tree = KDTree(self._params[:active_n])

    def save_to_disk(self, path: Path) -> None:
        """Serialize cache contents to an .npz file for persistence.

        Uses atomic temp-file + rename to prevent corruption on crash.
        """
        n = self.n_entries
        total_writes = int(self._header[1])
        tmp_path = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            str(tmp_path),
            params=self._params[:n].copy(),
            udata=self._udata[:n].copy(),
            n_entries=np.array([n]),
            total_writes=np.array([total_writes]),
            n_eta=np.array([self.n_eta]),
            n_components=np.array([self.n_components]),
            n_dofs=np.array([self.n_dofs]),
        )
        os.replace(str(tmp_path), str(path))

    def load_from_disk(self, path: Path) -> None:
        """Restore cache from a previously saved .npz file."""
        data = np.load(str(path))
        n = int(data["n_entries"][0])
        params = data["params"]
        udata = data["udata"]

        count = min(n, self.max_entries)
        self._params[:count] = params[:count]
        self._udata[:count] = udata[:count]
        self._header[0] = count
        # Restore total_writes so the ring buffer continues from the right slot
        if "total_writes" in data:
            self._header[1] = int(data["total_writes"][0])
        else:
            self._header[1] = count  # legacy fallback

        if count > 0:
            self._tree = KDTree(self._params[:count])

    @classmethod
    def open_worker_view(
        cls,
        shm_name: str,
        max_entries: int,
        n_eta: int,
        n_components: int,
        n_dofs: int,
    ) -> "SharedICCache":
        """Open a read-only worker view of an existing shared memory segment.

        Workers get a snapshot of the KD-tree at open time. Stale lookups
        (missing recently stored entries) simply fall through to the slow path.
        """
        instance = object.__new__(cls)
        instance.max_entries = max_entries
        instance.n_eta = n_eta
        instance.n_components = n_components
        instance.n_dofs = n_dofs
        instance.hit_count = 0
        instance.miss_count = 0

        instance._params_shape = (max_entries, 4)
        instance._udata_shape = (max_entries, n_eta, n_components, n_dofs)

        header_bytes = 2 * 8
        params_bytes = int(np.prod(instance._params_shape)) * 8

        instance._shm = shared_memory.SharedMemory(name=shm_name, create=False)
        buf = instance._shm.buf
        instance._header_offset = 0
        instance._params_offset = header_bytes
        instance._udata_offset = header_bytes + params_bytes

        instance._header = np.ndarray(
            (2,), dtype=np.int64, buffer=buf[instance._header_offset:]
        )
        instance._params = np.ndarray(
            instance._params_shape, dtype=np.float64,
            buffer=buf[instance._params_offset:],
        )
        instance._udata = np.ndarray(
            instance._udata_shape, dtype=np.float64,
            buffer=buf[instance._udata_offset:],
        )

        # Build KD-tree from current snapshot
        n = instance.n_entries
        if n > 0:
            instance._tree = KDTree(instance._params[:n].copy())
        else:
            instance._tree = None

        return instance


# ── Checkpoint save/load ──────────────────────────────────────────────

def _save_forward_checkpoint(
    path: Path,
    params: np.ndarray,
    cd: np.ndarray,
    pc: np.ndarray,
    converged: np.ndarray,
    timings: np.ndarray,
    phi_applied: np.ndarray,
    n_completed: int,
) -> None:
    """Atomic checkpoint save via os.replace().

    Format: training_data.npz with keys:
      parameters (N,4), current_density (N,n_eta), peroxide_current (N,n_eta),
      converged (N,), timings (N,), phi_applied (n_eta,), n_completed.
    """
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        str(tmp_path),
        parameters=params,
        current_density=cd,
        peroxide_current=pc,
        converged=converged,
        timings=timings,
        phi_applied=phi_applied,
        n_completed=np.array([n_completed]),
    )
    os.replace(str(tmp_path), str(path))


def _save_solutions_checkpoint(
    path: Path,
    U_data_dict: dict,
) -> None:
    """Save converged U_data arrays for adjoint pass.

    Format: batch_solutions.npz with keys u_data_{sample}_{eta}.

    Args:
        U_data_dict: dict mapping (sample_idx, eta_idx) -> numpy array.
    """
    arrays = {}
    for (sample_idx, eta_idx), arr in U_data_dict.items():
        arrays[f"u_data_{sample_idx}_{eta_idx}"] = arr
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(str(tmp_path), **arrays)
    os.replace(str(tmp_path), str(path))


def _load_forward_checkpoint(path: Path) -> dict | None:
    """Load and return forward checkpoint data for resume.

    Returns:
        dict with keys: parameters, current_density, peroxide_current,
        converged, timings, phi_applied, n_completed.
        Returns None if the file does not exist.
    """
    if not path.exists():
        return None
    data = np.load(str(path))
    return {
        "parameters": data["parameters"],
        "current_density": data["current_density"],
        "peroxide_current": data["peroxide_current"],
        "converged": data["converged"],
        "timings": data["timings"],
        "phi_applied": data["phi_applied"],
        "n_completed": int(data["n_completed"][0]),
    }


# ── Worker globals ────────────────────────────────────────────────────

_worker_mesh = None
_worker_cache_view: Optional[SharedICCache] = None


def _worker_init(
    shm_name: str,
    max_entries: int,
    n_eta: int,
    n_components: int,
    n_dofs: int,
) -> None:
    """Per-worker initialization: create mesh, open SharedMemory IC cache view.

    Workers MUST ignore SIGINT so only the main process handles Ctrl-C
    graceful shutdown. Without this, workers crash with KeyboardInterrupt
    instead of letting the main process orchestrate shutdown.
    """
    global _worker_mesh, _worker_cache_view
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    from Forward.bv_solver.mesh import make_graded_rectangle_mesh
    _worker_mesh = make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
    )

    _worker_cache_view = SharedICCache.open_worker_view(
        shm_name, max_entries, n_eta, n_components, n_dofs,
    )


def _worker_solve_group_v16(
    group_tasks: List[Tuple[int, float, float, float, float]],
    phi_applied_list: List[float],
) -> List[Dict[str, Any]]:
    """Process a group of parameter samples across the full voltage grid.

    IC source priority:
    1. Check SharedMemory IC cache via find_nearest() (threshold 0.3)
    2. If cache miss AND previous sample in group converged -> use that U_data
    3. If neither -> go directly to slow path (no warm-start)

    Solve cascade per sample:
    1. If ICs available -> fast path via _warm_start_sweep()
    2. If fast path fails -> discard all -> slow path for entire sample
    3. Slow path retries internally with relaxed SNES

    Returns list of result dicts with params, observables, U_data, valid flag,
    timing, and path_taken.
    """
    phi_applied = np.array(phi_applied_list)
    n_eta = len(phi_applied)
    results: List[Dict[str, Any]] = []
    prev_U_data: Optional[Dict[int, tuple]] = None  # within-group chain

    for task_idx, (sample_idx, k0_1, k0_2, alpha_1, alpha_2) in enumerate(group_tasks):
        t0 = time.time()
        path_taken = "none"

        # Build solver params for this sample
        solver_params = make_bv_solver_params(
            eta_hat=0.0,
            dt=DT,
            t_end=T_END,
            species=FOUR_SPECIES_CHARGED,
            snes_opts=SNES_OPTS_CHARGED,
            k0_hat_r1=k0_1,
            k0_hat_r2=k0_2,
            alpha_r1=alpha_1,
            alpha_r2=alpha_2,
        )

        # -- IC source priority --
        cached_U_data: Optional[Dict[int, tuple]] = None

        # Priority 1: SharedMemory IC cache
        if _worker_cache_view is not None:
            cached_U_data = _worker_cache_view.find_nearest(
                k0_1, k0_2, alpha_1, alpha_2, max_dist=0.3,
            )

        # Priority 2: within-group chain from previous converged sample
        if cached_U_data is None and prev_U_data is not None:
            cached_U_data = prev_U_data

        # -- Solve cascade --
        success = False
        obs_cd: Optional[np.ndarray] = None
        obs_pc: Optional[np.ndarray] = None
        U_data_per_eta: Optional[Dict[int, tuple]] = None
        converged_flags = np.zeros(n_eta, dtype=bool)
        z_factors = np.ones(n_eta, dtype=float)

        # Fast path: if we have ICs
        if cached_U_data is not None:
            fast_ok, fast_results = _warm_start_sweep(
                solver_params, cached_U_data, _worker_mesh, phi_applied,
                max_steps_per_point=20,
            )
            if fast_ok and fast_results is not None:
                path_taken = "fast"
                obs_cd = np.array([r["obs_cd"] for r in fast_results])
                obs_pc = np.array([r["obs_pc"] for r in fast_results])
                U_data_per_eta = {
                    i: r["U_data"] for i, r in enumerate(fast_results)
                }
                converged_flags[:] = True
                z_factors[:] = 1.0
                success = True

        # Slow path: if fast path unavailable or failed
        if not success:
            path_taken = "slow"
            slow_ok, slow_result, slow_obs = _slow_path_solve(
                solver_params, phi_applied, _worker_mesh,
            )
            if slow_ok and slow_result is not None:
                obs_cd = np.full(n_eta, np.nan)
                obs_pc = np.full(n_eta, np.nan)
                U_data_per_eta = {}
                for orig_idx, pt in slow_result.points.items():
                    converged_flags[orig_idx] = pt.converged
                    z_factors[orig_idx] = pt.achieved_z_factor
                    U_data_per_eta[orig_idx] = pt.U_data
                    if orig_idx in slow_obs:
                        obs_cd[orig_idx] = slow_obs[orig_idx]["current_density"]
                        obs_pc[orig_idx] = slow_obs[orig_idx]["peroxide_current"]
                # Check all n_eta points are present (sparse result → reject)
                all_present = len(slow_result.points) == n_eta
                success = (
                    all_present
                    and np.all(converged_flags)
                    and np.all(np.isfinite(obs_cd))
                    and np.all(np.isfinite(obs_pc))
                )
            else:
                success = False

        elapsed = time.time() - t0

        # Validate
        valid = False
        if success and obs_cd is not None and obs_pc is not None:
            valid = _validate_sample(converged_flags, z_factors, obs_cd, obs_pc, phi_applied)

        # Update within-group chain
        if valid and U_data_per_eta is not None:
            prev_U_data = U_data_per_eta
        elif not valid:
            prev_U_data = None  # break chain on failure

        # Serialize U_data for return (tuples of numpy arrays -> list of lists)
        serialized_U_data: Optional[Dict[int, List[np.ndarray]]] = None
        if valid and U_data_per_eta is not None:
            serialized_U_data = {
                eta_idx: [arr.copy() if hasattr(arr, "copy") else np.array(arr)
                          for arr in components]
                for eta_idx, components in U_data_per_eta.items()
            }

        results.append({
            "sample_idx": sample_idx,
            "params": [k0_1, k0_2, alpha_1, alpha_2],
            "obs_cd": obs_cd.tolist() if obs_cd is not None else None,
            "obs_pc": obs_pc.tolist() if obs_pc is not None else None,
            "U_data_per_eta": serialized_U_data,
            "valid": valid,
            "elapsed": elapsed,
            "path_taken": path_taken,
            "n_converged": int(converged_flags.sum()),
            "n_eta": n_eta,
        })

    return results


# ── Signal handling ───────────────────────────────────────────────────

_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    """Set shutdown flag on SIGINT/SIGTERM. Workers finish current group."""
    global _shutdown_requested
    logger.warning(
        "Signal %d received — finishing current group then exiting", signum,
    )
    _shutdown_requested = True


# ── Streamed metrics ─────────────────────────────────────────────────

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


def _log_sample_result(
    worker_id: int,
    sample_idx: int,
    total: int,
    result: Dict[str, Any],
) -> None:
    """Log a single-line per-sample result."""
    params = result["params"]
    status = "VALID" if result["valid"] else "FAIL "
    n_conv = result["n_converged"]
    n_eta = result["n_eta"]
    path = result["path_taken"]
    dt = result["elapsed"]
    logger.info(
        "[W%d] #%d/%d  idx=%d  k0=[%.2e, %.2e]  a=[%.2f, %.2f]  "
        "%s  conv=%d/%d  path=%s  dt=%.1fs",
        worker_id, sample_idx + 1, total, result["sample_idx"],
        params[0], params[1], params[2], params[3],
        status, n_conv, n_eta, path, dt,
    )


def _log_checkpoint_summary(
    n_completed: int,
    total: int,
    n_valid: int,
    n_failed: int,
    timings: List[float],
    cache: SharedICCache,
) -> None:
    """Log a checkpoint summary block."""
    pct = n_completed / max(total, 1) * 100
    fast_times = [t for t in timings if t > 0]
    avg_pace = np.mean(fast_times) if fast_times else 0.0
    total_hits = cache.hit_count
    total_lookups = cache.hit_count + cache.miss_count
    hit_rate = total_hits / max(total_lookups, 1) * 100

    logger.info(
        "=== CHECKPOINT at %d/%d (%.1f%%) | valid=%d fail=%d ===",
        n_completed, total, pct, n_valid, n_failed,
    )
    logger.info(
        "    Pace: %.1fs/pt  Cache: %d entries, fast-path hit rate: %.0f%%",
        avg_pace, cache.n_entries, hit_rate,
    )


# ── Resume logic ─────────────────────────────────────────────────────

def _detect_resume_point(batch_dir: Path) -> Optional[Dict[str, Any]]:
    """Check for existing checkpoint in batch_dir; return loaded data or None.

    Must load ALL THREE checkpoint artifacts:
    1. training_data.npz -- parameters, observables, convergence flags, timings
    2. batch_solutions.npz -- converged U_data per sample/eta
    3. ic_cache.npz -- SharedICCache state

    If any artifact is missing or corrupt, returns None (start fresh).
    """
    td_path = batch_dir / "training_data.npz"
    sol_path = batch_dir / "batch_solutions.npz"
    ic_path = batch_dir / "ic_cache.npz"

    if not (td_path.exists() and sol_path.exists() and ic_path.exists()):
        return None

    try:
        td = _load_forward_checkpoint(td_path)
        if td is None:
            return None
        sol_data = np.load(str(sol_path))
        # Reconstruct U_data dict from solution keys
        U_data_dict: Dict[Tuple[int, int], np.ndarray] = {}
        for key in sol_data.files:
            if key.startswith("u_data_"):
                parts = key.split("_")
                s_idx = int(parts[2])
                e_idx = int(parts[3])
                U_data_dict[(s_idx, e_idx)] = sol_data[key]
        return {
            "training_data": td,
            "U_data_dict": U_data_dict,
            "ic_cache_path": ic_path,
        }
    except Exception as e:
        logger.warning("Corrupt checkpoint in %s: %s — starting fresh", batch_dir, e)
        return None


def _auto_detect_start_batch(args: argparse.Namespace) -> int:
    """Scan OUTPUT_DIR for existing batch_NNNN/ dirs to find start point.

    Returns the next batch index. If a batch dir exists but has no final
    batch_data.npz, that batch needs resuming (return its index).
    """
    if args.start_batch > 0:
        return args.start_batch

    batch_idx = 0
    while True:
        batch_dir = OUTPUT_DIR / f"batch_{batch_idx:04d}"
        final_path = batch_dir / "batch_data.npz"
        if not batch_dir.exists():
            break
        if not final_path.exists():
            # In-progress batch — resume it
            logger.info(
                "Found in-progress batch_%04d (no batch_data.npz), resuming",
                batch_idx,
            )
            return batch_idx
        batch_idx += 1

    if batch_idx > 0:
        logger.info(
            "Auto-detected: %d batches complete, starting at batch %d",
            batch_idx, batch_idx,
        )
    return batch_idx


# ── Batch processing ─────────────────────────────────────────────────

def run_batch(
    batch_idx: int,
    args: argparse.Namespace,
    phi_applied: np.ndarray,
) -> None:
    """Run one batch: sample -> order -> group -> parallel solve -> checkpoint."""
    global _shutdown_requested

    batch_dir = OUTPUT_DIR / f"batch_{batch_idx:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    final_path = batch_dir / "batch_data.npz"

    # Skip completed batches
    if final_path.exists():
        logger.info("Batch %d already complete, skipping", batch_idx)
        return

    # -- Sampling --
    batch_size = args.batch_size
    n_wide = int(batch_size * 0.7)
    n_focused = batch_size - n_wide

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

    seed_wide = 1000 + batch_idx * 2
    seed_focused = 1001 + batch_idx * 2

    samples = generate_multi_region_lhs_samples(
        wide_bounds=wide_bounds,
        focused_bounds=focused_bounds,
        n_base=n_wide,
        n_focused=n_focused,
        seed_base=seed_wide,
        seed_focused=seed_focused,
        log_space_k0=True,
    )

    N = samples.shape[0]
    n_eta = len(phi_applied)

    logger.info("=" * 70)
    logger.info(
        "BATCH %d: %d samples, %d voltage points  seeds=(%d, %d)",
        batch_idx, N, n_eta, seed_wide, seed_focused,
    )
    logger.info("=" * 70)

    # -- Order by nearest-neighbor --
    nn_order = _order_samples_nearest_neighbor(samples)

    # -- Pre-allocate storage --
    all_params = samples.copy()
    all_cd = np.full((N, n_eta), np.nan, dtype=float)
    all_pc = np.full((N, n_eta), np.nan, dtype=float)
    all_converged = np.zeros(N, dtype=bool)
    all_timings = np.zeros(N, dtype=float)
    U_data_dict: Dict[Tuple[int, int], np.ndarray] = {}
    completed_indices: set = set()
    n_valid = 0
    n_failed = 0

    # -- Determine mesh DOF count for IC cache sizing --
    # We need n_components and n_dofs. Build a temporary mesh to query.
    # For MESH_NX=8, MESH_NY=200 with CG1 mixed space (5 components):
    # n_dofs ~ (NX+1)*(NY+1) = 9*201 = 1809
    n_components = 5  # 4 species + potential
    n_dofs = (MESH_NX + 1) * (MESH_NY + 1)  # approximate CG1 DOF count

    # -- Initialize IC cache --
    cache = SharedICCache(
        max_entries=IC_CACHE_MAX_ENTRIES,
        n_eta=n_eta,
        n_components=n_components,
        n_dofs=n_dofs,
    )

    # Register atexit handler to clean up shared memory even on unhandled exceptions.
    # SIGKILL still leaks ~1 GB; manual cleanup would be needed in that case.
    import atexit
    def _cleanup_shm():
        try:
            cache._shm.close()
            cache._shm.unlink()
        except Exception:
            pass
    atexit.register(_cleanup_shm)

    # -- Resume from checkpoint --
    resume_data = _detect_resume_point(batch_dir)
    if resume_data is not None:
        td = resume_data["training_data"]
        n_completed_prev = td["n_completed"]
        # Restore full arrays — the checkpoint stores all N rows,
        # with NaN/False for samples not yet processed.
        saved_len = min(n_completed_prev, N)
        all_cd[:saved_len] = td["current_density"][:saved_len]
        all_pc[:saved_len] = td["peroxide_current"][:saved_len]
        all_converged[:saved_len] = td["converged"][:saved_len]
        all_timings[:saved_len] = td["timings"][:saved_len]
        # Rebuild completed_indices from the actual converged mask and
        # non-NaN timing entries — NOT from a contiguous range 0..n-1,
        # because samples are processed in NN-permuted order.
        for i in range(N):
            if all_converged[i] or (all_timings[i] > 0):
                completed_indices.add(i)
        n_valid = int(all_converged.sum())
        n_failed = len(completed_indices) - n_valid
        U_data_dict = resume_data["U_data_dict"]
        cache.load_from_disk(resume_data["ic_cache_path"])
        logger.info(
            "RESUME: %d/%d completed (%d valid, %d failed), cache=%d entries",
            n_completed_prev, N, n_valid, n_failed, cache.n_entries,
        )

    # -- Divide into groups --
    n_groups = (N + GROUP_SIZE - 1) // GROUP_SIZE
    groups: List[List[Tuple[int, float, float, float, float]]] = []
    group_ids: List[int] = []
    for g in range(n_groups):
        start = g * GROUP_SIZE
        end = min(start + GROUP_SIZE, N)
        group_indices = nn_order[start:end]
        if all(int(i) in completed_indices for i in group_indices):
            continue
        group_tasks = [
            (int(i), *samples[i].tolist())
            for i in group_indices
        ]
        groups.append(group_tasks)
        group_ids.append(g)

    if not groups:
        logger.info("All groups already completed!")
        return

    t_batch_start = time.time()
    all_timings_list: List[float] = []
    phi_list = phi_applied.tolist()

    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                cache.shm_name,
                cache.max_entries,
                n_eta,
                n_components,
                n_dofs,
            ),
        ) as executor:
            future_to_gidx: Dict[Any, int] = {}
            groups_submitted = 0

            # Submit initial groups
            for g_offset, group in enumerate(groups):
                if _shutdown_requested:
                    break
                future = executor.submit(
                    _worker_solve_group_v16, group, phi_list,
                )
                future_to_gidx[future] = group_ids[g_offset]
                groups_submitted += 1

            pending = set(future_to_gidx.keys())

            while pending:
                done, pending = wait(
                    pending, timeout=300, return_when=FIRST_COMPLETED,
                )

                if not done:
                    wall = time.time() - t_batch_start
                    logger.info(
                        "[HEARTBEAT] %s  waiting...  %d/%d complete  wall=%s",
                        datetime.datetime.now().strftime("%H:%M:%S"),
                        len(completed_indices), N, _fmt_duration(wall),
                    )
                    continue

                for future in done:
                    g_idx = future_to_gidx[future]
                    try:
                        group_results = future.result()
                    except Exception as e:
                        logger.error("Worker group %d failed: %s", g_idx, e)
                        continue

                    for r_idx, r in enumerate(group_results):
                        idx = r["sample_idx"]
                        completed_indices.add(idx)
                        all_timings[idx] = r["elapsed"]
                        all_timings_list.append(r["elapsed"])

                        _log_sample_result(
                            g_idx % args.n_workers, r_idx, len(group_results), r,
                        )

                        if r["valid"] and r["obs_cd"] is not None:
                            all_cd[idx] = np.array(r["obs_cd"])
                            all_pc[idx] = np.array(r["obs_pc"])
                            all_converged[idx] = True
                            n_valid += 1

                            # Store U_data for adjoint pass
                            if r["U_data_per_eta"] is not None:
                                for eta_idx, components in r["U_data_per_eta"].items():
                                    # Flatten components into single array
                                    flat = np.concatenate(
                                        [np.asarray(c).ravel() for c in components]
                                    )
                                    U_data_dict[(idx, int(eta_idx))] = flat

                                # Update IC cache (main process only)
                                p = r["params"]
                                cache.store(
                                    p[0], p[1], p[2], p[3],
                                    {int(k): tuple(np.asarray(c) for c in v)
                                     for k, v in r["U_data_per_eta"].items()},
                                )
                        else:
                            n_failed += 1

                # Checkpoint after each group completes
                _save_forward_checkpoint(
                    batch_dir / "training_data.npz",
                    all_params, all_cd, all_pc,
                    all_converged, all_timings, phi_applied,
                    len(completed_indices),
                )
                _save_solutions_checkpoint(
                    batch_dir / "batch_solutions.npz",
                    U_data_dict,
                )
                cache.save_to_disk(batch_dir / "ic_cache.npz")

                _log_checkpoint_summary(
                    len(completed_indices), N, n_valid, n_failed,
                    all_timings_list, cache,
                )

                # Check for graceful shutdown
                if _shutdown_requested:
                    logger.warning(
                        "Shutdown requested — waiting for in-flight workers...",
                    )
                    # Let remaining futures complete
                    for remaining_future in pending:
                        try:
                            group_results = remaining_future.result(timeout=600)
                            g_idx = future_to_gidx[remaining_future]
                            for r_idx, r in enumerate(group_results):
                                idx = r["sample_idx"]
                                completed_indices.add(idx)
                                all_timings[idx] = r["elapsed"]
                                all_timings_list.append(r["elapsed"])
                                if r["valid"] and r["obs_cd"] is not None:
                                    all_cd[idx] = np.array(r["obs_cd"])
                                    all_pc[idx] = np.array(r["obs_pc"])
                                    all_converged[idx] = True
                                    n_valid += 1
                                    if r["U_data_per_eta"] is not None:
                                        for eta_idx, components in r["U_data_per_eta"].items():
                                            flat = np.concatenate(
                                                [np.asarray(c).ravel() for c in components]
                                            )
                                            U_data_dict[(idx, int(eta_idx))] = flat
                                        p = r["params"]
                                        cache.store(
                                            p[0], p[1], p[2], p[3],
                                            {int(k): tuple(np.asarray(c) for c in v)
                                             for k, v in r["U_data_per_eta"].items()},
                                        )
                                else:
                                    n_failed += 1
                        except Exception as e:
                            logger.error("In-flight group failed: %s", e)

                    # Final checkpoint before exit
                    _save_forward_checkpoint(
                        batch_dir / "training_data.npz",
                        all_params, all_cd, all_pc,
                        all_converged, all_timings, phi_applied,
                        len(completed_indices),
                    )
                    _save_solutions_checkpoint(
                        batch_dir / "batch_solutions.npz",
                        U_data_dict,
                    )
                    cache.save_to_disk(batch_dir / "ic_cache.npz")
                    logger.info("Graceful shutdown complete")
                    break

    finally:
        # Clean up shared memory
        try:
            cache._shm.close()
            cache._shm.unlink()
        except Exception:
            pass

    if _shutdown_requested:
        return

    # -- Save final batch data --
    valid_mask = all_converged
    n_valid_final = int(valid_mask.sum())

    tmp_final = batch_dir / "batch_data.tmp.npz"
    np.savez_compressed(
        str(tmp_final),
        parameters=all_params[valid_mask],
        current_density=all_cd[valid_mask],
        peroxide_current=all_pc[valid_mask],
        phi_applied=phi_applied,
        all_parameters=all_params,
        all_current_density=all_cd,
        all_peroxide_current=all_pc,
        all_converged=all_converged,
        all_timings=all_timings,
        n_completed=N,
        n_total=N,
    )
    os.replace(str(tmp_final), str(final_path))

    logger.info(
        "Batch %d: %d/%d valid samples saved to %s",
        batch_idx, n_valid_final, N, final_path,
    )


# ── Merge running totals ─────────────────────────────────────────────

def _merge_running_totals(up_to_batch: int) -> None:
    """Merge all batch training_data.npz into training_data_v16_running.npz."""
    merged_path = OUTPUT_DIR / "training_data_v16_running.npz"
    all_params_list: List[np.ndarray] = []
    all_cd_list: List[np.ndarray] = []
    all_pc_list: List[np.ndarray] = []
    phi_applied_saved: Optional[np.ndarray] = None

    for b in range(up_to_batch + 1):
        batch_path = OUTPUT_DIR / f"batch_{b:04d}" / "batch_data.npz"
        if not batch_path.exists():
            continue
        data = np.load(str(batch_path))
        if len(data["parameters"]) > 0:
            all_params_list.append(data["parameters"])
            all_cd_list.append(data["current_density"])
            all_pc_list.append(data["peroxide_current"])
            if phi_applied_saved is None:
                phi_applied_saved = data["phi_applied"]

    if not all_params_list:
        logger.info("No batch data to merge.")
        return

    merged_params = np.concatenate(all_params_list, axis=0)
    merged_cd = np.concatenate(all_cd_list, axis=0)
    merged_pc = np.concatenate(all_pc_list, axis=0)

    tmp_path = merged_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        str(tmp_path),
        parameters=merged_params,
        current_density=merged_cd,
        peroxide_current=merged_pc,
        phi_applied=phi_applied_saved,
        n_batches=up_to_batch + 1,
    )
    os.replace(str(tmp_path), str(merged_path))

    logger.info(
        "Merged %d batches -> %d total valid samples at %s",
        up_to_batch + 1, len(merged_params), merged_path,
    )


# ── CLI argument parser ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for v16 training."""
    p = argparse.ArgumentParser(
        description="v16 overnight training data generation",
    )
    p.add_argument(
        "--max-batches", type=int, default=0,
        help="0 = unlimited",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
    )
    p.add_argument(
        "--start-batch", type=int, default=0,
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick 5-sample test",
    )
    p.add_argument(
        "--n-workers", type=int, default=N_WORKERS,
        help=f"Number of parallel workers (default: {N_WORKERS})",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Top-level entry: parse args, build grid, run batch loop."""
    args = parse_args()
    phi_applied = _build_voltage_grid()

    if args.smoke_test:
        args.max_batches = 1
        args.batch_size = 5

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_eta = len(phi_applied)
    logger.info("#" * 70)
    logger.info("  SURROGATE TRAINING v16 — UNIFIED GRID CHARGE CONTINUATION")
    logger.info("#" * 70)
    logger.info("  Output dir     : %s", OUTPUT_DIR)
    logger.info("  Voltage points : %d  range [%.1f, %.1f]",
                n_eta, phi_applied[-1], phi_applied[0])
    logger.info("  Batch size     : %d", args.batch_size)
    logger.info("  Workers        : %d", args.n_workers)
    logger.info("  Max batches    : %s",
                "unlimited" if args.max_batches == 0 else args.max_batches)
    logger.info("  Mesh           : %dx%d, beta=%.1f", MESH_NX, MESH_NY, MESH_BETA)
    logger.info("#" * 70)

    # Register signal handlers (main process only)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_batch = _auto_detect_start_batch(args)
    batch_idx = start_batch
    t_grand = time.time()

    while args.max_batches == 0 or batch_idx < start_batch + args.max_batches:
        if _shutdown_requested:
            break

        t_batch = time.time()
        logger.info(
            "STARTING BATCH %d  %s",
            batch_idx, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        run_batch(batch_idx, args, phi_applied)

        if _shutdown_requested:
            break

        batch_elapsed = time.time() - t_batch
        total_elapsed = time.time() - t_grand
        logger.info("Batch %d elapsed: %s", batch_idx, _fmt_duration(batch_elapsed))
        logger.info("Total elapsed: %s", _fmt_duration(total_elapsed))

        _merge_running_totals(batch_idx)

        batch_idx += 1

    total_elapsed = time.time() - t_grand
    logger.info("#" * 70)
    logger.info(
        "DONE: %d batches in %s",
        batch_idx - start_batch, _fmt_duration(total_elapsed),
    )
    logger.info("#" * 70)


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
