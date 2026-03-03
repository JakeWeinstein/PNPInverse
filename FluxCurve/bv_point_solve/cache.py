"""Global state, caches, and pool management for BV point solves."""

from __future__ import annotations

from typing import Any, Dict

# P2: Cross-evaluation warm-start cache.
# Persists across calls to solve_bv_curve_points_with_warmstart so that
# the first point of eval N+1 uses eval N's converged solution as IC.
# Keyed by sorted_indices[0] (original index of the smallest-|eta| point).
_cross_eval_cache: Dict[int, tuple] = {}

# P7: Checkpoint-restart warm-start cache.
# After the first full sequential sweep, cache the converged solution for
# every point.  On subsequent evaluations, each point can be solved
# independently using its cached IC (no sequential sweep needed).
_all_points_cache: Dict[int, tuple] = {}
_cache_populated: bool = False

# P8: Module-level parallel pool for fast-path point solves.
# Set by set_parallel_pool() / cleared by close_parallel_pool().
# When not None and enabled, _solve_cached_fast_path will dispatch
# independent point solves to worker processes instead of looping.
_parallel_pool: Any = None

# P6: SER adaptive pseudo-timestep parameters.
_SER_GROWTH_CAP = 4.0    # max dt multiplier per step
_SER_SHRINK = 0.5        # dt multiplier when residual grows
_SER_DT_MAX_RATIO = 20.0  # max dt / dt_initial ratio

# Bridge point parameters.
_BRIDGE_MAX_STEPS = 40       # generous step cap for bridge points
_BRIDGE_SER_DT_MAX_RATIO = 50.0  # allow larger dt growth for bridges

# Default max pseudo-time steps for warm-started points.  When the initial
# guess comes from the previous voltage point's converged solution, only a
# few steps are needed to re-converge.  The full max_steps is still used
# for the first point (no warm-start) and for recovery attempts.
_WARMSTART_MAX_STEPS = 20


def _clear_caches() -> None:
    """Clear all module-level caches.

    Call between multi-fidelity phases (coarse -> fine) to prevent stale
    solutions from a different mesh being reused.
    """
    global _cache_populated
    _cross_eval_cache.clear()
    _all_points_cache.clear()
    _cache_populated = False


def set_parallel_pool(pool: Any) -> None:
    """Set the module-level parallel pool for fast-path point solves."""
    global _parallel_pool
    _parallel_pool = pool


def close_parallel_pool() -> None:
    """Shutdown and clear the module-level parallel pool."""
    global _parallel_pool
    if _parallel_pool is not None:
        try:
            _parallel_pool.close()
        except Exception:
            pass
        _parallel_pool = None
