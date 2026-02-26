"""Forward-solve recovery and solver-option relaxation utilities."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from FluxCurve.config import ForwardRecoveryConfig


def clip_kappa(kappa: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project kappa onto simple box bounds."""
    return np.minimum(np.maximum(kappa, lower), upper)


def _attempt_phase_state(
    attempt: int, recovery: ForwardRecoveryConfig
) -> Tuple[str, int, int]:
    """Return ``(phase, phase_step, cycle_index)`` for a retry attempt index."""
    if attempt <= 0:
        return "baseline", 1, 0

    max_it_only = max(1, int(recovery.max_it_only_attempts))
    anis_only = max(1, int(recovery.anisotropy_only_attempts))
    tol_only = max(1, int(recovery.tolerance_relax_attempts))
    cycle_len = max_it_only + anis_only + tol_only

    cycle_offset = int(attempt - 1)
    cycle_index = cycle_offset // cycle_len
    idx = cycle_offset % cycle_len

    if idx < max_it_only:
        return "max_it", idx + 1, cycle_index
    idx -= max_it_only
    if idx < anis_only:
        return "anisotropy", idx + 1, cycle_index
    idx -= anis_only
    return "tolerance_relax", idx + 1, cycle_index


def _relax_solver_options_for_attempt(
    solver_options: Dict[str, object],
    *,
    phase: str,
    phase_step: int,
    recovery: ForwardRecoveryConfig,
    baseline_options: Mapping[str, object],
) -> None:
    """Apply staged solver-option relaxation for one retry attempt."""
    base_max_it = int(baseline_options.get("snes_max_it", solver_options.get("snes_max_it", 80)))
    base_atol = float(baseline_options.get("snes_atol", solver_options.get("snes_atol", 1e-8)))
    base_rtol = float(baseline_options.get("snes_rtol", solver_options.get("snes_rtol", 1e-8)))
    base_ksp_rtol = float(
        baseline_options.get("ksp_rtol", solver_options.get("ksp_rtol", 1e-8))
    )
    base_linesearch = baseline_options.get(
        "snes_linesearch_type", solver_options.get("snes_linesearch_type")
    )

    # Make divergence explicit so failed solves are caught and retried.
    solver_options.setdefault("snes_error_if_not_converged", True)
    solver_options.setdefault("ksp_error_if_not_converged", True)

    def _reset_relaxation_knobs() -> None:
        solver_options["snes_atol"] = base_atol
        solver_options["snes_rtol"] = base_rtol
        solver_options["ksp_rtol"] = base_ksp_rtol
        solver_options["snes_max_it"] = base_max_it
        if base_linesearch is not None:
            solver_options["snes_linesearch_type"] = base_linesearch

    if phase == "baseline":
        _reset_relaxation_knobs()
        return

    if phase == "max_it":
        _reset_relaxation_knobs()
        solver_options["snes_max_it"] = int(
            min(
                float(recovery.max_it_cap),
                base_max_it * (float(recovery.max_it_growth) ** int(max(1, phase_step))),
            )
        )
        return

    if phase == "anisotropy":
        # This stage just resets relaxation knobs; kappa adjustment is handled at curve level.
        _reset_relaxation_knobs()
        return

    if phase != "tolerance_relax":
        return

    _reset_relaxation_knobs()
    local_step = int(max(1, phase_step))
    solver_options["snes_atol"] = base_atol * (float(recovery.atol_relax_factor) ** local_step)
    solver_options["snes_rtol"] = base_rtol * (float(recovery.rtol_relax_factor) ** local_step)
    solver_options["ksp_rtol"] = base_ksp_rtol * (
        float(recovery.ksp_rtol_relax_factor) ** local_step
    )

    if recovery.line_search_schedule:
        idx = min(local_step - 1, len(recovery.line_search_schedule) - 1)
        solver_options["snes_linesearch_type"] = recovery.line_search_schedule[idx]


def _reduce_kappa_anisotropy(
    kappa: np.ndarray,
    *,
    target_ratio: float,
    blend: float,
) -> np.ndarray:
    """Reduce anisotropy (max/min magnitude ratio) in a kappa vector."""
    arr = np.asarray(kappa, dtype=float).ravel()
    if arr.size < 2:
        return arr.copy()

    mags = np.maximum(np.abs(arr), 1e-14)
    current_ratio = float(np.max(mags) / np.min(mags))
    if current_ratio <= max(1.0, float(target_ratio)):
        return arr.copy()

    geo = float(np.exp(np.mean(np.log(mags))))
    isotropic = np.sign(arr) * geo
    isotropic[np.sign(arr) == 0.0] = geo

    beta = min(max(float(blend), 0.0), 1.0)
    return (1.0 - beta) * arr + beta * isotropic
