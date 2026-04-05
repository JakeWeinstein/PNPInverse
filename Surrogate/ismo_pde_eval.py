"""PDE evaluation and data integration for the ISMO refinement loop.

Bridges the acquisition strategy (4-03) and surrogate retraining (4-05).
Given candidate parameter points, evaluates them with the true PDE solver,
compares surrogate predictions against PDE truth, merges results into the
training dataset with provenance tracking, and reports quality diagnostics.

All Firedrake imports are deferred inside functions (matching training.py).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Surrogate.sampling import ParameterBounds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses (all frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDESolverBundle:
    """Immutable bundle of everything needed to run PDE evaluations.

    Created via :func:`make_standard_pde_bundle`.  The factory defers all
    Firedrake imports; this dataclass itself has no Firedrake dependency.
    """

    base_solver_params: Any  # SolverParams (11-element list)
    steady: Any  # SteadyStateConfig
    observable_scale: float  # -I_SCALE
    mesh_params: Tuple[int, int, float]  # (Nx, Ny, beta) for parallel workers
    phi_applied_values: np.ndarray  # voltage grid (loaded from training data)
    min_converged_fraction: float  # default 0.8
    max_batch_size: int  # safety bound on candidate batch size
    cd_reference_range: float  # full-dataset CD ptp for NRMSE normalization
    pc_reference_range: float  # full-dataset PC ptp for NRMSE normalization


@dataclass(frozen=True)
class PDEEvalResult:
    """Results from evaluating candidate parameters with the PDE solver."""

    candidate_params: np.ndarray  # (B, 4) input parameters
    current_density: np.ndarray  # (B_valid, n_eta) CD curves for valid solves
    peroxide_current: np.ndarray  # (B_valid, n_eta) PC curves for valid solves
    valid_mask: np.ndarray  # (B,) bool: which candidates converged
    timings: np.ndarray  # (B,) seconds per solve
    n_valid: int
    n_failed: int
    valid_params: np.ndarray  # (B_valid, 4) parameters for valid solves


@dataclass(frozen=True)
class SurrogatePDEComparison:
    """Per-candidate comparison of surrogate predictions vs PDE truth."""

    candidate_params: np.ndarray  # (B_valid, 4)
    cd_nrmse_per_candidate: np.ndarray  # (B_valid,)
    pc_nrmse_per_candidate: np.ndarray  # (B_valid,)
    cd_rmse_per_candidate: np.ndarray  # (B_valid,)
    pc_rmse_per_candidate: np.ndarray  # (B_valid,)
    cd_max_error: float  # worst-case CD NRMSE
    pc_max_error: float  # worst-case PC NRMSE
    cd_mean_nrmse: float  # mean CD NRMSE across candidates
    pc_mean_nrmse: float  # mean PC NRMSE across candidates
    is_converged: bool  # True if max(cd_mean, pc_mean) < threshold


@dataclass(frozen=True)
class AugmentedDataset:
    """Result of merging new PDE data into existing training set."""

    output_path: str  # where the augmented .npz was saved
    n_original: int  # samples from original dataset
    n_new: int  # valid new samples added
    n_total: int  # n_original + n_new
    provenance: np.ndarray  # (n_total,) string array: source tags


@dataclass(frozen=True)
class QualityReport:
    """Quality check results for a batch of PDE evaluations."""

    n_candidates: int
    n_converged: int
    n_nan_detected: int
    n_extreme_values: int
    n_bounds_violations: int
    n_passed_all_checks: int
    flagged_indices: np.ndarray  # indices of problematic samples
    flags: List[str]  # human-readable flag descriptions
    passed: bool  # True if n_passed == n_converged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_standard_pde_bundle(
    *,
    training_data_path: str = None,
    Nx: int = 8,
    Ny: int = 200,
    beta: float = 3.0,
    min_converged_fraction: float = 0.8,
    max_batch_size: int = 100,
) -> PDESolverBundle:
    """Create the standard PDE solver configuration bundle.

    Loads ``phi_applied`` from the existing training data file to guarantee
    exact match with the voltage grid used for the original dataset.  Also
    computes CD/PC reference ranges from the full training set for consistent
    NRMSE normalization.

    All Firedrake / solver imports are deferred inside this function body.

    Parameters
    ----------
    training_data_path : str
        Path to the existing training data ``.npz`` file.  Must contain
        ``phi_applied``, ``current_density``, and ``peroxide_current`` keys.
    Nx, Ny : int
        Mesh dimensions.
    beta : float
        Mesh grading parameter.
    min_converged_fraction : float
        Minimum fraction of voltage points that must converge.
    max_batch_size : int
        Safety bound on the number of candidates per batch.
    """
    # Resolve default training data path relative to package root
    _package_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_package_dir)
    if training_data_path is None:
        training_data_path = os.path.join(
            _repo_root, "data", "surrogate_models", "training_data_merged.npz"
        )

    # Deferred imports -- Firedrake must not be imported at module level
    import sys

    repo_root = _repo_root
    scripts_dir = os.path.join(repo_root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from _bv_common import (  # type: ignore[import-untyped]
        I_SCALE,
        FOUR_SPECIES_CHARGED,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    from Forward.steady_state import SteadyStateConfig

    base_solver_params = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.5,
        t_end=50.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
    )
    steady = SteadyStateConfig(relative_tolerance=1e-4)
    observable_scale = -I_SCALE

    # Load phi_applied and reference ranges from existing training data
    data = np.load(training_data_path, allow_pickle=True)
    phi_applied_values = data["phi_applied"]
    cd_reference_range = float(np.ptp(data["current_density"]))
    pc_reference_range = float(np.ptp(data["peroxide_current"]))

    return PDESolverBundle(
        base_solver_params=base_solver_params,
        steady=steady,
        observable_scale=observable_scale,
        mesh_params=(Nx, Ny, beta),
        phi_applied_values=phi_applied_values,
        min_converged_fraction=min_converged_fraction,
        max_batch_size=max_batch_size,
        cd_reference_range=cd_reference_range,
        pc_reference_range=pc_reference_range,
    )


# ---------------------------------------------------------------------------
# PDE evaluation
# ---------------------------------------------------------------------------

# Threshold for switching from sequential to parallel evaluation.
_PARALLEL_THRESHOLD = 8


def evaluate_candidates_with_pde(
    candidate_params: np.ndarray,
    pde_bundle: PDESolverBundle,
    *,
    n_workers: int = 4,
    output_path: Optional[str] = None,
) -> PDEEvalResult:
    """Evaluate candidate parameter points with the true PDE solver.

    Delegates to :func:`Surrogate.training.generate_training_dataset` (for
    small batches) or :func:`Surrogate.training.generate_training_dataset_parallel`
    (for larger batches).

    Parameters
    ----------
    candidate_params : np.ndarray of shape (B, 4)
        Parameter samples ``[k0_1, k0_2, alpha_1, alpha_2]``.
    pde_bundle : PDESolverBundle
        Solver configuration from :func:`make_standard_pde_bundle`.
    n_workers : int
        Number of parallel worker processes (only used for B > 8).
    output_path : str or None
        If provided, save raw PDE results to this path for debugging.

    Returns
    -------
    PDEEvalResult
        Structured results with valid/failed separation.
    """
    from Surrogate.training import (
        generate_training_dataset,
        generate_training_dataset_parallel,
    )

    if candidate_params.ndim != 2 or candidate_params.shape[1] != 4:
        raise ValueError(
            f"candidate_params must have shape (B, 4), got {candidate_params.shape}"
        )
    B = candidate_params.shape[0]
    if B > pde_bundle.max_batch_size:
        raise ValueError(
            f"Batch size {B} exceeds max_batch_size={pde_bundle.max_batch_size}. "
            f"This safety bound limits PDE budget per iteration."
        )
    if B == 0:
        return PDEEvalResult(
            candidate_params=candidate_params,
            current_density=np.empty((0, len(pde_bundle.phi_applied_values))),
            peroxide_current=np.empty((0, len(pde_bundle.phi_applied_values))),
            valid_mask=np.array([], dtype=bool),
            timings=np.array([], dtype=float),
            n_valid=0,
            n_failed=0,
            valid_params=np.empty((0, 4)),
        )

    # Choose a temp output path if none provided
    save_path = output_path or f"/tmp/ismo_pde_eval_{int(time.time())}.npz"

    t0 = time.time()

    if B <= _PARALLEL_THRESHOLD:
        # Sequential -- avoids process spawn overhead for small batches
        # Need a mesh for sequential mode
        from Forward.bv_solver import make_graded_rectangle_mesh

        Nx, Ny, beta = pde_bundle.mesh_params
        mesh = make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta)

        result = generate_training_dataset(
            candidate_params,
            phi_applied_values=pde_bundle.phi_applied_values,
            base_solver_params=pde_bundle.base_solver_params,
            steady=pde_bundle.steady,
            observable_scale=pde_bundle.observable_scale,
            mesh=mesh,
            output_path=save_path,
            min_converged_fraction=pde_bundle.min_converged_fraction,
            verbose=True,
        )
    else:
        # Parallel with warm-start chains
        result = generate_training_dataset_parallel(
            candidate_params,
            phi_applied_values=pde_bundle.phi_applied_values,
            base_solver_params=pde_bundle.base_solver_params,
            steady=pde_bundle.steady,
            observable_scale=pde_bundle.observable_scale,
            mesh_params=pde_bundle.mesh_params,
            output_path=save_path,
            n_workers=n_workers,
            min_converged_fraction=pde_bundle.min_converged_fraction,
            verbose=True,
        )

    wall_time = time.time() - t0

    # Build valid mask: generate_training_dataset returns only valid rows in
    # 'parameters', so we match back to original candidate_params.
    valid_params = result["parameters"]
    n_valid = result["n_valid"]
    n_failed = result["n_failed"]

    # Reconstruct per-candidate valid mask.  The training functions filter out
    # failed samples, so we compare the returned parameters against inputs.
    # Use the all_converged array from the saved npz for exact mask.
    if os.path.exists(save_path + ".npz") or os.path.exists(save_path):
        actual_path = save_path if os.path.exists(save_path) else save_path + ".npz"
        saved = np.load(actual_path, allow_pickle=True)
        if "all_converged" in saved:
            valid_mask = saved["all_converged"].astype(bool)
        else:
            # Fallback: mark first n_valid as valid
            valid_mask = np.zeros(B, dtype=bool)
            valid_mask[:n_valid] = True
    else:
        valid_mask = np.zeros(B, dtype=bool)
        valid_mask[:n_valid] = True

    timings = result.get("timings", np.zeros(B, dtype=float))

    # Log timing summary
    logger.info(
        "PDE evaluation complete: %d/%d valid (%.1f%%), "
        "wall=%.1fs, mean=%.1fs/sample, failed=%d",
        n_valid,
        B,
        100.0 * n_valid / B if B > 0 else 0.0,
        wall_time,
        wall_time / B if B > 0 else 0.0,
        n_failed,
    )

    return PDEEvalResult(
        candidate_params=candidate_params,
        current_density=result["current_density"],
        peroxide_current=result["peroxide_current"],
        valid_mask=valid_mask,
        timings=timings,
        n_valid=n_valid,
        n_failed=n_failed,
        valid_params=valid_params,
    )


# ---------------------------------------------------------------------------
# Surrogate-PDE comparison
# ---------------------------------------------------------------------------


def _compute_nrmse_with_reference_range(
    pred: np.ndarray,
    truth: np.ndarray,
    reference_range: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample NRMSE normalised by each sample's own ptp range.

    Despite the function name, ``reference_range`` is **not** used as the
    NRMSE denominator.  Instead, each sample is normalised by its own
    ``ptp(truth[i])``.  The ``reference_range`` is used only to derive a
    floor (1% of the training-set ptp, minimum 1e-12) that prevents
    division by near-zero ranges on flat-response samples.

    Parameters
    ----------
    pred, truth : np.ndarray of shape (N, n_eta)
    reference_range : float
        The ptp range from the full training set.  Used solely to compute
        a minimum denominator floor (``max(reference_range * 0.01, 1e-12)``).

    Returns
    -------
    nrmse_per_sample : np.ndarray of shape (N,)
    rmse_per_sample : np.ndarray of shape (N,)
    """
    diff = pred - truth
    rmse_per_sample = np.sqrt(np.mean(diff ** 2, axis=1))
    # Floor prevents division-by-zero for samples with nearly flat truth.
    denominator_floor = max(reference_range * 0.01, 1e-12)

    nrmse_per_sample = np.zeros(len(rmse_per_sample), dtype=float)
    for i in range(len(rmse_per_sample)):
        sample_range = max(float(np.ptp(truth[i])), denominator_floor)
        nrmse_per_sample[i] = rmse_per_sample[i] / sample_range

    return nrmse_per_sample, rmse_per_sample


def compare_surrogate_vs_pde(
    surrogate: Any,
    pde_result: PDEEvalResult,
    convergence_threshold: float = 0.02,
    *,
    cd_reference_range: Optional[float] = None,
    pc_reference_range: Optional[float] = None,
) -> SurrogatePDEComparison:
    """Compare surrogate predictions against PDE truth at candidate points.

    Uses a fixed reference range from the full training set for NRMSE
    normalization (review finding #2), ensuring the convergence threshold
    is comparable across batches of different size and composition.

    Parameters
    ----------
    surrogate : object
        Any model with a ``predict_batch(params) -> dict`` method returning
        ``current_density`` and ``peroxide_current`` arrays.
    pde_result : PDEEvalResult
        PDE solver results from :func:`evaluate_candidates_with_pde`.
    convergence_threshold : float
        ISMO converges when both ``cd_mean_nrmse`` and ``pc_mean_nrmse``
        are below this threshold.
    cd_reference_range, pc_reference_range : float or None
        Fixed reference ranges for NRMSE normalization.  If None, falls
        back to per-batch ptp (less robust for small batches).

    Returns
    -------
    SurrogatePDEComparison
    """
    # Handle all-candidates-failed case (review finding #4)
    if pde_result.n_valid == 0:
        empty = np.array([], dtype=float)
        return SurrogatePDEComparison(
            candidate_params=np.empty((0, 4)),
            cd_nrmse_per_candidate=empty,
            pc_nrmse_per_candidate=empty,
            cd_rmse_per_candidate=empty,
            pc_rmse_per_candidate=empty,
            cd_max_error=float("inf"),
            pc_max_error=float("inf"),
            cd_mean_nrmse=float("inf"),
            pc_mean_nrmse=float("inf"),
            is_converged=False,
        )

    pred = surrogate.predict_batch(pde_result.valid_params)
    pred_cd = pred["current_density"]
    pred_pc = pred["peroxide_current"]

    truth_cd = pde_result.current_density
    truth_pc = pde_result.peroxide_current

    # Use fixed reference range if provided; otherwise fall back to batch ptp
    cd_ref = cd_reference_range if cd_reference_range is not None else float(np.ptp(truth_cd))
    pc_ref = pc_reference_range if pc_reference_range is not None else float(np.ptp(truth_pc))

    cd_nrmse, cd_rmse = _compute_nrmse_with_reference_range(pred_cd, truth_cd, cd_ref)
    pc_nrmse, pc_rmse = _compute_nrmse_with_reference_range(pred_pc, truth_pc, pc_ref)

    cd_mean_nrmse = float(np.mean(cd_nrmse))
    pc_mean_nrmse = float(np.mean(pc_nrmse))

    is_converged = (cd_mean_nrmse < convergence_threshold) and (
        pc_mean_nrmse < convergence_threshold
    )

    return SurrogatePDEComparison(
        candidate_params=pde_result.valid_params,
        cd_nrmse_per_candidate=cd_nrmse,
        pc_nrmse_per_candidate=pc_nrmse,
        cd_rmse_per_candidate=cd_rmse,
        pc_rmse_per_candidate=pc_rmse,
        cd_max_error=float(np.max(cd_nrmse)),
        pc_max_error=float(np.max(pc_nrmse)),
        cd_mean_nrmse=cd_mean_nrmse,
        pc_mean_nrmse=pc_mean_nrmse,
        is_converged=is_converged,
    )


# ---------------------------------------------------------------------------
# Data integration
# ---------------------------------------------------------------------------

# Core keys expected in every training data .npz
_CORE_KEYS = {"parameters", "current_density", "peroxide_current", "phi_applied"}


def integrate_new_data(
    pde_result: PDEEvalResult,
    existing_data_path: str,
    output_path: str,
    *,
    iteration_tag: str = "ismo_iter_1",
    comparison: Optional[SurrogatePDEComparison] = None,
) -> AugmentedDataset:
    """Merge new PDE evaluation results into the existing training dataset.

    All ISMO-acquired points go into the training set only; existing test
    indices (which reference the first ``n_original`` rows) remain valid
    (review finding #3).

    The original ``existing_data_path`` is never modified.  Output is written
    to ``output_path`` with versioned naming (e.g.
    ``training_data_ismo_iter1.npz``).

    Extra metadata keys in the existing ``.npz`` (e.g. ``n_existing``,
    ``n_gapfill``) are carried forward to the output file (review finding #6).

    Parameters
    ----------
    pde_result : PDEEvalResult
        Results from :func:`evaluate_candidates_with_pde`.
    existing_data_path : str
        Path to the current training data ``.npz``.
    output_path : str
        Path for the augmented output ``.npz``.
    iteration_tag : str
        Provenance tag for the new samples (e.g. ``"ismo_iter_1"``).
    comparison : SurrogatePDEComparison or None
        If provided, per-candidate NRMSE values are stored in provenance.

    Returns
    -------
    AugmentedDataset
    """
    # Handle all-candidates-failed (review finding #4): no-op
    if pde_result.n_valid == 0:
        logger.warning(
            "No valid PDE results to integrate (n_valid=0). "
            "Returning original dataset path."
        )
        existing = np.load(existing_data_path, allow_pickle=True)
        n_original = existing["parameters"].shape[0]
        provenance = np.array(["original"] * n_original)
        return AugmentedDataset(
            output_path=existing_data_path,
            n_original=n_original,
            n_new=0,
            n_total=n_original,
            provenance=provenance,
        )

    # Load existing data
    existing = np.load(existing_data_path, allow_pickle=True)
    old_params = existing["parameters"]
    old_cd = existing["current_density"]
    old_pc = existing["peroxide_current"]
    old_phi = existing["phi_applied"]
    n_original = old_params.shape[0]

    # Validate voltage grid match
    new_n_eta = pde_result.current_density.shape[1]
    if len(old_phi) != new_n_eta:
        raise ValueError(
            f"Voltage grid mismatch: existing has {len(old_phi)} points, "
            f"new data has {new_n_eta}"
        )

    # Concatenate core arrays
    n_new = pde_result.n_valid
    merged_params = np.concatenate([old_params, pde_result.valid_params], axis=0)
    merged_cd = np.concatenate([old_cd, pde_result.current_density], axis=0)
    merged_pc = np.concatenate([old_pc, pde_result.peroxide_current], axis=0)
    n_total = n_original + n_new

    # Build provenance arrays
    provenance_source = np.array(
        ["original"] * n_original + [iteration_tag] * n_new
    )
    provenance_strategy = np.array(
        [""] * n_original + [iteration_tag] * n_new
    )
    provenance_nrmse_cd = np.full(n_total, np.nan, dtype=float)
    provenance_nrmse_pc = np.full(n_total, np.nan, dtype=float)

    if comparison is not None and comparison.cd_nrmse_per_candidate.size == n_new:
        provenance_nrmse_cd[n_original:] = comparison.cd_nrmse_per_candidate
        provenance_nrmse_pc[n_original:] = comparison.pc_nrmse_per_candidate

    # Build save dict, carrying forward extra keys (review finding #6)
    save_dict: Dict[str, Any] = {
        "parameters": merged_params,
        "current_density": merged_cd,
        "peroxide_current": merged_pc,
        "phi_applied": old_phi,
        "provenance_source": provenance_source,
        "provenance_strategy": provenance_strategy,
        "provenance_nrmse_cd": provenance_nrmse_cd,
        "provenance_nrmse_pc": provenance_nrmse_pc,
    }

    # Carry forward any extra keys from the existing file
    for key in existing.files:
        if key not in _CORE_KEYS and key not in save_dict:
            save_dict[key] = existing[key]

    # Add ISMO metadata
    ismo_meta = {
        "iteration_tag": iteration_tag,
        "n_original": int(n_original),
        "n_new": int(n_new),
        "n_total": int(n_total),
    }
    if comparison is not None:
        ismo_meta["cd_mean_nrmse"] = float(comparison.cd_mean_nrmse)
        ismo_meta["pc_mean_nrmse"] = float(comparison.pc_mean_nrmse)
        ismo_meta["is_converged"] = bool(comparison.is_converged)
    save_dict["ismo_metadata"] = np.array(json.dumps(ismo_meta))

    # Save (never overwrite existing)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    logger.info(
        "Integrated %d new samples into dataset: %d -> %d total, saved to %s",
        n_new,
        n_original,
        n_total,
        output_path,
    )

    return AugmentedDataset(
        output_path=output_path,
        n_original=n_original,
        n_new=n_new,
        n_total=n_total,
        provenance=provenance_source,
    )


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def check_pde_quality(
    pde_result: PDEEvalResult,
    bounds: ParameterBounds,
    *,
    extreme_cd_threshold: float = 50.0,
    extreme_pc_threshold: float = 50.0,
) -> QualityReport:
    """Run quality checks on PDE evaluation results.

    Checks applied to each valid PDE result:

    1. **NaN detection** -- any NaN in CD or PC curves.
    2. **Extreme value detection** -- ``|CD| > threshold`` or ``|PC| > threshold``
       at any voltage point (dimensionless units).
    3. **Parameter bounds compliance** -- all 4 parameters within
       ``ParameterBounds`` ranges.

    Samples that fail any check are *flagged* but NOT removed.  The
    :class:`QualityReport` is returned to the orchestrator which decides
    whether to include flagged samples.

    Parameters
    ----------
    pde_result : PDEEvalResult
        Results from :func:`evaluate_candidates_with_pde`.
    bounds : ParameterBounds
        Acceptable parameter ranges.
    extreme_cd_threshold, extreme_pc_threshold : float
        Thresholds for extreme value detection.

    Returns
    -------
    QualityReport
    """
    n_candidates = pde_result.candidate_params.shape[0]
    n_converged = pde_result.n_valid
    flags: List[str] = []
    flagged_set: set = set()
    n_nan = 0
    n_extreme = 0
    n_bounds = 0

    if n_converged == 0:
        return QualityReport(
            n_candidates=n_candidates,
            n_converged=0,
            n_nan_detected=0,
            n_extreme_values=0,
            n_bounds_violations=0,
            n_passed_all_checks=0,
            flagged_indices=np.array([], dtype=int),
            flags=["All candidates failed PDE evaluation"],
            passed=False,
        )

    cd = pde_result.current_density
    pc = pde_result.peroxide_current
    params = pde_result.valid_params

    for i in range(n_converged):
        sample_flagged = False

        # Check 1: NaN detection
        if np.any(np.isnan(cd[i])) or np.any(np.isnan(pc[i])):
            n_nan += 1
            flags.append(f"Sample {i}: NaN detected in CD/PC curves")
            sample_flagged = True

        # Check 2: Extreme values
        if np.any(np.abs(cd[i]) > extreme_cd_threshold):
            n_extreme += 1
            flags.append(
                f"Sample {i}: extreme CD value "
                f"(max |CD|={np.max(np.abs(cd[i])):.2f} > {extreme_cd_threshold})"
            )
            sample_flagged = True
        if np.any(np.abs(pc[i]) > extreme_pc_threshold):
            n_extreme += 1
            flags.append(
                f"Sample {i}: extreme PC value "
                f"(max |PC|={np.max(np.abs(pc[i])):.2f} > {extreme_pc_threshold})"
            )
            sample_flagged = True

        # Check 3: Parameter bounds
        p = params[i]
        violations = []
        if not (bounds.k0_1_range[0] <= p[0] <= bounds.k0_1_range[1]):
            violations.append(f"k0_1={p[0]:.4e} outside {bounds.k0_1_range}")
        if not (bounds.k0_2_range[0] <= p[1] <= bounds.k0_2_range[1]):
            violations.append(f"k0_2={p[1]:.4e} outside {bounds.k0_2_range}")
        if not (bounds.alpha_1_range[0] <= p[2] <= bounds.alpha_1_range[1]):
            violations.append(f"alpha_1={p[2]:.4f} outside {bounds.alpha_1_range}")
        if not (bounds.alpha_2_range[0] <= p[3] <= bounds.alpha_2_range[1]):
            violations.append(f"alpha_2={p[3]:.4f} outside {bounds.alpha_2_range}")
        if violations:
            n_bounds += 1
            flags.append(f"Sample {i}: bounds violation: {'; '.join(violations)}")
            sample_flagged = True

        if sample_flagged:
            flagged_set.add(i)

    n_passed = n_converged - len(flagged_set)
    passed = n_passed == n_converged

    return QualityReport(
        n_candidates=n_candidates,
        n_converged=n_converged,
        n_nan_detected=n_nan,
        n_extreme_values=n_extreme,
        n_bounds_violations=n_bounds,
        n_passed_all_checks=n_passed,
        flagged_indices=np.array(sorted(flagged_set), dtype=int),
        flags=flags,
        passed=passed,
    )
