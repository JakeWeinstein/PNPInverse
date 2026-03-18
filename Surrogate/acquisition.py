"""Acquisition strategy and sample selection for the ISMO refinement loop.

Decides where to place new PDE evaluations each ISMO iteration using three
complementary strategies:

1. **Optimizer trajectory**: extract endpoints from MultiStart / Cascade
   optimizers plus random neighbors in a ball around each endpoint.
2. **GP uncertainty**: select points with highest posterior variance from
   a large LHS candidate set (requires a fitted GPSurrogateModel).
3. **Space-filling LHS**: global coverage via Latin Hypercube sampling.

A configurable hybrid allocation splits the per-iteration budget across
the three strategies, and a shared de-duplication pipeline removes points
that are too close to existing training data or to each other.

Public API
----------
AcquisitionConfig
    Frozen configuration dataclass.
AcquisitionResult
    Frozen result container.
select_new_samples
    Main entry point (called by the ISMO orchestrator).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from Surrogate.sampling import ParameterBounds, generate_lhs_samples

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcquisitionConfig:
    """Controls how the acquisition budget is allocated.

    Attributes
    ----------
    budget : int
        Total number of new PDE samples to request.
    frac_optimizer : float
        Fraction of budget from optimizer trajectories.
    frac_uncertainty : float
        Fraction of budget from GP posterior variance.
    frac_spacefill : float
        Fraction of budget from space-filling LHS.
    neighborhood_radius_log : float
        Ball radius in normalized log-space for generating neighbors
        around optimizer endpoints.
    n_neighbors_per_candidate : int
        Random neighbors to generate per optimizer endpoint.
    n_uncertainty_candidates : int
        Number of LHS candidates to evaluate GP variance on.
    k0_2_sensitivity_weight : float
        Upweight factor for k0_2 dimension in GP variance scoring.
    min_distance_log : float
        Minimum distance in normalized log-space to existing data.
    min_distance_batch : float
        Minimum distance between points within the same batch.
    spacefill_seed : int
        Seed for space-filling LHS (vary per iteration for diversity).
    seed : int
        General random seed for neighborhood sampling, etc.
    verbose : bool
        If True, log progress and diagnostics.
    """

    budget: int = 30
    frac_optimizer: float = 0.5
    frac_uncertainty: float = 0.3
    frac_spacefill: float = 0.2
    neighborhood_radius_log: float = 0.3
    n_neighbors_per_candidate: int = 2
    n_uncertainty_candidates: int = 5000
    k0_2_sensitivity_weight: float = 2.0
    min_distance_log: float = 0.05
    min_distance_batch: float = 0.08
    spacefill_seed: int | None = None  # None = vary per call for diversity
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        total = self.frac_optimizer + self.frac_uncertainty + self.frac_spacefill
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Strategy fractions must sum to 1.0, got "
                f"{self.frac_optimizer} + {self.frac_uncertainty} + "
                f"{self.frac_spacefill} = {total}"
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcquisitionResult:
    """Output of the acquisition pipeline.

    Attributes
    ----------
    samples : np.ndarray
        (N_acquired, 4) in physical space [k0_1, k0_2, alpha_1, alpha_2].
    strategy_labels : tuple
        Length N_acquired; each element in
        {"optimizer", "uncertainty", "spacefill"}.
    n_requested : int
        The configured budget.
    n_acquired : int
        Actual number of accepted points (may be < budget).
    n_rejected_dedup : int
        Total points removed by distance checks.
    n_rejected_optimizer : int
        Optimizer-strategy points rejected by de-duplication.
    n_rejected_uncertainty : int
        Uncertainty-strategy points rejected by de-duplication.
    n_rejected_spacefill : int
        Space-fill-strategy points rejected by de-duplication.
    gp_variance_used : bool
        Whether GP uncertainty was actually available.
    """

    samples: np.ndarray
    strategy_labels: tuple
    n_requested: int
    n_acquired: int
    n_rejected_dedup: int
    n_rejected_optimizer: int
    n_rejected_uncertainty: int
    n_rejected_spacefill: int
    gp_variance_used: bool


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

def _to_normalized_log(
    params: np.ndarray,
    bounds: ParameterBounds,
) -> np.ndarray:
    """Convert physical-space parameters (N, 4) to normalized log-space [0, 1]^4.

    Dimensions 0, 1 (k0_1, k0_2): log10 then min-max scale to [0, 1].
    Dimension 2 (alpha_1): min-max scale using bounds.alpha_1_range.
    Dimension 3 (alpha_2): min-max scale using bounds.alpha_2_range.

    Parameters
    ----------
    params : np.ndarray of shape (N, 4) or (4,)
        Physical-space parameters [k0_1, k0_2, alpha_1, alpha_2].
    bounds : ParameterBounds
        Parameter space bounds.

    Returns
    -------
    np.ndarray of same shape
        Normalized parameters in [0, 1]^4.
    """
    squeeze = params.ndim == 1
    p = np.atleast_2d(params).copy()

    log_k0_1_lo = np.log10(max(bounds.k0_1_range[0], 1e-30))
    log_k0_1_hi = np.log10(bounds.k0_1_range[1])
    log_k0_2_lo = np.log10(max(bounds.k0_2_range[0], 1e-30))
    log_k0_2_hi = np.log10(bounds.k0_2_range[1])

    # k0_1: log10 then min-max
    log_vals_0 = np.log10(np.maximum(p[:, 0], 1e-30))
    p[:, 0] = (log_vals_0 - log_k0_1_lo) / (log_k0_1_hi - log_k0_1_lo)

    # k0_2: log10 then min-max
    log_vals_1 = np.log10(np.maximum(p[:, 1], 1e-30))
    p[:, 1] = (log_vals_1 - log_k0_2_lo) / (log_k0_2_hi - log_k0_2_lo)

    # alpha_1: min-max
    a1_lo, a1_hi = bounds.alpha_1_range
    p[:, 2] = (p[:, 2] - a1_lo) / (a1_hi - a1_lo)

    # alpha_2: min-max
    a2_lo, a2_hi = bounds.alpha_2_range
    p[:, 3] = (p[:, 3] - a2_lo) / (a2_hi - a2_lo)

    return p[0] if squeeze else p


def _from_normalized_log(
    params_norm: np.ndarray,
    bounds: ParameterBounds,
) -> np.ndarray:
    """Convert normalized log-space [0, 1]^4 back to physical space.

    Inverse of :func:`_to_normalized_log`.

    Parameters
    ----------
    params_norm : np.ndarray of shape (N, 4) or (4,)
        Normalized parameters in [0, 1]^4.
    bounds : ParameterBounds
        Parameter space bounds.

    Returns
    -------
    np.ndarray of same shape
        Physical-space parameters [k0_1, k0_2, alpha_1, alpha_2].
    """
    squeeze = params_norm.ndim == 1
    p = np.atleast_2d(params_norm).copy()

    log_k0_1_lo = np.log10(max(bounds.k0_1_range[0], 1e-30))
    log_k0_1_hi = np.log10(bounds.k0_1_range[1])
    log_k0_2_lo = np.log10(max(bounds.k0_2_range[0], 1e-30))
    log_k0_2_hi = np.log10(bounds.k0_2_range[1])

    # k0_1: denorm then 10^x
    p[:, 0] = 10.0 ** (p[:, 0] * (log_k0_1_hi - log_k0_1_lo) + log_k0_1_lo)

    # k0_2: denorm then 10^x
    p[:, 1] = 10.0 ** (p[:, 1] * (log_k0_2_hi - log_k0_2_lo) + log_k0_2_lo)

    # alpha_1: denorm
    a1_lo, a1_hi = bounds.alpha_1_range
    p[:, 2] = p[:, 2] * (a1_hi - a1_lo) + a1_lo

    # alpha_2: denorm
    a2_lo, a2_hi = bounds.alpha_2_range
    p[:, 3] = p[:, 3] * (a2_hi - a2_lo) + a2_lo

    return p[0] if squeeze else p


# ---------------------------------------------------------------------------
# Distance and de-duplication
# ---------------------------------------------------------------------------

def _min_distance_to_set(
    candidate: np.ndarray,
    existing: np.ndarray,
) -> float:
    """Euclidean distance from candidate to nearest point in existing set.

    Parameters
    ----------
    candidate : np.ndarray of shape (4,)
        Single point in normalized log-space.
    existing : np.ndarray of shape (M, 4)
        Set of points in normalized log-space.

    Returns
    -------
    float
        Minimum Euclidean distance, or inf if existing is empty.
    """
    if existing.shape[0] == 0:
        return float("inf")
    diffs = existing - candidate[np.newaxis, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.min(dists))


def _deduplicate_candidates(
    candidates: np.ndarray,
    labels: list,
    existing_data: np.ndarray,
    bounds: ParameterBounds,
    min_dist_to_existing: float,
    min_dist_within_batch: float,
) -> Tuple[np.ndarray, list, int, dict]:
    """Remove candidates too close to existing data or to each other.

    Algorithm:
    1. Convert all points to normalized log-space.
    2. For each candidate (in order), reject if:
       a. min distance to existing_data < min_dist_to_existing, OR
       b. min distance to already-accepted candidates < min_dist_within_batch.
    3. Return accepted candidates in physical space, preserving input order.

    Parameters
    ----------
    candidates : np.ndarray of shape (K, 4)
        Candidate points in physical space.
    labels : list of str
        Strategy label for each candidate (length K).
    existing_data : np.ndarray of shape (N_train, 4)
        Existing training points in physical space.
    bounds : ParameterBounds
        Parameter space bounds.
    min_dist_to_existing : float
        Minimum distance in normalized log-space to existing data.
    min_dist_within_batch : float
        Minimum distance in normalized log-space between accepted candidates.

    Returns
    -------
    (accepted, accepted_labels, n_rejected, per_strategy_rejections)
        accepted: np.ndarray (N_accepted, 4) in physical space.
        accepted_labels: list of str.
        n_rejected: int.
        per_strategy_rejections: dict mapping strategy name to rejection count.
    """
    if candidates.shape[0] == 0:
        return candidates, [], 0, {}

    cand_norm = _to_normalized_log(candidates, bounds)
    exist_norm = _to_normalized_log(existing_data, bounds) if existing_data.shape[0] > 0 else np.empty((0, 4))

    accepted_norm = []
    accepted_phys = []
    accepted_labels_out = []
    n_rejected = 0
    per_strategy_rejections: dict = {}

    for i in range(candidates.shape[0]):
        c_norm = cand_norm[i]

        # Check distance to existing training data
        dist_existing = _min_distance_to_set(c_norm, exist_norm)
        if dist_existing < min_dist_to_existing:
            n_rejected += 1
            lbl = labels[i]
            per_strategy_rejections[lbl] = per_strategy_rejections.get(lbl, 0) + 1
            continue

        # Check distance to already-accepted batch points
        if len(accepted_norm) > 0:
            accepted_arr = np.array(accepted_norm)
            dist_batch = _min_distance_to_set(c_norm, accepted_arr)
            if dist_batch < min_dist_within_batch:
                n_rejected += 1
                lbl = labels[i]
                per_strategy_rejections[lbl] = per_strategy_rejections.get(lbl, 0) + 1
                continue

        accepted_norm.append(c_norm)
        accepted_phys.append(candidates[i])
        accepted_labels_out.append(labels[i])

    if len(accepted_phys) == 0:
        return np.empty((0, 4)), [], n_rejected, per_strategy_rejections

    return (
        np.array(accepted_phys),
        accepted_labels_out,
        n_rejected,
        per_strategy_rejections,
    )


# ---------------------------------------------------------------------------
# Strategy 1: Optimizer trajectory acquisition
# ---------------------------------------------------------------------------

def _acquire_optimizer_trajectory(
    multistart_result,
    cascade_result,
    bounds: ParameterBounds,
    n_target: int,
    neighborhood_radius: float,
    n_neighbors_per_candidate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Extract candidate points from optimizer outputs + neighborhood ball.

    Algorithm:
    1. Collect unique optimizer endpoints from MultiStartResult.candidates
       and CascadeResult.pass_results.
    2. For each endpoint, generate n_neighbors_per_candidate points by
       sampling uniformly in a ball of radius neighborhood_radius in
       normalized log-space, then converting back to physical space.
    3. Return up to n_target points (endpoints + neighbors), prioritizing
       the lowest-loss endpoints first.

    Parameters
    ----------
    multistart_result : MultiStartResult or None
    cascade_result : CascadeResult or None
    bounds : ParameterBounds
    n_target : int
        Maximum number of points to return.
    neighborhood_radius : float
        Ball radius in normalized log-space.
    n_neighbors_per_candidate : int
        Neighbors to generate per endpoint.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (<= n_target, 4) in physical space.
    """
    # Collect endpoints with losses for sorting
    endpoints: list = []  # list of (params_array, loss)

    if multistart_result is not None:
        for c in multistart_result.candidates:
            params = np.array([c.k0_1, c.k0_2, c.alpha_1, c.alpha_2])
            endpoints.append((params, c.polished_loss))

    if cascade_result is not None:
        for pr in cascade_result.pass_results:
            params = np.array([pr.k0_1, pr.k0_2, pr.alpha_1, pr.alpha_2])
            endpoints.append((params, pr.loss))

    if len(endpoints) == 0:
        return np.empty((0, 4))

    # Sort by loss (best first)
    endpoints.sort(key=lambda x: x[1])

    # De-duplicate endpoints in normalized log-space (within a tight tolerance)
    dedup_endpoints = []
    dedup_norm = []
    for params, loss in endpoints:
        p_norm = _to_normalized_log(params, bounds)
        if len(dedup_norm) > 0:
            dist = _min_distance_to_set(p_norm, np.array(dedup_norm))
            if dist < 0.01:  # very close -> skip
                continue
        dedup_endpoints.append(params)
        dedup_norm.append(p_norm)

    # Generate neighbors around each endpoint
    all_points = []
    for params in dedup_endpoints:
        all_points.append(params)
        p_norm = _to_normalized_log(params, bounds)

        for _ in range(n_neighbors_per_candidate):
            # Sample direction uniformly on the 4-sphere
            direction = rng.standard_normal(4)
            direction /= np.linalg.norm(direction)
            # Sample radius uniformly in the ball
            radius = neighborhood_radius * rng.uniform() ** (1.0 / 4.0)
            neighbor_norm = p_norm + direction * radius
            # Clamp to [0, 1]^4
            neighbor_norm = np.clip(neighbor_norm, 0.0, 1.0)
            neighbor_phys = _from_normalized_log(neighbor_norm, bounds)
            all_points.append(neighbor_phys)

        if len(all_points) >= n_target:
            break

    result = np.array(all_points[:n_target])
    return result


# ---------------------------------------------------------------------------
# Strategy 2: GP uncertainty acquisition
# ---------------------------------------------------------------------------

def _acquire_uncertainty(
    gp_model,
    bounds: ParameterBounds,
    n_target: int,
    n_candidates: int,
    k0_2_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select points with highest GP posterior variance.

    Parameters
    ----------
    gp_model : GPSurrogateModel or None
    bounds : ParameterBounds
    n_target : int
    n_candidates : int
        Number of LHS candidates to evaluate.
    k0_2_weight : float
        Upweight factor for k0_2 sensitivity.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray of shape (<= n_target, 4) in physical space, or empty.
    """
    if gp_model is None:
        return np.empty((0, 4))

    # Check if fitted
    if not getattr(gp_model, "is_fitted", False):
        return np.empty((0, 4))

    # Generate candidate points
    seed_val = int(rng.integers(0, 2**31))
    candidates = generate_lhs_samples(
        bounds, n_candidates, seed=seed_val, log_space_k0=True,
    )

    # Evaluate GP posterior uncertainty
    try:
        pred = gp_model.predict_batch_with_uncertainty(candidates)
    except Exception as exc:
        logger.warning("GP uncertainty prediction failed: %s", exc)
        return np.empty((0, 4))

    cd_std = pred["current_density_std"]   # (M, n_eta)
    pc_std = pred["peroxide_current_std"]  # (M, n_eta)

    # Aggregate uncertainty: mean std across voltage points
    sigma_cd = np.mean(cd_std, axis=1)  # (M,)
    sigma_pc = np.mean(pc_std, axis=1)  # (M,)
    scores = sigma_cd + sigma_pc

    # k0_2 sensitivity weighting: upweight candidates that are further
    # from the mean k0_2 value (in log-space), encouraging exploration of
    # extreme k0_2 regions where the surrogate is likely less accurate.
    if k0_2_weight > 0:
        k0_2_vals = candidates[:, 1]  # k0_2 is column index 1
        log_k0_2 = np.log10(np.maximum(k0_2_vals, 1e-30))
        k0_2_range = log_k0_2.max() - log_k0_2.min()
        if k0_2_range > 0:
            k0_2_sensitivity = 1.0 + k0_2_weight * (
                np.abs(log_k0_2 - log_k0_2.mean()) / k0_2_range
            )
        else:
            k0_2_sensitivity = np.ones(n_candidates)
    else:
        k0_2_sensitivity = np.ones(n_candidates)
    scores = scores * k0_2_sensitivity

    # Select top n_target
    top_idx = np.argsort(scores)[::-1][:n_target]
    return candidates[top_idx]


# ---------------------------------------------------------------------------
# Strategy 3: Space-filling LHS acquisition
# ---------------------------------------------------------------------------

def _acquire_spacefill(
    bounds: ParameterBounds,
    n_target: int,
    seed: int | None,
) -> np.ndarray:
    """Generate space-filling LHS samples across the full parameter space.

    Uses the existing ``generate_lhs_samples()`` from ``Surrogate.sampling``
    with ``log_space_k0=True``.

    Parameters
    ----------
    bounds : ParameterBounds
    n_target : int
    seed : int or None
        If None, a random seed is used for diversity across calls.

    Returns
    -------
    np.ndarray of shape (n_target, 4) in physical space.
    """
    if seed is None:
        rng = np.random.default_rng()  # no seed → random
        seed = int(rng.integers(0, 2**31))
    return generate_lhs_samples(bounds, n_target, seed=seed, log_space_k0=True)


# ---------------------------------------------------------------------------
# Clamping utility
# ---------------------------------------------------------------------------

def _clamp_to_bounds(params: np.ndarray, bounds: ParameterBounds) -> np.ndarray:
    """Clamp parameter array to stay within bounds.

    Parameters
    ----------
    params : np.ndarray of shape (N, 4)
    bounds : ParameterBounds

    Returns
    -------
    np.ndarray of shape (N, 4)
        Clamped copy.
    """
    out = params.copy()
    out[:, 0] = np.clip(out[:, 0], bounds.k0_1_range[0], bounds.k0_1_range[1])
    out[:, 1] = np.clip(out[:, 1], bounds.k0_2_range[0], bounds.k0_2_range[1])
    out[:, 2] = np.clip(out[:, 2], bounds.alpha_1_range[0], bounds.alpha_1_range[1])
    out[:, 3] = np.clip(out[:, 3], bounds.alpha_2_range[0], bounds.alpha_2_range[1])
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_new_samples(
    existing_data: np.ndarray,
    bounds: ParameterBounds,
    config: AcquisitionConfig,
    multistart_result=None,
    cascade_result=None,
    gp_model=None,
) -> AcquisitionResult:
    """Select new PDE evaluation points using the hybrid acquisition strategy.

    This is the main entry point called by the ISMO loop orchestrator.

    Algorithm:
    1. Compute per-strategy budgets from config fractions.
    2. If gp_model is None, redistribute uncertainty budget.
    3. If both multistart_result and cascade_result are None, redistribute
       optimizer budget to space-filling.
    4. Generate candidates from each strategy.
    5. Concatenate with strategy labels (optimizer first, then uncertainty,
       then space-filling) for priority-ordered de-duplication.
    6. Run de-duplication against existing data and within the batch.
    7. Return AcquisitionResult.

    Parameters
    ----------
    existing_data : np.ndarray (N_train, 4)
        Current training parameters in physical space.
    bounds : ParameterBounds
        Parameter space bounds.
    config : AcquisitionConfig
        Acquisition configuration.
    multistart_result : MultiStartResult, optional
        Most recent multistart optimizer output.
    cascade_result : CascadeResult, optional
        Most recent cascade optimizer output.
    gp_model : GPSurrogateModel, optional
        Fitted GP model for uncertainty quantification.

    Returns
    -------
    AcquisitionResult
        New sample points and metadata.
    """
    rng = np.random.default_rng(config.seed)

    # --- Step 1: Compute per-strategy budgets ---
    n_opt = round(config.budget * config.frac_optimizer)
    n_unc = round(config.budget * config.frac_uncertainty)
    n_lhs = config.budget - n_opt - n_unc  # remainder to space-filling

    gp_variance_used = gp_model is not None and getattr(gp_model, "is_fitted", False)

    # --- Step 2: Redistribute if GP unavailable ---
    if not gp_variance_used and n_unc > 0:
        orig_n_unc = n_unc
        n_opt += n_unc // 2
        n_lhs += n_unc - n_unc // 2
        n_unc = 0
        if config.verbose:
            logger.info(
                "No GP model available; redistributing uncertainty budget "
                "(%d -> optimizer +%d, spacefill +%d)",
                orig_n_unc, orig_n_unc // 2, orig_n_unc - orig_n_unc // 2,
            )

    # --- Step 3: Redistribute if no optimizer results ---
    has_optimizer = multistart_result is not None or cascade_result is not None
    if not has_optimizer and n_opt > 0:
        if config.verbose:
            logger.info(
                "No optimizer results available; redistributing optimizer "
                "budget (%d) to space-filling",
                n_opt,
            )
        n_lhs += n_opt
        n_opt = 0

    # --- Step 4: Generate candidates from each strategy ---
    opt_candidates = np.empty((0, 4))
    if n_opt > 0:
        opt_candidates = _acquire_optimizer_trajectory(
            multistart_result=multistart_result,
            cascade_result=cascade_result,
            bounds=bounds,
            n_target=n_opt,
            neighborhood_radius=config.neighborhood_radius_log,
            n_neighbors_per_candidate=config.n_neighbors_per_candidate,
            rng=rng,
        )
        if opt_candidates.shape[0] > 0:
            opt_candidates = _clamp_to_bounds(opt_candidates, bounds)

    unc_candidates = np.empty((0, 4))
    if n_unc > 0:
        unc_candidates = _acquire_uncertainty(
            gp_model=gp_model,
            bounds=bounds,
            n_target=n_unc,
            n_candidates=config.n_uncertainty_candidates,
            k0_2_weight=config.k0_2_sensitivity_weight,
            rng=rng,
        )

    lhs_candidates = np.empty((0, 4))
    if n_lhs > 0:
        lhs_candidates = _acquire_spacefill(
            bounds=bounds,
            n_target=n_lhs,
            seed=config.spacefill_seed,
        )

    # --- Step 5: Concatenate with labels (priority order) ---
    all_candidates = []
    all_labels = []

    if opt_candidates.shape[0] > 0:
        all_candidates.append(opt_candidates)
        all_labels.extend(["optimizer"] * opt_candidates.shape[0])

    if unc_candidates.shape[0] > 0:
        all_candidates.append(unc_candidates)
        all_labels.extend(["uncertainty"] * unc_candidates.shape[0])

    if lhs_candidates.shape[0] > 0:
        all_candidates.append(lhs_candidates)
        all_labels.extend(["spacefill"] * lhs_candidates.shape[0])

    if len(all_candidates) == 0:
        return AcquisitionResult(
            samples=np.empty((0, 4)),
            strategy_labels=(),
            n_requested=config.budget,
            n_acquired=0,
            n_rejected_dedup=0,
            n_rejected_optimizer=0,
            n_rejected_uncertainty=0,
            n_rejected_spacefill=0,
            gp_variance_used=gp_variance_used,
        )

    pool = np.concatenate(all_candidates, axis=0)

    # --- Step 6: De-duplication ---
    accepted, accepted_labels, n_rejected, per_strat_rej = _deduplicate_candidates(
        candidates=pool,
        labels=all_labels,
        existing_data=existing_data,
        bounds=bounds,
        min_dist_to_existing=config.min_distance_log,
        min_dist_within_batch=config.min_distance_batch,
    )

    n_acquired = accepted.shape[0]

    # Warn if yield is low
    if n_acquired < config.budget * 0.5:
        warnings.warn(
            f"Acquisition yield is low: {n_acquired}/{config.budget} points "
            f"accepted ({n_rejected} rejected by de-duplication). "
            f"The parameter space may be saturated.",
            stacklevel=2,
        )

    if config.verbose:
        logger.info(
            "Acquisition: %d/%d accepted (%d rejected). "
            "Strategies: opt=%d, unc=%d, lhs=%d",
            n_acquired, config.budget, n_rejected,
            sum(1 for l in accepted_labels if l == "optimizer"),
            sum(1 for l in accepted_labels if l == "uncertainty"),
            sum(1 for l in accepted_labels if l == "spacefill"),
        )

    return AcquisitionResult(
        samples=accepted,
        strategy_labels=tuple(accepted_labels),
        n_requested=config.budget,
        n_acquired=n_acquired,
        n_rejected_dedup=n_rejected,
        n_rejected_optimizer=per_strat_rej.get("optimizer", 0),
        n_rejected_uncertainty=per_strat_rej.get("uncertainty", 0),
        n_rejected_spacefill=per_strat_rej.get("spacefill", 0),
        gp_variance_used=gp_variance_used,
    )
