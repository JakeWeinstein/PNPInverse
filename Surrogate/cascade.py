"""Per-observable inference cascade for multi-reaction BV kinetics.

Exploits the weight-sweep insight that CD-dominant weighting recovers
reaction-1 parameters (k0_1, alpha_1) excellently, while PC-dominant
weighting recovers k0_2 well.  Instead of jointly optimising all 4
parameters at a single weight, the cascade runs three sequential passes:

1. **Pass 1 (CD-dominant):** Optimize all 4 parameters with low
   secondary_weight.  Recovers k0_1 and alpha_1 to ~0.5% and ~2%.
2. **Pass 2 (PC-dominant):** FIX k0_1 and alpha_1 from Pass 1,
   optimise ONLY k0_2 and alpha_2 with high secondary_weight.
3. **Pass 3 (Joint polish, optional):** Fine-tune all 4 parameters
   from the cascade result with moderate weight.

Public API
----------
CascadeConfig
    Frozen configuration dataclass.
CascadePassResult
    Per-pass result snapshot.
CascadeResult
    Final result container.
run_cascade_inference
    Main entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from Surrogate.surrogate_model import BVSurrogateModel
from Surrogate.objectives import _has_autograd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CascadeConfig:
    """Frozen configuration for the per-observable inference cascade.

    Attributes
    ----------
    pass1_weight : float
        secondary_weight for Pass 1 (CD-dominant, low value).
    pass2_weight : float
        secondary_weight for Pass 2 (PC-dominant, high value).
    pass1_maxiter : int
        Maximum L-BFGS-B iterations for Pass 1.
    pass2_maxiter : int
        Maximum L-BFGS-B iterations for Pass 2.
    polish_maxiter : int
        Maximum L-BFGS-B iterations for Pass 3 (joint polish).
    polish_weight : float
        secondary_weight for Pass 3 (joint polish).
    skip_polish : bool
        If True, skip Pass 3 entirely.
    fd_step : float
        Finite difference step size for gradient computation.
    verbose : bool
        If True, print progress during optimisation.
    """

    pass1_weight: float = 0.5
    pass2_weight: float = 2.0
    pass1_maxiter: int = 60
    pass2_maxiter: int = 60
    polish_maxiter: int = 30
    polish_weight: float = 1.0
    skip_polish: bool = False
    fd_step: float = 1e-5
    verbose: bool = True


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CascadePassResult:
    """Snapshot of a single cascade pass.

    Attributes
    ----------
    pass_name : str
        Human-readable name for the pass.
    k0_1, k0_2 : float
        Recovered k0 values after this pass.
    alpha_1, alpha_2 : float
        Recovered alpha values after this pass.
    loss : float
        Objective value at the end of this pass.
    n_iters : int
        Number of L-BFGS-B iterations used.
    n_evals : int
        Number of surrogate evaluations performed.
    elapsed_s : float
        Wall-clock time for this pass (seconds).
    """

    pass_name: str
    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    loss: float
    n_iters: int
    n_evals: int
    elapsed_s: float


@dataclass(frozen=True)
class CascadeResult:
    """Final result of the per-observable inference cascade.

    Attributes
    ----------
    best_k0_1, best_k0_2 : float
        Best recovered k0 values.
    best_alpha_1, best_alpha_2 : float
        Best recovered alpha values.
    best_loss : float
        Objective value at the best result.
    pass_results : tuple of CascadePassResult
        Per-pass result snapshots.
    total_evals : int
        Total surrogate evaluations across all passes.
    total_time_s : float
        Total wall-clock time (seconds).
    """

    best_k0_1: float
    best_k0_2: float
    best_alpha_1: float
    best_alpha_2: float
    best_loss: float
    pass_results: tuple
    total_evals: int
    total_time_s: float


# ---------------------------------------------------------------------------
# Subset objective helper
# ---------------------------------------------------------------------------

def _make_subset_objective_fn(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    secondary_weight: float,
    subset_idx: np.ndarray | None,
) -> tuple:
    """Build a (objective, gradient) pair that respects subset_idx.

    Returns (objective_fn, gradient_fn, eval_counter_dict) where
    eval_counter_dict["n"] tracks the number of surrogate evaluations.
    The control vector is [log10(k0_1), log10(k0_2), alpha_1, alpha_2].

    When the surrogate supports ``predict_torch()``, gradients are computed
    via PyTorch autograd instead of finite differences.
    """
    if subset_idx is not None:
        tgt_cd = target_cd[subset_idx]
        tgt_pc = target_pc[subset_idx]
    else:
        tgt_cd = np.asarray(target_cd, dtype=float)
        tgt_pc = np.asarray(target_pc, dtype=float)

    valid_cd = ~np.isnan(tgt_cd)
    valid_pc = ~np.isnan(tgt_pc)
    counter = {"n": 0}

    # ----- Autograd path (NN / ensemble surrogates) -----
    if _has_autograd(surrogate):
        import torch

        # Cache target tensors
        _target_cd_t = torch.tensor(tgt_cd[valid_cd], dtype=torch.float64)
        _target_pc_t = torch.tensor(tgt_pc[valid_pc], dtype=torch.float64)
        _valid_cd_idx = torch.tensor(np.where(valid_cd)[0], dtype=torch.long)
        _valid_pc_idx = torch.tensor(np.where(valid_pc)[0], dtype=torch.long)
        _subset_idx_t = (
            torch.tensor(subset_idx, dtype=torch.long)
            if subset_idx is not None
            else None
        )

        def _autograd_obj_and_grad(x: np.ndarray):
            x_t = torch.tensor(
                np.asarray(x, dtype=np.float64),
                dtype=torch.float64,
                requires_grad=True,
            )
            y = surrogate.predict_torch(x_t)
            n_eta = len(y) // 2

            if _subset_idx_t is not None:
                cd_sim = y[_subset_idx_t]
                pc_sim = y[n_eta + _subset_idx_t]
            else:
                cd_sim = y[:n_eta]
                pc_sim = y[n_eta:]

            cd_diff = cd_sim[_valid_cd_idx] - _target_cd_t
            pc_diff = pc_sim[_valid_pc_idx] - _target_pc_t
            J = (
                0.5 * torch.sum(cd_diff ** 2)
                + secondary_weight * 0.5 * torch.sum(pc_diff ** 2)
            )
            J.backward()
            counter["n"] += 1
            return float(J.detach()), x_t.grad.numpy().copy()

        _cache = {}

        def _objective(x: np.ndarray) -> float:
            key = x.tobytes()
            if key not in _cache:
                J, g = _autograd_obj_and_grad(x)
                _cache.clear()
                _cache[key] = (J, g)
            return _cache[key][0]

        def _gradient(x: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
            key = x.tobytes()
            if key not in _cache:
                J, g = _autograd_obj_and_grad(x)
                _cache.clear()
                _cache[key] = (J, g)
            return _cache[key][1]

        return _objective, _gradient, counter

    # ----- Finite-difference path (RBF / POD surrogates) -----
    def _objective(x: np.ndarray) -> float:
        k0_1, k0_2 = 10.0 ** x[0], 10.0 ** x[1]
        alpha_1, alpha_2 = float(x[2]), float(x[3])
        pred = surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_sim = pred["current_density"]
        pc_sim = pred["peroxide_current"]
        if subset_idx is not None:
            cd_sim = cd_sim[subset_idx]
            pc_sim = pc_sim[subset_idx]
        cd_diff = cd_sim[valid_cd] - tgt_cd[valid_cd]
        pc_diff = pc_sim[valid_pc] - tgt_pc[valid_pc]
        j = 0.5 * np.sum(cd_diff ** 2) + secondary_weight * 0.5 * np.sum(pc_diff ** 2)
        counter["n"] += 1
        return float(j)

    def _gradient(x: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.zeros(4, dtype=float)
        for i in range(4):
            xp, xm = x.copy(), x.copy()
            xp[i] += fd_step
            xm[i] -= fd_step
            grad[i] = (_objective(xp) - _objective(xm)) / (2 * fd_step)
        return grad

    return _objective, _gradient, counter


def _make_subset_block_objective_fn(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    reaction_index: int,
    fixed_k0_other: float,
    fixed_alpha_other: float,
    secondary_weight: float,
    subset_idx: np.ndarray | None,
) -> tuple:
    """Build a (objective, gradient) pair for a 2D block with subset_idx.

    The control vector is [log10(k0_r), alpha_r] for the free reaction.
    Returns (objective_fn, gradient_fn, eval_counter_dict).

    When the surrogate supports ``predict_torch()``, gradients are computed
    via PyTorch autograd instead of finite differences.
    """
    if subset_idx is not None:
        tgt_cd = target_cd[subset_idx]
        tgt_pc = target_pc[subset_idx]
    else:
        tgt_cd = np.asarray(target_cd, dtype=float)
        tgt_pc = np.asarray(target_pc, dtype=float)

    valid_cd = ~np.isnan(tgt_cd)
    valid_pc = ~np.isnan(tgt_pc)
    counter = {"n": 0}

    # ----- Autograd path (NN / ensemble surrogates) -----
    if _has_autograd(surrogate):
        import torch

        _target_cd_t = torch.tensor(tgt_cd[valid_cd], dtype=torch.float64)
        _target_pc_t = torch.tensor(tgt_pc[valid_pc], dtype=torch.float64)
        _valid_cd_idx = torch.tensor(np.where(valid_cd)[0], dtype=torch.long)
        _valid_pc_idx = torch.tensor(np.where(valid_pc)[0], dtype=torch.long)
        _subset_idx_t = (
            torch.tensor(subset_idx, dtype=torch.long)
            if subset_idx is not None
            else None
        )
        _log_k0_other = torch.tensor(
            np.log10(max(fixed_k0_other, 1e-30)), dtype=torch.float64,
        )
        _alpha_other = torch.tensor(fixed_alpha_other, dtype=torch.float64)

        def _autograd_obj_and_grad(x: np.ndarray):
            x_t = torch.tensor(
                np.asarray(x, dtype=np.float64),
                dtype=torch.float64,
                requires_grad=True,
            )
            # Build full 4D input
            if reaction_index == 0:
                x_full = torch.stack([
                    x_t[0], _log_k0_other, x_t[1], _alpha_other,
                ])
            else:
                x_full = torch.stack([
                    _log_k0_other, x_t[0], _alpha_other, x_t[1],
                ])

            y = surrogate.predict_torch(x_full)
            n_eta = len(y) // 2

            if _subset_idx_t is not None:
                cd_sim = y[_subset_idx_t]
                pc_sim = y[n_eta + _subset_idx_t]
            else:
                cd_sim = y[:n_eta]
                pc_sim = y[n_eta:]

            cd_diff = cd_sim[_valid_cd_idx] - _target_cd_t
            pc_diff = pc_sim[_valid_pc_idx] - _target_pc_t
            J = (
                0.5 * torch.sum(cd_diff ** 2)
                + secondary_weight * 0.5 * torch.sum(pc_diff ** 2)
            )
            J.backward()
            counter["n"] += 1
            return float(J.detach()), x_t.grad.numpy().copy()

        _cache = {}

        def _objective(x: np.ndarray) -> float:
            key = x.tobytes()
            if key not in _cache:
                J, g = _autograd_obj_and_grad(x)
                _cache.clear()
                _cache[key] = (J, g)
            return _cache[key][0]

        def _gradient(x: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
            key = x.tobytes()
            if key not in _cache:
                J, g = _autograd_obj_and_grad(x)
                _cache.clear()
                _cache[key] = (J, g)
            return _cache[key][1]

        return _objective, _gradient, counter

    # ----- Finite-difference path (RBF / POD surrogates) -----
    def _to_full(x: np.ndarray) -> tuple:
        k0_r = 10.0 ** x[0]
        alpha_r = float(x[1])
        if reaction_index == 0:
            return k0_r, fixed_k0_other, alpha_r, fixed_alpha_other
        return fixed_k0_other, k0_r, fixed_alpha_other, alpha_r

    def _objective(x: np.ndarray) -> float:
        k0_1, k0_2, a1, a2 = _to_full(x)
        pred = surrogate.predict(k0_1, k0_2, a1, a2)
        cd_sim = pred["current_density"]
        pc_sim = pred["peroxide_current"]
        if subset_idx is not None:
            cd_sim = cd_sim[subset_idx]
            pc_sim = pc_sim[subset_idx]
        cd_diff = cd_sim[valid_cd] - tgt_cd[valid_cd]
        pc_diff = pc_sim[valid_pc] - tgt_pc[valid_pc]
        j = 0.5 * np.sum(cd_diff ** 2) + secondary_weight * 0.5 * np.sum(pc_diff ** 2)
        counter["n"] += 1
        return float(j)

    def _gradient(x: np.ndarray, fd_step: float = 1e-5) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        grad = np.zeros(2, dtype=float)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += fd_step
            xm[i] -= fd_step
            grad[i] = (_objective(xp) - _objective(xm)) / (2 * fd_step)
        return grad

    return _objective, _gradient, counter


# ---------------------------------------------------------------------------
# Individual pass runners
# ---------------------------------------------------------------------------

def _run_pass1(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    initial_k0: Sequence[float],
    initial_alpha: Sequence[float],
    bounds_log: list[Tuple[float, float]],
    config: CascadeConfig,
    subset_idx: np.ndarray | None,
) -> CascadePassResult:
    """Pass 1: CD-dominant, all 4 params free."""
    t0 = time.time()

    obj_fn, grad_fn, counter = _make_subset_objective_fn(
        surrogate, target_cd, target_pc,
        secondary_weight=config.pass1_weight,
        subset_idx=subset_idx,
    )

    fd = config.fd_step

    x0 = np.array([
        np.log10(max(initial_k0[0], 1e-30)),
        np.log10(max(initial_k0[1], 1e-30)),
        float(initial_alpha[0]),
        float(initial_alpha[1]),
    ], dtype=float)

    result = minimize(
        obj_fn,
        x0,
        jac=lambda x: grad_fn(x, fd),
        method="L-BFGS-B",
        bounds=bounds_log,
        options={"maxiter": config.pass1_maxiter, "ftol": 1e-14, "gtol": 1e-8},
    )

    k0_1 = 10.0 ** result.x[0]
    k0_2 = 10.0 ** result.x[1]
    alpha_1 = float(result.x[2])
    alpha_2 = float(result.x[3])
    elapsed = time.time() - t0

    return CascadePassResult(
        pass_name="Pass 1 (CD-dominant)",
        k0_1=k0_1, k0_2=k0_2,
        alpha_1=alpha_1, alpha_2=alpha_2,
        loss=float(result.fun),
        n_iters=int(result.get("nit", 0)),
        n_evals=counter["n"],
        elapsed_s=elapsed,
    )


def _run_pass2(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    fixed_k0_1: float,
    fixed_alpha_1: float,
    initial_k0_2: float,
    initial_alpha_2: float,
    bounds_log_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    config: CascadeConfig,
    subset_idx: np.ndarray | None,
) -> CascadePassResult:
    """Pass 2: PC-dominant, only k0_2 and alpha_2 free."""
    t0 = time.time()

    obj_fn, grad_fn, counter = _make_subset_block_objective_fn(
        surrogate, target_cd, target_pc,
        reaction_index=1,
        fixed_k0_other=fixed_k0_1,
        fixed_alpha_other=fixed_alpha_1,
        secondary_weight=config.pass2_weight,
        subset_idx=subset_idx,
    )

    fd = config.fd_step

    x0 = np.array([
        np.log10(max(initial_k0_2, 1e-30)),
        initial_alpha_2,
    ], dtype=float)

    bounds_p2 = [bounds_log_k0_2, bounds_alpha]

    result = minimize(
        obj_fn,
        x0,
        jac=lambda x: grad_fn(x, fd),
        method="L-BFGS-B",
        bounds=bounds_p2,
        options={"maxiter": config.pass2_maxiter, "ftol": 1e-14, "gtol": 1e-8},
    )

    k0_2 = 10.0 ** result.x[0]
    alpha_2 = float(result.x[1])
    elapsed = time.time() - t0

    return CascadePassResult(
        pass_name="Pass 2 (PC-dominant)",
        k0_1=fixed_k0_1, k0_2=k0_2,
        alpha_1=fixed_alpha_1, alpha_2=alpha_2,
        loss=float(result.fun),
        n_iters=int(result.get("nit", 0)),
        n_evals=counter["n"],
        elapsed_s=elapsed,
    )


def _run_polish(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    initial_k0_1: float,
    initial_k0_2: float,
    initial_alpha_1: float,
    initial_alpha_2: float,
    bounds_log: list[Tuple[float, float]],
    config: CascadeConfig,
    subset_idx: np.ndarray | None,
) -> CascadePassResult:
    """Pass 3: Joint polish with moderate weight."""
    t0 = time.time()

    obj_fn, grad_fn, counter = _make_subset_objective_fn(
        surrogate, target_cd, target_pc,
        secondary_weight=config.polish_weight,
        subset_idx=subset_idx,
    )

    fd = config.fd_step

    x0 = np.array([
        np.log10(max(initial_k0_1, 1e-30)),
        np.log10(max(initial_k0_2, 1e-30)),
        initial_alpha_1,
        initial_alpha_2,
    ], dtype=float)

    result = minimize(
        obj_fn,
        x0,
        jac=lambda x: grad_fn(x, fd),
        method="L-BFGS-B",
        bounds=bounds_log,
        options={"maxiter": config.polish_maxiter, "ftol": 1e-14, "gtol": 1e-8},
    )

    k0_1 = 10.0 ** result.x[0]
    k0_2 = 10.0 ** result.x[1]
    alpha_1 = float(result.x[2])
    alpha_2 = float(result.x[3])
    elapsed = time.time() - t0

    return CascadePassResult(
        pass_name="Pass 3 (Joint polish)",
        k0_1=k0_1, k0_2=k0_2,
        alpha_1=alpha_1, alpha_2=alpha_2,
        loss=float(result.fun),
        n_iters=int(result.get("nit", 0)),
        n_evals=counter["n"],
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Main cascade entry point
# ---------------------------------------------------------------------------

def run_cascade_inference(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    initial_k0: Sequence[float],
    initial_alpha: Sequence[float],
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    config: CascadeConfig | None = None,
    subset_idx: np.ndarray | None = None,
) -> CascadeResult:
    """Run per-observable inference cascade.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd, target_pc : np.ndarray
        Target I-V curves.  If ``subset_idx`` is provided, these should
        be the full-grid targets (subsetting is applied internally).
    initial_k0 : sequence of 2 floats
        Initial [k0_1, k0_2] guesses (physical space).
    initial_alpha : sequence of 2 floats
        Initial [alpha_1, alpha_2] guesses.
    bounds_k0_1, bounds_k0_2 : (lo, hi)
        Bounds for k0 in *physical* space.
    bounds_alpha : (lo, hi)
        Bounds for alpha_1 and alpha_2.
    config : CascadeConfig or None
        Algorithm configuration (defaults applied if None).
    subset_idx : np.ndarray or None
        Indices into the surrogate voltage grid for shallow subset.

    Returns
    -------
    CascadeResult
        Final result with per-pass snapshots.
    """
    if config is None:
        config = CascadeConfig()

    t_start = time.time()

    # Log-space bounds for k0
    bounds_log_k0_1 = (
        np.log10(max(bounds_k0_1[0], 1e-30)),
        np.log10(bounds_k0_1[1]),
    )
    bounds_log_k0_2 = (
        np.log10(max(bounds_k0_2[0], 1e-30)),
        np.log10(bounds_k0_2[1]),
    )
    bounds_log_full = [
        bounds_log_k0_1,
        bounds_log_k0_2,
        bounds_alpha,
        bounds_alpha,
    ]

    pass_results: list[CascadePassResult] = []
    total_evals = 0

    # --- Pass 1: CD-dominant, all 4 params free ---
    if config.verbose:
        print(f"\n  [Cascade] Pass 1: CD-dominant (w={config.pass1_weight}), "
              f"4 params free, maxiter={config.pass1_maxiter}")

    p1 = _run_pass1(
        surrogate, target_cd, target_pc,
        initial_k0, initial_alpha,
        bounds_log_full, config, subset_idx,
    )
    pass_results.append(p1)
    total_evals += p1.n_evals

    if config.verbose:
        print(f"    k0=[{p1.k0_1:.4e},{p1.k0_2:.4e}] "
              f"alpha=[{p1.alpha_1:.4f},{p1.alpha_2:.4f}] "
              f"loss={p1.loss:.4e} ({p1.n_iters} iters, {p1.elapsed_s:.2f}s)")

    # --- Pass 2: PC-dominant, k0_1+alpha_1 fixed from Pass 1 ---
    if config.verbose:
        print(f"\n  [Cascade] Pass 2: PC-dominant (w={config.pass2_weight}), "
              f"k0_2+alpha_2 free, maxiter={config.pass2_maxiter}")
        print(f"    Fixed: k0_1={p1.k0_1:.4e}, alpha_1={p1.alpha_1:.4f}")

    p2 = _run_pass2(
        surrogate, target_cd, target_pc,
        fixed_k0_1=p1.k0_1,
        fixed_alpha_1=p1.alpha_1,
        initial_k0_2=p1.k0_2,
        initial_alpha_2=p1.alpha_2,
        bounds_log_k0_2=bounds_log_k0_2,
        bounds_alpha=bounds_alpha,
        config=config,
        subset_idx=subset_idx,
    )
    pass_results.append(p2)
    total_evals += p2.n_evals

    if config.verbose:
        print(f"    k0=[{p2.k0_1:.4e},{p2.k0_2:.4e}] "
              f"alpha=[{p2.alpha_1:.4f},{p2.alpha_2:.4f}] "
              f"loss={p2.loss:.4e} ({p2.n_iters} iters, {p2.elapsed_s:.2f}s)")

    # --- Pass 3: Joint polish (optional) ---
    best = p2
    if not config.skip_polish:
        if config.verbose:
            print(f"\n  [Cascade] Pass 3: Joint polish (w={config.polish_weight}), "
                  f"4 params free, maxiter={config.polish_maxiter}")

        p3 = _run_polish(
            surrogate, target_cd, target_pc,
            initial_k0_1=p2.k0_1,
            initial_k0_2=p2.k0_2,
            initial_alpha_1=p2.alpha_1,
            initial_alpha_2=p2.alpha_2,
            bounds_log=bounds_log_full,
            config=config,
            subset_idx=subset_idx,
        )
        pass_results.append(p3)
        total_evals += p3.n_evals
        best = p3

        if config.verbose:
            print(f"    k0=[{p3.k0_1:.4e},{p3.k0_2:.4e}] "
                  f"alpha=[{p3.alpha_1:.4f},{p3.alpha_2:.4f}] "
                  f"loss={p3.loss:.4e} ({p3.n_iters} iters, {p3.elapsed_s:.2f}s)")

    total_time = time.time() - t_start

    # Re-evaluate best under canonical secondary_weight=1.0 for commensurable loss
    canonical_obj, _, _ = _make_subset_objective_fn(
        surrogate, target_cd, target_pc,
        secondary_weight=1.0,
        subset_idx=subset_idx,
    )
    canonical_x = np.array([
        np.log10(max(best.k0_1, 1e-30)),
        np.log10(max(best.k0_2, 1e-30)),
        best.alpha_1,
        best.alpha_2,
    ], dtype=float)
    canonical_loss = canonical_obj(canonical_x)

    if config.verbose:
        print(f"\n  [Cascade] Done: total_evals={total_evals}, "
              f"total_time={total_time:.2f}s")

    return CascadeResult(
        best_k0_1=best.k0_1,
        best_k0_2=best.k0_2,
        best_alpha_1=best.alpha_1,
        best_alpha_2=best.alpha_2,
        best_loss=canonical_loss,
        pass_results=tuple(pass_results),
        total_evals=total_evals,
        total_time_s=total_time,
    )
