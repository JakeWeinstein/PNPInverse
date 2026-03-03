"""Dataclass definitions for inference runner configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SyntheticData:
    """Clean/noisy terminal fields produced by one forward solve."""

    clean_concentration_vectors: List[np.ndarray]
    clean_phi_vector: np.ndarray
    noisy_concentration_vectors: List[np.ndarray]
    noisy_phi_vector: np.ndarray

    def select_targets(self, *, use_noisy_data: bool) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return objective targets from clean or noisy fields."""
        if use_noisy_data:
            return self.noisy_concentration_vectors, self.noisy_phi_vector
        return self.clean_concentration_vectors, self.clean_phi_vector


@dataclass
class InferenceRequest:
    """Configuration for one inverse run.

    Parameters
    ----------
    adapter:
        Forward solver adapter created from a compatible ``forsolve`` module.
    target:
        Parameter target definition from the target registry.
    base_solver_params:
        Baseline 11-entry solver parameter list. ``target`` will inject true and
        guess values into copies of this list.
    true_value:
        Ground-truth value used to generate synthetic observations.
    initial_guess:
        Starting value used to initialize optimization controls.
    noise_percent:
        Gaussian noise level in percent of RMS(field).
    seed:
        RNG seed for reproducible noise.
    optimizer_method:
        Method name forwarded to ``firedrake.adjoint.minimize``.
    optimizer_options:
        Optional method-specific options.
    tolerance:
        Optional optimizer tolerance.
    bounds:
        Optional bounds. If ``None``, target defaults are used.
    fit_to_noisy_data:
        If True, infer against noisy observations; else clean observations.
    blob_initial_condition:
        Passed through to forward solver initialization.
    print_interval_data:
        Forward solve print interval for data generation.
    print_interval_inverse:
        Forward solve print interval inside taped objective evaluations.
    recovery_attempts:
        Optional override for total resilient-minimization attempts. If set,
        this value takes priority over ``recovery.max_attempts``.
    recovery:
        Configuration for the always-on resilient minimizer.
    """

    adapter: Any  # ForwardSolverAdapter
    target: Any  # ParameterTarget
    base_solver_params: Sequence[Any]
    true_value: Any
    initial_guess: Any
    noise_percent: float = 10.0
    seed: Optional[int] = None
    optimizer_method: str = "L-BFGS-B"
    optimizer_options: Optional[Mapping[str, Any]] = None
    tolerance: Optional[float] = 1e-8
    bounds: Optional[Any] = None
    fit_to_noisy_data: bool = True
    blob_initial_condition: bool = True
    print_interval_data: int = 100
    print_interval_inverse: int = 100
    recovery_attempts: Optional[int] = None
    recovery: "RecoveryConfig" = field(default_factory=lambda: RecoveryConfig())


@dataclass
class RecoveryConfig:
    """Configuration for resilient minimization retries.

    The first attempt uses user-provided solver options. Subsequent attempts
    cycle through staged recovery:
    1) ``max_it`` attempts (increase ``snes_max_it`` only),
    2) ``anisotropy`` attempts (reduce parameter anisotropy),
    3) ``tolerance_relax`` attempts (loosen atol/rtol).

    The cycle then repeats. Entering the anisotropy stage resets max-iteration
    and tolerance relaxations to their baseline values for a fresh restart.
    """

    max_attempts: int = 15
    contraction_factor: float = 0.5
    fallback_shrink_if_stuck: float = 0.15
    max_it_only_attempts: int = 2
    anisotropy_only_attempts: int = 3
    tolerance_relax_attempts: int = 1
    anisotropy_target_ratio: float = 3.0
    anisotropy_blend: float = 0.5
    atol_relax_factor: float = 10.0
    rtol_relax_factor: float = 10.0
    ksp_rtol_relax_factor: float = 10.0
    max_it_growth: float = 1.5
    max_it_cap: int = 500
    line_search_schedule: Tuple[str, ...] = ("bt", "l2", "cp", "basic")
    verbose: bool = True


@dataclass
class RecoveryAttempt:
    """Per-attempt log entry produced by resilient minimization."""

    attempt_index: int
    phase: str
    solver_options: Dict[str, Any]
    start_guess: Any
    status: str
    reason: str
    best_objective_seen: Optional[float] = None
    best_estimate_seen: Optional[Any] = None


@dataclass
class InferenceResult:
    """Outputs returned by :func:`run_inverse_inference`."""

    target_key: str
    estimate: Any
    objective_value: float
    true_solver_params: List[Any]
    inverse_solver_params: List[Any]
    synthetic_data: SyntheticData
    optimized_controls: Any
    recovery_attempts: List[RecoveryAttempt]
