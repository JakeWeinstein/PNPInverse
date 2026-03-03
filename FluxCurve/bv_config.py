"""Configuration dataclass for BV k0 flux-curve inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from FluxCurve.config import ForwardRecoveryConfig


@dataclass
class BVFluxCurveInferenceRequest:
    """User-facing configuration for adjoint-gradient BV k0 inference.

    Mirrors :class:`RobinFluxCurveInferenceRequest` with BV-specific fields:
    - ``true_k0`` / ``initial_guess`` instead of ``true_value``
    - ``k0_lower`` / ``k0_upper`` bounds
    - ``current_density_scale`` conversion factor
    - ``mesh_Nx``, ``mesh_Ny``, ``mesh_beta`` for graded rectangle mesh
    """

    base_solver_params: Sequence[object]
    steady: Any  # SteadyStateConfig
    true_k0: Optional[Sequence[float]]
    initial_guess: Sequence[float]
    phi_applied_values: Sequence[float]
    target_csv_path: str
    output_dir: str
    regenerate_target: bool = False
    target_noise_percent: float = 2.0
    target_seed: int = 20260226
    # Observable mode for BV: "current_density" (default), "peroxide_current",
    # or "reaction".
    observable_mode: str = "current_density"
    observable_reaction_index: Optional[int] = None
    # i_scale: conversion from dimensionless BV rate to physical units.
    # Default corresponds to n_e * F * D_ref * c_bulk / L_ref * 0.1 (mA/cm²).
    current_density_scale: float = 1.0
    observable_label: str = "current density (mA/cm²)"
    observable_title: str = "BV k0 fit"
    # Control mode: "k0" (default), "alpha", or "joint" (k0 + alpha).
    control_mode: str = "k0"
    # k0 bounds for L-BFGS-B
    k0_lower: float = 1e-8
    k0_upper: float = 100.0
    # Log-space optimization: optimizer works in x = log10(k0) space.
    # Gradient transformed via chain rule: dJ/dx = dJ/dk0 * k0 * ln(10).
    # Essential for parameters spanning orders of magnitude.
    log_space: bool = True
    # Alpha inference settings
    true_alpha: Optional[Sequence[float]] = None
    initial_alpha_guess: Optional[Sequence[float]] = None
    alpha_lower: float = 0.05
    alpha_upper: float = 0.95
    alpha_log_space: bool = False
    # Per-component bound overrides (optional). When set, these replace the
    # scalar k0_lower/k0_upper/alpha_lower/alpha_upper for that component.
    # Length must match the number of k0 or alpha components.
    k0_lower_per_component: Optional[Sequence[float]] = None
    k0_upper_per_component: Optional[Sequence[float]] = None
    alpha_lower_per_component: Optional[Sequence[float]] = None
    alpha_upper_per_component: Optional[Sequence[float]] = None
    # Fixed k0 values when control_mode="alpha" (known k0, infer alpha only)
    fixed_k0: Optional[Sequence[float]] = None
    # Steric inference settings (control_mode="steric" or "full")
    # "steric" infers a_vals only; "full" infers k0 + alpha + a_vals
    true_steric_a: Optional[Sequence[float]] = None
    initial_steric_a_guess: Optional[Sequence[float]] = None
    steric_a_lower: float = 0.001
    steric_a_upper: float = 0.5
    # Fixed k0/alpha when control_mode="steric"
    fixed_alpha: Optional[Sequence[float]] = None
    # Bridge point auto-insertion: when max_eta_gap > 0, automatically insert
    # forward-only bridge points between inference points whose eta gap exceeds
    # this threshold.  Bridge points carry the warm-start solution but do NOT
    # contribute to the objective or gradient (no adjoint tape).
    # Default 0.0 means bridge points are OFF (backward compatible).
    max_eta_gap: float = 0.0
    # Graded rectangle mesh params
    mesh_Nx: int = 4
    mesh_Ny: int = 200
    mesh_beta: float = 3.0
    # Optimizer
    optimizer_method: str = "L-BFGS-B"
    optimizer_tolerance: Optional[float] = None
    optimizer_options: Optional[Mapping[str, Any]] = None
    max_iters: int = 30
    gtol: float = 1e-3
    fail_penalty: float = 1e9
    print_point_gradients: bool = True
    blob_initial_condition: bool = False
    # Live plot settings
    live_plot: bool = True
    live_plot_pause_seconds: float = 0.001
    live_plot_eval_lines: bool = True
    live_plot_eval_line_alpha: float = 0.30
    live_plot_eval_max_lines: int = 120
    live_plot_capture_frames_dir: Optional[str] = None
    live_plot_capture_every_n_updates: int = 1
    live_plot_capture_max_frames: int = 1000
    live_plot_export_gif_path: Optional[str] = None
    live_plot_export_gif_seconds: float = 5.0
    live_plot_export_gif_frames: int = 50
    live_plot_export_gif_dpi: int = 140
    # Tikhonov regularization: penalizes deviation from prior parameter values.
    # J_reg = lambda * sum((log10(k0) - log10(k0_prior))^2)  for k0 (log-space)
    # J_reg = lambda * sum((alpha - alpha_prior)^2)            for alpha (linear)
    # Backward compatible: lambda=0 disables regularization entirely.
    regularization_lambda: float = 0.0
    regularization_k0_prior: Optional[Sequence[float]] = None
    regularization_alpha_prior: Optional[Sequence[float]] = None
    # Multi-fidelity mesh coarsening: when enabled, run a short optimization
    # on a coarse mesh first to get an approximate solution, then use that
    # as initial guess for the fine mesh run.
    multifidelity_enabled: bool = False
    coarse_mesh_Nx: int = 4
    coarse_mesh_Ny: int = 100
    coarse_max_iters: int = 5
    # Checkpoint-restart warm-start: when True, cache all converged point
    # solutions and reuse them as ICs on subsequent evaluations (skipping
    # sequential sweep after first full pass).
    use_checkpoint_warmstart: bool = True
    # Multi-observable joint fitting: when secondary_observable_mode is set,
    # the objective sums contributions from the primary observable_mode and
    # the secondary one (e.g. "current_density" + "peroxide_current").
    # Each generates its own target curve and computes 0.5*(sim-target)^2.
    # secondary_observable_weight scales the secondary term relative to primary.
    secondary_observable_mode: Optional[str] = None
    secondary_observable_weight: float = 1.0
    secondary_current_density_scale: Optional[float] = None
    secondary_target_csv_path: Optional[str] = None
    # Multi-pH inference: fit I-V curves at multiple pH conditions simultaneously
    # with shared (k0, alpha) but pH-dependent c_H+ bulk concentrations.
    # Each entry: {"pH": float, "c_hp_hat": float, "c_hp_bulk_mol_m3": float,
    #              "weight": float, "target_csv_path": str}
    # When set, the pipeline loops over conditions and sums objectives/gradients.
    multi_ph_conditions: Optional[List[Dict[str, Any]]] = None
    # Parallel fast-path: when enabled, the checkpoint-restart fast path
    # (which runs after the first sequential sweep populates _all_points_cache)
    # dispatches independent per-point solves to a ProcessPoolExecutor.
    # Each worker spawns its own Firedrake instance (one-time ~2-5s cost).
    # parallel_workers=0 means auto-detect via os.cpu_count().
    parallel_fast_path: bool = False
    parallel_workers: int = 0
    # Recovery
    anisotropy_trigger_failed_points: int = 4
    anisotropy_trigger_failed_fraction: float = 0.25
    forward_recovery: ForwardRecoveryConfig = field(
        default_factory=lambda: ForwardRecoveryConfig()
    )
