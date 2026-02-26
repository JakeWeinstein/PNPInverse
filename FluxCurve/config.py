"""Configuration dataclasses for Robin kappa flux-curve inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


@dataclass
class ForwardRecoveryConfig:
    """Recovery settings for failed forward solves at a single phi_applied point.

    This mirrors resilient-minimize staging:
    1) max-it increases only,
    2) anisotropy stage (resets relaxation knobs),
    3) tolerance relaxation.
    """

    max_attempts: int = 8
    max_it_only_attempts: int = 2
    anisotropy_only_attempts: int = 1
    tolerance_relax_attempts: int = 2
    max_it_growth: float = 1.5
    max_it_cap: int = 500
    atol_relax_factor: float = 10.0
    rtol_relax_factor: float = 10.0
    ksp_rtol_relax_factor: float = 10.0
    line_search_schedule: Tuple[str, ...] = ("bt", "l2", "cp", "basic")
    anisotropy_target_ratio: float = 3.0
    anisotropy_blend: float = 0.5


@dataclass
class RobinFluxCurveInferenceRequest:
    """User-facing configuration for adjoint-gradient Robin-kappa inference."""

    base_solver_params: Sequence[object]
    steady: Any  # SteadyStateConfig imported at use-site to avoid heavy dep
    true_value: Optional[Sequence[float]]
    initial_guess: Sequence[float]
    phi_applied_values: Sequence[float]
    target_csv_path: str
    output_dir: str
    regenerate_target: bool = False
    target_noise_percent: float = 2.0
    target_seed: int = 20260220
    observable_mode: str = "total_species"
    observable_species_index: Optional[int] = None
    observable_scale: float = 1.0
    observable_label: str = "steady-state flux (observable)"
    observable_title: str = "Robin kappa fit"
    kappa_lower: float = 1e-6
    kappa_upper: float = 20.0
    optimizer_method: str = "L-BFGS-B"
    optimizer_tolerance: Optional[float] = None
    optimizer_options: Optional[Mapping[str, Any]] = None
    max_iters: int = 8
    gtol: float = 1e-4
    fail_penalty: float = 1e9
    print_point_gradients: bool = True
    # Use uniform+linear IC by default; empirically closer to steady state than blob.
    blob_initial_condition: bool = False
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
    anisotropy_trigger_failed_points: int = 4
    anisotropy_trigger_failed_fraction: float = 0.25
    # Point solves across phi_applied values can run in parallel worker processes.
    parallel_point_solves_enabled: bool = False
    parallel_point_workers: int = 4
    parallel_point_min_points: int = 4
    parallel_start_method: str = "spawn"
    # Replay mode keeps per-phi reduced functionals alive for fast re-evaluation.
    # TEMPORARILY DISABLED: replay can produce invalid/non-steady evaluations.
    replay_mode_enabled: bool = False
    replay_reenable_after_successes: int = 1
    replay_extra_steady_steps: int = 3
    replay_post_refine_enabled: bool = True
    replay_post_refine_max_iters: int = 20
    forward_recovery: ForwardRecoveryConfig = field(
        default_factory=lambda: ForwardRecoveryConfig()
    )


@dataclass(frozen=True)
class _ParallelPointConfig:
    """Worker-initialized immutable config for one-point forward/adjoint solves."""

    base_solver_params: Sequence[object]
    steady: Any
    blob_initial_condition: bool
    fail_penalty: float
    forward_recovery: ForwardRecoveryConfig
    observable_mode: str
    observable_species_index: Optional[int]
    observable_scale: float
