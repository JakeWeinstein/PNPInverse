"""Helper utilities for Robin kappa inference from a phi_applied-flux curve.

This module contains the full adjoint-gradient machinery used by
``InferenceScripts/Infer_RobinKappa_from_flux_curve.py`` so that the entry
script can stay short and user-config focused.
"""

from __future__ import annotations

import csv
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from Utils.robin_flux_experiment import (
    FARADAY_CONSTANT,
    SteadyStateConfig,
    add_percent_noise,
    all_results_converged,
    configure_robin_solver_params,
    read_phi_applied_flux_csv,
    results_to_flux_array,
    sweep_phi_applied_steady_flux,
    write_phi_applied_flux_csv,
)
from Utils.robin_forsolve import build_context, build_forms, set_initial_conditions


@dataclass
class RobinFluxCurveInferenceRequest:
    """User-facing configuration for adjoint-gradient Robin-kappa inference."""

    base_solver_params: Sequence[object]
    steady: SteadyStateConfig
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
    # Threading is intentionally avoided because firedrake-adjoint tape state is
    # process-global and not thread-safe for concurrent annotations.
    parallel_point_solves_enabled: bool = False
    parallel_point_workers: int = 4
    parallel_point_min_points: int = 4
    parallel_start_method: str = "spawn"
    # Replay mode keeps per-phi reduced functionals alive for fast re-evaluation.
    # TEMPORARILY DISABLED: replay can produce invalid/non-steady evaluations for
    # some kappa updates. Keep default False until replay validity is reworked.
    replay_mode_enabled: bool = False
    replay_reenable_after_successes: int = 1
    # Replay point models are built by solving to steady state at an anchor
    # kappa and then taking extra post-steady timesteps as a safety buffer.
    # This helps replay remain valid for nearby kappa values.
    replay_extra_steady_steps: int = 3
    # After replay-based optimization, run a replay-off refinement solve from the
    # replay best point so final reported kappa is on the full dynamic objective.
    replay_post_refine_enabled: bool = True
    replay_post_refine_max_iters: int = 20
    forward_recovery: "ForwardRecoveryConfig" = field(
        default_factory=lambda: ForwardRecoveryConfig()
    )


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
class RobinFluxCurveInferenceResult:
    """Inference outputs and generated artifact paths."""

    best_kappa: np.ndarray
    best_loss: float
    phi_applied_values: np.ndarray
    target_flux: np.ndarray
    best_simulated_flux: np.ndarray
    forward_failures_at_best: int
    fit_csv_path: str
    fit_plot_path: Optional[str]
    history_csv_path: str
    point_gradient_csv_path: str
    live_gif_path: Optional[str]
    optimization_success: bool
    optimization_message: str
    replay_rebuild_count: int
    replay_diag_rebuild_count: int
    replay_exception_rebuild_count: int


@dataclass
class PointAdjointResult:
    """Adjoint-evaluation result for one phi_applied point."""

    phi_applied: float
    target_flux: float
    simulated_flux: float
    objective: float
    gradient: np.ndarray
    converged: bool
    steps_taken: int
    reason: str = ""
    final_relative_change: Optional[float] = None
    final_absolute_change: Optional[float] = None
    diagnostics_valid: bool = True


@dataclass(frozen=True)
class _ParallelPointConfig:
    """Worker-initialized immutable config for one-point forward/adjoint solves."""

    base_solver_params: Sequence[object]
    steady: SteadyStateConfig
    blob_initial_condition: bool
    fail_penalty: float
    forward_recovery: "ForwardRecoveryConfig"
    observable_mode: str
    observable_species_index: Optional[int]
    observable_scale: float


@dataclass
class CurveAdjointResult:
    """Aggregated objective + gradient across the full phi_applied curve."""

    objective: float
    gradient: np.ndarray
    simulated_flux: np.ndarray
    points: List[PointAdjointResult]
    n_failed: int
    effective_kappa: np.ndarray
    used_anisotropy_recovery: bool = False
    used_replay_mode: bool = False


@dataclass
class _ReplayPointFunctional:
    """Persistent reduced-functional object for one phi_applied sweep point."""

    phi_applied: float
    tape: Any
    control_state: List[object]
    reduced_flux: Any
    reduced_flux_prev: Any
    reduced_state_delta_l2: Any
    reduced_state_norm_l2: Any
    steady_rel_tol: float
    steady_abs_tol: float
    steps_taken: int


@dataclass
class _ReplayBundle:
    """Collection of replay-ready point models for the full phi_applied sweep."""

    points: List[_ReplayPointFunctional]
    anchor_kappa: np.ndarray


_PARALLEL_POINT_CONFIG: Optional[_ParallelPointConfig] = None


def _point_result_to_payload(point: PointAdjointResult) -> Dict[str, object]:
    """Convert point result to a plain payload for inter-process transport."""
    return {
        "phi_applied": float(point.phi_applied),
        "target_flux": float(point.target_flux),
        "simulated_flux": float(point.simulated_flux),
        "objective": float(point.objective),
        "gradient": np.asarray(point.gradient, dtype=float).tolist(),
        "converged": bool(point.converged),
        "steps_taken": int(point.steps_taken),
        "reason": str(point.reason),
        "final_relative_change": (
            None if point.final_relative_change is None else float(point.final_relative_change)
        ),
        "final_absolute_change": (
            None if point.final_absolute_change is None else float(point.final_absolute_change)
        ),
        "diagnostics_valid": bool(point.diagnostics_valid),
    }


def _point_result_from_payload(payload: Mapping[str, object]) -> PointAdjointResult:
    """Reconstruct PointAdjointResult from plain payload."""
    gradient_raw = payload.get("gradient", [0.0, 0.0])
    gradient_arr = np.asarray(list(gradient_raw), dtype=float)
    return PointAdjointResult(
        phi_applied=float(payload.get("phi_applied", float("nan"))),
        target_flux=float(payload.get("target_flux", float("nan"))),
        simulated_flux=float(payload.get("simulated_flux", float("nan"))),
        objective=float(payload.get("objective", float("inf"))),
        gradient=gradient_arr,
        converged=bool(payload.get("converged", False)),
        steps_taken=int(payload.get("steps_taken", 0)),
        reason=str(payload.get("reason", "")),
        final_relative_change=(
            None
            if payload.get("final_relative_change", None) is None
            else float(payload.get("final_relative_change"))
        ),
        final_absolute_change=(
            None
            if payload.get("final_absolute_change", None) is None
            else float(payload.get("final_absolute_change"))
        ),
        diagnostics_valid=bool(payload.get("diagnostics_valid", False)),
    )


def _parallel_worker_init(config: _ParallelPointConfig) -> None:
    """Initialize one worker process with static point-solve configuration."""
    global _PARALLEL_POINT_CONFIG
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _PARALLEL_POINT_CONFIG = config


def _parallel_worker_solve_point(
    task: Tuple[int, float, float, Sequence[float]],
) -> Tuple[int, Dict[str, object]]:
    """Worker entrypoint: solve one point and return serializable payload."""
    global _PARALLEL_POINT_CONFIG
    if _PARALLEL_POINT_CONFIG is None:
        raise RuntimeError("Parallel worker config is not initialized.")

    idx, phi_applied, target_flux, kappa_values = task
    cfg = _PARALLEL_POINT_CONFIG
    point = solve_point_objective_and_gradient(
        base_solver_params=cfg.base_solver_params,
        steady=cfg.steady,
        phi_applied=float(phi_applied),
        target_flux=float(target_flux),
        kappa_values=[float(v) for v in kappa_values],
        blob_initial_condition=bool(cfg.blob_initial_condition),
        fail_penalty=float(cfg.fail_penalty),
        forward_recovery=cfg.forward_recovery,
        observable_mode=str(cfg.observable_mode),
        observable_species_index=cfg.observable_species_index,
        observable_scale=float(cfg.observable_scale),
    )
    return int(idx), _point_result_to_payload(point)


class _PointSolveExecutor:
    """Optional process-based executor for parallel phi_applied point solves."""

    def __init__(
        self,
        *,
        request: RobinFluxCurveInferenceRequest,
        n_points: int,
    ) -> None:
        self.enabled = False
        self.max_workers = 1
        self._executor: Optional[ProcessPoolExecutor] = None

        if not bool(request.parallel_point_solves_enabled):
            return
        n_points_i = int(max(0, n_points))
        if n_points_i < int(max(1, request.parallel_point_min_points)):
            return

        requested_workers = int(max(1, request.parallel_point_workers))
        workers = min(requested_workers, n_points_i)
        if workers <= 1:
            return

        method = str(request.parallel_start_method or "spawn").strip().lower()
        if method not in ("spawn", "forkserver"):
            method = "spawn"

        config = _ParallelPointConfig(
            base_solver_params=copy.deepcopy(request.base_solver_params),
            steady=copy.deepcopy(request.steady),
            blob_initial_condition=bool(request.blob_initial_condition),
            fail_penalty=float(request.fail_penalty),
            forward_recovery=copy.deepcopy(request.forward_recovery),
            observable_mode=str(request.observable_mode),
            observable_species_index=request.observable_species_index,
            observable_scale=float(request.observable_scale),
        )
        try:
            ctx = mp.get_context(method)
            self._executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_parallel_worker_init,
                initargs=(config,),
            )
            self.enabled = True
            self.max_workers = int(workers)
            print(
                "[parallel] enabled point-solve workers: "
                f"workers={self.max_workers} start_method={method}"
            )
        except Exception as exc:
            self.enabled = False
            self.max_workers = 1
            self._executor = None
            print(
                "[parallel] worker pool initialization failed; "
                f"using serial point solves ({type(exc).__name__}: {exc})"
            )

    def close(self) -> None:
        """Shutdown worker pool if active."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self.enabled = False

    def map_points(
        self,
        *,
        phi_applied_values: np.ndarray,
        target_flux: np.ndarray,
        kappa_values: np.ndarray,
    ) -> Optional[List[PointAdjointResult]]:
        """Solve all points in parallel. Returns None when executor unavailable."""
        if not self.enabled or self._executor is None:
            return None

        kappa_list = np.asarray(kappa_values, dtype=float).tolist()
        tasks: List[Tuple[int, float, float, Sequence[float]]] = []
        for i, (phi_i, target_i) in enumerate(
            zip(phi_applied_values.tolist(), target_flux.tolist())
        ):
            tasks.append((int(i), float(phi_i), float(target_i), kappa_list))

        results: List[Optional[PointAdjointResult]] = [None] * len(tasks)
        try:
            future_map = {
                self._executor.submit(_parallel_worker_solve_point, task): int(task[0])
                for task in tasks
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                payload_idx, payload = future.result()
                if int(payload_idx) != int(idx):
                    raise RuntimeError(
                        "Parallel worker returned mismatched point index "
                        f"(expected {idx}, got {payload_idx})."
                    )
                results[idx] = _point_result_from_payload(payload)
        except Exception as exc:
            print(
                "[parallel] worker execution failed; "
                f"falling back to serial point solves ({type(exc).__name__}: {exc})"
            )
            return None

        out: List[PointAdjointResult] = []
        for idx, point in enumerate(results):
            if point is None:
                raise RuntimeError(f"Missing parallel result for point index {idx}.")
            out.append(point)
        return out


class _LiveFitPlot:
    """Interactive plot that updates the fit curve during optimization."""

    def __init__(
        self,
        *,
        phi_applied_values: np.ndarray,
        target_flux: np.ndarray,
        y_label: str,
        title: str,
        enabled: bool,
        pause_seconds: float,
        show_eval_lines: bool,
        eval_line_alpha: float,
        eval_max_lines: int,
        capture_frames_dir: Optional[str],
        capture_every_n_updates: int,
        capture_max_frames: int,
    ) -> None:
        self.enabled = bool(enabled and plt is not None)
        self.pause_seconds = float(max(0.0, pause_seconds))
        self.show_eval_lines = bool(show_eval_lines)
        self.eval_line_alpha = float(min(max(eval_line_alpha, 0.0), 1.0))
        self.eval_max_lines = int(max(1, eval_max_lines))
        self.capture_frames_dir = capture_frames_dir
        self.capture_every_n_updates = int(max(1, capture_every_n_updates))
        self.capture_max_frames = int(max(1, capture_max_frames))
        self._update_counter = 0
        self._captured_frames = 0
        self.fig = None
        self.ax = None
        self.target_line = None
        self.best_line = None
        self.current_line = None
        self.status_text = None
        self.eval_lines: List[object] = []
        self.eval_cmap = None

        if not self.enabled:
            return

        try:
            if self.capture_frames_dir:
                if os.path.isdir(self.capture_frames_dir):
                    shutil.rmtree(self.capture_frames_dir)
                os.makedirs(self.capture_frames_dir, exist_ok=True)
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 4))
            x = np.asarray(phi_applied_values, dtype=float)
            y_target = np.asarray(target_flux, dtype=float)
            y_nan = np.full_like(y_target, np.nan)

            (self.target_line,) = self.ax.plot(
                x, y_target, marker="o", linewidth=2, label="target (true)"
            )
            (self.best_line,) = self.ax.plot(
                x, y_nan, marker="s", linewidth=2, label="best guess (so far)"
            )
            (self.current_line,) = self.ax.plot(
                x, y_nan, linestyle="--", linewidth=1.5, label="current iteration"
            )
            self.status_text = self.ax.text(
                0.01,
                0.99,
                "",
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
            self.ax.set_xlabel("applied voltage phi_applied")
            self.ax.set_ylabel(str(y_label))
            self.ax.set_title(f"{title} (live)")
            self.ax.grid(True, alpha=0.25)
            self.ax.legend()
            self.fig.tight_layout()
            # Keep one persistent, interactive window open for the entire fit.
            self.fig.show()
            self.eval_cmap = plt.get_cmap("turbo")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(self.pause_seconds)
            self._capture_frame(force=True)
        except Exception:
            # Fail open: disable live plotting but continue optimization.
            self.enabled = False
            self.fig = None
            self.ax = None

    def add_eval_curve(self, *, flux: np.ndarray, eval_id: int) -> None:
        """Add one colored line for a newly evaluated candidate curve."""
        if (
            not self.enabled
            or not self.show_eval_lines
            or self.fig is None
            or self.ax is None
            or self.current_line is None
        ):
            return
        try:
            y = np.asarray(flux, dtype=float)
            x = np.asarray(self.current_line.get_xdata(), dtype=float)
            if y.shape != x.shape:
                return

            if self.eval_cmap is None:
                self.eval_cmap = plt.get_cmap("turbo")
            color = self.eval_cmap((int(eval_id) % 20) / 20.0)

            (line,) = self.ax.plot(
                x,
                y,
                color=color,
                linewidth=1.0,
                alpha=self.eval_line_alpha,
                zorder=1,
            )
            self.eval_lines.append(line)

            while len(self.eval_lines) > self.eval_max_lines:
                old = self.eval_lines.pop(0)
                try:
                    old.remove()
                except Exception:
                    pass
        except Exception:
            pass

    def update(
        self,
        *,
        current_flux: np.ndarray,
        best_flux: np.ndarray,
        iteration: int,
        objective: float,
        n_failed: int,
        kappa: np.ndarray,
        eval_id: Optional[int] = None,
    ) -> None:
        """Refresh plot lines/text for the latest optimization state."""
        if not self.enabled or self.fig is None or self.ax is None:
            return
        try:
            current_flux = np.asarray(current_flux, dtype=float)
            best_flux = np.asarray(best_flux, dtype=float)
            self.current_line.set_ydata(current_flux)
            self.best_line.set_ydata(best_flux)
            eval_text = "" if eval_id is None else f"eval={int(eval_id):03d}  "
            self.status_text.set_text(
                f"{eval_text}"
                f"iter={int(iteration):02d}  "
                f"loss={float(objective):.6e}  "
                f"fails={int(n_failed):02d}  "
                f"kappa=[{float(kappa[0]):.6f}, {float(kappa[1]):.6f}]"
            )

            finite_vals = []
            for arr in (current_flux, best_flux, self.target_line.get_ydata()):
                vals = np.asarray(arr, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    finite_vals.append(vals)
            if finite_vals:
                all_vals = np.concatenate(finite_vals)
                y_min = float(np.min(all_vals))
                y_max = float(np.max(all_vals))
                span = max(1e-8, y_max - y_min)
                pad = 0.08 * span
                self.ax.set_ylim(y_min - pad, y_max + pad)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(self.pause_seconds)
            self._capture_frame(force=False)
        except Exception:
            pass

    def save(self, path: str) -> None:
        """Save current live figure to disk."""
        if not self.enabled or self.fig is None:
            return
        self.fig.savefig(path, dpi=160)

    def _capture_frame(self, *, force: bool) -> None:
        """Optionally save the current live figure as an animation frame."""
        if (
            not self.enabled
            or self.fig is None
            or not self.capture_frames_dir
            or self._captured_frames >= self.capture_max_frames
        ):
            return
        self._update_counter += 1
        if (not force) and (self._update_counter % self.capture_every_n_updates != 0):
            return
        frame_path = os.path.join(
            self.capture_frames_dir, f"frame_{self._captured_frames:04d}.png"
        )
        try:
            self.fig.savefig(frame_path, dpi=160)
            self._captured_frames += 1
        except Exception:
            pass


def _normalize_kappa(value: Optional[Sequence[float]], *, name: str) -> Optional[List[float]]:
    """Convert 2-species kappa-like input into a validated float list."""
    if value is None:
        return None
    vals = [float(v) for v in list(value)]
    if len(vals) != 2:
        raise ValueError(f"{name} must have length 2; got {len(vals)}.")
    if any(v <= 0.0 for v in vals):
        raise ValueError(f"{name} must be strictly positive.")
    return vals


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
    solver_options: Dict[str, Any],
    *,
    phase: str,
    phase_step: int,
    recovery: ForwardRecoveryConfig,
    baseline_options: Mapping[str, Any],
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

def ensure_target_curve(
    *,
    target_csv_path: str,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    true_kappa: Optional[Sequence[float]],
    noise_percent: float,
    seed: int,
    force_regenerate: bool,
    blob_initial_condition: bool,
) -> Dict[str, np.ndarray]:
    """Load target data from CSV or generate synthetic target if missing."""
    if os.path.exists(target_csv_path) and not force_regenerate:
        print(f"Loading target curve from: {target_csv_path}")
        return read_phi_applied_flux_csv(target_csv_path, flux_column="flux_noisy")

    kappa_true = _normalize_kappa(true_kappa, name="true_value")
    if kappa_true is None:
        kappa_true = [0.8, 0.8]

    if os.path.exists(target_csv_path):
        print(
            "Regenerating target curve with requested kappa_true; "
            f"overwriting existing CSV: {target_csv_path}"
        )
    else:
        print("Target CSV not found; generating synthetic target data first.")
    print(f"Synthetic target settings: kappa_true={kappa_true}, noise_percent={noise_percent}")

    target_results = sweep_phi_applied_steady_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        kappa_values=kappa_true,
        blob_initial_condition=bool(blob_initial_condition),
    )
    if not all_results_converged(target_results):
        failed = [f"{r.phi_applied:.6f}" for r in target_results if not r.converged]
        raise RuntimeError(
            "Synthetic target generation failed for some phi_applied values. "
            f"Failed values: {failed}"
        )

    target_flux_clean = results_to_flux_array(target_results)
    target_flux_noisy = add_percent_noise(target_flux_clean, noise_percent, seed=seed)

    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
    write_phi_applied_flux_csv(target_csv_path, target_results, noisy_flux=target_flux_noisy)
    print(f"Synthetic target saved to: {target_csv_path}")

    return {"phi_applied": phi_applied_values.copy(), "flux": target_flux_noisy}


def _build_species_boundary_flux_forms(
    ctx: Dict[str, object],
    *,
    state: Optional[object] = None,
) -> List[object]:
    """Build integrated Robin-boundary flux forms for a selected mixed state."""
    import firedrake as fd

    n_species = int(ctx["n_species"])
    robin = ctx["robin_settings"]
    electrode_marker = int(robin["electrode_marker"])
    c_inf_vals = [float(v) for v in robin["c_inf_vals"]]
    kappa_funcs = list(ctx["kappa_funcs"])
    mixed_state = ctx["U"] if state is None else state
    ci = fd.split(mixed_state)[:-1]
    ds = fd.Measure("ds", domain=ctx["mesh"])

    forms: List[object] = []
    for i in range(n_species):
        forms.append(kappa_funcs[i] * (ci[i] - fd.Constant(c_inf_vals[i])) * ds(electrode_marker))
    return forms


def _build_observable_form(
    ctx: Dict[str, object],
    *,
    mode: str,
    species_index: Optional[int],
    scale: float,
    state: Optional[object] = None,
) -> object:
    """Build scalar observable form from species boundary flux forms.

    Supported modes:
    - ``total_species``: sum_i flux_i
    - ``total_charge``: F * sum_i z_i * flux_i
    - ``charge_proxy_no_f``: sum_i z_i * flux_i (Faraday scaling omitted)
    - ``species``: flux_species_index
    """
    import firedrake as fd

    mode_norm = str(mode).strip().lower()
    forms = _build_species_boundary_flux_forms(ctx, state=state)
    n_species = len(forms)

    if mode_norm == "total_species":
        out = 0
        for form_i in forms:
            out += form_i
    elif mode_norm == "total_charge":
        z_consts = list(ctx.get("z_consts", []))
        if len(z_consts) != n_species:
            raise ValueError(
                f"z_consts length {len(z_consts)} does not match species count {n_species}."
            )
        out = 0
        for i in range(n_species):
            out += z_consts[i] * forms[i]
        out = fd.Constant(float(FARADAY_CONSTANT)) * out
    elif mode_norm == "charge_proxy_no_f":
        z_consts = list(ctx.get("z_consts", []))
        if len(z_consts) != n_species:
            raise ValueError(
                f"z_consts length {len(z_consts)} does not match species count {n_species}."
            )
        # Provisional observable for parameter studies: charge-weighted species
        # flux without Faraday scaling. Units are intentionally treated as a.u.
        # until physical current-density calibration is finalized.
        out = 0
        for i in range(n_species):
            out += z_consts[i] * forms[i]
    elif mode_norm == "species":
        if species_index is None:
            raise ValueError("species_index must be set when observable_mode='species'.")
        idx = int(species_index)
        if idx < 0 or idx >= n_species:
            raise ValueError(
                f"species_index {idx} out of bounds for n_species={n_species}."
            )
        out = forms[idx]
    else:
        raise ValueError(
            f"Unknown observable_mode '{mode}'. "
            "Use 'total_species', 'total_charge', 'charge_proxy_no_f', or 'species'."
        )

    return fd.Constant(float(scale)) * out


def _build_scalar_target_in_control_space(
    ctx: Dict[str, object], value: float, *, name: str
):
    """Create a scalar Function in the same R-space used by kappa controls."""
    import firedrake as fd

    kappa_funcs = list(ctx["kappa_funcs"])
    if not kappa_funcs:
        raise ValueError("Context has no kappa control functions.")
    control_space = kappa_funcs[0].function_space()
    target = fd.Function(control_space, name=name)
    target.assign(float(value))
    return target


def _gradient_controls_to_array(raw_gradient: object, n_species: int) -> np.ndarray:
    """Convert adjoint gradient output to dense numpy vector."""
    if isinstance(raw_gradient, (list, tuple)):
        grads = list(raw_gradient)
    else:
        grads = [raw_gradient]

    out = np.zeros(n_species, dtype=float)
    for i in range(min(n_species, len(grads))):
        gi = grads[i]
        if hasattr(gi, "dat"):
            out[i] = float(gi.dat.data_ro[0])
        else:
            out[i] = float(gi)
    return out


def solve_point_objective_and_gradient(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied: float,
    target_flux: float,
    kappa_values: Sequence[float],
    blob_initial_condition: bool,
    fail_penalty: float,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_species_index: Optional[int],
    observable_scale: float,
) -> PointAdjointResult:
    """Solve one phi_applied point and extract dJ_i/dkappa with firedrake-adjoint."""
    import firedrake as fd
    import firedrake.adjoint as adj

    kappa_list = [float(v) for v in kappa_values]
    baseline_params = configure_robin_solver_params(
        base_solver_params,
        phi_applied=float(phi_applied),
        kappa_values=kappa_list,
    )
    n_species = int(baseline_params[0])
    abs_tol = float(max(steady.absolute_tolerance, 1e-16))
    rel_tol = float(steady.relative_tolerance)
    max_steps = int(max(1, steady.max_steps))
    required_steady = int(max(1, steady.consecutive_steps))

    baseline_options: Mapping[str, Any] = {}
    if isinstance(baseline_params[10], dict):
        baseline_options = copy.deepcopy(baseline_params[10])

    last_reason = "forward solve did not converge"
    last_flux = float("nan")
    last_steps = 0

    max_attempts = max(1, int(forward_recovery.max_attempts))
    for attempt in range(max_attempts):
        params = configure_robin_solver_params(
            base_solver_params,
            phi_applied=float(phi_applied),
            kappa_values=kappa_list,
        )
        if isinstance(params[10], dict):
            phase, phase_step, _cycle_index = _attempt_phase_state(attempt, forward_recovery)
            _relax_solver_options_for_attempt(
                params[10],
                phase=phase,
                phase_step=phase_step,
                recovery=forward_recovery,
                baseline_options=baseline_options if baseline_options else params[10],
            )

        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()

        ctx = build_context(params)
        ctx = build_forms(ctx, params)
        set_initial_conditions(ctx, params, blob=blob_initial_condition)

        U = ctx["U"]
        U_prev = ctx["U_prev"]
        F_res = ctx["F_res"]
        bcs = ctx["bcs"]

        jac = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params[10])

        observable_form = _build_observable_form(
            ctx,
            mode=observable_mode,
            species_index=observable_species_index,
            scale=float(observable_scale),
        )
        prev_flux: Optional[float] = None
        steady_count = 0
        rel_metric: Optional[float] = None
        abs_metric: Optional[float] = None
        simulated_flux = float("nan")
        steps_taken = 0
        failed_by_exception = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            try:
                solver.solve()
            except Exception as exc:
                failed_by_exception = True
                last_reason = f"{type(exc).__name__}: {exc}"
                break

            U_prev.assign(U)

            # Non-annotated assembly for convergence check itself.
            with adj.stop_annotating():
                simulated_flux = float(fd.assemble(observable_form))

            if prev_flux is not None:
                delta = abs(simulated_flux - prev_flux)
                scale = max(abs(simulated_flux), abs(prev_flux), abs_tol)
                rel_metric = delta / scale
                abs_metric = delta
                is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                steady_count = steady_count + 1 if is_steady else 0
            else:
                steady_count = 0

            prev_flux = simulated_flux
            if steady_count >= required_steady:
                # Point objective in annotated mode: 0.5*(flux-target)^2.
                target_flux_control = _build_scalar_target_in_control_space(
                    ctx, target_flux, name="target_flux_value"
                )
                target_flux_scalar = fd.assemble(
                    target_flux_control * fd.dx(domain=ctx["mesh"])
                )
                simulated_flux_scalar = fd.assemble(observable_form)
                point_objective = 0.5 * (simulated_flux_scalar - target_flux_scalar) ** 2

                controls = [adj.Control(ctrl) for ctrl in list(ctx["kappa_funcs"])]
                rf = adj.ReducedFunctional(point_objective, controls)
                control_state = [ctrl for ctrl in list(ctx["kappa_funcs"])]
                point_objective_value = float(rf(control_state))
                point_gradient = _gradient_controls_to_array(rf.derivative(), n_species)

                return PointAdjointResult(
                    phi_applied=float(phi_applied),
                    target_flux=float(target_flux),
                    simulated_flux=float(simulated_flux_scalar),
                    objective=point_objective_value,
                    gradient=point_gradient,
                    converged=True,
                    steps_taken=int(steps_taken),
                    reason="",
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    diagnostics_valid=True,
                )

        last_flux = simulated_flux
        last_steps = int(steps_taken)
        if not failed_by_exception:
            last_reason = "steady-state criterion not satisfied before max_steps"

    return PointAdjointResult(
        phi_applied=float(phi_applied),
        target_flux=float(target_flux),
        simulated_flux=last_flux,
        objective=float(fail_penalty),
        gradient=np.zeros(n_species, dtype=float),
        converged=False,
        steps_taken=int(last_steps),
        reason=last_reason,
        final_relative_change=None,
        final_absolute_change=None,
        diagnostics_valid=False,
    )


def evaluate_curve_objective_and_gradient(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
    point_executor: Optional[_PointSolveExecutor] = None,
) -> CurveAdjointResult:
    """Evaluate curve objective + gradient, with anisotropy recovery fallback."""
    n_species = int(request.base_solver_params[0])

    def _evaluate_once(kappa_eval: np.ndarray) -> CurveAdjointResult:
        points: List[PointAdjointResult] = []
        simulated_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)
        total_objective = 0.0
        total_gradient = np.zeros(n_species, dtype=float)
        n_failed = 0

        parallel_points: Optional[List[PointAdjointResult]] = None
        if point_executor is not None and bool(point_executor.enabled):
            parallel_points = point_executor.map_points(
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                kappa_values=np.asarray(kappa_eval, dtype=float),
            )

        if parallel_points is not None:
            points = list(parallel_points)
        else:
            for phi_i, target_i in zip(phi_applied_values.tolist(), target_flux.tolist()):
                point = solve_point_objective_and_gradient(
                    base_solver_params=request.base_solver_params,
                    steady=request.steady,
                    phi_applied=float(phi_i),
                    target_flux=float(target_i),
                    kappa_values=kappa_eval.tolist(),
                    blob_initial_condition=bool(request.blob_initial_condition),
                    fail_penalty=float(request.fail_penalty),
                    forward_recovery=request.forward_recovery,
                    observable_mode=str(request.observable_mode),
                    observable_species_index=request.observable_species_index,
                    observable_scale=float(request.observable_scale),
                )
                points.append(point)

        for i, point in enumerate(points):
            simulated_flux[i] = point.simulated_flux
            total_objective += float(point.objective)
            if point.converged:
                total_gradient += point.gradient
            else:
                n_failed += 1

        return CurveAdjointResult(
            objective=float(total_objective),
            gradient=total_gradient,
            simulated_flux=simulated_flux,
            points=points,
            n_failed=n_failed,
            effective_kappa=np.asarray(kappa_eval, dtype=float).copy(),
            used_anisotropy_recovery=False,
        )

    primary = _evaluate_once(np.asarray(kappa_values, dtype=float))

    n_points = int(len(phi_applied_values))
    fail_threshold_by_points = max(1, int(request.anisotropy_trigger_failed_points))
    fail_threshold_by_fraction = int(
        np.ceil(
            max(0.0, float(request.anisotropy_trigger_failed_fraction)) * max(1, n_points)
        )
    )
    fail_trigger = max(fail_threshold_by_points, fail_threshold_by_fraction)

    if primary.n_failed < fail_trigger:
        return primary

    kappa_aniso = _reduce_kappa_anisotropy(
        np.asarray(kappa_values, dtype=float),
        target_ratio=float(request.forward_recovery.anisotropy_target_ratio),
        blend=float(request.forward_recovery.anisotropy_blend),
    )
    if np.allclose(kappa_aniso, np.asarray(kappa_values, dtype=float), rtol=1e-12, atol=1e-12):
        return primary

    print(
        "[recovery] many point failures detected "
        f"({primary.n_failed}/{n_points}); retrying with anisotropy-reduced kappa "
        f"[{kappa_aniso[0]:.6f}, {kappa_aniso[1]:.6f}]"
    )
    secondary = _evaluate_once(kappa_aniso)
    secondary.used_anisotropy_recovery = True

    # Prefer the evaluation with fewer failed points; tie-break on objective.
    if secondary.n_failed < primary.n_failed:
        return secondary
    if secondary.n_failed == primary.n_failed and secondary.objective < primary.objective:
        return secondary
    return primary


def _build_replay_point_flux_functional(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied: float,
    kappa_values: Sequence[float],
    blob_initial_condition: bool,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_species_index: Optional[int],
    observable_scale: float,
    replay_extra_steady_steps: int,
) -> Tuple[Optional[_ReplayPointFunctional], str]:
    """Build one persistent replay-ready reduced functional for a sweep point.

    Replay point tapes are built dynamically by:
    1) solving until the same steady-state criterion is satisfied, and
    2) taking a few additional steady steps as a buffer for nearby replayed
       kappa values.

    Diagnostics functionals (previous-step observable + state deltas) are also
    taped so each replay evaluation can validate whether replay still appears to
    be in steady state.
    """
    import firedrake as fd
    import firedrake.adjoint as adj

    kappa_list = [float(v) for v in kappa_values]
    baseline_params = configure_robin_solver_params(
        base_solver_params,
        phi_applied=float(phi_applied),
        kappa_values=kappa_list,
    )
    abs_tol = float(max(steady.absolute_tolerance, 1e-16))
    rel_tol = float(steady.relative_tolerance)
    max_steps = int(max(1, steady.max_steps))
    required_steady = int(max(1, steady.consecutive_steps))
    extra_steady = int(max(0, replay_extra_steady_steps))
    target_steady = int(required_steady + extra_steady)

    baseline_options: Mapping[str, Any] = {}
    if isinstance(baseline_params[10], dict):
        baseline_options = copy.deepcopy(baseline_params[10])

    last_reason = (
        "dynamic replay build failed before reaching steady state "
        f"(need {target_steady} steady steps)"
    )
    max_attempts = max(1, int(forward_recovery.max_attempts))
    for attempt in range(max_attempts):
        params = configure_robin_solver_params(
            base_solver_params,
            phi_applied=float(phi_applied),
            kappa_values=kappa_list,
        )
        if isinstance(params[10], dict):
            phase, phase_step, _cycle_index = _attempt_phase_state(attempt, forward_recovery)
            _relax_solver_options_for_attempt(
                params[10],
                phase=phase,
                phase_step=phase_step,
                recovery=forward_recovery,
                baseline_options=baseline_options if baseline_options else params[10],
            )

        tape = adj.Tape()
        failed_by_exception = False
        try:
            with adj.set_working_tape(tape):
                adj.continue_annotation()

                ctx = build_context(params)
                ctx = build_forms(ctx, params)
                set_initial_conditions(ctx, params, blob=blob_initial_condition)

                U = ctx["U"]
                U_prev = ctx["U_prev"]
                F_res = ctx["F_res"]
                bcs = ctx["bcs"]

                jac = fd.derivative(F_res, U)
                problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
                solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params[10])

                observable_form = _build_observable_form(
                    ctx,
                    mode=observable_mode,
                    species_index=observable_species_index,
                    scale=float(observable_scale),
                    state=U,
                )
                observable_prev_form = _build_observable_form(
                    ctx,
                    mode=observable_mode,
                    species_index=observable_species_index,
                    scale=float(observable_scale),
                    state=U_prev,
                )
                state_delta_sq_form = fd.inner(U - U_prev, U - U_prev) * fd.dx(domain=ctx["mesh"])
                state_norm_sq_form = fd.inner(U, U) * fd.dx(domain=ctx["mesh"])

                prev_flux: Optional[float] = None
                steady_count = 0

                for step in range(1, max_steps + 1):
                    try:
                        solver.solve()
                    except Exception as exc:
                        failed_by_exception = True
                        last_reason = f"{type(exc).__name__}: {exc}"
                        break

                    flux_now = float(fd.assemble(observable_form))
                    if prev_flux is not None:
                        delta = abs(flux_now - prev_flux)
                        scale = max(abs(flux_now), abs(prev_flux), abs_tol)
                        rel_metric = delta / scale
                        abs_metric = delta
                        is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                        steady_count = steady_count + 1 if is_steady else 0
                    else:
                        steady_count = 0

                    if steady_count >= target_steady:
                        observable_scalar = fd.assemble(observable_form)
                        observable_prev_scalar = fd.assemble(observable_prev_form)
                        state_delta_sq_scalar = fd.assemble(state_delta_sq_form)
                        state_norm_sq_scalar = fd.assemble(state_norm_sq_form)
                        controls = [adj.Control(ctrl) for ctrl in list(ctx["kappa_funcs"])]
                        reduced_flux = adj.ReducedFunctional(observable_scalar, controls)
                        reduced_flux_prev = adj.ReducedFunctional(
                            observable_prev_scalar, controls
                        )
                        reduced_state_delta_l2 = adj.ReducedFunctional(
                            state_delta_sq_scalar, controls
                        )
                        reduced_state_norm_l2 = adj.ReducedFunctional(
                            state_norm_sq_scalar, controls
                        )
                        control_state = [ctrl for ctrl in list(ctx["kappa_funcs"])]
                        return (
                            _ReplayPointFunctional(
                                phi_applied=float(phi_applied),
                                tape=tape,
                                control_state=control_state,
                                reduced_flux=reduced_flux,
                                reduced_flux_prev=reduced_flux_prev,
                                reduced_state_delta_l2=reduced_state_delta_l2,
                                reduced_state_norm_l2=reduced_state_norm_l2,
                                steady_rel_tol=rel_tol,
                                steady_abs_tol=abs_tol,
                                steps_taken=int(step),
                            ),
                            "",
                        )

                    prev_flux = flux_now
                    # Advance to next timestep with the newly solved state.
                    U_prev.assign(U)

                if not failed_by_exception:
                    observable_scalar = fd.assemble(observable_form)
                    last_reason = (
                        "steady-state criterion not satisfied before max_steps "
                        f"(max_steps={max_steps}, final_flux={float(observable_scalar):.6g})"
                    )
        except Exception as exc:
            last_reason = f"{type(exc).__name__}: {exc}"

    return None, str(last_reason)


def _build_replay_bundle(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    kappa_values: np.ndarray,
) -> Tuple[Optional[_ReplayBundle], str]:
    """Build persistent replay point models for all phi_applied values."""
    points: List[_ReplayPointFunctional] = []
    for point_idx, phi_i in enumerate(phi_applied_values.tolist()):
        point_model, reason = _build_replay_point_flux_functional(
            base_solver_params=request.base_solver_params,
            steady=request.steady,
            phi_applied=float(phi_i),
            kappa_values=np.asarray(kappa_values, dtype=float).tolist(),
            blob_initial_condition=bool(request.blob_initial_condition),
            forward_recovery=request.forward_recovery,
            observable_mode=str(request.observable_mode),
            observable_species_index=request.observable_species_index,
            observable_scale=float(request.observable_scale),
            replay_extra_steady_steps=int(request.replay_extra_steady_steps),
        )
        if point_model is None:
            return (
                None,
                f"phi_applied={float(phi_i):.6f} (point {point_idx}) build failed: {reason}",
            )
        points.append(point_model)

    return (
        _ReplayBundle(
            points=points,
            anchor_kappa=np.asarray(kappa_values, dtype=float).copy(),
        ),
        "",
    )


def _evaluate_curve_with_replay_bundle(
    *,
    bundle: _ReplayBundle,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
) -> CurveAdjointResult:
    """Evaluate full curve objective/gradient via persistent replay models.

    Each point also runs replay diagnostics:
    - absolute/relative change between current and previous-step observable
    - state change norm between current and previous timestep states

    Points failing diagnostics are marked non-converged so the caller can
    fallback to fresh forward solves and potentially rebuild replay.
    """
    import firedrake.adjoint as adj

    n_species = int(np.asarray(kappa_values, dtype=float).size)
    simulated_flux = np.full(target_flux.shape, np.nan, dtype=float)
    total_gradient = np.zeros(n_species, dtype=float)
    total_objective = 0.0
    point_rows: List[PointAdjointResult] = []
    n_failed = 0

    if int(len(bundle.points)) != int(target_flux.size):
        raise RuntimeError(
            "Replay bundle size does not match target curve size: "
            f"{len(bundle.points)} vs {target_flux.size}."
        )

    for i, point in enumerate(bundle.points):
        try:
            with adj.set_working_tape(point.tape):
                adj.continue_annotation()
                for j in range(min(n_species, len(point.control_state))):
                    point.control_state[j].assign(float(kappa_values[j]))
                flux_val = float(point.reduced_flux(point.control_state))
                flux_prev_val = float(point.reduced_flux_prev(point.control_state))
                state_delta_sq_val = float(
                    point.reduced_state_delta_l2(point.control_state)
                )
                state_norm_sq_val = float(point.reduced_state_norm_l2(point.control_state))
                dflux_dk = _gradient_controls_to_array(point.reduced_flux.derivative(), n_species)
        except Exception as exc:
            raise RuntimeError(
                f"Replay point evaluation failed at phi_applied={point.phi_applied:.6f}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        diagnostics_valid = bool(
            np.isfinite(flux_val)
            and np.isfinite(flux_prev_val)
            and np.isfinite(state_delta_sq_val)
            and np.isfinite(state_norm_sq_val)
            and np.all(np.isfinite(dflux_dk))
            and (state_delta_sq_val >= -1e-14)
            and (state_norm_sq_val >= -1e-14)
        )

        target_i = float(target_flux[i])
        residual_i = flux_val - target_i if np.isfinite(flux_val) else float("nan")
        point_objective = (
            0.5 * (residual_i**2) if np.isfinite(residual_i) else float("inf")
        )

        flux_abs_change = abs(flux_val - flux_prev_val) if diagnostics_valid else float("inf")
        flux_rel_change = (
            flux_abs_change
            / max(abs(flux_val), abs(flux_prev_val), float(point.steady_abs_tol), 1e-16)
            if diagnostics_valid
            else float("inf")
        )
        state_delta_l2 = (
            float(np.sqrt(max(0.0, state_delta_sq_val))) if diagnostics_valid else float("inf")
        )
        state_norm_l2 = (
            float(np.sqrt(max(0.0, state_norm_sq_val))) if diagnostics_valid else float("inf")
        )
        state_rel_change = (
            state_delta_l2 / max(state_norm_l2, 1e-16) if diagnostics_valid else float("inf")
        )

        steady_ok = bool(
            diagnostics_valid
            and (
                flux_rel_change <= float(point.steady_rel_tol)
                or flux_abs_change <= float(point.steady_abs_tol)
            )
        )
        point_converged = bool(diagnostics_valid and steady_ok)

        point_gradient = residual_i * dflux_dk if point_converged else np.zeros(n_species, dtype=float)

        simulated_flux[i] = flux_val
        if point_converged and np.isfinite(point_objective):
            total_objective += float(point_objective)
        else:
            # Force fallback path to run fresh forward solves for this candidate.
            total_objective += 1e12
        total_gradient += point_gradient
        if not point_converged:
            n_failed = n_failed + 1

        reason = "replay"
        if not diagnostics_valid:
            reason = "replay_diag_invalid"
        elif not steady_ok:
            reason = (
                "replay_not_steady"
                f"(rel={flux_rel_change:.3e},abs={flux_abs_change:.3e},"
                f"state_rel={state_rel_change:.3e})"
            )
        point_rows.append(
            PointAdjointResult(
                phi_applied=float(point.phi_applied),
                target_flux=target_i,
                simulated_flux=float(flux_val),
                objective=float(point_objective),
                gradient=np.asarray(point_gradient, dtype=float),
                converged=point_converged,
                steps_taken=int(point.steps_taken),
                reason=reason,
                final_relative_change=float(flux_rel_change)
                if np.isfinite(flux_rel_change)
                else None,
                final_absolute_change=float(flux_abs_change)
                if np.isfinite(flux_abs_change)
                else None,
                diagnostics_valid=diagnostics_valid,
            )
        )

    return CurveAdjointResult(
        objective=float(total_objective),
        gradient=np.asarray(total_gradient, dtype=float),
        simulated_flux=simulated_flux,
        points=point_rows,
        n_failed=int(n_failed),
        effective_kappa=np.asarray(kappa_values, dtype=float).copy(),
        used_anisotropy_recovery=False,
        used_replay_mode=True,
    )


class _DynamicReplayCurveEvaluator:
    """Hybrid curve evaluator with automatic replay enable/disable/re-enable."""

    def __init__(
        self,
        *,
        request: RobinFluxCurveInferenceRequest,
        phi_applied_values: np.ndarray,
        target_flux: np.ndarray,
        point_executor: Optional[_PointSolveExecutor] = None,
    ) -> None:
        self.request = request
        self.phi_applied_values = np.asarray(phi_applied_values, dtype=float)
        self.target_flux = np.asarray(target_flux, dtype=float)
        self.point_executor = point_executor
        self.replay_requested = bool(request.replay_mode_enabled)
        self.replay_bundle: Optional[_ReplayBundle] = None
        self.replay_enabled = False
        self.fallback_success_streak = 0
        self.reenable_after_successes = max(1, int(request.replay_reenable_after_successes))
        self.replay_rebuild_count = 0
        self.replay_diag_rebuild_count = 0
        self.replay_exception_rebuild_count = 0

    def initialize(self, *, kappa_anchor: np.ndarray) -> None:
        """Build replay bundle once at startup when replay mode is enabled."""
        if not self.replay_requested:
            return
        print(
            "[replay] building persistent point models at "
            f"kappa=[{float(kappa_anchor[0]):.6f}, {float(kappa_anchor[1]):.6f}] "
            "using dynamic steady-state replay "
            f"(extra_steady_steps={int(self.request.replay_extra_steady_steps)})"
        )
        self._enable_replay(np.asarray(kappa_anchor, dtype=float))

    def evaluate(self, *, kappa_values: np.ndarray) -> CurveAdjointResult:
        """Evaluate objective/gradient, preferring replay when available."""
        kappa_eval = np.asarray(kappa_values, dtype=float)

        if self.replay_enabled and self.replay_bundle is not None:
            try:
                replay_curve = _evaluate_curve_with_replay_bundle(
                    bundle=self.replay_bundle,
                    target_flux=self.target_flux,
                    kappa_values=kappa_eval,
                )
                if int(replay_curve.n_failed) == 0:
                    return replay_curve
                self.replay_diag_rebuild_count += 1
                self._disable_replay(
                    "replay diagnostics failed "
                    f"({int(replay_curve.n_failed)}/{len(replay_curve.points)} points); "
                    "refreshing via full steady-state solves."
                )
            except Exception as exc:
                self.replay_exception_rebuild_count += 1
                self._disable_replay(
                    "replay evaluation failure; "
                    f"falling back to resilient solves ({type(exc).__name__}: {exc})"
                )

        fallback = evaluate_curve_objective_and_gradient(
            request=self.request,
            phi_applied_values=self.phi_applied_values,
            target_flux=self.target_flux,
            kappa_values=kappa_eval,
            point_executor=self.point_executor,
        )
        fallback.used_replay_mode = False

        if not self.replay_requested:
            return fallback

        if int(fallback.n_failed) == 0:
            self.fallback_success_streak += 1
            if self.fallback_success_streak >= self.reenable_after_successes:
                print(
                    "[replay] fallback reconverged; "
                    "attempting replay re-enable at current kappa."
                )
                enabled = self._enable_replay(np.asarray(fallback.effective_kappa, dtype=float))
                if enabled:
                    self.fallback_success_streak = 0
                else:
                    # Keep retry cadence optimistic without rebuilding every call.
                    self.fallback_success_streak = self.reenable_after_successes - 1
        else:
            self.fallback_success_streak = 0

        return fallback

    def _enable_replay(self, kappa_anchor: np.ndarray) -> bool:
        if not self.replay_requested:
            return False
        bundle, reason = _build_replay_bundle(
            request=self.request,
            phi_applied_values=self.phi_applied_values,
            kappa_values=np.asarray(kappa_anchor, dtype=float),
        )
        if bundle is None:
            self.replay_bundle = None
            self.replay_enabled = False
            print(f"[replay] build failed: {reason}")
            return False

        self.replay_bundle = bundle
        self.replay_enabled = True
        self.replay_rebuild_count += 1
        print(
            "[replay] enabled with "
            f"{len(bundle.points)} persistent point models."
        )
        return True

    def _disable_replay(self, reason: str) -> None:
        if self.replay_enabled or self.replay_bundle is not None:
            print(f"[replay] disabled: {reason}")
        self.replay_enabled = False
        self.replay_bundle = None
        self.fallback_success_streak = 0

    def stats(self) -> Dict[str, int]:
        """Return replay lifecycle counters for runtime diagnostics."""
        return {
            "replay_rebuild_count": int(self.replay_rebuild_count),
            "replay_diag_rebuild_count": int(self.replay_diag_rebuild_count),
            "replay_exception_rebuild_count": int(self.replay_exception_rebuild_count),
        }


def evaluate_curve_loss_forward(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
    blob_initial_condition: bool,
    fail_penalty: float,
    observable_scale: float,
) -> Tuple[float, np.ndarray, int]:
    """Evaluate scalar curve loss (no derivatives), used by line search."""
    results = sweep_phi_applied_steady_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        kappa_values=kappa_values.tolist(),
        blob_initial_condition=blob_initial_condition,
    )
    simulated_flux = float(observable_scale) * results_to_flux_array(results)
    n_failed = sum(0 if r.converged else 1 for r in results)

    if n_failed > 0 or np.any(~np.isfinite(simulated_flux)):
        return float(fail_penalty * max(1, n_failed)), simulated_flux, int(n_failed)

    residual = simulated_flux - target_flux
    loss = 0.5 * float(np.sum(residual * residual))
    return float(loss), simulated_flux, 0


def _build_scipy_options(request: RobinFluxCurveInferenceRequest) -> Dict[str, Any]:
    """Construct SciPy minimize options with sensible defaults."""
    options: Dict[str, Any] = {}
    if request.optimizer_options is not None:
        options.update(dict(request.optimizer_options))
    options.setdefault("maxiter", int(request.max_iters))
    options.setdefault("disp", True)
    if "gtol" not in options and request.optimizer_method in (
        "L-BFGS-B",
        "BFGS",
        "CG",
        "Newton-CG",
        "TNC",
    ):
        options["gtol"] = float(request.gtol)
    return options


def run_scipy_adjoint_optimization(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    initial_kappa: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    np.ndarray,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Any,
    _LiveFitPlot,
    Dict[str, int],
]:
    """Optimize kappa using SciPy minimize with analytic adjoint Jacobian."""
    from scipy.optimize import minimize

    n_species = int(initial_kappa.size)
    x0 = clip_kappa(np.asarray(initial_kappa, dtype=float), lower_bounds, upper_bounds)
    bounds = [(float(lower_bounds[i]), float(upper_bounds[i])) for i in range(n_species)]
    options = _build_scipy_options(request)

    history_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []

    # Cache expensive PDE evaluations so separate fun/jac calls at same x reuse work.
    cache: Dict[Tuple[float, ...], Dict[str, object]] = {}
    eval_counter = {"n": 0}
    iteration_counter = {"n": 0}

    best_kappa = x0.copy()
    best_loss = float("inf")
    best_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)
    point_executor = _PointSolveExecutor(
        request=request, n_points=int(phi_applied_values.size)
    )
    curve_evaluator = _DynamicReplayCurveEvaluator(
        request=request,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        point_executor=point_executor,
    )
    curve_evaluator.initialize(kappa_anchor=x0.copy())
    live_plot = _LiveFitPlot(
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        y_label=str(request.observable_label),
        title=str(request.observable_title),
        enabled=bool(request.live_plot),
        pause_seconds=float(request.live_plot_pause_seconds),
        show_eval_lines=bool(request.live_plot_eval_lines),
        eval_line_alpha=float(request.live_plot_eval_line_alpha),
        eval_max_lines=int(request.live_plot_eval_max_lines),
        capture_frames_dir=request.live_plot_capture_frames_dir,
        capture_every_n_updates=int(request.live_plot_capture_every_n_updates),
        capture_max_frames=int(request.live_plot_capture_max_frames),
    )

    def _key_from_x(x: np.ndarray) -> Tuple[float, ...]:
        x_clip = clip_kappa(np.asarray(x, dtype=float), lower_bounds, upper_bounds)
        return tuple(float(f"{v:.12g}") for v in x_clip.tolist())

    def _record_point_rows(
        *,
        eval_id: int,
        iter_id: int,
        points: Sequence[PointAdjointResult],
    ) -> None:
        for idx, point in enumerate(points):
            point_rows.append(
                {
                    "evaluation": eval_id,
                    "iteration": iter_id,
                    "point_index": idx,
                    "phi_applied": point.phi_applied,
                    "target_observable": point.target_flux,
                    "simulated_observable": point.simulated_flux,
                    "target_flux": point.target_flux,
                    "simulated_flux": point.simulated_flux,
                    "point_objective": point.objective,
                    "dJ_dkappa0": float(point.gradient[0]) if n_species >= 1 else float("nan"),
                    "dJ_dkappa1": float(point.gradient[1]) if n_species >= 2 else float("nan"),
                    "converged": int(bool(point.converged)),
                    "steps_taken": int(point.steps_taken),
                    "final_relative_change": (
                        float(point.final_relative_change)
                        if point.final_relative_change is not None
                        else float("nan")
                    ),
                    "final_absolute_change": (
                        float(point.final_absolute_change)
                        if point.final_absolute_change is not None
                        else float("nan")
                    ),
                    "diagnostics_valid": int(bool(point.diagnostics_valid)),
                    "reason": point.reason,
                }
            )
            if bool(request.print_point_gradients):
                print(
                    f"  phi={point.phi_applied:8.5f} "
                    f"target={point.target_flux:12.6f} "
                    f"sim={point.simulated_flux:12.6f} "
                    f"dJ/dk=[{point.gradient[0]:11.6f}, {point.gradient[1]:11.6f}] "
                    f"conv={int(point.converged)} "
                    f"diag={int(bool(point.diagnostics_valid))} "
                    f"rel={(float(point.final_relative_change) if point.final_relative_change is not None else float('nan')):9.3e} "
                    f"abs={(float(point.final_absolute_change) if point.final_absolute_change is not None else float('nan')):9.3e}"
                )

    def _evaluate(x: np.ndarray) -> Dict[str, object]:
        nonlocal best_kappa, best_loss, best_flux

        x_clip = clip_kappa(np.asarray(x, dtype=float), lower_bounds, upper_bounds)
        key = _key_from_x(x_clip)
        if key in cache:
            return cache[key]

        curve = curve_evaluator.evaluate(kappa_values=x_clip)

        eval_counter["n"] += 1
        eval_id = int(eval_counter["n"])
        iter_id = int(iteration_counter["n"])
        objective = float(curve.objective)
        gradient = np.asarray(curve.gradient, dtype=float)
        grad_norm = float(np.linalg.norm(gradient))
        n_failed = int(curve.n_failed)
        simulated_flux = np.asarray(curve.simulated_flux, dtype=float)

        recovery_tag = " aniso_recovery=1" if curve.used_anisotropy_recovery else ""
        replay_tag = " replay=1" if curve.used_replay_mode else ""
        print(
            f"[eval={eval_id:03d}] "
            f"kappa=[{x_clip[0]:10.6f}, {x_clip[1]:10.6f}] "
            f"loss={objective:14.6e} "
            f"grad=[{gradient[0]:12.6f}, {gradient[1]:12.6f}] "
            f"|grad|={grad_norm:12.6f} "
            f"fails={n_failed:02d}"
            f"{recovery_tag}"
            f"{replay_tag}"
        )
        if curve.used_anisotropy_recovery:
            ek = curve.effective_kappa
            print(
                f"  effective_kappa=[{float(ek[0]):10.6f}, {float(ek[1]):10.6f}]"
            )

        _record_point_rows(eval_id=eval_id, iter_id=iter_id, points=curve.points)

        live_plot.add_eval_curve(flux=simulated_flux, eval_id=eval_id)
        live_plot.update(
            current_flux=simulated_flux,
            best_flux=best_flux.copy(),
            iteration=int(iter_id),
            objective=objective,
            n_failed=n_failed,
            kappa=x_clip.copy(),
            eval_id=eval_id,
        )

        history_rows.append(
            {
                "evaluation": eval_id,
                "iteration": iter_id,
                "kappa0": float(x_clip[0]),
                "kappa1": float(x_clip[1]),
                "objective": objective,
                "grad0": float(gradient[0]),
                "grad1": float(gradient[1]),
                "grad_norm": grad_norm,
                "n_failed_points": n_failed,
                "used_anisotropy_recovery": int(bool(curve.used_anisotropy_recovery)),
                "used_replay_mode": int(bool(curve.used_replay_mode)),
                "effective_kappa0": float(curve.effective_kappa[0]),
                "effective_kappa1": float(curve.effective_kappa[1]),
                "from_cache": 0,
            }
        )

        if n_failed == 0 and np.isfinite(objective) and objective < best_loss:
            best_loss = objective
            best_kappa = x_clip.copy()
            best_flux = simulated_flux.copy()

        payload = {
            "x": x_clip.copy(),
            "objective": objective,
            "gradient": gradient.copy(),
            "simulated_flux": simulated_flux.copy(),
            "n_failed": n_failed,
            "used_anisotropy_recovery": bool(curve.used_anisotropy_recovery),
            "used_replay_mode": bool(curve.used_replay_mode),
            "effective_kappa": np.asarray(curve.effective_kappa, dtype=float).copy(),
            "points": list(curve.points),
            "eval_id": eval_id,
        }
        cache[key] = payload
        return payload

    def _fun(x: np.ndarray) -> float:
        return float(_evaluate(x)["objective"])

    def _jac(x: np.ndarray) -> np.ndarray:
        return np.asarray(_evaluate(x)["gradient"], dtype=float)

    def _callback(xk: np.ndarray) -> None:
        iteration_counter["n"] += 1
        payload = _evaluate(xk)
        grad = np.asarray(payload["gradient"], dtype=float)
        recovery_tag = " aniso_recovery=1" if bool(payload.get("used_anisotropy_recovery", False)) else ""
        replay_tag = " replay=1" if bool(payload.get("used_replay_mode", False)) else ""
        print(
            f"[iter={iteration_counter['n']:02d}] "
            f"kappa=[{payload['x'][0]:10.6f}, {payload['x'][1]:10.6f}] "
            f"loss={float(payload['objective']):14.6e} "
            f"|grad|={float(np.linalg.norm(grad)):12.6f} "
            f"fails={int(payload['n_failed']):02d}"
            f"{recovery_tag}"
            f"{replay_tag}"
        )
        live_plot.update(
            current_flux=np.asarray(payload["simulated_flux"], dtype=float),
            best_flux=best_flux.copy(),
            iteration=int(iteration_counter["n"]),
            objective=float(payload["objective"]),
            n_failed=int(payload["n_failed"]),
            kappa=np.asarray(payload["x"], dtype=float),
            eval_id=int(payload.get("eval_id", -1)),
        )

    # Draw an initial curve before the first optimizer step.
    initial_payload = _evaluate(x0)
    live_plot.update(
        current_flux=np.asarray(initial_payload["simulated_flux"], dtype=float),
        best_flux=best_flux.copy(),
        iteration=0,
        objective=float(initial_payload["objective"]),
        n_failed=int(initial_payload["n_failed"]),
        kappa=np.asarray(initial_payload["x"], dtype=float),
        eval_id=int(initial_payload.get("eval_id", -1)),
    )

    try:
        result = minimize(
            _fun,
            x0=x0,
            jac=_jac,
            method=str(request.optimizer_method),
            bounds=bounds,
            tol=request.optimizer_tolerance,
            callback=_callback,
            options=options,
        )

        # Ensure final point is evaluated and available in outputs.
        final_x = clip_kappa(np.asarray(result.x, dtype=float), lower_bounds, upper_bounds)
        final_payload = _evaluate(final_x)

        if (
            int(final_payload["n_failed"]) == 0
            and np.isfinite(float(final_payload["objective"]))
            and float(final_payload["objective"]) < best_loss
        ):
            best_loss = float(final_payload["objective"])
            best_kappa = np.asarray(final_payload["x"], dtype=float).copy()
            best_flux = np.asarray(final_payload["simulated_flux"], dtype=float).copy()

        # If no fully converged point was seen, fallback to final-forward loss.
        if not np.isfinite(best_loss):
            best_loss, best_flux, _ = evaluate_curve_loss_forward(
                base_solver_params=request.base_solver_params,
                steady=request.steady,
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                kappa_values=best_kappa,
                blob_initial_condition=bool(request.blob_initial_condition),
                fail_penalty=float(request.fail_penalty),
                observable_scale=float(request.observable_scale),
            )

        # Ensure the final state is drawn even if callback was not called on last point.
        live_plot.update(
            current_flux=np.asarray(final_payload["simulated_flux"], dtype=float),
            best_flux=best_flux.copy(),
            iteration=int(iteration_counter["n"]),
            objective=float(final_payload["objective"]),
            n_failed=int(final_payload["n_failed"]),
            kappa=np.asarray(final_payload["x"], dtype=float),
            eval_id=int(final_payload.get("eval_id", -1)),
        )
    finally:
        point_executor.close()

    replay_stats = curve_evaluator.stats()
    print(
        "[replay] summary: "
        f"rebuilds={int(replay_stats.get('replay_rebuild_count', 0))} "
        f"diag_rebuilds={int(replay_stats.get('replay_diag_rebuild_count', 0))} "
        f"exception_rebuilds={int(replay_stats.get('replay_exception_rebuild_count', 0))}"
    )
    return (
        best_kappa,
        float(best_loss),
        best_flux,
        history_rows,
        point_rows,
        result,
        live_plot,
        replay_stats,
    )


def write_history_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write optimizer-iteration history to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = [
        "evaluation",
        "iteration",
        "kappa0",
        "kappa1",
        "objective",
        "grad0",
        "grad1",
        "grad_norm",
        "n_failed_points",
        "used_anisotropy_recovery",
        "used_replay_mode",
        "effective_kappa0",
        "effective_kappa1",
        "from_cache",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_point_gradient_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write per-point adjoint gradient diagnostics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = [
        "evaluation",
        "iteration",
        "point_index",
        "phi_applied",
        "target_observable",
        "simulated_observable",
        "target_flux",
        "simulated_flux",
        "point_objective",
        "dJ_dkappa0",
        "dJ_dkappa1",
        "converged",
        "steps_taken",
        "final_relative_change",
        "final_absolute_change",
        "diagnostics_valid",
        "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _as_int(value: object, default: int = 0) -> int:
    """Best-effort conversion to integer with fallback."""
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: object, default: float = float("nan")) -> float:
    """Best-effort conversion to float with fallback."""
    try:
        return float(value)
    except Exception:
        return float(default)


def export_live_fit_gif(
    *,
    path: str,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    history_rows: Sequence[Dict[str, object]],
    point_rows: Sequence[Dict[str, object]],
    seconds: float,
    n_frames: int,
    dpi: int,
    y_label: str = "steady-state flux (observable)",
    title: str = "Robin kappa fit progress",
) -> Optional[str]:
    """Render a standalone convergence GIF from per-evaluation diagnostics.

    This path is robust in headless/GUI-restricted environments because it
    re-renders the curves from recorded optimization rows rather than relying on
    interactive window screenshots.
    """
    if plt is None:
        return None

    try:
        from PIL import Image
    except Exception:
        return None

    phi = np.asarray(phi_applied_values, dtype=float).copy()
    target = np.asarray(target_flux, dtype=float).copy()
    n_points = int(phi.size)
    if n_points == 0:
        return None

    curves_by_eval: Dict[int, np.ndarray] = {}
    for row in point_rows:
        eval_id = _as_int(row.get("evaluation"), -1)
        point_idx = _as_int(row.get("point_index"), -1)
        if eval_id < 0 or point_idx < 0 or point_idx >= n_points:
            continue
        if eval_id not in curves_by_eval:
            curves_by_eval[eval_id] = np.full(n_points, np.nan, dtype=float)
        curves_by_eval[eval_id][point_idx] = _as_float(
            row.get("simulated_observable", row.get("simulated_flux"))
        )

    eval_ids = sorted(curves_by_eval.keys())
    if not eval_ids:
        return None

    history_by_eval: Dict[int, Dict[str, object]] = {}
    for row in history_rows:
        eval_id = _as_int(row.get("evaluation"), -1)
        if eval_id >= 0:
            history_by_eval[eval_id] = dict(row)

    y_arrays = [target]
    for eval_id in eval_ids:
        vals = curves_by_eval[eval_id]
        finite = vals[np.isfinite(vals)]
        if finite.size:
            y_arrays.append(finite)
    y_concat = np.concatenate([np.asarray(a, dtype=float).ravel() for a in y_arrays])
    y_min = float(np.min(y_concat))
    y_max = float(np.max(y_concat))
    y_span = max(1e-8, y_max - y_min)
    y_pad = 0.08 * y_span
    y_lim = (y_min - y_pad, y_max + y_pad)

    n_out = int(max(2, n_frames))
    pick = np.round(np.linspace(0, len(eval_ids) - 1, n_out)).astype(int)
    selected_eval_ids = [eval_ids[i] for i in pick]
    eval_pos = {eval_id: i for i, eval_id in enumerate(eval_ids)}

    best_eval_by_pos: List[int] = []
    best_eval = eval_ids[0]
    best_obj = float("inf")
    for eval_id in eval_ids:
        meta = history_by_eval.get(eval_id, {})
        obj = _as_float(meta.get("objective"), float("inf"))
        n_failed = _as_int(meta.get("n_failed_points"), 9999)
        if np.isfinite(obj) and n_failed == 0 and obj < best_obj:
            best_obj = obj
            best_eval = eval_id
        best_eval_by_pos.append(best_eval)

    frames: List[Image.Image] = []
    for eval_id in selected_eval_ids:
        pos = eval_pos[eval_id]
        best_eval_here = best_eval_by_pos[pos]
        current = curves_by_eval[eval_id]
        best_curve = curves_by_eval[best_eval_here]
        meta = history_by_eval.get(eval_id, {})

        fig, ax = plt.subplots(figsize=(7, 4), dpi=max(72, int(dpi)))

        for prev_eval in eval_ids[:pos]:
            prev = curves_by_eval[prev_eval]
            if np.any(np.isfinite(prev)):
                ax.plot(phi, prev, color="#BBBBBB", linewidth=1.0, alpha=0.22)

        ax.plot(
            phi,
            target,
            "o-",
            color="#1f77b4",
            linewidth=2.2,
            markersize=4.5,
            label="target data",
        )
        ax.plot(
            phi,
            best_curve,
            "s-",
            color="#2ca02c",
            linewidth=2.0,
            markersize=4.0,
            label="best fit so far",
        )
        ax.plot(
            phi,
            current,
            "D--",
            color="#ff7f0e",
            linewidth=1.8,
            markersize=3.8,
            label="current evaluation",
        )

        ax.set_xlim(float(np.min(phi)), float(np.max(phi)))
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("applied voltage phi_applied")
        ax.set_ylabel(str(y_label))
        ax.set_title(str(title))

        loss = _as_float(meta.get("objective"), float("nan"))
        kappa0 = _as_float(meta.get("kappa0"), float("nan"))
        kappa1 = _as_float(meta.get("kappa1"), float("nan"))
        n_failed = _as_int(meta.get("n_failed_points"), -1)
        status = (
            f"eval={eval_id:03d}  "
            f"loss={loss:.6e}  "
            f"kappa=[{kappa0:.6f}, {kappa1:.6f}]  "
            f"fails={n_failed:02d}"
        )
        ax.text(
            0.01,
            0.99,
            status,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
        )

        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = Image.fromarray(rgba, mode="RGBA").convert("P", palette=Image.Palette.ADAPTIVE)
        frames.append(frame)
        plt.close(fig)

    if not frames:
        return None

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    duration_ms = int(max(1, round((float(max(0.1, seconds)) * 1000.0) / len(frames))))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return path


def run_robin_kappa_flux_curve_inference(
    request: RobinFluxCurveInferenceRequest,
) -> RobinFluxCurveInferenceResult:
    """Run end-to-end Robin-kappa curve inference with adjoint gradients."""
    request_runtime = copy.deepcopy(request)
    if bool(request_runtime.replay_mode_enabled):
        print(
            "[replay] requested but currently disabled globally; "
            "running with replay_mode_enabled=False."
        )
    request_runtime.replay_mode_enabled = False
    request_runtime.steady.flux_observable = str(request_runtime.observable_mode)
    request_runtime.steady.species_index = request_runtime.observable_species_index

    phi_applied_values = np.asarray(request_runtime.phi_applied_values, dtype=float)
    target_data = ensure_target_curve(
        target_csv_path=request_runtime.target_csv_path,
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        true_kappa=request_runtime.true_value,
        noise_percent=float(request_runtime.target_noise_percent),
        seed=int(request_runtime.target_seed),
        force_regenerate=bool(request_runtime.regenerate_target),
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
    )
    target_phi_applied = np.asarray(target_data["phi_applied"], dtype=float)
    target_flux = float(request_runtime.observable_scale) * np.asarray(
        target_data["flux"], dtype=float
    )
    if target_phi_applied.size != target_flux.size:
        raise ValueError("Target phi_applied and flux lengths do not match.")
    phi_applied_values = target_phi_applied.copy()

    initial_kappa_list = _normalize_kappa(request_runtime.initial_guess, name="initial_guess")
    if initial_kappa_list is None:
        raise ValueError("initial_guess must be set.")
    initial_kappa = np.asarray(initial_kappa_list, dtype=float)

    n_species = int(request_runtime.base_solver_params[0])
    lower = np.full(n_species, float(request_runtime.kappa_lower), dtype=float)
    upper = np.full(n_species, float(request_runtime.kappa_upper), dtype=float)

    print("=== Adjoint-Gradient Robin Kappa Inference ===")
    print(f"Target points: {len(phi_applied_values)}")
    print(f"Initial kappa: {initial_kappa.tolist()}")
    print(f"Bounds: lower={lower.tolist()} upper={upper.tolist()}")
    print(
        "Objective: 0.5 * sum_i (observable(phi_i, kappa) - target_i)^2, "
        f"mode={request_runtime.observable_mode}, "
        f"scale={float(request_runtime.observable_scale):.16g}, "
        "with dJ/dkappa from firedrake-adjoint per point."
    )

    best_kappa, best_loss, best_sim_flux, history_rows, point_rows, opt_result, live_plot, replay_stats = run_scipy_adjoint_optimization(
        request=request_runtime,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        initial_kappa=initial_kappa,
        lower_bounds=lower,
        upper_bounds=upper,
    )
    print(
        "SciPy minimize summary: "
        f"success={bool(getattr(opt_result, 'success', False))} "
        f"status={getattr(opt_result, 'status', 'n/a')} "
        f"message={str(getattr(opt_result, 'message', '')).strip()}"
    )

    if bool(request_runtime.replay_mode_enabled) and bool(request_runtime.replay_post_refine_enabled):
        refine_iters = int(max(1, request_runtime.replay_post_refine_max_iters))
        print(
            "[refine] starting replay-off refinement/verification phase "
            f"from kappa=[{float(best_kappa[0]):.6f}, {float(best_kappa[1]):.6f}] "
            f"for up to {refine_iters} iterations."
        )
        refine_request = copy.deepcopy(request_runtime)
        refine_request.replay_mode_enabled = False
        refine_request.live_plot = False
        refine_request.live_plot_eval_lines = False
        refine_request.live_plot_capture_frames_dir = None
        refine_request.live_plot_export_gif_path = None
        refine_request.max_iters = refine_iters
        refine_opts = dict(refine_request.optimizer_options or {})
        refine_opts["maxiter"] = refine_iters
        refine_request.optimizer_options = refine_opts

        refine_best_kappa, refine_best_loss, refine_best_sim_flux, refine_history_rows, refine_point_rows, refine_opt_result, _, refine_replay_stats = run_scipy_adjoint_optimization(
            request=refine_request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            initial_kappa=np.asarray(best_kappa, dtype=float).copy(),
            lower_bounds=lower,
            upper_bounds=upper,
        )
        replay_stats = {
            "replay_rebuild_count": int(replay_stats.get("replay_rebuild_count", 0))
            + int(refine_replay_stats.get("replay_rebuild_count", 0)),
            "replay_diag_rebuild_count": int(
                replay_stats.get("replay_diag_rebuild_count", 0)
            )
            + int(refine_replay_stats.get("replay_diag_rebuild_count", 0)),
            "replay_exception_rebuild_count": int(
                replay_stats.get("replay_exception_rebuild_count", 0)
            )
            + int(refine_replay_stats.get("replay_exception_rebuild_count", 0)),
        }
        print(
            "[refine] SciPy summary: "
            f"success={bool(getattr(refine_opt_result, 'success', False))} "
            f"status={getattr(refine_opt_result, 'status', 'n/a')} "
            f"message={str(getattr(refine_opt_result, 'message', '')).strip()}"
        )

        eval_offset = max((_as_int(r.get("evaluation"), 0) for r in history_rows), default=0)
        iter_offset = max((_as_int(r.get("iteration"), 0) for r in history_rows), default=0)
        for row in refine_history_rows:
            row_out = dict(row)
            row_out["evaluation"] = _as_int(row_out.get("evaluation"), 0) + eval_offset
            row_out["iteration"] = _as_int(row_out.get("iteration"), 0) + iter_offset
            history_rows.append(row_out)
        for row in refine_point_rows:
            row_out = dict(row)
            row_out["evaluation"] = _as_int(row_out.get("evaluation"), 0) + eval_offset
            row_out["iteration"] = _as_int(row_out.get("iteration"), 0) + iter_offset
            point_rows.append(row_out)

        take_refined = False
        if np.isfinite(refine_best_loss) and (
            (not np.isfinite(best_loss)) or (float(refine_best_loss) <= float(best_loss))
        ):
            take_refined = True
        elif bool(getattr(refine_opt_result, "success", False)):
            # If both are finite but replay-off succeeds, prefer the refined endpoint
            # as the final reported full-objective solution.
            take_refined = True

        if take_refined:
            best_kappa = np.asarray(refine_best_kappa, dtype=float).copy()
            best_loss = float(refine_best_loss)
            best_sim_flux = np.asarray(refine_best_sim_flux, dtype=float).copy()
            opt_result = refine_opt_result
            print("[refine] accepted replay-off refined solution.")
        else:
            print("[refine] retained replay-phase solution (refinement not better).")

    final_loss, final_sim_flux, forward_failures_at_best = evaluate_curve_loss_forward(
        base_solver_params=request_runtime.base_solver_params,
        steady=request_runtime.steady,
        phi_applied_values=phi_applied_values,
        target_flux=target_flux,
        kappa_values=best_kappa,
        blob_initial_condition=bool(request_runtime.blob_initial_condition),
        fail_penalty=float(request_runtime.fail_penalty),
        observable_scale=float(request_runtime.observable_scale),
    )
    if np.isfinite(final_loss) and forward_failures_at_best == 0:
        best_loss = float(final_loss)
        best_sim_flux = final_sim_flux

    os.makedirs(request_runtime.output_dir, exist_ok=True)
    fit_csv_path = os.path.join(
        request_runtime.output_dir, "phi_applied_vs_steady_observable_fit.csv"
    )
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_observable,simulated_observable\n")
        for p, t, s in zip(
            phi_applied_values.tolist(), target_flux.tolist(), best_sim_flux.tolist()
        ):
            f.write(f"{p:.16g},{t:.16g},{s:.16g}\n")
    print(f"Saved fitted curve CSV: {fit_csv_path}")

    history_csv_path = os.path.join(
        request_runtime.output_dir, "robin_kappa_gradient_optimization_history.csv"
    )
    write_history_csv(history_csv_path, history_rows)
    print(f"Saved optimization history CSV: {history_csv_path}")

    point_csv_path = os.path.join(request_runtime.output_dir, "robin_kappa_point_gradients.csv")
    write_point_gradient_csv(point_csv_path, point_rows)
    print(f"Saved point-gradient CSV: {point_csv_path}")

    replay_diag_csv_path = os.path.join(
        request_runtime.output_dir, "replay_diagnostics_summary.csv"
    )
    with open(replay_diag_csv_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(
            f"replay_rebuild_count,{int(replay_stats.get('replay_rebuild_count', 0))}\n"
        )
        f.write(
            "replay_diag_rebuild_count,"
            f"{int(replay_stats.get('replay_diag_rebuild_count', 0))}\n"
        )
        f.write(
            "replay_exception_rebuild_count,"
            f"{int(replay_stats.get('replay_exception_rebuild_count', 0))}\n"
        )
    print(f"Saved replay diagnostics summary CSV: {replay_diag_csv_path}")

    fit_plot_path: Optional[str] = None
    if plt is None:
        print("matplotlib not available; skipping fit plot generation.")
    elif bool(request_runtime.live_plot) and getattr(live_plot, "enabled", False):
        fit_plot_path = os.path.join(
            request_runtime.output_dir, "phi_applied_vs_steady_observable_fit.png"
        )
        live_plot.update(
            current_flux=best_sim_flux.copy(),
            best_flux=best_sim_flux.copy(),
            iteration=-1,
            objective=float(best_loss),
            n_failed=int(forward_failures_at_best),
            kappa=best_kappa.copy(),
        )
        live_plot.save(fit_plot_path)
        print(f"Saved fit plot: {fit_plot_path}")
    else:
        fit_plot_path = os.path.join(
            request_runtime.output_dir, "phi_applied_vs_steady_observable_fit.png"
        )
        plt.figure(figsize=(7, 4))
        plt.plot(
            phi_applied_values,
            target_flux,
            marker="o",
            linewidth=2,
            label="target observable",
        )
        plt.plot(
            phi_applied_values,
            best_sim_flux,
            marker="s",
            label="best-fit simulated observable",
            linewidth=2,
        )
        plt.xlabel("applied voltage phi_applied")
        plt.ylabel(str(request_runtime.observable_label))
        plt.title(f"{request_runtime.observable_title} (adjoint-gradient curve fitting)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fit_plot_path, dpi=160)
        plt.close()
        print(f"Saved fit plot: {fit_plot_path}")

    live_gif_path: Optional[str] = None
    if request_runtime.live_plot_export_gif_path:
        live_gif_path = export_live_fit_gif(
            path=str(request_runtime.live_plot_export_gif_path),
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            history_rows=history_rows,
            point_rows=point_rows,
            seconds=float(request_runtime.live_plot_export_gif_seconds),
            n_frames=int(request_runtime.live_plot_export_gif_frames),
            dpi=int(request_runtime.live_plot_export_gif_dpi),
            y_label=str(request_runtime.observable_label),
            title=f"{str(request_runtime.observable_title)} progress",
        )
        if live_gif_path:
            print(f"Saved convergence GIF: {live_gif_path}")
        else:
            print("GIF export requested but could not be generated.")

    print(
        "[replay] run totals: "
        f"rebuilds={int(replay_stats.get('replay_rebuild_count', 0))} "
        f"diag_rebuilds={int(replay_stats.get('replay_diag_rebuild_count', 0))} "
        f"exception_rebuilds={int(replay_stats.get('replay_exception_rebuild_count', 0))}"
    )

    return RobinFluxCurveInferenceResult(
        best_kappa=best_kappa.copy(),
        best_loss=float(best_loss),
        phi_applied_values=phi_applied_values.copy(),
        target_flux=target_flux.copy(),
        best_simulated_flux=best_sim_flux.copy(),
        forward_failures_at_best=int(forward_failures_at_best),
        fit_csv_path=fit_csv_path,
        fit_plot_path=fit_plot_path,
        history_csv_path=history_csv_path,
        point_gradient_csv_path=point_csv_path,
        live_gif_path=live_gif_path,
        optimization_success=bool(getattr(opt_result, "success", False)),
        optimization_message=str(getattr(opt_result, "message", "")),
        replay_rebuild_count=int(replay_stats.get("replay_rebuild_count", 0)),
        replay_diag_rebuild_count=int(replay_stats.get("replay_diag_rebuild_count", 0)),
        replay_exception_rebuild_count=int(
            replay_stats.get("replay_exception_rebuild_count", 0)
        ),
    )
