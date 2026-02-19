"""Helper utilities for Robin kappa inference from a phi_applied-flux curve.

This module contains the full adjoint-gradient machinery used by
``InferenceScripts/Infer_RobinKappa_from_flux_curve.py`` so that the entry
script can stay short and user-config focused.
"""

from __future__ import annotations

import csv
import copy
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
    blob_initial_condition: bool = True
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
        blob_initial_condition=True,
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


def _build_species_boundary_flux_forms(ctx: Dict[str, object]) -> List[object]:
    """Build integrated Robin-boundary flux forms (one scalar form per species)."""
    import firedrake as fd

    n_species = int(ctx["n_species"])
    robin = ctx["robin_settings"]
    electrode_marker = int(robin["electrode_marker"])
    c_inf_vals = [float(v) for v in robin["c_inf_vals"]]
    kappa_funcs = list(ctx["kappa_funcs"])
    ci = fd.split(ctx["U"])[:-1]
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
    forms = _build_species_boundary_flux_forms(ctx)
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
    )


def evaluate_curve_objective_and_gradient(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
) -> CurveAdjointResult:
    """Evaluate curve objective + gradient, with anisotropy recovery fallback."""
    n_species = int(request.base_solver_params[0])

    def _evaluate_once(kappa_eval: np.ndarray) -> CurveAdjointResult:
        points: List[PointAdjointResult] = []
        simulated_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)
        total_objective = 0.0
        total_gradient = np.zeros(n_species, dtype=float)
        n_failed = 0

        for i, (phi_i, target_i) in enumerate(
            zip(phi_applied_values.tolist(), target_flux.tolist())
        ):
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
                    "target_flux": point.target_flux,
                    "simulated_flux": point.simulated_flux,
                    "point_objective": point.objective,
                    "dJ_dkappa0": float(point.gradient[0]) if n_species >= 1 else float("nan"),
                    "dJ_dkappa1": float(point.gradient[1]) if n_species >= 2 else float("nan"),
                    "converged": int(bool(point.converged)),
                    "steps_taken": int(point.steps_taken),
                    "reason": point.reason,
                }
            )
            if bool(request.print_point_gradients):
                print(
                    f"  phi={point.phi_applied:8.5f} "
                    f"target={point.target_flux:12.6f} "
                    f"sim={point.simulated_flux:12.6f} "
                    f"dJ/dk=[{point.gradient[0]:11.6f}, {point.gradient[1]:11.6f}] "
                    f"conv={int(point.converged)}"
                )

    def _evaluate(x: np.ndarray) -> Dict[str, object]:
        nonlocal best_kappa, best_loss, best_flux

        x_clip = clip_kappa(np.asarray(x, dtype=float), lower_bounds, upper_bounds)
        key = _key_from_x(x_clip)
        if key in cache:
            return cache[key]

        curve = evaluate_curve_objective_and_gradient(
            request=request,
            phi_applied_values=phi_applied_values,
            target_flux=target_flux,
            kappa_values=x_clip,
        )

        eval_counter["n"] += 1
        eval_id = int(eval_counter["n"])
        iter_id = int(iteration_counter["n"])
        objective = float(curve.objective)
        gradient = np.asarray(curve.gradient, dtype=float)
        grad_norm = float(np.linalg.norm(gradient))
        n_failed = int(curve.n_failed)
        simulated_flux = np.asarray(curve.simulated_flux, dtype=float)

        recovery_tag = " aniso_recovery=1" if curve.used_anisotropy_recovery else ""
        print(
            f"[eval={eval_id:03d}] "
            f"kappa=[{x_clip[0]:10.6f}, {x_clip[1]:10.6f}] "
            f"loss={objective:14.6e} "
            f"grad=[{gradient[0]:12.6f}, {gradient[1]:12.6f}] "
            f"|grad|={grad_norm:12.6f} "
            f"fails={n_failed:02d}"
            f"{recovery_tag}"
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
        print(
            f"[iter={iteration_counter['n']:02d}] "
            f"kappa=[{payload['x'][0]:10.6f}, {payload['x'][1]:10.6f}] "
            f"loss={float(payload['objective']):14.6e} "
            f"|grad|={float(np.linalg.norm(grad)):12.6f} "
            f"fails={int(payload['n_failed']):02d}"
            f"{recovery_tag}"
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

    return best_kappa, float(best_loss), best_flux, history_rows, point_rows, result, live_plot


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
        "target_flux",
        "simulated_flux",
        "point_objective",
        "dJ_dkappa0",
        "dJ_dkappa1",
        "converged",
        "steps_taken",
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
        curves_by_eval[eval_id][point_idx] = _as_float(row.get("simulated_flux"))

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

    best_kappa, best_loss, best_sim_flux, history_rows, point_rows, opt_result, live_plot = run_scipy_adjoint_optimization(
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
    fit_csv_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_steady_flux_fit.csv")
    with open(fit_csv_path, "w", encoding="utf-8") as f:
        f.write("phi_applied,target_flux,simulated_flux\n")
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

    fit_plot_path: Optional[str] = None
    if plt is None:
        print("matplotlib not available; skipping fit plot generation.")
    elif bool(request_runtime.live_plot) and getattr(live_plot, "enabled", False):
        fit_plot_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_steady_flux_fit.png")
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
        fit_plot_path = os.path.join(request_runtime.output_dir, "phi_applied_vs_steady_flux_fit.png")
        plt.figure(figsize=(7, 4))
        plt.plot(
            phi_applied_values,
            target_flux,
            marker="o",
            linewidth=2,
            label="target flux",
        )
        plt.plot(
            phi_applied_values,
            best_sim_flux,
            marker="s",
            label="best-fit simulated flux",
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
    )
