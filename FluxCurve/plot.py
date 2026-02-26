"""Live interactive fit plot and GIF export for flux-curve optimization."""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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
