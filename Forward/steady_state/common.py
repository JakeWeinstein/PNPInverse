"""Shared types, I/O helpers, noise utilities, and context managers for steady-state solves.

This module contains the data structures and utility functions used by both the
Robin and BV steady-state sub-modules.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import csv
import os
from typing import Any, Iterable, Sequence

# Keep Firedrake cache paths writable in sandboxed/restricted environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from Nondim.constants import FARADAY_CONSTANT

try:
    import firedrake.adjoint as adj
except Exception:
    adj = None


@dataclass
class SteadyStateConfig:
    """Settings for determining whether a run has reached steady state.

    Parameters
    ----------
    relative_tolerance:
        Max allowed relative change in per-species boundary flux between
        successive steps.
    absolute_tolerance:
        Absolute change floor used in the metric denominator and absolute check.
    consecutive_steps:
        Number of consecutive "steady" steps required before declaring
        convergence to steady state.
    max_steps:
        Hard cap on time steps for each solve.
    flux_observable:
        Which scalar observable to report:
        - ``"total_species"``: sum of species fluxes
        - ``"total_charge"``: Faradaic charge flux ``F * sum(z_i * J_i)``
        - ``"charge_proxy_no_f"``: signed charge proxy ``sum(z_i * J_i)``
          (Faraday scaling intentionally omitted; units are not physical current)
        - ``"species"``: one selected species flux
    species_index:
        Required when ``flux_observable="species"``.
    verbose:
        If True, prints periodic step diagnostics.
    print_every:
        Step print interval used when ``verbose=True``.
    """

    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 1e-8
    consecutive_steps: int = 5
    max_steps: int = 200
    flux_observable: str = "total_species"
    species_index: int | None = None
    verbose: bool = False
    print_every: int = 25


@dataclass
class SteadyStateResult:
    """Steady-state solve summary for one applied-voltage point."""

    phi_applied: float
    converged: bool
    steps_taken: int
    final_time: float
    species_flux: list[float]
    observed_flux: float
    final_relative_change: float | None
    final_absolute_change: float | None
    failure_reason: str = ""

    @property
    def phi0(self) -> float:
        """Backward-compatible alias for older code using ``result.phi0``."""
        return float(self.phi_applied)


def observed_flux_from_species_flux(
    species_flux: Sequence[float],
    *,
    z_vals: Sequence[float],
    flux_observable: str,
    species_index: int | None,
) -> float:
    """Map per-species flux vector to one scalar observable for fitting."""
    f = np.asarray(species_flux, dtype=float)
    mode = str(flux_observable).strip().lower()

    if mode == "total_species":
        return float(np.sum(f))

    if mode == "total_charge":
        z = np.asarray([float(v) for v in z_vals], dtype=float)
        if z.size != f.size:
            raise ValueError(f"z_vals size {z.size} does not match species flux size {f.size}.")
        return float(FARADAY_CONSTANT * np.dot(z, f))

    if mode == "charge_proxy_no_f":
        z = np.asarray([float(v) for v in z_vals], dtype=float)
        if z.size != f.size:
            raise ValueError(f"z_vals size {z.size} does not match species flux size {f.size}.")
        # Provisional observable for fitting only: signed charge-weighted flux
        # without Faraday scaling. Keeps magnitudes comparable to flux-space
        # studies while unit conventions are being finalized.
        return float(np.dot(z, f))

    if mode == "species":
        if species_index is None:
            raise ValueError("species_index must be set when flux_observable='species'.")
        idx = int(species_index)
        if idx < 0 or idx >= f.size:
            raise ValueError(f"species_index {idx} out of bounds for {f.size} species.")
        return float(f[idx])

    raise ValueError(
        f"Unknown flux_observable '{flux_observable}'. "
        "Use 'total_species', 'total_charge', 'charge_proxy_no_f', or 'species'."
    )


def add_percent_noise(
    values: Sequence[float],
    noise_percent: float,
    *,
    seed: int | None = None,
    mode: str = "rms",
) -> np.ndarray:
    """Add zero-mean Gaussian noise to *values*.

    Two noise modes are supported:

    ``mode="rms"`` (default, legacy)
        Global sigma: ``sigma = (noise_percent / 100) * RMS(finite values)``.
        Every finite entry receives noise drawn from the same Gaussian.
        This is the original behavior and all existing callers use it.

    ``mode="signal"``
        Per-point multiplicative sigma:
        ``sigma_i = (noise_percent / 100) * |values_i|`` for each point.
        A floor of 1e-12 is applied to sigma_i so that near-zero values
        still receive a tiny perturbation rather than exactly zero noise.
        Useful when the signal spans several orders of magnitude and a
        uniform percentage of the *local* signal is desired.

    In both modes NaN entries in *values* are preserved (noise is only
    added to finite entries).

    Parameters
    ----------
    values : Sequence[float]
        Input signal array.
    noise_percent : float
        Noise level as a percentage (e.g., 5.0 means 5 %).
    seed : int or None
        Random seed for reproducibility.
    mode : str
        ``"rms"`` for global-sigma noise (default) or ``"signal"`` for
        per-point multiplicative noise.

    Returns
    -------
    np.ndarray
        Noisy copy of *values*.
    """
    v = np.asarray(values, dtype=float)
    pct = float(noise_percent)
    if pct < 0:
        raise ValueError(f"noise_percent must be non-negative; got {pct}.")
    if mode not in ("rms", "signal"):
        raise ValueError(
            f"Unknown noise mode '{mode}'. Use 'rms' or 'signal'."
        )
    if pct == 0:
        return v.copy()
    rng = np.random.default_rng(seed)
    finite_mask = np.isfinite(v)
    v_finite = v[finite_mask]
    if v_finite.size == 0:
        return v.copy()

    out = v.copy()

    if mode == "rms":
        # Legacy global-sigma mode
        rms = float(np.sqrt(np.mean(v_finite * v_finite)))
        sigma = (pct / 100.0) * max(rms, 1e-12)
        # Draw noise for ALL positions (preserves RNG state reproducibility
        # relative to the original implementation when there are no NaN values)
        noise = rng.normal(0.0, sigma, size=v.shape)
        out[finite_mask] += noise[finite_mask]
    else:
        # Per-point multiplicative mode
        sigma_all = np.maximum((pct / 100.0) * np.abs(v), 1e-12)
        noise = rng.standard_normal(size=v.shape) * sigma_all
        out[finite_mask] += noise[finite_mask]

    return out


def write_phi_applied_flux_csv(
    csv_path: str,
    results: Sequence[SteadyStateResult],
    *,
    noisy_flux: Sequence[float] | None = None,
) -> None:
    """Write sweep results to CSV with both clean and optional noisy flux."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    noisy = None if noisy_flux is None else list(noisy_flux)

    header = [
        "phi_applied",
        "flux_clean",
        "flux_noisy",
        "converged",
        "steps_taken",
        "final_time",
        "final_relative_change",
        "final_absolute_change",
        "species_flux",
        "failure_reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, r in enumerate(results):
            writer.writerow(
                [
                    f"{float(r.phi_applied):.16g}",
                    f"{float(r.observed_flux):.16g}",
                    "" if noisy is None else f"{float(noisy[i]):.16g}",
                    int(bool(r.converged)),
                    int(r.steps_taken),
                    f"{float(r.final_time):.16g}",
                    ""
                    if r.final_relative_change is None
                    else f"{float(r.final_relative_change):.16g}",
                    ""
                    if r.final_absolute_change is None
                    else f"{float(r.final_absolute_change):.16g}",
                    ";".join(f"{float(v):.16g}" for v in r.species_flux),
                    str(r.failure_reason),
                ]
            )


def read_phi_applied_flux_csv(
    csv_path: str, *, flux_column: str = "flux_noisy"
) -> dict[str, np.ndarray]:
    """Read phi_applied-vs-flux CSV written by :func:`write_phi_applied_flux_csv`.

    Parameters
    ----------
    csv_path:
        Input CSV path.
    flux_column:
        Preferred flux column; if missing or empty, falls back to ``flux_clean``.
    """
    phi_applied_vals: list[float] = []
    flux_vals: list[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Backward compatibility: older files used "phi0".
            phi_key = "phi_applied" if "phi_applied" in row else "phi0"
            phi_applied_vals.append(float(row[phi_key]))
            selected = row.get(flux_column, "")
            if selected is None or str(selected).strip() == "" or str(selected).strip().lower() == "nan":
                selected = row["flux_clean"]
            flux_vals.append(float(selected))

    return {
        "phi_applied": np.asarray(phi_applied_vals, dtype=float),
        "flux": np.asarray(flux_vals, dtype=float),
    }


def results_to_flux_array(results: Sequence[SteadyStateResult]) -> np.ndarray:
    """Convert sweep results to numpy array of observed flux values."""
    return np.asarray([float(r.observed_flux) for r in results], dtype=float)


def all_results_converged(results: Iterable[SteadyStateResult]) -> bool:
    """Return True only when all sweep points reached steady-state criterion."""
    return all(bool(r.converged) for r in results)


@contextmanager
def _maybe_stop_annotating():
    """Disable pyadjoint annotation when available for pure forward studies."""
    if adj is None:
        yield
    else:
        with adj.stop_annotating():
            yield
