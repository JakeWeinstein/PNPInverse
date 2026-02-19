"""Utilities for Robin-boundary flux experiments and flux-curve inversion.

This module is designed for the experimental workflow where:
1) the controlled input is applied voltage (``phi_applied``), and
2) the measured output is steady-state Robin boundary flux.

The helpers here provide:
- robust stepping to a flux-defined steady state
- voltage sweep generation (phi_applied vs steady-state flux)
- CSV serialization helpers for experimental/synthetic datasets

The code intentionally avoids firedrake-adjoint optimization machinery because
the objective is a multi-solve, steady-state curve mismatch rather than a
single terminal-state functional.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Keep Firedrake cache paths writable in sandboxed/restricted environments.
# Set these before importing firedrake/pyop2.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import firedrake as fd
import numpy as np

from UnifiedInverse.solver_interface import as_species_list, deep_copy_solver_params
from Utils.robin_forsolve import build_context, build_forms, set_initial_conditions

try:
    import firedrake.adjoint as adj
except Exception:
    adj = None


FARADAY_CONSTANT = 96485.3329


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
    species_index: Optional[int] = None
    verbose: bool = False
    print_every: int = 25


@dataclass
class SteadyStateResult:
    """Steady-state solve summary for one applied-voltage point."""

    phi_applied: float
    converged: bool
    steps_taken: int
    final_time: float
    species_flux: List[float]
    observed_flux: float
    final_relative_change: Optional[float]
    final_absolute_change: Optional[float]
    failure_reason: str = ""

    @property
    def phi0(self) -> float:
        """Backward-compatible alias for older code using ``result.phi0``."""
        return float(self.phi_applied)


def configure_robin_solver_params(
    base_solver_params: Sequence[Any],
    *,
    phi_applied: float,
    kappa_values: Optional[Sequence[float]] = None,
) -> List[Any]:
    """Return a solver-parameter copy with applied voltage and optional kappa.

    Notes
    -----
    In the Robin forward solver, boundary voltage is read from slot 7
    (``phi_applied``). We also mirror it into slot 9 (``phi0``) for interface
    consistency and easier inspection in logs.
    """
    params = deep_copy_solver_params(base_solver_params)
    n_species = int(params[0])

    params[7] = float(phi_applied)  # Applied boundary voltage used by robin_forsolve.
    params[9] = float(phi_applied)  # Kept in sync for human readability.

    if kappa_values is not None:
        p = params[10]
        if not isinstance(p, dict):
            raise ValueError("solver_params[10] must be a dict when setting robin kappa.")
        robin_cfg = p.setdefault("robin_bc", {})
        if not isinstance(robin_cfg, dict):
            raise ValueError("params['robin_bc'] must be a dict when setting robin kappa.")
        robin_cfg["kappa"] = as_species_list(kappa_values, n_species, "robin_kappa")
    return params


def compute_species_flux_on_robin_boundary(ctx: Dict[str, Any]) -> List[float]:
    """Assemble outward species fluxes across the configured Robin boundary.

    The Robin condition in this code is:
        ``J_i · n = kappa_i * (c_i - c_inf_i)``
    so each species flux is assembled directly from this boundary expression.
    """
    n = int(ctx["n_species"])
    robin = ctx["robin_settings"]
    electrode_marker = int(robin["electrode_marker"])
    c_inf_vals = [float(v) for v in robin["c_inf_vals"]]
    kappa_funcs = list(ctx["kappa_funcs"])
    ci = fd.split(ctx["U"])[:-1]
    ds = fd.Measure("ds", domain=ctx["mesh"])

    fluxes: List[float] = []
    for i in range(n):
        form = kappa_funcs[i] * (ci[i] - fd.Constant(c_inf_vals[i])) * ds(electrode_marker)
        fluxes.append(float(fd.assemble(form)))
    return fluxes


def observed_flux_from_species_flux(
    species_flux: Sequence[float],
    *,
    z_vals: Sequence[float],
    flux_observable: str,
    species_index: Optional[int],
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
        # without Faraday scaling. This keeps magnitudes comparable to
        # flux-space studies while unit conventions are being finalized.
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


def solve_to_steady_state_for_phi_applied(
    solver_params: Sequence[Any],
    *,
    steady: SteadyStateConfig,
    blob_initial_condition: bool = True,
) -> SteadyStateResult:
    """Run one Robin forward solve until flux-defined steady state is reached.

    Steady-state metric:
    - Per step, compute per-species fluxes on the Robin boundary.
    - Let ``delta = |J_n - J_{n-1}|`` (vector).
    - Relative metric is ``max(delta / max(|J_n|, |J_{n-1}|, abs_tol))``.
    - Absolute metric is ``max(delta)``.
    - A step is considered "steady" when either metric passes tolerance:
      ``rel_metric <= relative_tolerance`` OR
      ``abs_metric <= absolute_tolerance``.
    - Convergence requires ``consecutive_steps`` steady steps in a row.
    """
    params = deep_copy_solver_params(solver_params)
    dt = float(params[2])
    phi_applied = float(params[7])
    z_vals = as_species_list(params[4], int(params[0]), "z_vals")

    with _maybe_stop_annotating():
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

        prev_flux: Optional[np.ndarray] = None
        steady_count = 0
        rel_metric: Optional[float] = None
        abs_metric: Optional[float] = None
        species_flux = np.zeros(int(params[0]), dtype=float)

        for step in range(1, max(1, int(steady.max_steps)) + 1):
            try:
                solver.solve()
            except Exception as exc:
                return SteadyStateResult(
                    phi_applied=phi_applied,
                    converged=False,
                    steps_taken=step,
                    final_time=step * dt,
                    species_flux=species_flux.tolist(),
                    observed_flux=float("nan"),
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )

            U_prev.assign(U)

            species_flux = np.asarray(compute_species_flux_on_robin_boundary(ctx), dtype=float)

            if prev_flux is not None:
                delta = np.abs(species_flux - prev_flux)
                scale = np.maximum(
                    np.maximum(np.abs(species_flux), np.abs(prev_flux)),
                    float(max(steady.absolute_tolerance, 1e-16)),
                )
                rel_metric = float(np.max(delta / scale))
                abs_metric = float(np.max(delta))

                is_steady = (
                    rel_metric <= float(steady.relative_tolerance)
                    or abs_metric <= float(steady.absolute_tolerance)
                )
                steady_count = steady_count + 1 if is_steady else 0
            else:
                steady_count = 0

            if steady.verbose and int(steady.print_every) > 0 and step % int(steady.print_every) == 0:
                obs = observed_flux_from_species_flux(
                    species_flux,
                    z_vals=z_vals,
                    flux_observable=steady.flux_observable,
                    species_index=steady.species_index,
                )
                print(
                    f"[steady] phi_applied={phi_applied:>9.4f} step={step:>4d} "
                    f"rel_change={(rel_metric if rel_metric is not None else float('nan')):>10.3e} "
                    f"abs_change={(abs_metric if abs_metric is not None else float('nan')):>10.3e} "
                    f"flux={obs:>12.6f}"
                )

            prev_flux = species_flux.copy()

            if steady_count >= int(max(1, steady.consecutive_steps)):
                observed = observed_flux_from_species_flux(
                    species_flux,
                    z_vals=z_vals,
                    flux_observable=steady.flux_observable,
                    species_index=steady.species_index,
                )
                return SteadyStateResult(
                    phi_applied=phi_applied,
                    converged=True,
                    steps_taken=step,
                    final_time=step * dt,
                    species_flux=species_flux.tolist(),
                    observed_flux=float(observed),
                    final_relative_change=rel_metric,
                    final_absolute_change=abs_metric,
                    failure_reason="",
                )

        observed = observed_flux_from_species_flux(
            species_flux,
            z_vals=z_vals,
            flux_observable=steady.flux_observable,
            species_index=steady.species_index,
        )
        return SteadyStateResult(
            phi_applied=phi_applied,
            converged=False,
            steps_taken=int(max(1, steady.max_steps)),
            final_time=float(max(1, steady.max_steps)) * dt,
            species_flux=species_flux.tolist(),
            observed_flux=float(observed),
            final_relative_change=rel_metric,
            final_absolute_change=abs_metric,
            failure_reason="steady-state criterion not satisfied before max_steps",
        )


def sweep_phi_applied_steady_flux(
    base_solver_params: Sequence[Any],
    *,
    phi_applied_values: Sequence[float],
    steady: SteadyStateConfig,
    kappa_values: Optional[Sequence[float]] = None,
    blob_initial_condition: bool = True,
) -> List[SteadyStateResult]:
    """Generate a phi_applied-vs-steady-state-flux curve for one kappa setting."""
    out: List[SteadyStateResult] = []
    for phi_applied in phi_applied_values:
        params = configure_robin_solver_params(
            base_solver_params,
            phi_applied=float(phi_applied),
            kappa_values=kappa_values,
        )
        result = solve_to_steady_state_for_phi_applied(
            params,
            steady=steady,
            blob_initial_condition=blob_initial_condition,
        )
        out.append(result)
    return out


def add_percent_noise(values: Sequence[float], noise_percent: float, *, seed: Optional[int] = None) -> np.ndarray:
    """Add zero-mean Gaussian noise using sigma = pct * RMS(values)."""
    v = np.asarray(values, dtype=float)
    pct = float(noise_percent)
    if pct < 0:
        raise ValueError(f"noise_percent must be non-negative; got {pct}.")
    if pct == 0:
        return v.copy()
    rng = np.random.default_rng(seed)
    rms = float(np.sqrt(np.mean(v * v)))
    sigma = (pct / 100.0) * max(rms, 1e-12)
    return v + rng.normal(0.0, sigma, size=v.shape)


def write_phi_applied_flux_csv(
    csv_path: str,
    results: Sequence[SteadyStateResult],
    *,
    noisy_flux: Optional[Sequence[float]] = None,
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
) -> Dict[str, np.ndarray]:
    """Read phi_applied-vs-flux CSV written by :func:`write_phi_applied_flux_csv`.

    Parameters
    ----------
    csv_path:
        Input CSV path.
    flux_column:
        Preferred flux column; if missing or empty, falls back to ``flux_clean``.
    """
    phi_applied_vals: List[float] = []
    flux_vals: List[float] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Backward compatibility: older files used "phi0".
            phi_key = "phi_applied" if "phi_applied" in row else "phi0"
            phi_applied_vals.append(float(row[phi_key]))
            selected = row.get(flux_column, "")
            if selected is None or str(selected).strip() == "":
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


# Backward-compatible aliases for older script names.
def solve_to_steady_state_for_phi0(*args, **kwargs):
    return solve_to_steady_state_for_phi_applied(*args, **kwargs)


def sweep_phi0_steady_flux(*args, **kwargs):
    return sweep_phi_applied_steady_flux(*args, **kwargs)


def write_phi0_flux_csv(*args, **kwargs):
    return write_phi_applied_flux_csv(*args, **kwargs)


def read_phi0_flux_csv(*args, **kwargs):
    return read_phi_applied_flux_csv(*args, **kwargs)


@contextmanager
def _maybe_stop_annotating():
    """Disable pyadjoint annotation when available for pure forward studies."""
    if adj is None:
        yield
    else:
        with adj.stop_annotating():
            yield
