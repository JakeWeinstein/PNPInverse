"""I/O helpers and target generation for BV flux-curve inference."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from Forward.steady_state import (
    SteadyStateConfig,
    read_phi_applied_flux_csv,
    sweep_phi_applied_steady_bv_flux,
    all_results_converged,
    results_to_flux_array,
    add_percent_noise,
    write_phi_applied_flux_csv,
)


def _normalize_k0(value: Optional[Sequence[float]], *, name: str) -> Optional[List[float]]:
    """Validate k0-like input into a positive float list."""
    if value is None:
        return None
    vals = [float(v) for v in list(value)]
    if any(v <= 0.0 for v in vals):
        raise ValueError(f"{name} must be strictly positive.")
    return vals


def ensure_bv_target_curve(
    *,
    target_csv_path: str,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    true_k0: Optional[Sequence[float]],
    current_density_scale: float,
    noise_percent: float,
    seed: int,
    force_regenerate: bool,
    blob_initial_condition: bool,
    mesh: Any = None,
) -> Dict[str, np.ndarray]:
    """Load target BV data from CSV or generate synthetic target if missing."""
    if os.path.exists(target_csv_path) and not force_regenerate:
        print(f"Loading BV target curve from: {target_csv_path}")
        return read_phi_applied_flux_csv(target_csv_path, flux_column="flux_noisy")

    k0_true = _normalize_k0(true_k0, name="true_k0")
    if k0_true is None:
        raise ValueError("true_k0 must be set for synthetic target generation.")

    if os.path.exists(target_csv_path):
        print(
            "Regenerating BV target curve with true k0; "
            f"overwriting existing CSV: {target_csv_path}"
        )
    else:
        print("BV target CSV not found; generating synthetic target data first.")
    print(f"Synthetic target settings: true_k0={k0_true}, noise_percent={noise_percent}")

    # compute_bv_current_density computes  I = -(sum R_j) * i_scale
    # The observable form computes           obs = scale * sum(R_j)
    # For consistency: i_scale = -current_density_scale
    # (current_density_scale already includes the sign, e.g. -I_SCALE)
    target_results = sweep_phi_applied_steady_bv_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        k0_values=k0_true,
        i_scale=-current_density_scale,
        mesh=mesh,
        blob_initial_condition=bool(blob_initial_condition),
    )
    if not all_results_converged(target_results):
        failed = [f"{r.phi_applied:.6f}" for r in target_results if not r.converged]
        print(f"WARNING: BV target generation failed for {len(failed)} points: {failed}")
        # Continue with partial data rather than raising

    target_flux_clean = results_to_flux_array(target_results)
    target_flux_noisy = add_percent_noise(target_flux_clean, noise_percent, seed=seed)

    os.makedirs(os.path.dirname(target_csv_path) or ".", exist_ok=True)
    write_phi_applied_flux_csv(target_csv_path, target_results, noisy_flux=target_flux_noisy)
    print(f"BV synthetic target saved to: {target_csv_path}")

    return {"phi_applied": phi_applied_values.copy(), "flux": target_flux_noisy}


def write_bv_history_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write optimizer-iteration history to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_bv_point_gradient_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    """Write per-point adjoint gradient diagnostics to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _generate_observable_target(
    *,
    base_solver_params: Sequence[object],
    steady: Any,
    phi_applied_values: np.ndarray,
    true_k0: Sequence[float],
    observable_mode: str,
    observable_scale: float,
    noise_percent: float,
    seed: int,
    mesh: Any,
    target_csv_path: str,
    force_regenerate: bool,
) -> np.ndarray:
    """Generate synthetic target data for a specific observable mode.

    Runs the BV point solver at the true k0 values with the specified
    observable mode, then adds noise. Returns the noisy target flux array.
    """
    from Forward.steady_state import add_percent_noise
    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )
    from FluxCurve.config import ForwardRecoveryConfig as _FRC

    if os.path.exists(target_csv_path) and not force_regenerate:
        data = read_phi_applied_flux_csv(target_csv_path, flux_column="flux_noisy")
        return np.asarray(data["flux"], dtype=float)

    _clear_caches()

    # Use a dummy target (zeros) -- we only need the simulated flux at true params
    dummy_target = np.zeros_like(phi_applied_values, dtype=float)
    k0_list = [float(v) for v in true_k0]

    points = solve_bv_curve_points_with_warmstart(
        base_solver_params=base_solver_params,
        steady=steady,
        phi_applied_values=phi_applied_values,
        target_flux=dummy_target,
        k0_values=k0_list,
        blob_initial_condition=False,
        fail_penalty=1e9,
        forward_recovery=_FRC(),
        observable_mode=observable_mode,
        observable_reaction_index=None,
        observable_scale=observable_scale,
        mesh=mesh,
        alpha_values=None,
        control_mode="k0",
        max_eta_gap=0.0,
    )

    _clear_caches()

    clean_flux = np.array([float(p.simulated_flux) for p in points], dtype=float)
    noisy_flux = add_percent_noise(clean_flux, noise_percent, seed=seed)

    # Save to CSV
    os.makedirs(os.path.dirname(target_csv_path) or ".", exist_ok=True)
    with open(target_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phi_applied", "flux_clean", "flux_noisy"])
        for phi, fc, fn in zip(phi_applied_values, clean_flux, noisy_flux):
            writer.writerow([f"{phi:.16g}", f"{fc:.16g}", f"{fn:.16g}"])
    print(f"Saved {observable_mode} target CSV: {target_csv_path}")

    return noisy_flux
