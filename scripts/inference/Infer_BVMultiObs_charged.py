"""Multi-observable joint (k0, alpha) inference -- total + peroxide current.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Fits two observables simultaneously:
    Primary:   total current density  = sum_j R_j * i_scale
    Secondary: peroxide current       = -(R0 - R1) * i_scale

The combined objective sums both terms:
    J = J_primary + weight * J_secondary

This improves identifiability of k0_2 (H2O2 decomposition rate constant),
since the peroxide observable directly shows the difference between
production (R0) and consumption (R1) of H2O2.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVMultiObs_charged.py
"""

from __future__ import annotations

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, K_SCALE, I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_multi_observable_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

print_params_summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 15-point cathodic range (same as hybrid Phase B, good for k0+alpha)
    eta_values = np.array([
        -1.0, -2.0, -3.0,
        -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0,
        -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
    )

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    observable_scale = -I_SCALE
    output_dir = os.path.join("StudyResults", "bv_multi_obs_inference_charged")

    print(f"\n{'='*70}")
    print(f"  Multi-Observable Joint (k0, alpha) Inference")
    print(f"  Primary:   current_density (total)")
    print(f"  Secondary: peroxide_current (R0 - R1)")
    print(f"  Points: {len(eta_values)}, eta in [{eta_values.min():.1f}, {eta_values.max():.1f}]")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"{'='*70}\n")

    t_start = time.time()

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "target_primary.csv"),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        # Primary observable: total current density
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Multi-observable joint fit",
        # Secondary observable: peroxide current
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(output_dir, "target_peroxide.csv"),
        # Joint control mode
        control_mode="joint",
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result = run_bv_multi_observable_flux_curve_inference(request)

    elapsed = time.time() - t_start
    best_k0 = np.asarray(result["best_k0"])
    best_alpha = np.asarray(result["best_alpha"])
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)

    print(f"\n{'='*70}")
    print(f"  MULTI-OBSERVABLE INFERENCE RESULTS")
    print(f"{'='*70}")
    print(f"  k0_1 = {best_k0[0]:.6e}  (true {K0_HAT_R1:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"  k0_2 = {best_k0[1]:.6e}  (true {K0_HAT_R2:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"  alpha_1 = {best_alpha[0]:.6f}  (true {ALPHA_R1:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"  alpha_2 = {best_alpha[1]:.6f}  (true {ALPHA_R2:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"  Final objective: {result['best_loss']:.12e}")
    print(f"  Optimizer converged: {result['optimization_success']}")
    print(f"  Time: {elapsed:.1f}s")

    best_k0_phys = best_k0 * K_SCALE
    true_k0_phys = true_k0_arr * K_SCALE
    print(f"\n  Physical units (m/s):")
    print(f"    True k0:  [{true_k0_phys[0]:.4e}, {true_k0_phys[1]:.4e}]")
    print(f"    Best k0:  [{best_k0_phys[0]:.4e}, {best_k0_phys[1]:.4e}]")
    print(f"{'='*70}")
    print(f"\n  Output: {output_dir}/")


if __name__ == "__main__":
    main()
