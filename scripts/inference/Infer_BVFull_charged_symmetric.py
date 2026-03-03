"""Full (k0 + alpha + steric) inference -- SYMMETRIC voltage range.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers [k0_1, k0_2, alpha_1, alpha_2, a_1, a_2, a_3, a_4] simultaneously
using symmetric voltage placement (anodic + cathodic).

This is the most challenging inference mode: 8 parameters from 20 I-V points.
The symmetric voltage range should improve k0/alpha recovery by providing
independent Tafel slope constraints from both branches. Better k0/alpha pinning
should reduce cross-talk with steric parameters.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVFull_charged_symmetric.py
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1,
    K0_HAT_R2,
    I_SCALE,
    ALPHA_R1,
    ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_full_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

TRUE_STERIC_A = [0.05, 0.05, 0.05, 0.05]

def main() -> None:
    # Symmetric 20-point placement
    eta_values = np.array([
        +5.0, +3.0, +2.0, +1.0, +0.5,
        -0.25, -0.5,
        -1.0, -1.5, -2.0, -3.0,
        -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0,
        -17.0, -22.0, -28.0,
    ])

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]
    initial_steric_a_guess = [0.1, 0.1, 0.1, 0.1]

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
    output_dir = os.path.join("StudyResults", "bv_full_charged_symmetric")

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "target.csv"),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260228,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Full inference (k0+alpha+steric, symmetric)",
        control_mode="full",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        alpha_log_space=False,
        true_steric_a=TRUE_STERIC_A,
        initial_steric_a_guess=initial_steric_a_guess,
        steric_a_lower=0.001,
        steric_a_upper=0.15,  # reduced from 0.5
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 60, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=60, gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        live_plot=False,
        forward_recovery=make_recovery_config(),
    )

    result = run_bv_full_flux_curve_inference(request)

    # Final summary
    print("\n" + "=" * 80)
    print("  FULL (k0 + alpha + steric) INFERENCE SUMMARY (SYMMETRIC)")
    print("=" * 80)
    print(f"  Voltage range: eta_hat in [{eta_values.min():+.1f}, {eta_values.max():+.1f}]")
    print(f"  {len(eta_values)} points: {sum(eta_values > 0)} anodic, "
          f"{sum(eta_values < -0.5)} cathodic")

    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)
    true_steric_arr = np.asarray(TRUE_STERIC_A)

    best_k0 = result["best_k0"]
    best_alpha = result["best_alpha"]
    best_steric = result["best_steric_a"]

    k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    steric_err = np.abs(best_steric - true_steric_arr) / np.maximum(np.abs(true_steric_arr), 1e-16)

    print(f"\n  True k0:       {true_k0}")
    print(f"  Best k0:       {best_k0.tolist()}")
    print(f"  k0 rel err:    [{k0_err[0]:.4f}, {k0_err[1]:.4f}]")
    print()
    print(f"  True alpha:    {true_alpha}")
    print(f"  Best alpha:    {best_alpha.tolist()}")
    print(f"  alpha rel err: [{alpha_err[0]:.4f}, {alpha_err[1]:.4f}]")
    print()
    print(f"  True steric a: {TRUE_STERIC_A}")
    print(f"  Best steric a: {best_steric.tolist()}")
    print(f"  steric rel err:{steric_err.tolist()}")
    print()
    print(f"  Final loss:       {result['best_loss']:.12e}")
    print(f"  Optimizer success: {result['optimization_success']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
