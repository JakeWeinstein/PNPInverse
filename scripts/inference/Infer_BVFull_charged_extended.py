"""Full (k0, alpha, steric_a) inference -- EXTENDED range.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers [k0_1, k0_2, alpha_1, alpha_2, a_1, a_2, a_3, a_4] simultaneously
via control_mode="full".

Extended range: 15 inference points spanning eta_hat in [-1, -46.5],
with bridge points auto-inserted for gaps > 3.0.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVFull_charged_extended.py
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

# Steric true values
TRUE_STERIC_A = [0.05, 0.05, 0.05, 0.05]

def main() -> None:
    # Extended 15-point placement
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
    output_dir = os.path.join("StudyResults", "bv_full_charged_extended")

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp, steady=steady,
        true_k0=true_k0, initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "phi_applied_vs_current_density_synthetic.csv"),
        output_dir=output_dir, regenerate_target=True,
        target_noise_percent=2.0, target_seed=20260228,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Full (k0+alpha+steric) inference (extended)",
        control_mode="full",
        k0_lower=1e-8, k0_upper=100.0, log_space=True,
        true_alpha=true_alpha, initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95, alpha_log_space=False,
        # Steric settings -- reduced upper bound
        true_steric_a=TRUE_STERIC_A,
        initial_steric_a_guess=initial_steric_a_guess,
        steric_a_lower=0.001,
        steric_a_upper=0.15,
        # Bridge points
        max_eta_gap=3.0,
        # Mesh
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        # Optimizer
        optimizer_method="L-BFGS-B", optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40, gtol=1e-6,
        fail_penalty=1e9, print_point_gradients=True,
        live_plot=False,
        forward_recovery=make_recovery_config(),
    )

    result = run_bv_full_flux_curve_inference(request)

    # Final summary
    print("\n" + "=" * 80)
    print("  FULL (k0 + alpha + steric) INFERENCE SUMMARY (EXTENDED)")
    print("=" * 80)

    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)
    true_steric_arr = np.asarray(TRUE_STERIC_A)

    best_k0 = result["best_k0"]
    best_alpha = result["best_alpha"]
    best_steric = result["best_steric_a"]

    k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    steric_err = np.abs(best_steric - true_steric_arr) / np.maximum(np.abs(true_steric_arr), 1e-16)

    print(f"  True k0:       {true_k0}")
    print(f"  Best k0:       {best_k0.tolist()}")
    print(f"  k0 rel err:    {k0_err.tolist()}")
    print()
    print(f"  True alpha:    {true_alpha}")
    print(f"  Best alpha:    {best_alpha.tolist()}")
    print(f"  alpha rel err: {alpha_err.tolist()}")
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
