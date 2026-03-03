"""k0 inference from peroxide current density curve -- EXTENDED range.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Extended range: 15 inference points spanning eta_hat in [-1, -46.5].
Uses observable_mode="peroxide_current".

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVk0_charged_peroxide_current_extended.py
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
    K_SCALE,
    I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_redimensionalized_results,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_k0_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

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
    initial_k0_guess = [0.003, 0.0001]

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

    output_dir = os.path.join("StudyResults", "bv_k0_peroxide_current_charged_extended")
    observable_scale = -I_SCALE

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp, steady=steady,
        true_k0=true_k0, initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "phi_applied_vs_peroxide_current_synthetic.csv"),
        output_dir=output_dir, regenerate_target=True,
        target_noise_percent=2.0, target_seed=20260226,
        observable_mode="peroxide_current",
        current_density_scale=observable_scale,
        observable_label="peroxide current density (mA/cm2)",
        observable_title="k0 inference from peroxide current (extended)",
        control_mode="k0",
        k0_lower=1e-8, k0_upper=100.0, log_space=True,
        max_eta_gap=3.0,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B", optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30, live_plot=False,
        forward_recovery=make_recovery_config(),
    )

    result = run_bv_k0_flux_curve_inference(request)

    best_k0 = np.asarray(result["best_k0"])
    true_k0_arr = np.asarray(true_k0)
    k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)

    print(f"\n=== k0 Inference from Peroxide Current (Extended Range) ===")
    print(f"True k0 (nondim):  {true_k0}")
    print(f"Best k0 (nondim):  {best_k0.tolist()}")
    print(f"Relative error:    {k0_err.tolist()}")
    print(f"Final objective:   {result['best_loss']:.12e}")


if __name__ == "__main__":
    main()
