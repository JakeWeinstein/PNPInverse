"""k0 inference from peroxide current density curve.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Uses observable_mode="peroxide_current" to compute I_peroxide = -(R0 - R1) * i_scale,
which isolates the net peroxide production/consumption signal.

This should improve identifiability of k0_2 (H2O2 -> H2O reaction) compared
to total current density, since the peroxide observable directly shows the
difference between production (R0) and consumption (R1).

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVk0_charged_peroxide_current.py
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
    K0_HAT_R1, K0_HAT_R2, K_SCALE, I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
    print_redimensionalized_results,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_k0_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

print_params_summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eta_values = np.linspace(-1.0, -10.0, 10)

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    initial_k0_guess = [0.003, 0.0001]  # ~2x true values

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

    output_dir = os.path.join("StudyResults", "bv_k0_peroxide_current_charged")

    # Peroxide current observable: I_pxd = -(R0 - R1) * i_scale
    # Use negative i_scale because cathodic gives negative current
    observable_scale = -I_SCALE

    print(f"[params] Observable: peroxide_current = -(R0 - R1) * scale")
    print(f"[params] Scale = {observable_scale:.4f}")

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "phi_applied_vs_peroxide_current_synthetic.csv"),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        # KEY CHANGE: use peroxide_current observable
        observable_mode="peroxide_current",
        current_density_scale=observable_scale,
        observable_label="peroxide current density (mA/cm2)",
        observable_title="k0 inference from peroxide current (charged 4-sp)",
        control_mode="k0",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        live_plot_export_gif_path=os.path.join(output_dir, "convergence.gif"),
        forward_recovery=make_recovery_config(),
    )

    result = run_bv_k0_flux_curve_inference(request)

    best_k0 = np.asarray(result["best_k0"], dtype=float)
    true_k0_arr = np.asarray(true_k0, dtype=float)
    print_redimensionalized_results(best_k0, true_k0_arr)


if __name__ == "__main__":
    main()
