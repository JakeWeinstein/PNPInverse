"""Adjoint-gradient inference of BV exchange rate constants (k0) from I-V curve.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers [k0_1, k0_2] using adjoint-gradient + L-BFGS-B, same framework
as the neutral 2-species script but with the Poisson-coupled EDL.

Physical parameters from Mangan2025 (pH 4):
  - 4-species charged: O2 (D=1.9e-9), H2O2 (D=1.6e-9),
    H+ (D=9.311e-9, z=+1), ClO4- (D=1.792e-9, z=-1)
  - c_O2 = 0.5 mol/m3, c_H+ = c_ClO4- = 0.1 mol/m3, L_ref = 100 um
  - R1: O2 + 2H+ + 2e- -> H2O2   (reversible, k0=2.4e-8 m/s, alpha=0.627)
  - R2: H2O2 + 2H+ + 2e- -> H2O  (irreversible, k0=1e-9 m/s, alpha=0.5)
  - cathodic_conc_factors: (c_H+/c_ref_H+)^2 in both reactions

Mesh: make_graded_rectangle_mesh(Nx=8, Ny=300, beta=3.0)
  Markers: 3=bottom(electrode), 4=top(bulk), 1/2=left/right(zero-flux)

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVk0_charged_from_current_density_curve.py
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
    # Sweep: eta_hat from -1 to -10 (10 points)
    # Conservative range: the charged 4-species system is stiffer than neutral.
    # Current saturates by eta~=-8 V_T, so most kinetic info is in [-1, -10].
    eta_values = np.linspace(-1.0, -10.0, 10)

    # True k0 values (dimensionless) -- ground truth for synthetic data
    true_k0 = [K0_HAT_R1, K0_HAT_R2]

    # Initial guess -- ~2x true values (charged system less tolerant of large errors)
    initial_guess = [0.003, 0.0001]

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps

    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
    )

    steady = SteadyStateConfig(
        relative_tolerance=1e-4,
        absolute_tolerance=1e-8,
        consecutive_steps=4,
        max_steps=max_ss_steps,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    output_dir = os.path.join("StudyResults", "bv_k0_inference_charged")

    # Observable scale: cathodic (reduction) gives negative current.
    # I = -(sum R_j) * I_SCALE, so observable_scale = -I_SCALE
    observable_scale = -I_SCALE

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(
            output_dir,
            "phi_applied_vs_current_density_synthetic.csv",
        ),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        observable_reaction_index=None,
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Charged 4-species BV k0 inference from I-V curve",
        k0_lower=1e-8,
        k0_upper=100.0,
        mesh_Nx=8,
        mesh_Ny=200,
        mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 15,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "disp": True,
        },
        max_iters=15,
        gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        blob_initial_condition=False,
        live_plot=False,
        live_plot_pause_seconds=0.001,
        live_plot_eval_lines=False,
        live_plot_eval_line_alpha=0.30,
        live_plot_eval_max_lines=120,
        live_plot_export_gif_path=os.path.join(output_dir, "bv_k0_convergence.gif"),
        anisotropy_trigger_failed_points=4,
        anisotropy_trigger_failed_fraction=0.25,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result = run_bv_k0_flux_curve_inference(request)

    # Print redimensionalized results
    best_k0 = np.asarray(result["best_k0"], dtype=float)
    true_k0_arr = np.asarray(true_k0, dtype=float)
    print_redimensionalized_results(best_k0, true_k0_arr)


if __name__ == "__main__":
    main()
