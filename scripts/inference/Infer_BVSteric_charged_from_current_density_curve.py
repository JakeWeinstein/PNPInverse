"""Adjoint-gradient steric 'a' inference from I-V curve data.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers Bikerman steric exclusion parameters [a_1, a_2, a_3, a_4]
with fixed k0 and alpha at true values.

The Bikerman model adds mu_steric = ln(1 - sum_j a_j c_j) to species fluxes.
Steric a values are O(0.01-0.1) and optimized in linear space (no log transform).

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVSteric_charged_from_current_density_curve.py
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
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, I_SCALE,
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
    run_bv_steric_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

print_params_summary()

# Steric true values: uniform steric exclusion across all species
TRUE_STERIC_A = [0.05, 0.05, 0.05, 0.05]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eta_values = np.linspace(-1.0, -10.0, 10)

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]

    # Initial guess for steric a: 2x the true value
    initial_steric_a_guess = [0.1, 0.1, 0.1, 0.1]

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps

    # NOTE: The steric script uses TRUE_STERIC_A as a_vals in the solver params
    # for target generation. We build a custom species config that overrides
    # the default a_vals. However, make_bv_solver_params uses species.a_vals_hat
    # which defaults to 0.01. For steric inference, the FluxCurve pipeline
    # handles steric a_vals internally, so we use the standard base_sp.
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

    output_dir = os.path.join("StudyResults", "bv_steric_charged")

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=true_k0,  # k0 fixed at true (not optimized)
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "phi_applied_vs_current_density_synthetic.csv"),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260228,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Steric 'a' inference (charged 4-species)",
        control_mode="steric",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        # Fixed k0 and alpha (at true values)
        fixed_k0=true_k0,
        fixed_alpha=true_alpha,
        true_alpha=true_alpha,
        # Steric inference settings
        true_steric_a=TRUE_STERIC_A,
        initial_steric_a_guess=initial_steric_a_guess,
        steric_a_lower=0.001,
        steric_a_upper=0.5,
        # Mesh
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        # Optimizer
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40, gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        live_plot=False,
        live_plot_export_gif_path=os.path.join(output_dir, "convergence.gif"),
        forward_recovery=make_recovery_config(),
    )

    result = run_bv_steric_flux_curve_inference(request)

    # Final summary
    print("\n" + "=" * 70)
    print("  STERIC INFERENCE SUMMARY")
    print("=" * 70)
    print(f"  True steric a:    {TRUE_STERIC_A}")
    print(f"  Initial guess:    {initial_steric_a_guess}")
    print(f"  Best steric a:    {result['best_steric_a'].tolist()}")
    true_arr = np.asarray(TRUE_STERIC_A)
    best_arr = result["best_steric_a"]
    rel_err = np.abs(best_arr - true_arr) / np.maximum(np.abs(true_arr), 1e-16)
    print(f"  Per-species error: {rel_err.tolist()}")
    print(f"  Max relative error: {np.max(rel_err):.6f}")
    print(f"  Final loss:       {result['best_loss']:.12e}")
    print(f"  Optimizer success: {result['optimization_success']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
