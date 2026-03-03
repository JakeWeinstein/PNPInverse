"""Multi-pH joint (k0, alpha) inference v2 -- Phase 1 solver robustness fixes.

Changes from v1:
  Fix 1: Denser eta sweep (19 points vs 12) with smaller gaps for elevated H+
  Fix 2: SNES backtracking linesearch (bt) + reduced maxlambda (0.3)
  Fix 3: Smaller dt (0.1) with more steps (500) for better convergence
  Fix 4: Conditions solved in order of increasing difficulty (baseline first)
  Fix 5: Tighter max_eta_gap (1.5) for more bridge points
  Fix 6: Softplus concentration regularization enabled

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVMultiPH_charged_v2.py
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
    C_HP_HAT,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_STRICT,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_multi_ph_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

print_params_summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---------------------------------------------------------------------------
    # Fix 1: Denser eta sweep -- 19 points with smaller gaps for elevated H+
    # Original: 12 points with gaps of 2-11.5 between consecutive eta values
    # Now: 19 points with max gap of 4 (and bridge points fill remaining gaps)
    # ---------------------------------------------------------------------------
    eta_values = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0,
        -6.0, -7.0, -8.0, -9.0, -10.0,
        -12.0, -14.0, -17.0, -20.0,
        -24.0, -28.0, -32.0, -38.0, -46.5,
    ])

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    # ---------------------------------------------------------------------------
    # Fix 3: Smaller dt + more steps for convergence of stiff elevated-H+ cases
    # Original: dt=0.5, max_ss_steps=100, t_end=50
    # Now:      dt=0.1, max_ss_steps=500, t_end=50 (same t_end but finer steps)
    # ---------------------------------------------------------------------------
    dt = 0.1           # Fix 3: was 0.5
    max_ss_steps = 500  # Fix 3: was 100
    t_end = dt * max_ss_steps

    # Uses SNES_OPTS_STRICT (maxlambda=0.3) + softplus=True for robustness
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_STRICT,
        softplus=True,  # Fix 6: enable softplus concentration regularization
        c_hp_hat=C_HP_HAT,
    )

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    observable_scale = -I_SCALE
    output_dir = os.path.join("StudyResults", "multi_ph_inference_v2")

    # ---------------------------------------------------------------------------
    # Fix 4: Solve conditions in order of increasing difficulty
    # Original order: 0.4 (hardest), 0.2 (baseline), 0.1 (easiest)
    # New order: 0.1 (easiest), 0.2 (baseline), 0.4 (hardest)
    # This helps because the multi-pH evaluator processes conditions sequentially
    # and the easier conditions warm the Firedrake kernel cache.
    # ---------------------------------------------------------------------------
    ph_conditions = [
        {
            "c_hp_hat": 0.1,    # c_H+ = 0.05 mol/m^3 (0.5x baseline) -- easiest
            "weight": 1.0,
            "c_hp_species_index": 2,
            "counterion_species_index": 3,
            "label": "c_H+=0.05 mol/m3 (easy)",
        },
        {
            "c_hp_hat": C_HP_HAT,    # c_H+ = 0.1 mol/m^3 -> c_hp_hat = 0.2 (baseline)
            "weight": 1.0,
            "c_hp_species_index": 2,
            "counterion_species_index": 3,
            "label": "c_H+=0.1 mol/m3 (baseline)",
        },
        {
            "c_hp_hat": 0.3,    # c_H+ = 0.15 mol/m^3 (1.5x baseline) -- hard
            "weight": 1.0,
            "c_hp_species_index": 2,
            "counterion_species_index": 3,
            "label": "c_H+=0.15 mol/m3 (hard)",
        },
    ]

    print(f"\n{'='*70}")
    print(f"  Multi-pH Joint (k0, alpha) Inference -- v2 (Phase 1 fixes)")
    print(f"  Conditions: {len(ph_conditions)} pH values")
    for ci, cond in enumerate(ph_conditions):
        print(f"    Condition {ci}: {cond.get('label', '')} "
              f"(c_hp_hat={cond['c_hp_hat']:.4f}, weight={cond['weight']})")
    print(f"  Points per condition: {len(eta_values)}")
    print(f"  eta in [{eta_values.min():.1f}, {eta_values.max():.1f}]")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Phase 1 fixes:")
    print(f"    - Denser eta sweep: {len(eta_values)} points (was 12)")
    print(f"    - SNES linesearch: l2, maxlambda=0.3 (was 0.5)")
    print(f"    - dt={dt} (was 0.5), max_steps={max_ss_steps} (was 100)")
    print(f"    - Conditions ordered easy->hard")
    print(f"    - max_eta_gap=1.5 (was 3.0)")
    print(f"    - Softplus concentration regularization enabled")
    print(f"{'='*70}\n")

    t_start = time.time()

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "target_base.csv"),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Multi-pH joint fit v2",
        # Joint control mode
        control_mode="joint",
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=300, mesh_beta=4.0,  # Phase 2: finer mesh for hard condition
        max_eta_gap=1.5,    # Fix 5: tighter gap (was 3.0)
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 15, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=15,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
        # Multi-pH conditions
        multi_ph_conditions=ph_conditions,
    )

    result = run_bv_multi_ph_flux_curve_inference(request)

    elapsed = time.time() - t_start
    best_k0 = np.asarray(result["best_k0"])
    best_alpha = np.asarray(result["best_alpha"])
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)

    print(f"\n{'='*70}")
    print(f"  MULTI-PH INFERENCE v2 RESULTS")
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
