"""Steric 'a' inference -- SYMMETRIC voltage range (anodic + cathodic).

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers Bikerman steric exclusion parameters [a_1, a_2, a_3, a_4]
with fixed k0 and alpha at true values.

Uses symmetric voltage placement including anodic overpotentials.
Primary steric information comes from the cathodic transition region
(eta -3 to -8) where intermediate concentrations produce the largest
steric gradient signal. Anodic data helps indirectly by constraining
the k0/alpha dependencies in a full inference context.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVSteric_charged_symmetric.py
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    ForwardRecoveryConfig,
    run_bv_steric_flux_curve_inference,
)
from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig


# ---------------------------------------------------------------------------
# Physical constants and scales
# ---------------------------------------------------------------------------

F_CONST = 96485.3329
R_GAS = 8.31446
T_REF = 298.15
V_T = R_GAS * T_REF / F_CONST
N_ELECTRONS = 2

D_O2 = 1.9e-9;  C_O2 = 0.5
D_H2O2 = 1.6e-9; C_H2O2 = 0.0
D_HP = 9.311e-9; C_HP = 0.1
D_CLO4 = 1.792e-9; C_CLO4 = 0.1

K0_PHYS = 2.4e-8; ALPHA_1 = 0.627
K0_2_PHYS = 1e-9; ALPHA_2 = 0.5

L_REF = 1.0e-4; D_REF = D_O2; C_SCALE = C_O2; K_SCALE = D_REF / L_REF

D_O2_HAT = D_O2 / D_REF; D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT = D_HP / D_REF; D_CLO4_HAT = D_CLO4 / D_REF
C_O2_HAT = C_O2 / C_SCALE; C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT = C_HP / C_SCALE; C_CLO4_HAT = C_CLO4 / C_SCALE

K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_SCALE / L_REF * 0.1

TRUE_STERIC_A = [0.05, 0.05, 0.05, 0.05]

SNES_OPTS = {
    "snes_type":                 "newtonls",
    "snes_max_it":               300,
    "snes_atol":                 1e-7,
    "snes_rtol":                 1e-10,
    "snes_stol":                 1e-12,
    "snes_linesearch_type":      "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type":                  "preonly",
    "pc_type":                   "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8":         77,
    "mat_mumps_icntl_14":        80,
}


def _make_bv_solver_params(eta_hat: float, dt: float, t_end: float) -> SolverParams:
    """Build SolverParams for 4-species charged BV with steric exclusion."""
    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent": True, "exponent_clip": 50.0,
        "regularize_concentration": True, "conc_floor": 1e-12,
        "use_eta_in_bv": True,
    }
    params["nondim"] = {
        "enabled": True,
        "diffusivity_scale_m2_s": D_REF,
        "concentration_scale_mol_m3": C_SCALE,
        "length_scale_m": L_REF,
        "potential_scale_v": V_T,
        "kappa_inputs_are_dimensionless": True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless": True,
        "time_inputs_are_dimensionless": True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_HAT, "alpha": ALPHA_1,
                "cathodic_species": 0, "anodic_species": 1,
                "c_ref": 1.0, "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2, "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
            {
                "k0": K0_2_HAT, "alpha": ALPHA_2,
                "cathodic_species": 1, "anodic_species": None,
                "c_ref": 0.0, "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2, "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
        ],
        "k0": [K0_HAT] * 4, "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0], "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    return SolverParams.from_list([
        4, 1, dt, t_end, [0, 0, 1, -1],
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
        TRUE_STERIC_A,
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
        0.0, params,
    ])


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

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    initial_steric_a_guess = [0.1, 0.1, 0.1, 0.1]

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    observable_scale = -I_SCALE
    output_dir = os.path.join("StudyResults", "bv_steric_charged_symmetric")

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=true_k0,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "target.csv"),
        output_dir=output_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260228,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Steric 'a' inference (symmetric voltage range)",
        control_mode="steric",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        fixed_k0=true_k0,
        fixed_alpha=true_alpha,
        true_alpha=true_alpha,
        true_steric_a=TRUE_STERIC_A,
        initial_steric_a_guess=initial_steric_a_guess,
        steric_a_lower=0.001,
        steric_a_upper=0.15,  # reduced from 0.5 per convergence analysis
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,  # bridge points for cathodic plateau gaps
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 50, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=50, gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        live_plot=False,
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=6, max_it_only_attempts=2,
            anisotropy_only_attempts=1, tolerance_relax_attempts=2,
        ),
    )

    result = run_bv_steric_flux_curve_inference(request)

    # Final summary
    print("\n" + "=" * 70)
    print("  STERIC INFERENCE SUMMARY (SYMMETRIC VOLTAGE RANGE)")
    print("=" * 70)
    print(f"  Voltage range: eta_hat in [{eta_values.min():+.1f}, {eta_values.max():+.1f}]")
    print(f"  {len(eta_values)} points: {sum(eta_values > 0)} anodic, "
          f"{sum(eta_values < -0.5)} cathodic")
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
