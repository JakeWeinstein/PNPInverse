"""Adjoint-gradient joint inference with Tikhonov regularization -- EXTENDED range.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers [k0_1, k0_2, alpha_1, alpha_2] with regularization penalty
across the extended voltage range eta_hat in [-1, -46.5].

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVJoint_charged_regularized_extended.py
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
    run_bv_joint_flux_curve_inference,
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
    """Build SolverParams for 4-species charged BV."""
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
        [0.01, 0.01, 0.01, 0.01],
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
        0.0, params,
    ])


def run_one_lambda(
    reg_lambda: float,
    *,
    output_subdir: str,
    eta_values: np.ndarray,
    true_k0: list,
    true_alpha: list,
    initial_k0_guess: list,
    initial_alpha_guess: list,
    k0_prior: list,
    alpha_prior: list,
) -> dict:
    """Run joint inference with a specific regularization lambda."""
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

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_subdir, "phi_applied_vs_current_density_synthetic.csv"),
        output_dir=output_subdir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title=f"Regularized joint inference (lambda={reg_lambda}, extended)",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        alpha_log_space=False,
        # Bridge points
        max_eta_gap=3.0,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30, gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        live_plot=False,
        live_plot_export_gif_path=os.path.join(output_subdir, "convergence.gif"),
        # Regularization
        regularization_lambda=reg_lambda,
        regularization_k0_prior=k0_prior,
        regularization_alpha_prior=alpha_prior,
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=6, max_it_only_attempts=2,
            anisotropy_only_attempts=1, tolerance_relax_attempts=2,
        ),
    )

    return run_bv_joint_flux_curve_inference(request)


def main() -> None:
    # Extended 15-point placement
    eta_values = np.array([
        -1.0, -2.0, -3.0,           # onset
        -4.0, -5.0, -6.5, -8.0,     # transition
        -10.0, -13.0,                # knee
        -17.0, -22.0, -28.0,        # plateau (sparse)
        -35.0, -41.0, -46.5,        # deep plateau
    ])

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    k0_prior = [K0_HAT * 1.5, K0_2_HAT * 2.0]
    alpha_prior = [0.5, 0.4]

    base_output = os.path.join("StudyResults", "bv_joint_regularized_charged_extended")

    lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0]

    all_results = []
    for lam in lambda_values:
        print(f"\n{'='*70}")
        print(f"  Regularization lambda = {lam}")
        print(f"{'='*70}")

        subdir = os.path.join(base_output, f"lambda_{lam:.4f}")
        try:
            result = run_one_lambda(
                lam,
                output_subdir=subdir,
                eta_values=eta_values,
                true_k0=true_k0,
                true_alpha=true_alpha,
                initial_k0_guess=initial_k0_guess,
                initial_alpha_guess=initial_alpha_guess,
                k0_prior=k0_prior,
                alpha_prior=alpha_prior,
            )
            all_results.append({"lambda": lam, "result": result, "error": None})
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"lambda": lam, "result": None, "error": str(e)})

    # Summary table
    print(f"\n{'='*90}")
    print(f"  Regularization Study Summary (Extended Range)")
    print(f"{'='*90}")
    print(f"{'lambda':>8} | {'k0_1 err':>10} {'k0_2 err':>10} {'a1 err':>10} {'a2 err':>10} | {'loss':>12}")
    print(f"{'-'*90}")

    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    for entry in all_results:
        lam = entry["lambda"]
        r = entry["result"]
        if r is None:
            print(f"{lam:>8.4f} | {'FAILED':>10} {'':>10} {'':>10} {'':>10} | {'N/A':>12}")
            continue
        best_k0 = np.asarray(r["best_k0"])
        best_alpha = np.asarray(r["best_alpha"])
        k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{lam:>8.4f} | {k0_err[0]:>10.4f} {k0_err[1]:>10.4f} "
              f"{alpha_err[0]:>10.4f} {alpha_err[1]:>10.4f} | {r['best_loss']:>12.6e}")

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
