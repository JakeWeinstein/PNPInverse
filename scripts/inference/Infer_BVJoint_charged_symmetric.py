"""Joint k0+alpha inference -- SYMMETRIC voltage range (anodic + cathodic).

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Symmetric range: 20 inference points spanning eta_hat in [-28, +5].

This script tests whether including anodic overpotentials improves
joint (k0, alpha) recovery by breaking the k0-alpha correlation.

At cathodic eta: I ~ k0 * exp(-alpha * eta)     -> Tafel slope = alpha
At anodic eta:   I ~ k0 * exp((1-alpha) * eta)  -> Tafel slope = (1-alpha)
Together: two independent constraints on alpha, breaking the k0-alpha correlation.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVJoint_charged_symmetric.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

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


def _make_recovery():
    return ForwardRecoveryConfig(
        max_attempts=6, max_it_only_attempts=2,
        anisotropy_only_attempts=1, tolerance_relax_attempts=2,
        max_it_growth=1.5, max_it_cap=600,
    )


def main() -> None:
    # Three voltage configurations for comparison
    configs = {
        "cathodic_only": {
            "eta": np.linspace(-1.0, -10.0, 10),
            "label": "Cathodic only [-1, -10], 10 points",
        },
        "symmetric_focused": {
            "eta": np.array([
                +5.0, +3.0, +1.0,           # anodic (3 pts)
                -0.5,                         # near-equilibrium
                -1.0, -2.0, -3.0,           # cathodic onset
                -5.0, -8.0,                  # transition
                -10.0, -15.0, -20.0,        # knee + plateau
            ]),
            "label": "Symmetric focused [-20, +5], 12 points",
        },
        "symmetric_full": {
            "eta": np.array([
                +5.0, +3.0, +2.0, +1.0, +0.5,
                -0.25, -0.5,
                -1.0, -1.5, -2.0, -3.0,
                -4.0, -5.0, -6.5, -8.0,
                -10.0, -13.0,
                -17.0, -22.0, -28.0,
            ]),
            "label": "Symmetric full [-28, +5], 20 points",
        },
    }

    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]
    observable_scale = -I_SCALE

    base_output = os.path.join("StudyResults", "bv_joint_inference_charged_symmetric")
    os.makedirs(base_output, exist_ok=True)

    all_results = {}

    for config_name, cfg in configs.items():
        eta_values = cfg["eta"]
        label = cfg["label"]

        print(f"\n{'='*70}")
        print(f"  Joint Inference: {label}")
        print(f"{'='*70}")

        out_dir = os.path.join(base_output, config_name)
        t0 = time.time()

        # Use bridge points for configurations with large gaps
        max_gap = 3.0 if eta_values.min() < -12 else 0.0

        request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=initial_k0_guess,
            phi_applied_values=eta_values.tolist(),
            target_csv_path=os.path.join(out_dir, "target.csv"),
            output_dir=out_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"Joint: {label}",
            control_mode="joint",
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            true_alpha=true_alpha,
            initial_alpha_guess=initial_alpha_guess,
            alpha_lower=0.05, alpha_upper=0.95,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=max_gap,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=40,
            live_plot=False,
            forward_recovery=_make_recovery(),
        )

        result = run_bv_joint_flux_curve_inference(request)
        elapsed = time.time() - t0

        best_k0 = np.asarray(result["best_k0"])
        best_alpha = np.asarray(result["best_alpha"])

        all_results[config_name] = {
            "k0": best_k0.tolist(),
            "alpha": best_alpha.tolist(),
            "loss": result["best_loss"],
            "time": elapsed,
            "label": label,
            "n_points": len(eta_values),
        }

        print(f"\n  Result: k0 = {best_k0.tolist()}, alpha = {best_alpha.tolist()}")
        print(f"  Time: {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    print(f"\n{'='*100}")
    print(f"  Joint Inference Comparison: Cathodic-Only vs Symmetric")
    print(f"{'='*100}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print()

    print(f"{'Config':<25} | {'pts':>4} | {'k0_1 err':>10} {'k0_2 err':>10} "
          f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12} | {'time':>6}")
    print(f"{'-'*100}")

    for config_name, r in all_results.items():
        k0_arr = np.asarray(r["k0"])
        alpha_arr = np.asarray(r["alpha"])
        k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{config_name:<25} | {r['n_points']:>4} | "
              f"{k0_err[0]:>10.4f} {k0_err[1]:>10.4f} "
              f"{alpha_err[0]:>10.4f} {alpha_err[1]:>10.4f} "
              f"| {r['loss']:>12.6e} | {r['time']:>5.0f}s")

    print(f"{'='*100}")

    # Save CSV
    csv_path = os.path.join(base_output, "joint_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "n_points", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err", "k0_2_err", "alpha_1_err", "alpha_2_err",
            "loss", "time_s",
        ])
        for config_name, r in all_results.items():
            k0_arr = np.asarray(r["k0"])
            alpha_arr = np.asarray(r["alpha"])
            k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
            alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
            writer.writerow([
                config_name, r["n_points"],
                f"{k0_arr[0]:.8e}", f"{k0_arr[1]:.8e}",
                f"{alpha_arr[0]:.6f}", f"{alpha_arr[1]:.6f}",
                f"{k0_err[0]:.6f}", f"{k0_err[1]:.6f}",
                f"{alpha_err[0]:.6f}", f"{alpha_err[1]:.6f}",
                f"{r['loss']:.12e}", f"{r['time']:.1f}",
            ])
    print(f"\n[csv] Comparison saved -> {csv_path}")

    print(f"\n=== Joint Inference Comparison Complete ===")
    print(f"Output: {base_output}/")


if __name__ == "__main__":
    main()
