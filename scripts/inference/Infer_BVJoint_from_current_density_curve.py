"""Adjoint-gradient joint inference of (k0, alpha) from I-V curve.

Recovers [k0_1, k0_2, alpha_1, alpha_2] simultaneously for the 2-species
coupled BV model (O2 + H2O2) from a synthetic current-density vs.
overpotential curve.

Physical parameters from Mangan2025 (pH 4):
  - 2-species neutral: O2 (D=2.1e-9), H2O2 (D=1.6e-9), z=[0,0]
  - c_bulk = 0.5 mol/m3, L_ref = 100 um
  - R1: O2 + 2H+ + 2e- -> H2O2   (reversible, k0=2.4e-8 m/s, alpha=0.627)
  - R2: H2O2 + 2H+ + 2e- -> H2O  (irreversible, k0=1e-9 m/s, alpha=0.5)

Mesh: make_graded_rectangle_mesh(Nx=4, Ny=200, beta=3.0)
  Markers: 3=bottom(electrode), 4=top(bulk), 1/2=left/right(zero-flux)

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVJoint_from_current_density_curve.py
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

F_CONST = 96485.3329     # C/mol
V_T = 0.025693           # thermal voltage at 25 C, V
N_ELECTRONS = 2

# Species 0: O2 (neutral)
D_O2 = 2.10e-9           # m2/s
C_BULK = 0.5             # mol/m3

# Species 1: H2O2 (neutral)
D_H2O2 = 1.60e-9         # m2/s

# R1 kinetics: O2 -> H2O2
K0_PHYS = 2.4e-8         # m/s
ALPHA_1 = 0.627

# R2 kinetics: H2O2 -> H2O (irreversible)
K0_2_PHYS = 1e-9          # m/s
ALPHA_2 = 0.5

# Reference scales
L_REF = 1.0e-4            # m (100 um)
D_REF = D_O2
K_SCALE = D_REF / L_REF   # m/s

# Dimensionless
D_O2_HAT = D_O2 / D_REF       # = 1.0
D_H2O2_HAT = D_H2O2 / D_REF   # ~ 0.762
K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

# Steric (Bikerman)
A_O2_HAT = 0.01
A_H2O2_HAT = 0.01

# Current density conversion: dimensionless rate -> mA/cm2
I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_BULK / L_REF * 0.1

print(f"[params] D_ref = {D_REF:.3e} m2/s")
print(f"[params] K_scale = {K_SCALE:.3e} m/s")
print(f"[params] k0_hat (R1) = {K0_HAT:.6f}")
print(f"[params] k0_2_hat (R2) = {K0_2_HAT:.8f}")
print(f"[params] True alpha_1 = {ALPHA_1}, alpha_2 = {ALPHA_2}")
print(f"[params] I_scale = {I_SCALE:.4f} mA/cm2")


# ---------------------------------------------------------------------------
# SNES options
# ---------------------------------------------------------------------------

SNES_OPTS = {
    "snes_type":                 "newtonls",
    "snes_max_it":               200,
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


# ---------------------------------------------------------------------------
# Build SolverParams
# ---------------------------------------------------------------------------

def _make_bv_solver_params(
    eta_hat: float,
    dt: float,
    t_end: float,
) -> SolverParams:
    """Build SolverParams for 2-species BV with graded rectangle mesh markers."""
    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent":            True,
        "exponent_clip":            50.0,
        "regularize_concentration": True,
        "conc_floor":               1e-12,
        "use_eta_in_bv":            True,
    }
    params["nondim"] = {
        "enabled":                              True,
        "diffusivity_scale_m2_s":               D_O2,
        "concentration_scale_mol_m3":           C_BULK,
        "length_scale_m":                       L_REF,
        "potential_scale_v":                     V_T,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_HAT,
                "alpha": ALPHA_1,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": 1.0,
                "stoichiometry": [-1, +1],
                "n_electrons": 2,
                "reversible": True,
            },
            {
                "k0": K0_2_HAT,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": [0, -1],
                "n_electrons": 2,
                "reversible": False,
            },
        ],
        # Legacy per-species fallback (needed by _get_bv_cfg for markers)
        "k0": [K0_HAT, K0_2_HAT],
        "alpha": [ALPHA_1, ALPHA_2],
        "stoichiometry": [-1, -1],
        "c_ref": [1.0, 0.0],
        "E_eq_v": 0.0,
        # Graded rectangle mesh markers
        "electrode_marker":      3,   # bottom (y=0)
        "concentration_marker":  4,   # top (y=1, bulk)
        "ground_marker":         4,   # top (y=1, ground)
    }
    return SolverParams.from_list([
        2,                               # n_species
        1,                               # FE order
        dt,                              # dt (nondim)
        t_end,                           # t_end (nondim)
        [0, 0],                          # z_vals (neutral)
        [D_O2_HAT, D_H2O2_HAT],         # D_vals (nondim)
        [A_O2_HAT, A_H2O2_HAT],         # a_vals (Bikerman)
        eta_hat,                         # phi_applied
        [1.0, 0.0],                      # c0_vals: O2=1, H2O2=0
        0.0,                             # phi0
        params,
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Sweep: eta_hat from 0 to cathodic (15 points)
    eta_values = np.linspace(-1.0, -20.0, 15)

    # True values -- ground truth for synthetic data
    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]

    # Initial guesses -- both k0 and alpha wrong
    initial_k0_guess = [0.005, 0.0005]   # ~4x too high
    initial_alpha_guess = [0.4, 0.3]     # wrong direction for alpha_1

    dt = 0.5
    max_ss_steps = 60
    t_end = dt * max_ss_steps

    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4,
        absolute_tolerance=1e-8,
        consecutive_steps=4,
        max_steps=max_ss_steps,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    output_dir = os.path.join("StudyResults", "bv_joint_inference")

    # Observable scale: cathodic (reduction) gives negative current.
    observable_scale = -I_SCALE

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
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
        observable_title="Joint (k0, alpha) inference from I-V curve",
        # Control mode
        control_mode="joint",
        # k0 bounds
        k0_lower=1e-8,
        k0_upper=100.0,
        log_space=True,
        # Alpha settings
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05,
        alpha_upper=0.95,
        alpha_log_space=False,
        # Mesh
        mesh_Nx=4,
        mesh_Ny=200,
        mesh_beta=3.0,
        # Optimizer
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 30,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "disp": True,
        },
        max_iters=30,
        gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        blob_initial_condition=False,
        # Live plot
        live_plot=False,
        live_plot_pause_seconds=0.001,
        live_plot_eval_lines=False,
        live_plot_eval_line_alpha=0.30,
        live_plot_eval_max_lines=120,
        live_plot_export_gif_path=os.path.join(output_dir, "bv_joint_convergence.gif"),
        anisotropy_trigger_failed_points=4,
        anisotropy_trigger_failed_fraction=0.25,
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=6,
            max_it_only_attempts=2,
            anisotropy_only_attempts=1,
            tolerance_relax_attempts=2,
            max_it_growth=1.5,
            max_it_cap=500,
            atol_relax_factor=10.0,
            rtol_relax_factor=10.0,
            ksp_rtol_relax_factor=10.0,
            line_search_schedule=("bt", "l2", "cp", "basic"),
            anisotropy_target_ratio=3.0,
            anisotropy_blend=0.5,
        ),
    )

    result = run_bv_joint_flux_curve_inference(request)

    # Print redimensionalized results
    best_k0 = np.asarray(result["best_k0"], dtype=float)
    best_alpha = np.asarray(result["best_alpha"], dtype=float)
    true_k0_arr = np.asarray(true_k0, dtype=float)
    true_alpha_arr = np.asarray(true_alpha, dtype=float)

    best_k0_phys = best_k0 * K_SCALE
    true_k0_phys = true_k0_arr * K_SCALE

    print("\n=== Redimensionalized Joint Inference Results ===")
    print(f"K_scale = {K_SCALE:.6e} m/s")
    print(f"True k0 (m/s):      [{true_k0_phys[0]:.4e}, {true_k0_phys[1]:.4e}]")
    print(f"Estimated k0 (m/s): [{best_k0_phys[0]:.4e}, {best_k0_phys[1]:.4e}]")
    k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    print(f"k0 relative error:  [{k0_err[0]:.4f}, {k0_err[1]:.4f}]")
    print(f"True alpha:         [{true_alpha_arr[0]:.4f}, {true_alpha_arr[1]:.4f}]")
    print(f"Estimated alpha:    [{best_alpha[0]:.4f}, {best_alpha[1]:.4f}]")
    alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    print(f"alpha relative error: [{alpha_err[0]:.4f}, {alpha_err[1]:.4f}]")


if __name__ == "__main__":
    main()
