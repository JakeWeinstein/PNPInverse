"""Adjoint-gradient inference of BV transfer coefficients (alpha) from I-V curve.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers [alpha_1, alpha_2] with k0 values held fixed at their true values.

Physical parameters from Mangan2025 (pH 4):
  - 4-species charged: O2 (D=1.9e-9), H2O2 (D=1.6e-9),
    H+ (D=9.311e-9, z=+1), ClO4- (D=1.792e-9, z=-1)
  - c_O2 = 0.5 mol/m3, c_H+ = c_ClO4- = 0.1 mol/m3, L_ref = 100 um
  - R1: O2 + 2H+ + 2e- -> H2O2   (reversible, k0=2.4e-8 m/s, alpha=0.627)
  - R2: H2O2 + 2H+ + 2e- -> H2O  (irreversible, k0=1e-9 m/s, alpha=0.5)
  - cathodic_conc_factors: (c_H+/c_ref_H+)^2 in both reactions

Mesh: make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
  Markers: 3=bottom(electrode), 4=top(bulk), 1/2=left/right(zero-flux)

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVAlpha_charged_from_current_density_curve.py
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
    run_bv_alpha_flux_curve_inference,
)
from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig


# ---------------------------------------------------------------------------
# Physical constants and scales
# ---------------------------------------------------------------------------

F_CONST = 96485.3329     # C/mol
R_GAS = 8.31446          # J/(mol K)
T_REF = 298.15           # K
V_T = R_GAS * T_REF / F_CONST   # 0.025693 V
N_ELECTRONS = 2

# Species 0: O2 (neutral)
D_O2 = 1.9e-9            # m2/s
C_O2 = 0.5               # mol/m3

# Species 1: H2O2 (neutral)
D_H2O2 = 1.6e-9          # m2/s
C_H2O2 = 0.0             # mol/m3

# Species 2: H+ (z=+1)
D_HP = 9.311e-9           # m2/s
C_HP = 0.1               # mol/m3 (pH 4)

# Species 3: ClO4- (z=-1)
D_CLO4 = 1.792e-9        # m2/s
C_CLO4 = 0.1             # mol/m3 (electroneutrality)

# R1 kinetics: O2 -> H2O2
K0_PHYS = 2.4e-8          # m/s
ALPHA_1 = 0.627           # true value (to be inferred)

# R2 kinetics: H2O2 -> H2O (irreversible)
K0_2_PHYS = 1e-9          # m/s
ALPHA_2 = 0.5             # true value (to be inferred)

# Reference scales
L_REF = 1.0e-4            # m (100 um)
D_REF = D_O2              # 1.9e-9 m2/s
C_SCALE = C_O2            # 0.5 mol/m3
K_SCALE = D_REF / L_REF   # m/s

# Dimensionless species quantities
D_O2_HAT = D_O2 / D_REF       # 1.0
D_H2O2_HAT = D_H2O2 / D_REF   # ~0.842
D_HP_HAT = D_HP / D_REF       # ~4.9
D_CLO4_HAT = D_CLO4 / D_REF   # ~0.943

C_O2_HAT = C_O2 / C_SCALE     # 1.0
C_H2O2_HAT = C_H2O2 / C_SCALE # 0.0
C_HP_HAT = C_HP / C_SCALE     # 0.2
C_CLO4_HAT = C_CLO4 / C_SCALE # 0.2

K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

# Current density conversion: dimensionless rate -> mA/cm2
I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_SCALE / L_REF * 0.1

print(f"[params] D_ref = {D_REF:.3e} m2/s")
print(f"[params] K_scale = {K_SCALE:.3e} m/s")
print(f"[params] k0_hat (R1) = {K0_HAT:.6f}  (FIXED)")
print(f"[params] k0_2_hat (R2) = {K0_2_HAT:.8f}  (FIXED)")
print(f"[params] True alpha_1 = {ALPHA_1}, alpha_2 = {ALPHA_2}")
print(f"[params] I_scale = {I_SCALE:.4f} mA/cm2")
print(f"[params] Species: O2(z=0), H2O2(z=0), H+(z=+1), ClO4-(z=-1)")
print(f"[params] D_hat = [{D_O2_HAT:.3f}, {D_H2O2_HAT:.3f}, {D_HP_HAT:.3f}, {D_CLO4_HAT:.3f}]")
print(f"[params] c0_hat = [{C_O2_HAT:.3f}, {C_H2O2_HAT:.3f}, {C_HP_HAT:.3f}, {C_CLO4_HAT:.3f}]")


# ---------------------------------------------------------------------------
# SNES options (conservative for charged system)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Build SolverParams for 4-species charged BV
# ---------------------------------------------------------------------------

def _make_bv_solver_params(
    eta_hat: float,
    dt: float,
    t_end: float,
) -> SolverParams:
    """Build SolverParams for 4-species charged BV with graded rectangle mesh."""
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
        "diffusivity_scale_m2_s":               D_REF,
        "concentration_scale_mol_m3":           C_SCALE,
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
                "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
            {
                "k0": K0_2_HAT,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
        ],
        # Legacy per-species fallback (needed by _get_bv_cfg for markers)
        "k0": [K0_HAT] * 4,
        "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0],
        "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        # Graded rectangle mesh markers
        "electrode_marker":      3,   # bottom (y=0)
        "concentration_marker":  4,   # top (y=1, bulk)
        "ground_marker":         4,   # top (y=1, ground)
    }
    return SolverParams.from_list([
        4,                                                  # n_species
        1,                                                  # FE order
        dt,                                                 # dt (nondim)
        t_end,                                              # t_end (nondim)
        [0, 0, 1, -1],                                     # z_vals
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],    # D_vals (nondim)
        [0.01, 0.01, 0.01, 0.01],                          # a_vals (Bikerman)
        eta_hat,                                             # phi_applied
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],    # c0_vals (nondim)
        0.0,                                                 # phi0
        params,
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Sweep: eta_hat from -1 to -10 (10 points)
    # Conservative range: the charged 4-species system is stiffer than neutral.
    eta_values = np.linspace(-1.0, -10.0, 10)

    # True alpha values -- ground truth for synthetic data
    true_alpha = [ALPHA_1, ALPHA_2]

    # Fixed k0 values (known, not being inferred)
    fixed_k0 = [K0_HAT, K0_2_HAT]

    # Initial guess -- deliberately wrong
    initial_alpha_guess = [0.3, 0.3]

    dt = 0.5
    max_ss_steps = 100
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

    output_dir = os.path.join("StudyResults", "bv_alpha_inference_charged")

    # Observable scale: cathodic (reduction) gives negative current.
    observable_scale = -I_SCALE

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=fixed_k0,
        initial_guess=fixed_k0,  # k0 initial = fixed (not optimized)
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
        observable_title="Charged 4-species BV alpha inference from I-V curve",
        # Control mode
        control_mode="alpha",
        # k0 bounds (not optimized but needed)
        k0_lower=1e-8,
        k0_upper=100.0,
        log_space=True,
        # Alpha settings
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05,
        alpha_upper=0.95,
        alpha_log_space=False,
        fixed_k0=fixed_k0,
        # Mesh
        mesh_Nx=8,
        mesh_Ny=200,
        mesh_beta=3.0,
        # Optimizer
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 40,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "disp": True,
        },
        max_iters=40,
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
        live_plot_export_gif_path=os.path.join(output_dir, "bv_alpha_convergence.gif"),
        anisotropy_trigger_failed_points=4,
        anisotropy_trigger_failed_fraction=0.25,
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=6,
            max_it_only_attempts=2,
            anisotropy_only_attempts=1,
            tolerance_relax_attempts=2,
            max_it_growth=1.5,
            max_it_cap=600,
            atol_relax_factor=10.0,
            rtol_relax_factor=10.0,
            ksp_rtol_relax_factor=10.0,
            line_search_schedule=("bt", "l2", "cp", "basic"),
            anisotropy_target_ratio=3.0,
            anisotropy_blend=0.5,
        ),
    )

    result = run_bv_alpha_flux_curve_inference(request)

    # Print results
    best_alpha = np.asarray(result["best_alpha"], dtype=float)
    true_alpha_arr = np.asarray(true_alpha, dtype=float)

    print("\n=== Alpha Inference Results (Charged 4-Species) ===")
    print(f"True alpha:      [{true_alpha_arr[0]:.4f}, {true_alpha_arr[1]:.4f}]")
    print(f"Estimated alpha: [{best_alpha[0]:.4f}, {best_alpha[1]:.4f}]")
    rel_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    print(f"Relative error:  [{rel_err[0]:.4f}, {rel_err[1]:.4f}]")
    print(f"Fixed k0 (nondim): [{K0_HAT:.6f}, {K0_2_HAT:.8f}]")


if __name__ == "__main__":
    main()
